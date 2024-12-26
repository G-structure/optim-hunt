import sys
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
import functools
from tqdm import tqdm
from IPython.display import display
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv

from optim_hunter.plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference, line
from optim_hunter.utils import prepare_prompt, slice_dataset
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.data_model import create_comparison_data
from optim_hunter.plot_html import create_logit_lens_plot
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

#device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

# Load directly from model path https://github.com/TransformerLensOrg/TransformerLens/issues/691
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random ]

MODEL_TYPE = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "/home/freiza/optim_hunter/.models/Llama-3.1-8B-Instruct/"

def load_llama_model(model_path=MODEL_PATH, model_type=MODEL_TYPE):
    """
    Load and configure a Llama model using HookedTransformer.
    
    Args:
        model_path (str): Path to the model files
        model_type (str): Type/name of the model
        
    Returns:
        HookedTransformer: The configured model instance
    """
    if not model_path:
        return None
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True,
                                                 #quantization_config=BitsAndBytesConfig(load_in_4bit=True), 
                                                 #torch_dtype = t.float32, 
                                                 #device_map = "cuda:0"
                                                 )

    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = HookedTransformer.from_pretrained(
        model_type,
        hf_model=hf_model,
        device="cuda",
        n_devices=2,
        fold_ln=True,
        # fold_value_biases=False,
        center_writing_weights=True,
        # refactor_factored_attn_matrices=True,
        center_unembed=True,
        # dtype=t.bfloat16,
        dtype=t.float16,
        default_padding_side='left',
        tokenizer=tokenizer,
        verbose=False
    )

    return model

def generate_linreg_tokens(
    model: HookedTransformer,
    dataset,
    seq_len = 5,
    batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of linear regression ICL tokens

    Outputs are:
        linreg_tokens: [batch, 1+linreg]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    zero_token = model.to_tokens('0', truncate=True)[0][-1]
    
    # Create list to store tokens for each batch
    batch_tokens = []
    data_store = []

    dataset_func = get_dataset_friedman_2
    
    # Generate tokens for each batch with different random seeds
    for i in range(batch):
        data = create_comparison_data(model, dataset_func, regressors, random_state=i, seq_len=seq_len)
        tokens = model.to_tokens(data['prompt'], truncate=True)
        batch_tokens.append(tokens[0])
        data_store.append(data)
    
    # Find the longest sequence length
    max_len = max(len(tokens) for tokens in batch_tokens)
    
    # Pad shorter sequences with token 0 at position -4
    for i in range(len(batch_tokens)):
        while len(batch_tokens[i]) < max_len:
            # Insert 0 at position -4 from the end
            print(f"Found mismatch in token length for batch {i}!\nLargest length: {max_len}\nBatch {i} length: {len(batch_tokens[i])}\nApplying padding...")
            print(f"\nBefore Zero Token Padding:\n#####\n{model.to_string(batch_tokens[i][-50:])}\n#####")
            batch_tokens[i] = t.cat([
                batch_tokens[i][:len(batch_tokens[i])-3],  
                zero_token.unsqueeze(0), # Add unsqueeze to make zero_token 1-dimensional
                batch_tokens[i][len(batch_tokens[i])-3:]
            ])
            print(f"\nAfter Zero Token Padding:\n#####\n{model.to_string(batch_tokens[i][-50:])}\n#####")

    
    # Stack all batches together 
    linreg_tokens = t.stack(batch_tokens).to(device)
    
    # Add prefix to each batch
    linreg_tokens = t.cat([prefix, linreg_tokens], dim=-1).to(device)
    return linreg_tokens, data_store

def run_and_cache_model_linreg_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> tuple[Tensor, Tensor, ActivationCache]:
    '''
    Generates a sequence of linear regression ICL tokens, and runs the model on it, returning (tokens, logits, cache)

    Should use the `generate_linreg_tokens` function above

    Outputs are:
        linreg_tokens: [batch, 1+linreg]
        linreg_logits: [batch, 1+linreg, d_vocab]
        linreg_cache: The cache of the model run on linreg_tokens
    '''
    linreg_tokens, linreg_data_store = generate_linreg_tokens(model, get_dataset_friedman_2, seq_len, batch)
    linreg_logits, linreg_cache = model.run_with_cache(linreg_tokens)
    return linreg_tokens, linreg_logits, linreg_cache, linreg_data_store
import torch as t
from transformer_lens import (
    HookedTransformer,
)

import logging

# Configure logging
logger = logging.getLogger(__name__)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

#device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

# Load directly from model path https://github.com/TransformerLensOrg/TransformerLens/issues/691
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_TYPE = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH = "/home/freiza/optim_hunter/.models/Llama-3.1-8B-Instruct/"

def load_llama_model(model_path=MODEL_PATH, model_type=MODEL_TYPE):
    """Load and configure a Llama model using HookedTransformer.

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

def load_gpt2_model():
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
        device="cuda"
    )
    return model

import torch as t
from torch import Tensor
from jaxtyping import Int
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.data_model import create_comparison_data
import logging
from typing import List, Tuple

regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random ]
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

def get_numerical_tokens(model):
    # Get the vocabulary
    vocab = model.tokenizer.get_vocab()

    # Search for numerical tokens
    numerical_tokens = {}
    for token, id in vocab.items():
        # Skip superscript numbers and other special characters
        if token in ['¹', '²', '³']:
            continue
            
        # Check if token is a number (integer or decimal)
        # Only include ASCII digits and decimal point
        if token.strip().replace('.','').replace('-','').isdigit() and all(c in '0123456789.-' for c in token):
            numerical_tokens[token] = id
    
    return numerical_tokens

def generate_linreg_tokens(
    model: HookedTransformer,
    dataset,
    seq_len = 5,
    batch: int = 1,
    sub_batch = None
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
    if sub_batch:
        data = create_comparison_data(model, dataset_func, regressors, random_state=sub_batch, seq_len=seq_len)
        tokens = model.to_tokens(data['prompt'], truncate=True)
        batch_tokens.append(tokens[0])
        data_store.append(data)
    else:
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

def run_and_cache_model_linreg_tokens_batched(
    model: HookedTransformer, 
    seq_len: int, 
    total_batch: int = 1,
) -> tuple[List[Tensor], List[Tensor], List[ActivationCache], List]:
    '''
    Generates sequences of linear regression ICL tokens in smaller batches, 
    runs the model, and offloads cache to CPU to manage memory.

    Args:
        model: The transformer model
        seq_len: Sequence length for each example
        total_batch: Total number of examples to process
        sub_batch: Size of each sub-batch to process at once

    Returns:
        all_tokens: List of token tensors for each batch
        all_logits: List of logit tensors for each batch
        all_caches: List of activation caches (on CPU) for each batch
        all_data_stores: List of data stores for each batch
    '''
    all_tokens = []
    all_logits = []
    all_caches = []
    all_data_stores = []

    # Process data in smaller batches
    for i in range(total_batch):
        # Generate and process current batch with sub_batch index
        batch_tokens, batch_data_store = generate_linreg_tokens(
            model, 
            get_dataset_friedman_2, 
            seq_len, 
            batch=1,  # Since we're processing one at a time
            sub_batch=i
        )
        batch_logits, batch_cache = model.run_with_cache(batch_tokens)

        # Move results to CPU to free GPU memory
        all_tokens.append(batch_tokens.cpu())
        all_logits.append(batch_logits.cpu())
        all_caches.append(batch_cache.to('cpu'))
        all_data_stores.extend(batch_data_store)

        # Optional: Clear CUDA cache
        if t.cuda.is_available():
            t.cuda.empty_cache()

    return all_tokens, all_logits, all_caches, all_data_stores
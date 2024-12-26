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

from optim_hunter.data_model import create_comparison_data
import logging
from typing import List, Tuple

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

def get_numerical_tokens(model):
    """
    Extract numerical tokens from a model's vocabulary.
    
    This function filters the model's vocabulary to find tokens that represent 
    numerical values (integers or decimals), excluding special characters like 
    superscripts.
    
    Args:
        model: A transformer model with a tokenizer attribute that has a get_vocab() method
        
    Returns:
        dict: A dictionary mapping numerical tokens (str) to their corresponding token IDs (int).
              Only includes tokens containing ASCII digits, decimal points, and minus signs.
              
    Example:
        >>> numerical_tokens = get_numerical_tokens(model)
        >>> print(numerical_tokens)
        {'0': 1, '1': 2, '1.5': 3, '-2': 4, ...}
    """
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
    regressors,
    seq_len = 5,
    batch: int = 1,
    sub_batch = None,
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of linear regression in-context learning tokens for multiple batches.

    Args:
        model (HookedTransformer): The transformer model to use
        dataset: Dataset to generate tokens from
        regressors: List of regression functions to compare
        seq_len (int): Length of sequence to generate
        batch (int, optional): Number of batches to generate. Defaults to 1
        sub_batch (int, optional): If provided, generates tokens for just this specific batch index. 
                                 Defaults to None.

    Returns:
        tuple: A tuple containing:
            - linreg_tokens (Tensor): Tokenized sequences with shape [batch, sequence_length]
            - data_store (list): List of dictionaries containing comparison data for each batch
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    zero_token = model.to_tokens('0', truncate=True)[0][-1]
    
    # Create list to store tokens for each batch
    batch_tokens = []
    data_store = []

    dataset_func = dataset
    
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

def run_and_cache_model_linreg_tokens(model: HookedTransformer, dataset, regressors, seq_len: int, batch: int = 1) -> tuple[Tensor, Tensor, ActivationCache]:
    '''
    Generates a sequence of linear regression in-context learning tokens, runs the model on them, and returns tokens, logits and cache.

    Args:
        model (HookedTransformer): The transformer model to run
        dataset: Dataset function or object to generate examples from
        regressors: List of regression models to compare against
        seq_len (int): Length of sequence for each example
        batch (int, optional): Batch size. Defaults to 1.

    Returns:
        tuple:
            - linreg_tokens (Tensor): Tokenized sequences with shape [batch, 1+seq_len] 
            - linreg_logits (Tensor): Model logits with shape [batch, 1+seq_len, vocab_size]
            - linreg_cache (ActivationCache): Cache of model activations
            - linreg_data_store (list): List of dictionaries containing comparison data for each batch
    '''
    linreg_tokens, linreg_data_store = generate_linreg_tokens(model, dataset, regressors, seq_len, batch)
    linreg_logits, linreg_cache = model.run_with_cache(linreg_tokens)
    return linreg_tokens, linreg_logits, linreg_cache, linreg_data_store

def run_and_cache_model_linreg_tokens_batched(
    model: HookedTransformer,
    dataset,
    regressors,
    seq_len: int, 
    total_batch: int = 1,
) -> tuple[List[Tensor], List[Tensor], List[ActivationCache], List]:
    '''
    Generates sequences of linear regression in-context learning tokens in smaller batches,
    runs the model, and offloads cache to CPU to manage memory.

    Args:
        model (HookedTransformer): The transformer model to run
        dataset: Dataset function or object to generate examples from
        regressors: List of regression models to compare against
        seq_len (int): Length of sequence for each example
        total_batch (int, optional): Total number of batches to process. Defaults to 1.

    Returns:
        tuple:
            - all_tokens (List[Tensor]): List of tokenized sequences, each with shape [1, 1+seq_len]
            - all_logits (List[Tensor]): List of model logits, each with shape [1, 1+seq_len, vocab_size]
            - all_caches (List[ActivationCache]): List of model activation caches stored on CPU
            - all_data_stores (List): List of dictionaries containing comparison data for each batch
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
            dataset,
            regressors, 
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
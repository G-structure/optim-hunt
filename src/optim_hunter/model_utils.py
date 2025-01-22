"""Main module for optimisation and analysis.

This module provides functionality for token analysis, data generation,
and model interaction with transformer models for optimization experiments.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from optim_hunter.LR_methods import RegressionResults

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as t
from jaxtyping import Int
from torch import Tensor
from transformer_lens import (
    ActivationCache,
    HookedTransformer,
)
from transformers import PreTrainedTokenizer

from optim_hunter.data_model import create_comparison_data
from optim_hunter.utils import (
    pad_numeric_tokens,
    prepare_prompt_from_tokens,
    slice_dataset,
)

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


def get_numerical_tokens(model: HookedTransformer) -> dict[str, int]:
    """Extract numerical tokens from a model's vocabulary.

    This function filters the model's vocabulary to find tokens that represent
    numerical values (integers or decimals), excluding special characters like
    superscripts.

    Args:
        model: A transformer model with a tokenizer attribute that has a
            get_vocab() method

    Returns:
        dict: A dictionary mapping numerical tokens (str) to their
            corresponding token IDs (int). Only includes tokens containing
            ASCII digits, decimal points, and minus signs.

    Example:
        >>> numerical_tokens = get_numerical_tokens(model)
        >>> print(numerical_tokens)
        {'0': 1, '1': 2, '1.5': 3, '-2': 4, ...}

    """
    if (not hasattr(model, 'tokenizer') or
            not isinstance(model.tokenizer, PreTrainedTokenizer)):
        raise AttributeError(
            "Model must have a PreTrainedTokenizer as its tokenizer attribute"
        )

    # Get the vocabulary
    vocab = model.tokenizer.get_vocab()

    # Search for numerical tokens
    numerical_tokens: dict[str, int] = {}
    for token, id in vocab.items():
        # Skip superscript numbers and other special characters
        if token in ["¹", "²", "³"]:
            continue

        # Check if token is a number (integer or decimal)
        # Only include ASCII digits and decimal point
        if token.strip().replace(".", "").replace("-", "").isdigit() and all(
            c in "0123456789.-" for c in token
        ):
            numerical_tokens[token] = id

    return numerical_tokens


def get_tokenized_prompt(
    model: HookedTransformer,
    seq_len: int,
    random_int: int,
    dataset: Callable[
        [int],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    print_prompt: bool = True
) -> t.Tensor:
    """Get tokenized prompt for in-context learning.

    Args:
        model (HookedTransformer): The transformer model to use for tokenization
        seq_len (int): Length of sequence to generate
        random_int (int): Random seed for dataset generation
        dataset (Callable): Function that returns (x_train, y_train, x_test,
            y_test) tuple of (DataFrame, Series, DataFrame, Series) containing
            training and test data
        print_prompt (bool): Whether to print the text prompt

    Returns:
        torch.Tensor: Tokenized prompt tensor

    """
    # Dataset returns types are defined by the Callable type hint
    x_train, y_train, x_test, y_test = dataset(random_int)

    # slice_dataset returns same types as input
    x_train_sliced, y_train_sliced, x_test_sliced, _ = slice_dataset(
        x_train, y_train, x_test, y_test, seq_len
    )

    # Cast return values from pad_numeric_tokens to DataFrames/Series
    tokens = pad_numeric_tokens(
        model,
        x_train_sliced,
        y_train_sliced,
        x_test_sliced
    )
    x_train_tokens = pd.DataFrame(tokens[0])
    y_train_tokens = pd.Series(tokens[1])
    x_test_tokens = pd.DataFrame(tokens[2])

    # prepare_prompt_from_tokens returns torch tensor
    tokenized_prompt: t.Tensor = prepare_prompt_from_tokens(
        model,
        x_train_tokens,
        y_train_tokens,
        x_test_tokens,
        prepend_bos=True,
        prepend_inst=True,
    )

    # model.to_string returns str, cast to ensure string type
    prompt: str = str(model.to_string(tokenized_prompt[0]))
    if print_prompt:
        print(prompt)

    return tokenized_prompt


def check_token_positions(
    model: HookedTransformer,
    dataset: Callable[
        [int],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    seq_len: int,
    seed: int = 0,
    print_info: bool = True
) -> tuple[list[int], list[int]]:
    """Check token positions for a single sequence length and seed.

    Args:
        model: The transformer model
        dataset: The dataset being used
        seq_len: Sequence length to test
        seed: Random seed (default: 0)
        print_info: Whether to print diagnostic information (default: True)

    Returns:
        tuple: Lists of output and feature positions

    """
    # Get tokenized prompt for the specified seed
    tokenized_prompt = get_tokenized_prompt(
        model, seq_len, seed, dataset, print_prompt=print_info
    )
    input_tokens = tokenized_prompt.to(model.cfg.device)

    # Get string tokens for the full sequence
    str_tokens = model.to_str_tokens(input_tokens[0])

    # Find indices where "Output:" and "Features:" appear
    output_indices = [
        i
        for i, token in enumerate(str_tokens[24:], start=24)
        if token == "Output"
    ]
    feature_indices = [
        i
        for i, token in enumerate(str_tokens[24:], start=24)
        if token == "Feature"
    ]

    # Get positions of first number token after each marker
    output_positions: list[int] = []
    for idx in output_indices:
        current_pos = idx + 3  # Skip "Output: "
        if current_pos < len(str_tokens) - 1:
            output_positions.append(current_pos)

    feature_positions: list[int] = []
    for idx in feature_indices:
        current_pos = idx + 5  # Skip "Features n: "
        if current_pos < len(str_tokens) - 1:
            feature_positions.append(current_pos)

    # Print results
    if print_info:
        print("\nPositions of first number token after 'Output:':")
        print(f"Positions: {output_positions}")

        print("\nPositions of first number token after 'Feature:':")
        print(f"Positions: {feature_positions}")

        # Print example tokens at these positions
        print("\nExample tokens at these positions:")
        print("After Output:", [str_tokens[pos] for pos in output_positions])
        print("After Features:", [str_tokens[pos] for pos in feature_positions])

    return output_positions, feature_positions


def generate_linreg_tokens(
    model: HookedTransformer,
    dataset: Callable[
        [int],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[
        Callable[
            [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
            RegressionResults
        ]
    ],
    seq_len: int = 5,
    batch_size: int = 1,
    sub_batch: Optional[int] = None,
) -> Tuple[Int[t.Tensor, "batch"], List[Dict[str, Any]]]:  # noqa: F821
    """Generate linear regression in-context learning tokens for batches.

    Args:
        model (HookedTransformer): The transformer model to use
        dataset: Dataset to generate tokens from
        regressors: List of regression functions to compare
        seq_len (int): Length of sequence to generate
        batch_size (int, optional): Number of batches to generate. Defaults to 1
        sub_batch (int, optional): If provided, generates tokens for just this
            specific batch index. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - linreg_tokens (Tensor): Tokenized sequences with shape
              [batch, sequence_length]
            - data_store (List[Dict]): List of dictionaries containing
               comparison data for each batch

    """
    # Verify model has a tokenizer
    if not model.tokenizer:
        raise ValueError("Model must have a tokenizer attribute")

    # Cast tokenizer to correct type
    tokenizer = cast(PreTrainedTokenizer, model.tokenizer)

    # Verify tokenizer has required attributes
    if not hasattr(tokenizer, "bos_token_id"):
        raise ValueError("Model tokenizer must have bos_token_id")

    prefix = (t.ones(batch_size, 1) * tokenizer.bos_token_id).long().to(device)
    zero_token = model.to_tokens("0", truncate=True)[0][-1]

    # Create list to store tokens for each batch
    batch_tokens: List[t.Tensor] = []
    data_store: List[Dict[str, Any]] = []

    dataset_func = dataset

    # Generate tokens for each batch with different random seeds
    if sub_batch is not None:
        data = create_comparison_data(
            model,
            dataset_func,
            regressors,  # Type conversion handled by create_comparison_data
            random_state=sub_batch,
            seq_len=seq_len,
        )
        tokens = model.to_tokens(str(data["prompt"]), truncate=True)
        batch_tokens.append(tokens[0])
        data_store.append(data)
    else:
        for i in range(batch_size):
            data = create_comparison_data(
                model,
                dataset_func,
                regressors,  # Type conversion handled by create_comparison_data
                random_state=i,
                seq_len=seq_len,
            )
            tokens = model.to_tokens(str(data["prompt"]), truncate=True)
            batch_tokens.append(tokens[0])
            data_store.append(data)

    # Find the longest sequence length
    max_len = max(len(tensor) for tensor in batch_tokens)

    # Pad shorter sequences with token 0 at position -4
    for i in range(len(batch_tokens)):
        current_tensor = batch_tokens[i]
        while len(current_tensor) < max_len:
            print(
                f"Found mismatch in token length for batch {i}!\n"
                f"Largest length: {max_len}\n"
                f"Batch {i} length: {len(current_tensor)}\n"
                f"Applying padding..."
            )
            print(
                f"\nBefore Zero Token Padding:\n#####\n"
                f"{model.to_string(current_tensor[-50:])}\n#####"
            )
            batch_tokens[i] = t.cat(
                [
                    current_tensor[: len(current_tensor) - 3],
                    zero_token.unsqueeze(0),
                    current_tensor[len(current_tensor) - 3:],
                ]
            )
            current_tensor = batch_tokens[i]
            print(
                f"\nAfter Zero Token Padding:\n#####\n"
                f"{model.to_string(current_tensor[-50:])}\n#####"
            )

    # Stack all batches together
    linreg_tokens = t.stack(batch_tokens).to(device)

    # Add prefix to each batch
    linreg_tokens = t.cat([prefix, linreg_tokens], dim=-1).to(device)
    return linreg_tokens, data_store


def run_and_cache_model_linreg_tokens(
    model: HookedTransformer,
    dataset: Callable[
        [int],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[
        Callable[
            [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
            RegressionResults
        ]
    ],
    seq_len: int,
    batch: int = 1
) -> tuple[Tensor, Tensor, ActivationCache, List[Dict[str, Any]]]:
    """Generate tokens, run model and return results with cache.

    Args:
        model (HookedTransformer): The transformer model to run
        dataset: Dataset function or object to generate examples from
        regressors: List of regression models to compare against
        seq_len (int): Length of sequence for each example
        batch (int, optional): Batch size. Defaults to 1.

    Returns:
        tuple:
            - linreg_tokens (Tensor): Tokenized sequences with shape
              [batch, 1+seq_len]
            - linreg_logits (Tensor): Model logits with shape
              [batch, 1+seq_len, vocab_size]
            - linreg_cache (ActivationCache): Cache of model activations
            - linreg_data_store (list): List of dictionaries containing
              comparison data for each batch

    """
    linreg_tokens, linreg_data_store = generate_linreg_tokens(
        model, dataset, regressors, seq_len, batch
    )
    linreg_logits, linreg_cache = model.run_with_cache(linreg_tokens)
    return (
        linreg_tokens,
        cast(Tensor, linreg_logits),
        linreg_cache,
        linreg_data_store
    )


def run_and_cache_model_linreg_tokens_batched(
    model: HookedTransformer,
    dataset: Callable[
        [int],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[
        Callable[
            [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
            RegressionResults
        ]
    ],
    seq_len: int,
    total_batch: int = 1,
) -> tuple[
    List[Tensor],
    List[Tensor],
    List[ActivationCache],
    List[Dict[str, Any]]
]:
    """Generate batched linear regression tokens, run model and cache results.

    Args:
        model (HookedTransformer): The transformer model to run
        dataset: Dataset function or object to generate examples from
        regressors: List of regression models to compare against
        seq_len (int): Length of sequence for each example
        total_batch (int, optional): Total number of batches to process.
            Defaults to 1.

    Returns:
        tuple:
            - all_tokens (List[Tensor]): List of tokenized sequences with
              shape [1, 1+seq_len]
            - all_logits (List[Tensor]): List of model logits with shape
              [1, 1+seq_len, vocab_size]
            - all_caches (List[ActivationCache]): List of model activation
              caches stored on CPU
            - all_data_stores (List): List of dictionaries containing comparison
              data for each batch

    """
    all_tokens: List[Tensor] = []
    all_logits: List[Tensor] = []
    all_caches: List[ActivationCache] = []
    all_data_stores: List[Dict[str, Any]] = []

    # Process data in smaller batches
    for i in range(total_batch):
        # Generate and process current batch with sub_batch index
        batch_tokens, batch_data_store = generate_linreg_tokens(
            model,
            dataset,
            regressors,
            seq_len,
            batch_size=1,  # Since we're processing one at a time
            sub_batch=i,
        )
        batch_logits, batch_cache = model.run_with_cache(batch_tokens)

        # Move results to CPU to free GPU memory
        all_tokens.append(batch_tokens.cpu())
        all_logits.append(cast(Tensor, batch_logits).cpu())
        all_caches.append(batch_cache.to("cpu"))
        all_data_stores.extend(batch_data_store)

        # Optional: Clear CUDA cache
        if t.cuda.is_available():
            t.cuda.empty_cache()

    return all_tokens, all_logits, all_caches, all_data_stores

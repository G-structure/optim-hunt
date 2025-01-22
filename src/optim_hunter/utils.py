"""Entry point for data transformation utilities.

This module provides functions for handling data transformation, prompt
preparation,
and tokenization for language model interactions.
"""

from math import e
from typing import Dict, Hashable, List, Tuple, Union, cast, Callable, Any, Optional, Sequence

from optim_hunter.LR_methods import RegressionResults

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.utils import Sequence
import torch
from transformer_lens import (
    HookedTransformer,
)
import re


def prepare_prompt(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame
) -> str:
    """Prepare the prompt for model input from the linear / non linear dataset.

    Args:
        x_train (pd.DataFrame): Training features dataframe
        y_train (pd.Series): Training labels series
        x_test (pd.DataFrame): Test features dataframe

    Returns:
        str: The formatted prompt string containing training examples
             and test case

    """
    # Format numeric columns to 3 sig figs
    x_train = x_train.round(3)
    y_train = y_train.round(3)
    x_test = x_test.round(3)

    # Create the template for examples
    template = [
        f"{col_name}: {{{col_name}}}"
        for col_name in x_train.columns.astype(str)
    ]
    template.append(f"{y_train.name}: {{{y_train.name}}}")
    template = "\n".join(template)

    # Create suffix (test case format)
    suffix = [
        f"{col_name}: {{{col_name}}}"
        for col_name in x_train.columns.astype(str)
    ]
    suffix.append(f"{y_train.name}: ")
    suffix = "\n".join(suffix)

    # Format the test case using the suffix
    test_dict = cast(Dict[str, float], x_test.to_dict("records")[0])
    test_case = suffix.format(**test_dict)
    examples_text = []
    for x_row, y_val in zip(
        cast(List[Dict[str, float]], x_train.to_dict("records")),
        cast(npt.NDArray[np.float64], y_train.values),
        strict=False
    ):
        example_dict = {**x_row, y_train.name: y_val}
        formatted_example = template.format(**example_dict)
        examples_text.append(formatted_example)

    # Join all examples with double newlines
    examples_text = "\n\n".join(examples_text)

    # Add instruction prefix
    prefix_instruction = (
        'The task is to provide your best estimate for "Output". '
        'Please provide that and only that, without any additional text.'
        '\n\n\n\n\n'
    )

    # Combine everything
    final_prompt = f"{prefix_instruction}{examples_text}\n\n{test_case}"

    return final_prompt


def prepare_prompt_from_tokens(
    model: HookedTransformer,
    x_train_tokens: pd.DataFrame,  # DataFrame w/ tokenized training features
    y_train_tokens: pd.Series,     # Series w/ tokenized training labels
    x_test_tokens: pd.DataFrame,   # DataFrame w/ tokenized test features
    prepend_bos: bool = True,
    prepend_inst: bool = True,
) -> torch.Tensor:
    """Prepare a prompt tensor from pre-tokenized numeric data.

    Args:
        model: The language model with tokenizer
        x_train_tokens (pd.DataFrame): DataFrame containing tokenized training
            features
        y_train_tokens (pd.Series): Series containing tokenized training
            labels
        x_test_tokens (pd.DataFrame): DataFrame containing tokenized test
            features
        prepend_bos (bool): Whether to prepend the beginning of sequence token.
            Defaults to True.
        prepend_inst (bool): Whether to prepend instruction tokens.
            Defaults to True.

    Returns:
        torch.Tensor: A tensor of tokens representing the complete prompt with
            shape [1, sequence_length]

    """
    # Get tokens for static text elements
    instruction = model.to_tokens(
        'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.\n\n\n\n\n',  # noqa: E501
        prepend_bos=prepend_bos,
    )[0]

    newline = model.to_tokens("\n", prepend_bos=False)[0]
    double_newline = model.to_tokens("\n\n", prepend_bos=False)[0]
    colon_space = model.to_tokens(": ", prepend_bos=False)[0]

    # Initialize list to store all tokens
    all_tokens = []

    # Add instruction tokens
    if prepend_inst:
        all_tokens.extend(instruction.tolist())

    # Process training examples
    for idx in range(len(x_train_tokens)):
        if idx > 0:
            # Add separator between examples
            all_tokens.extend(double_newline.tolist())

        # Add features
        for col in x_train_tokens.columns:
            col = cast(str, col)
            # Add feature name
            feature_tokens = model.to_tokens(f"{col}", prepend_bos=False)[0]
            all_tokens.extend(feature_tokens.tolist())

            # Add colon and space
            all_tokens.extend(colon_space.tolist())

            # Add feature value
            value_tokens = cast(torch.Tensor, x_train_tokens[col].iloc[idx])
            all_tokens.extend(value_tokens.tolist())

            # Add newline
            all_tokens.extend(newline.tolist())

        # Add output label
        output_name_tokens = model.to_tokens("Output", prepend_bos=False)[0]
        all_tokens.extend(output_name_tokens.tolist())

        # Add colon and space
        all_tokens.extend(colon_space.tolist())

        # Add output value
        value = cast(torch.Tensor, y_train_tokens.iloc[idx])
        all_tokens.extend(value.tolist())

    # Add separator before test case
    all_tokens.extend(double_newline.tolist())

    # Print the numerical values in x_test_tokens
    print(f"Test values:")
    for col in x_test_tokens.columns:
        print(f"{col}: {model.to_string(x_test_tokens[col].iloc[0])}")
    # Add test features
    for col in x_test_tokens.columns:
        # Add feature name
        feature_tokens = model.to_tokens(f"{col}", prepend_bos=False)[0]
        all_tokens.extend(feature_tokens.tolist())

        # Add colon and space
        all_tokens.extend(colon_space.tolist())

        # Add feature value
        value = cast(torch.Tensor, x_test_tokens[col].iloc[0])
        all_tokens.extend(value.tolist())

        # Add newline
        all_tokens.extend(newline.tolist())

    # Add final "Output: "
    output_tokens = model.to_tokens("Output: ", prepend_bos=False)[0]
    all_tokens.extend(output_tokens.tolist())

    # Convert to tensor and ensure on correct device
    prompt_tensor = torch.tensor(all_tokens, device=model.cfg.device)

    return prompt_tensor.unsqueeze(0)  # Add batch dimension


def slice_dataset(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n: int = 10
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Slice the first n items from each dataset while preserving DataFrame.

    Args:
        x_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        x_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        n (int, optional): Number of items to keep. Defaults to 10.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple
        containing:
            - x_train_slice (pd.DataFrame): Sliced training features
            - y_train_slice (pd.Series): Sliced training labels
            - x_test_slice (pd.DataFrame): Sliced test features
            - y_test_slice (pd.Series): Sliced test labels

    """
    x_train_slice = cast(pd.DataFrame, x_train.iloc[:n])
    y_train_slice = cast(pd.Series, y_train.iloc[:n])
    x_test_slice = cast(pd.DataFrame, x_test.iloc[:n])
    y_test_slice = cast(pd.Series, y_test.iloc[:n])

    return x_train_slice, y_train_slice, x_test_slice, y_test_slice


def pad_numeric_tokens(
    model: HookedTransformer,
    x_train: Union[pd.DataFrame, List[pd.DataFrame]],
    y_train: Union[pd.Series, List[pd.Series]],
    x_test: Union[pd.DataFrame, List[pd.DataFrame]]
) -> Union[
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    Tuple[List[pd.DataFrame], List[pd.Series], List[pd.DataFrame]]
]:
    """Create new dataframes/lists with tokenized and padded numeric values.

    Args:
        model: The language model with tokenizer
        x_train: DataFrame or list of DataFrames with training features
        y_train: Series or list of Series with training labels
        x_test: DataFrame or list of DataFrames with test features

    Returns:
        tuple: (x_train_tokens, y_train_tokens, x_test_tokens) matching input
            type

    """
    # Convert to lists if single DataFrame/Series
    is_single = not isinstance(x_train, list)
    if is_single:
        x_train = [x_train]
        y_train = [cast(pd.Series, y_train)]
        x_test = [cast(pd.DataFrame, x_test)]

    # Get zero token for padding
    zero_token = model.to_tokens("0", truncate=True, prepend_bos=False)[0].cpu()

    # Format numeric columns to 3 sig figs
    x_train_rounded = [x_df.round(3) for x_df in x_train]
    y_train_rounded = [y_s.round(3) for y_s in cast(List[pd.Series], y_train)]
    x_test_rounded = [
        x_df.round(3) for x_df in cast(List[pd.DataFrame], x_test)
    ]


    # Function to tokenize a single number
    def tokenize_number(num: float) -> torch.Tensor:
        return model.to_tokens(str(num), prepend_bos=False, truncate=True)[
            0
        ].cpu()  # Move to CPU

    # Function to pad tokens to target length
    def pad_tokens(tokens: torch.Tensor, max_len: int) -> torch.Tensor:
        if len(tokens) < max_len:
            padding = zero_token.repeat(max_len - len(tokens))
            return torch.cat([tokens, padding])
        return tokens

    # Get all numeric values (both x and y)
    all_values: List[float] = []
    for x_df in x_train_rounded + x_test_rounded:
        for col in cast(List[str], x_df.columns):
            all_values.extend(cast(List[float], x_df[str(col)].values.tolist()))
    for y_series in y_train_rounded:
        all_values.extend(cast(List[float], y_series.values.tolist()))

    # Tokenize all values and find global maximum length
    all_tokenized = [tokenize_number(val) for val in all_values]
    global_max_len = max(len(tokens) for tokens in all_tokenized)

    # Process X training data
    x_train_tokens: List[pd.DataFrame] = []
    for x_df in x_train_rounded:
        x_tokens_df = pd.DataFrame(index=x_df.index)
        for col in cast(List[str], x_df.columns):
            x_tokens_df[str(col)] = [
                pad_tokens(tokenize_number(val), global_max_len)
                for val in cast(List[float], x_df[str(col)].values.tolist())
            ]
        x_train_tokens.append(x_tokens_df)

    # Process X test data
    x_test_tokens: List[pd.DataFrame] = []
    for x_df in x_test_rounded:
        x_tokens_df = pd.DataFrame(index=x_df.index)
        for col in cast(List[str], x_df.columns):
            x_tokens_df[str(col)] = [
                pad_tokens(tokenize_number(val), global_max_len)
                for val in cast(List[float], x_df[str(col)].values.tolist())
            ]
        x_test_tokens.append(x_tokens_df)

    # Process y values
    y_train_tokens: List[pd.Series] = []
    for y_series in y_train_rounded:
        y_tokens = pd.Series(
            [
                pad_tokens(tokenize_number(val), global_max_len)
                for val in cast(List[float], y_series.values.tolist())
            ],
            index=y_series.index,
        )
        y_train_tokens.append(y_tokens)

    # Return single items if input was single items
    if is_single:
        return (
            x_train_tokens[0],
            y_train_tokens[0],
            x_test_tokens[0]
        )

    return x_train_tokens, y_train_tokens, x_test_tokens


def prepare_dataset_prompts(
    dataset_fns: Union[Callable, List[Callable]],
    n_samples: int,
    model: HookedTransformer,
    seq_len: int = 10,  # Added parameter for data slicing
    random_seeds: List[int] = None
) -> List[Tuple[torch.Tensor, pd.DataFrame, str]]:
    """Prepare multiple dataset samples with corresponding tokenized prompts.

    Args:
        dataset_fns: Single dataset function or list of functions that generate
                    datasets (e.g. get_dataset_friedman_1)
        n_samples: Number of dataset samples to generate per dataset function
        model: The language model with tokenizer
        seq_len: Number of examples to include in each prompt. Defaults to 10.
        random_seeds: List of random seeds for dataset generation. If None,
                     uses range(n_samples)

    Returns:
        List[Tuple[torch.Tensor, pd.DataFrame, str]]: List of tuples containing:
            - tokenized prompt tensor
            - test features dataframe
            - dataset function name

    """
    # Input validation and setup
    if random_seeds is None:
        random_seeds = range(n_samples)
    elif len(random_seeds) < n_samples:
        raise ValueError(
            "Not enough random seeds provided for requested samples"
        )

    dataset_fns = (
        [dataset_fns] if not isinstance(dataset_fns, list) else dataset_fns
    )
    total_samples = len(dataset_fns) * n_samples

    # Pre-allocate lists with known sizes
    x_train_samples = [None] * total_samples
    y_train_samples = [None] * total_samples
    x_test_samples = [None] * total_samples
    dataset_names = [None] * total_samples

    # Generate all samples
    for i, dataset_fn in enumerate(dataset_fns):
        fn_name = dataset_fn.__name__
        base_idx = i * n_samples

        for j, seed in enumerate(random_seeds[:n_samples]):
            idx = base_idx + j
            # Generate dataset
            x_train, y_train, x_test, y_test = dataset_fn(random_state=seed)

            # Slice the dataset
            x_train, y_train, x_test, y_test = slice_dataset(
                x_train, y_train, x_test, y_test, n=seq_len
            )

            # Store samples
            x_train_samples[idx] = x_train
            y_train_samples[idx] = y_train
            x_test_samples[idx] = x_test
            dataset_names[idx] = fn_name

    # Pad all numeric tokens together
    x_train_tokens, y_train_tokens, x_test_tokens = pad_numeric_tokens(
        model=model,
        x_train=x_train_samples,
        y_train=y_train_samples,
        x_test=x_test_samples
    )

    # Pre-allocate result list
    prompts_and_data = [None] * total_samples

    # Create prompts for each padded sample
    for idx in range(total_samples):
        prompt_tensor = prepare_prompt_from_tokens(
            model=model,
            x_train_tokens=x_train_tokens[idx],
            y_train_tokens=y_train_tokens[idx],
            x_test_tokens=x_test_tokens[idx]
        )

        prompts_and_data[idx] = (
            prompt_tensor,
            x_test_samples[idx],
            dataset_names[idx]
        )

    return prompts_and_data


def create_regressor_results(
    dataset: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],  # Simplified since we don't use List[Tuple] case
    regressors: Sequence[Callable[
        [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],  # Input types
        RegressionResults  # Return type
    ]],
    random_state: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create DataFrame(s) with regressor predictions and performance metrics.

    Args:
        dataset: Single tuple or list of tuples containing:
            (x_train, y_train, x_test, y_test)
        regressors: List of regression functions to compare
        random_state: Random seed for reproducibility. Defaults to 1.

    Returns:
        Union[
            Tuple[pd.DataFrame, pd.DataFrame],
            List[Tuple[pd.DataFrame, pd.DataFrame]]
        ]:
            For single dataset:
                Tuple containing:
                - DataFrame with predictions and true values
                - DataFrame with MSE for each regressor
            For multiple datasets:
                List of such tuples

    """
    # Handle single dataset case
    is_single = not isinstance(dataset, list)
    datasets = [dataset] if is_single else dataset

    results = []

    for dataset_tuple in datasets:
        x_train, y_train, x_test, y_test = dataset_tuple

        # Initialize predictions dict with true values
        predictions = {
            'true_value': cast(np.ndarray, y_test.values)
        }

        # Initialize MSE dict
        mse_dict = {}

        # Get predictions and calculate MSE for each regressor
        for regressor in regressors:
            result = regressor(
                x_train,
                x_test,
                y_train,
                y_test,
                **{'random_state': random_state}
            )
            model_name = result.model_name
            model_predictions = result.y_predict

            # Store predictions
            predictions[model_name] = model_predictions

            # Get MSE from results metadata
            mse = result.metadata["performance_metrics"]["mse"]
            mse_dict[model_name] = mse

        # Create DataFrames
        predictions_df = pd.DataFrame(predictions)
        mse_df = pd.DataFrame([mse_dict]).T.rename(columns={0: 'MSE'})

        results.append((predictions_df, mse_df))

    # Return single tuple if input was single dataset
    if is_single:
        return cast(Tuple[pd.DataFrame, pd.DataFrame], results[0])

    return cast(Tuple[pd.DataFrame, pd.DataFrame], results)


def extract_model_prediction(
    model: HookedTransformer,
    prompt_tensor: torch.Tensor,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    sample_id: Optional[int] = None
) -> Optional[float]:
    """Extract numeric prediction from model generation output.

    Args:
        model: The transformer model
        prompt_tensor: Input prompt as tensor
        max_new_tokens: Maximum number of tokens to generate. Defaults to 16.
        temperature: Sampling temperature. Defaults to 0.0.
        sample_id: Optional sample identifier for warning messages.
                  Defaults to None.

    Returns:
        Optional[float]: Extracted numeric prediction or None if extraction
        fails

    """
    # Generate prediction
    pred_text = str(model.generate(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_type="str",
        prepend_bos=False
    ))

    # Extract the numeric prediction
    try:
        # Clean the prediction text
        generated_part = pred_text.split("Output: ")[-1].strip()
        pattern = r"[+-]?(?:\d*\.)?\d+"
        match = re.search(pattern, generated_part)
        if match:
            end_pos = match.end()
            generated_part = generated_part[:end_pos]
        return float(generated_part)
    except (ValueError, IndexError):
        sample_info = (
            f" for sample {sample_id}" if sample_id is not None else ""
        )
        print(f"Warning: Could not parse model prediction{sample_info}")
        return None

def extract_numeric_from_logits(
    model: HookedTransformer,
    model_out: torch.Tensor,
    sample_id: Optional[int] = None
) -> Optional[float]:
    """Extract numeric prediction from model.run_with_cache() output logits.

    Args:
        model: The transformer model
        model_out: Output logits from model.run_with_cache()
        sample_id: Optional sample identifier for warning messages.
                  Defaults to None.

    Returns:
        Optional[float]: Extracted numeric prediction or None if extraction fails

    """
    # Get most likely token predictions
    print("Model output shape:", model_out.shape)
    next_token = model_out.argmax(dim=-1)

    # Convert tokens to text
    try:
        pred_text = model.to_string(next_token)
        print(pred_text[0])
        # Clean the prediction text
        generated_part = pred_text.split("Output: ")[-1].strip()
        pattern = r"[+-]?(?:\d*\.)?\d+"
        match = re.search(pattern, generated_part)
        if match:
            end_pos = match.end()
            generated_part = generated_part[:end_pos]
        return float(generated_part)

    except (ValueError, IndexError):
        sample_info = (
            f" for sample {sample_id}" if sample_id is not None else ""
        )
        print(f"Warning: Could not parse model output{sample_info}")
        return None

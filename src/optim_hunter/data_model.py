"""Provides utilities for comparing regression model outputs."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from optim_hunter.LR_methods import RegressionResults
import numpy.typing as npt
import pandas as pd
import torch as t

from optim_hunter.utils import prepare_prompt, slice_dataset


def create_comparison_data(
    model: Any,
    dataset_func: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame,
        pd.Series]],
    regressors: List[Callable[..., RegressionResults]],
    random_state: int = 1,
    seq_len: Optional[int] = None
) -> Dict[str, Union[str, t.Tensor, Dict[str, float], List[str]]]:
    """Create a structured comparison dataset for analyzing regression models.

    Args:
        model (HookedTransformer): The transformer model used for tokenization
        dataset_func (callable): Function that returns (x_train, y_train, x_test
            y_test)
        regressors (list): List of regression functions to compare
        random_state (int, optional): Random seed for dataset generation.
            Defaults to 1
        seq_len (int, optional): Maximum sequence length to use.
                                 Defaults to None

    Returns:
        dict: A structured dictionary containing:
        {
            'dataset_name': str,  # Name of the dataset function
            'prompt': str,        # Generated prompt text for the model
            'predictions': {      # Dictionary of predictions from each model
                'gold': float,    # True value
                'model_name1': float,  # Prediction from first model
                'model_name2': float,  # Prediction from second model
                ...
            },
            'comparison_names': [  # List of comparison descriptors
                'model1 vs model2',
                'model1 vs model3',
                ...
            ],
            'token_pairs': tensor  # Shape: [num_comparisons, 1, 2]
                                # Each pair contains first tokens of predictions
        }

    Note:
         The function generates unique combinations (not permutations)
        - Only the first token of each prediction is stored in token_pairs
        - All possible combinations between gold and regressors are included
        - Token pairs maintain the order specified in comparison_names

    """
    # Get dataset
    x_train, y_train, x_test, y_test = dataset_func(random_state=random_state)
    if seq_len:
        x_train, y_train, x_test, y_test = slice_dataset(
            x_train, y_train, x_test, y_test, seq_len)

    # Get prompt
    prompt = prepare_prompt(x_train, y_train, x_test)

    # Get gold value
    gold = float(y_test.values[0])

    # Get predictions from each regressor
    predictions: Dict[str, float] = {}
    predictions['gold'] = gold
    for regressor in regressors:
        result = regressor(x_train, x_test, y_train, y_test)
        model_name = str(result.model_name)
        predictions[model_name] = float(result.y_predict[0])

    # Create comparison names and token pairs
    comparison_names: List[str] = []
    token_pairs: List[t.Tensor] = []

    # Create list of all predictors (including gold)
    all_predictors = ['gold'] + [
        str(reg(x_train, x_test, y_train, y_test).model_name)
        for reg in regressors
    ]

    # Generate unique combinations (not permutations)
    for i, pred1 in enumerate(all_predictors):
        for pred2 in all_predictors[i+1:]:
            comparison_name = f"{pred1} vs {pred2}"
            comparison_names.append(comparison_name)

            # Tokenize each prediction separately and get their first tokens
            tokens1 = model.to_tokens(str(predictions[pred1]),
                                   prepend_bos=False)[0, 0]
            tokens2 = model.to_tokens(str(predictions[pred2]),
                                   prepend_bos=False)[0, 0]

            # Combine the first tokens into a pair
            first_tokens = t.tensor([tokens1, tokens2],
                                 device=tokens1.device).unsqueeze(0)
            token_pairs.append(first_tokens)

    # Verify each comparison_name matches the corresponding token_pair
    for idx, (comp_name, token_pair) in enumerate(zip(comparison_names,
                                                     token_pairs, strict=True)):
        pred1_name, pred2_name = comp_name.split(' vs ')
        pred1_value = predictions[pred1_name]
        pred2_value = predictions[pred2_name]

        # Tokenize the actual prediction values
        actual_tokens1 = model.to_tokens(str(pred1_value),
                                      prepend_bos=False)[0, 0].item()
        actual_tokens2 = model.to_tokens(str(pred2_value),
                                      prepend_bos=False)[0, 0].item()

        # Extract tokens from token_pair
        token1, token2 = cast(List[int], token_pair.squeeze(0).tolist())

        # Assert that tokens match
        assert token1 == actual_tokens1, (
            f"Mismatch in token1 for comparison '{comp_name}' at index {idx}")
        assert token2 == actual_tokens2, (
            f"Mismatch in token2 for comparison '{comp_name}' at index {idx}")

    return {
        'dataset_name': dataset_func.__name__,
        'prompt': prompt,
        'predictions': predictions,
        'comparison_names': comparison_names,
        'token_pairs': t.stack(token_pairs),  # Shape: [num_comparisons, 1, 2]
    }

# # Create the data store
# datasets = [ get_dataset_friedman_2 ]
# regressors = [ linear_regression, knn_regression, random_forest,
# baseline_average, baseline_last, baseline_random ]
# data_store = {}
# for dataset_func in datasets:
#     data_store[dataset_func.__name__] = create_comparison_data(model,
#                                     dataset_func, regressors)
# # Print out the token pairs and comparison names
# for dataset_name, dataset_info in data_store.items():
#     print(f"\nDataset: {dataset_name}")
#     print("Token pairs:")
#     print(dataset_info['token_pairs'])
#     print("\nComparison names:")
#     print(dataset_info['comparison_names'])

# # Get the first token pair from the first dataset
# first_dataset_name = next(iter(data_store))
# first_token_pair = data_store[first_dataset_name]['token_pairs'][0]
# print("\nFirst token pair:")
# print(first_token_pair)

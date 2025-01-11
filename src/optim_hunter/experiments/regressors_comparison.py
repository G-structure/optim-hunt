"""Module for generating and comparing predictions between
machine learning models using regression.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy.typing as npt
import pandas as pd
import torch as t
from transformer_lens import (
    HookedTransformer,
)

from optim_hunter.logging_config import setup_logging
from optim_hunter.plot_html import create_bar_plot, with_identifier
from optim_hunter.utils import prepare_dataset_prompts, create_regressor_results, extract_model_prediction
import re

# Set up logging
setup_logging("DEBUG")

# Create logger for this module
logger = logging.getLogger(__name__)

t.set_grad_enabled(False)

# device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive set of regression metrics.

    Args:
        y_true: Array of true values
        y_pred: Array of predicted values

    Returns:
        Dict containing calculated metrics
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    # Basic error calculations
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Core metrics
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)

    # R-squared calculation
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # MAPE calculation (avoiding division by zero)
    non_zero_mask = y_true != 0
    mape = np.mean(abs_errors[non_zero_mask] / np.abs(y_true[non_zero_mask])) * 100 if any(non_zero_mask) else np.inf

    # Bias/Variance metrics
    bias = np.mean(errors)
    variance = np.var(y_pred)
    std_error = np.std(errors)

    # Distribution metrics
    max_error = np.max(abs_errors)
    error_25th = np.percentile(abs_errors, 25)
    error_50th = np.percentile(abs_errors, 50)
    error_75th = np.percentile(abs_errors, 75)

    return {
        # Core metrics
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape if mape != np.inf else None,

        # Bias/Variance metrics
        'Bias': bias,
        'Variance': variance,
        'Std_Error': std_error,

        # Distribution metrics
        'Max_Error': max_error,
        'Error_25th': error_25th,
        'Error_50th': error_50th,
        'Error_75th': error_75th
    }

def generate_and_compare_predictions(
    model: HookedTransformer,
    dataset_func: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    regressors: List[Callable[..., Dict[str, Union[str, npt.NDArray[Any]]]]],
    num_samples: int = 5,
    seq_len: Optional[int] = None
) -> Dict[str, Union[List[Dict[str, Any]], Dict[str, float]]]:
    """Generate model predictions and compare against regression baselines using
    MSE across multiple prompts.

    Args:
        model (HookedTransformer): The transformer model
        dataset_func (callable): Function that returns dataset splits
        regressors (list): List of regression functions to compare
        num_samples (int): Number of different prompts to generate and test
        seq_len (int, optional): Length to slice dataset

    Returns:
        dict: MSE scores and predictions for each sample

    """
    # Generate prompts and datasets
    prompts_and_data = prepare_dataset_prompts(
        dataset_fns=dataset_func,
        n_samples=num_samples,
        model=model,
        seq_len=seq_len or 10,
        random_seeds=list(range(num_samples))
    )

    all_results: List[Dict[str, Any]] = []

    # Process each prompt and dataset
    for i, (prompt_tensor, x_test, dataset_name) in enumerate(prompts_and_data):
        # Get dataset with same random seed
        dataset_tuple = dataset_func(random_state=i)

        # Get regressor predictions and MSE
        predictions_df, mse_df = create_regressor_results(
            dataset=dataset_tuple,
            regressors=regressors,
            random_state=i
        )

        # Get LLM prediction
        model_pred = extract_model_prediction(
            model=model,
            prompt_tensor=prompt_tensor,
            sample_id=i,
        )


        # Add LLM prediction to results
        true_value = float(predictions_df['true_value'].iloc[0])
        predictions = {
            "true_value": true_value,
            **{col: predictions_df[col].iloc[0]
               for col in predictions_df.columns
               if col != 'true_value'},
            "LLaMA-7B": model_pred  # Use model name instead of generic "llm"
        }

        # Calculate MSE for all predictions including LLM
        mse_scores = {name: float(mse_df.loc[name, 'MSE'])
                     for name in mse_df.index}
        if model_pred is not None:
            mse_scores['LLaMA-7B'] = float((model_pred - true_value) ** 2)

        sample_results = {
            "sample_id": i,
            "dataset": dataset_name,
            "predictions": predictions,
            "mse_scores": mse_scores
        }

        all_results.append(sample_results)

    # Calculate average MSE across all samples
    methods = set().union(*(r['mse_scores'].keys() for r in all_results))
    avg_mse = {method: [] for method in methods}

    for result in all_results:
        for method, mse in result['mse_scores'].items():
            if mse is not None:  # Only include valid MSE scores
                avg_mse[method].append(float(mse))

    # Calculate final averages, excluding any methods with no valid scores
    avg_mse_final = {
        method: sum(scores) / len(scores)
        for method, scores in avg_mse.items()
        if scores
    }

    return {
        "individual_results": all_results,
        "average_mse": avg_mse_final
    }

def compare_llm_and_regressors(
    dataset: Callable[
        ...,
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[Callable[..., Dict[str, Union[str, npt.NDArray[Any]]]]],
    seq_len: Optional[int],
    batches: int,
    model: HookedTransformer
) -> str:
    """Compare predictions between language model and regression baselines.

    Args:
        dataset: Function that returns dataset splits
        regressors: List of regression functions to compare
        seq_len: Optional sequence length to slice dataset
        batches: Number of batches/samples to generate
        model: The language model to evaluate

    Returns:
        str: HTML of the MSE comparison plot

    """
    results = generate_and_compare_predictions(
        model=model,
        dataset_func=dataset,
        regressors=regressors,
        num_samples=batches,
        seq_len=seq_len,
    )

    @with_identifier("mse-comparison")
    def create_mse_plot(
        results: Dict[str, Union[List[Dict[str, Any]], Dict[str, float]]]
    ) -> str:
        avg_mse = cast(Dict[str, float], results["average_mse"])
        return create_bar_plot(
            x_values=list(avg_mse.keys()),
            y_values=list(avg_mse.values()),
            title="Average MSE Across Methods",
            x_label="Method",
            y_label="Mean Squared Error",
            include_plotlyjs=True,
            include_theme_js=True,
        )

    mse_plot = create_mse_plot(results)
    return mse_plot

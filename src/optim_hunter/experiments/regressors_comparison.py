"""Module for comparing ML model predictions using regression."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from optim_hunter.LR_methods import RegressionResults
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as t
from transformer_lens import (
    HookedTransformer,
)

from optim_hunter.logging_config import setup_logging
from optim_hunter.plot_html import create_bar_plot, with_identifier
from optim_hunter.utils import (
    create_regressor_results,
    extract_model_prediction,
    prepare_dataset_prompts,
)

# Set up logging
setup_logging("DEBUG")

# Create logger for this module
logger = logging.getLogger(__name__)

t.set_grad_enabled(False)

# device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"

def calculate_metrics(
    y_true: Union[np.ndarray, RegressionResults],
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive set of regression metrics.

    Args:
        y_true: Array of true values or RegressionResults object
        y_pred: Array of predicted values (not needed if y_true is RegressionResults)

    Returns:
        Dict containing calculated metrics

    """
    # Handle RegressionResults input
    if isinstance(y_true, RegressionResults):
        y_true_array = y_true.y_test.values
        y_pred_array = y_true.y_predict
    else:
        y_true_array = y_true
        y_pred_array = y_pred if y_pred is not None else np.array([])
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
    if any(non_zero_mask):
        numerator = abs_errors[non_zero_mask]
        denominator = np.abs(y_true[non_zero_mask])
        mape = np.mean(numerator / denominator) * 100
    else:
        mape = np.inf

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
    regressors: List[Callable[..., RegressionResults]],
    num_samples: int = 5,
    seq_len: Optional[int] = None
) -> Dict[str, Union[List[Dict[str, Any]], Dict[str, float]]]:
    """Compare ML model predictions with various regression baselines.

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

        # Get dataset with same random seed
        x_train, y_train, x_test, y_test = dataset_tuple

        # Get regressor predictions
        regressor_results = []
        predictions = {"true_value": float(y_test.iloc[0])}
        mse_scores = {}

        for regressor in regressors:
            result = regressor(x_train, x_test, y_train, y_test, **{"random_state": i})
            regressor_results.append(result)
            predictions[result.model_name] = float(result.y_predict[0])
            mse_scores[result.model_name] = result.metadata["performance_metrics"]["mse"]

        # Get LLM prediction
        model_pred = extract_model_prediction(
            model=model,
            prompt_tensor=prompt_tensor,
            sample_id=i,
        )

        # Add LLM prediction to results
        if model_pred is not None:
            predictions["LLaMA-3.1 8B"] = model_pred
            true_value = float(y_test.iloc[0])
            mse_scores['LLaMA-3.1 8B'] = float((model_pred - true_value) ** 2)

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

def analyze_low_mse_seeds(
    results: Dict[str, Union[List[Dict[str, Any]], Dict[str, float]]]
) -> Tuple[List[int], str]:
    """Analyze and plot results for seeds with low MSE (<30,000) for LLaMA-7B.

    Args:
        results: Results dictionary from generate_and_compare_predictions

    Returns:
        Tuple of low MSE seeds list and HTML plot

    """
    individual_results = cast(List[Dict[str, Any]],
        results["individual_results"]
    )

    # Find seeds with low MSE
    low_mse_seeds = []
    low_mse_results = {}

    for result in individual_results:
        seed = result["sample_id"]
        if "LLaMA-3.1 8B" in result["mse_scores"]:
            llm_mse = result["mse_scores"]["LLaMA-3.1 8B"]
            if llm_mse < 30000:
                low_mse_seeds.append(seed)
                # Store all method scores for this seed
                low_mse_results[seed] = result["mse_scores"]

    # Create plot for low MSE seeds only
    @with_identifier("low-mse-comparison")
    def create_low_mse_plot() -> str:
        # Reorganize data for plotting
        methods = set().union(
            *(scores.keys() for scores in low_mse_results.values())
        )
        avg_mse = {method: [] for method in methods}

        for scores in low_mse_results.values():
            for method, mse in scores.items():
                if mse is not None:
                    avg_mse[method].append(float(mse))

        # Calculate averages for low MSE seeds
        avg_mse_final = {
            method: sum(scores) / len(scores)
            for method, scores in avg_mse.items()
            if scores
        }

        return create_bar_plot(
            x_values=list(avg_mse_final.keys()),
            y_values=list(avg_mse_final.values()),
            title=(f"Average MSE Across Methods "
                   f"(Low MSE Seeds: {low_mse_seeds})"),
            x_label="Method",
            y_label="Mean Squared Error",
            include_plotlyjs=True,
            include_theme_js=True
        )

    return low_mse_seeds, create_low_mse_plot()

def compare_llm_and_regressors(
    dataset: Callable[
        ..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[Callable[..., RegressionResults]],
    seq_len: Optional[int],
    batches: int,
    model: HookedTransformer
) -> str:
    """Compare predictions between language model and regression baselines.

    Returns:
        Tuple[str, str]: HTML of both the full MSE comparison plot and low MSE

    """
    results = generate_and_compare_predictions(
        model=model,
        dataset_func=dataset,
        regressors=regressors,
        num_samples=batches,
        seq_len=seq_len,
    )

    # Original MSE plot
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

    # Get low MSE analysis
    low_mse_seeds, low_mse_plot = analyze_low_mse_seeds(results)

    # Print low MSE seeds
    text = f"Seeds with LLM MSE < 30,000: {low_mse_seeds}"

    plots = f"{mse_plot} \n {text} \n {low_mse_plot}"

    return plots

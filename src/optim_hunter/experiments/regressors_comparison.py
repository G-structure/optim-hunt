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

from optim_hunter.data_model import create_comparison_data
from optim_hunter.logging_config import setup_logging
from optim_hunter.plot_html import create_bar_plot, with_identifier

# Set up logging
setup_logging("DEBUG")

# Create logger for this module
logger = logging.getLogger(__name__)

t.set_grad_enabled(False)

# device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"


def generate_and_compare_predictions(
    model: HookedTransformer,
    dataset_func: Callable[
        ...,
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[
        Callable[..., Dict[str, Union[str, npt.NDArray[Any]]]]
    ],
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
    all_results: List[Dict[str, Any]] = []

    # Generate multiple samples
    for i in range(num_samples):
        # Get data and predictions using create_comparison_data
        data = create_comparison_data(
            model, dataset_func, regressors,
            random_state=i, seq_len=seq_len
        )

        # Get the prompt
        prompt = str(data["prompt"])

        # Generate model prediction 4 tokens
        pred_text = str(
            model.generate(prompt, max_new_tokens=16, temperature=0)
        )

        # Extract the numeric prediction from the generated text
        try:
            # Clean the prediction text - remove the prompt and keep only the
            # generated part
            generated_part = str(pred_text.replace(prompt, "").strip())
            # Find first number with space after it
            import re

            pattern = r"[+-]?(?:\d*\.)?\d+"
            match = re.search(pattern, str(generated_part))
            if match:
                end_pos = match.end()
                generated_part = generated_part[:end_pos]
            model_pred = float(generated_part)
        except ValueError:
            print(
                f"Warning: Could not parse model prediction for sample {i}: "
                f"{pred_text}"
            )
            model_pred = None

        # Access predictions dictionary and get gold value safely using get()
        predictions_dict = cast(Dict[str, Any], data["predictions"])
        gold_value = float(predictions_dict.get("gold", 0))

        sample_results: Dict[str, Any] = {
            "sample_id": i,
            "predictions": {
                "llama 8b 4 tokens": model_pred,
                "gold": gold_value
            },
            "mse_scores": {},
        }

        # Add predictions from all regressors
        predictions = cast(Dict[str, Any], data["predictions"])
        for reg_name, pred_value in predictions.items():
            if reg_name != "gold":
                sample_results["predictions"][reg_name] = float(pred_value)

        # Calculate MSE scores for all predictions including the model's
        predictions_dict = cast(Dict[str, Any], sample_results["predictions"])
        for method, pred in predictions_dict.items():
            if method != "gold" and pred is not None:
                mse_scores = cast(Dict[str, float],
                                         sample_results["mse_scores"])
                mse_scores[method] = float((pred - gold_value) ** 2)

        all_results.append(sample_results)

    # Calculate average MSE across all samples
    avg_mse: Dict[str, List[float]] = {
        method: [] for method in all_results[0]["mse_scores"].keys()
    }
    for result in all_results:
        for method, mse in result["mse_scores"].items():
            avg_mse[method].append(float(mse))

    avg_mse_final: Dict[str, float] = {
        method: sum(scores) / len(scores)
        for method, scores in avg_mse.items()
    }

    return {"individual_results": all_results, "average_mse": avg_mse_final}

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

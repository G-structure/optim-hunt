"""Entry point for stacked regression comparisons.

This module implements functionality for comparing regression models
using a stacked prompt approach. It loads models, runs predictions,
and generates visualizations of the results.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
import torch as t
from transformer_lens import HookedTransformer

from optim_hunter.datasets import get_dataset_friedman_2, get_dataset_friedman_3
from optim_hunter.llama_model import load_llama_model
from optim_hunter.plot_html import (
    create_bar_plot,
    create_multi_line_plot,
    with_identifier,
)
from optim_hunter.sklearn_regressors import (
    adaboost,
    bagging,
    baseline_average,
    baseline_last,
    baseline_random,
    gradient_boosting,
    kernel_ridge_regression,
    lasso,
    linear_regression,
    mlp_deep1,
    mlp_deep2,
    mlp_deep3,
    mlp_universal_approximation_theorem1,
    mlp_universal_approximation_theorem2,
    mlp_universal_approximation_theorem3,
    random_forest,
    ridge,
    svm_and_scaler_regression,
    svm_regression,
)
from optim_hunter.utils import (
    pad_numeric_tokens,
    prepare_prompt_from_tokens,
    slice_dataset,
)

# Configure logging
logger = logging.getLogger(__name__)
t.set_grad_enabled(False)


def model_predict(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 16
) -> Optional[float]:
    """Generate and extract a numeric prediction from a language model.

    Args:
        model: The language model to use for prediction
        prompt: Input prompt text to feed to the model
        max_new_tokens: Maximum number of new tokens to generate.
                        Defaults to 16.

    Returns:
        float or None: The extracted numeric prediction,
                        or None if parsing fails

    The function:
    1. Generates text from the model using the prompt
    2. Extracts just the generated portion (removes prompt)
    3. Finds the first number in the generated text
    4. Converts it to float and returns it

    """
    pred_text: str = str(model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0
    ))

    # Extract the numeric prediction from the generated text
    try:
        # Clean prediction text - remove prompt and keep only generated part
        # This assumes the model's output follows the prompt
        generated_part: str = str(pred_text.replace(prompt, '').strip())

        # Find first number with space after it
        import re
        pattern = r'[+-]?(?:\d*\.)?\d+'
        match = re.search(pattern, generated_part)
        if match:
            end_pos = match.end()
            generated_part = generated_part[:end_pos]
        model_pred: Optional[float] = float(generated_part)
    except ValueError:
        print(f"Warning: Could not parse model prediction: {pred_text}")
        model_pred = None

    print(f"Model Pred: {model_pred}")
    return model_pred

def stacked_compare() -> Dict[
        str,
        Union[
            List[Dict[str, Any]],
            Dict[str, float]
        ]
    ]:
    """Compare predictions between multiple regression models stacking inputs.

    This function:
    1. Runs predictions with LLM and sklearn regressors on Friedman datasets
    2. Stacks multiple dataset prompts together for comparison
    3. Calculates MSE scores for each model
    4. Generates comparison visualizations

    Returns:
        Dict with:
            individual_results (List[Dict]): Per-batch predictions & MSE scores
            average_mse (Dict[str, float]): Mean MSE for each model

    """
    regressors = [
        linear_regression, ridge, lasso, mlp_universal_approximation_theorem1,
        mlp_universal_approximation_theorem2,
        mlp_universal_approximation_theorem3, mlp_deep1, mlp_deep2, mlp_deep3,
        random_forest, bagging, gradient_boosting, adaboost, svm_regression,
        svm_and_scaler_regression, kernel_ridge_regression, baseline_average,
        baseline_last, baseline_random
    ]
    datasets = [get_dataset_friedman_2, get_dataset_friedman_3]

    llama_model = load_llama_model()
    if llama_model is None:
        raise ValueError("Could not load LLaMA model")

    seq_len = 25  # Number of examples to show the model
    n_batches = 1 # should be even for stacked prompt

    # Lists to store batches
    x_trains: List[pd.DataFrame] = []
    y_trains: List[pd.Series] = []
    x_tests: List[pd.DataFrame] = []
    y_tests: List[pd.Series] = []
    tokenized_prompts: List[torch.Tensor] = []

    stack_prompts = True # Combine datasets into one big prompt
    double_instruct = True # when stacking prompts double up the inst message

    def stack_tokenized_prompts(
        model: HookedTransformer,
        tokens_1: List[torch.Tensor],
        tokens_2: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Stack tokenized prompts from two datasets with separator.

        Args:
            model: The LLaMA model instance
            tokens_1: List of tokenized prompts from first dataset
            tokens_2: List of tokenized prompts from second dataset

        Returns:
            List of stacked tokenized prompts

        """
        separator = model.to_tokens('\n\n\n\n\n', prepend_bos=False)[0]
        stacked_prompts: List[torch.Tensor] = []

        for t1, t2 in zip(tokens_1, tokens_2):
            stacked = torch.cat([
                t1,
                separator.repeat(t1.size(0), 1),
                t2
            ], dim=1)
            stacked_prompts.append(stacked)

        return stacked_prompts

    for i, dataset in enumerate(datasets):
        print(i)
        temp_tokenized_prompts: List[torch.Tensor] = []
        prepend_bos = not stack_prompts or i == 0
        prepend_inst = not stack_prompts or double_instruct or i == 0

        for seed in range(n_batches):
            x_train, y_train, x_test, y_test = dataset(random_state=seed)
            x_train, y_train, x_test, y_test = slice_dataset(
                x_train, y_train, x_test, y_test, seq_len
            )

            if stack_prompts and i == 0:
                x_trains.append(x_train)
                y_trains.append(y_train)
                x_tests.append(x_test)
                y_tests.append(y_test)

            x_train_tokens, y_train_tokens, x_test_tokens = cast(
                Tuple[pd.DataFrame, pd.Series, pd.DataFrame],
                pad_numeric_tokens(llama_model, x_train, y_train, x_test)
            )
            tokenized_prompt = prepare_prompt_from_tokens(
                llama_model, x_train_tokens, y_train_tokens, x_test_tokens,
                prepend_bos=prepend_bos, prepend_inst=prepend_inst
            )
            temp_tokenized_prompts.append(tokenized_prompt)

        if i == 0:
            tokenized_prompts = temp_tokenized_prompts
        elif stack_prompts:
            tokenized_prompts = stack_tokenized_prompts(
                llama_model, tokenized_prompts, temp_tokenized_prompts
            )
        else:
            tokenized_prompts.extend(temp_tokenized_prompts)

    batched_prompts = torch.cat(tokenized_prompts, dim=0)
    print(f"\nFinal batched prompts shape: {batched_prompts.shape}")

    print(f"\n X trains: {len(x_trains)}")
    print(f"\n Y trains: {len(y_trains)}")
    print(f"\n X tests: {len(x_tests)}")
    print(f"\n X tests: {len(y_tests)}")

    x_trains = x_trains[:len(x_trains)//2]
    y_trains = y_trains[:len(y_trains)//2]
    x_tests = x_tests[:len(x_tests)//2]
    y_tests = y_tests[:len(y_tests)//2]

    all_results: List[Dict[str, Any]] = []

    for i in range(len(x_trains)):
        batch_results: Dict[str, Any] = {
            'batch_id': i,
            'predictions': {},
            'mse_scores': {}
        }

        x_train = x_trains[i]
        y_train = y_trains[i]
        x_test = x_tests[i]
        y_test = y_tests[i]

        prompt = cast(str, llama_model.to_string(tokenized_prompts[i][0]))
        llm_pred = model_predict(llama_model, prompt)
        batch_results['predictions']['llm'] = llm_pred

        for regressor in regressors:
            reg_result = regressor(x_train, x_test, y_train, y_test)
            reg_name = reg_result['model_name']
            reg_pred = cast(float, reg_result['y_predict'][0])
            batch_results['predictions'][reg_name] = reg_pred

        gold = float(cast(float, y_test.iloc[0]))
        print(f"Gold value: {gold}")
        batch_results['predictions']['gold'] = gold

        for method_name, pred in batch_results['predictions'].items():
            if method_name != 'gold' and pred is not None:
                try:
                    gold_value = float(batch_results['predictions']['gold'])
                    pred_value = float(pred)
                    mse = (pred_value - gold_value) ** 2
                    batch_results['mse_scores'][method_name] = mse
                except (ValueError, TypeError) as e:
                    print(f"Error calculating MSE for {method_name}: {e}")
                    continue

        all_results.append(batch_results)

    avg_mse: Dict[str, float] = {}
    method_set: set[str] = set().union(
        *(r['mse_scores'].keys() for r in all_results)
    )
    for method_name in method_set:
        scores = [
            r['mse_scores'][method_name] for r in all_results
            if method_name in r['mse_scores']
        ]
        if scores:
            avg_mse[method_name] = sum(scores) / len(scores)

    @with_identifier("predictions-comparison")
    def create_predictions_plot(all_results: List[Dict[str, Any]]) -> str:
        methods = list(all_results[0]['predictions'].keys())
        y_values_list: List[List[float]] = []
        labels: List[str] = []

        for method in methods:
            y_values = [
                float(r['predictions'][method]) for r in all_results
                if method in r['predictions'] and
                r['predictions'][method] is not None
            ]
            if y_values:
                y_values_list.append(y_values)
                labels.append(str(method))

        return create_multi_line_plot(
            y_values_list=y_values_list,
            labels=labels,
            title='Predictions Across Batches',
            x_label='Batch',
            y_label='Prediction Value',
            include_plotlyjs=True,
            include_theme_js=True
        )

    @with_identifier("mse-comparison")
    def create_mse_plot(avg_mse: Dict[str, float]) -> str:
            sorted_items = sorted(avg_mse.items(), key=lambda x: x[1])
            x_values = [str(item[0]) for item in sorted_items]
            y_values = [item[1] for item in sorted_items]

            if y_values and max(y_values) / (min(y_values) + 1e-10) > 1000:
                y_values_for_plot = [float(np.log10(v)) for v in y_values]
                y_label = 'Log10(MSE)'
            else:
                y_values_for_plot = [float(v) for v in y_values]
                y_label = 'Mean Squared Error'

            return create_bar_plot(
                x_values=[str(x) for x in x_values],
                y_values=y_values_for_plot,
                title='Average MSE Across Methods',
                x_label='Method',
                y_label=y_label,
                include_plotlyjs=True,
                include_theme_js=True
            )

    predictions_plot = create_predictions_plot(all_results)
    mse_plot = create_mse_plot(avg_mse)

    print(predictions_plot)
    print(mse_plot)

    return {
        'individual_results': all_results,
        'average_mse': avg_mse
    }

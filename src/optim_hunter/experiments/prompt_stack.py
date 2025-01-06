from optim_hunter.datasets import get_dataset_friedman_2, get_dataset_friedman_3
from optim_hunter.utils import slice_dataset, prepare_prompt_from_tokens, pad_numeric_tokens
from optim_hunter.llama_model import load_llama_model
from optim_hunter.sklearn_regressors import (
    linear_regression, ridge, lasso, mlp_universal_approximation_theorem1,
    mlp_universal_approximation_theorem2, mlp_universal_approximation_theorem3,
    mlp_deep1, mlp_deep2, mlp_deep3, random_forest, bagging,
    gradient_boosting, adaboost, svm_regression, svm_and_scaler_regression, kernel_ridge_regression,
    baseline_average, baseline_last, baseline_random
)
import torch
from optim_hunter.plot_html import with_identifier, create_bar_plot, create_multi_line_plot
import numpy as np

def model_predict(model, prompt, max_new_tokens=16):
    """Generate and extract a numeric prediction from a language model.

    Args:
        model: The language model to use for prediction
        prompt: Input prompt text to feed to the model

    Returns:
        float or None: The extracted numeric prediction, or None if parsing fails

    The function:
    1. Generates text from the model using the prompt
    2. Extracts just the generated portion (removes prompt)
    3. Finds the first number in the generated text
    4. Converts it to float and returns it

    """
    pred_text = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)
    # No need to convert to string since generate() returns string directly

    # Extract the numeric prediction from the generated text
    try:
        # Clean the prediction text - remove the prompt and keep only the generated part
        # This assumes the model's output follows the prompt
        generated_part = pred_text.replace(prompt, '').strip()
        # Find first number with space after it
        import re
        pattern = r'[+-]?(?:\d*\.)?\d+'
        match = re.search(pattern, generated_part)
        if match:
            end_pos = match.end()
            generated_part = generated_part[:end_pos]
        model_pred = float(generated_part)
    except ValueError:
        print(f"Warning: Could not parse model prediction: {pred_text}")
        model_pred = None

    print(f"Model Pred: {model_pred}")
    return model_pred

def stacked_compare():
    regressors = [ linear_regression, ridge, lasso, mlp_universal_approximation_theorem1, mlp_universal_approximation_theorem2, mlp_universal_approximation_theorem3, mlp_deep1, mlp_deep2, mlp_deep3, random_forest, bagging, gradient_boosting, adaboost, svm_regression, svm_and_scaler_regression, kernel_ridge_regression, baseline_average, baseline_last, baseline_random]
    datasets = [get_dataset_friedman_2, get_dataset_friedman_3]

    llama_model = load_llama_model()
    seq_len = 25  # Number of examples to show the model
    n_batches = 1 # should be even for stached prompt

    # Lists to store batches
    x_trains, y_trains, x_tests, y_tests = [], [], [], []
    prompts = []
    tokenized_prompts = []

    stack_prompts = True # Combine datasets into one big prompt
    double_instruct = True # when stacking prompts double up the inst message

    def stack_tokenized_prompts(model, tokens_1, tokens_2):
        """Stack tokenized prompts from two datasets with a separator.

        Args:
            model: The LLaMA model instance
            tokens_1: List of tokenized prompts from first dataset
            tokens_2: List of tokenized prompts from second dataset

        Returns:
            List of stacked tokenized prompts

        """
        # Get separator tokens (multiple newlines)
        separator = model.to_tokens('\n\n\n\n\n', prepend_bos=False)[0]

        # Initialize list for stacked prompts
        stacked_prompts = []

        # Stack corresponding prompts from both datasets
        for t1, t2 in zip(tokens_1, tokens_2):
            # Concatenate prompts with separator
            stacked = torch.cat([
                t1,  # Shape: [batch_size, seq_len_1]
                separator.repeat(t1.size(0), 1),  # Shape: [batch_size, separator_len]
                t2   # Shape: [batch_size, seq_len_2]
            ], dim=1)
            stacked_prompts.append(stacked)

        return stacked_prompts

    for i, dataset in enumerate(datasets):
        print(i)
        temp_tokenized_prompts = []
        # Set default values and adjust based on stacking configuration
        prepend_bos = not stack_prompts or i == 0
        prepend_inst = not stack_prompts or double_instruct or i == 0
        # Generate batches with different seeds
        for seed in range(n_batches):
            # Get dataset with current seed
            x_train, y_train, x_test, y_test = dataset(random_state=seed)

            # Slice the dataset
            x_train, y_train, x_test, y_test = slice_dataset(
                x_train, y_train, x_test, y_test, seq_len
            )

            # Store the batches
            if stack_prompts and i == 0:
                x_trains.append(x_train)
                y_trains.append(y_train)
                x_tests.append(x_test)
                y_tests.append(y_test)

            # Create tokenized prompt
            x_train_tokens, y_train_tokens, x_test_tokens = pad_numeric_tokens(
                llama_model, x_train, y_train, x_test
            )
            tokenized_prompt = prepare_prompt_from_tokens(
                llama_model, x_train_tokens, y_train_tokens, x_test_tokens, prepend_bos=prepend_bos, prepend_inst=prepend_inst
            )
            temp_tokenized_prompts.append(tokenized_prompt)

        if i == 0:
            tokenized_prompts = temp_tokenized_prompts
        elif stack_prompts:
            tokenized_prompts = stack_tokenized_prompts(llama_model, tokenized_prompts, temp_tokenized_prompts)
        else:
            tokenized_prompts.extend(temp_tokenized_prompts)

    # Stack all tokenized prompts into a single batch
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

    all_results = []  # Initialize the list

    for i in range(len(x_trains)):
        batch_results = {
            'batch_id': i,
            'predictions': {},
            'mse_scores': {}
        }

        # Get current batch data
        x_train = x_trains[i]
        y_train = y_trains[i]
        x_test = x_tests[i]
        y_test = y_tests[i]

        # Get LLM prediction
        prompt = llama_model.to_string(tokenized_prompts[i][0])
        llm_pred = model_predict(llama_model, prompt)
        batch_results['predictions']['llm'] = llm_pred

        # Get predictions from each regressor
        for regressor in regressors:
            reg_result = regressor(x_train, x_test, y_train, y_test)
            reg_name = reg_result['model_name']
            reg_pred = reg_result['y_predict'][0]
            batch_results['predictions'][reg_name] = reg_pred

        # Ground truth value
        gold = y_test
        print(f"Gold value: {gold}")
        batch_results['predictions']['gold'] = gold

        for method, pred in batch_results['predictions'].items():
            if method != 'gold' and pred is not None:
                try:
                    gold_value = float(batch_results['predictions']['gold'])
                    pred_value = float(pred)
                    mse = (pred_value - gold_value) ** 2
                    if not np.isnan(mse) and not np.isinf(mse):
                        batch_results['mse_scores'][method] = mse
                except (ValueError, TypeError) as e:
                    print(f"Error calculating MSE for {method}: {e}")
                    continue

        # Add this line to append the batch results
        all_results.append(batch_results)

    # Calculate average MSE across batches
    avg_mse = {}
    for method in set().union(*(r['mse_scores'].keys() for r in all_results)):
        scores = [r['mse_scores'][method] for r in all_results
                    if method in r['mse_scores']
                    and r['mse_scores'][method] is not None
                    and not np.isnan(r['mse_scores'][method])
                    and not np.isinf(r['mse_scores'][method])]
        if scores:  # Only calculate average if we have valid scores
            avg_mse[method] = sum(scores) / len(scores)

    # Create HTML plots
    @with_identifier("predictions-comparison")
    def create_predictions_plot(all_results):
        methods = list(all_results[0]['predictions'].keys())
        x_values = [f"Batch {r['batch_id']}" for r in all_results]

        # Create list of y-values for each method
        y_values_list = []
        labels = []

        for method in methods:
            y_values = [r['predictions'][method] for r in all_results
                       if method in r['predictions'] and r['predictions'][method] is not None]
            if y_values:  # Only add if we have values
                y_values_list.append(y_values)
                labels.append(method)

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
    def create_mse_plot(avg_mse):
        # Filter out any None values and handle potential numerical issues
        filtered_mse = {
            k: v for k, v in avg_mse.items()
            if v is not None and not np.isnan(v) and not np.isinf(v)
        }

        # Sort by MSE value for better visualization
        sorted_items = sorted(filtered_mse.items(), key=lambda x: x[1])
        x_values = [item[0] for item in sorted_items]
        y_values = [item[1] for item in sorted_items]

        # Optionally log transform values if they span many orders of magnitude
        y_values = np.log10(y_values) if max(y_values) / (min(y_values) + 1e-10) > 1000 else y_values

        return create_bar_plot(
            x_values=x_values,
            y_values=y_values,
            title='Average MSE Across Methods',
            x_label='Method',
            y_label='Log10(MSE)' if y_values != y_values else 'Mean Squared Error',
            include_plotlyjs=True,
            include_theme_js=True
        )

    # Generate and print plots
    predictions_plot = create_predictions_plot(all_results)
    mse_plot = create_mse_plot(avg_mse)

    print(predictions_plot)
    print(mse_plot)

    return {
        'individual_results': all_results,
        'average_mse': avg_mse
    }

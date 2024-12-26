import sys
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
import functools
from tqdm import tqdm
from IPython.display import display
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv

from optim_hunter.plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference, line
from optim_hunter.utils import prepare_prompt, slice_dataset
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.data_model import create_comparison_data
from optim_hunter.plot_html import get_theme_sync_js, create_line_plot, with_identifier, create_bar_plot
from optim_hunter.llama_model import load_llama_model
from optim_hunter.model_utils import run_and_cache_model_linreg_tokens_batched, run_and_cache_model_linreg_tokens
from typing import List, Tuple

import logging
from optim_hunter.logging_config import setup_logging

# Set up logging
setup_logging("DEBUG")

# Create logger for this module
logger = logging.getLogger(__name__)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

#device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"

def generate_and_compare_predictions(model, dataset_func, regressors, num_samples=5, seq_len=None):
    """
    Generate model predictions and compare against regression baselines using MSE across multiple prompts
    
    Args:
        model (HookedTransformer): The transformer model
        dataset_func (callable): Function that returns dataset splits
        regressors (list): List of regression functions to compare
        num_samples (int): Number of different prompts to generate and test
        seq_len (int, optional): Length to slice dataset
        
    Returns:
        dict: MSE scores and predictions for each sample
    """
    all_results = []
    
    # Generate multiple samples
    for i in range(num_samples):
        # Get data and predictions using create_comparison_data
        data = create_comparison_data(model, dataset_func, regressors, random_state=i, seq_len=seq_len)
        
        # Get the prompt
        prompt = data['prompt']
        
        # Generate model prediction
        pred_text = model.generate(prompt, max_new_tokens=4, temperature=0)
        # No need to convert to string since generate() returns string directly
        
        # Extract the numeric prediction from the generated text
        try:
            # Clean the prediction text - remove the prompt and keep only the generated part
            # This assumes the model's output follows the prompt
            generated_part = pred_text.replace(prompt, '').strip()
            model_pred = float(generated_part)
        except ValueError:
            print(f"Warning: Could not parse model prediction for sample {i}: {pred_text}")
            model_pred = None
            
        # Get gold value and regression predictions
        sample_results = {
            'sample_id': i,
            'predictions': {
                'llama': model_pred,  # Changed 'model' to 'llama' for clarity
                'gold': data['predictions']['gold'],
            },
            'mse_scores': {}
        }
        
        # Add predictions from all regressors
        for reg_name, pred_value in data['predictions'].items():
            if reg_name != 'gold':
                sample_results['predictions'][reg_name] = pred_value
        
        # Calculate MSE scores for all predictions including the model's
        gold = sample_results['predictions']['gold']
        for method, pred in sample_results['predictions'].items():
            if method != 'gold' and pred is not None:
                sample_results['mse_scores'][method] = (pred - gold) ** 2
                
        all_results.append(sample_results)
    
    # Calculate average MSE across all samples
    avg_mse = {method: [] for method in all_results[0]['mse_scores'].keys()}
    for result in all_results:
        for method, mse in result['mse_scores'].items():
            avg_mse[method].append(mse)
    
    avg_mse = {method: sum(scores)/len(scores) for method, scores in avg_mse.items()}
    
    return {
        'individual_results': all_results,
        'average_mse': avg_mse
    }

def plot_comparison_results(results):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Individual Sample Predictions', 'Average MSE Across Samples'),
                       vertical_spacing=0.3)
    
    # Colors for different methods
    methods = list(results['individual_results'][0]['predictions'].keys())
    methods.remove('gold')
    colors = px.colors.qualitative.Set3[:len(methods)]
    color_map = dict(zip(methods, colors))
    
    # Plot individual predictions
    for method in methods:
        x_vals = []
        y_vals = []
        for sample in results['individual_results']:
            x_vals.append(f"Sample {sample['sample_id']}")
            y_vals.append(sample['predictions'][method])
        
        fig.add_trace(
            go.Scatter(
                name=method,
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=color_map[method])
            ),
            row=1, col=1
        )
        
        # Add gold values
        if method == methods[0]:  # Only add gold once
            gold_vals = [sample['predictions']['gold'] for sample in results['individual_results']]
            fig.add_trace(
                go.Scatter(
                    name='Gold',
                    x=x_vals,
                    y=gold_vals,
                    mode='lines+markers',
                    line=dict(color='black', dash='dash')
                ),
                row=1, col=1
            )
    
    # Plot average MSE
    fig.add_trace(
        go.Bar(
            name='Average MSE',
            x=list(results['average_mse'].keys()),
            y=list(results['average_mse'].values()),
            marker_color=[color_map[method] for method in results['average_mse'].keys()]
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Model vs Regression Methods Comparison Across Multiple Samples',
        showlegend=True,
        height=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Prediction Value", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=2, col=1)
    
    return fig

def compare_llm_and_regressors(dataset, regressors):
    model = load_llama_model()

    results = generate_and_compare_predictions(
        model=model,
        dataset_func=dataset,
        regressors=regressors,
        num_samples=1,
        seq_len=None
    )

    @with_identifier("mse-comparison")
    def create_mse_plot(results):
        return create_bar_plot(
            x_values=list(results['average_mse'].keys()),
            y_values=list(results['average_mse'].values()),
            title='Average MSE Across Methods',
            x_label='Method',
            y_label='Mean Squared Error',
            include_plotlyjs=True,
            include_theme_js=True
        )
    mse_plot = create_mse_plot(results)
    
    # Output the HTML
    print(mse_plot)
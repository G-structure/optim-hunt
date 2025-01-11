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
from optim_hunter.utils import prepare_prompt, slice_dataset, prepare_dataset_prompts
from optim_hunter.sklearn_regressors import linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random
from optim_hunter.datasets import get_dataset_friedman_2
from optim_hunter.data_model import create_comparison_data
from optim_hunter.plot_html import get_theme_sync_js, create_line_plot, with_identifier, create_multi_line_plot, create_multi_line_plot_layer_names
from optim_hunter.llama_model import load_llama_model
from optim_hunter.model_utils import run_and_cache_model_linreg_tokens_batched, run_and_cache_model_linreg_tokens
from typing import List, Tuple, Callable, Dict,Union, Optional
import pandas as pd

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

def generate_logit_diff_batched(
    dataset: Callable,
    regressors: List[Callable],
    seq_len: int,
    batches: int,
    model: HookedTransformer,
    random_seeds: Optional[List[int]] = None
) -> str:
    """Generate logit difference analysis across batches without storing activations."""
    # Generate all prompts and datasets upfront
    prompts_and_data = prepare_dataset_prompts(
        dataset_fns=dataset,
        n_samples=batches,
        model=model,
        seq_len=seq_len,
        random_seeds=random_seeds if random_seeds is not None else list(range(batches))
    )

    # Get comparison names and token pairs from first batch to verify consistency
    dataset_tuple = dataset(random_state=0)
    first_comparison = create_comparison_data(
        model, dataset, regressors, random_state=0, seq_len=seq_len
    )
    token_pairs_names = first_comparison["comparison_names"]
    base_token_pairs = first_comparison["token_pairs"]

    # Move base_token_pairs to the same device as model
    base_token_pairs = base_token_pairs.to(model.cfg.device)

    # Initialize accumulators for averaging
    accumulated_diffs_sum = None
    per_layer_diffs_sum = None
    n_valid_batches = 0
    labels = None

    # Process each prompt
    for batch_idx, (prompt_tensor, x_test, dataset_name) in enumerate(prompts_and_data):
        logger.info(f"Processing batch {batch_idx}")

        # Ensure prompt tensor is on correct device
        prompt_tensor = prompt_tensor.to(model.cfg.device)

        # Run model and get logits
        logits, cache = model.run_with_cache(prompt_tensor)

        cache = cache.to("cpu")

        # Get final residual stream
        final_residual_stream = cache["resid_post", -1]
        final_token_residual_stream = final_residual_stream[:, -1, :]

        # Process each token pair
        batch_accumulated_diffs = []
        batch_per_layer_diffs = []

        for pair_idx, token_pair in enumerate(base_token_pairs):
            # Ensure token pair is on correct device
            token_pair = token_pair.to("cuda:1")

            # Compute residual directions
            pair_residual_directions = model.tokens_to_residual_directions(token_pair)
            correct_residual_directions, incorrect_residual_directions = pair_residual_directions.unbind(dim=1)
            logit_diff_directions = correct_residual_directions - incorrect_residual_directions

            # Move logit diff directions to cuda:1
            logit_diff_directions = logit_diff_directions.to('cuda')
            # Accumulate residuals
            accumulated_residual, curr_labels = cache.accumulated_resid(
                layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
            )
            if labels is None:
                labels = curr_labels

            # Calculate logit differences
            logit_lens_diffs = residual_stack_to_logit_diff(
                accumulated_residual, cache, logit_diff_directions
            ).half()
            batch_accumulated_diffs.append(logit_lens_diffs)

            # Per layer analysis
            per_layer_residual, _ = cache.decompose_resid(
                layer=-1, pos_slice=-1, return_labels=True
            )
            per_layer_diffs = residual_stack_to_logit_diff(
                per_layer_residual, cache, logit_diff_directions
            )
            batch_per_layer_diffs.append(per_layer_diffs)

        # Stack batch results
        batch_accumulated_diffs = t.stack(batch_accumulated_diffs)
        batch_per_layer_diffs = t.stack(batch_per_layer_diffs)

        # Add to running sums
        if accumulated_diffs_sum is None:
            accumulated_diffs_sum = batch_accumulated_diffs
            per_layer_diffs_sum = batch_per_layer_diffs
        else:
            accumulated_diffs_sum += batch_accumulated_diffs
            per_layer_diffs_sum += batch_per_layer_diffs

        n_valid_batches += 1

        # Clear cache to free memory
        del cache
        if t.cuda.is_available():
            t.cuda.empty_cache()

    # Calculate averages
    avg_accumulated_diffs = accumulated_diffs_sum / n_valid_batches
    avg_per_layer_diffs = per_layer_diffs_sum / n_valid_batches

    # Create plots
    plots = create_logit_diff_plots(
        avg_accumulated_diffs,
        avg_per_layer_diffs,
        token_pairs_names,
        labels
    )

    return plots

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    """Calculate logit differences from residual stack."""
    scaled_residual_stream = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    logit_diff_directions = logit_diff_directions.to(
        dtype=scaled_residual_stream.dtype
    ).to('cpu')

    batch_size = residual_stack.size(-2)
    avg_logit_diff = einops.einsum(
        scaled_residual_stream,
        logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size
    return avg_logit_diff

def create_logit_diff_plots(
    accumulated_diffs: Tensor,
    per_layer_diffs: Tensor,
    token_pairs_names: List[str],
    layer_names: List[str]
) -> str:
    """Create plots for logit difference analysis."""
    @with_identifier("accumulated-residual-plot")
    def create_accumulated_residual_plot():
        return create_multi_line_plot_layer_names(
            y_values_list=accumulated_diffs,
            labels=token_pairs_names,
            title="Average Logit Difference From Accumulated Residual Stream",
            x_label="Layer",
            y_label="Logit Diff",
            layer_names=layer_names,
            include_plotlyjs=True,
            include_theme_js=False,
            active_lines=[-4]
        )

    @with_identifier("per-layer-plot")
    def create_per_layer_plot():
        return create_multi_line_plot_layer_names(
            y_values_list=per_layer_diffs,
            labels=token_pairs_names,
            title="Average Per Layer Logit Difference",
            x_label="Layer",
            y_label="Logit Diff",
            layer_names=layer_names,
            include_plotlyjs=False,
            include_theme_js=True,
            active_lines=[-4]
        )

    accumulated_plot = create_accumulated_residual_plot()
    per_layer_plot = create_per_layer_plot()

    return f"{accumulated_plot}\n{per_layer_plot}"

"""Module for analyzing and plotting logit differences of transformer layers."""
import logging
from typing import Callable, List, Optional

import einops
import torch
import torch as t
from jaxtyping import Float
from torch import Tensor
from transformer_lens import (
    ActivationCache,
    HookedTransformer,
)

from optim_hunter.data_model import create_comparison_data
from optim_hunter.logging_config import setup_logging
from optim_hunter.plot_html import (
    create_multi_line_plot_layer_names,
    with_identifier,
)

from optim_hunter.utils import (
    prepare_prompt,
)

# Set up logging
setup_logging("DEBUG")

# Create logger for this module
logger = logging.getLogger(__name__)

# Saves computation time, since we don't need it
t.set_grad_enabled(False)

#device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],  # noqa: F722
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],  # noqa: F722
) -> Float[Tensor, "..."]:
    """Calculate logit differences from residual stack."""
    # Get the device from the input tensor
    device = residual_stack.device

    # Apply layer normalization (keeping on same device)
    scaled_residual_stream = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )

    # Ensure consistent device and dtype
    logit_diff_directions = logit_diff_directions.to(
        device=device,
        dtype=scaled_residual_stream.dtype
    )

    batch_size = residual_stack.size(-2)
    avg_logit_diff = einops.einsum(
        scaled_residual_stream,
        logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size

    return avg_logit_diff

def generate_logit_diff_batched(
    dataset: Callable,
    regressors: List[Callable],
    seq_len: int,
    batches: int,
    model: HookedTransformer,
    random_seeds: Optional[List[int]] = None
) -> str:
    """Generate logit difference across batches with seed-specific token pairs."""
    random_seeds = random_seeds if random_seeds is not None else list(range(batches))
    model_device = model.W_U.device

    # Initialize results storage
    all_diffs = []
    token_pairs_names = None
    layer_names = None

    for batch_idx, seed in enumerate(random_seeds):
        print(f"Processing seed {seed} (batch {batch_idx+1}/{len(random_seeds)})")

        # Get dataset and run model
        dataset_tuple = dataset(random_state=seed)
        x_train, y_train, x_test, y_test = dataset_tuple
        prompt = prepare_prompt(x_train, y_train, x_test)
        prompt_tensor = model.to_tokens(prompt, prepend_bos=True).to(model_device)

        comparison_data = create_comparison_data(
            model, dataset, regressors, random_state=seed, seq_len=seq_len
        )

        if token_pairs_names is None:
            token_pairs_names = comparison_data["comparison_names"]

        # Run model
        with torch.no_grad():
            _, cache = model.run_with_cache(prompt_tensor)

        # Get accumulated and per-layer residuals once
        accumulated_residual, curr_labels = cache.accumulated_resid(
            layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
        )
        per_layer_residual, _ = cache.decompose_resid(
            layer=-1, pos_slice=-1, return_labels=True
        )

        if layer_names is None:
            layer_names = curr_labels

        # Process all token pairs at once
        token_pairs = comparison_data["token_pairs"].to(model_device)

        # Store results for this seed
        seed_diffs = []

        for token_pair in token_pairs:
            try:
                # Get directions (on GPU)
                token_directions = model.tokens_to_residual_directions(token_pair)
                if token_directions.shape[1] == 1:
                    logit_diff_directions = token_directions.squeeze(1)
                else:
                    correct_dir, incorrect_dir = token_directions.unbind(dim=1)
                    logit_diff_directions = (correct_dir - incorrect_dir)

                # Calculate both diffs (keeping on GPU)
                acc_diffs = residual_stack_to_logit_diff(
                    accumulated_residual, cache, logit_diff_directions
                )
                per_layer_diffs = residual_stack_to_logit_diff(
                    per_layer_residual, cache, logit_diff_directions
                )

                # Combine and store results (still on GPU)
                combined_diffs = torch.stack([acc_diffs, per_layer_diffs])
                seed_diffs.append(combined_diffs)

            except Exception as e:
                print(f"Error processing pair for seed {seed}: {e}")
                if seed_diffs:
                    # Use shape from previous successful computation
                    placeholder = torch.zeros_like(seed_diffs[-1])
                    seed_diffs.append(placeholder)

        if seed_diffs:
            # Stack all results for this seed (still on GPU)
            seed_diffs = torch.stack(seed_diffs)
            # Move to CPU only once per seed
            all_diffs.append(seed_diffs.cpu())

        # Clear GPU memory
        del cache, accumulated_residual, per_layer_residual
        torch.cuda.empty_cache()

    # Average results (on CPU)
    if all_diffs:
        avg_diffs = torch.stack(all_diffs).mean(dim=0)
        # Split accumulated and per-layer diffs
        avg_accumulated_diffs = avg_diffs[:, 0]
        avg_per_layer_diffs = avg_diffs[:, 1]

        # Create plots
        plots = create_logit_diff_plots(
            avg_accumulated_diffs,
            avg_per_layer_diffs,
            token_pairs_names,
            layer_names
        )
        return plots

    return "Error: No valid results to plot"

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

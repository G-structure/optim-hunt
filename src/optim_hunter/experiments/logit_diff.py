from typing import Callable, Dict, List, Tuple, Union
import torch as t
from torch import Tensor
import einops
from tqdm import tqdm
from functools import partial
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import pandas as pd
import numpy.typing as npt
import numpy as np
from jaxtyping import Float, Int

from optim_hunter.plot_html import (
    create_multi_line_plot_layer_names,
    create_heatmap_plot,
    with_identifier,
)
from optim_hunter.model_utils import generate_linreg_tokens

def generate_logit_diff_hooked(
    dataset: Callable[
        ...,
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[
        Callable[
            [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int],
            Dict[str, Union[str, npt.NDArray[np.float64]]]
        ]
    ],
    seq_len: int,
    batches: int,
    model: HookedTransformer
) -> None:
    """Generate logit difference visualizations using memory-efficient hooks.

    Args:
        dataset: Dataset function that returns training/testing data splits
        regressors: List of regression functions to compare
        seq_len: Maximum sequence length
        batches: Number of batches to process
        model: Transformer model for generating predictions
    """
    # Get initial batch to determine shapes and comparison names
    # Get initial dataset split
    x_train, y_train, x_test, y_test = dataset(0)  # Use 0 as initial random seed

    # Generate comparison data for each regressor
    comparison_names: List[str] = []
    for regressor in regressors:
        result = regressor(x_train, x_test, y_train, y_test, random_state=0)
        model_name = str(result["model_name"])
        # Add comparison between model and true value
        comparison_names.append(f"{model_name}_vs_true")
        # Add comparison between model and each other model
        for other_regressor in regressors:
            if other_regressor != regressor:
                other_result = other_regressor(
                    x_train, x_test, y_train, y_test, random_state=0
                )
                other_name = str(other_result["model_name"])
                if f"{other_name}_vs_{model_name}" not in comparison_names:
                    comparison_names.append(f"{model_name}_vs_{other_name}")

    n_comparisons: int = len(comparison_names)

    # Initialize accumulators
    accumulated_diffs = t.zeros(
        (n_comparisons, model.cfg.n_layers + 1),  # Layers + embed
        device=model.cfg.device
    )

    per_layer_diffs = t.zeros(
        (n_comparisons, model.cfg.n_layers),  # Just the layers
        device=model.cfg.device
    )
    per_head_diffs: Float[Tensor, "n_comp n_layers n_heads"] = t.zeros(
        (n_comparisons, model.cfg.n_layers, model.cfg.n_heads),
        device=model.cfg.device
    )
    n_samples: int = 0

    def get_logit_diff_directions(
        model: HookedTransformer,
        token_pairs: List[Int[Tensor, "batch 2"]]
    ) -> List[Float[Tensor, "d_model"]]:
        """Calculate logit difference directions for each comparison."""
        directions: List[Float[Tensor, "d_model"]] = []
        for token_pair in token_pairs:
            pair_dirs: Float[Tensor, "batch 2 d_model"] = \
                model.tokens_to_residual_directions(token_pair)
            # Get directions for correct and incorrect tokens
            correct_dir, incorrect_dir = pair_dirs.unbind(dim=1)
            # Calculate difference direction
            directions.append((correct_dir - incorrect_dir)[0])  # Take first batch element
        return directions

    def process_batch(batch_idx: int) -> None:
        """Process a single batch of data.

        Args:
            batch_idx: Index of current batch
        """
        nonlocal n_samples

        # Generate data for this batch
        tokens, data_store = generate_linreg_tokens(
            model, dataset, regressors, seq_len,
            batch_size=1, sub_batch=batch_idx
        )

        # Extract token pairs for each comparison
        token_pairs: List[Int[Tensor, "batch 2"]] = [
            t.stack([data_store[0]["token_pairs"][i]])[0]
            for i in range(n_comparisons)
        ]

        # Get logit difference directions
        # Move token pairs to device
        token_pairs = [pair.to("cuda:1") for pair in token_pairs]
        logit_diff_directions = get_logit_diff_directions(model, token_pairs)

        def residual_hook(
            activation: Float[Tensor, "batch pos d_model"],
            hook: HookPoint,
            comp_idx: int,
            logit_diff_direction: Float[Tensor, "d_model"]
        ) -> None:
            """Process residual stream activations."""
            # Get layer index from hook name
            if hook.name is None or not isinstance(hook.name, str):
                return
            layer_idx: int = int(hook.name.split('.')[1]) if 'blocks' in hook.name else -1

            # Scale activation with layer norm
            if layer_idx == -1:  # Embedding
                scaled_activation = model.ln_final(activation[:, -1, :])
            else:
                # Get the layernorm from the transformer block
                block = model.blocks[layer_idx]
                scaled_activation = block.ln2(activation[:, -1, :])

            # Move tensors to same device
            scaled_activation = scaled_activation.to(logit_diff_direction.device)

            # Calculate logit difference by projecting onto the direction
            logit_diff = einops.einsum(
                scaled_activation,
                logit_diff_direction,
                "batch d_model, d_model -> batch"
            ).mean()

            # Ensure both tensors are on cuda
            logit_diff = logit_diff.to("cuda")
            # Accumulate in appropriate stores
            if layer_idx == -1:
                accumulated_diffs[comp_idx, 0] += logit_diff  # Embedding goes in first position
            else:
                accumulated_diffs[comp_idx, layer_idx + 1] += logit_diff  # Layer outputs after
                per_layer_diffs[comp_idx, layer_idx] += logit_diff

        def attention_hook(
            pattern: Float[Tensor, "batch head_idx pos pos"],
            hook: HookPoint,
            comp_idx: int
        ) -> None:
            """Process attention patterns.

            Args:
                pattern: Attention pattern tensor
                hook: Hook point in the model
                comp_idx: Index of current comparison
            """
            if hook.name is None or not isinstance(hook.name, str):
                return
            layer_idx: int = int(hook.name.split('.')[1])
            # Average attention per head
            head_pattern: Float[Tensor, "n_heads"] = pattern.mean(dim=(0, 2))
            per_head_diffs[comp_idx, layer_idx, :] += head_pattern

        # Set up hooks for each comparison
        # Set up hooks for each comparison
        hooks: List[Tuple[str, Callable[..., None]]] = []

        # First add embedding hook
        for comp_idx, logit_diff_direction in enumerate(logit_diff_directions):
            hooks.append(
                ('hook_embed',
                 partial(residual_hook, comp_idx=comp_idx,
                        logit_diff_direction=logit_diff_direction))
            )

        # Then add layer hooks
        for comp_idx, logit_diff_direction in enumerate(logit_diff_directions):
            for layer in range(model.cfg.n_layers):
                hooks.append(
                    (f'blocks.{layer}.hook_resid_post',
                     partial(residual_hook, comp_idx=comp_idx,
                            logit_diff_direction=logit_diff_direction))
                )

        # Run model with hooks
        model.run_with_hooks(
            tokens,
            fwd_hooks=hooks,
            return_type=None
        )

        n_samples += 1

        # Clear GPU memory
        if t.cuda.is_available():
            t.cuda.empty_cache()

    # Process all batches
    for batch_idx in tqdm(range(batches), desc="Processing batches"):
        process_batch(batch_idx)

    # Average accumulated metrics
    accumulated_diffs /= n_samples
    per_layer_diffs /= n_samples
    per_head_diffs /= n_samples

    # Move tensors to CPU for plotting
    accumulated_diffs = accumulated_diffs.cpu()
    per_layer_diffs = per_layer_diffs.cpu()
    per_head_diffs = per_head_diffs.cpu()

    # Get layer names for plotting
    layer_names: List[str] = ['embed'] + [f'layer_{i}' for i in range(model.cfg.n_layers)]

    # Create plots using the original plotting functions
    @with_identifier("accumulated-residual-plot")
    def create_accumulated_residual_plot() -> str:
        return create_multi_line_plot_layer_names(
            y_values_list=[[float(x) for x in diffs] for diffs in accumulated_diffs],
            labels=comparison_names,
            title="Average Logit Difference From Accumulated Residual Stream",
            x_label="Layer",
            y_label="Logit Diff",
            layer_names=layer_names,
            include_plotlyjs=True,
            include_theme_js=False,
            active_lines=[-4]
        )

    @with_identifier("per-layer-plot")
    def create_per_layer_plot() -> str:
        return create_multi_line_plot_layer_names(
            y_values_list=[[float(x) for x in diffs] for diffs in per_layer_diffs],
            labels=comparison_names,
            title="Average Per Layer Logit Difference",
            x_label="Layer",
            y_label="Logit Diff",
            layer_names=layer_names[1:],  # Skip embed
            include_plotlyjs=False,
            include_theme_js=True,
            active_lines=[-4]
        )

    @with_identifier("per-head-heatmap")
    def create_per_head_heatmap() -> str:
        # Reshape per_head_diffs for heatmap
        per_head_matrix: Float[Tensor, "n_comp n_cells"] = \
            per_head_diffs.reshape(n_comparisons, -1)
        return create_heatmap_plot(
            z_values=per_head_matrix,
            title="Per-Head Logit Differences Across Comparisons",
            x_label="Attention Head",
            y_label="Comparison",
            include_plotlyjs=False,
            include_theme_js=True
        )

    # Generate and output all plots
    accumulated_plot: str = create_accumulated_residual_plot()
    per_layer_plot: str = create_per_layer_plot()
    per_head_heatmap: str = create_per_head_heatmap()
    print(accumulated_plot + per_layer_plot + per_head_heatmap)

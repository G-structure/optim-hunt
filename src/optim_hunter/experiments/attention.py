"""Module for attention visualization and analysis in transformer models.

Contains functions for calculating and visualizing different types of attention
scores including induction scores, per-example scores, and accumulated attention
scores.
"""
import functools

import einops
import torch as t
from jaxtyping import Float
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer

from optim_hunter.model_utils import check_token_positions, get_tokenized_prompt
from optim_hunter.plot_html import create_heatmap_plot

from typing import Callable, Tuple
import pandas as pd


def attention(
    model: HookedTransformer,
    num_seeds: int,
    seq_len: int,
    dataset: Callable[
        [int],
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
) -> str:
    """Generate attention viz plots comparing different attention mechanisms.

    Args:
        model: The transformer model to analyze
        num_seeds: Number of random seeds to use
        seq_len: Maximum sequence length
        dataset: Function that generates the dataset splits

    Returns:
        str: Combined HTML of all generated plots

    """
    tokenized_prompt = get_tokenized_prompt(model, seq_len, 1, dataset,
        print_prompt=False)
    output_pos, feature_pos = check_token_positions(model, dataset, seq_len,
        print_info=False)

    # Explicitly type output_pos and feature_pos
    output_pos: list[int] = output_pos  # type: ignore
    feature_pos: list[int] = feature_pos  # type: ignore

    cfg = model.cfg
    device = cfg.device
    n_layers, n_heads = cfg.n_layers, cfg.n_heads

    # Initialize score tensors
    induction_score_store = t.zeros((n_layers, n_heads), device=device)
    per_example_score_store = t.zeros((n_layers, n_heads, len(output_pos)),
        device=device)
    per_example_accumulated_score_store = t.zeros((n_layers, n_heads,
        len(output_pos)), device=device)
    all_examples_score_store = t.zeros((n_layers, n_heads), device=device)
    all_example_accumulated_score_store = t.zeros((n_layers, n_heads),
        device=device)

    output_pos_len = len(output_pos)

    # Initialize accumulators for scores
    induction_score_accumulator = t.zeros((n_layers, n_heads), device=device)
    all_examples_score_accumulator = t.zeros((n_layers, n_heads), device=device)
    all_example_accumulated_score_accumulator = t.zeros((n_layers, n_heads),
        device=device)
    per_example_score_store_accumulator = t.zeros((n_layers, n_heads,
        output_pos_len), device=device)
    per_example_accumulated_score_store_accumulator = t.zeros((n_layers,
        n_heads, output_pos_len), device=device)

    def induction_score_hook(
        pattern: Float[t.Tensor, "batch head_index dest_pos source_pos"],  # noqa: F722
        hook: HookPoint,
    ) -> None:
        # Take diagonal of attention paid from each destination position to
        # source positions seq_len-1 tokens back
        # (Only entries for tokens with index>=seq_len)
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
        # Get average score per head
        induction_score = einops.reduce(induction_stripe,
            "batch head_index position -> head_index", "mean")
        # Store result
        induction_score_store[hook.layer(), :] = induction_score

    def accumulated_attention_hook(
        pattern: Float[t.Tensor, "batch head_index dest_pos source_pos"],  # noqa: F722
        hook: HookPoint,
        output_positions: list[int],
        feature_positions: list[int]
    ) -> None:
        """Measure accumulated attention between output positions and previous
        positions.

        Args:
            pattern: Attention pattern tensor with shape
                [batch, head_index, dest_pos, source_pos]
            hook: HookPoint object containing layer information
            output_positions: List of positions of first number after "Output:"
            feature_positions: List of positions of first numbers after
                "Features:"

        """
        scores: list[t.Tensor] = []

        # For each output position
        for i, output_pos in enumerate(output_positions):
            # Get all previous output and feature positions
            relevant_positions = [pos for pos in output_positions
                if pos < output_pos] + \
                [pos for pos in feature_positions if pos < output_pos]

            # Get attention scores from current output position
            # Shape: [head_index, 1, source_pos]
            output_attention = pattern[0, :, output_pos:output_pos+1, :]

            # Calculate mean attention to relevant positions
            # Shape: [head_index]
            if relevant_positions:
                accumulated_attention = output_attention[:, 0,
                    relevant_positions].mean(dim=-1)
                per_example_accumulated_score_store[hook.layer(), : , i] = \
                    accumulated_attention
                scores.append(accumulated_attention)

        # Average across outputs
        if scores:
            example_score = t.stack(scores).mean(dim=0)
            # Store result
            all_example_accumulated_score_store[hook.layer(), :] = example_score

    def all_example_hook(
        pattern: Float[t.Tensor, "batch head_index dest_pos source_pos"],  # noqa: F722
        hook: HookPoint,
        output_positions: list[int],
        feature_positions: list[int]
    ) -> None:
        """Measure attention from output positions to feature positions.

        Args:
            pattern: Attention pattern tensor with shape
                [batch, head_index, dest_pos, source_pos]
            hook: HookPoint object containing layer information
            output_positions: List of positions of first number after "Output:"
            feature_positions: List of positions of first numbers after
                "Features:"

        """
        scores: list[t.Tensor] = []

        # For each output position
        for i, output_pos in enumerate(output_positions):
            # Get the 3 relevant feature positions that come before this output
            relevant_feature_pos = [pos for pos in feature_positions
                if pos < output_pos][-3:]

            # Get attention scores from output position
            # Shape: [head_index, 1, source_pos]
            output_attention = pattern[0, :, output_pos:output_pos+1, :]

            # Calculate mean attention to relevant feature positions
            # Shape: [head_index]
            feature_attention = output_attention[:, 0,
                relevant_feature_pos].mean(dim=-1)
            per_example_score_store[hook.layer(), : , i] = feature_attention
            scores.append(feature_attention)

        # Average across outputs
        example_score = t.stack(scores).mean(dim=0)

        # Store result
        all_examples_score_store[hook.layer(), :] = example_score

    # Filter for attention pattern names
    def pattern_hook_names_filter(name: str) -> bool:
        return str(name).endswith("pattern")

    # Loop over seeds
    for seed in tqdm(range(num_seeds), desc="Running seeds"):
        # Get tokenized prompt for current seed
        tokenized_prompt = get_tokenized_prompt(model, seq_len, seed, dataset,
            print_prompt=False)
        output_pos, feature_pos = check_token_positions(model, dataset, seq_len,
            print_info=False)

        # Run model with hooks
        model.run_with_hooks(
            tokenized_prompt,
            return_type=None,  # For efficiency, don't calculate logits
            fwd_hooks=[
                (
                    pattern_hook_names_filter,
                    induction_score_hook
                ),
                (
                    pattern_hook_names_filter,
                    functools.partial(
                        all_example_hook,
                        output_positions=output_pos,
                        feature_positions=feature_pos
                    )
                ),
                (
                    pattern_hook_names_filter,
                    functools.partial(
                        accumulated_attention_hook,
                        output_positions=output_pos,
                        feature_positions=feature_pos
                    )
                )
            ]
        )

        # Accumulate scores
        induction_score_accumulator += induction_score_store
        all_examples_score_accumulator += all_examples_score_store
        all_example_accumulated_score_accumulator += \
            all_example_accumulated_score_store
        per_example_accumulated_score_store_accumulator += \
            per_example_accumulated_score_store
        per_example_score_store_accumulator += per_example_score_store

    # Average scores
    induction_score_avg = induction_score_accumulator / num_seeds
    all_examples_score_avg = all_examples_score_accumulator / num_seeds
    all_example_accumulated_score_avg = \
        all_example_accumulated_score_accumulator / num_seeds
    per_example_accumulated_score_store_avg = \
        per_example_accumulated_score_store_accumulator / num_seeds
    per_example_score_store_avg = \
        per_example_score_store_accumulator / num_seeds

    # Create HTML plots
    html_output: list[str] = []

    # Include plotlyjs and theme_js only in first plot
    html_output.append(create_heatmap_plot(
        induction_score_avg,
        "Average Induction Score by Head",
        x_label="Head",
        y_label="Layer",
        include_plotlyjs=True,
        include_theme_js=True
    ))

    html_output.append(create_heatmap_plot(
        all_examples_score_avg,
        "Average Per Example Score by Head",
        x_label="Head",
        y_label="Layer"
    ))

    html_output.append(create_heatmap_plot(
        all_example_accumulated_score_avg,
        "Average Accumulated Example Score by Head",
        x_label="Head",
        y_label="Layer"
    ))

    # Create multi-plot for per-example scores
    for i in range(per_example_score_store_avg.shape[-1]):
        html_output.append(create_heatmap_plot(
            per_example_score_store_avg[..., i],
            f"Example {i} Score by Head",
            x_label="Head",
            y_label="Layer"
        ))

    # Create multi-plot for accumulated per-example scores
    for i in range(per_example_accumulated_score_store_avg.shape[-1]):
        html_output.append(create_heatmap_plot(
            per_example_accumulated_score_store_avg[..., i],
            f"Accumulated Example {i} Score by Head",
            x_label="Head",
            y_label="Layer"
        ))

    # Combine all HTML plots
    combined_html = "\n".join(html_output)

    return combined_html

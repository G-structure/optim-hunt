"""Module for testing logit differences and visualizing transformer attention.

Contains functions for generating and analyzing logit differences between
correct/incorrect predictions and creating visualizations.
"""
import logging
from typing import Any, Callable, Dict, List

import einops
import pandas as pd
import torch as t
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer

from optim_hunter.llama_model import load_llama_model
from optim_hunter.logging_config import setup_logging
from optim_hunter.model_utils import (
    run_and_cache_model_linreg_tokens,
    run_and_cache_model_linreg_tokens_batched,
)
from optim_hunter.plot_html import (
    create_heatmap_plot,
    create_line_plot,
    create_multi_line_plot_layer_names,
    with_identifier,
)

# Set up logging
setup_logging("DEBUG")

# Create logger for this module
logger = logging.getLogger(__name__)

# Saves computation time, since we don't need it for the contents of this
# notebook
t.set_grad_enabled(False)

#device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"


def generate_logit_diff_plots(
    dataset: Any,
    regressors: List[Callable[..., Dict[str, Any]]]
) -> None:
    """Generate and plot logit difference visualizations.

    Args:
        dataset: Dataset generator function that returns splits for training
            and testing
        regressors: List of regression functions to compare the model against

    Returns:
        None: Plots are printed to stdout as HTML

    """
    all_plots_html: List[str] = []
    model = load_llama_model()
    if model is None:
        raise ValueError("Could not load model")

    seq_len = 25
    # TODO we need to be able to run more batches but not over
    batch = 1
    (linreg_tokens, linreg_logits, linreg_cache, linreg_data_store) = \
        run_and_cache_model_linreg_tokens(
            model, dataset, regressors, seq_len, batch)

    model.clear_contexts()
    linreg_tokens = linreg_tokens.to('cpu')
    linreg_logits = linreg_logits.to('cpu')
    linreg_cache = linreg_cache.to('cpu')

    # Verify all datasets have the same comparison names
    base_comparison_names = linreg_data_store[0]["comparison_names"]
    all_match = all(
        dataset["comparison_names"] == base_comparison_names
        for dataset in linreg_data_store[1:]
    )
    assert all_match, "Mismatch in comparison names across datasets."

    # Extract comparison names from the first dataset
    token_pairs_names = base_comparison_names.copy()

    # Extract token pairs across all datasets for each comparison
    token_pairs = [
        t.stack([dataset["token_pairs"][i] for dataset in linreg_data_store])[0]
        for i in range(len(token_pairs_names))
    ]

    logger.info(f"Number of comparisons: {len(token_pairs_names)}")
    logger.info(f"Number of token_pairs: {len(token_pairs)}")

    # Iterate over token pairs and generate plots
    for i, token_pair in enumerate(token_pairs):
        logger.info(f"Processing comparison {i}: {token_pairs_names[i]}")
        token_pair = token_pair.to('cpu')

        def logits_to_ave_logit_diff(
            logits: Tensor,  # Shape: [batch seq d_vocab]
            answer_tokens: Tensor = token_pair,  # Shape: [batch 2]
            per_prompt: bool = False
        ) -> Tensor:  # Shape: [batch] or single value
            """Calculate logit difference between correct and incorrect answer.

            If per_prompt=True, return array of differences rather than the
            average.
            """
            # Extract token IDs for correct and incorrect answers
            correct = answer_tokens[:, 0]  # Correct token IDs
            incorrect = answer_tokens[:, 1]  # Incorrect token IDs

            # Extract logits for the final token in the sequence
            final_logits = logits[:, -1, :]  # Shape: (batch, d_vocab)

            # Get logits for the correct and incorrect answers
            correct_logits = final_logits[
                t.arange(final_logits.size(0)),
                correct
            ]
            incorrect_logits = final_logits[t.arange(final_logits.size(0)),
                incorrect]

            # Calculate logit difference
            logit_diff = correct_logits - incorrect_logits  # Shape: (batch,)

            if per_prompt:
                return logit_diff  # Return per-prompt logit differences
            else:
                return logit_diff.mean()  # Return mean logit difference / batch

        original_per_prompt_diff = logits_to_ave_logit_diff(
            linreg_logits, token_pair, per_prompt=True)
        logger.debug(
            f"Per prompt logit difference for comparison "
            f"'{token_pairs_names[i]}': {original_per_prompt_diff}")
        original_average_logit_diff = logits_to_ave_logit_diff(linreg_logits,
            token_pair)
        logger.debug(
            f"Average logit difference for comparison "
            f"'{token_pairs_names[i]}': {original_average_logit_diff}")

        # Retrieve final residual stream
        final_residual_stream = linreg_cache["resid_post", -1]
                                  # Shape: [batch seq d_model]
        logger.debug(
            f"Final residual stream shape: "
            f"{final_residual_stream.shape}"
        )
        final_token_residual_stream = final_residual_stream[:, -1, :]
                                     # Shape: [batch d_model]

        # Compute residual directions
        pair_residual_directions = model.tokens_to_residual_directions(
            token_pair.to('cpu'))
        logger.debug(
            f"Answer residual directions shape: "
            f"{pair_residual_directions.shape}")

        correct_residual_directions, incorrect_residual_directions = \
            pair_residual_directions.unbind(dim=1)
        logit_diff_directions = correct_residual_directions - \
            incorrect_residual_directions
        logger.debug(
            f"Logit difference directions shape: {logit_diff_directions.shape}")

        def residual_stack_to_logit_diff(
            residual_stack: Tensor,  # Shape: [...batch d_model]
            cache: ActivationCache,
            logit_diff_directions: Tensor = logit_diff_directions
                                            # [batch d_model]
        ) -> Tensor:
            """Get average logit difference between correct & incorrect answer.

            Args:
                residual_stack: Stack of residual stream components
                cache: Activation cache
                logit_diff_directions: Token logit difference directions

            Returns:
                Average logit difference

            """
            # Apply LayerNorm scaling (to just the final sequence position)
            scaled_residual_stream = cache.apply_ln_to_stack(
                residual_stack, layer=-1, pos_slice=-1)

            logit_diff_directions = logit_diff_directions.to(
                dtype=scaled_residual_stream.dtype)

            logit_diff_directions = logit_diff_directions.to('cpu')

            logger.debug(
                f"Scaled residual stream shape: {scaled_residual_stream.shape}")
            logger.debug(
                f"Logit diff directions shape: {logit_diff_directions.shape}")

            # Projection
            batch_size = residual_stack.size(-2)
            avg_logit_diff = einops.einsum(
                scaled_residual_stream,
                logit_diff_directions,
                "... batch d_model, batch d_model -> ..."
            ) / batch_size
            return avg_logit_diff

        # Verify residual stack computation
        t.testing.assert_close(
            residual_stack_to_logit_diff(
                final_token_residual_stream.to(device='cuda'),
                linreg_cache.to(device='cuda')
            ),
            original_average_logit_diff.to(device='cuda'),
            rtol=5e-3,  # Increased tolerance
            atol=5e-3
        )

        # Accumulate residuals
        accumulated_residual, labels = linreg_cache.accumulated_resid(
            layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
        # accumulated_residual has shape (component, batch, d_model)

        logit_lens_logit_diffs = residual_stack_to_logit_diff(
            accumulated_residual, linreg_cache)
        # Convert to half precision
        logit_lens_logit_diffs = logit_lens_logit_diffs.half()

        # Generate plots with unique identifiers for each
        @with_identifier(f"logit-lens-plot-{token_pairs_names[i].lower().\
            replace(' ', '-')}")
        def create_logit_lens_plot_with_id():
                    # Convert labels to list of strings
                    labels_list = [str(label) for label in labels]
                    return create_line_plot(
                        y_values=logit_lens_logit_diffs,
                        title=(
                            "Average Logit Difference From Accumulated "
                            f"Residual Stream for {token_pairs_names[i]}"
                        ),
                        labels=labels_list,
                        x_label="Layer",
                        y_label="Logit Diff",
                        hover_mode="x unified",
                        include_plotlyjs=(i == 0),
                        include_theme_js=(i == len(token_pairs) - 1)
                    )

        # Generate and collect the plots
        logit_lens_plot_html = create_logit_lens_plot_with_id()
        all_plots_html.append(logit_lens_plot_html)

    # Combine all plots
    combined_html = "\n".join(all_plots_html)

    # Output combined HTML to stdout
    print(combined_html)

def generate_logit_diff_batched(
    dataset: Callable[
        ...,
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    ],
    regressors: List[Callable[..., Dict[str, Any]]],
    seq_len: int,
    batches: int,
    model: HookedTransformer
) -> None:
    """Generate batched logit difference visualizations for regression outputs.

    Args:
        dataset: Dataset function that returns training/testing data splits
        regressors: List of regression functions to compare against
        seq_len: Maximum sequence length for inputs
        batches: Number of batches to process
        model: Transformer model for generating predictions

    Returns:
        None: Plots are printed to stdout as HTML

    """
    if t.cuda.is_available():
        t.cuda.empty_cache()

    (linreg_tokens, linreg_logits, linreg_caches, linreg_data_store) = \
        run_and_cache_model_linreg_tokens_batched(
            model,
            dataset,
            regressors,
            seq_len=seq_len,
            total_batch=batches
        )

    model.clear_contexts()

    # Move all tokens and logits to CPU
    linreg_tokens = [tokens.to('cpu') for tokens in linreg_tokens]
    linreg_logits = [logits.to('cpu') for logits in linreg_logits]

    # Verify all datasets have the same comparison names
    base_comparison_names = linreg_data_store[0]["comparison_names"]
    all_match = all(
        dataset["comparison_names"] == base_comparison_names
        for dataset in linreg_data_store[1:]
    )
    assert all_match, "Mismatch in comparison names across datasets."

    # Extract comparison names from the first dataset
    token_pairs_names = base_comparison_names.copy()

    # Extract token pairs across all datasets for each comparison
    token_pairs = [
        t.stack([dataset["token_pairs"][i] for dataset in linreg_data_store])[0]
        for i in range(len(token_pairs_names))
    ]

    logger.info(f"Number of comparisons: {len(token_pairs_names)}")
    logger.info(f"Number of token_pairs: {len(token_pairs)}")

     # Lists to store accumulated results for all token pairs
    all_accumulated_diffs: List[t.Tensor] = []
    all_per_layer_diffs: List[t.Tensor] = []
    all_per_head_diffs: List[t.Tensor] = []

    # Iterate over token pairs and generate plots
    for i, token_pair in enumerate(token_pairs):
        logger.info(f"Processing comparison {i}: {token_pairs_names[i]}")
        token_pair = token_pair.to('cpu')

        def logits_to_ave_logit_diff(
            logits_list: List[t.Tensor],
            answer_tokens: t.Tensor = token_pair,
            per_prompt: bool = False
        ) -> t.Tensor:
            # Process each batch separately
            all_logit_diffs: List[t.Tensor] = []

            for logits in logits_list:
                final_logits = logits[:, -1, :]  # Take final position from each

                correct = answer_tokens[:, 0]
                incorrect = answer_tokens[:, 1]

                correct_logits = final_logits[
                    t.arange(final_logits.size(0)),
                    correct
                ]
                incorrect_logits = final_logits[
                    t.arange(final_logits.size(0)),
                    incorrect
                ]

                logit_diff = correct_logits - incorrect_logits
                all_logit_diffs.append(logit_diff)

            # Combine results
            combined_logit_diffs = t.cat(all_logit_diffs)

            if per_prompt:
                return combined_logit_diffs
            else:
                return combined_logit_diffs.mean()

        _ = logits_to_ave_logit_diff(linreg_logits, token_pair, per_prompt=True)
        _ = logits_to_ave_logit_diff(linreg_logits, token_pair)

        # Initialize lists to store results from all caches
        token_pair_accumulated_diffs: List[t.Tensor] = []
        token_pair_per_layer_diffs: List[t.Tensor] = []
        token_pair_per_head_diffs: List[t.Tensor] = []

        # Process each cache
        for cache_idx, current_cache in enumerate(linreg_caches):
            logger.info(f"Processing cache {cache_idx}")

            # Retrieve final residual stream
            # final_residual_stream = current_cache["resid_post", -1]

            # Compute residual directions
            pair_residual_directions = model.tokens_to_residual_directions(
                token_pair.to('cpu')
            )
            correct_residual_directions, incorrect_residual_directions = \
                pair_residual_directions.unbind(dim=1)
            logit_diff_directions = correct_residual_directions - \
                incorrect_residual_directions

            def residual_stack_to_logit_diff(
                residual_stack: t.Tensor,
                cache: ActivationCache,
                logit_diff_directions: t.Tensor = logit_diff_directions,
            ) -> t.Tensor:
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

            # Accumulate residuals
            accumulated_residual, labels = current_cache.accumulated_resid(
                layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
            )
            logit_lens_logit_diffs = residual_stack_to_logit_diff(
                accumulated_residual, current_cache
            ).half()
            token_pair_accumulated_diffs.append(logit_lens_logit_diffs)

            # Per layer analysis
            per_layer_residual, labels = current_cache.decompose_resid(
                layer=-1, pos_slice=-1, return_labels=True
            )
            per_layer_logit_diffs = residual_stack_to_logit_diff(
                per_layer_residual, current_cache
            )
            token_pair_per_layer_diffs.append(per_layer_logit_diffs)

            per_head_residual, _ = current_cache.stack_head_results(
                layer=-1, pos_slice=-1, return_labels=True
            )
            per_head_residual = einops.rearrange(
                per_head_residual,
                "(layer head) ... -> layer head ...",
                layer=model.cfg.n_layers
            )
            per_head_logit_diffs = residual_stack_to_logit_diff(
                per_head_residual, current_cache
            )
            token_pair_per_head_diffs.append(per_head_logit_diffs)

            # Clear GPU memory
            current_cache = current_cache.to('cpu')
            if t.cuda.is_available():
                t.cuda.empty_cache()

        # Average results across all caches
        stacked_diffs = t.stack(token_pair_accumulated_diffs)
        avg_accumulated_diffs = stacked_diffs.mean(dim=0)
        avg_per_layer_diffs = t.stack(token_pair_per_layer_diffs).mean(dim=0)
        avg_per_head_diffs = t.stack(token_pair_per_head_diffs).mean(dim=0)

        all_accumulated_diffs.append(avg_accumulated_diffs)
        all_per_layer_diffs.append(avg_per_layer_diffs)
        all_per_head_diffs.append(avg_per_head_diffs)

    logger.info(f"Size of all_accumulated_diffs: {len(all_accumulated_diffs)}")
    logger.info(f"Size of token_pairs_names: {len(token_pairs_names)}")
    # List out all token pairs names
    logger.info("Token pairs names:")
    for name in token_pairs_names:
        logger.info(f"- {name}")

    layer_names = [str(l) for l in labels]

    # Create the two multi-line plots
    @with_identifier("accumulated-residual-plot")
    def create_accumulated_residual_plot():
        return create_multi_line_plot_layer_names(
            y_values_list=all_accumulated_diffs,
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
            y_values_list=all_per_layer_diffs,
            labels=token_pairs_names,
            title="Average Per Layer Logit Difference",
            x_label="Layer",
            y_label="Logit Diff",
            layer_names=layer_names,
            include_plotlyjs=False,
            include_theme_js=True,
            active_lines=[-4]
        )

    @with_identifier("per-head-heatmap")
    def create_per_head_heatmap():
        # Stack tensors to create a 2D array
        # Each row represents a comparison, each column represents a head
        per_head_matrix = t.stack(all_per_head_diffs)

        return create_heatmap_plot(
            z_values=per_head_matrix,
            title="Per-Head Logit Differences Across Comparisons",
            x_label="Attention Head",
            y_label="Comparison",
            include_plotlyjs=False,
            include_theme_js=True
        )

    # Generate and output all plots
    accumulated_plot = create_accumulated_residual_plot()
    per_layer_plot = create_per_layer_plot()
    per_head_heatmap = create_per_head_heatmap()
    print(accumulated_plot + per_layer_plot + per_head_heatmap)

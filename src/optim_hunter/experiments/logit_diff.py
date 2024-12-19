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
from optim_hunter.plot_html import create_logit_lens_plot, get_theme_sync_js
from optim_hunter.llama_model import load_llama_model
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

#device = t.device("cuda:0,1" if t.cuda.is_available() else "cpu")
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# device = t.device("cpu")

MAIN = __name__ == "__main__"



regressors = [ linear_regression, knn_regression, random_forest, baseline_average, baseline_last, baseline_random ]



def generate_linreg_tokens(
    model: HookedTransformer,
    dataset,
    seq_len = 5,
    batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of linear regression ICL tokens

    Outputs are:
        linreg_tokens: [batch, 1+linreg]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long().to(device)
    zero_token = model.to_tokens('0', truncate=True)[0][-1]
    
    # Create list to store tokens for each batch
    batch_tokens = []
    data_store = []

    dataset_func = get_dataset_friedman_2
    
    # Generate tokens for each batch with different random seeds
    for i in range(batch):
        data = create_comparison_data(model, dataset_func, regressors, random_state=i, seq_len=seq_len)
        tokens = model.to_tokens(data['prompt'], truncate=True)
        batch_tokens.append(tokens[0])
        data_store.append(data)
    
    # Find the longest sequence length
    max_len = max(len(tokens) for tokens in batch_tokens)
    
    # Pad shorter sequences with token 0 at position -4
    for i in range(len(batch_tokens)):
        while len(batch_tokens[i]) < max_len:
            # Insert 0 at position -4 from the end
            print(f"Found mismatch in token length for batch {i}!\nLargest length: {max_len}\nBatch {i} length: {len(batch_tokens[i])}\nApplying padding...")
            print(f"\nBefore Zero Token Padding:\n#####\n{model.to_string(batch_tokens[i][-50:])}\n#####")
            batch_tokens[i] = t.cat([
                batch_tokens[i][:len(batch_tokens[i])-3],  
                zero_token.unsqueeze(0), # Add unsqueeze to make zero_token 1-dimensional
                batch_tokens[i][len(batch_tokens[i])-3:]
            ])
            print(f"\nAfter Zero Token Padding:\n#####\n{model.to_string(batch_tokens[i][-50:])}\n#####")

    
    # Stack all batches together 
    linreg_tokens = t.stack(batch_tokens).to(device)
    
    # Add prefix to each batch
    linreg_tokens = t.cat([prefix, linreg_tokens], dim=-1).to(device)
    return linreg_tokens, data_store

def run_and_cache_model_linreg_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> tuple[Tensor, Tensor, ActivationCache]:
    '''
    Generates a sequence of linear regression ICL tokens, and runs the model on it, returning (tokens, logits, cache)

    Should use the `generate_linreg_tokens` function above

    Outputs are:
        linreg_tokens: [batch, 1+linreg]
        linreg_logits: [batch, 1+linreg, d_vocab]
        linreg_cache: The cache of the model run on linreg_tokens
    '''
    linreg_tokens, linreg_data_store = generate_linreg_tokens(model, get_dataset_friedman_2, seq_len, batch)
    linreg_logits, linreg_cache = model.run_with_cache(linreg_tokens)
    return linreg_tokens, linreg_logits, linreg_cache, linreg_data_store

def generate_logit_diff_plots():
    all_plots_html = []
    model = load_llama_model()
    seq_len = 25
    # TODO we need to be able to run more batches but not over 
    batch = 1
    (linreg_tokens, linreg_logits, linreg_cache, linreg_data_store) = run_and_cache_model_linreg_tokens(model, seq_len, batch)
    model.clear_contexts()
    linreg_tokens = linreg_tokens.to('cpu')
    linreg_logits = linreg_logits.to('cpu')
    linreg_cache = linreg_cache.to('cpu')

    # Verify all datasets have the same comparison names
    base_comparison_names = linreg_data_store[0]["comparison_names"]
    all_match = all(dataset["comparison_names"] == base_comparison_names for dataset in linreg_data_store[1:])
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
            logits: Float[Tensor, "batch seq d_vocab"],
            answer_tokens: Float[Tensor, "batch 2"] = token_pair,
            per_prompt: bool = False
        ) -> Float[Tensor, "*batch"]:
            '''
            Returns logit difference between the correct and incorrect answer.

            If per_prompt=True, return the array of differences rather than the average.
            '''
            # Extract token IDs for correct and incorrect answers
            correct = answer_tokens[:, 0]  # Correct token IDs
            incorrect = answer_tokens[:, 1]  # Incorrect token IDs

            # Extract logits for the final token in the sequence
            final_logits = logits[:, -1, :]  # Shape: (batch, d_vocab)

            # Get logits for the correct and incorrect answers
            correct_logits = final_logits[t.arange(final_logits.size(0)), correct]  # Shape: (batch,)
            incorrect_logits = final_logits[t.arange(final_logits.size(0)), incorrect]  # Shape: (batch,)

            # Calculate logit difference
            logit_diff = correct_logits - incorrect_logits  # Shape: (batch,)

            if per_prompt:
                return logit_diff  # Return per-prompt logit differences
            else:
                return logit_diff.mean()  # Return mean logit difference over the batch

        original_per_prompt_diff = logits_to_ave_logit_diff(linreg_logits, token_pair, per_prompt=True)
        logger.debug(f"Per prompt logit difference for comparison '{token_pairs_names[i]}': {original_per_prompt_diff}")
        original_average_logit_diff = logits_to_ave_logit_diff(linreg_logits, token_pair)
        logger.debug(f"Average logit difference for comparison '{token_pairs_names[i]}': {original_average_logit_diff}")

        # Retrieve final residual stream
        final_residual_stream: Float[Tensor, "batch seq d_model"] = linreg_cache["resid_post", -1]
        logger.debug(f"Final residual stream shape: {final_residual_stream.shape}")
        final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

        # Compute residual directions
        pair_residual_directions = model.tokens_to_residual_directions(token_pair.to('cpu'))  # [batch 2 d_model]
        logger.debug(f"Answer residual directions shape: {pair_residual_directions.shape}")

        correct_residual_directions, incorrect_residual_directions = pair_residual_directions.unbind(dim=1)
        logit_diff_directions = correct_residual_directions - incorrect_residual_directions  # [batch d_model]
        logger.debug(f"Logit difference directions shape: {logit_diff_directions.shape}")

        def residual_stack_to_logit_diff(
            residual_stack: Float[Tensor, "... batch d_model"],
            cache: ActivationCache,
            logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
        ) -> Float[Tensor, "..."]:
            '''
            Gets the avg logit difference between the correct and incorrect answer for a given
            stack of components in the residual stream.
            '''
            # Apply LayerNorm scaling (to just the final sequence position)
            scaled_residual_stream = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)

            logit_diff_directions = logit_diff_directions.to(dtype=scaled_residual_stream.dtype)

            logit_diff_directions = logit_diff_directions.to('cpu')

            logger.debug(f"Scaled residual stream shape: {scaled_residual_stream.shape}")
            logger.debug(f"Logit diff directions shape: {logit_diff_directions.shape}")

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
            residual_stack_to_logit_diff(final_token_residual_stream.to(t.float32), linreg_cache.to(t.float32)),
            original_average_logit_diff.to(t.float32),
            rtol=5e-3,  # Increased tolerance
            atol=5e-3
        )

        # Accumulate residuals
        accumulated_residual, labels = linreg_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
        # accumulated_residual has shape (component, batch, d_model)

        logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(accumulated_residual, linreg_cache)
        # Convert to half precision
        logit_lens_logit_diffs = logit_lens_logit_diffs.half()

        # # # Generate plot
        # line(
        #     logit_lens_logit_diffs,
        #     hovermode="x unified",
        #     title=f"Logit Difference From Accumulated Residual Stream for {token_pairs_names[i]}",
        #     labels={"x": "Layer", "y": "Logit Diff"},
        #     xaxis_tickvals=labels,
        #     width=800
        # )
        # # Save the logit lens plot
        # fig = line(
        #     logit_lens_logit_diffs,
        #     hovermode="x unified", 
        #     title=f"Logit Difference From Accumulated Residual Stream for {token_pairs_names[i]}",
        #     labels={"x": "Layer", "y": "Logit Diff"},
        #     xaxis_tickvals=labels,
        #     width=800,
        #     return_fig=True
        # )
        # # Save the file, creating it if it doesn't exist
        # output_path = f"../docs/logit_lens_{token_pairs_names[i].replace(' ', '_')}.html"
        # with open(output_path, 'w') as f:
        #     fig.write_html(f)
            # Create the plot
        # fig = create_logit_lens_plot(
        #     logit_lens_logit_diffs,
        #     labels,
        #     token_pairs_names[i]
        # )
        
        # # Save just the plot HTML
        # output_path = f"../docs/logit_lens_{token_pairs_names[i].replace(' ', '_')}.html"
        # with open(output_path, 'w') as f:
        #     f.write(fig.to_html(
        #         full_html=False,
        #         include_plotlyjs='cdn',
        #         config={'displayModeBar': False}
        #     ))
        plot_html = create_logit_lens_plot(
            logit_lens_logit_diffs,
            labels,
            token_pairs_names[i],
            include_theme_js=(i == len(token_pairs) - 1),  # Include theme JS with last plot
            include_plotlyjs=(i == 0)  # Include Plotly.js only with first plot
        )
        
        all_plots_html.append(plot_html)
    
    # Combine all plots
    combined_html = "\n".join(all_plots_html)
    
    # Output combined HTML to stdout
    print(combined_html)
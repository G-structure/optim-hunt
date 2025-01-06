import torch as t
import functools
from tqdm import tqdm
from optim_hunter.model_utils import check_token_positions, get_tokenized_prompt
from optim_hunter.plot_html import create_heatmap_plot
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import einops

def attention(model, num_seeds, seq_len, dataset):
    tokenized_prompt = get_tokenized_prompt(model, seq_len, 1, dataset, print_prompt=False)
    output_pos, feature_pos = check_token_positions(model, dataset, seq_len, print_info=False)
    
    induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    per_example_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads, len(output_pos)), device=model.cfg.device)
    per_example_accumulated_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads, len(output_pos)), device=model.cfg.device)
    all_examples_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    all_example_accumulated_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Define the number of seeds
    num_seeds = 100

    output_pos, feature_pos = check_token_positions(model, dataset, seq_len, print_info=False)


    # Initialize accumulators for scores
    induction_score_accumulator = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    all_examples_score_accumulator = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    all_example_accumulated_score_accumulator = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    per_example_score_store_accumulator = t.zeros((model.cfg.n_layers, model.cfg.n_heads, len(output_pos)), device=model.cfg.device)
    per_example_accumulated_score_store_accumulator = t.zeros((model.cfg.n_layers, model.cfg.n_heads, len(output_pos)), device=model.cfg.device)


    def induction_score_hook(
        pattern: Float[t.Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint,
    ):
        # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
        # (This only has entries for tokens with index>=seq_len)
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
        # Get an average score per head
        induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
        # Store the result.
        induction_score_store[hook.layer(), :] = induction_score

    def accumulated_attention_hook(
        pattern: Float[t.Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint,
        output_positions,
        feature_positions
    ):
        """Hook to measure accumulated attention from current output positions to all previous output and feature positions.
        
        Args:
            pattern: Attention pattern tensor with shape [batch, head_index, dest_pos, source_pos]
            hook: HookPoint object containing layer information
            output_positions: List of positions of first number after "Output:"
            feature_positions: List of positions of first numbers after "Features:"

        """
        batch_size = pattern.shape[0]
        n_heads = pattern.shape[1]
        scores = []
        
        # For each output position
        for i, output_pos in enumerate(output_positions):
            # Get all previous output and feature positions
            relevant_positions = [pos for pos in output_positions if pos < output_pos] + \
                                [pos for pos in feature_positions if pos < output_pos]
            
            # Get attention scores from current output position to all previous relevant positions
            # Shape: [head_index, 1, source_pos]
            output_attention = pattern[0, :, output_pos:output_pos+1, :]  # Using first batch element
            
            # Calculate mean attention to the relevant positions
            # Shape: [head_index]
            if relevant_positions:
                accumulated_attention = output_attention[:, 0, relevant_positions].mean(dim=-1)
                per_example_accumulated_score_store[hook.layer(), : , i] = accumulated_attention
                scores.append(accumulated_attention)
        
        # Average across outputs
        if scores:
            example_score = t.stack(scores).mean(dim=0)
            # Store the result in the global store
            all_example_accumulated_score_store[hook.layer(), :] = example_score

    def all_example_hook(
        pattern: Float[t.Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint,
        output_positions,
        feature_positions
    ):
        """Hook to measure attention from output positions to previous feature positions.
        
        Args:
            pattern: Attention pattern tensor with shape [batch, head_index, dest_pos, source_pos]
            hook: HookPoint object containing layer information
            output_positions: List of positions of first number after "Output:"
            feature_positions: List of positions of first numbers after "Features:"

        """
        batch_size = pattern.shape[0]
        n_heads = pattern.shape[1]
        scores = []
        
        # For each output position
        for i, output_pos in enumerate(output_positions):
            # Get the 3 relevant feature positions that come before this output
            relevant_feature_pos = [pos for pos in feature_positions if pos < output_pos][-3:]
            
            
            # Get attention scores from output position to feature positions
            # Shape: [head_index, 1, source_pos]
            output_attention = pattern[0, :, output_pos:output_pos+1, :]  # Using first batch element
            
            # Calculate mean attention to the relevant feature positions
            # Shape: [head_index]
            feature_attention = output_attention[:, 0, relevant_feature_pos].mean(dim=-1)
            per_example_score_store[hook.layer(), : , i] = feature_attention
            scores.append(feature_attention)
        
        # Average across outputs
        example_score = t.stack(scores).mean(dim=0)
        
        # Store the result in the global store
        all_examples_score_store[hook.layer(), :] = example_score

    # We make a boolean filter on activation names, that's true only on attention pattern names.
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Loop over seeds
    for seed in tqdm(range(num_seeds), desc="Running seeds"):
        # Get tokenized prompt for the current seed
        tokenized_prompt = get_tokenized_prompt(model, seq_len, seed, dataset, print_prompt=False)
        output_pos, feature_pos = check_token_positions(model, dataset, seq_len, print_info=False)

        # Run the model with hooks
        model.run_with_hooks(
            tokenized_prompt, 
            return_type=None,  # For efficiency, we don't need to calculate the logits
            fwd_hooks=[
                (
                    pattern_hook_names_filter,
                    induction_score_hook
                ),
                (
                    pattern_hook_names_filter,
                    functools.partial(
                        all_example_hook,
                        output_positions=output_pos,  # Use positions for the current seed
                        feature_positions=feature_pos  # Use positions for the current seed
                    )
                ),
                (
                    pattern_hook_names_filter,
                    functools.partial(
                        accumulated_attention_hook,
                        output_positions=output_pos,  # Use positions for the current seed
                        feature_positions=feature_pos  # Use positions for the current seed
                    )
                )
            ]
        )

        # Accumulate scores
        induction_score_accumulator += induction_score_store
        all_examples_score_accumulator += all_examples_score_store
        all_example_accumulated_score_accumulator += all_example_accumulated_score_store
        per_example_accumulated_score_store_accumulator += per_example_accumulated_score_store
        per_example_score_store_accumulator += per_example_score_store

    # Average the scores
    induction_score_avg = induction_score_accumulator / num_seeds
    all_examples_score_avg = all_examples_score_accumulator / num_seeds
    all_example_accumulated_score_avg = all_example_accumulated_score_accumulator / num_seeds
    per_example_accumulated_score_store_avg = per_example_accumulated_score_store_accumulator / num_seeds
    per_example_score_store_avg = per_example_score_store_accumulator / num_seeds

       # Create HTML plots
    html_output = []
    
    # Include plotlyjs and theme_js only in the first plot
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
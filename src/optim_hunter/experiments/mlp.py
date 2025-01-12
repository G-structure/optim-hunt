"""Utilities for analyzing MLP layer activations within transformer models."""

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from transformer_lens import (
    HookedTransformer,
)
from transformer_lens.hook_points import HookPoint

from optim_hunter.plot_html import create_heatmap_plot


def analyze_mlp_for_specific_tokens(
    model: HookedTransformer,
    input_tokens: torch.Tensor,
    output_pos: int,
    feature_pos: int,
    num_last_layers: int = 10
) -> str:
    """Analyze MLP activations for token positions in transformer model.

    Args:
        model: The transformer model to analyze
        input_tokens: Input token tensor
        output_pos: Position index for output token analysis
        feature_pos: Position index for feature token analysis
        num_last_layers: Number of final layers to analyze (default: 10)

    Returns:
        HTML string containing activation heatmap plots

    """
    mlp_activations: Dict[str, torch.Tensor] = {}
    plots_html: List[str] = []

    def mlp_hook(
        act: torch.Tensor,
        hook: HookPoint,
        layer_num: int
    ) -> torch.Tensor:
        if layer_num >= model.cfg.n_layers - num_last_layers:
            mlp_activations[f'layer_{layer_num}'] = act.detach()
        return act

    # Create hooks for each layer
    hooks: Sequence[
        Tuple[
            str,
            Callable[[Any, Any], Any]  # More generic callable type
        ]
    ] = []

    for layer_num in range(model.cfg.n_layers):
        hooks.append((
            f'blocks.{layer_num}.hook_mlp_out',
            lambda act, hook, ln=layer_num: mlp_hook(act, hook, ln)
        ))

    # Run model with hooks
    model.set_use_hook_mlp_in(True)
    _ = model.run_with_hooks(
        input_tokens,
        fwd_hooks=hooks
    )

    for layer_num in range(
        model.cfg.n_layers - num_last_layers,
        model.cfg.n_layers
    ):
        layer_key = f'layer_{layer_num}'
        if layer_key in mlp_activations:
            acts: torch.Tensor = mlp_activations[layer_key].squeeze(0)

            # Plot output token activations
            output_acts = acts[output_pos, :].cpu().numpy()
            output_plot = create_heatmap_plot(
                output_acts.reshape(1, -1),
                f"Layer {layer_num} - Output Number Tokens",
                include_plotlyjs=(
                    layer_num == model.cfg.n_layers - num_last_layers
                )
            )
            plots_html.append(output_plot)

            # Plot feature token activations
            feature_acts = acts[feature_pos, :].cpu().numpy()
            feature_plot = create_heatmap_plot(
                feature_acts.reshape(1, -1),
                f"Layer {layer_num} - Feature Number Tokens",
                include_theme_js=(layer_num == model.cfg.n_layers - 1)
            )
            plots_html.append(feature_plot)

            # Print statistics
            print(f"\nLayer {layer_num} Statistics:")
            print("\nOutput Token Statistics:")
            print(f"Mean activation: {output_acts.mean():.4f}")
            print(f"Std deviation: {output_acts.std():.4f}")
            print(f"Max activation: {output_acts.max():.4f}")
            print(f"Min activation: {output_acts.min():.4f}")
            print(f"Sparsity: {(output_acts == 0).mean() * 100:.2f}%")

            print("\nFeature Token Statistics:")
            print(f"Mean activation: {feature_acts.mean():.4f}")
            print(f"Std deviation: {feature_acts.std():.4f}")
            print(f"Max activation: {feature_acts.max():.4f}")
            print(f"Min activation: {feature_acts.min():.4f}")
            print(f"Sparsity: {(feature_acts == 0).mean() * 100:.2f}%")

    return "\n".join(plots_html)

from optim_hunter.plot_html import create_heatmap_plot

def analyze_mlp_for_specific_tokens(model, input_tokens, output_pos, feature_pos, num_last_layers=10):
    """
    Analyze MLP activations specifically for output and feature number tokens
    with zero values shown in white. Returns HTML string of plots.
    """
    mlp_activations = {}
    plots_html = []
    
    def mlp_hook(act, hook, layer_num):
        if layer_num >= model.cfg.n_layers - num_last_layers:
            mlp_activations[f'layer_{layer_num}'] = act.detach()
        return act

    # Create hooks for each layer
    hooks = []
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

    for layer_num in range(model.cfg.n_layers - num_last_layers, model.cfg.n_layers):
        layer_key = f'layer_{layer_num}'
        if layer_key in mlp_activations:
            acts = mlp_activations[layer_key].squeeze(0)
            
            # Plot output token activations
            output_acts = acts[output_pos, :].cpu().numpy()
            output_plot = create_heatmap_plot(
                output_acts,
                f"Layer {layer_num} - Output Number Tokens",
                include_plotlyjs=(layer_num == model.cfg.n_layers - num_last_layers)
            )
            plots_html.append(output_plot)
            
            # Plot feature token activations
            feature_acts = acts[feature_pos, :].cpu().numpy()
            feature_plot = create_heatmap_plot(
                feature_acts,
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
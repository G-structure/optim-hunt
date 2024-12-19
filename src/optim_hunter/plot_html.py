def create_logit_lens_plot(logit_lens_logit_diffs, labels, comparison_name):
    """
    Creates a lightweight, themed logit lens plot suitable for web embedding.
    
    Args:
        logit_lens_logit_diffs: Tensor of logit differences
        labels: List of layer labels
        comparison_name: Name of the comparison being plotted
    
    Returns:
        str: HTML/JavaScript code for the plot using Plotly
    """
    import plotly.graph_objects as go
    
    # Convert tensor to list
    logit_diffs = logit_lens_logit_diffs.tolist()
    
    # Create the trace
    trace = go.Scatter(
        x=list(range(len(logit_diffs))),
        y=logit_diffs,
        mode='lines+markers',
        line=dict(
            color='#9ec5fe',  # Matches website theme
            width=2
        ),
        marker=dict(
            size=6,
            color='#9ec5fe',
            line=dict(
                color='#ffffff',
                width=1
            )
        ),
        hovertemplate='Layer: %{x}<br>Logit Diff: %{y:.3f}<extra></extra>'
    )

    # Create the layout
    layout = go.Layout(
        template='plotly_dark',  # Dark theme to match website
        plot_bgcolor='rgba(26,26,26,0)',  # Transparent background
        paper_bgcolor='rgba(26,26,26,0)',
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(
            title='Layer',
            gridcolor='#444',
            ticktext=labels,
            tickvals=list(range(len(labels))),
            tickmode='array',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Logit Difference',
            gridcolor='#444',
            showgrid=True,
            zeroline=False
        ),
        title=dict(
            text=f'Logit Difference Across Layers<br><sub>{comparison_name}</sub>',
            font=dict(
                size=14,
                color='#e0e0e0'
            ),
            x=0.5,
            xanchor='center'
        ),
        hoverlabel=dict(
            bgcolor='#333',
            font_size=12,
            font_family="monospace"
        ),
        width=600,
        height=400
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)
    
    # Add light theme configuration
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Light Theme",
                    method="relayout",
                    args=[{
                        "plot_bgcolor": "rgba(253,246,227,0)",
                        "paper_bgcolor": "rgba(253,246,227,0)",
                        "font.color": "#073642",
                        "xaxis.gridcolor": "#93a1a1",
                        "yaxis.gridcolor": "#93a1a1",
                        "title.font.color": "#073642"
                    }]
                ),
                dict(
                    label="Dark Theme",
                    method="relayout",
                    args=[{
                        "plot_bgcolor": "rgba(26,26,26,0)",
                        "paper_bgcolor": "rgba(26,26,26,0)",
                        "font.color": "#e0e0e0",
                        "xaxis.gridcolor": "#444",
                        "yaxis.gridcolor": "#444",
                        "title.font.color": "#e0e0e0"
                    }]
                )
            ],
            x=0.9,
            y=1.1,
            xanchor="right",
            yanchor="top",
        )]
    )

    # Generate minimal HTML
    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={'displayModeBar': False}
    )
    
    return plot_html

# # Modify your existing loop to use the new plotting function
# for i, token_pair in enumerate(token_pairs):
#     logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, linreg_cache)
    # plot_html = create_logit_lens_plot(
    #     logit_lens_logit_diffs,
    #     labels,
    #     token_pairs_names[i]
    # )
    
    # # Save just the plot HTML
    # output_path = f"../docs/logit_lens_{token_pairs_names[i].replace(' ', '_')}.html"
    # with open(output_path, 'w') as f:
    #     f.write(plot_html)
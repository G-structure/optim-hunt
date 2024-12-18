def get_theme_sync_js():
    """
    Returns the JavaScript code for theme synchronization.
    This should be included once in the page.
    """
    return """
    <script>
    (function() {
        function updatePlotTheme(plotDiv, isLight) {
            console.log('Updating plot theme for', plotDiv.id, 'isLight:', isLight);
            try {
                if (isLight) {
                    Plotly.relayout(plotDiv, {
                        'plot_bgcolor': 'rgba(253,246,227,0)',
                        'paper_bgcolor': 'rgba(253,246,227,0)',
                        'xaxis.gridcolor': '#93a1a1',
                        'yaxis.gridcolor': '#93a1a1',
                        'xaxis.color': '#073642',
                        'yaxis.color': '#073642',
                        'title.font.color': '#073642',
                        'autosize': true
                    });
                    Plotly.restyle(plotDiv, {
                        'line.color': '#2075c7',
                        'marker.color': '#2075c7',
                        'marker.line.color': '#fdf6e3'
                    });
                } else {
                    Plotly.relayout(plotDiv, {
                        'plot_bgcolor': 'rgba(26,26,26,0)',
                        'paper_bgcolor': 'rgba(26,26,26,0)',
                        'xaxis.gridcolor': '#444',
                        'yaxis.gridcolor': '#444',
                        'xaxis.color': '#e0e0e0',
                        'yaxis.color': '#e0e0e0',
                        'title.font.color': '#e0e0e0',
                        'autosize': true
                    });
                    Plotly.restyle(plotDiv, {
                        'line.color': '#9ec5fe',
                        'marker.color': '#9ec5fe',
                        'marker.line.color': '#1a1a1a'
                    });
                }
                
                // Force a resize after theme update
                window.dispatchEvent(new Event('resize'));
                
            } catch (e) {
                console.error('Error updating plot theme:', e);
            }
        }

        // In template.html, modify the theme change handler
        function updateAllPlots() {
            // Debounce the update
            if (window.plotUpdateTimeout) {
                clearTimeout(window.plotUpdateTimeout);
            }
            
            window.plotUpdateTimeout = setTimeout(() => {
                const isLight = document.body.classList.contains('light-theme');
                const plots = document.querySelectorAll('.plotly-graph-div');
                
                // Update plots in batches
                const batchSize = 3;
                for (let i = 0; i < plots.length; i += batchSize) {
                    setTimeout(() => {
                        const batch = Array.from(plots).slice(i, i + batchSize);
                        batch.forEach(plot => updatePlotTheme(plot, isLight));
                    }, Math.floor(i/batchSize) * 100);
                }
            }, 250);
        }

        // Add resize handler with debouncing
        window.addEventListener('resize', function() {
            if (window.resizeTimeout) {
                clearTimeout(window.resizeTimeout);
            }
            window.resizeTimeout = setTimeout(function() {
                document.querySelectorAll('.plotly-graph-div').forEach(function(plot) {
                    Plotly.Plots.resize(plot);
                });
            }, 250);
        });

        // Wait for Plotly to be fully loaded
        function ensurePlotly() {
            if (window.Plotly) {
                console.log('Plotly loaded, setting up theme handling');
                // Initial theme setup
                updateAllPlots();
                
                // Listen for theme changes
                document.body.addEventListener('themeChanged', function(e) {
                    console.log('Theme changed event received');
                    updateAllPlots();
                });

                // Initial resize to ensure proper dimensions
                window.dispatchEvent(new Event('resize'));
            } else {
                console.log('Waiting for Plotly...');
                setTimeout(ensurePlotly, 100);
            }
        }

        // Start checking for Plotly
        ensurePlotly();
    })();
    </script>
    """

def create_logit_lens_plot(logit_lens_logit_diffs, labels, comparison_name, include_theme_js=False, include_plotlyjs=False):
    """
    Creates a lightweight, themed logit lens plot suitable for web embedding.
    
    Args:
        logit_lens_logit_diffs: Tensor of logit differences
        labels: List of layer labels
        comparison_name: Name of the comparison being plotted
        include_theme_js: Whether to include the theme sync JavaScript (should only be True once)
        include_plotlyjs: Whether to include the Plotly.js library (should only be True once)
    
    Returns:
        str: HTML/JavaScript code for the plot
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
            color='#9ec5fe',
            width=2
        ),
        marker=dict(
            size=6,
            color='#9ec5fe',
            line=dict(
                color='#1a1a1a',
                width=1
            )
        ),
        hovertemplate='Layer: %{x}<br>Logit Diff: %{y:.3f}<extra></extra>'
    )

    # Create the layout
    layout = go.Layout(
        plot_bgcolor='rgba(26,26,26,0)',
        paper_bgcolor='rgba(26,26,26,0)',
        margin=dict(l=50, r=20, t=50, b=50, pad=4),
        xaxis=dict(
            title='Layer',
            gridcolor='#444',
            ticktext=labels,
            tickvals=list(range(len(labels))),
            tickmode='array',
            showgrid=True,
            zeroline=False,
            color='#e0e0e0'
        ),
        yaxis=dict(
            title='Logit Difference',
            gridcolor='#444',
            showgrid=True,
            zeroline=False,
            color='#e0e0e0'
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
        autosize=True
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    config = {
        'displayModeBar': False,
        'staticPlot': False,
        'responsive': True,  # Enable responsiveness
    }
    
    # Generate plot HTML with controlled script inclusion
    plot_html = f"""
    <div class="plot-container">
        {fig.to_html(
            full_html=False,
            include_plotlyjs='cdn' if include_plotlyjs else False,
            config=config
        )}
    </div>
    """
    
    # Add theme sync JavaScript if requested
    if include_theme_js:
        plot_html += get_theme_sync_js()
    
    return plot_html
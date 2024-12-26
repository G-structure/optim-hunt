import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

THEME_COLORS = {
    'light': {
        'plot_bgcolor': 'rgba(253,246,227,0)',
        'paper_bgcolor': 'rgba(253,246,227,0)',
        'gridcolor': '#93a1a1',
        'text_color': '#073642',
        'bar_color': '#2075c7',
        'multi_line_colors': [
            '#2075c7', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#1f77b4', '#ff9896', '#98df8a', '#c5b0d5', '#c49c94',
            '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#393b79',
            '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3'
        ]
    },
    'dark': {
        'plot_bgcolor': 'rgba(26,26,26,0)',
        'paper_bgcolor': 'rgba(26,26,26,0)',
        'gridcolor': '#444',
        'text_color': '#e0e0e0',
        'bar_color': '#9ec5fe',
        'multi_line_colors': [
            '#9ec5fe', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
            '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3'
        ]
    }
}

def get_theme_sync_js():
    """
    Returns the JavaScript code for theme synchronization.
    This should be included once in the page.
    """
    return f"""
    <script>
    (function() {{
        function updatePlotTheme(plotDiv, isLight) {{
            console.log('Updating plot theme for', plotDiv.id, 'isLight:', isLight);
            try {{
                if (isLight) {{
                    Plotly.relayout(plotDiv, {{
                        'plot_bgcolor': '{THEME_COLORS["light"]["plot_bgcolor"]}',
                        'paper_bgcolor': '{THEME_COLORS["light"]["paper_bgcolor"]}',
                        'xaxis.gridcolor': '{THEME_COLORS["light"]["gridcolor"]}',
                        'yaxis.gridcolor': '{THEME_COLORS["light"]["gridcolor"]}',
                        'xaxis.color': '{THEME_COLORS["light"]["text_color"]}',
                        'yaxis.color': '{THEME_COLORS["light"]["text_color"]}',
                        'title.font.color': '{THEME_COLORS["light"]["text_color"]}',
                        'autosize': true
                    }});
                    Plotly.restyle(plotDiv, {{
                        'line.color': '{THEME_COLORS["light"]["bar_color"]}',
                        'marker.color': '{THEME_COLORS["light"]["bar_color"]}',
                        'marker.line.color': '{THEME_COLORS["light"]["plot_bgcolor"]}'
                    }});
                }} else {{
                    Plotly.relayout(plotDiv, {{
                        'plot_bgcolor': '{THEME_COLORS["dark"]["plot_bgcolor"]}',
                        'paper_bgcolor': '{THEME_COLORS["dark"]["paper_bgcolor"]}',
                        'xaxis.gridcolor': '{THEME_COLORS["dark"]["gridcolor"]}',
                        'yaxis.gridcolor': '{THEME_COLORS["dark"]["gridcolor"]}',
                        'xaxis.color': '{THEME_COLORS["dark"]["text_color"]}',
                        'yaxis.color': '{THEME_COLORS["dark"]["text_color"]}',
                        'title.font.color': '{THEME_COLORS["dark"]["text_color"]}',
                        'autosize': true
                    }});
                    Plotly.restyle(plotDiv, {{
                        'line.color': '{THEME_COLORS["dark"]["bar_color"]}',
                        'marker.color': '{THEME_COLORS["dark"]["bar_color"]}',
                        'marker.line.color': '{THEME_COLORS["dark"]["plot_bgcolor"]}'
                    }});
                }}
                
                // Force a resize after theme update
                window.dispatchEvent(new Event('resize'));
                
            }} catch (e) {{
                console.error('Error updating plot theme:', e);
            }}
        }}

        // In template.html, modify the theme change handler
        function updateAllPlots() {{
            // Debounce the update
            if (window.plotUpdateTimeout) {{
                clearTimeout(window.plotUpdateTimeout);
            }}
            
            window.plotUpdateTimeout = setTimeout(() => {{
                const isLight = document.body.classList.contains('light-theme');
                const plots = document.querySelectorAll('.plotly-graph-div');
                
                // Update plots in batches
                const batchSize = 3;
                for (let i = 0; i < plots.length; i += batchSize) {{
                    setTimeout(() => {{
                        const batch = Array.from(plots).slice(i, i + batchSize);
                        batch.forEach(plot => updatePlotTheme(plot, isLight));
                    }}, Math.floor(i/batchSize) * 100);
                }}
            }}, 250);
        }}

        // Add resize handler with debouncing
        window.addEventListener('resize', function() {{
            if (window.resizeTimeout) {{
                clearTimeout(window.resizeTimeout);
            }}
            window.resizeTimeout = setTimeout(function() {{
                document.querySelectorAll('.plotly-graph-div').forEach(function(plot) {{
                    Plotly.Plots.resize(plot);
                }});
            }}, 250);
        }});

        // Wait for Plotly to be fully loaded
        function ensurePlotly() {{
            if (window.Plotly) {{
                console.log('Plotly loaded, setting up theme handling');
                // Initial theme setup
                updateAllPlots();
                
                // Listen for theme changes
                document.body.addEventListener('themeChanged', function(e) {{
                    console.log('Theme changed event received');
                    updateAllPlots();
                }});

                // Initial resize to ensure proper dimensions
                window.dispatchEvent(new Event('resize'));
            }} else {{
                console.log('Waiting for Plotly...');
                setTimeout(ensurePlotly, 100);
            }}
        }}

        // Start checking for Plotly
        ensurePlotly();
    }})();
    </script>
    """
  
def create_line_plot(y_values, title, labels=None, x_label="Layer", y_label="Value", 
                hover_mode="x unified", include_theme_js=False, include_plotlyjs=False):
    """
    Creates a lightweight, themed plot suitable for web embedding.
    
    Args:
        y_values: Tensor or list of values to plot
        title: Plot title
        labels: List of x-axis tick labels (optional)
        x_label: Label for x-axis (default: "Layer")
        y_label: Label for y-axis (default: "Value")
        hover_mode: Plotly hover mode (default: "x unified")
        include_theme_js: Whether to include the theme sync JavaScript (should only be True once)
        include_plotlyjs: Whether to include the Plotly.js library (should only be True once)
    
    Returns:
        str: HTML/JavaScript code for the plot
    """    
    # Convert tensor to list if necessary
    if hasattr(y_values, 'tolist'):
        y_values = y_values.tolist()
    
    # Create the trace
    trace = go.Scatter(
        x=list(range(len(y_values))),
        y=y_values,
        mode='lines+markers',
        line=dict(
            color=THEME_COLORS['dark']['bar_color'],
            width=2
        ),
        marker=dict(
            size=6,
            color=THEME_COLORS['dark']['bar_color'],
            line=dict(
                color=THEME_COLORS['dark']['plot_bgcolor'],
                width=1
            )
        ),
        hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
    )

    # Create the layout
    layout = go.Layout(
        plot_bgcolor=THEME_COLORS['dark']['plot_bgcolor'],
        paper_bgcolor=THEME_COLORS['dark']['paper_bgcolor'],
        margin=dict(l=50, r=20, t=50, b=50, pad=4),
        xaxis=dict(
            title=x_label,
            gridcolor=THEME_COLORS['dark']['gridcolor'],
            ticktext=labels if labels is not None else None,
            tickvals=list(range(len(labels))) if labels is not None else None,
            tickmode='array' if labels is not None else 'auto',
            showgrid=True,
            zeroline=False,
            color=THEME_COLORS['dark']['text_color']
        ),
        yaxis=dict(
            title=y_label,
            gridcolor=THEME_COLORS['dark']['gridcolor'],
            showgrid=True,
            zeroline=False,
            color=THEME_COLORS['dark']['text_color']
        ),
        title=dict(
            text=title,
            font=dict(
                size=14,
                color=THEME_COLORS['dark']['text_color']
            ),
            x=0.5,
            xanchor='center'
        ),
        hoverlabel=dict(
            bgcolor='#333',
            font_size=12,
            font_family="monospace"
        ),
        hovermode=hover_mode,
        autosize=True
    )

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    config = {
        'displayModeBar': False,
        'staticPlot': False,
        'responsive': True,
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

def create_bar_plot(
    x_values: list,
    y_values: list,
    title: str,
    x_label: str = "",
    y_label: str = "",
    include_theme_js: bool = False,
    include_plotlyjs: bool = False,
    hover_template: str = None
) -> str:
    """
    Creates a themed bar plot suitable for web embedding.
    
    Args:
        x_values: List of x-axis values (categories)
        y_values: List of y-axis values (heights)
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        include_theme_js: Whether to include theme sync JavaScript
        include_plotlyjs: Whether to include Plotly.js library
        hover_template: Custom hover template (optional)
    
    Returns:
        str: HTML/JavaScript code for the plot
    """
    # Create the trace
    trace = go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(
            color='#9ec5fe',  # Default to dark theme color
            line=dict(
                color='#1a1a1a',
                width=1
            )
        ),
        hovertemplate=hover_template if hover_template else f"{x_label}: %{{x}}<br>{y_label}: %{{y:.3f}}<extra></extra>"
    )

    # Create the layout
    layout = go.Layout(
        plot_bgcolor=THEME_COLORS['dark']['plot_bgcolor'],
        paper_bgcolor=THEME_COLORS['dark']['paper_bgcolor'],
        margin=dict(l=50, r=20, t=50, b=50, pad=4),
        xaxis=dict(
            title=x_label,
            gridcolor=THEME_COLORS['dark']['gridcolor'],
            showgrid=True,
            zeroline=False,
            color=THEME_COLORS['dark']['text_color']
        ),
        yaxis=dict(
            title=y_label,
            gridcolor=THEME_COLORS['dark']['gridcolor'],
            showgrid=True,
            zeroline=False,
            color=THEME_COLORS['dark']['text_color']
        ),
        title=dict(
            text=title,
            font=dict(
                size=14,
                color=THEME_COLORS['dark']['text_color']
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
        'responsive': True,
    }
    
    # Generate plot HTML
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

# Add new function to plot_html.py
def create_multi_line_plot(y_values_list, labels, title, x_label="Layer", y_label="Value",
                          hover_mode="x unified", include_theme_js=False, include_plotlyjs=False):
    """
    Creates a multi-line plot with different colors for each line.
    
    Args:
        y_values_list: List of lists/tensors containing y-values for each line
        labels: List of labels for each line
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        hover_mode: Plotly hover mode
        include_theme_js: Whether to include theme sync JavaScript
        include_plotlyjs: Whether to include Plotly.js library
    """
    traces = []
    
    # Convert tensors to lists if necessary
    y_values_list = [y.tolist() if hasattr(y, 'tolist') else y for y in y_values_list]
    
    for i, (y_values, label) in enumerate(zip(y_values_list, labels)):
        trace = go.Scatter(
            x=list(range(len(y_values))),
            y=y_values,
            name=label,
            mode='lines+markers',
            line=dict(
                color=THEME_COLORS['dark']['multi_line_colors'][i],
                width=2
            ),
            marker=dict(
                size=6,
                color=THEME_COLORS['dark']['multi_line_colors'][i],
                line=dict(
                    color=THEME_COLORS['dark']['plot_bgcolor'],
                    width=1
                )
            ),
            hovertemplate=f'{label}<br>{x_label}: %{{x}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
        )
        traces.append(trace)

    layout = go.Layout(
        plot_bgcolor=THEME_COLORS['dark']['plot_bgcolor'],
        paper_bgcolor=THEME_COLORS['dark']['paper_bgcolor'],
        margin=dict(l=50, r=20, t=50, b=50, pad=4),
        xaxis=dict(
            title=x_label,
            gridcolor=THEME_COLORS['dark']['gridcolor'],
            showgrid=True,
            zeroline=False,
            color=THEME_COLORS['dark']['text_color']
        ),
        yaxis=dict(
            title=y_label,
            gridcolor=THEME_COLORS['dark']['gridcolor'],
            showgrid=True,
            zeroline=False,
            color=THEME_COLORS['dark']['text_color']
        ),
        title=dict(
            text=title,
            font=dict(size=14, color=THEME_COLORS['dark']['text_color']),
            x=0.5,
            xanchor='center'
        ),
        hoverlabel=dict(
            bgcolor='#333',
            font_size=12,
            font_family="monospace"
        ),
        hovermode=hover_mode,
        autosize=True,
        showlegend=True,
        legend=dict(
            font=dict(color=THEME_COLORS['dark']['text_color']),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    
    config = {
        'displayModeBar': False,
        'staticPlot': False,
        'responsive': True,
    }
    
    plot_html = f"""
    <div class="plot-container">
        {fig.to_html(
            full_html=False,
            include_plotlyjs='cdn' if include_plotlyjs else False,
            config=config
        )}
    </div>
    """
    
    if include_theme_js:
        plot_html += get_theme_sync_js()
    
    return plot_html

def with_identifier(identifier):
    """
    Decorator to wrap output in a div with an identifier.
    Can be used with any function that returns HTML content.
    
    Usage:
    @with_identifier("my-plot-1")
    def create_plot(...):
        ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Creating plot with identifier: {identifier}")
            output = func(*args, **kwargs)
            return f'<div id="{identifier}" class="identified-content">{output}</div>'
        return wrapper
    return decorator
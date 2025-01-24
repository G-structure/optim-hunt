"""Module for generating plots with plotly and managing plot themes."""

import logging
from typing import Any, Callable, Union, cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import torch

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
    """Return the JavaScript code for theme synchronization.
    This should be included once in the page.
    """
    return f"""
<script>
(function() {{
    function updatePlotTheme(plotDiv, isLight) {{
        console.log('Updating plot theme for', plotDiv.id, 'isLight:',
            isLight);
        try {{
            if (isLight) {{
                Plotly.relayout(plotDiv, {{
                    'plot_bgcolor': '{THEME_COLORS["light"]["plot_bgcolor"]}'
                    ,
                    'paper_bgcolor':
                        '{THEME_COLORS["light"]["paper_bgcolor"]}',
                    'xaxis.gridcolor':
                        '{THEME_COLORS["light"]["gridcolor"]}',
                    'yaxis.gridcolor':
                        '{THEME_COLORS["light"]["gridcolor"]}',
                    'xaxis.color': '{THEME_COLORS["light"]["text_color"]}',
                    'yaxis.color': '{THEME_COLORS["light"]["text_color"]}',
                    'title.font.color':
                        '{THEME_COLORS["light"]["text_color"]}',
                    'autosize': true
                }});
                Plotly.restyle(plotDiv, {{
                    'line.color': '{THEME_COLORS["light"]["bar_color"]}',
                    'marker.color': '{THEME_COLORS["light"]["bar_color"]}',
                    'marker.line.color':
                        '{THEME_COLORS["light"]["plot_bgcolor"]}'
                }});
            }} else {{
                Plotly.relayout(plotDiv, {{
                    'plot_bgcolor': '{THEME_COLORS["dark"]["plot_bgcolor"]}',
                    'paper_bgcolor':
                        '{THEME_COLORS["dark"]["paper_bgcolor"]}',
                    'xaxis.gridcolor': '{THEME_COLORS["dark"]["gridcolor"]}',
                    'yaxis.gridcolor': '{THEME_COLORS["dark"]["gridcolor"]}',
                    'xaxis.color': '{THEME_COLORS["dark"]["text_color"]}',
                    'yaxis.color': '{THEME_COLORS["dark"]["text_color"]}',
                    'title.font.color':
                        '{THEME_COLORS["dark"]["text_color"]}',
                    'autosize': true
                }});
                Plotly.restyle(plotDiv, {{
                    'line.color': '{THEME_COLORS["dark"]["bar_color"]}',
                    'marker.color': '{THEME_COLORS["dark"]["bar_color"]}',
                    'marker.line.color':
                        '{THEME_COLORS["dark"]["plot_bgcolor"]}'
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
            document.querySelectorAll('.plotly-graph-div')
                .forEach(function(plot) {{
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

def create_line_plot(
    y_values: Union[list[float], torch.Tensor],
    title: str,
    labels: list[str] | None = None,
    x_label: str = "Layer",
    y_label: str = "Value",
    hover_mode: str = "x unified",
    include_theme_js: bool = False,
    include_plotlyjs: bool = False
) -> str:
    """Create a lightweight, themed plot suitable for web embedding.

    Args:
        y_values: Tensor or list of values to plot
        title: Plot title
        labels: List of x-axis tick labels (optional)
        x_label: Label for x-axis (default: "Layer")
        y_label: Label for y-axis (default: "Value")
        hover_mode: Plotly hover mode (default: "x unified")
        include_theme_js: Whether to include the theme sync JavaScript
            (should only be True once)
        include_plotlyjs: Whether to include the Plotly.js library
            (should only be True once)

    Returns:
        str: HTML/JavaScript code for the plot

    """
    # Convert tensor to list if necessary
    if isinstance(y_values, torch.Tensor):
        y_values = cast(list[float], y_values.tolist())
    else:
        y_values = y_values

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
        hovertemplate=(
            f'{x_label}: %{{x}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
        )
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
    x_values: list[Union[str, float]],
    y_values: list[float],
    title: str,
    x_label: str = "",
    y_label: str = "",
    include_theme_js: bool = False,
    include_plotlyjs: bool = False,
    hover_template: str | None = None
) -> str:
    """Create a themed bar plot suitable for web embedding.

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
        hovertemplate=(hover_template if hover_template
                      else f"{x_label}: %{{x}}<br>{y_label}: %{{y:.3f}}"
                           f"<extra></extra>")
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

def create_multi_line_plot(
    y_values_list: list[list[float]],
    labels: list[str],
    title: str,
    x_label: str = "Layer",
    y_label: str = "Value",
    hover_mode: str = "x unified",
    include_theme_js: bool = False,
    include_plotlyjs: bool = False
) -> str:
    """Create a multi-line plot with different colors for each line.

    Args:
        y_values_list: List of lists/tensors containing y-values for each line
        labels: List of labels for each line
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        hover_mode: Plotly hover mode
        include_theme_js: Whether to include theme sync JavaScript
        include_plotlyjs: Whether to include Plotly.js library

    Returns:
        str: HTML string containing the plot

    """
    traces = []

    # Get the number of colors available
    num_colors = len(THEME_COLORS['dark']['multi_line_colors'])

    # Calculate number of legend rows needed (assuming ~3 items per row)
    num_legend_rows = (len(labels) + 2) // 3  # Add 2 for rounding up
    bottom_margin = max(100, 50 + (num_legend_rows * 20))  # 50px + 20px per row

    for i, (y_values, label) in enumerate(zip(y_values_list,
                                          labels, strict=True)):
        # Use modulo to cycle through colors
        color_index = i % num_colors

        trace = go.Scatter(
            x=list(range(len(y_values))),
            y=y_values,
            name=label,
            mode='lines+markers',
            line=dict(
                color=THEME_COLORS['dark']['multi_line_colors'][color_index],
                width=2
            ),
            marker=dict(
                size=6,
                color=THEME_COLORS['dark']['multi_line_colors'][color_index],
                line=dict(
                    color=THEME_COLORS['dark']['plot_bgcolor'],
                    width=1
                )
            ),
            hovertemplate=(
                f'{label}<br>{x_label}: %{{x}}<br>'
                f'{y_label}: %{{y:.3f}}<extra></extra>'
            )
        )
        traces.append(trace)

    layout = go.Layout(
        plot_bgcolor=THEME_COLORS['dark']['plot_bgcolor'],
        paper_bgcolor=THEME_COLORS['dark']['paper_bgcolor'],
        margin=dict(l=50, r=20, t=50, b=bottom_margin),  # Dynamic bottom margin
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
            bgcolor='rgba(0,0,0,0)',
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor to bottom
            y=-0.2 * (num_legend_rows / 3),  # Scale position based on rows
            xanchor="center",  # Center horizontally
            x=0.5,  # Center position
            traceorder="normal"
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

def create_multi_line_plot_layer_names(
    y_values_list: list[list[float]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    layer_names: list[str],
    hover_mode: str = 'x unified',
    include_plotlyjs: bool = True,
    include_theme_js: bool = True,
    active_lines: list[int] | None = None
) -> str:
    """Create a multi-line plot with custom layer names on x-axis.

    Args:
        y_values_list: List of lists containing y-values for each line
        labels: List of labels for each line
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        layer_names: List of names for x-axis ticks
        hover_mode: Plotly hover mode. Defaults to 'x unified'
        include_plotlyjs: Whether to include Plotly.js library
        include_theme_js: Whether to include theme sync JavaScript
        active_lines: List of line indices to show by default. Defaults to None.

    Returns:
        str: HTML string containing the plot

    """
    traces = []

    # Get the number of colors available
    num_colors = len(THEME_COLORS['dark']['multi_line_colors'])

    for i, (y_values, label) in enumerate(zip(y_values_list,
                                          labels, strict=True)):
        # Use modulo to cycle through colors
        color_index = i % num_colors

        # Determine if line is visable
        is_visible = (True if active_lines and
                     (i in active_lines or
                      i == len(y_values_list) + active_lines[0])
                     else "legendonly")

        trace = go.Scatter(
            x=layer_names,
            y=y_values,
            name=label,
            mode='lines+markers',
            line=dict(
                color=THEME_COLORS['dark']['multi_line_colors'][color_index],
                width=2
            ),
            marker=dict(
                size=2,
                color=THEME_COLORS['dark']['multi_line_colors'][color_index],
                line=dict(
                    color=THEME_COLORS['dark']['plot_bgcolor'],
                    width=1
                )
            ),
            opacity=1.0,
            hovertemplate=(f'{label}<br>{x_label}: %{{x}}<br>'
                          f'{y_label}: %{{y:.3f}}<extra></extra>'),
            visible=is_visible
        )
        traces.append(trace)

    layout = go.Layout(
        plot_bgcolor=THEME_COLORS['dark']['plot_bgcolor'],
        paper_bgcolor=THEME_COLORS['dark']['paper_bgcolor'],
        margin=dict(l=50, r=20, t=20, b=20),  # Dynamic bottom margin
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
            font=dict(color=THEME_COLORS['dark']['text_color'], size=10),
            bgcolor='rgba(0,0,0,0)',
            orientation="h",  # Horizontal orientation
            yanchor="top",  # Anchor to bottom
            y = -0.3,
            xanchor="center",  # Center horizontally
            x=0.5,  # Center position
            traceorder="normal"
        )
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

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

def create_heatmap_plot(
    z_values: Union[torch.Tensor, npt.NDArray[np.float64]],
    title: str,
    x_label: str = "Hidden Dimension",
    y_label: str = "Token Index",
    colorscale: list[list[float | str]] | None = None,
    zmid: float = 0,
    include_theme_js: bool = False,
    include_plotlyjs: bool = False
) -> str:
    """Create a heatmap plot with white at zero.

    Args:
        z_values: 2D array of values to plot (PyTorch tensor or NumPy array)
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        colorscale: Custom colorscale (optional)
        zmid: Middle value for color scale
        include_theme_js: Whether to include theme sync JavaScript
        include_plotlyjs: Whether to include Plotly.js library

    Returns:
        str: HTML string containing the plot

    """
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(z_values, torch.Tensor):
        z_values = cast(npt.NDArray[np.float64], z_values.detach()
                                                         .cpu()
                                                         .numpy())
    elif hasattr(z_values, 'numpy'):
        z_values = np.array(z_values, dtype=np.float64)

    if colorscale is None:
        colorscale = [
            [0, THEME_COLORS['dark']['multi_line_colors'][0]],
            [0.5, 'white'],  # White at zero
            [1, THEME_COLORS['dark']['multi_line_colors'][1]]
        ]

    # Get max absolute value for symmetric color scale
    max_abs_val = float(max(
        abs(np.min(z_values)),
        abs(np.max(z_values))
    ))

    trace = go.Heatmap(
        z=z_values,
        colorscale=colorscale,
        zmid=zmid,
        zmin=-max_abs_val,
        zmax=max_abs_val,
        showscale=True,
        colorbar=dict(
            title='Activation Value',
            titleside='right',
            thickness=15,
            len=0.75,
            tickfont=dict(color=THEME_COLORS['dark']['text_color']),
            title_font=dict(color=THEME_COLORS['dark']['text_color'])
        )
    )

    layout = go.Layout(
        plot_bgcolor=THEME_COLORS['dark']['plot_bgcolor'],
        paper_bgcolor=THEME_COLORS['dark']['paper_bgcolor'],
        margin=dict(l=50, r=50, t=50, b=50),
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
        autosize=True
    )

    fig = go.Figure(data=[trace], layout=layout)

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

def render_html_plot(html: str):
    """Display HTML plot with additional container wrapper.

    This adds a containing div with class plot-container around the raw HTML
    plot content and prints the result. Useful for consistent styling and layout.

    Args:
        html: Raw HTML plot content to wrap in container div.

    Returns:
        None - prints the wrapped HTML directly to output.

    """
    plot_html = f"""
    <div class="plot-container">
        {html}
    </div>"""
    print(plot_html)

def render_html_text(text: str):
    """Wrap text content in a text container div for rendering.

    Useful for displaying text content in a consistent style with plots
    and other HTML components.

    Args:
        text: Raw text content to wrap in HTML container div

    Returns:
        None - prints wrapped HTML text directly to output

    """
    text_html = f"""
    <div class="text-container">
        {text}
    </div>"""
    print(text_html)

def with_identifier(identifier: str) -> Callable[..., Callable[..., str]]:
    """Wrap output in a div with an identifier.

    Can be used with any function that returns HTML content.

    Args:
        identifier: String identifier for the div

    Usage:
        @with_identifier("my-plot-1")
        def create_plot(...):
            ...

    Returns:
        Decorated function that wraps HTML output in identified div

    """
    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        def wrapper(*args: Any, **kwargs: Any) -> str:
            logger.info(f"Creating plot with identifier: {identifier}")
            output: str = func(*args, **kwargs)
            return (
                f'<div id="{identifier}" class="identified-content">'
                f'{output}</div>'
            )
        return wrapper
    return decorator

def full_html():
    """Decorator to make plots use full HTML output."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store original to_html method
            original_to_html = go.Figure.to_html

            # Create modified to_html that forces full_html=True
            def modified_to_html(self, *th_args, **th_kwargs):
                th_kwargs['full_html'] = True
                return original_to_html(self, *th_args, **th_kwargs)

            # Replace to_html method
            go.Figure.to_html = modified_to_html

            try:
                # Run the original function
                result = func(*args, **kwargs)
            finally:
                # Restore original to_html method
                go.Figure.to_html = original_to_html

            return result
        return wrapper
    return decorator

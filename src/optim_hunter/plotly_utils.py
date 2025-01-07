"""Utilities for plotly visualization functions and interactive plotting.

This module provides a collection of helper functions for creating interactive
visualizations using plotly, particularly focused on visualizing transformer
model behavior and attention patterns.

Source https://github.com/callummcdougall/ARENA_3.0/blob/301319f65c1339a9204a466c075e4e80d8a9c94f/chapter1_transformer_interp/exercises/plotly_utils.py
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import einops
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from jaxtyping import Float
from plotly.subplots import make_subplots
from torch import Tensor


def to_numpy(
    tensor: Union[
        t.Tensor,
        List[Union[float, int]],
        Tuple[Union[float, int], ...],
        npt.NDArray[Any],
        int,
        float,
        bool,
        str
    ]
) -> npt.NDArray[Any]:
    """Convert input to numpy array format.

    Args:
        tensor: Input to convert. Can be tensor, list, tuple, array or scalar.

    Returns:
        np.ndarray: The input converted to a numpy array.

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif tensor in (int, float, bool, str):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


# GENERIC PLOTTING FUNCTIONS

update_layout_set = {
    "xaxis_range",
    "yaxis_range",
    "hovermode",
    "xaxis_title",
    "yaxis_title",
    "colorbar",
    "colorscale",
    "coloraxis",
    "title_x",
    "bargap",
    "bargroupgap",
    "xaxis_tickformat",
    "yaxis_tickformat",
    "title_y",
    "legend_title_text",
    "xaxis_showgrid",
    "xaxis_gridwidth",
    "xaxis_gridcolor",
    "yaxis_showgrid",
    "yaxis_gridwidth",
    "yaxis_gridcolor",
    "showlegend",
    "xaxis_tickmode",
    "yaxis_tickmode",
    "margin",
    "xaxis_visible",
    "yaxis_visible",
    "bargap",
    "bargroupgap",
    "coloraxis_showscale",
    "xaxis_tickangle",
    "yaxis_scaleanchor",
    "xaxis_tickfont",
    "yaxis_tickfont",
}

update_traces_set = {"textposition"}


def imshow(
    tensor: t.Tensor,
    renderer: Optional[str] = None,
    **kwargs: Union[str, int, float, bool, List[Any], Dict[str, Any]]
) -> Optional[go.Figure]:
    """Create and display an image plot using plotly express.

    Args:
        tensor: Input tensor to display as an image
        renderer: Plotly renderer to use for display
        **kwargs: Additional arguments passed to px.imshow() and update_layout()
            Valid kwargs include size/shape, facet_labels, border, return_fig,
            text, xaxis_tickangle, and any plotly layout parameters

    Returns:
        Optional[Figure]: If return_fig=True, returns the plotly Figure object
                         Otherwise returns None after displaying the plot

    """
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = cast(Optional[List[str]],
                       kwargs_pre.pop("facet_labels", None))
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size  # type: ignore
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = cast(Optional[List[List[str]]], kwargs_pre.pop("text", None))
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(
            list("tblr"),
            kwargs_post["margin"]
        )

    fig = px.imshow(
        to_numpy(tensor),
        **cast(Dict[str, Any], kwargs_pre)
    ).update_layout(**cast(Dict[str, Any], kwargs_post))

    if facet_labels:
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(
                facet_labels,
                cast(int, kwargs_pre["facet_col_wrap"])
            )
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]["text"] = label  # type: ignore

    if border:
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True
        )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True
        )

    if text:
        if tensor.ndim == 2:
            text = [text]  # type: ignore
        else:
            text_list = text
            if any(x for x in text_list):  # Fixed unnecessary isinstance check
                text = [text for _ in range(len(fig.data))]  # type: ignore
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text,
                texttemplate="%{text}",
                textfont={"size": 12}
            )

    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1 + n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle  # type: ignore

    return (fig if return_fig else fig.show(renderer=renderer,
            config={"staticPlot": static}))


def reorder_list_in_plotly_way[T](
    input_list: List[T],
    col_wrap: int
) -> List[T]:
    """Reorder a list according to Plotly's column wrap logic.

    Args:
        input_list: List to reorder
        col_wrap: Column width for wrapping

    Returns:
        List reordered according to Plotly's column wrap ordering

    """
    output_list: List[T] = []
    remaining = input_list.copy()
    while remaining:
        output_list.extend(remaining[-col_wrap:])
        remaining = remaining[:-col_wrap]
    return output_list


def line(
    y: Union[t.Tensor, List[Union[float, t.Tensor]]],
    renderer: Optional[str] = None,
    **kwargs: Union[str, int, float, bool, Dict[str, Any]]
) -> Optional[go.Figure]:
    """Create and display line plots with customizable layout.

    Args:
        y: Data to plot - can be tensor or list of values
        renderer: Plotly renderer to use
        **kwargs: Additional arguments passed to px.line() and update_layout()

    Returns:
        Optional[Figure]: Figure object if return_fig=True, otherwise None

    """
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        if isinstance(size, (list, tuple)) and len(size) == 2:
            kwargs_pre["height"], kwargs_pre["width"] = size
    return_fig = kwargs_pre.pop("return_fig", False)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(
            list("tblr"),
            kwargs_post["margin"]
        )

    if "xaxis_tickvals" in kwargs_pre:
        tickvals = cast(List[Any], kwargs_pre.pop("xaxis_tickvals"))
        x_range = kwargs_pre.get("x", np.arange(len(tickvals)))
        kwargs_post["xaxis"] = dict(
            tickmode="array",
            tickvals=x_range,
            ticktext=tickvals
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
        if ("use_secondary_yaxis" in kwargs_pre and
                kwargs_pre["use_secondary_yaxis"]):
                    del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels = cast(Dict[str, str], kwargs_pre.pop("labels"))
            kwargs_post["yaxis_title_text"] = labels.get("y1", "")
            kwargs_post["yaxis2_title_text"] = labels.get("y2", "")
            kwargs_post["xaxis_title_text"] = labels.get("x", "")
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(**cast(Dict[str, bool], kwargs_post))
        y0 = to_numpy(cast(t.Tensor, y[0]))
        y1 = to_numpy(cast(t.Tensor, y[1]))
        x0, x1 = cast(List[Any], kwargs_pre.pop(
            "x", [np.arange(len(y0)), np.arange(len(y1))]
        ))
        name0, name1 = cast(List[str], kwargs_pre.pop(
            "names", ["yaxis1", "yaxis2"]
        ))
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y_arr = (
            list(map(to_numpy, cast(List[Any], y)))
            if isinstance(y, list) and not (isinstance(y[0], (int, float)))
            else to_numpy(cast(t.Tensor, y))
        )
        names = kwargs_pre.pop("names", None)
        fig = px.line(y=y_arr, **cast(Dict[str, Any], kwargs_pre))
        fig.update_layout(**cast(Dict[str, bool], kwargs_post))
        if names is not None:
            names_list = cast(List[str], names)
            for trace in cast(List[Any], fig.data):
                trace.update(name=names_list.pop(0))
    return fig if return_fig else fig.show(renderer=renderer)


def scatter(
    x: Union[t.Tensor, List[float], npt.NDArray[np.float64]],
    y: Union[t.Tensor, List[float], npt.NDArray[np.float64]],
    renderer: Optional[str] = None,
    **kwargs: Union[str, int, float, bool, List[Any], Dict[str, Any]]
) -> Optional[go.Figure]:
    """Create a scatter plot with optional line and layout customization.

    Args:
        x: X-axis data points
        y: Y-axis data points
        renderer: Plotly renderer to use
        **kwargs: Additional arguments for plot customization including:
            add_line: Add reference line (e.g. "x=y" or "x=5")
            facet_labels: Labels for faceted subplots
            and any other valid plotly express scatter or layout parameters

    Returns:
        Optional[Figure]: Figure object if return_fig=True, else None
            after display

    """
    x_np = to_numpy(x)
    y_np = to_numpy(y)
    add_line: Optional[str] = None
    if "add_line" in kwargs:
        add_line = cast(str, kwargs.pop("add_line"))
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_traces = {k: v for k, v in kwargs.items() if k in update_traces_set}
    kwargs_pre = {
        k: v for k, v in kwargs.items()
        if k not in (update_layout_set | update_traces_set)
    }
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = cast(Tuple[int, int], size)
    return_fig = kwargs_pre.pop("return_fig", False)
    facet_labels = cast(
        Optional[List[str]],
        kwargs_pre.pop("facet_labels", None)
    )
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(
            list("tblr"),
            kwargs_post["margin"]
        )

    fig = px.scatter(y=y_np, x=x_np, **cast(Dict[str, Any], kwargs_pre))
    fig = fig.update_layout(**cast(Dict[str, Any], kwargs_post))

    if add_line is not None:
        x_min, x_max = float(x_np.min()), float(x_np.max())
        y_min, y_max = float(y_np.min()), float(y_np.max())
        xrange = [x_min, x_max]
        yrange = [y_min, y_max]

        add_line = add_line.replace(" ", "")
        if add_line in ["x=y", "y=x"]:
            fig.add_trace(
                go.Scatter(mode="lines", x=xrange, y=xrange, showlegend=False)
            )
        elif re.match("(x|y)=", add_line):
            try:
                c = float(add_line.split("=")[1])
            except ValueError:
                raise ValueError(
                    f"Unrecognized add_line: {add_line}. Use 'x=y', 'x=c' or "
                    "'y=c' for float c."
                )
            x_line = [c, c] if add_line[0] == "x" else xrange
            y_line = yrange if add_line[0] == "x" else [c, c]
            fig.add_trace(
                go.Scatter(mode="lines", x=x_line, y=y_line, showlegend=False)
            )
        else:
            raise ValueError(
                f"Unrecognized add_line: {add_line}. Use 'x=y', 'x=c' or "
                "'y=c' for float c."
            )

    if facet_labels and isinstance(fig.layout, go.Layout):
        if hasattr(fig.layout, 'annotations'):
            for i, label in enumerate(facet_labels):
                fig.layout.annotations[i]["text"] = label  # type: ignore

    fig.update_traces(**cast(Dict[str, Any], kwargs_traces))
    return fig if return_fig else fig.show(renderer=renderer)


def bar(
    tensor: Union[t.Tensor, List[Union[float, t.Tensor]], npt.NDArray[Any]],
    renderer: Optional[str] = None,
    **kwargs: Union[str, int, float, bool, List[Any], Dict[str, Any]]
) -> Optional[go.Figure]:
    """Create and display a bar plot using plotly express.

    Args:
        tensor: Input data to plot - can be tensor, list, or array
        renderer: Plotly renderer to use for display
        **kwargs: Additional arguments for plot customization

    Returns:
        Optional[Figure]: If return_fig=True, returns plotly Figure object
                         Otherwise returns None after displaying plot

    """
    if isinstance(tensor, list):
        if isinstance(tensor[0], t.Tensor):
            arr = [to_numpy(tn) for tn in tensor]
        elif isinstance(tensor[0], list):
            arr = [np.array(tn) for tn in tensor]
        else:
            arr = np.array(tensor)
    else:
        arr = to_numpy(tensor)

    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    return_fig = cast(bool, kwargs_pre.pop("return_fig", False))
    names = cast(Optional[List[str]], kwargs_pre.pop("names", None))

    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(
            list("tblr"),
            kwargs_post["margin"]
        )

    fig = (
        px.bar(y=arr, **cast(Dict[str, Any], kwargs_pre))
        .update_layout(**cast(Dict[str, Any], kwargs_post))
    )

    if names is not None:
        data_len = len(cast(List[Any], fig.data))
        for i in range(data_len):
            fig.data[i]["name"] = names[
                i // 2 if "marginal" in kwargs_pre else i
            ]

    return fig if return_fig else fig.show(renderer=renderer)


def hist(
    tensor: Union[t.Tensor, List[Union[float, t.Tensor]],
            npt.NDArray[np.float64]],
    renderer: Optional[str] = None,
    **kwargs: Union[str, int, float, bool, List[Any], Dict[str, Any]]
) -> Optional[go.Figure]:
    """Create and display a histogram using plotly express.

    Args:
        tensor: Input data to plot - can be tensor, list or array
        renderer: Plotly renderer to use
        **kwargs: Additional arguments for plot customization including:
            - nbins: Number of histogram bins
            - add_mean_line: Add vertical line at mean
            - names: Custom trace names
            and any other valid plotly express histogram parameters

    Returns:
        Optional[Figure]: If return_fig=True, returns the plotly Figure object
                         Otherwise returns None after displaying plot

    """
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    return_fig = cast(bool, kwargs_pre.pop("return_fig", False))

    if isinstance(tensor, list):
        if isinstance(tensor[0], t.Tensor):
            arr = [to_numpy(tn) for tn in tensor]
        elif isinstance(tensor[0], list):
            arr = [np.array(tn) for tn in tensor]
        else:
            arr = np.array(tensor)
    else:
        arr = to_numpy(tensor)

    if "modebar_add" not in kwargs_post:
        kwargs_post["modebar_add"] = [
            "drawline", "drawopenpath", "drawclosedpath",
            "drawcircle", "drawrect", "eraseshape"
        ]

    add_mean_line = cast(bool, kwargs_pre.pop("add_mean_line", False))
    names = cast(Optional[List[str]], kwargs_pre.pop("names", None))

    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        keys = list("tblr")
        kwargs_post["margin"] = dict.fromkeys(keys, kwargs_post["margin"])
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "autosize" not in kwargs_post:
        kwargs_post["autosize"] = False

    is_list_of_arrays = isinstance(arr, list) and any(x.ndim > 0 for x in arr)

    if is_list_of_arrays:
        assert "marginal" not in kwargs_pre, (
            "Can't use `marginal` with list of arrays"
        )
        for key in ["title", "template", "height", "width", "labels"]:
            if key in kwargs_pre:
                kwargs_post[key] = kwargs_pre.pop(key)
        if "labels" in kwargs_post:
            labels = cast(Dict[str, str], kwargs_post["labels"])
            kwargs_post["xaxis_title_text"] = labels.get("x", "")
            kwargs_post["yaxis_title_text"] = labels.get("y", "")
            del kwargs_post["labels"]

        fig = go.Figure(layout=go.Layout(**cast(Dict[str, Any], kwargs_post)))

        if "nbins" in kwargs_pre:
            kwargs_pre["nbinsx"] = int(cast(int, kwargs_pre.pop("nbins")))

        for x in arr:
            fig.add_trace(
                go.Histogram(
                    x=x,
                    name=names.pop(0) if names is not None else None,
                    **cast(Dict[str, Any], kwargs_pre)
                )
            )

    else:
        fig = (
            px.histogram(x=arr, **cast(Dict[str, Any], kwargs_pre))
            .update_layout(**cast(Dict[str, Any], kwargs_post))
        )
        if names is not None:
            data = cast(List[Any], fig.data)
            for i in range(len(data)):
                fig.data[i]["name"] = names[
                    i // 2 if "marginal" in kwargs_pre else i
                ]

    assert isinstance(arr, (np.ndarray, t.Tensor))

    if add_mean_line:
        if arr.ndim == 1:
            fig.add_vline(
                x=arr.mean(),
                line_width=3,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Mean = {arr.mean():.3f}",
                annotation_position="top",
            )
        elif arr.ndim == 2:
            for i in range(arr.shape[0]):
                fig.add_vline(
                    x=arr[i].mean(),
                    line_width=3,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"Mean = {arr.mean():.3f}",
                    annotation_position="top",
                )
    return fig if return_fig else fig.show(renderer=renderer)


# PLOTTING FUNCTIONS FOR PART 2: INTRO TO MECH INTERP


def plot_comp_scores(
    model: Any,
    comp_scores: t.Tensor,
    title: str = "",
    baseline: Optional[t.Tensor] = None
) -> None:
    """Plot component scores heatmap comparing attention head interactions.

    Args:
        model: The model being analyzed
        comp_scores: Tensor of component scores to visualize
        title: Optional plot title
        baseline: Optional tensor to use as color scale midpoint

    Returns:
        None

    """
    px.imshow(
        to_numpy(comp_scores),
        y=[f"L0H{h}" for h in range(cast(int, model.cfg.n_heads))],
        x=[f"L1H{h}" for h in range(cast(int, model.cfg.n_heads))],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title=title,
        color_continuous_scale="RdBu" if baseline is not None else "Blues",
        color_continuous_midpoint=baseline if baseline is not None else None,
        zmin=None if baseline is not None else 0.0,
    ).show()


def convert_tokens_to_string(
    model: Any,
    tokens: t.Tensor,
    batch_index: int = 0
) -> List[str]:
    """Convert tokens into a list of strings for printing."""
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [
        f"|{model.tokenizer.decode(int(tok))}|_{c}"
        for c, tok in enumerate(tokens)
    ]


def plot_logit_attribution(
    model: Any,
    logit_attr: t.Tensor,
    tokens: t.Tensor,
    title: str = ""
) -> None:
    """Plot logit attributions for tokens using an attention heatmap.

    Args:
        model: Model being analyzed
        logit_attr: Tensor of attributions to plot
        tokens: Input token IDs
        title: Optional plot title

    Returns:
        None

    """
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(model, tokens[:-1])
    x_labels = ["Direct"] + [
        f"L{layer}H{head}"
        for layer in range(cast(int, model.cfg.n_layers))
        for head in range(cast(int, model.cfg.n_heads))
    ]
    imshow(
        to_numpy(logit_attr),  # type: ignore
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        title=title if title else "",
        height=18 * len(y_labels),
        width=24 * len(x_labels),
    )


# PLOTTING FUNCTIONS FOR PART 4: INTERP ON ALGORITHMIC MODEL

color_discrete_map = dict(
    zip(
        ["both failures", "just neg failure", "balanced",
         "just total elevation failure"],
        px.colors.qualitative.D3,
    )
)
# names = ["balanced", "just total elevation failure",
#     "just neg failure", "both failures"]
# colors = ['#2CA02C', '#1c96eb', '#b300ff', '#ff4800']
# color_discrete_map = dict(zip(names, colors))


def plot_failure_types_scatter(
    unbalanced_component_1: Float[t.Tensor, "*batch"],  # noqa: F722
    unbalanced_component_2: Float[t.Tensor, "*batch"],  # noqa: F722
    failure_types_dict: dict[str, Float[t.Tensor, "*batch"]],  # noqa: F722
    data: Any,
) -> None:
    """Create scatter plot comparing different failure types between two heads.

    Args:
        unbalanced_component_1: First head component contributions
        unbalanced_component_2: Second head component contributions
        failure_types_dict: Maps failure type names to boolean tensor masks
        data: Dataset containing failure information and filters

    Returns:
        None

    """
    failure_types = np.full(
        len(unbalanced_component_1), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)

    failures_df = cast(pd.DataFrame, pd.DataFrame({
        "Head 2.0 contribution": to_numpy(unbalanced_component_1),
        "Head 2.1 contribution": to_numpy(unbalanced_component_2),
        "Failure type": to_numpy(failure_types),
    })[data.starts_open.tolist()])

    fig = px.scatter(
        failures_df,
        color_discrete_map=color_discrete_map,
        x="Head 2.0 contribution",
        y="Head 2.1 contribution",
        color="Failure type",
        title="h20 vs h21 for different failure types",
        template="simple_white",
        height=600,
        width=800,
    ).update_traces(marker_size=4)
    fig.show()


def plot_contribution_vs_open_proportion(
    unbalanced_component: Float[Tensor, "batch"],  # noqa: F821
    title: str,
    failure_types_dict: Dict[str, Tensor],
    data: Any
) -> None:
    """Plot scatter plot comparing component contributions vs open proportion.

    Args:
        unbalanced_component: Tensor of component contributions in unbalanced
            direction
        title: Plot title string
        failure_types_dict: Maps failure type names to boolean tensor masks
        data: Dataset containing failure examples

    Returns:
        None

    """
    # Initialize failure types array
    failure_types = np.full(len(unbalanced_component), "",
                            dtype=np.dtype("U32"))

    # Fill in failure types based on masks
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)

    # Create scatter plot
    fig = (
        px.scatter(
            x=to_numpy(data.open_proportion),
            y=to_numpy(unbalanced_component),
            color=failure_types,
            color_discrete_map=color_discrete_map,
            title=title,
            template="simple_white",
            height=500,
            width=800,
            labels={
                "x": "Open-proportion",
                "y": f"Head {title} contribution"
            },
        )
        .update_traces(marker_size=4, opacity=0.5)
        .update_layout(legend_title_text="Failure type")
    )
    fig.show()


def mlp_attribution_scatter(
    out_by_component_in_pre_20_unbalanced_dir: Float[
        Tensor, "components batches"],  # noqa: F722
    data: Any,
    failure_types_dict: Dict[str, Tensor],
) -> None:
    """Plot MLP attribution scatterplots.

    Args:
        out_by_component_in_pre_20_unbalanced_dir: Tensor of component outputs
            in unbalanced direction
        data: Dataset containing failure examples and metadata
        failure_types_dict: Dictionary mapping failure type
                            names to boolean masks

    Returns:
        None

    """
    # Initialize failure type array
    failure_types = np.full(
        out_by_component_in_pre_20_unbalanced_dir.shape[-1], "",
        dtype=np.dtype("U32")
    )

    # Fill in failure types based on masks
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)

    # Plot scatter for each layer
    for layer in range(2):
        mlp_output = out_by_component_in_pre_20_unbalanced_dir[3 + layer * 3]
        fig = (
            px.scatter(
                x=to_numpy(data.open_proportion[data.starts_open]),
                y=to_numpy(mlp_output[data.starts_open]),
                color_discrete_map=color_discrete_map,
                color=to_numpy(failure_types)[to_numpy(data.starts_open)],
                title=(
                    f"Amount MLP {layer} writes in unbalanced direction "
                    f"for Head 2.0"
                ),
                template="simple_white",
                height=500,
                width=800,
                labels={"x": "Open-proportion", "y": "Head 2.0 contribution"},
            )
            .update_traces(marker_size=4, opacity=0.5)
            .update_layout(legend_title_text="Failure type")
        )
        fig.show()


def plot_neurons(
    neurons_in_unbalanced_dir: Float[Tensor, "batch neurons"],  # noqa: F722
    model: Any,
    data: Any,
    failure_types_dict: Dict[str, Tensor],
    layer: int,
    renderer: Optional[str] = None,
) -> None:
    """Plot neuron contributions showing failure type patterns.

    Args:
        neurons_in_unbalanced_dir: Tensor of neuron activations in
           unbalanced direction
        model: The transformer model being analyzed
        data: Dataset containing failure examples
        failure_types_dict: Dictionary mapping failure type names to
           boolean masks
        layer: Network layer index to plot
        renderer: Optional plotly renderer to use

    Returns:
        None

    """
    failure_types = np.full(neurons_in_unbalanced_dir.shape[0], "",
        dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(
            to_numpy(mask[to_numpy(data.starts_open)]),
            name,
            failure_types
        )

    # Get data that can be turned into a dataframe (plotly express is sometimes
    # easier to use with a dataframe)
    # Plot scatter plot of neuron contributions, color-coded by failure type,
    # with slider to view neurons
    neuron_numbers = einops.repeat(
        t.arange(model.cfg.d_model), "n -> (s n)", s=data.starts_open.sum()
    )
    failure_types = einops.repeat(
        failure_types, "s -> (s n)", n=model.cfg.d_model
    )
    data_open_proportion = einops.repeat(
        data.open_proportion[data.starts_open], "s -> (s n)",
        n=model.cfg.d_model
    )
    df = pd.DataFrame(
        {
            "Output in 2.0 direction": to_numpy(
                neurons_in_unbalanced_dir.flatten()),
            "Neuron number": to_numpy(neuron_numbers),
            "Open-proportion": to_numpy(data_open_proportion),
            "Failure type": failure_types,
        }
    )
    fig = (
        px.scatter(
            df,
            x="Open-proportion",
            y="Output in 2.0 direction",
            color="Failure type",
            animation_frame="Neuron number",
            title=f"Neuron contributions from layer {layer}",
            template="simple_white",
            height=800,
            width=1100,
        )
        .update_traces(marker_size=3)
        .update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5])
    )
    fig.show(renderer=renderer)


def plot_attn_pattern(
    pattern: Float[Tensor, "batch head_idx seqQ seqK"]  # noqa: F722
) -> None:
    """Plot attention pattern heatmap for parentheses attention analysis.

    Args:
        pattern: Attention pattern tensor w/ shape [batch, head_idx, seqQ, seqK]
                containing attention probabilities

    Returns:
        None

    """
    fig = px.imshow(
        pattern,
        title="Estimate for avg attn probabilities when query is from '('",
        labels={
            "x": "Key tokens (avg of left & right parens)",
            "y": "Query tokens (all left parens)",
        },
        height=900,
        width=900,
        color_continuous_scale="RdBu_r",
        range_color=[0, pattern.max().item()],
    ).update_layout(
        xaxis=dict(
            tickmode="array",
            ticktext=["[start]", *[f"{i+1}" for i in range(40)], "[end]"],
            tickvals=list(range(42)),
            tickangle=0,
        ),
        yaxis=dict(
            tickmode="array",
            ticktext=["[start]", *[f"{i+1}" for i in range(40)], "[end]"],
            tickvals=list(range(42)),
        ),
    )
    fig.show()


def hists_per_comp(
    out_by_component_in_unbalanced_dir: Float[Tensor, "components batches"],  # noqa: F722
    data: Any,
    xaxis_range: tuple[float, float] = (-1, 1)
):
    """Plot histograms showing component contributions in unbalanced direction.

    Args:
        out_by_component_in_unbalanced_dir: Tensor of component outputs
            projected onto unbalanced direction
        data: Dataset containing examples and metadata
        xaxis_range: Optional tuple defining x-axis range for plots

    Returns:
        None

    """
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0",
        (2, 2): "head 0.1",
        (2, 3): "mlp 0",
        (3, 1): "head `1.0`",
        (3, 2): "head `1.1`",
        (3, 3): "mlp 1",
        (4, 1): "head 2.0",
        (4, 2): "head 2.1",
        (4, 3): "mlp 2",
    }
    n_layers = out_by_component_in_unbalanced_dir.shape[0] // 3
    fig = make_subplots(rows=n_layers + 1, cols=3)
    for ((row, col), title), in_dir in zip(titles.items(),
                                          out_by_component_in_unbalanced_dir):
        fig.add_trace(
            go.Histogram(
                x=to_numpy(in_dir[data.isbal]),
                name="Balanced",
                marker_color="blue",
                opacity=0.5,
                legendgroup="1",
                showlegend=title == "embeddings",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Histogram(
                x=to_numpy(in_dir[~data.isbal]),
                name="Unbalanced",
                marker_color="red",
                opacity=0.5,
                legendgroup="2",
                showlegend=title == "embeddings",
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text=title,
            row=row,
            col=col,
            range=list(xaxis_range)
        )
    fig.update_layout(
        width=1200,
        height=250 * (n_layers + 1),
        barmode="overlay",
        legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4),
        title="Histograms of component significance",
    )
    fig.show()


def plot_loss_difference(
    log_probs: Tensor,
    rep_str: list[str],
    seq_len: int
) -> None:
    """Plot log probabilities comparison between two sequence sections.

    Args:
        log_probs: Tensor of log probabilities for each position
        rep_str: List of string tokens for hover labels
        seq_len: Length of one sequence section

    Returns:
        None

    """
    fig = px.line(
        to_numpy(log_probs),
        hover_name=rep_str[1:],
        title=(
            f"Per token log prob on correct token, for sequence of length "
            f"{seq_len}*2 (repeated twice)"
        ),
        labels={"index": "Sequence position", "value": "Log prob"},
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(
        x0=0,
        x1=seq_len - 0.5,
        fillcolor="red",
        opacity=0.2,
        line_width=0
    )
    fig.add_vrect(
        x0=seq_len - 0.5,
        x1=2 * seq_len - 1,
        fillcolor="green",
        opacity=0.2,
        line_width=0
    )
    fig.show()

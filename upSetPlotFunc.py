"""Customizable UpSet plots built on matplotlib.

An UpSet plot shows set membership without the 2^n overlap soup of a Venn
diagram. Exclusive intersections become columns (horizontal) or rows
(vertical); sets become the other axis.

Vertical orientation grows downward, so more intersections fit, and the set
bars on top can stack Present vs Missing (NaN).

Do not call ``tight_layout()``; panel positions are already computed.

Notebook / interactive display
------------------------------
Creating a figure already registers it with matplotlib's pyplot state, so
Jupyter's inline backend will render it once when the cell finishes. Calling
``display(result.fig)`` **and** ``plt.show()`` (or returning the figure as the
cell's last expression on top of that) draws the same plot twice. Prefer
exactly one of: leave the figure open, ``display(result.fig)``, or
``plt.show()``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

Color: TypeAlias = str | tuple[float, ...]
SortIntersections: TypeAlias = Literal["size", "degree", "input"]
SortSets: TypeAlias = Literal["size", "input"]
IntersectionColorMode: TypeAlias = Literal["uniform", "degree", "exclusive-set"]
Orientation: TypeAlias = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class UpSetResult:
    """Drawn figure plus the data that was actually plotted."""

    fig: Figure
    ax_intersections: Axes
    ax_sets: Axes
    ax_matrix: Axes
    ax_labels: Axes
    set_names: tuple[str, ...]
    set_sizes: tuple[int, ...]
    nan_counts: tuple[int, ...]
    intersections: tuple[frozenset[str], ...]
    intersection_sizes: tuple[int, ...]
    truncated: bool
    orientation: Orientation


@dataclass
class UpSetStyle:
    """Visual knobs. Any field can be overridden when calling :func:`upset_plot`."""

    intersection_color: Color | Sequence[Color] | Mapping[frozenset[str], Color] = (
        "#3A3A3A"
    )
    set_color: Color | Sequence[Color] | Mapping[str, Color] = "#3A3A3A"
    matrix_fill_color: Color = "#2B2B2B"
    matrix_empty_color: Color = "#D4D4D4"
    matrix_empty_alpha: float = 0.55
    matrix_line_color: Color | None = None
    row_stripe_colors: tuple[Color, Color] = ("#FFFFFF", "#F0F0F0")
    highlight_color: Color = "#C44E52"
    nan_color: Color = "#C8C8C8"
    bar_edgecolor: Color | None = None
    bar_linewidth: float = 0.0
    intersection_bar_width: float = 0.62
    set_bar_height: float = 0.62
    line_width: float | None = None
    dot_size: float | None = None
    fontsize: float | None = None
    label_fontsize: float | None = None
    count_fontsize: float | None = None
    title_fontsize: float | None = None
    colormap: str | Colormap = "viridis"
    color_dots_by_set: bool = False
    facecolor: Color = "white"


def _as_boolean_frame(
    data: pd.DataFrame | Mapping[str, Collection[Any]],
    sets: Sequence[str] | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Accept a boolean membership table or a mapping of set name -> members.

    Original NaNs are counted per set, then treated as False for intersections.
    Values that are neither numeric membership nor NaN still raise.
    """
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if sets is not None:
            missing = [name for name in sets if name not in frame.columns]
            if missing:
                raise KeyError(f"Requested sets are not columns: {missing}")
            frame = frame.loc[:, list(sets)]
        original_na = frame.isna()
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        introduced_na = numeric.isna() & ~original_na
        if introduced_na.any().any():
            bad = introduced_na.any()
            raise TypeError(
                "DataFrame columns must be boolean, 0/1, or NaN membership. "
                f"Non-numeric values in: {list(bad[bad].index)}. "
                "Pass a dict[str, Collection] if you have element IDs per set."
            )
        nan_counts = original_na.sum().astype(int).to_dict()
        return numeric.fillna(0).astype(bool), nan_counts

    if isinstance(data, Mapping):
        names = list(sets) if sets is not None else list(data.keys())
        missing = [name for name in names if name not in data]
        if missing:
            raise KeyError(f"Requested sets are not in the mapping: {missing}")
        memberships = {name: set(data[name]) for name in names}
        universe: set[Any] = set()
        for members in memberships.values():
            universe.update(members)
        index = sorted(universe, key=lambda item: str(item))
        frame = pd.DataFrame(
            {name: [item in memberships[name] for item in index] for name in names},
            index=index,
        )
        return frame, {name: 0 for name in names}

    raise TypeError(
        "data must be a pandas DataFrame or a mapping of set name -> members"
    )


def _exclusive_intersections(frame: pd.DataFrame) -> list[tuple[frozenset[str], int]]:
    """Count rows by exclusive membership pattern. All-false rows are dropped."""
    if frame.empty or frame.shape[1] == 0:
        return []

    active = frame.loc[frame.any(axis=1)]
    if active.empty:
        return []

    columns = list(active.columns)
    counts = active.groupby(columns, sort=False).size()
    records: list[tuple[frozenset[str], int]] = []
    for key, count in counts.items():
        flags = key if isinstance(key, tuple) else (key,)
        members = frozenset(name for name, flag in zip(columns, flags) if flag)
        if members:
            records.append((members, int(count)))
    return records


def _sort_sets(
    names: Sequence[str],
    sizes: Mapping[str, int],
    how: SortSets,
) -> list[str]:
    if how == "input":
        return list(names)
    if how == "size":
        # Largest set first in the list: bottom of a horizontal matrix, left of
        # a vertical matrix.
        return sorted(names, key=lambda name: (sizes[name], name), reverse=True)
    raise ValueError("sort_sets must be 'size' or 'input'")


def _sort_intersections(
    records: Sequence[tuple[frozenset[str], int]],
    how: SortIntersections,
    set_order: Sequence[str],
) -> list[tuple[frozenset[str], int]]:
    rank = {name: index for index, name in enumerate(set_order)}

    def pattern_key(members: frozenset[str]) -> tuple[int, ...]:
        return tuple(sorted(rank[name] for name in members))

    if how == "input":
        return list(records)
    if how == "size":
        return sorted(
            records, key=lambda item: (-item[1], -len(item[0]), pattern_key(item[0]))
        )
    if how == "degree":
        return sorted(
            records, key=lambda item: (-len(item[0]), -item[1], pattern_key(item[0]))
        )
    raise ValueError("sort_intersections must be 'size', 'degree', or 'input'")


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _label_span(longest_label: int, fontsize: float) -> float:
    return max(0.8, longest_label * fontsize * 0.62 / 72.0 + 0.32)


def _panel_geometry(
    n_sets: int,
    n_intersections: int,
    longest_label: int,
    element_size: float,
    set_bar_width: float,
    intersection_bar_height: float,
    fontsize: float,
    show_intersection_counts: bool,
    show_set_counts: bool,
    has_title: bool,
    show_nan: bool,
    orientation: Orientation,
    count_span: float,
    fig_width: float | None,
    fig_height: float | None,
) -> dict[str, Any]:
    """Figure-fraction rects that keep the four panels spine-aligned."""
    label_span = _label_span(longest_label, fontsize)
    if orientation == "horizontal":
        return _geometry_horizontal(
            n_sets=n_sets,
            n_intersections=n_intersections,
            label_span=label_span,
            element_size=element_size,
            set_bar_width=set_bar_width,
            intersection_bar_height=intersection_bar_height,
            show_intersection_counts=show_intersection_counts,
            show_set_counts=show_set_counts,
            has_title=has_title,
            show_nan=show_nan,
            count_span=count_span,
            fig_width=fig_width,
            fig_height=fig_height,
        )
    return _geometry_vertical(
        n_sets=n_sets,
        n_intersections=n_intersections,
        label_span=label_span,
        element_size=element_size,
        set_bar_width=set_bar_width,
        intersection_bar_height=intersection_bar_height,
        show_intersection_counts=show_intersection_counts,
        show_set_counts=show_set_counts,
        has_title=has_title,
        show_nan=show_nan,
        count_span=count_span,
        fig_width=fig_width,
        fig_height=fig_height,
    )


def _scale_to_figure(
    content_w: float,
    content_h: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    fig_width: float | None,
    fig_height: float | None,
) -> tuple[float, float, float, float]:
    needed_w = left + content_w + right
    needed_h = bottom + content_h + top
    fig_w = float(fig_width or needed_w)
    fig_h = float(fig_height or needed_h)
    scale_x = max(fig_w - left - right, 0.5) / content_w
    scale_y = max(fig_h - bottom - top, 0.5) / content_h
    return fig_w, fig_h, scale_x, scale_y


def _geometry_horizontal(
    n_sets: int,
    n_intersections: int,
    label_span: float,
    element_size: float,
    set_bar_width: float,
    intersection_bar_height: float,
    show_intersection_counts: bool,
    show_set_counts: bool,
    has_title: bool,
    show_nan: bool,
    count_span: float,
    fig_width: float | None,
    fig_height: float | None,
) -> dict[str, Any]:
    matrix_w = max(n_intersections, 1) * element_size
    matrix_h = max(n_sets, 1) * element_size
    inter_ylabel_w = 0.62
    h_gap = 0.12
    w_gap_sets_labels = 0.05
    left = 0.58
    right = 0.16
    bottom = 0.55
    top = 0.50 if show_intersection_counts else 0.24
    if has_title:
        top += 0.42
    if show_nan:
        top = max(top, 0.55)
    _ = count_span
    _ = show_set_counts

    content_w = set_bar_width + w_gap_sets_labels + label_span + inter_ylabel_w + matrix_w
    content_h = intersection_bar_height + h_gap + matrix_h
    fig_w, fig_h, scale_x, scale_y = _scale_to_figure(
        content_w, content_h, left, right, bottom, top, fig_width, fig_height
    )
    set_bar_width *= scale_x
    label_span *= scale_x
    inter_ylabel_w *= scale_x
    matrix_w *= scale_x
    w_gap_sets_labels *= scale_x
    intersection_bar_height *= scale_y
    matrix_h *= scale_y
    h_gap *= scale_y

    x_sets = left / fig_w
    x_labels = (left + set_bar_width + w_gap_sets_labels) / fig_w
    x_matrix = (
        left + set_bar_width + w_gap_sets_labels + label_span + inter_ylabel_w
    ) / fig_w
    y_matrix = bottom / fig_h
    y_inter = (bottom + matrix_h + h_gap) / fig_h
    return {
        "fig_width": fig_w,
        "fig_height": fig_h,
        "sets": (x_sets, y_matrix, set_bar_width / fig_w, matrix_h / fig_h),
        "labels": (x_labels, y_matrix, label_span / fig_w, matrix_h / fig_h),
        "matrix": (x_matrix, y_matrix, matrix_w / fig_w, matrix_h / fig_h),
        "counts": None,
        "inter": (x_matrix, y_inter, matrix_w / fig_w, intersection_bar_height / fig_h),
    }


def _geometry_vertical(
    n_sets: int,
    n_intersections: int,
    label_span: float,
    element_size: float,
    set_bar_width: float,
    intersection_bar_height: float,
    show_intersection_counts: bool,
    show_set_counts: bool,
    has_title: bool,
    show_nan: bool,
    count_span: float,
    fig_width: float | None,
    fig_height: float | None,
) -> dict[str, Any]:
    """Sets along x (bars on top), intersections along y (bars on the right)."""
    matrix_w = max(n_sets, 1) * element_size
    matrix_h = max(n_intersections, 1) * element_size
    set_panel_h = set_bar_width
    inter_panel_w = intersection_bar_height
    h_gap = 0.08
    w_gap = count_span if show_intersection_counts else 0.12
    inter_xlabel_h = 0.50
    set_ylabel_w = 0.58
    left = set_ylabel_w
    right = 0.22
    bottom = inter_xlabel_h
    top = 0.18
    if show_set_counts:
        top += 0.58
    if has_title:
        top += 0.48
    if show_nan:
        top += 0.10

    content_w = matrix_w + w_gap + inter_panel_w
    content_h = set_panel_h + h_gap + label_span + h_gap + matrix_h
    fig_w, fig_h, scale_x, scale_y = _scale_to_figure(
        content_w, content_h, left, right, bottom, top, fig_width, fig_height
    )
    matrix_w *= scale_x
    inter_panel_w *= scale_x
    w_gap *= scale_x
    set_panel_h *= scale_y
    label_span *= scale_y
    matrix_h *= scale_y
    h_gap *= scale_y

    x_matrix = left / fig_w
    y_matrix = bottom / fig_h
    y_labels = (bottom + matrix_h + h_gap) / fig_h
    y_sets = (bottom + matrix_h + h_gap + label_span + h_gap) / fig_h
    x_counts = (left + matrix_w) / fig_w
    x_inter = (left + matrix_w + w_gap) / fig_w
    w_matrix = matrix_w / fig_w
    h_matrix = matrix_h / fig_h
    return {
        "fig_width": fig_w,
        "fig_height": fig_h,
        "sets": (x_matrix, y_sets, w_matrix, set_panel_h / fig_h),
        "labels": (x_matrix, y_labels, w_matrix, label_span / fig_h),
        "matrix": (x_matrix, y_matrix, w_matrix, h_matrix),
        "counts": (x_counts, y_matrix, w_gap / fig_w, h_matrix),
        "inter": (x_inter, y_matrix, inter_panel_w / fig_w, h_matrix),
    }


def _colors_for_sets(
    spec: Color | Sequence[Color] | Mapping[Any, Color],
    input_names: Sequence[str],
    display_names: Sequence[str],
) -> list[Color]:
    """Sequence colors follow input column order; mappings follow set names."""
    if isinstance(spec, Mapping) or _is_single_color(spec):
        return _resolve_colors(spec, display_names, "#3A3A3A")
    by_input = dict(zip(input_names, _resolve_colors(spec, input_names, "#3A3A3A")))
    return [by_input.get(name, "#3A3A3A") for name in display_names]


def _resolve_colors(
    spec: Color | Sequence[Color] | Mapping[Any, Color],
    keys: Sequence[Any],
    default: Color,
) -> list[Color]:
    if isinstance(spec, Mapping):
        return [spec.get(key, default) for key in keys]
    if isinstance(spec, (list, tuple, np.ndarray)) and not _is_single_color(spec):
        palette = list(spec)
        if not palette:
            return [default] * len(keys)
        return [palette[index % len(palette)] for index in range(len(keys))]
    return [spec] * len(keys)  # type: ignore[list-item]


def _is_single_color(spec: Any) -> bool:
    if isinstance(spec, str):
        return True
    if (
        isinstance(spec, tuple)
        and spec
        and all(isinstance(part, (int, float, np.floating)) for part in spec)
    ):
        return True
    try:
        to_rgba(spec)
        return True
    except (ValueError, TypeError):
        return False


def _intersection_colors(
    intersections: Sequence[frozenset[str]],
    sizes: Sequence[int],
    set_names: Sequence[str],
    set_colors: Sequence[Color],
    style: UpSetStyle,
    mode: IntersectionColorMode,
    highlight: Collection[frozenset[str]],
) -> list[Color]:
    set_color_map = dict(zip(set_names, set_colors))
    if mode == "uniform":
        colors = _resolve_colors(style.intersection_color, intersections, "#3A3A3A")
    elif mode == "exclusive-set":
        colors = []
        fallback = (
            style.intersection_color
            if _is_single_color(style.intersection_color)
            else "#3A3A3A"
        )
        for members in intersections:
            if len(members) == 1:
                name = next(iter(members))
                colors.append(set_color_map.get(name, fallback))  # type: ignore[arg-type]
            else:
                colors.append(fallback)  # type: ignore[arg-type]
    elif mode == "degree":
        cmap = (
            plt.get_cmap(style.colormap)
            if isinstance(style.colormap, str)
            else style.colormap
        )
        degrees = np.array([len(members) for members in intersections], dtype=float)
        norm = Normalize(vmin=max(degrees.min(), 1.0), vmax=max(degrees.max(), 1.0))
        colors = [cmap(norm(degree)) for degree in degrees]
    else:
        raise ValueError(
            "intersection_color_mode must be 'uniform', 'degree', or 'exclusive-set'"
        )

    highlighted = {frozenset(item) for item in highlight}
    if highlighted:
        colors = [
            style.highlight_color if members in highlighted else color
            for members, color in zip(intersections, colors)
        ]
    _ = sizes
    return colors


def _strip_spines(ax: Axes, keep: Collection[str] = ()) -> None:
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


def _count_label_span(values: Sequence[int], fontsize: float) -> float:
    if not values:
        return 0.4
    widest = max(len(f"{value:g}") for value in values)
    return max(0.55, widest * fontsize * 0.62 / 72.0 + 0.22)


def _annotate_counts(
    ax: Axes,
    positions: np.ndarray,
    values: Sequence[float],
    *,
    along: Literal["x", "y"],
    fontsize: float,
    ha: str,
    va: str,
    at: Literal["end", "origin"] = "end",
) -> None:
    peak = float(max(values)) if len(values) else 0.0
    offset = peak * 0.02 if peak else 0.1
    box = {"facecolor": "white", "edgecolor": "none", "pad": 0.35, "alpha": 0.92}
    for pos, value in zip(positions, values):
        label = f"{value:g}"
        if along == "x":
            y = (value + offset) if at == "end" else 0.0
            ax.text(
                pos,
                y,
                label,
                ha=ha,
                va=va,
                fontsize=fontsize,
                color="#222222",
                clip_on=False,
                zorder=6,
                bbox=box,
            )
        else:
            x = (value + offset) if at == "end" else 0.0
            ax.text(
                x,
                pos,
                label,
                ha=ha,
                va=va,
                fontsize=fontsize,
                color="#222222",
                clip_on=False,
                zorder=6,
                bbox=box,
            )


def _draw_set_bars(
    ax: Axes,
    ordered_sets: Sequence[str],
    set_sizes: Mapping[str, int],
    nan_counts: Mapping[str, int],
    set_colors: Sequence[Color],
    style: UpSetStyle,
    *,
    orientation: Orientation,
    invert_set_bars: bool,
    show_nan: bool,
    show_set_counts: bool,
    set_xlabel: str,
    label_font: float,
    count_font: float,
) -> None:
    positions = np.arange(len(ordered_sets))
    present = np.array([set_sizes[name] for name in ordered_sets], dtype=float)
    missing = np.array([nan_counts.get(name, 0) for name in ordered_sets], dtype=float)
    totals = present + missing if show_nan else present
    bar_kwargs = {
        "color": set_colors,
        "edgecolor": style.bar_edgecolor,
        "linewidth": style.bar_linewidth,
        "zorder": 3,
        "label": "Present",
    }
    if orientation == "horizontal":
        ax.barh(positions, present, height=style.set_bar_height, **bar_kwargs)
        if show_nan:
            ax.barh(
                positions,
                missing,
                left=present,
                height=style.set_bar_height,
                color=style.nan_color,
                edgecolor=style.bar_edgecolor,
                linewidth=style.bar_linewidth,
                zorder=3,
                label="Missing (NaN)",
            )
        ax.set_xlabel(set_xlabel if not show_nan else f"{set_xlabel} / missing", fontsize=label_font)
        ax.tick_params(axis="x", labelsize=count_font, length=3)
        ax.tick_params(axis="y", left=False, labelleft=False)
        _strip_spines(ax, keep=("left", "bottom"))
        ax.set_ylim(-0.5, len(ordered_sets) - 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.6)
        if invert_set_bars:
            ax.invert_xaxis()
        if show_set_counts:
            _annotate_counts(
                ax,
                positions,
                totals,
                along="y",
                fontsize=count_font,
                ha="right" if invert_set_bars else "left",
                va="center",
            )
        return

    ax.bar(positions, present, width=style.set_bar_height, **bar_kwargs)
    if show_nan:
        ax.bar(
            positions,
            missing,
            bottom=present,
            width=style.set_bar_height,
            color=style.nan_color,
            edgecolor=style.bar_edgecolor,
            linewidth=style.bar_linewidth,
            zorder=3,
            label="Missing (NaN)",
        )
    ax.set_ylabel(set_xlabel if not show_nan else f"{set_xlabel} / missing", fontsize=label_font)
    ax.tick_params(axis="y", labelsize=count_font, length=3)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    _strip_spines(ax, keep=("left", "bottom"))
    ax.set_xlim(-0.5, len(ordered_sets) - 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    if show_set_counts:
        peak = float(max(totals)) if len(totals) else 0.0
        offset = peak * 0.025 if peak else 0.1
        longest = max((len(f"{value:g}") for value in totals), default=1)
        size = count_font
        if len(ordered_sets) >= 6 and longest >= 5:
            size = max(6.5, count_font * 0.82)
        for index, (pos, value) in enumerate(zip(positions, totals)):
            bump = offset * (3.2 if index % 2 else 1.0)
            ax.text(
                pos,
                value + bump,
                f"{value:g}",
                ha="center",
                va="bottom",
                fontsize=size,
                color="#222222",
                clip_on=False,
                zorder=6,
            )
        ax.set_ylim(0, peak * 1.36 if peak else 1.0)


def _draw_intersection_count_gutter(
    ax: Axes,
    intersection_sizes: Sequence[int],
    count_font: float,
    facecolor: Color,
) -> None:
    """Numbers to the left of horizontal intersection bars (vertical layout)."""
    ax.set_facecolor(facecolor)
    ax.set_xlim(0, 1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    _strip_spines(ax)
    for ypos, value in enumerate(intersection_sizes):
        ax.text(
            0.5,
            ypos,
            f"{value:g}",
            ha="center",
            va="center",
            fontsize=count_font,
            color="#222222",
            clip_on=True,
            zorder=6,
        )


def _draw_intersection_bars(
    ax: Axes,
    intersection_sizes: Sequence[int],
    inter_colors: Sequence[Color],
    style: UpSetStyle,
    *,
    orientation: Orientation,
    intersection_ylabel: str,
    show_intersection_counts: bool,
    label_font: float,
    count_font: float,
) -> None:
    sizes = np.array(intersection_sizes, dtype=float)
    positions = np.arange(len(sizes))
    bar_kwargs = {
        "color": inter_colors,
        "edgecolor": style.bar_edgecolor,
        "linewidth": style.bar_linewidth,
        "zorder": 4,
    }
    if orientation == "horizontal":
        ax.bar(positions, sizes, width=style.intersection_bar_width, **bar_kwargs)
        ax.set_ylabel(intersection_ylabel, fontsize=label_font)
        ax.tick_params(axis="y", labelsize=count_font, length=3)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        _strip_spines(ax, keep=("left", "bottom"))
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=3))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
        if show_intersection_counts:
            _annotate_counts(
                ax,
                positions,
                sizes,
                along="x",
                fontsize=count_font,
                ha="center",
                va="bottom",
                at="end",
            )
            ax.margins(y=0.16)
        return

    ax.barh(positions, sizes, height=style.intersection_bar_width, **bar_kwargs)
    ax.set_xlabel(intersection_ylabel, fontsize=label_font)
    ax.tick_params(axis="x", labelsize=count_font, length=3)
    ax.tick_params(axis="y", left=False, labelleft=False)
    _strip_spines(ax, keep=("left", "bottom"))
    ax.set_ylim(-0.6, len(sizes) - 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=3))
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#E6E6E6", linewidth=0.6)


def _draw_matrix(
    ax: Axes,
    ordered_sets: Sequence[str],
    intersections: Sequence[frozenset[str]],
    set_colors: Sequence[Color],
    style: UpSetStyle,
    *,
    orientation: Orientation,
    with_lines: bool,
    highlight_keys: Sequence[frozenset[str]],
    marker_size: float,
    line_width: float,
) -> None:
    n_sets = len(ordered_sets)
    n_inter = len(intersections)
    set_index = {name: index for index, name in enumerate(ordered_sets)}
    highlight = set(highlight_keys)
    line_color = style.matrix_line_color or style.matrix_fill_color
    set_color_map = dict(zip(ordered_sets, set_colors))
    sets_are_rows = orientation == "horizontal"

    stripe_count = n_sets if sets_are_rows else n_inter
    for index in range(stripe_count):
        stripe = style.row_stripe_colors[index % 2]
        if sets_are_rows:
            ax.axhspan(index - 0.5, index + 0.5, color=stripe, zorder=0, lw=0)
        else:
            ax.axhspan(index - 0.5, index + 0.5, color=stripe, zorder=0, lw=0)

    empty_x: list[float] = []
    empty_y: list[float] = []
    filled_x: list[float] = []
    filled_y: list[float] = []
    filled_colors: list[Color] = []
    connectors: list[tuple[list[float], list[float], Color]] = []

    for inter_i, members in enumerate(intersections):
        member_indexes = [set_index[name] for name in members]
        if with_lines and len(member_indexes) >= 2:
            color = style.highlight_color if members in highlight else line_color
            lo, hi = min(member_indexes), max(member_indexes)
            if sets_are_rows:
                connectors.append(([inter_i, inter_i], [lo, hi], color))
            else:
                connectors.append(([lo, hi], [inter_i, inter_i], color))
        for name, set_i in set_index.items():
            filled = name in members
            x, y = (inter_i, set_i) if sets_are_rows else (set_i, inter_i)
            if not filled:
                empty_x.append(x)
                empty_y.append(y)
                continue
            if style.color_dots_by_set:
                face = set_color_map[name]
            elif members in highlight:
                face = style.highlight_color
            else:
                face = style.matrix_fill_color
            filled_x.append(x)
            filled_y.append(y)
            filled_colors.append(face)

    if empty_x:
        ax.scatter(
            empty_x,
            empty_y,
            s=marker_size,
            color=style.matrix_empty_color,
            alpha=style.matrix_empty_alpha,
            zorder=1,
            linewidths=0,
            clip_on=True,
        )
    for xs, ys, color in connectors:
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=line_width,
            solid_capstyle="round",
            zorder=3,
            clip_on=True,
        )
    if filled_x:
        ax.scatter(
            filled_x,
            filled_y,
            s=marker_size,
            c=filled_colors,
            zorder=5,
            linewidths=0,
            clip_on=True,
        )

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    _strip_spines(ax)
    if sets_are_rows:
        ax.set_xlim(-0.6, n_inter - 0.4)
        ax.set_ylim(-0.5, n_sets - 0.5)
    else:
        ax.set_xlim(-0.5, n_sets - 0.5)
        ax.set_ylim(-0.6, n_inter - 0.4)


def _draw_set_labels(
    ax: Axes,
    ordered_sets: Sequence[str],
    style: UpSetStyle,
    *,
    orientation: Orientation,
    n_intersections: int,
    label_font: float,
) -> None:
    n_sets = len(ordered_sets)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    _strip_spines(ax)
    if orientation == "horizontal":
        for ypos in range(n_sets):
            stripe = style.row_stripe_colors[ypos % 2]
            ax.axhspan(ypos - 0.5, ypos + 0.5, color=stripe, zorder=0, lw=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_sets - 0.5)
        for ypos, name in enumerate(ordered_sets):
            ax.text(
                0.98,
                ypos,
                str(name),
                ha="right",
                va="center",
                fontsize=label_font,
                color="#1A1A1A",
                clip_on=False,
            )
        return

    for xpos in range(n_sets):
        stripe = style.row_stripe_colors[xpos % 2]
        ax.axvspan(xpos - 0.5, xpos + 0.5, color=stripe, zorder=0, lw=0)
    ax.set_xlim(-0.5, n_sets - 0.5)
    ax.set_ylim(0, 1)
    for xpos, name in enumerate(ordered_sets):
        ax.text(
            xpos,
            0.5,
            str(name),
            rotation=90,
            ha="center",
            va="center",
            fontsize=label_font,
            color="#1A1A1A",
            clip_on=False,
        )


def _add_nan_legend(fig: Figure, style: UpSetStyle, orientation: Orientation) -> None:
    handles = [
        Patch(facecolor="#3A3A3A", edgecolor="none", label="Present"),
        Patch(facecolor=style.nan_color, edgecolor="none", label="Missing (NaN)"),
    ]
    loc = "upper left" if orientation == "horizontal" else "upper right"
    fig.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=style.count_fontsize or style.fontsize or 8,
        borderaxespad=0.8,
    )


def upset_plot(
    data: pd.DataFrame | Mapping[str, Collection[Any]],
    *,
    sets: Sequence[str] | None = None,
    orientation: Orientation = "horizontal",
    min_degree: int = 1,
    max_degree: int | None = None,
    min_intersection_size: int = 1,
    max_intersections: int | None = 50,
    sort_intersections: SortIntersections = "size",
    sort_sets: SortSets = "size",
    intersection_order: Sequence[Collection[str]] | None = None,
    set_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    fig: Figure | None = None,
    element_size: float | None = None,
    set_bar_width: float = 1.55,
    intersection_bar_height: float | None = None,
    intersection_ylabel: str = "Intersection Size",
    set_xlabel: str = "Set Size",
    show_intersection_counts: bool = True,
    show_set_counts: bool = False,
    show_nan: bool | None = None,
    nan_counts: Mapping[str, int] | None = None,
    invert_set_bars: bool = True,
    with_lines: bool = True,
    highlight: Sequence[Collection[str]] | None = None,
    intersection_color_mode: IntersectionColorMode = "uniform",
    title: str | None = None,
    ax_kwargs: Mapping[str, Any] | None = None,
    style: UpSetStyle | None = None,
    **style_overrides: Any,
) -> UpSetResult:
    """Draw an UpSet plot whose geometry follows the number of features.

    Parameters
    ----------
    data
        Boolean / 0-1 / NaN DataFrame (columns = sets, rows = elements) **or** a
        mapping ``{set_name: iterable of element ids}``. NaN means "unknown":
        it is not membership, and it is counted separately when shown.
    orientation
        ``horizontal`` (classic: intersections as columns) or ``vertical``
        (intersections as rows, set bars on top). Vertical is the better fit
        when you want many intersections and stacked Present / Missing bars.
    sets
        Optional subset / display order of features. Unknown at call time is
        fine: all columns / mapping keys are used when this is omitted.
    min_degree, max_degree
        Keep intersections whose number of member sets is in this range.
    min_intersection_size
        Drop exclusive intersections smaller than this count.
    max_intersections
        Cap the number of columns (horizontal) or rows (vertical) after sorting.
        ``None`` plots every remaining intersection.
    sort_intersections
        ``size`` (default), ``degree``, or ``input`` (preserve computed order).
    sort_sets
        ``size`` (largest at the bottom / left) or ``input``.
    intersection_order, set_order
        Explicit orders. When given they win over the sort flags.
    figsize
        ``(width, height)`` in inches. ``None`` sizes the figure from the
        number of sets and intersections.
    fig
        Existing figure to draw into. Its size is left alone.
    element_size
        Inches per matrix cell. ``None`` picks a size from how many cells exist.
    set_bar_width, intersection_bar_height
        Inches for the set-size and intersection-size bar panels along their
        value axis. In vertical mode, ``set_bar_width`` is the *height* of the
        top bars and ``intersection_bar_height`` is the *width* of the side bars.
    show_nan
        Stack Missing (NaN) on the set-size bars. ``None`` turns this on
        automatically when any set has NaNs.
    nan_counts
        Optional override ``{set_name: n_missing}``. If omitted, counts come
        from NaNs in ``data``.
    highlight
        Intersections (as iterables of set names) drawn with ``highlight_color``.
    intersection_color_mode
        ``uniform``, ``degree`` (colormap by number of member sets), or
        ``exclusive-set`` (singleton columns inherit that set's color).
    style, **style_overrides
        :class:`UpSetStyle` fields. Overrides are applied on top of ``style``.

    Returns
    -------
    UpSetResult
        Figure, axes, plotted set / intersection values, and per-set NaN counts.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    unknown_style = sorted(
        key for key in style_overrides if key not in UpSetStyle.__dataclass_fields__
    )
    if unknown_style:
        raise TypeError(f"Unknown style field: {unknown_style}")
    style_obj = replace(style or UpSetStyle(), **style_overrides)

    frame, detected_nan = _as_boolean_frame(data, sets)
    if frame.shape[1] == 0:
        raise ValueError("No sets to plot.")

    set_sizes = frame.sum(axis=0).astype(int).to_dict()
    ordered_sets = (
        list(set_order)
        if set_order is not None
        else _sort_sets(frame.columns, set_sizes, sort_sets)
    )
    unknown_sets = [name for name in ordered_sets if name not in frame.columns]
    if unknown_sets:
        raise KeyError(f"set_order contains unknown sets: {unknown_sets}")

    resolved_nan = {
        name: int(nan_counts[name]) if nan_counts is not None and name in nan_counts else int(detected_nan.get(name, 0))
        for name in ordered_sets
    }
    if nan_counts is not None:
        extra = [name for name in nan_counts if name not in frame.columns]
        if extra:
            raise KeyError(f"nan_counts contains unknown sets: {extra}")

    show_nan_bars = (
        any(value > 0 for value in resolved_nan.values())
        if show_nan is None
        else bool(show_nan)
    )

    records = _exclusive_intersections(frame)
    records = [
        (members, count)
        for members, count in records
        if count >= min_intersection_size
        and len(members) >= min_degree
        and (max_degree is None or len(members) <= max_degree)
        and members.issubset(ordered_sets)
    ]

    truncated = False
    if intersection_order is not None:
        wanted = [frozenset(item) for item in intersection_order]
        lookup = {members: count for members, count in records}
        missing = [tuple(sorted(item)) for item in wanted if item not in lookup]
        if missing:
            raise KeyError(
                f"intersection_order contains combinations with no rows: {missing}"
            )
        records = [(members, lookup[members]) for members in wanted]
    else:
        records = _sort_intersections(records, sort_intersections, ordered_sets)
        if max_intersections is not None and len(records) > max_intersections:
            records = records[:max_intersections]
            truncated = True

    if not records:
        raise ValueError(
            "No intersections remain after filtering. Relax min_degree / min_intersection_size."
        )

    intersections = [members for members, _ in records]
    intersection_sizes = [count for _, count in records]
    n_sets = len(ordered_sets)
    n_inter = len(intersections)

    if element_size is None:
        element_size = _clamp(0.52 - 0.004 * max(n_sets, n_inter), 0.28, 0.55)
    if intersection_bar_height is None:
        intersection_bar_height = _clamp(1.45 + 0.12 * n_sets, 1.5, 2.6)

    base_font = (
        style_obj.fontsize
        if style_obj.fontsize is not None
        else _clamp(element_size * 72 * 0.32, 7.0, 12.0)
    )
    label_font = (
        style_obj.label_fontsize if style_obj.label_fontsize is not None else base_font
    )
    count_font = (
        style_obj.count_fontsize
        if style_obj.count_fontsize is not None
        else _clamp(base_font * 0.9, 6.0, 10.0)
    )
    title_font = (
        style_obj.title_fontsize
        if style_obj.title_fontsize is not None
        else base_font + 2
    )
    longest_label = max((len(str(name)) for name in ordered_sets), default=1)

    existing_size = None if fig is None else tuple(fig.get_size_inches())
    requested_size = figsize or existing_size
    geometry = _panel_geometry(
        n_sets=n_sets,
        n_intersections=n_inter,
        longest_label=longest_label,
        element_size=element_size,
        set_bar_width=set_bar_width,
        intersection_bar_height=intersection_bar_height,
        fontsize=label_font,
        show_intersection_counts=show_intersection_counts,
        show_set_counts=show_set_counts,
        has_title=title is not None,
        show_nan=show_nan_bars,
        orientation=orientation,
        fig_width=None if requested_size is None else requested_size[0],
        fig_height=None if requested_size is None else requested_size[1],
        count_span=(
            _count_label_span(intersection_sizes, count_font)
            if show_intersection_counts
            else 0.12
        ),
    )

    if fig is None:
        fig = plt.figure(
            figsize=(geometry["fig_width"], geometry["fig_height"]),
            facecolor=style_obj.facecolor,
        )
    else:
        fig.patch.set_facecolor(style_obj.facecolor)

    if orientation == "horizontal":
        ax_inter = fig.add_axes(geometry["inter"])
        ax_sets = fig.add_axes(geometry["sets"])
        ax_labels = fig.add_axes(geometry["labels"], sharey=ax_sets)
        ax_matrix = fig.add_axes(geometry["matrix"], sharex=ax_inter, sharey=ax_sets)
        ax_counts = None
    else:
        ax_sets = fig.add_axes(geometry["sets"])
        ax_labels = fig.add_axes(geometry["labels"], sharex=ax_sets)
        ax_matrix = fig.add_axes(geometry["matrix"], sharex=ax_sets)
        ax_inter = fig.add_axes(geometry["inter"], sharey=ax_matrix)
        ax_counts = (
            fig.add_axes(geometry["counts"], sharey=ax_matrix)
            if geometry.get("counts") is not None and show_intersection_counts
            else None
        )

    for ax in (ax_inter, ax_sets, ax_labels, ax_matrix):
        ax.set_facecolor(style_obj.facecolor)
        if ax_kwargs:
            ax.set(**ax_kwargs)

    set_colors = _colors_for_sets(
        style_obj.set_color, list(frame.columns), ordered_sets
    )
    highlight_keys = [frozenset(item) for item in highlight] if highlight else []
    inter_colors = _intersection_colors(
        intersections,
        intersection_sizes,
        ordered_sets,
        set_colors,
        style_obj,
        intersection_color_mode,
        highlight_keys,
    )
    diameter_pt = element_size * 72.0 * 0.48
    marker_size = (
        style_obj.dot_size if style_obj.dot_size is not None else diameter_pt**2
    )
    line_width = (
        style_obj.line_width
        if style_obj.line_width is not None
        else max(1.4, element_size * 3.2)
    )

    _draw_set_labels(
        ax_labels,
        ordered_sets,
        style_obj,
        orientation=orientation,
        n_intersections=n_inter,
        label_font=label_font,
    )
    _draw_matrix(
        ax_matrix,
        ordered_sets,
        intersections,
        set_colors,
        style_obj,
        orientation=orientation,
        with_lines=with_lines,
        highlight_keys=highlight_keys,
        marker_size=marker_size,
        line_width=line_width,
    )
    _draw_set_bars(
        ax_sets,
        ordered_sets,
        set_sizes,
        resolved_nan,
        set_colors,
        style_obj,
        orientation=orientation,
        invert_set_bars=invert_set_bars if orientation == "horizontal" else False,
        show_nan=show_nan_bars,
        show_set_counts=show_set_counts,
        set_xlabel=set_xlabel,
        label_font=label_font,
        count_font=count_font,
    )
    _draw_intersection_bars(
        ax_inter,
        intersection_sizes,
        inter_colors,
        style_obj,
        orientation=orientation,
        intersection_ylabel=intersection_ylabel,
        show_intersection_counts=show_intersection_counts,
        label_font=label_font,
        count_font=count_font,
    )

    if orientation == "vertical":
        ax_matrix.invert_yaxis()
        if ax_counts is not None:
            _draw_intersection_count_gutter(
                ax_counts,
                intersection_sizes,
                count_font,
                style_obj.facecolor,
            )

    if show_nan_bars:
        _add_nan_legend(fig, style_obj, orientation)

    if title:
        # Keep title in the reserved top band so it does not sit on panel art.
        fig.suptitle(title, fontsize=title_font, y=0.98, va="top")

    # Detach from pyplot's active-figure list. The returned Figure stays fully
    # usable for savefig / display, but Jupyter will not also auto-render it
    # at cell end — so callers that ``display(result.fig)`` see it once.
    plt.close(fig)

    return UpSetResult(
        fig=fig,
        ax_intersections=ax_inter,
        ax_sets=ax_sets,
        ax_matrix=ax_matrix,
        ax_labels=ax_labels,
        set_names=tuple(ordered_sets),
        set_sizes=tuple(set_sizes[name] for name in ordered_sets),
        nan_counts=tuple(resolved_nan[name] for name in ordered_sets),
        intersections=tuple(intersections),
        intersection_sizes=tuple(intersection_sizes),
        truncated=truncated,
        orientation=orientation,
    )

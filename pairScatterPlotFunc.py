"""Pairwise 2-D scatterplots with per-class marginal densities.

For every unordered pair of selected features the figure can draw:

* a **scatter** of objects, coloured / shaped by class
* **KDE density** of each class along the x-feature (top) and y-feature (right)

Visual knobs (axis colour / width, major & minor ticks, grid, square panels,
fonts, legend / title on-off, point & density transparency) live on
:class:`PairScatterStyle` and can be overridden as kwargs to
:func:`pair_scatter_plot`.

Layout scales with the number of pairs (figure size, fonts, spacing) unless
overridden — same adaptivity idea as ``upSetPlotFunc`` / ``raincloudPlotFunc``.

Do not call ``tight_layout()`` after ``pair_scatter_plot``; panel positions
are already computed. Leave the returned figure open (or call ``plt.show()``
once) — do not combine ``display(fig)`` with ``plt.show()`` or the plot
appears twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from scipy.stats import gaussian_kde

Color: TypeAlias = str | tuple[float, ...]

# Same palette as raincloudPlotFunc / upSetPlotFunc / test.py
_PALETTE: tuple[str, ...] = (
    "#4C78A8",
    "#E45756",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#72B7B2",
    "#ECAE3E",
    "#9D755D",
)

_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


@dataclass(frozen=True)
class PairScatterResult:
    """Drawn figure plus the pairs / classes that were plotted."""

    fig: Figure
    axes: tuple[Axes, ...]
    features: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    groups: tuple[str, ...]


@dataclass
class PairScatterStyle:
    """Visual knobs. Override any field when calling :func:`pair_scatter_plot`."""

    palette: Sequence[Color] = _PALETTE
    markers: Sequence[str] = _MARKERS
    class_colors: Mapping[str, Color] | None = None
    class_markers: Mapping[str, str] | None = None

    # Scatter
    point_size: float | None = None
    point_alpha: float = 0.55
    point_edgecolor: Color | None = "none"
    point_linewidth: float = 0.0

    # Marginal densities
    density_alpha: float = 0.35
    density_linewidth: float = 1.4
    density_fill: bool = True
    density_n: int = 200
    density_bw_method: float | str | None = None
    marginal_ratio: float | None = None
    marginal_pad: float = 0.02

    # Axes / spines
    axis_color: Color = "#333333"
    axis_linewidth: float = 1.0
    spine_top: bool = False
    spine_right: bool = False

    # Ticks: prefer step when set, else count (MaxNLocator)
    major_tick_count: int | None = 5
    minor_tick_count: int | None = 2
    major_tick_step: float | None = None
    minor_tick_step: float | None = None
    tick_length_major: float = 4.0
    tick_length_minor: float = 2.0
    tick_width: float = 0.8

    # Grid
    show_grid: bool = True
    grid_which: str = "major"
    grid_color: Color = "#CCCCCC"
    grid_alpha: float = 0.7
    grid_linewidth: float = 0.7
    grid_linestyle: str = "-"

    # Panel geometry (None = adaptive from n_pairs)
    square: bool = True
    panel_size: float | None = None
    wspace: float | None = None
    hspace: float | None = None

    # Fonts (None size = adaptive)
    fontfamily: str = "sans-serif"
    title_fontsize: float | None = None
    title_fontfamily: str | None = None
    label_fontsize: float | None = None
    label_fontfamily: str | None = None
    tick_fontsize: float | None = None
    tick_fontfamily: str | None = None
    legend_fontsize: float | None = None
    legend_fontfamily: str | None = None

    facecolor: Color = "white"


def _as_str_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: "NA" if pd.isna(v) else str(v))


def _colors_for(
    groups: Sequence[str],
    style: PairScatterStyle,
) -> dict[str, Color]:
    if style.class_colors is not None:
        return {
            g: style.class_colors.get(g, _PALETTE[i % len(_PALETTE)])
            for i, g in enumerate(groups)
        }
    palette = list(style.palette) or list(_PALETTE)
    return {g: palette[i % len(palette)] for i, g in enumerate(groups)}


def _markers_for(
    groups: Sequence[str],
    style: PairScatterStyle,
) -> dict[str, str]:
    if style.class_markers is not None:
        return {
            g: style.class_markers.get(g, _MARKERS[i % len(_MARKERS)])
            for i, g in enumerate(groups)
        }
    markers = list(style.markers) or list(_MARKERS)
    return {g: markers[i % len(markers)] for i, g in enumerate(groups)}


def _adaptive_layout(n_pairs: int, style: PairScatterStyle) -> dict[str, float]:
    """Scale panel / font sizes with the number of pair panels."""
    # More panels → slightly smaller cells and fonts so labels stay readable.
    scale = float(np.clip(1.15 - 0.06 * max(n_pairs - 1, 0), 0.72, 1.15))
    panel = style.panel_size if style.panel_size is not None else 3.4 * scale
    base = 9.5 * scale
    return {
        "panel_size": panel,
        "point_size": style.point_size
        if style.point_size is not None
        else 16.0 * scale,
        "title_fontsize": style.title_fontsize
        if style.title_fontsize is not None
        else max(11.0, 13.0 * scale),
        "label_fontsize": style.label_fontsize
        if style.label_fontsize is not None
        else max(8.0, base),
        "tick_fontsize": style.tick_fontsize
        if style.tick_fontsize is not None
        else max(7.0, base - 1.5),
        "legend_fontsize": style.legend_fontsize
        if style.legend_fontsize is not None
        else max(8.0, base),
        "wspace": style.wspace
        if style.wspace is not None
        else 0.42 + 0.04 * min(n_pairs, 6),
        "hspace": style.hspace
        if style.hspace is not None
        else 0.42 + 0.04 * min(n_pairs, 6),
        "marginal_ratio": (
            style.marginal_ratio if style.marginal_ratio is not None else 0.28
        ),
    }


def _configure_ticks(ax: Axes, style: PairScatterStyle, *, which: str = "both") -> None:
    axes = []
    if which in ("both", "x"):
        axes.append(ax.xaxis)
    if which in ("both", "y"):
        axes.append(ax.yaxis)

    for axis in axes:
        if style.major_tick_step is not None:
            axis.set_major_locator(MultipleLocator(style.major_tick_step))
        elif style.major_tick_count is not None:
            axis.set_major_locator(MaxNLocator(nbins=style.major_tick_count))

        if style.minor_tick_step is not None:
            axis.set_minor_locator(MultipleLocator(style.minor_tick_step))
        elif style.minor_tick_count is not None and style.minor_tick_count > 0:
            axis.set_minor_locator(AutoMinorLocator(style.minor_tick_count + 1))


def _style_axes(
    ax: Axes,
    style: PairScatterStyle,
    *,
    tick_fontsize: float,
    hide_x: bool = False,
    hide_y: bool = False,
) -> None:
    for spine_name, spine in ax.spines.items():
        spine.set_color(style.axis_color)
        spine.set_linewidth(style.axis_linewidth)
        if spine_name == "top":
            spine.set_visible(style.spine_top)
        elif spine_name == "right":
            spine.set_visible(style.spine_right)

    ax.tick_params(
        which="major",
        colors=style.axis_color,
        length=style.tick_length_major,
        width=style.tick_width,
        labelsize=tick_fontsize,
        labelcolor=style.axis_color,
    )
    ax.tick_params(
        which="minor",
        colors=style.axis_color,
        length=style.tick_length_minor,
        width=style.tick_width * 0.8,
    )
    tick_family = style.tick_fontfamily or style.fontfamily
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(tick_family)

    if hide_x:
        ax.tick_params(axis="x", labelbottom=False)
    if hide_y:
        ax.tick_params(axis="y", labelleft=False)

    if style.show_grid:
        ax.grid(
            True,
            which=style.grid_which,
            color=style.grid_color,
            alpha=style.grid_alpha,
            linewidth=style.grid_linewidth,
            linestyle=style.grid_linestyle,
        )
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _kde_curve(
    values: np.ndarray,
    grid: np.ndarray,
    *,
    bw_method: float | str | None,
) -> np.ndarray | None:
    clean = values[np.isfinite(values)]
    if clean.size < 2 or np.unique(clean).size < 2:
        return None
    try:
        kde = gaussian_kde(clean, bw_method=bw_method)
    except Exception:  # noqa: BLE001
        return None
    dens = kde(grid)
    peak = float(np.nanmax(dens))
    if not np.isfinite(peak) or peak <= 0:
        return None
    return dens / peak


def _draw_joint_panel(
    fig: Figure,
    outer: Any,
    *,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    groups: Sequence[str],
    colors: Mapping[str, Color],
    feature_colors: Mapping[str, Color],
    markers: Mapping[str, str],
    style: PairScatterStyle,
    layout: Mapping[str, float],
    show_scatter: bool,
    show_density: bool,
    panel_title: str | None = None,
    panel_title_fontsize: float | None = None,
) -> Axes:
    """Scatter + top/right class densities inside one outer GridSpec cell."""
    ratio = layout["marginal_ratio"] if show_density else 0.001
    pad = style.marginal_pad
    if show_density and not show_scatter:
        # Density-focused: enlarge the marginal strips.
        ratio = max(ratio, 0.42)

    title_fs = (
        panel_title_fontsize
        if panel_title_fontsize is not None
        else max(10.0, float(layout["label_fontsize"]) + 1.5)
    )
    if panel_title:
        inner = outer.subgridspec(
            3,
            2,
            width_ratios=[1.0, ratio],
            height_ratios=[0.32, ratio, 1.0],
            wspace=pad,
            hspace=max(pad, 0.05),
        )
        ax_title = fig.add_subplot(inner[0, :])
        ax_title.set_axis_off()
        ax_title.text(
            0.5,
            0.5,
            panel_title,
            transform=ax_title.transAxes,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color=style.axis_color,
            clip_on=False,
        )
        ax_dens_x = fig.add_subplot(inner[1, 0])
        ax_scatter = fig.add_subplot(inner[2, 0], sharex=ax_dens_x)
        ax_dens_y = fig.add_subplot(inner[2, 1], sharey=ax_scatter)
        ax_corner = fig.add_subplot(inner[1, 1])
    else:
        inner = outer.subgridspec(
            2,
            2,
            width_ratios=[1.0, ratio],
            height_ratios=[ratio, 1.0],
            wspace=pad,
            hspace=pad,
        )
        ax_dens_x = fig.add_subplot(inner[0, 0])
        ax_scatter = fig.add_subplot(inner[1, 0], sharex=ax_dens_x)
        ax_dens_y = fig.add_subplot(inner[1, 1], sharey=ax_scatter)
        ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.set_axis_off()

    x_all = data[x_col].to_numpy(dtype=float)
    y_all = data[y_col].to_numpy(dtype=float)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if finite.any():
        x_min, x_max = float(np.nanmin(x_all[finite])), float(np.nanmax(x_all[finite]))
        y_min, y_max = float(np.nanmin(y_all[finite])), float(np.nanmax(y_all[finite]))
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    x_pad = 0.05 * (x_max - x_min or 1.0)
    y_pad = 0.05 * (y_max - y_min or 1.0)
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    x_grid = np.linspace(x_lim[0], x_lim[1], style.density_n)
    y_grid = np.linspace(y_lim[0], y_lim[1], style.density_n)

    point_size = layout["point_size"]
    tick_fs = layout["tick_fontsize"]
    label_fs = layout["label_fontsize"]
    label_family = style.label_fontfamily or style.fontfamily

    for g in groups:
        mask = (data[group_col] == g).to_numpy() & finite
        class_color = colors[g]
        marker = markers[g]
        # Feature colours (same A/B/C/D map as raincloud): x vs y differ on each panel.
        x_feat_color = feature_colors.get(x_col, class_color)
        y_feat_color = feature_colors.get(y_col, class_color)
        # Multi-class: face = class colour, edge = still hints the two features.
        # Single class: face = x-feature, edge = y-feature so A and B both show.
        if len(groups) > 1:
            face = class_color
            edge = (
                style.point_edgecolor
                if style.point_edgecolor not in (None, "none")
                else "none"
            )
            lw = style.point_linewidth
        else:
            face = x_feat_color
            edge = y_feat_color
            lw = max(style.point_linewidth, 0.45)

        if show_scatter and mask.any():
            ax_scatter.scatter(
                x_all[mask],
                y_all[mask],
                s=point_size,
                color=face,
                marker=marker,
                alpha=style.point_alpha,
                edgecolors=edge,
                linewidths=lw,
                label=g,
                zorder=3,
            )

        if show_density:
            dens_x = _kde_curve(x_all[mask], x_grid, bw_method=style.density_bw_method)
            if dens_x is not None:
                dens_x_color = class_color if len(groups) > 1 else x_feat_color
                if style.density_fill:
                    ax_dens_x.fill_between(
                        x_grid,
                        dens_x,
                        color=dens_x_color,
                        alpha=style.density_alpha,
                        linewidth=0,
                    )
                ax_dens_x.plot(
                    x_grid,
                    dens_x,
                    color=dens_x_color,
                    lw=style.density_linewidth,
                    alpha=min(1.0, style.density_alpha + 0.35),
                )

            dens_y = _kde_curve(y_all[mask], y_grid, bw_method=style.density_bw_method)
            if dens_y is not None:
                dens_y_color = class_color if len(groups) > 1 else y_feat_color
                if style.density_fill:
                    ax_dens_y.fill_betweenx(
                        y_grid,
                        dens_y,
                        color=dens_y_color,
                        alpha=style.density_alpha,
                        linewidth=0,
                    )
                ax_dens_y.plot(
                    dens_y,
                    y_grid,
                    color=dens_y_color,
                    lw=style.density_linewidth,
                    alpha=min(1.0, style.density_alpha + 0.35),
                )

    ax_scatter.set_xlim(x_lim)
    ax_scatter.set_ylim(y_lim)
    if style.square:
        ax_scatter.set_box_aspect(1)

    ax_scatter.set_xlabel(
        x_col,
        fontsize=label_fs,
        fontfamily=label_family,
        color=feature_colors.get(x_col, style.axis_color),
        fontweight="bold",
    )
    ax_scatter.set_ylabel(
        y_col,
        fontsize=label_fs,
        fontfamily=label_family,
        color=feature_colors.get(y_col, style.axis_color),
        fontweight="bold",
    )

    _configure_ticks(ax_scatter, style, which="both")
    _style_axes(ax_scatter, style, tick_fontsize=tick_fs)
    if not show_scatter:
        # Keep axes for labels / limits; hide empty scatter frame clutter.
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        for spine in ax_scatter.spines.values():
            spine.set_visible(False)
        ax_scatter.grid(False)
        ax_scatter.set_xlabel(
            f"{x_col}  |  {y_col}", fontsize=label_fs, fontfamily=label_family
        )
        ax_scatter.set_ylabel("")

    if show_density:
        ax_dens_x.set_xlim(x_lim)
        ax_dens_x.set_ylim(0.0, 1.15)
        ax_dens_x.set_yticks([])
        _configure_ticks(ax_dens_x, style, which="x")
        _style_axes(ax_dens_x, style, tick_fontsize=tick_fs, hide_x=True, hide_y=True)
        ax_dens_x.spines["left"].set_visible(False)
        ax_dens_x.grid(False)

        ax_dens_y.set_ylim(y_lim)
        ax_dens_y.set_xlim(0.0, 1.15)
        ax_dens_y.set_xticks([])
        _configure_ticks(ax_dens_y, style, which="y")
        _style_axes(ax_dens_y, style, tick_fontsize=tick_fs, hide_x=True, hide_y=True)
        ax_dens_y.spines["bottom"].set_visible(False)
        ax_dens_y.grid(False)
    else:
        ax_dens_x.set_axis_off()
        ax_dens_y.set_axis_off()

    for ax in (ax_scatter, ax_dens_x, ax_dens_y):
        ax.set_facecolor(style.facecolor)

    return ax_scatter


def pair_scatter_plot(
    data: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    group: str = "class",
    title: str | None = None,
    show_title: bool = True,
    show_legend: bool = False,
    show_scatter: bool = True,
    show_density: bool = True,
    ncols: int | None = None,
    style: PairScatterStyle | None = None,
    **style_overrides: Any,
) -> PairScatterResult:
    """Scatter every feature pair with optional per-class marginal densities.

    Parameters
    ----------
    data
        DataFrame with numeric feature columns and a class column.
    columns
        Features to pair. Default: every numeric column except ``group``.
    group
        Class / grouping column. Point colour & marker, and density colour,
        vary by group level.
    title, show_title
        Figure title; ``show_title=False`` suppresses it even if ``title`` is set.
    show_legend
        Draw a class legend (colour + marker). Default off (more plot space),
        matching ``raincloud_plot``.
    show_scatter, show_density
        Toggle the two lab deliverables independently.
    ncols
        Panel columns in the pair grid. Default: ``ceil(sqrt(n_pairs))``.
    style, **style_overrides
        :class:`PairScatterStyle` fields (axis colour/width, tick count/step,
        grid, square panels, fonts, alphas, …).

    Returns
    -------
    PairScatterResult
    """
    if data is None or data.empty:
        raise ValueError("data must be a non-empty DataFrame")
    if group not in data.columns:
        raise KeyError(f"group column {group!r} not in data")
    if not show_scatter and not show_density:
        raise ValueError("at least one of show_scatter / show_density must be True")

    unknown = sorted(
        k for k in style_overrides if k not in PairScatterStyle.__dataclass_fields__
    )
    if unknown:
        raise TypeError(f"unknown style fields: {unknown}")
    style_obj = replace(style or PairScatterStyle(), **style_overrides)

    if columns is None:
        features = [
            c
            for c in data.columns
            if c != group and pd.api.types.is_numeric_dtype(data[c])
        ]
    else:
        features = list(columns)
    if len(features) < 2:
        raise ValueError("need at least two feature columns")
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise KeyError(f"columns not in data: {missing}")

    work = data[list(features) + [group]].copy()
    work[group] = _as_str_series(work[group])
    groups = tuple(dict.fromkeys(work[group].tolist()))
    colors = _colors_for(groups, style_obj)
    markers = _markers_for(groups, style_obj)
    # Same A/B/C/D colour map raincloud uses when features are categories.
    palette = list(style_obj.palette) if style_obj.palette else list(_PALETTE)
    feature_colors = {
        name: palette[i % len(palette)] for i, name in enumerate(features)
    }

    pairs = tuple(combinations(features, 2))
    n_pairs = len(pairs)
    layout = _adaptive_layout(n_pairs, style_obj)

    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n_pairs)))
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_pairs / ncols))

    has_title = bool(show_title and title)
    # Extra top margin so the title never sits on panel labels / legend.
    top = 0.84 if has_title else (0.94 if show_legend else 0.96)
    right = 0.86 if show_legend else 0.98

    fig_w = layout["panel_size"] * ncols
    fig_h = layout["panel_size"] * nrows + (0.55 if has_title or show_legend else 0.1)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=style_obj.facecolor)
    outer = fig.add_gridspec(
        nrows,
        ncols,
        left=0.07,
        right=right,
        bottom=0.07,
        top=top,
        wspace=layout["wspace"],
        hspace=layout["hspace"],
    )

    scatter_axes: list[Axes] = []
    for idx, (x_col, y_col) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        ax = _draw_joint_panel(
            fig,
            outer[r, c],
            data=work,
            x_col=x_col,
            y_col=y_col,
            group_col=group,
            groups=groups,
            colors=colors,
            feature_colors=feature_colors,
            markers=markers,
            style=style_obj,
            layout=layout,
            show_scatter=show_scatter,
            show_density=show_density,
        )
        scatter_axes.append(ax)

    for idx in range(n_pairs, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_blank = fig.add_subplot(outer[r, c])
        ax_blank.set_axis_off()

    if show_legend and groups:
        legend_family = style_obj.legend_fontfamily or style_obj.fontfamily
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=markers[g],
                markersize=max(6.0, np.sqrt(layout["point_size"])),
                markerfacecolor=colors[g],
                markeredgecolor="none",
                alpha=min(1.0, style_obj.point_alpha + 0.25),
                label=str(g),
            )
            for g in groups
        ]
        fig.legend(
            handles=handles,
            title=group,
            loc="upper right",
            frameon=False,
            fontsize=layout["legend_fontsize"],
            title_fontsize=layout["legend_fontsize"],
            prop={"family": legend_family, "size": layout["legend_fontsize"]},
            borderaxespad=0.3,
        )

    if has_title:
        title_family = style_obj.title_fontfamily or style_obj.fontfamily
        fig.suptitle(
            title,
            fontsize=layout["title_fontsize"],
            fontfamily=title_family,
            y=0.98,
            color=style_obj.axis_color,
        )

    return PairScatterResult(
        fig=fig,
        axes=tuple(scatter_axes),
        features=tuple(features),
        pairs=pairs,
        groups=groups,
    )


def pair_scatter_dataset_panel(
    datasets: Mapping[str, pd.DataFrame],
    x: str,
    y: str,
    *,
    group: str = "class",
    title: str | None = None,
    show_title: bool = True,
    show_legend: bool = False,
    show_scatter: bool = True,
    show_density: bool = True,
    ncols: int | None = None,
    style: PairScatterStyle | None = None,
    **style_overrides: Any,
) -> PairScatterResult:
    """m×n panel of joint scatter+density plots — one dataset per cell.

    Same visual language as :func:`pair_scatter_plot` (feature-coloured
    points/densities, class markers when several classes). Used by lab
    steps that compare many clones on a single feature pair.
    """
    if not datasets:
        raise ValueError("datasets must be a non-empty mapping")
    unknown = sorted(
        k for k in style_overrides if k not in PairScatterStyle.__dataclass_fields__
    )
    if unknown:
        raise TypeError(f"unknown style fields: {unknown}")
    style_obj = replace(style or PairScatterStyle(), **style_overrides)

    names = list(datasets.keys())
    n = len(names)
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n)))
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    layout = _adaptive_layout(max(n, 4), style_obj)

    # Shared class / feature colour maps across all panels.
    all_groups: list[str] = []
    for frame in datasets.values():
        if group not in frame.columns:
            raise KeyError(f"group column {group!r} not in one of the datasets")
        for g in _as_str_series(frame[group]).tolist():
            if g not in all_groups:
                all_groups.append(g)
    groups = tuple(all_groups)
    colors = _colors_for(groups, style_obj)
    markers = _markers_for(groups, style_obj)
    palette = list(style_obj.palette) if style_obj.palette else list(_PALETTE)
    # Keep A→palette[0], B→[1], … so colours match raincloud / [1.4].
    known = ("A", "B", "C", "D", "E", "F", "G", "H")
    feature_colors: dict[str, Color] = {}
    for i, name in enumerate((x, y)):
        feature_colors[name] = (
            palette[known.index(name) % len(palette)]
            if name in known
            else palette[i % len(palette)]
        )

    has_title = bool(show_title and title)
    # Extra room above panels so bold dataset titles do not collide with densities.
    top = 0.80 if has_title else 0.90
    right = 0.86 if show_legend else 0.98
    fig_w = layout["panel_size"] * ncols * 1.05
    fig_h = layout["panel_size"] * nrows * 1.12 + (
        0.85 if has_title or show_legend else 0.30
    )
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=style_obj.facecolor)
    outer = fig.add_gridspec(
        nrows,
        ncols,
        left=0.06,
        right=right,
        bottom=0.06,
        top=top,
        wspace=max(layout["wspace"], 0.60),
        hspace=max(layout["hspace"] + 0.28, 0.75),
    )

    scatter_axes: list[Axes] = []
    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        frame = datasets[name]
        work = frame[[x, y, group]].copy()
        work[group] = _as_str_series(work[group])
        panel_groups = tuple(dict.fromkeys(work[group].tolist()))
        ax = _draw_joint_panel(
            fig,
            outer[r, c],
            data=work,
            x_col=x,
            y_col=y,
            group_col=group,
            groups=panel_groups,
            colors=colors,
            feature_colors=feature_colors,
            markers=markers,
            style=style_obj,
            layout=layout,
            show_scatter=show_scatter,
            show_density=show_density,
            panel_title=str(name),
            panel_title_fontsize=max(12.5, layout["label_fontsize"] + 3.5),
        )
        scatter_axes.append(ax)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax_blank = fig.add_subplot(outer[r, c])
        ax_blank.set_axis_off()

    if show_legend and groups:
        legend_family = style_obj.legend_fontfamily or style_obj.fontfamily
        handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=markers[g],
                markersize=max(6.0, np.sqrt(layout["point_size"])),
                markerfacecolor=colors[g],
                markeredgecolor="none",
                alpha=min(1.0, style_obj.point_alpha + 0.25),
                label=str(g),
            )
            for g in groups
        ]
        fig.legend(
            handles=handles,
            title=group,
            loc="upper right",
            frameon=False,
            fontsize=layout["legend_fontsize"],
            title_fontsize=layout["legend_fontsize"],
            prop={"family": legend_family, "size": layout["legend_fontsize"]},
            borderaxespad=0.3,
        )

    if has_title:
        title_family = style_obj.title_fontfamily or style_obj.fontfamily
        fig.suptitle(
            title,
            fontsize=layout["title_fontsize"],
            fontfamily=title_family,
            y=0.98,
            color=style_obj.axis_color,
        )

    return PairScatterResult(
        fig=fig,
        axes=tuple(scatter_axes),
        features=(x, y),
        pairs=((x, y),),
        groups=groups,
    )

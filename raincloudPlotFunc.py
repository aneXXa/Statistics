"""Customizable raincloud plots built on matplotlib.

A raincloud is three 1-D views of the same sample, packed as:

* **rain** — jittered strip of the raw points
* **box** — replaceable sample estimates (default: median, IQR, range)
* **cloud** — half-violin (KDE)

Orientation
-----------
* ``vertical`` — numeric values on **y** (classic raincloud / reference figure).
  Parts sit left → right: rain | box | cloud.
* ``horizontal`` — numeric values on **x**. Parts sit bottom → top:
  rain | box | cloud.

When ``group`` is omitted, each requested feature becomes a category on one
axes (like genotypes on a single raincloud). When ``group`` is set, one
axes is drawn per feature and group levels are the categories.

The box is not a Tukey ``boxplot``. Its three layers are named estimators
(or callables) so a lab can swap median/IQR/range for mean/std/… without
redrawing the rest.

Do not call ``tight_layout()`` after ``raincloud_plot``; panel positions
are already computed. ``plt.show()`` is fine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

Color: TypeAlias = str | tuple[float, ...]
Orientation: TypeAlias = Literal["horizontal", "vertical"]
CenterName: TypeAlias = Literal["median", "mean", "mode"]
BoxName: TypeAlias = Literal["iqr", "std", "mad", "q10_q90"]
WhiskerName: TypeAlias = Literal["range", "tukey", "p05_p95", "std2"]
CenterSpec: TypeAlias = CenterName | Callable[[np.ndarray], float]
IntervalSpec: TypeAlias = (
    BoxName | WhiskerName | Callable[[np.ndarray], tuple[float, float]]
)

# Same palette as upSetPlotFunc / test.py
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

# Offsets of rain / box / cloud relative to a category centre.
_RAIN_OFFSET = -0.24
_BOX_OFFSET = 0.0
_CLOUD_OFFSET = 0.24
_CATEGORY_PITCH = 1.05


@dataclass(frozen=True)
class BoxLayerValues:
    """Resolved numbers actually drawn for one (feature, group) box."""

    center: float
    box_low: float
    box_high: float
    whisker_low: float
    whisker_high: float
    n: int
    center_name: str
    box_name: str
    whisker_name: str


@dataclass(frozen=True)
class RaincloudResult:
    """Drawn figure plus the estimates that were plotted."""

    fig: Figure
    axes: tuple[Axes, ...]
    features: tuple[str, ...]
    groups: tuple[str, ...]
    categories: tuple[str, ...]
    estimates: dict[tuple[str, str], BoxLayerValues]
    orientation: Orientation


@dataclass
class BoxEstimates:
    """What the three box layers represent.

    Each field is a built-in name or a callable on a 1-D ``ndarray``
    (NaNs already dropped). Callables for ``box`` / ``whiskers`` return
    ``(low, high)``.
    """

    center: CenterSpec = "median"
    box: IntervalSpec = "iqr"
    whiskers: IntervalSpec = "range"


@dataclass
class RaincloudStyle:
    """Visual knobs. Any field can be overridden when calling :func:`raincloud_plot`."""

    palette: Sequence[Color] = _PALETTE
    cloud_color: Color | Sequence[Color] | Mapping[str, Color] | None = None
    rain_color: Color | Sequence[Color] | Mapping[str, Color] | None = None
    box_facecolor: Color | Sequence[Color] | Mapping[str, Color] | None = None
    box_edgecolor: Color = "#2B2B2B"
    center_color: Color = "#1A1A1A"
    whisker_color: Color = "#2B2B2B"
    cloud_alpha: float = 0.72
    rain_alpha: float = 0.45
    box_alpha: float = 0.88
    rain_size: float = 14.0
    box_width: float = 0.14
    cloud_width: float = 0.34
    rain_span: float = 0.18
    linewidth: float = 1.2
    fontsize: float | None = None
    label_fontsize: float | None = None
    title_fontsize: float | None = None
    facecolor: Color = "white"
    grid: bool = False
    grid_color: Color = "#E6E6E6"
    spine_color: Color = "#444444"


def _clean(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _spec_name(spec: object, fallback: str) -> str:
    if callable(spec) and not isinstance(spec, str):
        return getattr(spec, "__name__", fallback)
    return str(spec)


def estimate_mode(values: np.ndarray, grid_size: int = 512) -> float:
    """Mode of a continuous sample: ``x`` that maximises a Gaussian KDE."""
    clean = _clean(values)
    if clean.size == 0:
        return float("nan")
    if np.allclose(clean, clean[0]):
        return float(clean[0])
    grid, density = _kde_curve(clean, grid_size=grid_size)
    return float(grid[int(np.argmax(density))])


def _center_value(values: np.ndarray, spec: CenterSpec) -> float:
    clean = _clean(values)
    if clean.size == 0:
        return float("nan")
    if callable(spec) and not isinstance(spec, str):
        return float(spec(clean))
    if spec == "median":
        return float(np.median(clean))
    if spec == "mean":
        return float(np.mean(clean))
    if spec == "mode":
        return estimate_mode(clean)
    raise ValueError(
        f"Unknown center estimate {spec!r}. Use median/mean/mode or a callable."
    )


def _interval(
    values: np.ndarray, spec: IntervalSpec, *, kind: str
) -> tuple[float, float]:
    clean = _clean(values)
    if clean.size == 0:
        return float("nan"), float("nan")
    if callable(spec) and not isinstance(spec, str):
        low, high = spec(clean)
        return float(low), float(high)

    q1, q3 = np.quantile(clean, [0.25, 0.75])
    iqr = float(q3 - q1)
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))

    named: dict[str, tuple[float, float]] = {
        "iqr": (float(q1), float(q3)),
        "std": (mean - std, mean + std),
        "mad": (median - 1.4826 * mad, median + 1.4826 * mad),
        "q10_q90": (float(np.quantile(clean, 0.10)), float(np.quantile(clean, 0.90))),
        "range": (float(np.min(clean)), float(np.max(clean))),
        "tukey": (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)),
        "p05_p95": (float(np.quantile(clean, 0.05)), float(np.quantile(clean, 0.95))),
        "std2": (mean - 2.0 * std, mean + 2.0 * std),
    }
    if spec not in named:
        raise ValueError(
            f"Unknown {kind} estimate {spec!r}. "
            f"Built-ins: {', '.join(sorted(named))} — or pass a callable."
        )
    return named[str(spec)]


def resolve_box_layers(values: np.ndarray, estimates: BoxEstimates) -> BoxLayerValues:
    """Compute the three box layers for one numeric sample."""
    clean = _clean(values)
    center = _center_value(clean, estimates.center)
    box_low, box_high = _interval(clean, estimates.box, kind="box")
    whis_low, whis_high = _interval(clean, estimates.whiskers, kind="whiskers")
    if np.isfinite(box_low) and np.isfinite(box_high) and box_low > box_high:
        box_low, box_high = box_high, box_low
    if np.isfinite(whis_low) and np.isfinite(whis_high) and whis_low > whis_high:
        whis_low, whis_high = whis_high, whis_low
    return BoxLayerValues(
        center=center,
        box_low=box_low,
        box_high=box_high,
        whisker_low=whis_low,
        whisker_high=whis_high,
        n=int(clean.size),
        center_name=_spec_name(estimates.center, "center"),
        box_name=_spec_name(estimates.box, "box"),
        whisker_name=_spec_name(estimates.whiskers, "whiskers"),
    )


def _kde_curve(
    values: np.ndarray, grid_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    clean = _clean(values)
    vmin, vmax = float(np.min(clean)), float(np.max(clean))
    if np.isclose(vmin, vmax):
        return np.array([vmin, vmax]), np.array([1.0, 1.0])
    pad = 0.05 * (vmax - vmin)
    grid = np.linspace(vmin - pad, vmax + pad, grid_size)
    try:
        from scipy.stats import gaussian_kde

        density = np.asarray(gaussian_kde(clean)(grid), dtype=float)
    except Exception:  # noqa: BLE001
        counts, edges = np.histogram(clean, bins="fd", density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = np.interp(grid, centers, counts, left=0.0, right=0.0)
    peak = float(np.max(density))
    if peak <= 0:
        density = np.ones_like(grid)
    return grid, density


def _as_long_frame(
    data: pd.DataFrame | Mapping[str, Any],
    columns: Sequence[str] | None,
    group: str | None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return a long table with columns ``feature``, ``value``, ``group``."""
    if isinstance(data, Mapping) and not isinstance(data, pd.DataFrame):
        frame = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise TypeError("data must be a DataFrame or a column mapping")

    if columns is None:
        numeric = frame.select_dtypes(include="number")
        features = [name for name in numeric.columns if name != group]
    else:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise KeyError(f"Requested columns are not in data: {missing}")
        features = list(columns)

    if not features:
        raise ValueError("No numeric features to plot.")

    if group is not None:
        if group not in frame.columns:
            raise KeyError(f"group column {group!r} is not in data")
        group_values = frame[group]
    else:
        group_values = pd.Series(["all"] * len(frame), index=frame.index)

    long = (
        frame.loc[:, features]
        .assign(_group=group_values.to_numpy())
        .melt(id_vars="_group", var_name="feature", value_name="value")
        .rename(columns={"_group": "group"})
    )
    long["group"] = long["group"].map(
        lambda item: "all" if pd.isna(item) else str(item)
    )
    long["feature"] = pd.Categorical(long["feature"], categories=features, ordered=True)
    groups = list(dict.fromkeys(long["group"].tolist()))
    return long, features, groups


def _colors_for(
    spec: Color | Sequence[Color] | Mapping[str, Color] | None,
    names: Sequence[str],
    fallback: Sequence[Color],
) -> dict[str, Color]:
    palette = list(fallback) if fallback else list(_PALETTE)
    if spec is None:
        return {name: palette[i % len(palette)] for i, name in enumerate(names)}
    if isinstance(spec, Mapping):
        return {
            name: spec.get(name, palette[i % len(palette)])
            for i, name in enumerate(names)
        }
    if isinstance(spec, str) or (isinstance(spec, tuple) and len(spec) in (3, 4)):
        if len(names) == 1:
            return {names[0]: spec}  # type: ignore[dict-item]
        return {name: palette[i % len(palette)] for i, name in enumerate(names)}
    sequence = list(spec)  # type: ignore[arg-type]
    if not sequence:
        sequence = palette
    return {name: sequence[i % len(sequence)] for i, name in enumerate(names)}


def _l_frame(ax: Axes, style: RaincloudStyle, *, value_is_y: bool) -> None:
    """Keep only the L-shaped value + category spines."""
    keep = {"left", "bottom"} if value_is_y else {"bottom", "left"}
    for name, spine in ax.spines.items():
        if name in keep:
            spine.set_visible(True)
            spine.set_color(style.spine_color)
            spine.set_linewidth(1.0)
        else:
            spine.set_visible(False)
    ax.tick_params(colors=style.spine_color, width=0.8)


def _draw_cloud(
    ax: Axes,
    values: np.ndarray,
    *,
    cat_pos: float,
    color: Color,
    style: RaincloudStyle,
    value_is_y: bool,
) -> None:
    clean = _clean(values)
    if clean.size == 0:
        return
    grid, density = _kde_curve(clean)
    peak = float(np.max(density))
    width = density / peak * style.cloud_width
    outer = cat_pos + width
    if value_is_y:
        ax.fill_betweenx(
            grid,
            cat_pos,
            outer,
            color=color,
            alpha=style.cloud_alpha,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            outer,
            grid,
            color=color,
            lw=style.linewidth,
            solid_capstyle="round",
            zorder=3,
        )
    else:
        ax.fill_between(
            grid,
            cat_pos,
            outer,
            color=color,
            alpha=style.cloud_alpha,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            grid,
            outer,
            color=color,
            lw=style.linewidth,
            solid_capstyle="round",
            zorder=3,
        )


def _draw_rain(
    ax: Axes,
    values: np.ndarray,
    *,
    cat_pos: float,
    color: Color,
    style: RaincloudStyle,
    value_is_y: bool,
    rng: np.random.RandomState,
) -> None:
    clean = _clean(values)
    if clean.size == 0:
        return
    jitter = rng.uniform(-style.rain_span * 0.5, style.rain_span * 0.5, size=clean.size)
    cats = np.full(clean.size, cat_pos) + jitter
    kwargs = {
        "s": style.rain_size,
        "c": color,
        "alpha": style.rain_alpha,
        "linewidths": 0,
        "zorder": 2,
        "rasterized": True,
    }
    if value_is_y:
        ax.scatter(cats, clean, **kwargs)
    else:
        ax.scatter(clean, cats, **kwargs)


def _draw_box(
    ax: Axes,
    layers: BoxLayerValues,
    *,
    cat_pos: float,
    facecolor: Color,
    style: RaincloudStyle,
    value_is_y: bool,
    show_fliers: bool,
    values: np.ndarray,
) -> None:
    half = style.box_width * 0.5
    box_span = layers.box_high - layers.box_low
    edge = style.box_edgecolor
    if value_is_y:
        ax.plot(
            [cat_pos, cat_pos],
            [layers.whisker_low, layers.whisker_high],
            color=style.whisker_color,
            lw=style.linewidth,
            solid_capstyle="butt",
            zorder=3,
        )
        for y in (layers.whisker_low, layers.whisker_high):
            ax.plot(
                [cat_pos - half * 0.75, cat_pos + half * 0.75],
                [y, y],
                color=style.whisker_color,
                lw=style.linewidth,
                zorder=3,
            )
        ax.add_patch(
            Rectangle(
                (cat_pos - half, layers.box_low),
                style.box_width,
                box_span if box_span != 0 else 1e-12,
                facecolor=facecolor,
                edgecolor=edge,
                lw=style.linewidth,
                alpha=style.box_alpha,
                zorder=4,
            )
        )
        ax.plot(
            [cat_pos - half, cat_pos + half],
            [layers.center, layers.center],
            color=style.center_color,
            lw=style.linewidth + 0.8,
            solid_capstyle="butt",
            zorder=5,
        )
        if show_fliers:
            clean = _clean(values)
            mask = (clean < layers.whisker_low) | (clean > layers.whisker_high)
            if np.any(mask):
                ax.scatter(
                    np.full(int(mask.sum()), cat_pos),
                    clean[mask],
                    s=style.rain_size * 0.9,
                    facecolors=edge,
                    edgecolors=edge,
                    linewidths=0.6,
                    zorder=5,
                )
        return

    ax.plot(
        [layers.whisker_low, layers.whisker_high],
        [cat_pos, cat_pos],
        color=style.whisker_color,
        lw=style.linewidth,
        solid_capstyle="butt",
        zorder=3,
    )
    for x in (layers.whisker_low, layers.whisker_high):
        ax.plot(
            [x, x],
            [cat_pos - half * 0.75, cat_pos + half * 0.75],
            color=style.whisker_color,
            lw=style.linewidth,
            zorder=3,
        )
    ax.add_patch(
        Rectangle(
            (layers.box_low, cat_pos - half),
            box_span if box_span != 0 else 1e-12,
            style.box_width,
            facecolor=facecolor,
            edgecolor=edge,
            lw=style.linewidth,
            alpha=style.box_alpha,
            zorder=4,
        )
    )
    ax.plot(
        [layers.center, layers.center],
        [cat_pos - half, cat_pos + half],
        color=style.center_color,
        lw=style.linewidth + 0.8,
        solid_capstyle="butt",
        zorder=5,
    )
    if show_fliers:
        clean = _clean(values)
        mask = (clean < layers.whisker_low) | (clean > layers.whisker_high)
        if np.any(mask):
            ax.scatter(
                clean[mask],
                np.full(int(mask.sum()), cat_pos),
                s=style.rain_size * 0.9,
                facecolors=edge,
                edgecolors=edge,
                linewidths=0.6,
                zorder=5,
            )


def _annotate_box_layers(
    ax: Axes,
    layers: BoxLayerValues,
    *,
    cat_pos: float,
    style: RaincloudStyle,
    value_is_y: bool,
    fontsize: float,
) -> None:
    """Write numeric values next to median / IQR / range edges."""
    half = style.box_width * 0.5
    # Sit just outside the box on the cloud side (not over the rain).
    label_cat = cat_pos + half + 0.05
    marks: list[tuple[float, str, str]] = [
        (layers.whisker_high, f"{layers.whisker_high:.3f}", "whisker"),
        (layers.box_high, f"{layers.box_high:.3f}", "box"),
        (layers.center, f"{layers.center:.3f}", "center"),
        (layers.box_low, f"{layers.box_low:.3f}", "box"),
        (layers.whisker_low, f"{layers.whisker_low:.3f}", "whisker"),
    ]
    kept: list[tuple[float, str, str]] = []
    for value, text, kind in marks:
        if not np.isfinite(value):
            continue
        if kept and abs(value - kept[-1][0]) < 1e-9:
            continue
        kept.append((value, text, kind))

    weight = {"center": "bold", "box": "normal", "whisker": "normal"}
    color = {
        "center": style.center_color,
        "box": "#333333",
        "whisker": style.whisker_color,
    }
    for value, text, kind in kept:
        common = {
            "s": text,
            "fontsize": fontsize,
            "fontweight": weight[kind],
            "color": color[kind],
            "clip_on": False,
            "zorder": 6,
        }
        if value_is_y:
            ax.text(label_cat, value, ha="left", va="center", **common)
        else:
            ax.text(value, label_cat, ha="center", va="bottom", **common)


def _draw_panel(
    ax: Axes,
    *,
    rows: pd.DataFrame,
    categories: Sequence[str],
    category_key: str,
    value_key: str,
    colors: Mapping[str, Color],
    box_colors: Mapping[str, Color],
    rain_colors: Mapping[str, Color],
    box_est: BoxEstimates,
    style: RaincloudStyle,
    value_is_y: bool,
    show_cloud: bool,
    show_rain: bool,
    show_box: bool,
    show_fliers: bool,
    show_estimate_values: bool,
    label_font: float,
    base_font: float,
    ylabel: str | None,
    rng: np.random.RandomState,
    resolved: dict[tuple[str, str], BoxLayerValues],
    feature_name: str,
) -> None:
    ax.set_facecolor(style.facecolor)
    n_cat = len(categories)
    centres = np.arange(n_cat, dtype=float) * _CATEGORY_PITCH
    annotate_font = max(base_font - 2.5, 6.5)

    for i, cat in enumerate(categories):
        centre = float(centres[i])
        values = rows.loc[rows[category_key] == cat, value_key].to_numpy()
        layers = resolve_box_layers(values, box_est)
        resolved[(feature_name, str(cat))] = layers
        color = colors[cat]
        if show_rain:
            _draw_rain(
                ax,
                values,
                cat_pos=centre + _RAIN_OFFSET,
                color=rain_colors[cat],
                style=style,
                value_is_y=value_is_y,
                rng=rng,
            )
        if show_box:
            _draw_box(
                ax,
                layers,
                cat_pos=centre + _BOX_OFFSET,
                facecolor=box_colors[cat],
                style=style,
                value_is_y=value_is_y,
                show_fliers=show_fliers,
                values=values,
            )
            if show_estimate_values:
                _annotate_box_layers(
                    ax,
                    layers,
                    cat_pos=centre + _BOX_OFFSET,
                    style=style,
                    value_is_y=value_is_y,
                    fontsize=annotate_font,
                )
        if show_cloud:
            _draw_cloud(
                ax,
                values,
                cat_pos=centre + _CLOUD_OFFSET,
                color=color,
                style=style,
                value_is_y=value_is_y,
            )

    # Extra category padding so edge labels are not clipped.
    pad = 0.55
    cat_lo = float(centres[0] + _RAIN_OFFSET) - pad
    cat_hi = float(centres[-1] + _CLOUD_OFFSET + style.cloud_width) + (
        0.55 if show_estimate_values else 0.35
    )

    if value_is_y:
        ax.set_xlim(cat_lo, cat_hi)
        ax.set_xticks(centres)
        ax.set_xticklabels(list(categories), fontsize=label_font)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_font)
        _l_frame(ax, style, value_is_y=True)
        if style.grid:
            ax.yaxis.grid(True, color=style.grid_color, lw=0.7)
            ax.set_axisbelow(True)
    else:
        ax.set_ylim(cat_lo, cat_hi)
        ax.set_yticks(centres)
        ax.set_yticklabels(list(categories), fontsize=label_font)
        if ylabel:
            ax.set_xlabel(ylabel, fontsize=label_font)
        _l_frame(ax, style, value_is_y=False)
        if style.grid:
            ax.xaxis.grid(True, color=style.grid_color, lw=0.7)
            ax.set_axisbelow(True)


def raincloud_plot(
    data: pd.DataFrame | Mapping[str, Any],
    *,
    columns: Sequence[str] | None = None,
    group: str | None = None,
    orientation: Orientation = "vertical",
    estimates: BoxEstimates | None = None,
    center: CenterSpec | None = None,
    box: IntervalSpec | None = None,
    whiskers: IntervalSpec | None = None,
    show_cloud: bool = True,
    show_rain: bool = True,
    show_box: bool = True,
    show_fliers: bool | None = None,
    show_estimate_values: bool = True,
    show_legend: bool = False,
    figsize: tuple[float, float] | None = None,
    fig: Figure | None = None,
    title: str | None = None,
    random_state: int | None = 0,
    ax_kwargs: Mapping[str, Any] | None = None,
    style: RaincloudStyle | None = None,
    **style_overrides: Any,
) -> RaincloudResult:
    """Draw a raincloud (rain | box | cloud) for each feature / group.

    Parameters
    ----------
    data
        Wide DataFrame, or a mapping of column name → values.
    columns
        Features to plot. ``None`` uses every numeric column except ``group``.
    group
        Optional categorical column. If omitted, features themselves become
        the categories on a single axes. If set, one axes is drawn per
        feature and group levels are the categories.
    orientation
        ``vertical`` — value on **y**, rain | box | cloud left-to-right.
        ``horizontal`` — value on **x**, rain | box | cloud bottom-to-top.
    estimates, center, box, whiskers
        Box layers. Defaults are median / IQR / range. Built-in names:

        * center: ``median``, ``mean``, ``mode``
        * box: ``iqr``, ``std``, ``mad``, ``q10_q90``
        * whiskers: ``range``, ``tukey``, ``p05_p95``, ``std2``

        Any of the three may be a callable on a 1-D array.
    show_fliers
        Draw points outside the whiskers. ``None`` turns them on only when
        whiskers are not the full range.
    show_estimate_values
        Annotate median / IQR / range edge values next to the box.
    show_legend
        Draw a color / layer legend. Default is off so the plot can use the
        full figure width.
    style, **style_overrides
        :class:`RaincloudStyle` fields. Default palette matches ``test.py``.

    Returns
    -------
    RaincloudResult
        Figure, axes, categories, and resolved estimates.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError("orientation must be 'horizontal' or 'vertical'")

    unknown_style = sorted(
        key for key in style_overrides if key not in RaincloudStyle.__dataclass_fields__
    )
    if unknown_style:
        raise TypeError(f"Unknown style field: {unknown_style}")
    style_obj = replace(style or RaincloudStyle(), **style_overrides)

    box_est = estimates or BoxEstimates()
    if center is not None or box is not None or whiskers is not None:
        box_est = replace(
            box_est,
            **{
                key: value
                for key, value in {
                    "center": center,
                    "box": box,
                    "whiskers": whiskers,
                }.items()
                if value is not None
            },
        )

    long, features, groups = _as_long_frame(data, columns, group)
    value_is_y = orientation == "vertical"
    draw_fliers = (
        str(box_est.whiskers) != "range" if show_fliers is None else bool(show_fliers)
    )

    # Layout mode: features-as-categories (one axes) vs groups-as-categories (one axes / feature).
    feature_as_category = group is None or groups == ["all"]
    if feature_as_category:
        panel_features = ["__all__"]
        categories = features
        panel_rows = {"__all__": long}
        ylabels = {"__all__": None}
        color_names = features
    else:
        panel_features = features
        categories = groups
        panel_rows = {
            name: long.loc[long["feature"] == name].copy() for name in features
        }
        for frame in panel_rows.values():
            frame["category"] = frame["group"]
        ylabels = {name: str(name) for name in features}
        color_names = groups

    n_panels = len(panel_features)
    n_cat = len(categories)
    if n_cat == 0:
        raise ValueError("No categories to plot.")

    palette = list(style_obj.palette) if style_obj.palette else list(_PALETTE)
    colors = _colors_for(style_obj.cloud_color, color_names, palette)
    box_colors = _colors_for(style_obj.box_facecolor, color_names, palette)
    rain_colors = _colors_for(style_obj.rain_color, color_names, palette)

    base_font = style_obj.fontsize if style_obj.fontsize is not None else 10.5
    label_font = (
        style_obj.label_fontsize if style_obj.label_fontsize is not None else base_font
    )
    title_font = (
        style_obj.title_fontsize
        if style_obj.title_fontsize is not None
        else base_font + 2.5
    )

    if figsize is None:
        if feature_as_category:
            width = max(6.5, 1.7 * n_cat + 2.2)
            height = 5.2
            figsize = (
                (width, height)
                if value_is_y
                else (height + 0.8, max(4.8, 1.35 * n_cat + 1.4))
            )
        elif value_is_y:
            figsize = (max(6.5, 1.5 * n_cat + 2.0), 2.6 * n_panels + 0.8)
        else:
            figsize = (2.8 * n_panels + 1.4, max(4.8, 1.35 * n_cat + 1.4))

    if fig is None:
        if feature_as_category:
            fig, ax0 = plt.subplots(
                1, 1, figsize=figsize, facecolor=style_obj.facecolor
            )
            axes = [ax0]
        elif value_is_y:
            fig, axes_arr = plt.subplots(
                n_panels,
                1,
                figsize=figsize,
                facecolor=style_obj.facecolor,
                squeeze=False,
            )
            axes = [row[0] for row in axes_arr]
        else:
            fig, axes_arr = plt.subplots(
                1,
                n_panels,
                figsize=figsize,
                facecolor=style_obj.facecolor,
                squeeze=False,
            )
            axes = list(axes_arr[0])
    else:
        fig.patch.set_facecolor(style_obj.facecolor)
        axes = list(fig.axes)
        if len(axes) < n_panels:
            raise ValueError(f"fig has {len(axes)} axes, need at least {n_panels}")
        axes = axes[:n_panels]

    rng = np.random.RandomState(random_state)
    resolved: dict[tuple[str, str], BoxLayerValues] = {}

    for ax, panel in zip(axes, panel_features):
        if ax_kwargs:
            ax.set(**ax_kwargs)
        rows = panel_rows[panel].copy()
        if feature_as_category:
            plot_rows = rows
            panel_categories = features
            cat_key = "feature"
            feat_label = "features"
            panel_colors = colors
            panel_box = box_colors
            panel_rain = rain_colors
            panel_ylabel = None
        else:
            plot_rows = rows.assign(category=rows["group"].astype(str))
            panel_categories = categories
            cat_key = "category"
            feat_label = str(panel)
            panel_colors = colors
            panel_box = box_colors
            panel_rain = rain_colors
            panel_ylabel = ylabels[panel]

        _draw_panel(
            ax,
            rows=plot_rows,
            categories=panel_categories,
            category_key=cat_key,
            value_key="value",
            colors=panel_colors,
            box_colors=panel_box,
            rain_colors=panel_rain,
            box_est=box_est,
            style=style_obj,
            value_is_y=value_is_y,
            show_cloud=show_cloud,
            show_rain=show_rain,
            show_box=show_box,
            show_fliers=draw_fliers,
            show_estimate_values=show_estimate_values,
            label_font=label_font,
            base_font=base_font,
            ylabel=panel_ylabel,
            rng=rng,
            resolved=resolved,
            feature_name=feat_label,
        )

    # Legend: one swatch per category + box-layer keys.
    if show_legend:
        legend_cats = features if feature_as_category else groups
        legend_colors = (
            _colors_for(style_obj.cloud_color, features, palette)
            if feature_as_category
            else colors
        )
        first = next(iter(resolved.values()))
        handles: list[Any] = [
            Patch(facecolor=legend_colors[name], edgecolor="none", label=str(name))
            for name in legend_cats
        ]
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=style_obj.center_color,
                    lw=2.4,
                    label=f"center: {first.center_name}",
                ),
                Patch(
                    facecolor="#DDDDDD",
                    edgecolor=style_obj.box_edgecolor,
                    label=f"box: {first.box_name}",
                ),
                Line2D(
                    [0],
                    [0],
                    color=style_obj.whisker_color,
                    lw=1.5,
                    label=f"whiskers: {first.whisker_name}",
                ),
            ]
        )
        fig.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=base_font - 1,
            borderaxespad=0.5,
        )

    if title:
        fig.suptitle(title, fontsize=title_font, y=0.98)
    fig.subplots_adjust(
        left=0.10 if value_is_y else 0.14,
        right=0.82 if show_legend else 0.98,
        top=0.90 if title else 0.96,
        bottom=0.12,
        hspace=0.35,
        wspace=0.28,
    )

    result_categories = tuple(features if feature_as_category else groups)
    return RaincloudResult(
        fig=fig,
        axes=tuple(axes),
        features=tuple(features),
        groups=tuple(groups),
        categories=result_categories,
        estimates=resolved,
        orientation=orientation,
    )

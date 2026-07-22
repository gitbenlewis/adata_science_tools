''' correlation dotplots '''
# module at _plotting/_corr_dotplots.py
import warnings
from collections.abc import Mapping as _Mapping, Sequence
from numbers import Real as _Real
from typing import Any, Literal

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

from . import palettes
from ._utils import _draw_reference_lines, _normalize_reference_lines

_NONLINEAR_LINE_SAMPLES = 257


def _is_categorical_series(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _scale_lower_bound(scale: str) -> float | None:
    if scale in {"log", "log2"}:
        return 0.0
    if scale == "log1p":
        return -1.0
    return None


def _log1p_forward(values: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(values)


def _forward_scale(values: Any, scale: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if scale == "log":
        return np.log10(numeric)
    if scale == "log2":
        return np.log2(numeric)
    if scale == "log1p":
        return _log1p_forward(numeric)
    return numeric


def _inverse_scale(values: Any, scale: str) -> np.ndarray:
    numeric = np.asarray(values, dtype=float)
    if scale == "log":
        return np.power(10.0, numeric)
    if scale == "log2":
        return np.exp2(numeric)
    if scale == "log1p":
        return np.expm1(numeric)
    return numeric


def _sample_linear_relation(
    x_endpoints: Sequence[float],
    *,
    intercept: float,
    slope: float,
    x_scale: str,
    y_scale: str,
) -> tuple[Sequence[float], Sequence[float]]:
    x_endpoints_array = np.asarray(x_endpoints, dtype=float)
    y_endpoints_array = intercept + slope * x_endpoints_array
    if x_scale == "linear" and y_scale == "linear":
        return x_endpoints_array.tolist(), y_endpoints_array.tolist()

    transformed_x = _forward_scale(x_endpoints_array, x_scale)
    x_samples = _inverse_scale(
        np.linspace(transformed_x[0], transformed_x[1], _NONLINEAR_LINE_SAMPLES),
        x_scale,
    )
    sample_sets = [x_endpoints_array, x_samples]
    if y_scale != "linear" and slope != 0:
        transformed_y = _forward_scale(y_endpoints_array, y_scale)
        y_samples = _inverse_scale(
            np.linspace(transformed_y[0], transformed_y[1], _NONLINEAR_LINE_SAMPLES),
            y_scale,
        )
        sample_sets.append((y_samples - intercept) / slope)

    x_coordinates = np.unique(
        np.clip(
            np.concatenate(sample_sets),
            x_endpoints_array[0],
            x_endpoints_array[1],
        )
    )
    return x_coordinates, intercept + slope * x_coordinates


def _compute_corr_and_fit(
    x_vals: pd.Series,
    y_vals: pd.Series,
    method: Literal["spearman", "pearson"],
) -> tuple[Any, float, float]:
    if method == "spearman":
        corr_res = stats.spearmanr(x_vals, y_vals)
        corr_value = corr_res.statistic
        corr_pvalue = corr_res.pvalue
    else:
        corr_value, corr_pvalue = stats.pearsonr(x_vals, y_vals)

    fit = linregress(x_vals, y_vals)
    return fit, corr_value, corr_pvalue


def _try_compute_corr_and_fit(
    x_vals: pd.Series,
    y_vals: pd.Series,
    method: Literal["spearman", "pearson"],
) -> tuple[Any | None, float | None, float | None]:
    try:
        return _compute_corr_and_fit(x_vals, y_vals, method)
    except Exception:
        return None, None, None


def _fit_line_coordinates(
    x_vals: pd.Series,
    fit: Any,
    x_scale: str,
    y_scale: str,
) -> tuple[Sequence[Any], Sequence[Any]]:
    x_endpoints = [float(x_vals.min()), float(x_vals.max())]
    y_endpoints = [fit.intercept + fit.slope * value for value in x_endpoints]
    y_lower_bound = _scale_lower_bound(y_scale)
    if y_lower_bound is not None and any(
        not np.isfinite(value) or value <= y_lower_bound for value in y_endpoints
    ):
        raise ValueError(
            "The rendered fit line must be finite and greater than "
            f"{y_lower_bound:g} for '{y_scale}'."
        )
    return _sample_linear_relation(
        x_endpoints,
        intercept=fit.intercept,
        slope=fit.slope,
        x_scale=x_scale,
        y_scale=y_scale,
    )


def _plot_fit_line(
    axes: plt.Axes,
    x_vals: pd.Series,
    fit: Any,
    *,
    show_y_intercept: bool,
    color: Any,
    linestyle: str,
    label: str | None,
    use_data_range: bool = False,
    x_scale: str = "linear",
    y_scale: str = "linear",
    coordinates: tuple[Sequence[Any], Sequence[Any]] | None = None,
):
    line_kwargs: dict[str, Any] = {"color": color, "linestyle": linestyle}
    if label is not None:
        line_kwargs["label"] = label

    if show_y_intercept and not use_data_range:
        return axes.axline(xy1=(0, fit.intercept), slope=fit.slope, **line_kwargs)

    if coordinates is None:
        coordinates = _fit_line_coordinates(x_vals, fit, x_scale, y_scale)
    x_coordinates, y_coordinates = coordinates
    (line,) = axes.plot(
        x_coordinates,
        y_coordinates,
        **line_kwargs,
    )
    return line


def _format_subset_stats_line(
    label: str,
    method: Literal["spearman", "pearson"],
    fit: Any | None,
    corr_value: float | None,
    corr_pvalue: float | None,
) -> str:
    if fit is None or corr_value is None or corr_pvalue is None:
        return f"{label}: fit unavailable"

    corr_label = method.capitalize()
    return (
        f"{label}: {corr_label} Corr = {corr_value:.3f} pvalue = {corr_pvalue:.6f} "
        f"y = {fit.intercept:.3f} + {fit.slope:.3f}x R^2: {fit.rvalue ** 2:.3f}"
    )


def _format_subset_fit_legend_label(
    label: str,
    corr_value: float,
    corr_pvalue: float,
) -> str:
    return f"{label}\nCorr={corr_value:.3f},p={corr_pvalue:.2e}"


def _normalize_bbox_to_anchor(
    bbox_to_anchor: Sequence[float] | None,
    *,
    param_name: str,
) -> tuple[float, ...] | None:
    if bbox_to_anchor is None:
        return None

    bbox_tuple = tuple(bbox_to_anchor)
    if len(bbox_tuple) not in {2, 4}:
        raise ValueError(f"'{param_name}' must contain 2 or 4 numeric values.")
    return bbox_tuple


def _normalize_axis_limits(
    limits: Sequence[float] | None,
    *,
    param_name: str,
    scale: str,
) -> tuple[float, float] | None:
    if limits is None:
        return None
    values = tuple(limits)
    if len(values) != 2:
        raise ValueError(f"'{param_name}' must contain exactly two values.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, _Real) for value in values):
        raise ValueError(f"'{param_name}' must contain numeric values.")
    lower, upper = float(values[0]), float(values[1])
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(f"'{param_name}' must contain finite increasing values.")
    lower_bound = _scale_lower_bound(scale)
    if lower_bound is not None and lower <= lower_bound:
        raise ValueError(
            f"'{param_name}' values must be greater than {lower_bound:g} for '{scale}'."
        )
    return lower, upper


def _normalize_padding(value: float | None, *, param_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, _Real):
        raise ValueError(f"'{param_name}' must be numeric.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"'{param_name}' must be finite and nonnegative.")
    return value


def _resolve_axis_limits(
    values: pd.Series,
    current_limits: tuple[float, float],
    *,
    explicit_limits: tuple[float, float] | None,
    padding_fraction: float | None,
    scale: str,
    extra_values: Sequence[float] = (),
) -> tuple[float, float]:
    if explicit_limits is not None:
        return explicit_limits
    if padding_fraction is None:
        if scale != "log1p":
            return current_limits
        padding_fraction = 0.05
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if extra_values:
        extra_numeric = np.asarray(tuple(extra_values), dtype=float)
        finite = np.concatenate([finite, extra_numeric[np.isfinite(extra_numeric)]])
    if finite.size == 0:
        raise ValueError("Cannot resolve limits without finite rendered values.")
    lower = float(finite.min())
    upper = float(finite.max())
    if scale in {"log", "log2", "log1p"}:
        transformed_lower, transformed_upper = _forward_scale([lower, upper], scale)
        if transformed_lower == transformed_upper:
            delta = max(abs(transformed_lower) * padding_fraction, 0.05)
        else:
            delta = (transformed_upper - transformed_lower) * padding_fraction
        resolved = _inverse_scale(
            [transformed_lower - delta, transformed_upper + delta],
            scale,
        )
        return (
            float(resolved[0]),
            float(resolved[1]),
        )
    if lower == upper:
        delta = max(abs(lower) * padding_fraction, 0.5)
    else:
        delta = (upper - lower) * padding_fraction
    return lower - delta, upper + delta


def _validate_scale_values(values: pd.Series, *, param_name: str, scale: str) -> None:
    lower_bound = _scale_lower_bound(scale)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= lower_bound).any():
        if lower_bound == 0:
            raise ValueError(
                f"'{param_name}' contains nonpositive or non-finite values for '{scale}'."
            )
        raise ValueError(
            f"'{param_name}' values must be finite and greater than {lower_bound:g} for '{scale}'."
        )


def corr_dotplot(
    df: pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    layer: str | None = None,
    x_df: Any | None = None,
    var_df: pd.DataFrame | None = None,
    obs_df: pd.DataFrame | None = None,
    column_key_x: str | None = None,
    column_key_y: str | None = None,
    hue: str | None = None,
    subset_key: str | None = None,
    figsize: tuple[float, float] = (20, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    axes_title: str | None = None,
    axes_lines: bool = True,
    show_y_intercept: bool = True,
    palette: Sequence[Any] | str | None = palettes.godsnot_102,
    subset_palette: Sequence[Any] | str | None = None,
    dot_size: float = 200,
    title_fontsize: int = 20,
    stats_fontsize: int | None = None,
    axes_title_y: float | None = None,
    axis_label_fontsize: int = 20,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    fit_legend_bbox_to_anchor: Sequence[float] | None = None,
    hue_legend_bbox_to_anchor: Sequence[float] | None = None,
    show_all_obs_fit: bool = False,
    show_fit: bool = True,
    show_fit_legend: bool = True,
    show_hue_legend: bool = True,
    show_stats_text: bool = True,
    show_identity_line: bool = False,
    identity_line_label: str | None = "Identity",
    identity_line_style: _Mapping[str, Any] | None = None,
    identity_limits: Literal["shared_axes", "data"] = "shared_axes",
    nas2zeros: bool = False,
    dropna: bool = False,
    dropzeros: bool = False,
    method: Literal["spearman", "pearson"] = "pearson",
    show_x_marginal_hist: bool = False,
    show_y_marginal_hist: bool = False,
    x_marginal_hist_bins: int | Sequence[float] = 20,
    y_marginal_hist_bins: int | Sequence[float] = 20,
    x_marginal_hist_fill: bool = True,
    x_marginal_hist_KDE: bool = True,
    y_marginal_hist_fill: bool = True,
    y_marginal_hist_KDE: bool = True,
    show_all_obs_x_hist: bool = False,
    show_all_obs_y_hist: bool = False,
    x_marginal_hist_height_ratio: float = 0.18,
    y_marginal_hist_width_ratio: float = 0.18,
    xscale: str = "linear",
    yscale: str = "linear",
    xlims: Sequence[float] | None = None,
    ylims: Sequence[float] | None = None,
    xlim_padding_fraction: float | None = None,
    ylim_padding_fraction: float | None = None,
    x_reference_lines: Sequence[_Mapping[str, Any]] | None = None,
    y_reference_lines: Sequence[_Mapping[str, Any]] | None = None,
    show: bool = True,
) -> tuple[
    plt.Figure,
    plt.Axes | dict[str, plt.Axes | None],
    Any,
    float,
    float,
]:
    """Plot a correlation scatter with optional marginal histograms.

    Parameters
    ----------
    df : pandas.DataFrame | None
        Pre-assembled DataFrame containing the required columns. When supplied,
        all AnnData-related parameters are ignored.
    adata : anndata.AnnData | None
        AnnData object that provides expression values and observation metadata.
    layer : str | None
        Name of ``adata.layers`` matrix used instead of ``adata.X`` when pulling
        expression values.
    x_df : pandas.DataFrame or other 2D matrix-like object, optional
        Expression matrix with observations as rows and features as columns. Overrides
        AnnData-derived matrices when supplied and is converted to a DataFrame when needed.
    var_df : pandas.DataFrame | None
        Feature metadata used to supply column names when ``x_df`` is not a DataFrame.
    obs_df : pandas.DataFrame | None
        Observation metadata. Overrides ``adata.obs`` when provided.
    column_key_x, column_key_y : str | None
        Keys selecting the x- and y-value columns used for the correlation.
    hue : str | None
        Observation column used for colouring points in the scatter plot. When ``None``,
        points are plotted without grouping or a legend.
    subset_key : str | None
        Column used to split the filtered observations into subgroup fit lines.
    figsize : tuple[float, float]
        Figure size passed to ``plt.subplots``.
    xlabel, ylabel : str | None
        Optional axis labels overriding the default column names.
    axes_title : str | None
        Optional axes title applied with ``Axes.set_title``.
    axes_lines : bool
        Draw horizontal and vertical reference lines through the origin when ``True``.
    palette : Sequence | str | None
        Palette forwarded to ``seaborn.scatterplot`` for ``hue``-coloured points.
    subset_palette : Sequence | str | None
        Palette used for ``subset_key``-driven subgroup fit lines. When ``None``,
        subgroup fits fall back to ``palette`` for backward compatibility.
    dot_size : float
        Point size passed to ``seaborn.scatterplot`` as ``s``.
    title_fontsize : int
        Font size used for the optional ``axes_title``.
    stats_fontsize : int | None
        Font size used for the statistical summary footer. Defaults to ``title_fontsize``
        when ``None``.
    axes_title_y : float | None
        Vertical position passed to ``Axes.set_title`` when ``axes_title`` is provided.
    axis_label_fontsize : int
        Font size used for explicitly provided ``xlabel`` and ``ylabel`` values.
    tick_label_fontsize : int | None
        Tick label font size applied to both axes when provided.
    legend_fontsize : int | None
        Legend label and title font size applied when a legend is drawn.
    fit_legend_bbox_to_anchor : Sequence[float] | None
        List-like anchor passed to the fit legend. Defaults to the current
        outside-right placement when not provided.
    hue_legend_bbox_to_anchor : Sequence[float] | None
        List-like anchor passed to the hue legend. Defaults to the current
        outside-right placement when not provided.
    show_all_obs_fit : bool
        In subset mode, also draw and report the overall fit across all filtered observations.
    show_fit_legend : bool
        Show the fit-line legend when subgroup fitting is enabled.
    show_hue_legend : bool
        Show the scatter/hue legend when ``hue`` is provided.
    show_stats_text : bool
        Show the statistical summary footer text when ``True``.
    nas2zeros : bool
        Replace missing x/y values with zeros when ``True``. occurs before / overrides ``dropna``.
    dropna : bool
        Remove observations with missing x/y values when ``True``.
    dropzeros : bool
        Remove observations where either the x or y value equals zero (after numeric coercion).
    method : {"spearman", "pearson"}
        Correlation statistic to report and place in the title.
    show_x_marginal_hist, show_y_marginal_hist : bool
        Add a histogram above the scatter for x values and/or to its right for y values.
    x_marginal_hist_bins, y_marginal_hist_bins : int | Sequence[float]
        Histogram bin counts or explicit bin edges for each marginal axis.
    x_marginal_hist_fill, y_marginal_hist_fill : bool
        Fill the corresponding step histogram when true.
    x_marginal_hist_KDE, y_marginal_hist_KDE : bool
        Draw a kernel-density estimate on the corresponding marginal when true.
    show_all_obs_x_hist, show_all_obs_y_hist : bool
        In subset mode, overlay the all-observation distribution on the corresponding marginal.
    x_marginal_hist_height_ratio, y_marginal_hist_width_ratio : float
        Marginal-to-main subplot size ratios used by the composite layout.
    xscale, yscale : {"linear", "log", "log2", "log1p"}
        Axis scales. ``log2`` uses a base-2 logarithm; ``log1p`` uses
        ``log1p``/``expm1`` and accepts only finite values greater than -1.
        Correlation and fit calculations always use the untransformed values.
    show : bool
        Call ``plt.show()`` before returning when ``True``.

    Notes
    -----
    When observation metadata and feature columns share a name in the assembled
    ``adata`` / ``obs_df`` path, the metadata column is renamed to ``<name>_obs``
    before concatenation so the bare name continues to refer to the feature column.
    When ``subset_key`` is provided, subgroup fits and subgroup footer stats are
    displayed, but the returned ``fit``, ``corr_value``, and ``corr_pvalue`` remain
    tied to the overall filtered observations for backward compatibility.
    Marginal histograms use the same filtered observations as the scatter and use
    ``subset_key`` rather than ``hue`` for grouping. If no valid subset values remain,
    the plot falls back to all-data fit, statistics, and enabled marginals.

    Returns
    -------
    tuple
        ``(fig, axes, fit, corr_value, corr_pvalue)``. ``axes`` is the single main
        axes when both marginals are disabled. When either marginal is enabled,
        it is a dictionary containing ``"main"``, ``"x_marginal"``, and
        ``"y_marginal"``; disabled marginal entries are ``None``.
    """

    if column_key_x is None or column_key_y is None:
        raise ValueError("Both 'column_key_x' and 'column_key_y' must be provided.")

    method = method.lower()
    if method not in {"spearman", "pearson"}:
        raise ValueError("'method' must be either 'spearman' or 'pearson'.")
    if identity_limits not in {"shared_axes", "data"}:
        raise ValueError("'identity_limits' must be 'shared_axes' or 'data'.")
    for scale, param_name in ((xscale, "xscale"), (yscale, "yscale")):
        if not isinstance(scale, str) or scale not in {"linear", "log", "log2", "log1p"}:
            raise ValueError(f"'{param_name}' must be 'linear', 'log', 'log2', or 'log1p'.")
    if x_marginal_hist_height_ratio <= 0 or y_marginal_hist_width_ratio <= 0:
        raise ValueError("Marginal panel ratios must be positive.")
    xlims_tuple = _normalize_axis_limits(xlims, param_name="xlims", scale=xscale)
    ylims_tuple = _normalize_axis_limits(ylims, param_name="ylims", scale=yscale)
    x_padding = _normalize_padding(xlim_padding_fraction, param_name="xlim_padding_fraction")
    y_padding = _normalize_padding(ylim_padding_fraction, param_name="ylim_padding_fraction")
    normalized_x_reference_lines = _normalize_reference_lines(
        x_reference_lines,
        param_name="x_reference_lines",
    )
    normalized_y_reference_lines = _normalize_reference_lines(
        y_reference_lines,
        param_name="y_reference_lines",
    )

    if df is not None:
        plot_df = df.copy()
    else:
        if obs_df is not None:
            _obs_df = obs_df.copy()
        elif adata is not None:
            _obs_df = adata.obs.copy()
        else:
            raise ValueError("Provide either 'df' or observation information via 'adata'/'obs_df'.")

        if not isinstance(_obs_df, pd.DataFrame):
            _obs_df = pd.DataFrame(_obs_df)

        requested_columns = list(
            dict.fromkeys(
                column
                for column in (column_key_x, column_key_y, hue, subset_key)
                if column is not None
            )
        )
        feature_df: pd.DataFrame | None = None
        if x_df is not None:
            if isinstance(x_df, pd.DataFrame):
                if len(x_df.index) != len(_obs_df.index):
                    raise ValueError(
                        "'x_df' row count must match observation metadata."
                    )
                feature_columns = pd.Index(x_df.columns)
                selected_feature_names = [
                    name
                    for name in feature_columns
                    if name in requested_columns
                    or (
                        name in _obs_df.columns
                        and f"{name}_obs" in requested_columns
                    )
                ]
                if selected_feature_names:
                    feature_df = x_df.loc[:, selected_feature_names].copy()
            else:
                if var_df is not None:
                    feature_columns = pd.Index(var_df.index)
                elif adata is not None:
                    feature_columns = pd.Index(adata.var_names)
                else:
                    raise ValueError(
                        "Provide 'var_df' so that feature columns can be named."
                    )
                matrix = x_df
                matrix_shape = getattr(matrix, "shape", None)
                if matrix_shape is None:
                    matrix = np.asarray(matrix)
                    matrix_shape = matrix.shape
                if len(matrix_shape) != 2:
                    raise ValueError("'x_df' must be a two-dimensional matrix.")
                if matrix_shape[0] != len(_obs_df.index):
                    raise ValueError(
                        "'x_df' row count must match observation metadata."
                    )
                if matrix_shape[1] != len(feature_columns):
                    raise ValueError(
                        "'x_df' column count must match variable metadata."
                    )
                selected_feature_names = [
                    name
                    for name in feature_columns
                    if name in requested_columns
                    or (
                        name in _obs_df.columns
                        and f"{name}_obs" in requested_columns
                    )
                ]
                if selected_feature_names:
                    positions = feature_columns.get_indexer(selected_feature_names)
                    selected_matrix = matrix[:, positions]
                    if hasattr(selected_matrix, "toarray"):
                        selected_matrix = selected_matrix.toarray()
                    feature_df = pd.DataFrame(
                        selected_matrix,
                        index=_obs_df.index,
                        columns=selected_feature_names,
                    )
        elif adata is not None:
            if layer is not None:
                if layer not in adata.layers:
                    raise ValueError(f"Layer '{layer}' not found in adata.layers.")
                matrix = adata.layers[layer]
            else:
                matrix = adata.X
            feature_columns = pd.Index(adata.var_names)
            selected_feature_names = [
                name
                for name in feature_columns
                if name in requested_columns
                or (
                    name in _obs_df.columns
                    and f"{name}_obs" in requested_columns
                )
            ]
            if selected_feature_names:
                positions = feature_columns.get_indexer(selected_feature_names)
                selected_matrix = matrix[:, positions]
                if hasattr(selected_matrix, "toarray"):
                    selected_matrix = selected_matrix.toarray()
                feature_df = pd.DataFrame(
                    selected_matrix,
                    index=adata.obs_names,
                    columns=selected_feature_names,
                )

        if feature_df is not None:
            if feature_df.index is None or not feature_df.index.equals(_obs_df.index):
                feature_df = feature_df.reindex(_obs_df.index)
            colliding_obs_cols = sorted(set(_obs_df.columns).intersection(feature_df.columns))
            if colliding_obs_cols:
                rename_map = {col: f"{col}_obs" for col in colliding_obs_cols}
                rename_targets = list(rename_map.values())
                if len(rename_targets) != len(set(rename_targets)):
                    raise ValueError("Observation column collision renaming produced duplicate column names.")
                existing_obs_cols = set(_obs_df.columns) - set(colliding_obs_cols)
                conflicting_targets = [
                    new_name
                    for new_name in rename_targets
                    if new_name in existing_obs_cols or new_name in feature_df.columns
                ]
                if conflicting_targets:
                    conflicts_str = ", ".join(sorted(conflicting_targets))
                    raise ValueError(
                        "Observation column collision renaming would overwrite existing columns: "
                        f"{conflicts_str}."
                    )
                # Keep feature names stable while making colliding obs metadata addressable.
                _obs_df = _obs_df.rename(columns=rename_map)
            plot_df = pd.concat([_obs_df, feature_df], axis=1)
        else:
            plot_df = _obs_df.copy()

    required_cols = {column_key_x, column_key_y}
    if hue is not None:
        required_cols.add(hue)
    if subset_key is not None:
        required_cols.add(subset_key)
    missing = [col for col in required_cols if col not in plot_df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Column(s) not found in the assembled DataFrame: {missing_str}.")

    selected_cols = [
        col
        for col in (column_key_x, column_key_y, hue, subset_key)
        if col is not None
    ]
    selected_cols = list(dict.fromkeys(selected_cols))
    working_df = plot_df[selected_cols].copy()

    if nas2zeros:
        working_df[column_key_x].fillna(0, inplace=True)
        working_df[column_key_y].fillna(0, inplace=True)

    if dropna:
        working_df = working_df.dropna(subset=[column_key_x, column_key_y])

    if dropzeros:
        x_numeric = pd.to_numeric(working_df[column_key_x], errors="coerce")
        y_numeric = pd.to_numeric(working_df[column_key_y], errors="coerce")
        non_numeric_mask = x_numeric.isna() | y_numeric.isna()
        working_df = working_df.loc[~non_numeric_mask]
        x_numeric = x_numeric.loc[working_df.index]
        y_numeric = y_numeric.loc[working_df.index]
        zero_mask = (x_numeric == 0) | (y_numeric == 0)
        working_df = working_df.loc[~zero_mask]

    if working_df.empty:
        raise ValueError("No data available after filtering missing values.")

    for grouped_col in (hue, subset_key):
        if grouped_col is not None and _is_categorical_series(working_df[grouped_col]):
            working_df[grouped_col] = working_df[grouped_col].cat.remove_unused_categories()

    x_vals = working_df[column_key_x]
    y_vals = working_df[column_key_y]
    x_lower_bound = _scale_lower_bound(xscale)
    if x_lower_bound is not None:
        _validate_scale_values(x_vals, param_name=column_key_x, scale=xscale)
        if xscale in {"log", "log2"} and axes_lines and x_reference_lines is None:
            raise ValueError("'axes_lines=True' would render x=0 on a logarithmic x axis.")
        if any(line["value"] <= x_lower_bound for line in normalized_x_reference_lines):
            raise ValueError(
                f"x reference lines must be greater than {x_lower_bound:g} for '{xscale}'."
            )
    y_lower_bound = _scale_lower_bound(yscale)
    if y_lower_bound is not None:
        _validate_scale_values(y_vals, param_name=column_key_y, scale=yscale)
        if yscale in {"log", "log2"} and axes_lines and y_reference_lines is None:
            raise ValueError("'axes_lines=True' would render y=0 on a logarithmic y axis.")
        if any(line["value"] <= y_lower_bound for line in normalized_y_reference_lines):
            raise ValueError(
                f"y reference lines must be greater than {y_lower_bound:g} for '{yscale}'."
            )
    fit, corr_value, corr_pvalue = _compute_corr_and_fit(x_vals, y_vals, method)
    corr_label = method.capitalize()
    use_data_range_for_fit = xscale != "linear" or yscale != "linear"
    subset_palette_to_use = subset_palette or palette
    subset_series = working_df[subset_key] if subset_key is not None else None
    subset_values: list[Any] = []
    subset_color_map: dict[Any, Any] = {}
    subset_fit_results = []
    fallback_to_all_data = False
    fit_legend_title: str | None = None

    if subset_key is not None:
        non_null_subset = subset_series.dropna()
        if _is_categorical_series(subset_series):
            subset_values = list(subset_series.cat.categories)
        elif pd.api.types.is_numeric_dtype(non_null_subset):
            subset_values = sorted(pd.unique(non_null_subset))
        else:
            subset_values = list(pd.unique(non_null_subset))

        subset_colors = sns.color_palette(
            subset_palette_to_use,
            n_colors=max(len(subset_values), 1),
        )
        subset_color_map = dict(zip(subset_values, subset_colors))
        fallback_to_all_data = not subset_values
        fit_legend_title = f"{subset_key} fit\n{corr_label}_corr"
        if fallback_to_all_data:
            fit_legend_title = f"All data fit\n{corr_label}_corr"

        for subset_value in subset_values:
            group_df = working_df.loc[subset_series == subset_value]
            group_x = group_df[column_key_x]
            group_y = group_df[column_key_y]
            group_fit, group_corr_value, group_corr_pvalue = _try_compute_corr_and_fit(
                group_x,
                group_y,
                method,
            )
            group_coordinates = None
            if group_fit is not None and show_fit:
                group_coordinates = _fit_line_coordinates(
                    group_x,
                    group_fit,
                    xscale,
                    yscale,
                )
            subset_fit_results.append(
                (
                    subset_value,
                    group_x,
                    group_fit,
                    group_corr_value,
                    group_corr_pvalue,
                    f"{subset_key}={subset_value}",
                    group_coordinates,
                )
            )

    overall_fit_coordinates = None
    if show_fit and (
        subset_key is None
        or show_all_obs_fit
        or fallback_to_all_data
    ):
        overall_fit_coordinates = _fit_line_coordinates(
            x_vals,
            fit,
            xscale,
            yscale,
        )

    x_limit_extras: list[float] = []
    y_limit_extras: list[float] = []
    if x_reference_lines is None:
        if axes_lines:
            x_limit_extras.append(0.0)
    else:
        x_limit_extras.extend(line["value"] for line in normalized_x_reference_lines)
    if y_reference_lines is None:
        if axes_lines:
            y_limit_extras.append(0.0)
    else:
        y_limit_extras.extend(line["value"] for line in normalized_y_reference_lines)

    if overall_fit_coordinates is not None:
        x_limit_extras.extend(overall_fit_coordinates[0])
        y_limit_extras.extend(overall_fit_coordinates[1])
    for fit_result in subset_fit_results:
        group_coordinates = fit_result[-1]
        if group_coordinates is not None:
            x_limit_extras.extend(group_coordinates[0])
            y_limit_extras.extend(group_coordinates[1])

    identity_domain_minimum = max(
        (bound for bound in (x_lower_bound, y_lower_bound) if bound is not None),
        default=None,
    )
    identity_data_limits = None
    if show_identity_line and identity_limits == "data":
        combined = np.concatenate(
            [
                pd.to_numeric(x_vals, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(y_vals, errors="coerce").to_numpy(dtype=float),
            ]
        )
        combined = combined[np.isfinite(combined)]
        identity_data_limits = (
            float(combined.min()),
            float(combined.max()),
        )
        if identity_domain_minimum is not None and (
            identity_data_limits[0] <= identity_domain_minimum
            or identity_data_limits[1] <= identity_domain_minimum
        ):
            raise ValueError(
                "Identity-line coordinates must be greater than "
                f"{identity_domain_minimum:g} for the configured scales."
            )

    stats_fontsize = stats_fontsize or title_fontsize
    fit_legend_anchor = _normalize_bbox_to_anchor(
        fit_legend_bbox_to_anchor,
        param_name="fit_legend_bbox_to_anchor",
    )
    hue_legend_anchor = _normalize_bbox_to_anchor(
        hue_legend_bbox_to_anchor,
        param_name="hue_legend_bbox_to_anchor",
    )
    fig = plt.figure(figsize=figsize)
    axes_x_marginal = None
    axes_y_marginal = None

    if show_x_marginal_hist and show_y_marginal_hist:
        grid_spec = fig.add_gridspec(
            2,
            2,
            height_ratios=[x_marginal_hist_height_ratio, 1],
            width_ratios=[1, y_marginal_hist_width_ratio],
            hspace=0.05,
            wspace=0.05,
        )
        axes_x_marginal = fig.add_subplot(grid_spec[0, 0])
        axes = fig.add_subplot(grid_spec[1, 0], sharex=axes_x_marginal)
        axes_y_marginal = fig.add_subplot(grid_spec[1, 1], sharey=axes)
    elif show_x_marginal_hist:
        grid_spec = fig.add_gridspec(
            2,
            1,
            height_ratios=[x_marginal_hist_height_ratio, 1],
            hspace=0.05,
        )
        axes_x_marginal = fig.add_subplot(grid_spec[0, 0])
        axes = fig.add_subplot(grid_spec[1, 0], sharex=axes_x_marginal)
    elif show_y_marginal_hist:
        grid_spec = fig.add_gridspec(
            1,
            2,
            width_ratios=[1, y_marginal_hist_width_ratio],
            wspace=0.05,
        )
        axes = fig.add_subplot(grid_spec[0, 0])
        axes_y_marginal = fig.add_subplot(grid_spec[0, 1], sharey=axes)
    else:
        axes = fig.add_subplot(1, 1, 1)

    scatter_kwargs: dict[str, Any] = {
        "data": working_df,
        "x": column_key_x,
        "y": column_key_y,
        "s": dot_size,
        "ax": axes,
    }
    if hue is not None:
        scatter_kwargs["hue"] = hue
        scatter_kwargs["legend"] = "full" if show_hue_legend else False
        scatter_kwargs["palette"] = palette
    sns.scatterplot(**scatter_kwargs)

    hue_legend = None
    if hue is not None:
        if show_hue_legend:
            legend_kwargs: dict[str, Any] = {
                "loc": 2,
                "borderaxespad": 0.0,
            }
            if show_y_marginal_hist and axes_y_marginal is not None and hue_legend_anchor is None:
                legend_kwargs["bbox_to_anchor"] = (1.04, 1)
                legend_kwargs["bbox_transform"] = axes_y_marginal.transAxes
            else:
                legend_kwargs["bbox_to_anchor"] = hue_legend_anchor or (1.05, 1)
                if hue_legend_anchor is not None:
                    legend_kwargs["bbox_transform"] = axes.transAxes
            if legend_fontsize is not None:
                legend_kwargs["fontsize"] = legend_fontsize
                legend_kwargs["title_fontsize"] = legend_fontsize
            hue_legend = axes.legend(**legend_kwargs)
            hue_legend.set_title(hue, prop={"size": legend_fontsize} if legend_fontsize is not None else None)
        else:
            existing_legend = axes.get_legend()
            if existing_legend is not None:
                existing_legend.remove()

    fit_handles: list[Any] = []
    if subset_key is None:
        if show_fit:
            _plot_fit_line(
                axes,
                x_vals,
                fit,
                show_y_intercept=show_y_intercept,
                color="C0",
                linestyle="-",
                label=None,
                use_data_range=use_data_range_for_fit,
                x_scale=xscale,
                y_scale=yscale,
                coordinates=overall_fit_coordinates,
            )
        stats_text = (
            f"{corr_label} Corr = {corr_value:.3f} pvalue = {corr_pvalue:.6f}\n"
            f"y = {fit.intercept:.3f} + {fit.slope:.3f}x R^2: {fit.rvalue ** 2:.3f}"
        )
    else:
        stats_lines: list[str] = []

        if fallback_to_all_data:
            stats_lines.append(
                f"No valid {subset_key} groups after filtering; showing All data fit."
            )

        if show_all_obs_fit or fallback_to_all_data:
            if show_fit:
                fit_handles.append(
                    _plot_fit_line(
                        axes,
                        x_vals,
                        fit,
                        show_y_intercept=show_y_intercept,
                        color="black",
                        linestyle="--",
                        label=_format_subset_fit_legend_label("All data", corr_value, corr_pvalue),
                        use_data_range=use_data_range_for_fit,
                        x_scale=xscale,
                        y_scale=yscale,
                        coordinates=overall_fit_coordinates,
                    )
                )
            stats_lines.append(
                _format_subset_stats_line("All data", method, fit, corr_value, corr_pvalue)
            )

        for (
            subset_value,
            group_x,
            group_fit,
            group_corr_value,
            group_corr_pvalue,
            group_label,
            group_coordinates,
        ) in subset_fit_results:
            if group_fit is not None and show_fit:
                fit_handles.append(
                    _plot_fit_line(
                        axes,
                        group_x,
                        group_fit,
                        show_y_intercept=show_y_intercept,
                        color=subset_color_map[subset_value],
                        linestyle="-",
                        label=_format_subset_fit_legend_label(
                            group_label,
                            group_corr_value,
                            group_corr_pvalue,
                        ),
                        use_data_range=use_data_range_for_fit,
                        x_scale=xscale,
                        y_scale=yscale,
                        coordinates=group_coordinates,
                    )
                )
            stats_lines.append(
                _format_subset_stats_line(
                    group_label,
                    method,
                    group_fit,
                    group_corr_value,
                    group_corr_pvalue,
                )
            )

        if not stats_lines:
            stats_lines.append(f"No valid {subset_key} groups after filtering.")
        stats_text = "\n".join(stats_lines)

    reference_handles: list[Any] = []
    if axes_lines and y_reference_lines is None:
        axes.axhline(0, color="black")
    if axes_lines and x_reference_lines is None:
        axes.axvline(0, color="black")
    if x_reference_lines is not None:
        reference_handles.extend(
            _draw_reference_lines(
                axes,
                normalized_x_reference_lines,
                axis="x",
                param_name="x_reference_lines",
            )
        )
    if y_reference_lines is not None:
        reference_handles.extend(
            _draw_reference_lines(
                axes,
                normalized_y_reference_lines,
                axis="y",
                param_name="y_reference_lines",
            )
        )

    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    if axes_title is not None:
        title_axes = axes_x_marginal if axes_x_marginal is not None else axes
        axes_title_kwargs: dict[str, Any] = {"fontsize": title_fontsize}
        if axes_title_y is not None:
            axes_title_kwargs["y"] = axes_title_y
        title_axes.set_title(axes_title, **axes_title_kwargs)
    if tick_label_fontsize is not None:
        axes.tick_params(axis="both", labelsize=tick_label_fontsize)

    main_xlim = axes.get_xlim()
    main_ylim = axes.get_ylim()
    subgroup_hist_alpha = 0.35
    all_obs_hist_alpha = 0.20
    all_obs_hist_color = "0.7"

    if axes_x_marginal is not None:
        x_hist_kwargs: dict[str, Any] = {
            "bins": x_marginal_hist_bins,
            "element": "step",
            "fill": x_marginal_hist_fill,
            "kde": x_marginal_hist_KDE,
            "ax": axes_x_marginal,
        }
        if isinstance(x_marginal_hist_bins, int):
            x_hist_kwargs["binrange"] = (x_vals.min(), x_vals.max())

        if subset_key is None:
            sns.histplot(
                data=working_df,
                x=column_key_x,
                color="C0",
                alpha=subgroup_hist_alpha,
                **x_hist_kwargs,
            )
        else:
            if show_all_obs_x_hist or fallback_to_all_data:
                sns.histplot(
                    data=working_df,
                    x=column_key_x,
                    color=all_obs_hist_color,
                    alpha=all_obs_hist_alpha,
                    **x_hist_kwargs,
                )
            for subset_value in subset_values:
                group_df = working_df.loc[subset_series == subset_value]
                if group_df.empty:
                    continue
                sns.histplot(
                    data=group_df,
                    x=column_key_x,
                    color=subset_color_map[subset_value],
                    alpha=subgroup_hist_alpha,
                    **x_hist_kwargs,
                )

        axes_x_marginal.tick_params(axis="x", labelbottom=False)
        axes_x_marginal.tick_params(axis="y", left=False)
        axes_x_marginal.set_xlabel("")
        axes_x_marginal.set_ylabel("")
        axes_x_marginal.set_yticks([])
        if tick_label_fontsize is not None:
            axes_x_marginal.tick_params(axis="both", labelsize=tick_label_fontsize)

    if axes_y_marginal is not None:
        y_hist_kwargs: dict[str, Any] = {
            "bins": y_marginal_hist_bins,
            "element": "step",
            "fill": y_marginal_hist_fill,
            "kde": y_marginal_hist_KDE,
            "ax": axes_y_marginal,
        }
        if isinstance(y_marginal_hist_bins, int):
            y_hist_kwargs["binrange"] = (y_vals.min(), y_vals.max())

        if subset_key is None:
            sns.histplot(
                data=working_df,
                y=column_key_y,
                color="C0",
                alpha=subgroup_hist_alpha,
                **y_hist_kwargs,
            )
        else:
            if show_all_obs_y_hist or fallback_to_all_data:
                sns.histplot(
                    data=working_df,
                    y=column_key_y,
                    color=all_obs_hist_color,
                    alpha=all_obs_hist_alpha,
                    **y_hist_kwargs,
                )
            for subset_value in subset_values:
                group_df = working_df.loc[subset_series == subset_value]
                if group_df.empty:
                    continue
                sns.histplot(
                    data=group_df,
                    y=column_key_y,
                    color=subset_color_map[subset_value],
                    alpha=subgroup_hist_alpha,
                    **y_hist_kwargs,
                )

        axes_y_marginal.tick_params(axis="y", labelleft=False)
        axes_y_marginal.set_ylabel("")
        axes_y_marginal.set_xlabel("Count")
        if tick_label_fontsize is not None:
            axes_y_marginal.tick_params(axis="both", labelsize=tick_label_fontsize)

    if axes_x_marginal is not None or axes_y_marginal is not None:
        axes.set_xlim(main_xlim)
        axes.set_ylim(main_ylim)

    xscale_name = {"log2": "log", "log1p": "function"}.get(xscale, xscale)
    yscale_name = {"log2": "log", "log1p": "function"}.get(yscale, yscale)
    if xscale == "log2":
        xscale_kwargs = {"base": 2}
    elif xscale == "log1p":
        xscale_kwargs = {"functions": (_log1p_forward, np.expm1)}
    else:
        xscale_kwargs = {}
    if yscale == "log2":
        yscale_kwargs = {"base": 2}
    elif yscale == "log1p":
        yscale_kwargs = {"functions": (_log1p_forward, np.expm1)}
    else:
        yscale_kwargs = {}
    axes.set_xscale(xscale_name, **xscale_kwargs)
    axes.set_yscale(yscale_name, **yscale_kwargs)
    if axes_x_marginal is not None:
        axes_x_marginal.set_xscale(xscale_name, **xscale_kwargs)
    if axes_y_marginal is not None:
        axes_y_marginal.set_yscale(yscale_name, **yscale_kwargs)
    resolved_xlim = _resolve_axis_limits(
        x_vals,
        axes.get_xlim(),
        explicit_limits=xlims_tuple,
        padding_fraction=x_padding,
        scale=xscale,
        extra_values=x_limit_extras,
    )
    resolved_ylim = _resolve_axis_limits(
        y_vals,
        axes.get_ylim(),
        explicit_limits=ylims_tuple,
        padding_fraction=y_padding,
        scale=yscale,
        extra_values=y_limit_extras,
    )
    axes.set_xlim(resolved_xlim)
    axes.set_ylim(resolved_ylim)
    if axes_x_marginal is not None:
        axes_x_marginal.set_xlim(resolved_xlim)
    if axes_y_marginal is not None:
        axes_y_marginal.set_ylim(resolved_ylim)

    if show_identity_line:
        if identity_line_style is not None and not isinstance(identity_line_style, _Mapping):
            raise ValueError("'identity_line_style' must be a mapping.")
        if identity_limits == "shared_axes":
            identity_lower = max(resolved_xlim[0], resolved_ylim[0])
            identity_upper = min(resolved_xlim[1], resolved_ylim[1])
            if identity_lower >= identity_upper:
                raise ValueError("The visible x and y ranges do not overlap for the identity line.")
        else:
            identity_lower, identity_upper = identity_data_limits
        if identity_domain_minimum is not None and (
            identity_lower <= identity_domain_minimum
            or identity_upper <= identity_domain_minimum
        ):
            raise ValueError(
                "Identity-line coordinates must be greater than "
                f"{identity_domain_minimum:g} for the configured scales."
            )
        identity_style: dict[str, Any] = {
            "color": "0.3",
            "linestyle": "--",
            "linewidth": 1.5,
            "zorder": 1,
        }
        identity_style.update(dict(identity_line_style or {}))
        identity_style.pop("label", None)
        if identity_line_label is not None:
            identity_style["label"] = identity_line_label
        identity_coordinates = _sample_linear_relation(
            [identity_lower, identity_upper],
            intercept=0.0,
            slope=1.0,
            x_scale=xscale,
            y_scale=yscale,
        )
        (identity_handle,) = axes.plot(*identity_coordinates, **identity_style)
        reference_handles.insert(0, identity_handle)
        axes.set_xlim(resolved_xlim)
        axes.set_ylim(resolved_ylim)

    if axes_x_marginal is None and axes_y_marginal is None:
        fig.tight_layout()

    stats_footer = None
    if show_stats_text:
        stats_footer = fig.text(
            0.5,
            0.01,
            stats_text,
            ha="center",
            va="bottom",
            fontsize=stats_fontsize,
        )

    labeled_reference_handles = [
        handle
        for handle in reference_handles
        if handle.get_label() and not str(handle.get_label()).startswith("_")
    ]
    legend_fit_handles = fit_handles if subset_key is not None and show_fit_legend else []
    combined_legend_handles = legend_fit_handles + labeled_reference_handles
    if combined_legend_handles:
        fit_legend_kwargs: dict[str, Any] = {
            "loc": 2,
            "borderaxespad": 0.0,
            "handlelength": 2.0,
        }
        if show_y_marginal_hist and axes_y_marginal is not None and fit_legend_anchor is None:
            fit_legend_kwargs["bbox_to_anchor"] = (1.04, 1)
            fit_legend_kwargs["bbox_transform"] = axes_y_marginal.transAxes
        else:
            fit_legend_kwargs["bbox_to_anchor"] = fit_legend_anchor or (1.04, 1)
            if fit_legend_anchor is not None:
                fit_legend_kwargs["bbox_transform"] = axes.transAxes
        if hue_legend is not None:
            if hue_legend_anchor is None and show_y_marginal_hist and axes_y_marginal is not None:
                hue_legend.set_bbox_to_anchor(
                    (1.04, 0.55),
                    transform=axes_y_marginal.transAxes,
                )
            else:
                hue_legend.set_bbox_to_anchor(
                    hue_legend_anchor or (1.04, 0.55),
                    transform=axes.transAxes,
                )
            axes.add_artist(hue_legend)
        if legend_fontsize is not None:
            fit_legend_kwargs["fontsize"] = legend_fontsize
            fit_legend_kwargs["title_fontsize"] = legend_fontsize
        fit_legend = axes.legend(
            handles=combined_legend_handles,
            title=fit_legend_title if legend_fit_handles else None,
            **fit_legend_kwargs,
        )
        for legend_handle in fit_legend.legend_handles:
            if hasattr(legend_handle, "set_linewidth"):
                legend_handle.set_linewidth(3.0)

    if stats_footer is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        footer_bbox = stats_footer.get_window_extent(renderer=renderer)
        axes_bbox_y0 = min(
            axis.get_tightbbox(renderer=renderer).y0
            for axis in (axes, axes_x_marginal, axes_y_marginal)
            if axis is not None
        )
        overlap_pixels = footer_bbox.y1 - axes_bbox_y0
        if overlap_pixels >= 0:
            padding_pixels = overlap_pixels + 8
            padding_fraction = padding_pixels / fig.bbox.height
            new_bottom = min(0.99, fig.subplotpars.bottom + padding_fraction)
            fig.subplots_adjust(bottom=new_bottom)

    if show:
        plt.show()
    else:
        # Prevent notebook backends from auto-rendering and retaining figures
        # when callers explicitly requested show=False.
        plt.close(fig)

    if axes_x_marginal is None and axes_y_marginal is None:
        axes_result = axes
    else:
        axes_result = {
            "main": axes,
            "x_marginal": axes_x_marginal,
            "y_marginal": axes_y_marginal,
        }
    return fig, axes_result, fit, corr_value, corr_pvalue


def corr_dotplot_dev(
    df: pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    layer: str | None = None,
    x_df: Any | None = None,
    var_df: pd.DataFrame | None = None,
    obs_df: pd.DataFrame | None = None,
    column_key_x: str | None = None,
    column_key_y: str | None = None,
    hue: str | None = None,
    subset_key: str | None = None,
    figsize: tuple[float, float] = (20, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    axes_title: str | None = None,
    axes_lines: bool = True,
    show_y_intercept: bool = True,
    palette: Sequence[Any] | str | None = palettes.godsnot_102,
    subset_palette: Sequence[Any] | str | None = None,
    dot_size: float = 200,
    title_fontsize: int = 20,
    stats_fontsize: int | None = None,
    axes_title_y: float | None = None,
    axis_label_fontsize: int = 20,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    fit_legend_bbox_to_anchor: Sequence[float] | None = None,
    hue_legend_bbox_to_anchor: Sequence[float] | None = None,
    show_all_obs_fit: bool = False,
    show_fit: bool = True,
    show_fit_legend: bool = True,
    show_hue_legend: bool = True,
    show_stats_text: bool = True,
    show_identity_line: bool = False,
    identity_line_label: str | None = "Identity",
    identity_line_style: _Mapping[str, Any] | None = None,
    identity_limits: Literal["shared_axes", "data"] = "shared_axes",
    nas2zeros: bool = False,
    dropna: bool = False,
    dropzeros: bool = False,
    method: Literal["spearman", "pearson"] = "pearson",
    show_x_marginal_hist: bool = False,
    show_y_marginal_hist: bool = False,
    x_marginal_hist_bins: int | Sequence[float] = 20,
    y_marginal_hist_bins: int | Sequence[float] = 20,
    x_marginal_hist_fill: bool = True,
    x_marginal_hist_KDE: bool = True,
    y_marginal_hist_fill: bool = True,
    y_marginal_hist_KDE: bool = True,
    show_all_obs_x_hist: bool = False,
    show_all_obs_y_hist: bool = False,
    x_marginal_hist_height_ratio: float = 0.18,
    y_marginal_hist_width_ratio: float = 0.18,
    xscale: str = "linear",
    yscale: str = "linear",
    xlims: Sequence[float] | None = None,
    ylims: Sequence[float] | None = None,
    xlim_padding_fraction: float | None = None,
    ylim_padding_fraction: float | None = None,
    x_reference_lines: Sequence[_Mapping[str, Any]] | None = None,
    y_reference_lines: Sequence[_Mapping[str, Any]] | None = None,
    show: bool = True,
):
    """Deprecated compatibility wrapper for :func:`corr_dotplot`."""

    warnings.warn(
        "corr_dotplot_dev() is deprecated; use corr_dotplot() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = corr_dotplot(
        df=df,
        adata=adata,
        layer=layer,
        x_df=x_df,
        var_df=var_df,
        obs_df=obs_df,
        column_key_x=column_key_x,
        column_key_y=column_key_y,
        hue=hue,
        subset_key=subset_key,
        figsize=figsize,
        xlabel=xlabel,
        ylabel=ylabel,
        axes_title=axes_title,
        axes_lines=axes_lines,
        show_y_intercept=show_y_intercept,
        palette=palette,
        subset_palette=subset_palette,
        dot_size=dot_size,
        title_fontsize=title_fontsize,
        stats_fontsize=stats_fontsize,
        axes_title_y=axes_title_y,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        fit_legend_bbox_to_anchor=fit_legend_bbox_to_anchor,
        hue_legend_bbox_to_anchor=hue_legend_bbox_to_anchor,
        show_all_obs_fit=show_all_obs_fit,
        show_fit=show_fit,
        show_fit_legend=show_fit_legend,
        show_hue_legend=show_hue_legend,
        show_stats_text=show_stats_text,
        show_identity_line=show_identity_line,
        identity_line_label=identity_line_label,
        identity_line_style=identity_line_style,
        identity_limits=identity_limits,
        nas2zeros=nas2zeros,
        dropna=dropna,
        dropzeros=dropzeros,
        method=method,
        show_x_marginal_hist=show_x_marginal_hist,
        show_y_marginal_hist=show_y_marginal_hist,
        x_marginal_hist_bins=x_marginal_hist_bins,
        y_marginal_hist_bins=y_marginal_hist_bins,
        x_marginal_hist_fill=x_marginal_hist_fill,
        x_marginal_hist_KDE=x_marginal_hist_KDE,
        y_marginal_hist_fill=y_marginal_hist_fill,
        y_marginal_hist_KDE=y_marginal_hist_KDE,
        show_all_obs_x_hist=show_all_obs_x_hist,
        show_all_obs_y_hist=show_all_obs_y_hist,
        x_marginal_hist_height_ratio=x_marginal_hist_height_ratio,
        y_marginal_hist_width_ratio=y_marginal_hist_width_ratio,
        xscale=xscale,
        yscale=yscale,
        xlims=xlims,
        ylims=ylims,
        xlim_padding_fraction=xlim_padding_fraction,
        ylim_padding_fraction=ylim_padding_fraction,
        x_reference_lines=x_reference_lines,
        y_reference_lines=y_reference_lines,
        show=show,
    )
    fig, axes_result, fit, corr_value, corr_pvalue = result
    if isinstance(axes_result, dict):
        axes_dict = axes_result
    else:
        axes_dict = {
            "main": axes_result,
            "x_marginal": None,
            "y_marginal": None,
        }
    return fig, axes_dict, fit, corr_value, corr_pvalue


def spearman_cor_dotplot(*args, **kwargs):
    """Backward-compatible wrapper around :func:`corr_dotplot` using Spearman correlation."""

    if args:
        if len(args) < 4:
            raise TypeError(
                "spearman_cor_dotplot positional usage requires at least df, column_key_x, column_key_y, and hue."
            )

        df_arg, col_x_arg, col_y_arg, hue, *rest = args

        kwargs.setdefault("df", df_arg)
        kwargs.setdefault("column_key_x", col_x_arg)
        kwargs.setdefault("column_key_y", col_y_arg)
        kwargs.setdefault("hue", hue)

        optional_names = ("figsize", "xlabel", "ylabel", "axes_lines")
        for name, value in zip(optional_names, rest):
            kwargs.setdefault(name, value)

        if len(rest) > len(optional_names):
            raise TypeError("Too many positional arguments provided to spearman_cor_dotplot().")

    kwargs["method"] = "spearman"
    return corr_dotplot(**kwargs)


def spearman_cor_dotplot_2(df, column_key_x, column_key_y, hue, hue_right, figsize=(20, 10)):
    df = df.loc[(df[column_key_x].isna() == False) & (df[column_key_y].isna() == False), :].copy()
    df[hue].cat.remove_unused_categories(inplace=True)
    df[hue_right].cat.remove_unused_categories(inplace=True)

    XY_spearman = df[column_key_x].corr(df[column_key_y], method='spearman')

    figure1, axes = plt.subplots(1, 2, figsize=figsize)
    sns.scatterplot(
        data=df,
        x=column_key_x,
        y=column_key_y,
        hue=hue,
        legend=2,
        s=200,
        palette=palettes.godsnot_102,
        ax=axes[0],
    )
    sns.scatterplot(
        data=df,
        x=column_key_x,
        y=column_key_y,
        hue=hue_right,
        legend=2,
        s=200,
        palette=palettes.godsnot_102,
        ax=axes[1],
    )

    axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.tight_layout()

    fit = linregress(df[column_key_x], df[column_key_y])
    axes[0].axline(xy1=(0, fit.intercept), slope=fit.slope)
    axes[1].axline(xy1=(0, fit.intercept), slope=fit.slope)

    axes[0].axhline(0, color='black')
    axes[0].axvline(0, color='black')
    axes[1].axhline(0, color='black')
    axes[1].axvline(0, color='black')

    figure1.suptitle(
        f"{column_key_y} (Y-axis) and {column_key_x} (X-axis)\nSpearman Correlation = {round(XY_spearman, 3)}\n"
        f"y = {round(fit.intercept, 3)} + {round(fit.slope, 3)}x R-squared: {fit.rvalue ** 2:.6f}",
        y=1.1,
    )

    return figure1, axes


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def plot_rank_scatter(
    list_y,
    list_x,
    extra_title="",
    x_label="Rank in list_y",
    y_label="Rank in list_x",
    figsize=(8, 6),
    show_diagonal=True,
):
    """
    Plots a scatter plot of the ranks of common elements between two lists,
    and includes the Spearman correlation and p-value in the plot title.
    
    Parameters:
      list_y (list): First ranked list (e.g., gene IDs).
      list_x (list): Second ranked list.
      extra_title (str): Additional string to include in the title.
      x_label (str): Label for the x-axis.
      y_label (str): Label for the y-axis.
      figsize (tuple): Size of the figure.
      show_diagonal (bool): When True, draw an x=y reference line.
      
    Returns:
      tuple: (correlation, p_value) computed from the common elements.
    """
    # If lists are provided as tuples (e.g., due to trailing commas), extract the list.
    if isinstance(list_y, tuple):
        list_y = list_y[0]
    if isinstance(list_x, tuple):
        list_x = list_x[0]
    
    # Find common elements between the two lists.
    common = set(list_y) & set(list_x)
    if not common:
        print("No common elements found!")
        return None, None
    
    # Create dictionaries mapping each common element to its rank (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list_y, start=1) if gene in common}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list_x, start=1) if gene in common}
    
    # Sort common elements for consistent ordering.
    common_sorted = sorted(common)
    
    # Build arrays of ranks.
    x_ranks = np.array([rank_dict1[gene] for gene in common_sorted])
    y_ranks = np.array([rank_dict2[gene] for gene in common_sorted])
    
    # Compute Spearman rank correlation.
    corr, p_value = spearmanr(x_ranks, y_ranks)
    
    # Create the scatter plot.
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(x_ranks, y_ranks, color='blue', alpha=0.2)

    # Match axis limits and aspect to keep the plot square.
    min_val = min(x_ranks.min(), y_ranks.min())
    max_val = max(x_ranks.max(), y_ranks.max())
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 1
    lower = min_val - pad
    upper = max_val + pad
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    if show_diagonal:
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{extra_title}\nSpearman Corr: {corr:.3f}, p-value: {p_value:.3e}")
    ax.grid(True)
    plt.show()
    
    return corr, p_value


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

def plot_rank_heatmap(
    list_y,
    list_x,
    extra_title="",
    x_label="Rank in list_y",
    y_label="Rank in list_x",
    gridsize=50,
    figsize=(8, 6),
    show_diagonal=True,
):
    """
    Plots a hexbin heatmap of the ranks of common elements between two lists,
    and includes the Spearman correlation and p-value in the plot title.

    Parameters:
      list_y (list): First ranked list (e.g., gene IDs).
      list_x (list): Second ranked list.
      extra_title (str): Additional title string.
      x_label (str): Label for the x-axis.
      y_label (str): Label for the y-axis.
      gridsize (int): Number of hexagons in the x-direction (affects resolution).
      show_diagonal (bool): When True, draw an x=y reference line.

    Returns:
      tuple: (correlation, p_value) computed from the common elements.
    """
    # If lists are provided as tuples, extract the list.
    if isinstance(list_y, tuple):
        list_y = list_y[0]
    if isinstance(list_x, tuple):
        list_x = list_x[0]
    
    # Find common elements.
    common = set(list_y) & set(list_x)
    if not common:
        print("No common elements found!")
        return None, None
    
    # Create dictionaries mapping each common element to its rank (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list_y, start=1) if gene in common}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list_x, start=1) if gene in common}
    
    # Sort common elements for consistent ordering.
    common_sorted = sorted(common)
    
    # Create arrays of ranks.
    x_ranks = np.array([rank_dict1[gene] for gene in common_sorted])
    y_ranks = np.array([rank_dict2[gene] for gene in common_sorted])
    
    # Compute Spearman rank correlation and p-value.
    corr, p_value = spearmanr(x_ranks, y_ranks)
    
    # Create a hexbin (density) plot.
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    hb = ax.hexbin(x_ranks, y_ranks, gridsize=gridsize, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Density')

    # Match axis limits and aspect to keep the plot square.
    min_val = min(x_ranks.min(), y_ranks.min())
    max_val = max(x_ranks.max(), y_ranks.max())
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 1
    lower = min_val - pad
    upper = max_val + pad
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    if show_diagonal:
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{extra_title}\nSpearman Corr: {corr:.3f}, p-value: {p_value:.3e}")
    ax.grid(True)
    plt.show()
    
    return corr, p_value

'''
# Example usage:
list_y = drug_vehicle_list_nested
list_x = drug_lmm_diff_list
extra_title = "xx hr Comparison"
corr, p_val = plot_rank_heatmap(list_y, list_x, extra_title, 
                                 y_label="Rank in drug_vehicle_list_nested", 
                                 x_label="Rank in drug_lmm_diff_list", 
                                  gridsize=50)
'''

                                  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, gaussian_kde

def plot_rank_scatter_density(
    list_y,
    list_x,
    extra_title="",
    y_label="Rank in list_y",
    x_label="Rank in list_x",
    dot_size=20,
    cmap="viridis",
    figsize=(8, 6),
    show_diagonal=True,
):
    """
    Plots a scatter plot of the ranks of common elements between two lists.
    Each dot is colored based on its local density (computed via a Gaussian KDE).
    The plot title includes the Spearman correlation and p-value.
    
    Parameters:
      list_y (list): First ranked list (e.g., gene IDs). (y-axis)
      list_x (list): Second ranked list.(x-axis)
      extra_title (str): Additional string to include in the title.
      y_label (str): Label for the y-axis.
      x_label (str): Label for the x-axis.
      dot_size (int): Size of the dots.
      cmap (str): Colormap for density.
      show_diagonal (bool): When True, draw an x=y reference line.
      
    Returns:
      tuple: (correlation, p_value) computed from the common elements.
    """
    # If lists are accidentally provided as tuples, extract the list.
    if isinstance(list_y, tuple):
        list_y = list_y[0]
    if isinstance(list_x, tuple):
        list_x = list_x[0]
        
    # Find common elements.
    common = set(list_y) & set(list_x)
    if not common:
        print("No common elements found!")
        return None, None
    
    # Map common genes to their ranks (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list_y, start=1) if gene in common}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list_x, start=1) if gene in common}
    
    # Sort common genes for consistent ordering.
    common_sorted = sorted(common)
    
    # Create arrays of ranks.
    x_ranks = np.array([rank_dict1[gene] for gene in common_sorted])
    y_ranks = np.array([rank_dict2[gene] for gene in common_sorted])
    
    # Compute Spearman rank correlation.
    corr, p_value = spearmanr(x_ranks, y_ranks)
    
    # Compute point density using Gaussian KDE.
    xy = np.vstack([x_ranks, y_ranks])
    z = gaussian_kde(xy)(xy)
    
    # Create the scatter plot with density coloring.
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sc = ax.scatter(x_ranks, y_ranks, c=z, s=dot_size, cmap=cmap)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{extra_title}\nSpearman Corr: {corr:.3f}, p-value: {p_value:.3e}")

    # Match axis limits and aspect to keep the plot square.
    min_val = min(x_ranks.min(), y_ranks.min())
    max_val = max(x_ranks.max(), y_ranks.max())
    span = max_val - min_val
    pad = 0.05 * span if span > 0 else 1
    lower = min_val - pad
    upper = max_val + pad
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    if show_diagonal:
        ax.plot([lower, upper], [lower, upper], color="gray", linestyle="--", linewidth=1)

    fig.colorbar(sc, ax=ax, label="Density")
    ax.grid(True)
    plt.show()
    
    return corr, p_value


'''
# Example usage:
list_y = drug_vehicle_list_nested
list_x = drug_lmm_diff_list
extra_title = "xx hr Comparison"
corr, p_val = plot_rank_scatter_density(list_y, list_x, extra_title, 
                                 y_label="Rank in drug_vehicle_list_nested", 
                                 x_label="Rank in drug_lmm_diff_list", 
                                  dot_size=20)
'''



from scipy.stats import spearmanr


def pairwise_spearman_corr_matrix(lists_dict):
    """
    Build a pairwise Spearman correlation matrix for ranked lists.
    
    Parameters:
      lists_dict (dict): Dictionary where keys are list names and values are ranked lists.
    
    Returns:
      pandas.DataFrame: A DataFrame containing pairwise Spearman correlation coefficients.
    """
    keys = list(lists_dict.keys())
    # Initialize an empty DataFrame with keys as both rows and columns
    corr_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i == j:
                corr_matrix.loc[key1, key2] = 1.0
            elif i < j:
                res = compare_ranked_lists(lists_dict[key1], lists_dict[key2])
                # If res is a tuple (correlation, p_value), extract the first element.
                if isinstance(res, tuple):
                    corr = res[0]
                else:
                    corr = res
                corr_matrix.loc[key1, key2] = corr
                corr_matrix.loc[key2, key1] = corr
    return corr_matrix


from scipy.stats import spearmanr

def compare_ranked_lists(list1, list2):
    """
    Compare two ranked lists using Spearman rank correlation.
    
    Parameters:
      list1, list2 (list): Two lists containing ranked items (e.g. gene IDs).
      
    Returns:
      tuple: (correlation, p_value) from Spearman rank correlation,
             or (None, None) if no common elements.
    """
    # Ensure inputs are actual lists (not tuples)
    if isinstance(list1, tuple):
        list1 = list1[0]
    if isinstance(list2, tuple):
        list2 = list2[0]

    # Find the intersection of the two lists.
    common_genes = set(list1) & set(list2)
    if not common_genes:
        print("No common elements to compare.")
        return None, None

    # Create dictionaries mapping gene to rank (starting at 1)
    rank_dict1 = {gene: rank for rank, gene in enumerate(list1, start=1) if gene in common_genes}
    rank_dict2 = {gene: rank for rank, gene in enumerate(list2, start=1) if gene in common_genes}

    # For consistent ordering, sort the common genes (or you can use any fixed order)
    common_genes_sorted = sorted(common_genes)

    # Create rank arrays
    ranks1 = [rank_dict1[gene] for gene in common_genes_sorted]
    ranks2 = [rank_dict2[gene] for gene in common_genes_sorted]

    # Compute Spearman rank correlation.
    correlation, p_value = spearmanr(ranks1, ranks2)
    return correlation, p_value


import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(
    corr_matrix,
    title: str = "Heatmap",
    vmin: float | None = -1,
    vmax: float | None = 1,
    figsize: tuple[float, float] = (8, 6),
    *,
    cluster: bool = True,
    metric: str = "euclidean",
    method: str = "average",
    annot: bool = True,
    cmap: str = "coolwarm",
    show: bool = True,
):
    """
    Plot a correlation heatmap, optionally clustered by hierarchical linkage.
    
    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix to visualise.
    title : str
        Figure title.
    vmin, vmax : float | None
        Color scale limits forwarded to seaborn.
    figsize : tuple[float, float]
        Figure size.
    cluster : bool
        When True, cluster rows/columns with ``seaborn.clustermap``.
    metric : str
        Distance metric used for clustering when ``cluster`` is True.
    method : str
        Linkage method used for clustering when ``cluster`` is True.
    annot : bool
        Annotate cells with correlation values.
    cmap : str
        Colormap for the heatmap.
    show : bool
        Call ``plt.show()`` before returning.
    
    Returns
    -------
    seaborn.matrix.ClusterGrid | tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        ClusterGrid when clustering, otherwise the figure/axes tuple from ``sns.heatmap``.
    """
    if cluster:
        g = sns.clustermap(
            corr_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            method=method,
            metric=metric,
            linewidths=0.5,
            annot=annot,
            fmt=".2f" if annot else "",
            figsize=figsize,
        )
        g.fig.suptitle(title, y=1.02)
        if show:
            plt.show()
        return g

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, vmin=vmin, vmax=vmax, fmt=".2f" if annot else "", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax

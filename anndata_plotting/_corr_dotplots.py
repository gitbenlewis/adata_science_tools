''' correlation dotplots '''

from collections.abc import Sequence
from typing import Any, Literal

import anndata
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import linregress

from . import palettes


def corr_dotplot(
    df: pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    layer: str | None = None,
    x_df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    obs_df: pd.DataFrame | None = None,
    column_key_x: str | None = None,
    column_key_y: str | None = None,
    hue: str | None = None,
    figsize: tuple[float, float] = (20, 10),
    xlabel: str | None = None,
    ylabel: str | None = None,
    axes_lines: bool = True,
    show_y_intercept: bool = True,
    palette: Sequence[Any] | str | None = palettes.godsnot_102,
    nas2zeros: bool = False,
    dropna: bool = False,
    dropzeros: bool = False,
    method: Literal["spearman", "pearson"] = "pearson",
    show: bool = True,
):
    """Plot a correlation scatter coloured by a grouping column.

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
    x_df : pandas.DataFrame | None
        Expression matrix with observations as rows and features as columns. Overrides
        AnnData-derived matrices when supplied.
    var_df : pandas.DataFrame | None
        Feature metadata used to supply column names when ``x_df`` is not a DataFrame.
    obs_df : pandas.DataFrame | None
        Observation metadata. Overrides ``adata.obs`` when provided.
    column_key_x, column_key_y : str | None
        Keys selecting the x- and y-value columns used for the correlation.
    hue : str | None
        Observation column used for colouring points in the scatter plot. When ``None``,
        points are plotted without grouping or a legend.
    figsize : tuple[float, float]
        Figure size passed to ``plt.subplots``.
    xlabel, ylabel : str | None
        Optional axis labels overriding the default column names.
    axes_lines : bool
        Draw horizontal and vertical reference lines through the origin when ``True``.
    palette : Sequence | str | None
        Palette forwarded to ``seaborn.scatterplot``.
    nas2zeros : bool
        Replace missing x/y values with zeros when ``True``. occurs before / overrides ``dropna``.
    dropna : bool
        Remove observations with missing x/y values when ``True``.
    dropzeros : bool
        Remove observations where either the x or y value equals zero (after numeric coercion).
    method : {"spearman", "pearson"}
        Correlation statistic to report and place in the title.
    show : bool
        Call ``plt.show()`` before returning when ``True``.
    """

    if column_key_x is None or column_key_y is None:
        raise ValueError("Both 'column_key_x' and 'column_key_y' must be provided.")

    method = method.lower()
    if method not in {"spearman", "pearson"}:
        raise ValueError("'method' must be either 'spearman' or 'pearson'.")

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

        feature_df: pd.DataFrame | None
        if x_df is not None:
            feature_df = x_df.copy()
        elif adata is not None:
            if layer is not None:
                if layer not in adata.layers:
                    raise ValueError(f"Layer '{layer}' not found in adata.layers.")
                matrix = adata.layers[layer]
            else:
                matrix = adata.X

            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()

            feature_df = pd.DataFrame(matrix, index=adata.obs_names, columns=adata.var_names)
        else:
            feature_df = None

        if feature_df is not None and not isinstance(feature_df, pd.DataFrame):
            if var_df is not None:
                columns = var_df.index
            elif adata is not None:
                columns = adata.var_names
            else:
                raise ValueError("Provide 'var_df' so that feature columns can be named.")
            feature_df = pd.DataFrame(feature_df, index=_obs_df.index, columns=columns)

        if feature_df is not None:
            if feature_df.index is None or not feature_df.index.equals(_obs_df.index):
                feature_df = feature_df.reindex(_obs_df.index)
            plot_df = pd.concat([_obs_df, feature_df], axis=1)
        else:
            plot_df = _obs_df.copy()

    required_cols = {column_key_x, column_key_y}
    if hue is not None:
        required_cols.add(hue)
    missing = [col for col in required_cols if col not in plot_df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Column(s) not found in the assembled DataFrame: {missing_str}.")

    selected_cols = [column_key_x, column_key_y]
    if hue is not None:
        selected_cols.append(hue)
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

    if hue is not None and pd.api.types.is_categorical_dtype(working_df[hue]):
        working_df[hue] = working_df[hue].cat.remove_unused_categories()

    x_vals = working_df[column_key_x]
    y_vals = working_df[column_key_y]

    if method == "spearman":
        corr_res = stats.spearmanr(x_vals, y_vals)
        corr_value = corr_res.statistic
        corr_pvalue = corr_res.pvalue
    else:
        corr_value, corr_pvalue = stats.pearsonr(x_vals, y_vals)

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    scatter_kwargs: dict[str, Any] = {
        "data": working_df,
        "x": column_key_x,
        "y": column_key_y,
        "s": 200,
        "ax": axes,
    }
    if hue is not None:
        scatter_kwargs["hue"] = hue
        scatter_kwargs["legend"] = "full"
        scatter_kwargs["palette"] = palette
    sns.scatterplot(**scatter_kwargs)

    if hue is not None:
        axes.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0).set_title(hue)
        # add legend name
        #axes.legend().set_title(hue)

    fit = linregress(x_vals, y_vals)
    if show_y_intercept:
        axes.axline(xy1=(0, fit.intercept), slope=fit.slope)
    else:
        x_min, x_max = x_vals.min(), x_vals.max()
        axes.plot(
            [x_min, x_max],
            [fit.intercept + fit.slope * x_min, fit.intercept + fit.slope * x_max],
            color="C0",
        )

    if axes_lines:
        axes.axhline(0, color="black")
        axes.axvline(0, color="black")

    if xlabel is not None:
        axes.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=20)

    fig.tight_layout()

    corr_label = method.capitalize()
    fig.suptitle(
        (
            f"{corr_label} Corr = {corr_value:.3f} pvalue = {corr_pvalue:.6f}\n"
            f"y = {fit.intercept:.3f} + {fit.slope:.3f}x R^2: {fit.rvalue ** 2:.3f}"
        ),
        y=1.05,
        fontsize=20
    )

    if show:
        plt.show()

    return fig, axes


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

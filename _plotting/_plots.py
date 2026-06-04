''' plotting functions for anndata data science tools '''
# module level imports
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import palettes
from ._histograms import _apply_isin_filters

LOGGER = logging.getLogger(__name__)

#######  START  ############. Volcano plots ###################.###################.###################.###################.



# Todo add accept adata or df input
# 2026.01.19 updated to change padj_col to pvalue_col
# 2026.01.19 updated updated to have more font size parameters and a character limit for feature labels
# 2025.08.22 updated to re org hue / label logic and add horizontal pvalue threshold to match updates elsewhere 
# 2025.02.28 updated to used the hue_column to set the hue values
# 2025.11.12 updated to add hue_palette_color_list parameter to accept custom color palettes for hue_column plotting mode
def volcano_plot_generic(
        _df, 
        l2fc_col: str | None = 'log2FoldChange',
        set_xlabel: str | None = 'log2fc model',
        xlimit: str | None = None,
        pvalue_col: str | None = 'pvalue',
        set_ylabel: str | None = '-log10(pvalue)',
        ylimit: str | None = None,
        title_text: str | None = 'volcano_plot',
        comparison_label: str | None = ' Comparison',
        hue_column: str | None = None,
        hue_palette_color_list: list | None = None,
        log2FoldChange_threshold: float | None = .1,
        pvalue_threshold: float | None = None,
        figsize: tuple | None = (15, 10),
        legend_bbox_to_anchor: tuple | None = (1.15, 1),
        title_fontsize: int | None = None,
        axis_label_and_tick_fontsize: int | None = None,
        legend_fontsize: int | None = None,
        label_top_features: bool | None = False,
        only_label_hue_dots: bool | None = True,
        label_top_features_fontsize: int | None = None,
        label_features_char_limit: int | None = 40,
        feature_label_col: str | None = 'gene_names',
        n_top_features: int | None = 50,
        dot_size_shrink_factor: int | None = 300,
        savefig: bool | None = False,
        file_name: str | None = 'volcano_plot.png',
                     ):

    """
    Generate a volcano plot for a single differential expression comparison.

    Creates a volcano plot using Seaborn/Matplotlib with log2 fold change on the x-axis 
    and -log10 adjusted p-value (pvalue) on the y-axis. Data points are categorized by
    significance thresholds (alpha=0.05, 0.1, 0.2) and optionally labeled with top features.

    Preprocessing includes:
      - Replacing missing pvalue values with 1 and computing -log10(pvalue).
      - Adding significance flags at alpha levels (0.2, 0.1, 0.05) with a log2FC cutoff.
      - Combining flags into a categorical "Significance" column for consistent legend order.
      - Clipping extreme values to calculated axis limits (x, y).
      - Optionally labeling top features by significance or fold change extremes.

    Parameters
    ----------
    df : pandas.DataFrame
        Differential expression results with log2 fold change and adjusted p-values.
    l2fc_col : str, optional
        Column for log2 fold change (default 'log2FoldChange').
    pvalue_col : str, optional
        Column for p-values (default 'pvalue').
    set_xlabel, set_ylabel : str, optional
        Axis labels (default 'log2fc model' and '-log10(pvalue)').
    xlimit, ylimit : float, optional
        Axis limits; computed automatically if None.
    title_text, comparison_label : str, optional
        Plot title and comparison label.
    hue_column : str, optional
        Column for coloring points; defaults to "Significance".
    hue_palette_color_list : list, optional 
        List of colors for the hue palette. 
        default uses Paul Tol’s 10-color set + gray
    log2FoldChange_threshold : float, optional
        Minimum absolute log2FC for significance (default 0.1).
    pvalue_threshold : float, optional
        Optional horizontal cutoff line (in p-value scale).
    figsize : tuple, optional
        Figure size (default (15, 10)).
    legend_bbox_to_anchor : tuple, optional
        Legend placement (default (1.15, 1)).
    title_fontsize : int, optional
        Font size for the plot title; if None, use Matplotlib defaults.
    axis_label_and_tick_fontsize : int, optional
        Font size for x/y axis labels and tick labels; if None, use Matplotlib defaults.
    legend_fontsize : int, optional
        Font size for legend text; if None, use Matplotlib defaults.
    label_top_features : bool, optional
        Whether to label top features (default False).
    label_top_features_fontsize : int, optional
        Font size for feature labels; if None, use Matplotlib defaults.
    label_features_char_limit : int, optional
        Max characters to display for feature labels; truncated labels use "..." (default 40).
    feature_label_col : str, optional
        Column used for feature labels (default 'gene_names').
    n_top_features : int, optional
        Number of features to label (default 50).
    dot_size_shrink_factor : int, optional
        Factor to scale dot size by dataset size (default 300).
    savefig : bool, optional
        Save figure to file if True (default False).
    file_name : str, optional
        Output filename for saved plot.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Matplotlib Axes object containing the plot.

    Notes
    -----
    - Vertical dashed lines mark ±log2FoldChange_threshold.
    - A 'Marker' column distinguishes in-range vs. out-of-range points.
    - Two plotting modes: (1) hue by significance or (2) hue by custom column.
    - Out-of-range values are clipped for visualization clarity.
    """

    # -------------------------
    # Imports
    # -------------------------
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -------------------------
    # Define custom color palettes
    # -------------------------
    # Paul Tol’s 10-color set + gray
    tol_colors_w_grey = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77",
        "#661100", "#CC6677", "#882255", "#AA4499",
        "#8D8D8D"
    ]

    # Reorder palettes for significance and hue mapping
    significance_custom_palette = [tol_colors_w_grey[10]] + [tol_colors_w_grey[7]] + [tol_colors_w_grey[0]] + [tol_colors_w_grey[3]]
    
    if hue_palette_color_list is not None:
        hue_palette_custom_palette = hue_palette_color_list
    else:
        hue_palette_custom_palette = [
            tol_colors_w_grey[0], tol_colors_w_grey[3], tol_colors_w_grey[4],
            tol_colors_w_grey[6], tol_colors_w_grey[1], tol_colors_w_grey[8],
            tol_colors_w_grey[7], tol_colors_w_grey[2], tol_colors_w_grey[5],
            tol_colors_w_grey[9]]


    # -------------------------
    # Input data checks and setup
    # -------------------------
    df = _df.copy()
    print(df.shape)

    # If no custom hue column is given, default to "Significance"
    if hue_column is None:
        hue_value = 'Significance'
    else:
        hue_value = hue_column

    # Ensure required columns exist
    required_columns = {l2fc_col, pvalue_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame is missing one of the required columns: {required_columns}")

    # -------------------------
    # Preprocessing: p-value transformations and significance flags
    # -------------------------
    # Fill missing pvalue values with 1 (max nonsignificance)
    df[pvalue_col] = df[pvalue_col].fillna(1)

    # Add -log10(pvalue) column for y-axis
    df['-log10(pvalue)'] = -np.log10(df[pvalue_col].replace(0, np.nextafter(0, 1)))

    # Add boolean columns for multiple alpha thresholds (0.2, 0.1, 0.05)
    df['alpha=0.2'] = ((df[pvalue_col] < 0.2) & (abs(df[l2fc_col]) >= log2FoldChange_threshold))
    df['alpha=0.1'] = ((df[pvalue_col] < 0.1) & (abs(df[l2fc_col]) >= log2FoldChange_threshold))
    df['alpha=0.05'] = ((df[pvalue_col] < 0.05) & (abs(df[l2fc_col]) >= log2FoldChange_threshold))

    # -------------------------
    # Combine significance levels into a single categorical column
    # -------------------------
    df['Significance'] = pd.Categorical(
        ['Not Significant' for _ in range(df.shape[0])],
        categories=['Not Significant', 'alpha=0.2', 'alpha=0.1', 'alpha=0.05'],
        ordered=True
    )
    df.loc[df['alpha=0.2'], 'Significance'] = 'alpha=0.2'
    df.loc[df['alpha=0.1'], 'Significance'] = 'alpha=0.1'
    df.loc[df['alpha=0.05'], 'Significance'] = 'alpha=0.05'

    # Convert pvalue_threshold to -log10 scale if provided
    if pvalue_threshold is not None:
        nlog10_pvalue_threshold = -np.log10(pvalue_threshold)

    # -------------------------
    # Axis limit calculations
    # -------------------------
    # Y-axis limit: 99th percentile of -log10(pvalue) among significant hits
    if not ylimit:
        ylimit = df[(df[pvalue_col] < 0.05) & (df[l2fc_col].abs() > log2FoldChange_threshold)]['-log10(pvalue)'].quantile(0.99)
        if np.isnan(ylimit):
            ylimit = df['-log10(pvalue)'].quantile(0.99)

    # X-axis limit: 99th percentile of absolute log2FC among significant hits
    if not xlimit:
        xlimit = df[(df[pvalue_col] < 0.05) & (df[l2fc_col].abs() > log2FoldChange_threshold)][l2fc_col].abs().quantile(0.99)
        if np.isnan(xlimit):
            xlimit = df[l2fc_col].abs().quantile(0.99)

    # -------------------------
    # Marker column: distinguish in-range vs out-of-range points
    # -------------------------
    df['Marker'] = pd.Categorical(['In_Range' for _ in range(df.shape[0])],
                                  categories=['In_Range', 'Out_of_Range'], ordered=True)
    df.loc[df['-log10(pvalue)'] >= ylimit, 'Marker'] = 'Out_of_Range'
    df.loc[abs(df[l2fc_col]) >= xlimit, 'Marker'] = 'Out_of_Range'

    # Clip values at limits to improve readability
    if ylimit:
        df['-log10(pvalue)'] = df['-log10(pvalue)'].apply(lambda x: (ylimit * 0.99) if x >= ylimit else x)
    else:
        ylimit = df['-log10(pvalue)'].max()

    if xlimit:
        df[l2fc_col] = df[l2fc_col].apply(lambda x: (xlimit * 0.99) if x >= xlimit else x)
        df[l2fc_col] = df[l2fc_col].apply(lambda x: (-xlimit * 0.99) if x <= -xlimit else x)
    else:
        xlimit = max(abs(df[l2fc_col].min()), df[l2fc_col].max())

    # -------------------------
    # Marker size scaling based on dataset size
    # -------------------------
    rel_size = df.shape[0] / dot_size_shrink_factor

    # -------------------------
    # Plotting logic: two modes (with vs. without hue_column)
    # -------------------------
    if hue_column is None:
        # Case 1: hue = Significance
        fig, ax = plt.subplots(figsize=figsize)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(pvalue)', hue=hue_value,
                            style='Marker', palette=significance_custom_palette,
                            sizes=rel_size, s=rel_size, ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n', fontsize=title_fontsize)

        # Add significance threshold lines
        if pvalue_threshold is not None:
            p.axhline(y=nlog10_pvalue_threshold, color='red', linestyle='--', label=f'pvalue<{pvalue_threshold} ')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--', label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')

        # Axis labels + legend
        p.set_xlabel(set_xlabel, fontsize=axis_label_and_tick_fontsize)
        p.set_ylabel(set_ylabel, fontsize=axis_label_and_tick_fontsize)
        if axis_label_and_tick_fontsize is not None:
            p.tick_params(axis="both", labelsize=axis_label_and_tick_fontsize)
        p.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=1, borderaxespad=0.05, fontsize=legend_fontsize)

    elif hue_column is not None:
        # Case 2: custom hue column
        fig, ax = plt.subplots(figsize=figsize)

        # First plot: all dots in gray (background layer)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(pvalue)', style='Marker',
                            color='gray', s=rel_size/2, alpha=0.5, ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.legend_.remove()  # Remove legend from background layer

        # Second plot: overlay hue-colored points
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(pvalue)', hue=hue_value, style='Marker',
                            palette=hue_palette_custom_palette[:], s=rel_size, ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n', fontsize=title_fontsize)

        # Add threshold lines
        if pvalue_threshold is not None:
            p.axhline(y=nlog10_pvalue_threshold, color='red', linestyle='--', label=f'pvalue<{pvalue_threshold} ')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--', label=f'log2fc>|{log2FoldChange_threshold}|')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')

        # Axis labels + legend cleanup
        p.set_xlabel(set_xlabel, fontsize=axis_label_and_tick_fontsize)
        p.set_ylabel(set_ylabel, fontsize=axis_label_and_tick_fontsize)
        if axis_label_and_tick_fontsize is not None:
            p.tick_params(axis="both", labelsize=axis_label_and_tick_fontsize)
        handles = p.get_legend_handles_labels()[0][2:]  # Skip legends from gray layer
        labels = p.get_legend_handles_labels()[1][2:]
        p.legend(handles, labels, bbox_to_anchor=legend_bbox_to_anchor, loc=1, borderaxespad=0.05, fontsize=legend_fontsize)

    # -------------------------
    # Optional: label top features
    # -------------------------
    if label_top_features:
        def _truncate_label(value: object) -> str:
            label = "" if value is None else str(value)
            if label_features_char_limit is None:
                return label
            if label_features_char_limit <= 0:
                return ""
            if len(label) <= label_features_char_limit:
                return label
            if label_features_char_limit <= 3:
                return label[:label_features_char_limit]
            return f"{label[:label_features_char_limit - 3]}..."

        label_kwargs = {"horizontalalignment": "left", "color": "black"}
        if label_top_features_fontsize is not None:
            label_kwargs["size"] = label_top_features_fontsize

        if ((hue_column is not None) and (only_label_hue_dots == True)):
            # Restrict labeling to rows with non-null hue values
            df = df[df[hue_column].notna()].sort_values(by=pvalue_col)

        # Label top genes by pvalue
        for line in range(0, n_top_features):
            p.text(df.sort_values(by=pvalue_col)[l2fc_col].to_list()[line],
                   df.sort_values(by=pvalue_col)['-log10(pvalue)'].to_list()[line],
                   _truncate_label(df.sort_values(by=pvalue_col)[feature_label_col].to_list()[line]),
                   **label_kwargs)

        # Label top genes by most negative log2FC
        for line in range(0, int(n_top_features/2)):
            p.text(df.sort_values(by=l2fc_col)[l2fc_col].to_list()[line],
                   df.sort_values(by=l2fc_col)['-log10(pvalue)'].to_list()[line],
                   _truncate_label(df.sort_values(by=l2fc_col)[feature_label_col].to_list()[line]),
                   **label_kwargs)

        # Label top genes by most positive log2FC
        for line in range(0, int(n_top_features/2)):
            p.text(df.sort_values(by=l2fc_col, ascending=False)[l2fc_col].to_list()[line],
                   df.sort_values(by=l2fc_col, ascending=False)['-log10(pvalue)'].to_list()[line],
                   _truncate_label(df.sort_values(by=l2fc_col, ascending=False)[feature_label_col].to_list()[line]),
                   **label_kwargs)

    # -------------------------
    # Save figure if requested
    # -------------------------
    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {file_name}")

    return p

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
#######  END ############. Volcano plots ###################.###################.###################.###################.

def qqplot(
    data,
    pvalue_column: str | None = None,
    *,
    source: str = "auto",      # "auto" | "var" | "obs" (for AnnData) | "df"
    title: str | None = None,
    pvalue_column_plot_label: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple = (5, 5),
    show: bool = True,
    return_points: bool = False,
    annotate_lambda: bool = True,
    savefig: bool = False,
    filename: str = "qqplot_pvalues.png",
    plotting_position: str = "Blom"  # "Blom" or "Weibull"
):
    """
    QQ plot for p-values (observed vs expected -log10 p).
    for expected p-values uses  (i - 0.5)/n rule, sometimes called “Blom's plotting position”.

    Parameters
    ----------
    data : array-like | pandas.DataFrame | anndata.AnnData
        If array-like: raw p-values.
        If DataFrame: provide `pvalue_column`.
        If AnnData: use `source` ("var" or "obs") + `pvalue_column`.
    pvalue_column : str, optional
        Column name containing p-values when data is a DataFrame or AnnData.
    source : str
        How to read from AnnData ("var" or "obs") or force DataFrame ("df"). "auto" will infer.
    label : str, optional
        Label for axes; defaults to `pvalue_column` or "p-value".
    return_points : bool
        If True, returns (expected, observed) arrays in the output dict.
    annotate_lambda : bool
        If True, computes and annotates genomic inflation factor λ_GC.
    """

    # ---- Extract p-values ----
    if isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        p = np.asarray(data, dtype=float)
        src = "array"
    elif isinstance(data, pd.DataFrame):
        if pvalue_column is None:
            raise ValueError("Provide `pvalue_column` when data is a DataFrame.")
        p = pd.to_numeric(data[pvalue_column], errors="coerce").values
        src = "dataframe"
    else:
        # Possibly AnnData
        if anndata is not None and isinstance(data, anndata.AnnData):
            if pvalue_column is None:
                raise ValueError("Provide `column` when data is an AnnData.")
            # decide source
            if source == "auto":
                source = "var" if pvalue_column in data.var.columns else "obs"
            if source == "var":
                if pvalue_column not in data.var.columns:
                    raise ValueError(f"Column '{pvalue_column}' not found in adata.var.")
                p = pd.to_numeric(data.var[pvalue_column], errors="coerce").values
                src = "adata.var"
            elif source == "obs":
                if pvalue_column not in data.obs.columns:
                    raise ValueError(f"Column '{pvalue_column}' not found in adata.obs.")
                p = pd.to_numeric(data.obs[pvalue_column], errors="coerce").values
                src = "adata.obs"
            else:
                raise ValueError("`source` must be 'var' or 'obs' for AnnData.")
        else:
            raise ValueError("Unsupported `data` type.")
    # ---- Clean p-values ----
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    p = p[(p >= 0) & (p <= 1)]
    if p.size == 0:
        raise ValueError("No valid p-values in [0,1].")
    # protect against zeros
    eps = np.finfo(float).tiny
    p = np.clip(p, eps, 1.0)
    # ---- Sort + expected (plotting positions) ----
    p = np.sort(p)
    n = p.size
    # plotting positions (i - 0.5) / n
    if plotting_position == "Blom":
        exp = (np.arange(1, n + 1) - 0.5) / n
    elif plotting_position == "Weibull":
        exp = np.linspace(1/(n+1), n/(n+1), n)
    else:
        raise ValueError("`plotting_position` must be 'Blom' or 'Weibull'.")
    exp_log = -np.log10(exp)
    obs_log = -np.log10(p)
    # ---- Figure/Axes ----
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
    ax.scatter(exp_log, obs_log, edgecolor='k', s=12, alpha=0.7)
    max_val = max(exp_log.max(), obs_log.max())
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="red", )
    if title is None:
        if pvalue_column_plot_label is not None:
            title = f"QQ plot: {pvalue_column_plot_label}"
        elif pvalue_column is not None:
            title = f"QQ plot: {pvalue_column}"
        else:
            title = "QQ plot of p-values"
    ax.set_title(title)
    if pvalue_column_plot_label is None:
        pvalue_column_plot_label =  "pvalue"
    ax.set_xlabel(f"Expected -log10({pvalue_column_plot_label})")
    ax.set_ylabel(f"Observed -log10({pvalue_column_plot_label})")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    # ---- Optional λ_GC annotation ----
    lam = None
    if annotate_lambda:
        # transform to chi-square with 1 df: chi2 = qchisq(1 - p, 1)
        # Use scipy only if available; otherwise an approximation via inverse survival of chi2
        try:
            from scipy.stats import norm, chi2
            chi_obs = chi2.isf(p, df=1)   # isf = inverse survival function = quantile of 1-p
            lam = np.median(chi_obs) / 0.456  # median of chi2_1 ≈ 0.454936..., commonly rounded to 0.456
            ax.text(0.02, 0.95, f"λ = {lam:.3f}", transform=ax.transAxes, ha="left", va="top")
            #z = norm.isf(p / 2.0)
            #lam_z = np.median(z**2) / 0.456
            #ax.text(0.02, 0.95, f"λ = {lam:.3f} / λ_Z = {lam_z:.3f}", transform=ax.transAxes, ha="left", va="top")
        except Exception as e:
            warnings.warn(f"Could not compute λ: {e}")
    # ---- Save / show ----
    if savefig:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    if created_fig and show:
        plt.tight_layout()
        plt.show()
    # ---- Return ----
    out = {"fig": fig, "ax": ax, "source": src, "n": n}
    if lam is not None:
        out["lambda_gc"] = lam
    if return_points:
        out["expected"] = exp_log
        out["observed"] = obs_log
    return out

####### START ############. datapoint plots ###################.###################.###################.###################.

def paired_datapoints(
    input_data: anndata.AnnData | pd.DataFrame | None = None,
    *,
    adata: anndata.AnnData | None = None,
    df: pd.DataFrame | None = None,
    var_df: pd.DataFrame | None = None,
    var_names: Sequence[str] | None = None,
    var_groupby_key: str | None = None,
    collapse_mode: Literal["stack", "aggregate", "all"] = "aggregate",
    collapse_func: Literal[
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "count",
        "select_max_ref_value",
    ] = "mean",
    layer: str | None = None,
    use_raw: bool = False,
    groupby_key: str = "Pre_or_Post_obs_col",
    groupby_key_target_value: Any = "Post",
    groupby_key_ref_value: Any = "Pre",
    pair_by_key: str | None = None,
    subject_col: str = "Subject_ID",
    ref_values_obsm_key: str | None = None,
    target_values_obsm_key: str | None = None,
    target_min_value: float | None = None,
    target_max_value: float | None = None,
    ref_min_value: float | None = None,
    ref_max_value: float | None = None,
    filter_vars_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    filter_obs_by_isin_lists: Mapping[str, Sequence[Any]] | None = None,
    subset_obs_key: str | None = None,
    subset_order: Sequence[Any] | None = None,
    palette: Sequence[Any] | str | None = palettes.tol_colors,
    subset_palette: Sequence[Any] | str | None = None,
    connect_lines: bool = True,
    line_alpha: float = 0.55,
    line_color: Any = "0.55",
    line_width: float = 0.9,
    line_style: str = "--",
    jitter_amount: float = 0.2,
    random_seed: int | None = 0,
    point_size: float = 80,
    point_alpha: float = 0.85,
    boxplot: bool = True,
    boxplot_width: float = 0.55,
    boxplot_showfliers: bool = False,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    sharey: bool = False,
    ylims: Sequence[float] | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    subplot_title_var_col: str | None = None,
    title_fontsize: int = 14,
    axis_label_fontsize: int = 12,
    tick_label_fontsize: int | None = None,
    legend_fontsize: int | None = None,
    legend: bool = False,
    dropna: bool = True,
    nas2zeros: bool = False,
    dropzeros: bool = False,
    show: bool = True,
    savefig: bool = False,
    file_name: str = "paired_datapoints.png",
    logger: logging.Logger | None = None,
    log_level: int | str | None = None,
    allow_unused_params: bool = False,
    **params: Any,
) -> tuple[plt.Figure, dict[str, plt.Axes], pd.DataFrame]:
    """Plot paired reference and target datapoints from AnnData or a wide DataFrame."""

    log = logger or LOGGER
    if log_level is not None:
        log.setLevel(log_level)

    params = dict(params)
    if "input" in params:
        if input_data is not None:
            raise ValueError("Provide only one of 'input' or 'input_data'.")
        input_data = params.pop("input")
    if params and not allow_unused_params:
        raise ValueError(f"Unused params: {sorted(params)}")

    if input_data is not None:
        if adata is not None or df is not None:
            raise ValueError("Provide 'input_data' or explicit 'adata'/'df', not both.")
        if isinstance(input_data, anndata.AnnData):
            adata = input_data
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("'input_data' must be an AnnData object or pandas DataFrame.")

    if (adata is None) == (df is None):
        raise ValueError("Provide exactly one of 'adata' or 'df'.")
    if use_raw and layer is not None:
        raise ValueError("'layer' cannot be used when use_raw=True.")
    if collapse_mode not in {"stack", "aggregate", "all"}:
        raise ValueError("'collapse_mode' must be one of 'stack', 'aggregate', or 'all'.")
    if collapse_func not in {"mean", "median", "sum", "min", "max", "count", "select_max_ref_value"}:
        raise ValueError(
            "'collapse_func' must be one of 'mean', 'median', 'sum', 'min', 'max', 'count', "
            "or 'select_max_ref_value'."
        )
    if collapse_mode == "all" and var_groupby_key is not None:
        raise ValueError("'collapse_mode=\"all\"' is only supported when 'var_groupby_key' is None.")
    if collapse_func == "select_max_ref_value":
        if adata is None:
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires AnnData input.")
        if var_groupby_key is None:
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires 'var_groupby_key'.")
        if collapse_mode != "aggregate":
            raise ValueError("'collapse_func=\"select_max_ref_value\"' requires collapse_mode=\"aggregate\".")
    if ncols < 1:
        raise ValueError("'ncols' must be at least 1.")
    if ylims is not None:
        ylims_tuple = tuple(ylims)
        if len(ylims_tuple) != 2:
            raise ValueError("'ylims' must contain exactly two values.")
        if ylims_tuple[0] >= ylims_tuple[1]:
            raise ValueError("'ylims' lower bound must be less than upper bound.")
    else:
        ylims_tuple = None

    source_obsm_pair: tuple[str, str] | None = None
    if adata is not None:
        if ref_values_obsm_key is not None or target_values_obsm_key is not None:
            if ref_values_obsm_key is None or target_values_obsm_key is None:
                raise ValueError("Provide both 'ref_values_obsm_key' and 'target_values_obsm_key'.")
            if ref_values_obsm_key not in adata.obsm:
                raise ValueError(f"Reference values obsm '{ref_values_obsm_key}' not found in adata.obsm.")
            if target_values_obsm_key not in adata.obsm:
                raise ValueError(f"Target values obsm '{target_values_obsm_key}' not found in adata.obsm.")
            source_obsm_pair = (ref_values_obsm_key, target_values_obsm_key)
        else:
            looks_like_ref_vs_target = "ref_vs_target_adata" in adata.uns or groupby_key not in adata.obs.columns
            if looks_like_ref_vs_target:
                for candidate_pair in (("pre_values", "post_values"), ("pre", "post"), ("ref_values", "target_values")):
                    if candidate_pair[0] in adata.obsm and candidate_pair[1] in adata.obsm:
                        source_obsm_pair = candidate_pair
                        break

    using_source_obsm = source_obsm_pair is not None
    if using_source_obsm:
        log.info(
            "Using paired source values from adata.obsm[%r] and adata.obsm[%r].",
            source_obsm_pair[0],
            source_obsm_pair[1],
        )

    if adata is not None:
        obs_metadata_df = adata.obs.copy()
        if using_source_obsm:
            if use_raw:
                raise ValueError("use_raw=True is not supported when plotting paired source-value obsm arrays.")
            var_metadata_df = adata.var.copy()
            matrix = None
            matrix_var_names = pd.Index(adata.var_names)
        elif use_raw:
            if adata.raw is None:
                raise ValueError("use_raw=True but adata.raw is None.")
            matrix = adata.raw.X
            var_metadata_df = adata.raw.var.copy()
            matrix_var_names = pd.Index(adata.raw.var_names)
        else:
            if layer is not None:
                if layer not in adata.layers:
                    raise ValueError(f"Layer '{layer}' not found in adata.layers.")
                matrix = adata.layers[layer]
            else:
                matrix = adata.X
            var_metadata_df = adata.var.copy()
            matrix_var_names = pd.Index(adata.var_names)
    else:
        obs_metadata_df = df
        matrix = None
        matrix_var_names = pd.Index(df.columns)
        if var_df is None:
            if var_groupby_key is not None:
                raise ValueError("For df input with 'var_groupby_key', provide 'var_df'.")
            if var_names is None:
                raise ValueError("For df input, provide 'var_names' or 'var_df'.")
            var_metadata_df = pd.DataFrame(index=pd.Index(var_names))
        else:
            var_metadata_df = var_df.copy()

    has_var_groups = var_groupby_key is not None
    has_all_vars_panel = collapse_mode == "all"
    if has_var_groups:
        if var_groupby_key not in var_metadata_df.columns:
            raise ValueError(f"Column '{var_groupby_key}' not found in variable metadata.")
        if subplot_title_var_col is not None:
            raise ValueError("'subplot_title_var_col' is not supported with 'var_groupby_key'.")
    elif has_all_vars_panel and subplot_title_var_col is not None:
        raise ValueError("'subplot_title_var_col' is not supported with 'collapse_mode=\"all\"'.")
    elif subplot_title_var_col is not None and subplot_title_var_col not in var_metadata_df.columns:
        raise ValueError(f"Column '{subplot_title_var_col}' not found in variable metadata.")

    obs_mask = _apply_isin_filters(
        obs_metadata_df,
        filter_obs_by_isin_lists,
        frame_label="observation metadata",
        param_name="filter_obs_by_isin_lists",
    )
    filtered_obs_df = obs_metadata_df.loc[obs_mask].copy()
    if filtered_obs_df.empty:
        raise ValueError("No observations remain after filtering.")
    if subset_obs_key is not None and subset_obs_key not in filtered_obs_df.columns:
        raise ValueError(f"Column '{subset_obs_key}' not found in observation metadata.")

    var_filter_mask = _apply_isin_filters(
        var_metadata_df,
        filter_vars_by_isin_lists,
        frame_label="variable metadata",
        param_name="filter_vars_by_isin_lists",
    )

    group_to_variant_names: dict[Any, list[str]] = {}
    if has_var_groups:
        filtered_var_metadata_df = var_metadata_df.loc[var_filter_mask].copy()
        missing_vars = [name for name in filtered_var_metadata_df.index if name not in matrix_var_names]
        if missing_vars:
            raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
        filtered_var_metadata_df = filtered_var_metadata_df.loc[
            filtered_var_metadata_df[var_groupby_key].notna()
        ]
        group_values = filtered_var_metadata_df[var_groupby_key]
        if var_names is None:
            selected_panel_names = list(pd.unique(group_values))
        else:
            selected_panel_names = list(var_names)
            observed_group_values = set(group_values)
            missing_groups = [name for name in selected_panel_names if name not in observed_group_values]
            if missing_groups:
                raise ValueError(f"Variable group(s) not found after filtering: {missing_groups}.")
        for group_name in selected_panel_names:
            group_to_variant_names[group_name] = list(
                filtered_var_metadata_df.index[group_values == group_name]
            )
    else:
        candidate_var_names = list(var_names) if var_names is not None else list(var_metadata_df.index)
        missing_vars = [name for name in candidate_var_names if name not in matrix_var_names]
        if missing_vars:
            raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
        missing_metadata_vars = [name for name in candidate_var_names if name not in var_metadata_df.index]
        if missing_metadata_vars:
            raise ValueError(f"Variable(s) not found in variable metadata: {missing_metadata_vars}.")
        selected_var_names = [name for name in candidate_var_names if bool(var_filter_mask.loc[name])]
        if has_all_vars_panel:
            selected_panel_names = ["all"]
            group_to_variant_names["all"] = selected_var_names
        else:
            selected_panel_names = selected_var_names
            for var_name in selected_var_names:
                group_to_variant_names[var_name] = [var_name]

    if not selected_panel_names:
        if has_var_groups:
            raise ValueError("No variable groups remain after filtering.")
        raise ValueError("No variables remain after filtering.")

    selected_raw_var_names: list[str] = []
    seen_raw_vars: set[str] = set()
    for panel_name in selected_panel_names:
        for raw_var_name in group_to_variant_names[panel_name]:
            if raw_var_name not in seen_raw_vars:
                selected_raw_var_names.append(raw_var_name)
                seen_raw_vars.add(raw_var_name)
    if not selected_raw_var_names:
        raise ValueError("No variables remain after filtering.")

    effective_pair_key = pair_by_key or subject_col

    def _matrix_to_frame(obs_index: pd.Index, raw_var_names: Sequence[str]) -> pd.DataFrame:
        if df is not None:
            values_df = df.loc[obs_index, list(raw_var_names)]
        else:
            obs_positions = adata.obs_names.get_indexer(obs_index)
            if (obs_positions < 0).any():
                missing_obs = list(obs_index[obs_positions < 0])
                raise ValueError(f"Observation(s) not found in AnnData input: {missing_obs}.")
            var_positions = matrix_var_names.get_indexer(raw_var_names)
            if (var_positions < 0).any():
                missing_vars = [raw_var_names[idx] for idx, pos in enumerate(var_positions) if pos < 0]
                raise ValueError(f"Variable(s) not found in input data: {missing_vars}.")
            values = matrix[obs_positions, :][:, var_positions]
            if hasattr(values, "toarray"):
                values = values.toarray()
            values_df = pd.DataFrame(
                np.asarray(values),
                index=obs_index,
                columns=list(raw_var_names),
            )
        if values_df.ndim == 1:
            values_df = values_df.to_frame()
        return values_df.apply(pd.to_numeric, errors="coerce")

    def _obsm_to_frame(obsm_key: str, obs_index: pd.Index, raw_var_names: Sequence[str]) -> pd.DataFrame:
        values = adata.obsm[obsm_key]
        if isinstance(values, pd.DataFrame):
            missing_obs = [obs_name for obs_name in obs_index if obs_name not in values.index]
            if missing_obs:
                raise ValueError(f"obsm '{obsm_key}' is missing observation(s): {missing_obs}.")
            missing_vars = [var_name for var_name in raw_var_names if var_name not in values.columns]
            if missing_vars:
                raise ValueError(f"obsm '{obsm_key}' is missing variable(s): {missing_vars}.")
            values_df = values.loc[obs_index, list(raw_var_names)]
        else:
            values_array = values if hasattr(values, "shape") else np.asarray(values)
            expected_shape = (adata.n_obs, len(matrix_var_names))
            if len(values_array.shape) != 2 or values_array.shape != expected_shape:
                raise ValueError(f"obsm '{obsm_key}' must have shape {expected_shape} when it is not a DataFrame.")
            obs_positions = adata.obs_names.get_indexer(obs_index)
            var_positions = matrix_var_names.get_indexer(raw_var_names)
            values_subset = values_array[obs_positions, :][:, var_positions]
            if hasattr(values_subset, "toarray"):
                values_subset = values_subset.toarray()
            values_df = pd.DataFrame(
                np.asarray(values_subset),
                index=obs_index,
                columns=list(raw_var_names),
            )
        return values_df.apply(pd.to_numeric, errors="coerce")

    def _aggregate_values(values_df: pd.DataFrame) -> pd.Series:
        if collapse_func == "mean":
            return values_df.mean(axis=1, skipna=True)
        if collapse_func == "median":
            return values_df.median(axis=1, skipna=True)
        if collapse_func == "sum":
            return values_df.sum(axis=1, skipna=True, min_count=1)
        if collapse_func == "min":
            return values_df.min(axis=1, skipna=True)
        if collapse_func == "max":
            return values_df.max(axis=1, skipna=True)
        if collapse_func == "count":
            return values_df.count(axis=1)
        raise ValueError("'select_max_ref_value' is handled separately.")

    if using_source_obsm:
        if pair_by_key is not None:
            if effective_pair_key not in filtered_obs_df.columns:
                raise ValueError(f"Column '{effective_pair_key}' not found in observation metadata.")
            pair_ids = filtered_obs_df[effective_pair_key].astype(str)
        elif effective_pair_key in filtered_obs_df.columns:
            pair_ids = filtered_obs_df[effective_pair_key].astype(str)
        else:
            pair_ids = pd.Series(filtered_obs_df.index.astype(str), index=filtered_obs_df.index)
        duplicate_pair_ids = sorted(pair_ids[pair_ids.duplicated(keep=False)].unique())
        if duplicate_pair_ids:
            raise ValueError(f"Duplicate pair IDs found in source-value observations: {duplicate_pair_ids}.")
        pair_order = list(pair_ids)
        pair_obs_index = filtered_obs_df.index
        ref_values_df = _obsm_to_frame(source_obsm_pair[0], pair_obs_index, selected_raw_var_names)
        target_values_df = _obsm_to_frame(source_obsm_pair[1], pair_obs_index, selected_raw_var_names)
        ref_values_df.index = pair_order
        target_values_df.index = pair_order
        pair_index = pd.Index(pair_order, name=effective_pair_key if effective_pair_key in filtered_obs_df.columns else None)
        if subset_obs_key is None:
            subset_values_by_pair = pd.Series(index=pair_index, dtype=object)
        else:
            subset_values_by_pair = pd.Series(
                filtered_obs_df[subset_obs_key].to_numpy(),
                index=pair_index,
            )
    else:
        if groupby_key not in filtered_obs_df.columns:
            raise ValueError(f"Column '{groupby_key}' not found in observation metadata.")
        if effective_pair_key not in filtered_obs_df.columns:
            raise ValueError(f"Column '{effective_pair_key}' not found in observation metadata.")
        ref_obs = filtered_obs_df.loc[filtered_obs_df[groupby_key] == groupby_key_ref_value].copy()
        target_obs = filtered_obs_df.loc[filtered_obs_df[groupby_key] == groupby_key_target_value].copy()
        if ref_obs.empty or target_obs.empty:
            raise ValueError("Reference and target groups must each contain at least one observation.")

        ref_missing_obs = ref_obs.index[ref_obs[effective_pair_key].isna()].tolist()
        target_missing_obs = target_obs.index[target_obs[effective_pair_key].isna()].tolist()
        if ref_missing_obs or target_missing_obs:
            log.warning(
                "Dropping observations with missing pair IDs: ref=%s, target=%s.",
                ref_missing_obs,
                target_missing_obs,
            )
            ref_obs = ref_obs.loc[ref_obs[effective_pair_key].notna()].copy()
            target_obs = target_obs.loc[target_obs[effective_pair_key].notna()].copy()

        ref_pair_ids = ref_obs[effective_pair_key].astype(str)
        target_pair_ids = target_obs[effective_pair_key].astype(str)
        ref_duplicate_ids = sorted(ref_pair_ids[ref_pair_ids.duplicated(keep=False)].unique())
        target_duplicate_ids = sorted(target_pair_ids[target_pair_ids.duplicated(keep=False)].unique())
        if ref_duplicate_ids or target_duplicate_ids:
            raise ValueError(
                "Duplicate pair IDs found in selected observations: "
                f"ref={ref_duplicate_ids}, target={target_duplicate_ids}."
            )

        ref_pair_id_set = set(ref_pair_ids.tolist())
        target_pair_id_set = set(target_pair_ids.tolist())
        pair_order = sorted(ref_pair_id_set.intersection(target_pair_id_set))
        if not pair_order:
            raise ValueError("No complete ref/target pairs remain after filtering.")

        dropped_ref_only_pair_ids = sorted(ref_pair_id_set.difference(target_pair_id_set))
        dropped_target_only_pair_ids = sorted(target_pair_id_set.difference(ref_pair_id_set))
        if dropped_ref_only_pair_ids or dropped_target_only_pair_ids:
            log.warning(
                "Dropping incomplete pair IDs: ref_only=%s, target_only=%s.",
                dropped_ref_only_pair_ids,
                dropped_target_only_pair_ids,
            )

        ref_obs_name_by_pair = pd.Series(ref_obs.index.to_numpy(), index=ref_pair_ids.to_numpy())
        target_obs_name_by_pair = pd.Series(target_obs.index.to_numpy(), index=target_pair_ids.to_numpy())
        ref_obs_index = pd.Index(ref_obs_name_by_pair.loc[pair_order].to_numpy())
        target_obs_index = pd.Index(target_obs_name_by_pair.loc[pair_order].to_numpy())
        ref_values_df = _matrix_to_frame(ref_obs_index, selected_raw_var_names)
        target_values_df = _matrix_to_frame(target_obs_index, selected_raw_var_names)
        ref_values_df.index = pair_order
        target_values_df.index = pair_order
        pair_index = pd.Index(pair_order, name=effective_pair_key)
        if subset_obs_key is None:
            subset_values_by_pair = pd.Series(index=pair_index, dtype=object)
        else:
            subset_values_by_pair = pd.Series(
                target_obs.loc[target_obs_index, subset_obs_key].to_numpy(),
                index=pair_index,
            )

    if ref_min_value is not None or ref_max_value is not None:
        ref_values_df = ref_values_df.clip(lower=ref_min_value, upper=ref_max_value)
    if target_min_value is not None or target_max_value is not None:
        target_values_df = target_values_df.clip(lower=target_min_value, upper=target_max_value)

    records: list[dict[str, Any]] = []
    ref_label = str(groupby_key_ref_value)
    target_label = str(groupby_key_target_value)

    def _append_pair_records(
        *,
        panel_name: Any,
        variable_name: Any,
        pair_id: Any,
        ref_value: Any,
        target_value: Any,
        source_variable: Any,
        line_id: str,
        subset_value: Any,
    ) -> None:
        panel_label = str(panel_name)
        variable_label = str(variable_name)
        records.append(
            {
                "panel": panel_label,
                "variable": variable_label,
                "source_variable": source_variable,
                "pair_id": str(pair_id),
                "x_label": ref_label,
                "x_order": 1,
                "value": ref_value,
                "line_id": line_id,
                "side": "ref",
                "subset_value": subset_value,
            }
        )
        records.append(
            {
                "panel": panel_label,
                "variable": variable_label,
                "source_variable": source_variable,
                "pair_id": str(pair_id),
                "x_label": target_label,
                "x_order": 2,
                "value": target_value,
                "line_id": line_id,
                "side": "target",
                "subset_value": subset_value,
            }
        )

    if connect_lines and collapse_mode in {"stack", "all"}:
        log.info("Connecting paired lines by pair ID and source variable for collapse_mode=%r.", collapse_mode)

    for panel_name in selected_panel_names:
        panel_raw_vars = group_to_variant_names[panel_name]
        panel_label = str(panel_name)
        if has_all_vars_panel or collapse_mode == "stack":
            for raw_var_name in panel_raw_vars:
                for pair_id in pair_index:
                    _append_pair_records(
                        panel_name=panel_label,
                        variable_name=panel_label,
                        pair_id=pair_id,
                        ref_value=ref_values_df.loc[pair_id, raw_var_name],
                        target_value=target_values_df.loc[pair_id, raw_var_name],
                        source_variable=raw_var_name,
                        line_id=f"{panel_label}|{pair_id}|{raw_var_name}",
                        subset_value=subset_values_by_pair.loc[pair_id] if subset_obs_key is not None else pd.NA,
                    )
        elif has_var_groups:
            ref_panel_values_df = ref_values_df.loc[:, panel_raw_vars]
            target_panel_values_df = target_values_df.loc[:, panel_raw_vars]
            if collapse_func == "select_max_ref_value":
                ref_values_matrix = ref_panel_values_df.to_numpy(dtype=float, copy=True)
                target_values_matrix = target_panel_values_df.to_numpy(dtype=float, copy=True)
                valid_ref_values = ~np.isnan(ref_values_matrix)
                has_ref_value = valid_ref_values.any(axis=1)
                ref_selection_matrix = np.where(valid_ref_values, ref_values_matrix, -np.inf)
                selected_variant_positions = np.argmax(ref_selection_matrix, axis=1)
                max_ref_values = ref_selection_matrix[
                    np.arange(ref_selection_matrix.shape[0]),
                    selected_variant_positions,
                ]
                tie_counts = (
                    (ref_selection_matrix == max_ref_values[:, None])
                    & valid_ref_values
                ).sum(axis=1)
                tied_pair_count = int(((tie_counts > 1) & has_ref_value).sum())
                if tied_pair_count:
                    log.warning(
                        "select_max_ref_value found tied maximum ref values for %d pair(s) "
                        "in panel '%s'; using the first variable in filtered variable order.",
                        tied_pair_count,
                        panel_label,
                    )
                for row_idx, pair_id in enumerate(pair_index):
                    if not has_ref_value[row_idx]:
                        selected_source_variable = pd.NA
                        ref_value = np.nan
                        target_value = np.nan
                    else:
                        selected_position = selected_variant_positions[row_idx]
                        selected_source_variable = panel_raw_vars[selected_position]
                        ref_value = ref_values_matrix[row_idx, selected_position]
                        target_value = target_values_matrix[row_idx, selected_position]
                    _append_pair_records(
                        panel_name=panel_label,
                        variable_name=panel_label,
                        pair_id=pair_id,
                        ref_value=ref_value,
                        target_value=target_value,
                        source_variable=selected_source_variable,
                        line_id=f"{panel_label}|{pair_id}",
                        subset_value=subset_values_by_pair.loc[pair_id] if subset_obs_key is not None else pd.NA,
                    )
            else:
                ref_values = _aggregate_values(ref_panel_values_df)
                target_values = _aggregate_values(target_panel_values_df)
                for pair_id in pair_index:
                    _append_pair_records(
                        panel_name=panel_label,
                        variable_name=panel_label,
                        pair_id=pair_id,
                        ref_value=ref_values.loc[pair_id],
                        target_value=target_values.loc[pair_id],
                        source_variable=pd.NA,
                        line_id=f"{panel_label}|{pair_id}",
                        subset_value=subset_values_by_pair.loc[pair_id] if subset_obs_key is not None else pd.NA,
                    )
        else:
            raw_var_name = panel_raw_vars[0]
            for pair_id in pair_index:
                _append_pair_records(
                    panel_name=panel_label,
                    variable_name=panel_label,
                    pair_id=pair_id,
                    ref_value=ref_values_df.loc[pair_id, raw_var_name],
                    target_value=target_values_df.loc[pair_id, raw_var_name],
                    source_variable=raw_var_name,
                    line_id=f"{panel_label}|{pair_id}",
                    subset_value=subset_values_by_pair.loc[pair_id] if subset_obs_key is not None else pd.NA,
                )

    plot_df = pd.DataFrame.from_records(records)
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    if subset_obs_key is not None:
        plot_df[subset_obs_key] = plot_df["subset_value"]
    if nas2zeros:
        plot_df["value"] = plot_df["value"].fillna(0)
    if dropna:
        plot_df = plot_df.dropna(subset=["value"])
    if dropzeros:
        plot_df = plot_df.loc[plot_df["value"] != 0]
    if plot_df.empty:
        raise ValueError("No paired datapoints remain after value filtering.")

    plot_panel_names = [str(panel_name) for panel_name in selected_panel_names]
    plot_ncols = min(ncols, len(plot_panel_names))
    plot_nrows = math.ceil(len(plot_panel_names) / plot_ncols)
    if figsize is None:
        figsize = (4.5 * plot_ncols, 4.0 * plot_nrows)

    fig, axes_array = plt.subplots(plot_nrows, plot_ncols, figsize=figsize, squeeze=False, sharey=sharey)
    axes_flat = axes_array.ravel()
    axes_by_panel: dict[str, plt.Axes] = {}
    rng = np.random.default_rng(random_seed)
    plot_df = plot_df.copy()
    plot_df["_jittered_x"] = plot_df["x_order"] + rng.uniform(
        -jitter_amount,
        jitter_amount,
        len(plot_df),
    )

    subset_hue_order: list[Any] = []
    subset_palette_map: dict[Any, Any] | None = None
    if subset_obs_key is not None:
        subset_values = plot_df[subset_obs_key].dropna()
        if subset_order is not None:
            observed_subset_values = set(subset_values)
            subset_hue_order = [value for value in subset_order if value in observed_subset_values]
        elif isinstance(filtered_obs_df[subset_obs_key].dtype, pd.CategoricalDtype):
            subset_hue_order = list(
                filtered_obs_df[subset_obs_key]
                .cat.remove_unused_categories()
                .cat.categories
            )
            subset_hue_order = [value for value in subset_hue_order if value in set(subset_values)]
        else:
            subset_hue_order = list(pd.unique(subset_values))
        subset_palette_to_use = subset_palette or palette
        if subset_palette_to_use is not None and subset_hue_order:
            if isinstance(subset_palette_to_use, str):
                subset_colors = sns.color_palette(subset_palette_to_use, n_colors=len(subset_hue_order))
            else:
                subset_colors = list(subset_palette_to_use)
                if not subset_colors:
                    raise ValueError("'subset_palette'/'palette' cannot be an empty sequence.")
            subset_palette_map = {
                subset_value: subset_colors[idx % len(subset_colors)]
                for idx, subset_value in enumerate(subset_hue_order)
            }

    if palette is None:
        default_point_color = "black"
    elif isinstance(palette, str):
        default_point_color = sns.color_palette(palette, n_colors=1)[0]
    else:
        palette_colors = list(palette)
        if not palette_colors:
            raise ValueError("'palette' cannot be an empty sequence.")
        default_point_color = palette_colors[0]

    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize)

    for plot_idx, panel_name in enumerate(plot_panel_names):
        ax = axes_flat[plot_idx]
        axes_by_panel[panel_name] = ax
        panel_df = plot_df.loc[plot_df["panel"] == panel_name].copy()
        if panel_df.empty:
            ax.text(0.5, 0.5, "No plottable values", ha="center", va="center", transform=ax.transAxes)
            continue

        if boxplot:
            box_values = [
                panel_df.loc[panel_df["x_order"] == x_order, "value"].dropna().to_numpy()
                for x_order in (1, 2)
            ]
            if any(len(values) for values in box_values):
                boxplot_artists = ax.boxplot(
                    box_values,
                    positions=[1, 2],
                    patch_artist=False,
                    showfliers=boxplot_showfliers,
                    widths=boxplot_width,
                )
                for element in ("boxes", "medians", "whiskers"):
                    for item in boxplot_artists[element]:
                        item.set(color="black", linewidth=0.75)
                for cap in boxplot_artists["caps"]:
                    cap.set_visible(False)

        if connect_lines:
            for _, line_df in panel_df.groupby("line_id", sort=False):
                if set(line_df["x_order"]) >= {1, 2}:
                    line_df = line_df.sort_values("x_order")
                    ax.plot(
                        line_df["_jittered_x"],
                        line_df["value"],
                        color=line_color,
                        linestyle=line_style,
                        linewidth=line_width,
                        alpha=line_alpha,
                        zorder=1,
                    )

        if subset_obs_key is None:
            ax.scatter(
                panel_df["_jittered_x"],
                panel_df["value"],
                color=default_point_color,
                s=point_size,
                alpha=point_alpha,
                zorder=2,
            )
        else:
            for subset_value in subset_hue_order:
                subset_df = panel_df.loc[panel_df[subset_obs_key] == subset_value]
                if subset_df.empty:
                    continue
                color = subset_palette_map.get(subset_value) if subset_palette_map is not None else None
                ax.scatter(
                    subset_df["_jittered_x"],
                    subset_df["value"],
                    color=color,
                    s=point_size,
                    alpha=point_alpha,
                    label=str(subset_value),
                    zorder=2,
                )
            missing_subset_df = panel_df.loc[panel_df[subset_obs_key].isna()]
            if not missing_subset_df.empty:
                ax.scatter(
                    missing_subset_df["_jittered_x"],
                    missing_subset_df["value"],
                    color="black",
                    s=point_size,
                    alpha=point_alpha,
                    zorder=2,
                )

        if subplot_title_var_col is not None and panel_name in var_metadata_df.index:
            panel_title = str(var_metadata_df.loc[panel_name, subplot_title_var_col])
        else:
            panel_title = panel_name
        ax.set_title(panel_title)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([ref_label, target_label], rotation=45, ha="right")
        ax.set_xlabel(groupby_key, fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel or "value", fontsize=axis_label_fontsize)
        if tick_label_fontsize is not None:
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        if ylims_tuple is not None:
            ax.set_ylim(ylims_tuple)
        if legend and subset_obs_key is not None and subset_hue_order:
            ax.legend(title=subset_obs_key, fontsize=legend_fontsize)

    for ax in axes_flat[len(plot_panel_names):]:
        ax.set_visible(False)

    plt.tight_layout()
    if savefig:
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    plot_df = plot_df.drop(columns=["_jittered_x"])
    return fig, axes_by_panel, plot_df


def timeseries_paired_datapoints(
    adata,
    feature_name,
    x_col='TimePoint',
    feature_name_label_col=None,
    layer='norm',
    Hue='Treatment_unique',
    subplotby=None,
    analyte_label='analyte_Level',
    savefig=False,
    file_name='test',
    pvalue_label1='paired-ttest',
    pvalue_col_in_var1=None,
    pvalue_label2=None,
    pvalue_col_in_var2=None,
    pvalue_label3=None,
    pvalue_col_in_var3=None,
    pvalue_label4=None,
    pvalue_col_in_var4=None,
    pvalue_label5=None,
    pvalue_col_in_var5=None,
    pvalue_label6=None,
    pvalue_col_in_var6=None,
    pvalue_label7=None,
    pvalue_col_in_var7=None,
    pvalue_label8=None,
    pvalue_col_in_var8=None,
    pvalue_label9=None,
    pvalue_col_in_var9=None,
    pvalue_label10=None,
    pvalue_col_in_var10=None,
    pvalue_label11=None,
    pvalue_col_in_var11=None,
    pvalue_label12=None,
    pvalue_col_in_var12=None,
    pvalue_label13=None,
    pvalue_col_in_var13=None,
    pvalue_label14=None,
    pvalue_col_in_var14=None,
    pvalue_label15=None,
    pvalue_col_in_var15=None,
    pvalue_label16=None,
    pvalue_col_in_var16=None,

    subject_col='Subject_ID',
    connect_lines=True,
    jitter_amount=0.2,
    legend=False,
    figsize=(10, 6),
    color_list=["#88CCEE", "#AA4499", "#117733", "#44AA99", "#332288", "#999933", "#DDCC77", "#661100", "#CC6677", "#882255"],
    jump_n_colors=0,
):
    """
    Plots `x_col` vs. a single feature's intensity (RFU) from an AnnData object.
    
    If `subplotby` is provided, the plot is split into two vertical subplots.
    Otherwise, a single plot is generated.
    Allows optional jittering and connecting lines between repeated measures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    import numpy as np
    import pandas.api.types as ptypes

    # Validate feature presence
    if feature_name not in adata.var_names:
        raise ValueError(f"Feature '{feature_name}' not found in adata.var_names.")

    df = adata.obs.copy()

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")

    # Extract feature data
    feature_idx = adata.var_names.get_loc(feature_name)
    df[analyte_label] = adata.layers[layer][:, feature_idx].ravel()

    # Extract display name for feature
    if feature_name_label_col and feature_name_label_col in adata.var.columns:
        feature_name_label = str(adata.var.loc[feature_name, feature_name_label_col])[:40]
    else:
        feature_name_label = feature_name

    # Extract p-values if provided
    pvalue_strs = []
    for pval_col, pval_label in [
        (pvalue_col_in_var1, pvalue_label1),
        (pvalue_col_in_var2, pvalue_label2),
        (pvalue_col_in_var3, pvalue_label3),
        (pvalue_col_in_var4, pvalue_label4),
        (pvalue_col_in_var5, pvalue_label5),
        (pvalue_col_in_var6, pvalue_label6),
        (pvalue_col_in_var7, pvalue_label7),
        (pvalue_col_in_var8, pvalue_label8),
         (pvalue_col_in_var9, pvalue_label9),
         (pvalue_col_in_var10, pvalue_label10),
         (pvalue_col_in_var11, pvalue_label11),
         (pvalue_col_in_var12, pvalue_label12),
         (pvalue_col_in_var13, pvalue_label13),
         (pvalue_col_in_var14, pvalue_label14),
         (pvalue_col_in_var15, pvalue_label15),
         (pvalue_col_in_var16, pvalue_label16),

    ]:
        if pval_col and pval_col in adata.var.columns:
            pval_val = adata.var.loc[feature_name, pval_col]
            if pval_val is not None:
                pvalue_strs.append(f"{pval_label} {pval_val:.2e}")

    # Handle cases where subplotby is None
    if subplotby is not None:
        if subplotby not in df.columns:
            raise ValueError(f"Column '{subplotby}' not found in adata.obs.")

        unique_groups = df[subplotby].unique()
        #if len(unique_groups) != 2:
        #    raise ValueError(f"'{subplotby}' must have exactly 2 unique values. Found: {unique_groups}")

        #group1, group2 = sorted(unique_groups)
        nrows = len(unique_groups)
    else:
        nrows = 1  # Single plot case

    # Color mapping for Hue
    if Hue not in df.columns:
        raise ValueError(f"Column '{Hue}' not found in adata.obs.")
    unique_hue_vals = df[Hue].cat.categories
    cmap = mcolors.ListedColormap(color_list,name='tol_cmap')
    hue_color_dict = {val: cmap(i) for i, val in enumerate(unique_hue_vals)}
    hue_color_dict = {}
    for i, val in enumerate(unique_hue_vals):
        # i % len(color_list) ensures we wrap around when i >= 10
        color_idx = (i + jump_n_colors) % len(color_list)
        hue_color_dict[val] = cmap.colors[color_idx]

    # Preserve categorical order for x_col and convert to string
    if ptypes.is_categorical_dtype(adata.obs[x_col]):
        xvals = list(map(str, adata.obs[x_col].cat.categories))  # Use categorical order
    else:
        xvals = sorted(map(str, df[x_col].unique()))  # Default: sorted unique values

    df[x_col] = df[x_col].astype(str)  # Ensure x_col is a string before mapping
    x_map = {v: i for i, v in enumerate(xvals, start=1)}  # Assign numeric values

    # Initialize figure
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True, sharey=True)
    if nrows == 1:
        axes = [axes]  # Convert to list for consistent handling

    # Title with p-values
    title =  f"{feature_name_label}\n {x_col} vs. {analyte_label}"
    fig.suptitle(title, fontsize=16)


    # add p values below plot
    #combineed_pvalue_strs = "\n".join(pvalue_strs)
    # make twp pvalue strings appear on the same line
    combined_lines = []
    for i in range(0, len(pvalue_strs), 2):
        # If there is a pair, combine them
        if i+1 < len(pvalue_strs):
            combined_lines.append(pvalue_strs[i] + "  " + pvalue_strs[i+1])
        # Otherwise (odd number of strings), just add the last one alone
        else:
            combined_lines.append(pvalue_strs[i])
    # Now join them with a newline for display
    combined_pvalue_strs = "\n".join(combined_lines)
    if len(combined_lines) <= 4:
        bottom_text_y_position = -0.08
    elif len(combined_lines) > 4:
        bottom_text_y_position = -0.15
    fig.text(
        0.,            # x-position in figure coordinates (0.0 left, 1.0 right)
        bottom_text_y_position,          # y-position in figure coordinates (0.0 bottom, 1.0 top)
        combined_pvalue_strs, 
        ha="left",    # center horizontally around x=0.5
        fontsize=12,
    )

    # Helper function to draw a plot
    def draw_plot(ax, group_val=None):
        """Inner function to draw the boxplot + scatter plot for a given group."""
        if group_val is not None:
            group_df = df[df[subplotby] == group_val].copy()
        else:
            group_df = df.copy()  # For subplotby=None, use all data

        data_for_boxplot = [group_df.loc[group_df[x_col] == xval, analyte_label].values for xval in xvals]

        # Ensure x_col is a string before mapping
        group_df[x_col] = group_df[x_col].astype(str)

        # Assign jittered x-values
        group_df["JITTERED_X"] = group_df[x_col].map(x_map) + np.random.uniform(-jitter_amount, jitter_amount, len(group_df))

        # Draw boxplot
        bp = ax.boxplot(
            data_for_boxplot,
            positions=range(1, len(xvals) + 1),
            patch_artist=False,
            showfliers=False,
            widths=0.6
        )

        for element in ["boxes", "medians", "whiskers"]:
            for item in bp[element]:
                item.set(color="black", linewidth=0.75)
        for cap in bp["caps"]:
            cap.set_visible(False)

        
        # Optionally connect points with the same subject
        #if connect_lines and subject_col in group_df.columns:
        #    for subj_id, sdata in group_df.groupby(subject_col):
        #        sdata = sdata.sort_values(by=x_col)
        #       xi_vals = sdata["JITTERED_X"].values
        #       yi_vals = sdata["METABOLITE_LEVEL"].values
        #       ax.plot(xi_vals, yi_vals, color="gray", linestyle="--", alpha=0.6)
        
        # Optionally connect points with the same subject
        if connect_lines and subject_col in group_df.columns:
            for subj_id, sdata in group_df.groupby(subject_col):
                sdata = sdata.copy()  # Avoid modifying original data

                # Ensure x_col exists and drop NaNs
                if x_col not in sdata.columns:
                    continue  # Skip if column is missing
                sdata = sdata.dropna(subset=[x_col])

                # Map x_col to numeric values, using reindex to handle missing keys safely
                sdata["_X_NUM"] = sdata[x_col].map(x_map).dropna()

                # Sort by the numeric x_col order
                sdata = sdata.sort_values(by="_X_NUM")

                # Ensure we have at least two points to draw a line
                if len(sdata) > 1:
                    xi_vals = sdata["JITTERED_X"].values
                    yi_vals = sdata[analyte_label].values
                    ax.plot(xi_vals, yi_vals, color="gray", linestyle="--", alpha=0.6)

        # Scatter plot with jitter
        ax.scatter(
            group_df["JITTERED_X"], 
            group_df[analyte_label], 
            color=[hue_color_dict.get(v, "black") for v in group_df[Hue]], 
            s=120, 
            alpha=0.85
        )
        ax.set_ylabel(analyte_label)
        ax.set_xticks(range(1, len(xvals) + 1))
        ax.set_xticklabels(xvals, rotation=45, ha="right")  # Preserve original order

    # Draw plots correctly, ensuring each subplot only gets its respective data
    if subplotby is None:
        draw_plot(axes[0])  # No subplotting, use all data
        axes[0].set_xlabel(x_col)
    else:
        for index, subplot_group in enumerate(unique_groups):
            draw_plot(axes[index], subplot_group)
            axes[index].set_title(f"{subplotby} = {subplot_group}", fontsize=12)
        axes[len(unique_groups)-1].set_xlabel(x_col)
    # Legend
    if legend:
        # Create legend elements preserving order
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label=str(val), markerfacecolor=hue_color_dict[val], markersize=8)
            for val in unique_hue_vals ]
        axes[0].legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            title=Hue
        )

    plt.tight_layout()
    #plt.tight_layout(rect=[0, 0.12, 1, 1])
    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight"
                    )
        print(f"Saved plot to {file_name}")
    plt.show()
    plt.close(fig)

####### END ############. datapoint plots ###################.###################.###################.###################.

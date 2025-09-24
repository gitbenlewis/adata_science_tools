''' plotting functions for anndata data science tools '''
# module level imports
import matplotlib.pyplot as plt



#######  START  ############. Volcano plots ###################.###################.###################.###################.



# Todo add accept adata or df input
# 2025.08.22 updated to re org hue / label logic and add horizontal pvalue threshold to match updates elsewhere 
# 2025.02.28 updated to used the hue_column to set the hue values 
def volcano_plot_generic(
        _df, 
        l2fc_col: str | None = 'log2FoldChange',
        set_xlabel: str | None = 'log2fc model',
        xlimit: str | None = None,
        padj_col: str | None = 'padj',
        set_ylabel: str | None = '-log10(padj)',
        ylimit: str | None = None,
        title_text: str | None = 'volcano_plot',
        comparison_label: str | None = ' Comparison',
        hue_column: str | None = None,
        log2FoldChange_threshold: float | None = .1,
        pvalue_threshold: float | None = None,
        figsize: tuple | None = (15, 10),
        legend_bbox_to_anchor: tuple | None = (1.15, 1),
        label_top_features: bool | None = False,
        only_label_hue_dots: bool | None = True,
        feature_label_col: str | None = 'gene_names',
        n_top_features: int | None = 50,
        dot_size_shrink_factor: int | None = 300,
        savefig: bool | None = False,
        file_name: str | None = 'volcano_plot.png',
                     ):

    """
    Generate a volcano plot for a single differential expression comparison.

    Creates a volcano plot using Seaborn/Matplotlib with log2 fold change on the x-axis 
    and -log10 adjusted p-value (padj) on the y-axis. Data points are categorized by 
    significance thresholds (alpha=0.05, 0.1, 0.2) and optionally labeled with top features.

    Preprocessing includes:
      - Replacing missing padj values with 1 and computing -log10(padj).
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
    padj_col : str, optional
        Column for adjusted p-values (default 'padj').
    set_xlabel, set_ylabel : str, optional
        Axis labels (default 'log2fc model' and '-log10(padj)').
    xlimit, ylimit : float, optional
        Axis limits; computed automatically if None.
    title_text, comparison_label : str, optional
        Plot title and comparison label.
    hue_column : str, optional
        Column for coloring points; defaults to "Significance".
    log2FoldChange_threshold : float, optional
        Minimum absolute log2FC for significance (default 0.1).
    pvalue_threshold : float, optional
        Optional horizontal cutoff line (in p-value scale).
    figsize : tuple, optional
        Figure size (default (15, 10)).
    legend_bbox_to_anchor : tuple, optional
        Legend placement (default (1.15, 1)).
    label_top_features : bool, optional
        Whether to label top features (default False).
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
    hue_palette_custom_palette = [
        tol_colors_w_grey[0], tol_colors_w_grey[3], tol_colors_w_grey[4],
        tol_colors_w_grey[6], tol_colors_w_grey[1], tol_colors_w_grey[8],
        tol_colors_w_grey[7], tol_colors_w_grey[2], tol_colors_w_grey[5],
        tol_colors_w_grey[9]
    ]

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
    required_columns = {l2fc_col, padj_col}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame is missing one of the required columns: {required_columns}")

    # -------------------------
    # Preprocessing: p-value transformations and significance flags
    # -------------------------
    # Fill missing padj values with 1 (max nonsignificance)
    df[padj_col] = df[padj_col].fillna(1)

    # Add -log10(padj) column for y-axis
    df['-log10(padj)'] = -np.log10(df[padj_col].replace(0, np.nextafter(0, 1)))

    # Add boolean columns for multiple alpha thresholds (0.2, 0.1, 0.05)
    df['alpha=0.2'] = ((df[padj_col] < 0.2) & (abs(df[l2fc_col]) >= log2FoldChange_threshold))
    df['alpha=0.1'] = ((df[padj_col] < 0.1) & (abs(df[l2fc_col]) >= log2FoldChange_threshold))
    df['alpha=0.05'] = ((df[padj_col] < 0.05) & (abs(df[l2fc_col]) >= log2FoldChange_threshold))

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
    # Y-axis limit: 99th percentile of -log10(padj) among significant hits
    if not ylimit:
        ylimit = df[(df[padj_col] < 0.05) & (df[l2fc_col].abs() > log2FoldChange_threshold)]['-log10(padj)'].quantile(0.99)
        if np.isnan(ylimit):
            ylimit = df['-log10(padj)'].quantile(0.99)

    # X-axis limit: 99th percentile of absolute log2FC among significant hits
    if not xlimit:
        xlimit = df[(df[padj_col] < 0.05) & (df[l2fc_col].abs() > log2FoldChange_threshold)][l2fc_col].abs().quantile(0.99)
        if np.isnan(xlimit):
            xlimit = df[l2fc_col].abs().quantile(0.99)

    # -------------------------
    # Marker column: distinguish in-range vs out-of-range points
    # -------------------------
    df['Marker'] = pd.Categorical(['In_Range' for _ in range(df.shape[0])],
                                  categories=['In_Range', 'Out_of_Range'], ordered=True)
    df.loc[df['-log10(padj)'] >= ylimit, 'Marker'] = 'Out_of_Range'
    df.loc[abs(df[l2fc_col]) >= xlimit, 'Marker'] = 'Out_of_Range'

    # Clip values at limits to improve readability
    if ylimit:
        df['-log10(padj)'] = df['-log10(padj)'].apply(lambda x: (ylimit * 0.99) if x >= ylimit else x)
    else:
        ylimit = df['-log10(padj)'].max()

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
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', hue=hue_value,
                            style='Marker', palette=significance_custom_palette,
                            sizes=rel_size, s=rel_size, ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')

        # Add significance threshold lines
        if pvalue_threshold is not None:
            p.axhline(y=nlog10_pvalue_threshold, color='red', linestyle='--', label=f'pvalue<{pvalue_threshold} ')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--', label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')

        # Axis labels + legend
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=1, borderaxespad=0.05)

    elif hue_column is not None:
        # Case 2: custom hue column
        fig, ax = plt.subplots(figsize=figsize)

        # First plot: all dots in gray (background layer)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', style='Marker',
                            color='gray', s=rel_size/2, alpha=0.5, ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.legend_.remove()  # Remove legend from background layer

        # Second plot: overlay hue-colored points
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', hue=hue_value, style='Marker',
                            palette=hue_palette_custom_palette[:], s=rel_size, ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')

        # Add threshold lines
        if pvalue_threshold is not None:
            p.axhline(y=nlog10_pvalue_threshold, color='red', linestyle='--', label=f'pvalue<{pvalue_threshold} ')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--', label=f'log2fc>|{log2FoldChange_threshold}|')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')

        # Axis labels + legend cleanup
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        handles = p.get_legend_handles_labels()[0][2:]  # Skip legends from gray layer
        labels = p.get_legend_handles_labels()[1][2:]
        p.legend(handles, labels, bbox_to_anchor=legend_bbox_to_anchor, loc=1, borderaxespad=0.05)

    # -------------------------
    # Optional: label top features
    # -------------------------
    if label_top_features:
        if ((hue_column is not None) and (only_label_hue_dots == True)):
            # Restrict labeling to rows with non-null hue values
            df = df[df[hue_column].notna()].sort_values(by=padj_col)

        # Label top genes by padj
        for line in range(0, n_top_features):
            p.text(df.sort_values(by=padj_col)[l2fc_col].to_list()[line],
                   df.sort_values(by=padj_col)['-log10(padj)'].to_list()[line],
                   df.sort_values(by=padj_col)[feature_label_col].to_list()[line],
                   horizontalalignment='left', size='small', color='black')

        # Label top genes by most negative log2FC
        for line in range(0, int(n_top_features/2)):
            p.text(df.sort_values(by=l2fc_col)[l2fc_col].to_list()[line],
                   df.sort_values(by=l2fc_col)['-log10(padj)'].to_list()[line],
                   df.sort_values(by=l2fc_col)[feature_label_col].to_list()[line],
                   horizontalalignment='left', size='small', color='black')

        # Label top genes by most positive log2FC
        for line in range(0, int(n_top_features/2)):
            p.text(df.sort_values(by=l2fc_col, ascending=False)[l2fc_col].to_list()[line],
                   df.sort_values(by=l2fc_col, ascending=False)['-log10(padj)'].to_list()[line],
                   df.sort_values(by=l2fc_col, ascending=False)[feature_label_col].to_list()[line],
                   horizontalalignment='left', size='small', color='black')

    # -------------------------
    # Save figure if requested
    # -------------------------
    if savefig:
        plt.savefig(file_name, dpi=600, bbox_inches="tight")
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

####### END ############. datapoint plots ###################.###################.###################.###################.

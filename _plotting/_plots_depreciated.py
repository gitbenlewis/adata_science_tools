import matplotlib.pyplot as plt






#######  START  ############. Volcano plots ###################.###################.###################.###################.



# Todo add accept adata or df input
# 2025.08.22 updated to re org hue / label logic and add horizontal pvalue threshold to match updates elsewhere 
# 2025.02.28 updated to used the hue_column to set the hue values 
def volcano_plot_sns_single_comparison_generic(
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
try:
    import anndata as ad  # optional
except Exception:
    ad = None

def qqplot_pvalues(
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

#######  END ############. Volcano plots ###################.###################.###################.###################.

####### START ############. datapoint plots ###################.###################.###################.###################.

def plot_paired_point_anndata(
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




import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata  # or use the quoted type hint instead
from matplotlib.patches import Patch
import anndata 
import numpy as np
from matplotlib.ticker import StrMethodFormatter

def plot_column_of_bar_h_2groups_GEX_adata(
        adata: anndata.AnnData | None = None,
        layer: str | None = 'salmon_effective_TPM',
        x_df: pd.DataFrame | None = None,       
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list=None,
        feature_label_vars_col: str | None ='SeqIdEntrezGeneSymbol',
        feature_label_char_limit: int | None= 25,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int] = (10, 30),
        fig_title: str | None = None,
        fig_title_y: float | None = .99,
        fig_title_fontsize: int | None = 30,
        feature_label_fontsize: int | None= 24,
        tick_label_fontsize: int | None= 20,
        legend_fontsize: int | None= 24,
        tight_layout_rect_arg=[0, .05, 1, .99],
        comparison_col: str | None = 'Treatment',
        remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
        subplot_xlabel: str | None = 'Expression (TPM)',
        sharex: bool = False,
        legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        savefig: bool = False,
        file_name: str = 'test_plot.png'):
    
    ############ prep input tables / parse adata ############
    if feature_list is None:
        raise ValueError("feature_list must be provided.") 
    if adata is not None:
        print(f"AnnData object provideed with shape {adata.shape} and {len(adata.var_names)} features.")
        # if adata is provided, use it to get the data
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        print(f"Using provided x_df with shape {x_df.shape}")
        _x_df = x_df.copy()
    elif layer is None:
        print("No layer provided, using adata.X with shape {adata.X.shape}")
        _x_df = adata.X.copy()  # use the raw data if no layer
    elif adata is not None and layer in adata.layers:
        print(f"No x_df provided, using adata.layers['{layer}'] with shape {adata.layers[layer].shape}")
        _x_df = adata.layers[layer].copy()

    if var_df is not None:
        print(f"Using provided var_df with shape {var_df.shape}")
        _var_df = var_df.copy()
    else:
        print(f"No var_df provided, using adata.var with shape {adata.var.shape}")
        _var_df = adata.var.copy()

    if obs_df is not None:
        print(f"Using provided obs_df with shape {obs_df.shape}")
        _obs_df = obs_df.copy()
    else:
        print(f"No obs_df provided, using adata.obs with shape {adata.obs.shape}")
        _obs_df = adata.obs.copy()

    # #) make df_obs_x, which is a tidy df with obs + expression columns
    if hasattr(_x_df, "toarray"):  # Convert sparse matrix to dense if necessary
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)

    # Build feature labels for subplot y-labels
    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _bar_feature_label_series = _var_df[feature_label_vars_col]
        _bar_feature_label_series = _bar_feature_label_series.where(
            _bar_feature_label_series.notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        if feature_label_vars_col is not None and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found in var_df; using index for labels.")
        _bar_feature_label_series = _var_df.index.to_series().astype(str)

    if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
        _bar_feature_label_series = _bar_feature_label_series.str.slice(0, int(feature_label_char_limit))
    _bar_feature_label_map = _bar_feature_label_series.to_dict()


    # Determine category order
    if comparison_order is None:
        # keep observed order
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)

    # Build a fixed palette used for every subplot
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))

    gene_list_len = len(feature_list)
    fig, axes = plt.subplots(
        gene_list_len, 1,
        sharex=sharex, 
        figsize=figsize, 
    )
    if gene_list_len == 1:
        axes = [axes]  # make iterable

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=fig_title_fontsize, y=fig_title_y )
    else:
        fig.suptitle(f"{subplot_xlabel} grouped by {comparison_col}\n", fontsize=fig_title_fontsize, y=fig_title_y)

    for plot_num, gene in enumerate(feature_list):
        ax = axes[plot_num]

        # Horizontal bars (aggregated by category)
        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax,
            hue=comparison_col, # Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect
            legend=False,
            palette=[color_map[c] for c in categories]
        )

        if remove_yticklabels:
            ax.set_yticklabels([])

        # Overlay points (each sample), same order as bars
        sns.stripplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax,
            color='black',
            legend=False
        )
        # set x-axis tic fontsize
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        # remove xlabel for all but the last subplot
        ax.set_xlabel('')
        # set ylabel for each subplot using mapped feature label
        _bar_feat_label = _bar_feature_label_map.get(gene, str(gene))
        ax.set_ylabel(_bar_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax.yaxis.set_label_coords(feature_label_x, 0.5)

        
    # outside of the loop, set the xlabel for the last subplot
    ax.set_xlabel(subplot_xlabel, fontsize=legend_fontsize)

    # Figure-level legend at bottom with the same bar colors
    if legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        fig.legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )
        # Leave space for the bottom legend
        rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.01, 0, 0])).tolist()
        plt.tight_layout(rect=rect_used)
    else:
        plt.tight_layout(rect=tight_layout_rect_arg)
    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight" )
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, axes

'''
# example usage
### input parameters
adata_layer='salmon_effective_TPM' # layer to use for the expression data
sharex=False
figsize=(10,30)
comparison_col='Treatment' 
comparison_order=None # use default order
figure_plot_title=f'Top 10 Upregulated Genes \nGrouped by {comparison_col}' # title for the figure
subplot_xlabel='Expression (TPM)' # label for the feature plot
figsize=(8,16)

print(top_up_genes)


adtl.plot_column_of_bar_h_2groups_GEX_adata(adata, feature_list=top_up_genes, layer=adata_layer, comparison_col=comparison_col,
                                         figure_plot_title=figure_plot_title, subplot_xlabel=subplot_xlabel,
                                        legend=True, legend_fontsize=20, remove_yticklabels=True,
                                        sharex=sharex, figsize=figsize)
'''

####### END ############. datapoint plots ###################.###################.###################.###################.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata  # or use the quoted type hint instead
from matplotlib.patches import Patch
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

import anndata 
def plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata(
        # shared parameters
        adata: anndata.AnnData | None = None,
        layer: str | None = 'salmon_effective_TPM',
        x_df: pd.DataFrame | None = None,       
        var_df: pd.DataFrame | None = None,
        obs_df: pd.DataFrame | None = None,
        feature_list: list[str] | None = None, # index of adata
        feature_label_vars_col: str | None = None, # if None than adata index used to label
        feature_label_char_limit: int | None = 40,
        feature_label_x: float = -0.02,
        figsize: tuple[int, int]| None = (10, 15),
        fig_title: str | None = None,
        fig_title_y: float = 1.03,
        subfig_title_y: float = 99,
        fig_title_fontsize: int | None = 30,
        subfig_title_fontsize: int | None = 24,
        feature_label_fontsize: int | None= 24,
        tick_label_fontsize: int | None= 20,
        legend_fontsize: int | None= 24,
        bar2dotplot_width_ratios: list[float] | None = [1.5, 1.],
        tight_layout_rect_arg: list[float] | None = [0, 0, 1, 1],
        savefig: bool = False,
        file_name: str = 'test_plot.png',
        # barh specific parameters
        comparison_col: str | None = 'Treatment',
        barh_remove_yticklabels: bool = True,
        comparison_order: list[str] | None = None,
        barh_figure_plot_title: str | None = f'Expression (TPM)',
        barh_subplot_xlabel: str | None = 'Expression (TPM)',
        barh_sharex: bool = False,
        barh_set_xaxis_lims: tuple[int, int]| None = None,
        barh_legend: bool = True,
        barh_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),

        # dotplot specific parameters
        dotplot_figure_plot_title: str | None = 'log2fc',
        dotplot_pval_vars_col_label: str | None = 'pvalue',
        dotplot_l2fc_vars_col_label: str | None ='log2FoldChange',
        dotplot_subplot_xlabel: str | None = 'log2fc ((target)/(ref))',
        pval_label: str = 'p-value',
        l2fc_label: str = 'log2FoldChange',
        pvalue_cutoff_ring: float = 0.1,
        sizes: tuple[int, int] | None = (20, 2000),
        dotplot_sharex: bool = False,
        dotplot_set_xaxis_lims: tuple[int, int]| None = None,
        dotplot_legend: bool = True,
        dotplot_legend_bins: int | None = 4,
        dotplot_legend_bbox_to_anchor: tuple[int, int] | None = (0.5, -.05),
        # Optional annotation on the dotplot with l2fc and p-value
        dotplot_annotate: bool = False,
        dotplot_annotate_xy: tuple[float, float] | None = (0.8, 1.2),
        dotplot_annotate_fontsize: int | None = None,
        # 
        ):
    
    #from .. import anndata_io as adio not needed wrote new io code here

    ############ prep input tables / parse adata ############
    if feature_list is None:
        raise ValueError("feature_list must be provided.") 
    if adata is not None:
        print(f"AnnData object provideed with shape {adata.shape} and {len(adata.var_names)} features.")
        # if adata is provided, use it to get the data
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        if comparison_col not in adata.obs.columns:
            raise ValueError(f"Column '{comparison_col}' not found in adata.obs.")
    if x_df is not None:
        print(f"Using provided x_df with shape {x_df.shape}")
        _x_df = x_df.copy()
    elif layer is None:
        print("No layer provided, using adata.X with shape {adata.X.shape}")
        _x_df = adata.X.copy()  # use the raw data if no layer
    elif adata is not None and layer in adata.layers:
        print(f"No x_df provided, using adata.layers['{layer}'] with shape {adata.layers[layer].shape}")
        _x_df = adata.layers[layer].copy()

    if var_df is not None:
        print(f"Using provided var_df with shape {var_df.shape}")
        _var_df = var_df.copy()
    else:
        print(f"No var_df provided, using adata.var with shape {adata.var.shape}")
        _var_df = adata.var.copy()

    if obs_df is not None:
        print(f"Using provided obs_df with shape {obs_df.shape}")
        _obs_df = obs_df.copy()
    else:
        print(f"No obs_df provided, using adata.obs with shape {adata.obs.shape}")
        _obs_df = adata.obs.copy()

    # #) make df_obs_x, which is a tidy df with obs + expression columns
    if hasattr(_x_df, "toarray"):  # Convert sparse matrix to dense if necessary
        _x_df = _x_df.toarray()
    df_obs_x = pd.DataFrame(_x_df, columns=_var_df.index, index=_obs_df.index)
    df_obs_x = pd.concat([_obs_df, df_obs_x], axis=1)


    # Determine category order
    if comparison_order is None:
        # keep observed order
        categories = list(pd.Series(df_obs_x[comparison_col]).astype('category').cat.categories) \
                     or list(df_obs_x[comparison_col].unique())
    else:
        categories = list(comparison_order)

    # Build a fixed palette used for every subplot
    palette = sns.color_palette('tab10', n_colors=len(categories))
    color_map = dict(zip(categories, palette))


    ############ prep dotplots ############
    # #) get the p-value and l2fc columns from the adata.var
    #adata_var_df = adata.var.copy()  # make a copy of the var metadata
    # #) Compute -log10 p-values for coloring/legend, and a size metric
    log10pval_label = f'-log10({pval_label})'
    _pvals = pd.to_numeric(_var_df[dotplot_pval_vars_col_label], errors='coerce')
    _pvals = _pvals.clip(lower=1e-300, upper=1.0)
    _var_df[log10pval_label] = -np.log10(_pvals)

    # Size metric: proportional to -log10(p) unless raw p > 0.5, then set to minimum
    size_metric_col = 'dotplot_size_metric'
    _var_df[size_metric_col] = np.where(_pvals > 0.5, 0.0, _var_df[log10pval_label])
    # Establish sizing/normalization bounds using only plotted features
    size_min = 0.0
    _size_vals = pd.to_numeric(_var_df.loc[feature_list, size_metric_col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    size_max = float(_size_vals.max()) if np.isfinite(_size_vals.max()) else 0.0
    # #) compute l2fc abs().max()   for axis limits
    l2fc_x_limit = _var_df.loc[feature_list][dotplot_l2fc_vars_col_label].abs().max()
    # Also store a column for the ring overlay cutoff, truncated to 2 decimals
    ring_col = 'ring_cutoff'
    log10_thresh = float(-np.log10(pvalue_cutoff_ring))
    # round the scalar threshold safely
    _var_df[ring_col] = np.round(log10_thresh, 2)
    # Ensure we have a non-degenerate scale and include the ring value
    size_max = float(max(size_max, log10_thresh, 1e-6))
    # Colormap for significant points (>= threshold). Below threshold will be grey.
    _cmap = plt.get_cmap('viridis_r')
    _color_norm = plt.Normalize(vmin=log10_thresh, vmax=max(size_max, log10_thresh), clip=True)
    # #) Build feature labels for dotplot and bar labels
    # If feature_label_vars_col provided and present, use it; otherwise fallback to index
    if (feature_label_vars_col is not None) and (feature_label_vars_col in _var_df.columns):
        _feature_label_series = _var_df[feature_label_vars_col]
        # Fill NaNs in provided label column with the index values
        _feature_label_series = _feature_label_series.where(
            _feature_label_series.notna(), _var_df.index.to_series()
        ).astype(str)
    else:
        if feature_label_vars_col is not None and feature_label_vars_col not in _var_df.columns:
            print(f"Warning: feature_label_vars_col '{feature_label_vars_col}' not found in var_df; using index for labels.")
        _feature_label_series = _var_df.index.to_series().astype(str)

    # Optionally truncate labels to a maximum character length
    if (feature_label_char_limit is not None) and (feature_label_char_limit > 0):
        _feature_label_series = _feature_label_series.str.slice(0, int(feature_label_char_limit))

    # Set the dotplot y-axis label column
    _var_df['dotplot_feature_name'] = _feature_label_series
    # Map for bar subplot y-axis labels
    _feature_label_map = _feature_label_series.astype(str).to_dict()


    ############ ############ ############ ############
    # #) set up the figure and subfigures
    gene_list_len = len(feature_list)
    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=bar2dotplot_width_ratios)
    # Optional overall title for the whole figure
    if fig_title is not None:
        ft_size = fig_title_fontsize or subfig_title_fontsize or (legend_fontsize + 2)
        fig.suptitle(fig_title, fontsize=ft_size, y=fig_title_y)

    ###### Create subplots for subfigs[0] - horizontal bar plots
    axes0 = subfigs[0].subplots(gene_list_len, 1, sharex=barh_sharex, )
    # set subfig[0] title
    if barh_figure_plot_title is not None:
        subfigs[0].suptitle(barh_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[0].suptitle(f"{barh_subplot_xlabel} grouped by {comparison_col}\n", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    ####### Create subplots subfigs[1] - for dot plots
    axes1 = subfigs[1].subplots(gene_list_len, 1, sharex=dotplot_sharex)
    # set subfig[1] title
    if dotplot_figure_plot_title is not None:
        subfigs[1].suptitle(dotplot_figure_plot_title, fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)
    else:
        subfigs[1].suptitle(f"{dotplot_subplot_xlabel} grouped by {comparison_col}\n", fontsize=(subfig_title_fontsize or legend_fontsize), y=subfig_title_y)

    ################## loop through features and create subplots ##################
    for plot_num, gene in enumerate(feature_list):
        if gene_list_len == 1:
            ax0 = axes0
            ax1 = axes1
        else:
            ax0 = axes0[plot_num]
            ax1 = axes1[plot_num]
        ############ barh plots ############
        # Horizontal bars (aggregated by category)
        sns.barplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            hue=comparison_col,
            hue_order=categories,
            legend=False,
            palette=color_map,
        )
        if barh_remove_yticklabels:
            ax0.set_yticklabels([])
        # Overlay points (each sample), same order as bars
        sns.stripplot(
            x=gene, y=comparison_col,
            data=df_obs_x,
            order=categories,
            ax=ax0,
            color='black',
            legend=False
        )
        # set x-axis limits
        if barh_set_xaxis_lims is not None:
            ax0.set_xlim(barh_set_xaxis_lims)
        # set x-axis tic fontsize
        ax0.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax0.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        # remove xlabel for all but the last subplot
        ax0.set_xlabel('')
        # set ylabel for each subplot using mapped feature label
        _feat_label = _feature_label_map.get(gene, str(gene))
        ax0.set_ylabel(_feat_label, rotation=0, fontsize=feature_label_fontsize, ha='right', va='center')
        ax0.yaxis.set_label_coords(feature_label_x, 0.5)

        ############ dot plots ############
        # A) Plot the ring (facecolors="none") using the ring_col
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=dotplot_l2fc_vars_col_label,
            y='dotplot_feature_name',
            size=ring_col,            # ring size is the ring_cutoff column
            size_norm=(size_min, size_max),
            sizes=sizes,
            facecolors="none",
            edgecolors="red",
            linewidths=1,
            zorder=4,  # ensure ring draws above the filled dot
            legend=False,
            ax=ax1,
        )
        # B) Plot the main points, colored & sized by actual -log10 p-value
        # Determine dot color: grey if below threshold, colormap otherwise
        _val = float(_var_df.loc[gene, log10pval_label]) if gene in _var_df.index else np.nan
        if np.isfinite(_val) and (_val >= log10_thresh):
            _dot_color = _cmap(_color_norm(_val))
        else:
            _dot_color = 'grey'
        sns.scatterplot(
            data=_var_df.loc[[gene]],
            x=dotplot_l2fc_vars_col_label,
            y='dotplot_feature_name',
            size=size_metric_col,
            size_norm=(size_min, size_max),
            sizes=sizes,
            color=_dot_color,
            edgecolors="black",
            linewidths=.5,
            zorder=3,
            legend=False,
            ax=ax1,
        )
        # C) Optional compact annotation (l2fc and p-value) on the dotplot
        if dotplot_annotate and (gene in _var_df.index):
            try:
                _l2fc_val = _var_df.loc[gene, dotplot_l2fc_vars_col_label]
                _pval_val = _var_df.loc[gene, dotplot_pval_vars_col_label]
                if np.isfinite(_l2fc_val) and np.isfinite(_pval_val):
                    _ann_text = f"l2fc: {_l2fc_val:.2g} | p:{_pval_val:.2g}"
                    _ann_fs = dotplot_annotate_fontsize or max(8, int(tick_label_fontsize))
                    _xy = dotplot_annotate_xy or (0.8, 1.2)
                    ax1.text(
                        _xy[0], _xy[1], _ann_text,
                        transform=ax1.transAxes,
                        ha='right', va='center',
                        fontsize=_ann_fs, color='black'
                    )
            except Exception as e:
                print(f"Dotplot annotation failed for feature '{gene}': {e}")
        # set x-axis limits
        if dotplot_set_xaxis_lims is not None:
            ax1.set_xlim(dotplot_set_xaxis_lims)
        else:
            l2fc_xaxis_pad=1.05
            ax1.set_xlim((-l2fc_x_limit*l2fc_xaxis_pad), (l2fc_x_limit* l2fc_xaxis_pad))  # add a bit of padding
        # set x-axis tic fontsize
        ax1.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        # Vertical line at x=0
        ax1.axvline(x=0, color="red", linestyle="--")
        # remove xlabel for all but the last subplot
        ax1.set_xlabel('')
        # remove ylabel for all subplots
        ax1.set_ylabel('')
        ax1.set_yticklabels([])
        if dotplot_sharex and plot_num < gene_list_len - 1:
            ax1.set_xlabel('')


    # outside of the loop, set the xlabel for the last subplot
    ax0.set_xlabel(barh_subplot_xlabel, fontsize=legend_fontsize)
    ax1.set_xlabel(dotplot_subplot_xlabel, fontsize=legend_fontsize)

    # subfigs[0] Figure-level legend at bottom with the same bar colors
    if barh_legend:
        handles = [Patch(facecolor=color_map[c], edgecolor='none', label=str(c)) for c in categories]
        subfigs[0].legend(
            handles=handles,
            labels=[str(c) for c in categories],
            loc='lower center',
            ncol=min(len(categories), 6),
            title=comparison_col,
            bbox_to_anchor=barh_legend_bbox_to_anchor,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
        )

    # subfigs[1] figure-level legend styled like the example (4 interval dots + ring)
    if dotplot_legend:
        from matplotlib.lines import Line2D
        cmap_min = float(-np.log10(pvalue_cutoff_ring))
        cmap = plt.get_cmap('viridis_r')
        # Legend normalization works on -log10(p) values from threshold to max
        vmin_leg = cmap_min
        vmax_leg = max(size_max, cmap_min)
        norm = plt.Normalize(vmin=vmin_leg, vmax=vmax_leg, clip=True)

        # Compute the ring value up front (fixes UnboundLocalError and simplifies logic)
        v_ring = float(-np.log10(pvalue_cutoff_ring))

        # Build bins above the threshold in -log10(p) space
        n_bins = max(1, int(dotplot_legend_bins or 3))
        edges = np.linspace(vmin_leg, vmax_leg, n_bins + 1)
        uppers = edges[1:]
        # Deduplicate and drop values that are effectively == ring threshold
        uniq_vals = []
        seen = set()
        for u in uppers:
            # Round to 1 decimal for label stability and duplicate removal
            key = round(float(u), 1)
            if key <= round(v_ring, 1) + 1e-6:
                continue
            if key in seen:
                continue
            seen.add(key)
            uniq_vals.append(float(u))
        labels = [f"{round(u,1):.1f}" for u in uniq_vals]

        # Helper to map value -> scatter area -> legend marker size (points)
        def _area_for(v):
            return float(np.interp(v, [size_min, size_max], sizes))
        def _ms_for(v):
            return max(4.0, np.sqrt(_area_for(v)))

        handles = []
        # Ring handle labelled in -log10(p)
        ms_ring = _ms_for(v_ring)
        ring_handle = Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor='none', markeredgecolor='red', markeredgewidth=1.5,
            markersize=ms_ring,
            label=f"{v_ring:.1f} (-log10 p) ring",
        )
        # Grey handle for below-threshold dots, sized just below the ring (e.g., 0.99 if ring=1.0)
        v_grey = max(size_min, min(v_ring - 0.01, vmax_leg))
        grey_handle = Line2D(
            [0], [0], marker='o', linestyle='',
            markerfacecolor='grey', markeredgecolor='black',
            markersize=_ms_for(v_grey), label=f"< {v_ring:.1f}"
        )

        # One colored dot per interval, using the UPPER bound for color and size
        for u, lab in zip(uniq_vals, labels):
            handles.append(
                Line2D([0], [0], marker='o', linestyle='',
                       markerfacecolor=cmap(norm(u)), markeredgecolor='black',
                       markersize=_ms_for(u), label=lab
                       )
            )

        # Compose final order: ring, grey indicator, then colored bins
        legend_handles = [ring_handle, grey_handle]
        legend_handles.extend(handles)

        # Second legend: stacked entries (kept single column to avoid cramped layout)
        dot_handles = legend_handles
        leg1 = subfigs[1].legend(
            handles=dot_handles,
            loc='lower center',
            ncol=1,
            bbox_to_anchor=dotplot_legend_bbox_to_anchor,
            title=f"{log10pval_label}",
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize,
            frameon=True,
            markerfirst=True,
            handletextpad=1.5,
        )

    # Leave space for the bottom legend
    if dotplot_legend or barh_legend:
        rect_used = (np.array(tight_layout_rect_arg) + np.array([0, 0.12, 0, 0])).tolist()
    else:
        rect_used = tight_layout_rect_arg
    plt.tight_layout(rect=rect_used)


    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight" )
        print(f"Saved plot to {file_name}")
    plt.show()
    return fig, subfigs


####### START ############. l2fc_pvalue plots ###################.###################.###################.###################.

def l2fc_pvalue_dotplot_protein_metabolite(
    diff_tests,
    feature_idx_list,
    index_column='EntrezGeneSymbol',
    analyte_label_column='TargetFullName',
    analyte_label='Protein',
    comparison_column='Timepoint',
    comparison='Target_vs_Reference',
    pval_col='ttest_rel_pvals_corrected_Target_vs_Reference',
    l2fc_col='ttest_rel_mean_paired_l2fcfc_Target_vs_Reference',
    pval_label='p-value',
    x_axis_label='log2fc ((target)/(ref))',
    sort_x_axis=False,
    pvalue_cutoff=0.2,
    sizes=(20, 2000),
    figsize=(6,10),
    bbox_to_anchor=(0.5, -0.25),
    plot_title='Target_vs_Reference l2fc ((target)/(ref))',
    savefig=False,
    file_name='test_plot.png'
):
    """
    Create a ring-overlay dot plot of selected metabolites from 'diff_tests'.

    Parameters
    ----------
    diff_tests : pd.DataFrame
        DataFrame containing the differential test results. Must have columns:
        [index_column, analyte_label_column, analyte_label, pval_col, l2fc_col]
    feature_idx_list : list
        List of feature_idx_list values to include in the plot.
    index_column : str
        Column name for features (default 'var_names').
    analyte_label_column: str
    analyte_label : str
        Column name to use for labeling the y-axis (default 'Metabolite').
    comparison_column : str
        Value in 'comparison_column' to select for plotting (default 'Target_vs_Reference').
    pval_col : str
        Column containing the p-values (default 'ttest_rel_pvals_corrected_Target_vs_Reference').
    l2fc_col : str
        Column containing the log2 fold change (default 'ttest_rel_mean_paired_l2fcfc_Target_vs_Reference').
    pval_label : str
        Label to use in the plot for p-value axis (default 'p-value').
    x_axis_label : str
        X-axis label (default 'log2fc ((target)/(ref))').
    sort_x_axis : bool
        Whether to sort the x-axis by the x_axis_label (default False).
    pvalue_cutoff : float
        Numeric cutoff for ring overlay (default 0.2).
    bbox_to_anchor: legend postion default bbox_to_anchor=(0.5, -0.25),,
    figsize : tuple
        Figure size (default (6,10)).
    plot_title : str
        Title string (default 'Target_vs_Reference l2fc ((target)/(ref))').
    savefig : bool
        Whether to save the figure (default False).
    file_name : str
        File name for saving (default 'test_plot.png').

    Returns
    -------
    None (displays plot or saves figure).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # Default columns to keep if not provided
    columns2keep = [comparison_column, index_column, analyte_label_column, analyte_label, pval_col, l2fc_col]
    columns2keep_labels = [comparison_column, index_column, analyte_label_column, analyte_label, pval_label, x_axis_label]
    if comparison_column is None:
        columns2keep = [index_column, analyte_label_column, analyte_label, pval_col, l2fc_col]
        columns2keep_labels = [index_column, analyte_label_column, analyte_label, pval_label, x_axis_label]

    # 1) Copy and prepare DataFrame
    df = diff_tests.copy()
    df[index_column] = df[index_column].astype(str)
    # Make truncated metabolite name (40 chars)
    df[analyte_label] = df[analyte_label_column].astype(str).str[:40]

    # 2) Filter by comparison_column if not None
    if comparison_column:
        df = df[df[comparison_column] == comparison].copy()

    # 3) Select columns & rename
    df = df[columns2keep].copy()
    df.columns = columns2keep_labels

    # 4) Compute -log10 p-value
    log10pval_label = f'-log10{pval_label}'
    df[log10pval_label] = -np.log10(df[pval_label])

    #size_min =0.01 #df[log10pval_label].min()
    size_min= -np.log10(.9)
    size_max = df[log10pval_label].max()

    # Also store a column for the ring overlay cutoff, truncated to 2 decimals
    ring_col = 'ring_cutoff'
    df[ring_col] = (-np.log10(pvalue_cutoff)).round(2)

    # 5) Filter for desired feature_idx_list list
    df = df[df[index_column].isin(feature_idx_list)].copy()
    # re order by index_list
    df[index_column] = pd.Categorical(df[index_column], categories=feature_idx_list, ordered=True)
    df = df.sort_values(index_column)

    # 6) Sort by x_axis_label
    if sort_x_axis:
        df = df.sort_values(by=x_axis_label, ascending=True)

    # Create the figure
    plt.figure(figsize=figsize)

    # Define size scale from actual -log10 p-values
    #size_min =0.01 #df[log10pval_label].min()
    #size_max = df[log10pval_label].max()
    
    #df.loc['test'] = [Timepoint_comparison,'test','test','test',pvalue_cutoff,0.1, -np.log10(pvalue_cutoff).round(2),-np.log10(pvalue_cutoff).round(2)]

    # A) Plot the ring (facecolors="none") using the ring_col
    ax = sns.scatterplot(
        data=df,
        x=x_axis_label,
        y=analyte_label,
        size=ring_col,            # ring size is the ring_cutoff column
        size_norm=(size_min, size_max),
        sizes=sizes,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
        zorder=3,
    )

    # B) Plot the main points, colored & sized by actual -log10 p-value
    ax = sns.scatterplot(
        data=df,
        x=x_axis_label,
        y=analyte_label,
        size=log10pval_label,
        size_norm=(size_min, size_max),
        sizes=sizes,
        hue=log10pval_label,
        palette="viridis_r",
        edgecolors="black",
        linewidths=.5,
        legend="brief",
        ax=ax,
    )

    # Vertical line at x=0
    ax.axvline(x=0, color="red", linestyle="--")

    # Title & labels
    plt.title(f'{plot_title}\n', fontsize=16)
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(analyte_label, fontsize=16)

    # Legend adjustments
    ax.legend(
        #loc="center left",
        bbox_to_anchor=bbox_to_anchor,
        title=f"-log10 {pval_label}",
        markerscale=0.4,    # shrink markers in legend
        labelspacing=1.2,
        borderpad=1.2,
        # make columns
        loc='lower center',
        ncol=10,
        
    )
    #plt.tight_layout()

    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {file_name}")
    else:
        plt.show()

'''
### **Example Usage**


# Suppose diff_tests is your main DataFrame of results
diff_tests_res_df = diff_tests.copy()  # or your real data

analyte_list=['CHEM_ID1', 'CHEM_ID2', 'CHEM_ID3', 'CHEM_ID4', 'CHEM_ID5']
    
l2fc_pvalue_dotplot_protein_metabolite(
    diff_tests,
    feature_idx_list,
    index_column='EntrezGeneSymbol',
    analyte_label_column='TargetFullName',
    analyte_label='Protein',
    comparison_column='Timepoint',
    comparison='Target_vs_Reference',
    pval_col='ttest_rel_pvals_corrected_Target_vs_Reference',
    l2fc_col='ttest_rel_mean_paired_l2fcfc_Target_vs_Reference',
    pval_label='p-value',
    x_axis_label='log2fc ((target)/(ref))',
    sort_x_axis=False,
    pvalue_cutoff=0.2,
    sizes=(20, 2000),
    figsize=(6,10),
    bbox_to_anchor=(0.5, -0.25),
    plot_title='Target_vs_Reference l2fc ((target)/(ref))',
    savefig=False,
    file_name='test_plot.png'
)
'''



def l2fc_pvalue_dotplot_gex(
    diff_tests,
    feature_idx_list,
    index_column='var_name',
    analyte_label_column='analyte_label',
    analyte_label='Gene_Expression',
    comparison_column='Timepoint',
    comparison='Target_vs_Reference',
    pval_col='ttest_rel_pvals_corrected_Target_vs_Reference',
    l2fc_col='ttest_rel_mean_paired_l2fcfc_Target_vs_Reference',
    pval_label='p-value',
    x_axis_label='log2fc ((target)/(ref))',
    sort_x_axis=False,
    pvalue_cutoff=0.2,
    sizes=(20, 2000),
    figsize=(6,10),
    bbox_to_anchor=(0.5, -0.25),
    dotplot_set_xaxis_lims=None,
    plot_title='Target_vs_Reference l2fc ((target)/(ref))',
    savefig=False,
    file_name='test_plot.png'
    
):
    """
    Create a ring-overlay dot plot of selected metabolites from 'diff_tests'.

    Parameters
    ----------
    diff_tests : pd.DataFrame
        DataFrame containing the differential test results. Must have columns:
        [index_column, analyte_label_column, analyte_label, pval_col, l2fc_col]
    feature_idx_list : list
        List of feature_idx_list values to include in the plot.
    index_column : str
        Column name for features (default 'var_names').
    analyte_label_column: str
    analyte_label : str
        Column name to use for labeling the y-axis (default 'Metabolite').
    comparison_column : str
        Value in 'comparison_column' to select for plotting (default 'Target_vs_Reference').
    pval_col : str
        Column containing the p-values (default 'ttest_rel_pvals_corrected_Target_vs_Reference').
    l2fc_col : str
        Column containing the log2 fold change (default 'ttest_rel_mean_paired_l2fcfc_Target_vs_Reference').
    pval_label : str
        Label to use in the plot for p-value axis (default 'p-value').
    x_axis_label : str
        X-axis label (default 'log2fc ((target)/(ref))').
    sort_x_axis : bool
        Whether to sort the x-axis by the x_axis_label (default False).
    pvalue_cutoff : float
        Numeric cutoff for ring overlay (default 0.2).
    bbox_to_anchor: legend postion default bbox_to_anchor=(0.5, -0.25),,
    figsize : tuple
        Figure size (default (6,10)).
    plot_title : str
        Title string (default 'Target_vs_Reference l2fc ((target)/(ref))').
    savefig : bool
        Whether to save the figure (default False).
    file_name : str
        File name for saving (default 'test_plot.png').

    Returns
    -------
    None (displays plot or saves figure).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    # Default columns to keep if not provided
    columns2keep = [comparison_column, index_column, analyte_label_column, analyte_label, pval_col, l2fc_col]
    columns2keep_labels = [comparison_column, index_column, analyte_label_column, analyte_label, pval_label, x_axis_label]
    if comparison_column is None:
        columns2keep = [index_column, analyte_label_column, analyte_label, pval_col, l2fc_col]
        columns2keep_labels = [index_column, analyte_label_column, analyte_label, pval_label, x_axis_label]

    # 1) Copy and prepare DataFrame
    df = diff_tests.copy()
    df[index_column] = df[index_column].astype(str)
    # Make truncated metabolite name (40 chars)
    df[analyte_label] = df[analyte_label_column].astype(str).str[:40]

    # 2) Filter by comparison_column if not None
    if comparison_column:
        df = df[df[comparison_column] == comparison].copy()

    # 3) Select columns & rename
    df = df[columns2keep].copy()
    df.columns = columns2keep_labels

    # 4) Compute -log10 p-value
    log10pval_label = f'-log10{pval_label}'
    df[log10pval_label] = -np.log10(df[pval_label])

    #size_min =0.01 #df[log10pval_label].min()
    size_min= -np.log10(.9)
    size_max = df[log10pval_label].max()

    # Also store a column for the ring overlay cutoff, truncated to 2 decimals
    ring_col = 'ring_cutoff'
    df[ring_col] = (-np.log10(pvalue_cutoff)).round(2)

    # 5) Filter for desired feature_idx_list list
    df = df[df[index_column].isin(feature_idx_list)].copy()
    # re order by index_list
    df[index_column] = pd.Categorical(df[index_column], categories=feature_idx_list, ordered=True)
    df = df.sort_values(index_column)

    # 6) Sort by x_axis_label
    if sort_x_axis:
        df = df.sort_values(by=x_axis_label, ascending=True)

    # Create the figure
    plt.figure(figsize=figsize)

    # Define size scale from actual -log10 p-values
    #size_min =0.01 #df[log10pval_label].min()
    #size_max = df[log10pval_label].max()
    
    #df.loc['test'] = [Timepoint_comparison,'test','test','test',pvalue_cutoff,0.1, -np.log10(pvalue_cutoff).round(2),-np.log10(pvalue_cutoff).round(2)]

    # A) Plot the ring (facecolors="none") using the ring_col
    ax = sns.scatterplot(
        data=df,
        x=x_axis_label,
        y=analyte_label,
        size=ring_col,            # ring size is the ring_cutoff column
        size_norm=(size_min, size_max),
        sizes=sizes,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
        zorder=3,
    )

    # B) Plot the main points, colored & sized by actual -log10 p-value
    ax = sns.scatterplot(
        data=df,
        x=x_axis_label,
        y=analyte_label,
        size=log10pval_label,
        size_norm=(size_min, size_max),
        sizes=sizes,
        hue=log10pval_label,
        palette="viridis_r",
        edgecolors="black",
        linewidths=.5,
        legend="brief",
        ax=ax,
    )

        # x limits and ticks
    if dotplot_set_xaxis_lims is not None:
        ax.set_xlim(dotplot_set_xaxis_lims)

    # Vertical line at x=0
    ax.axvline(x=0, color="red", linestyle="--")

    # Title & labels
    plt.title(f'{plot_title}\n', fontsize=16)
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(analyte_label, fontsize=16)

    # Legend adjustments
    ax.legend(
        #loc="center left",
        bbox_to_anchor=bbox_to_anchor,
        title=f"-log10 {pval_label}",
        markerscale=0.4,    # shrink markers in legend
        labelspacing=1.2,
        borderpad=1.2,
        # make columns
        loc='lower center',
        ncol=10,
        
    )
    #plt.tight_layout()

    if savefig:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {file_name}")
    else:
        plt.show()

'''
### **Example Usage**


# Suppose diff_tests is your main DataFrame of results
diff_tests_res_df = diff_tests.copy()  # or your real data

    
l2fc_pvalue_dotplot_gex(
    diff_tests,
    feature_idx_list,
    index_column='var_name',
    analyte_label_column='analyte_label',
    analyte_label='Gene_Expression',
    comparison_column='Timepoint',
    comparison='Target_vs_Reference',
    pval_col='ttest_rel_pvals_corrected_Target_vs_Reference',
    l2fc_col='ttest_rel_mean_paired_l2fcfc_Target_vs_Reference',
    pval_label='p-value',
    x_axis_label='log2fc ((target)/(ref))',
    sort_x_axis=False,
    pvalue_cutoff=0.2,
    sizes=(20, 2000),
    figsize=(6,10),
    bbox_to_anchor=(0.5, -0.25),
    plot_title='Target_vs_Reference l2fc ((target)/(ref))',
    savefig=False,
    file_name='test_plot.png'
)
'''

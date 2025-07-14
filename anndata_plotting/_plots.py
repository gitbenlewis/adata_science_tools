import matplotlib.pyplot as plt

# Paul Tol’s 10-color set
tol_colors = [
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#661100", "#CC6677",
    "#882255", "#AA4499"
]


def show_tol_colors(colors=None):
    """
    Creates a bar plot where each bar has one of the given colors.
    The x-axis is labeled with the hex color codes.
    """
    import matplotlib.pyplot as plt
    if colors is None:
        tol_colors = [
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#661100", "#CC6677",
    "#882255", "#AA4499"
        ]
        colors=tol_colors
    n = len(colors)
    x_vals = range(n)
    y_vals = [1]*n  # All bars have the same height (1)

    fig, ax = plt.subplots(figsize=(8, 2))
    bars = ax.bar(x_vals, y_vals)

    # Set each bar’s color and label
    for i, bar in enumerate(bars):
        bar.set_color(colors[i])
        # Put the hex code as an x-axis tick label
        ax.text(
            i, 0.5, colors[i],
            rotation=90, fontsize=9,
            color='white', ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.3, boxstyle='round')
        )

    # Remove extra chart details
    ax.set_xticks(x_vals)
    ax.set_xticklabels(['']*n)   # we place color codes in the bars, so x tick labels can be blank
    ax.set_yticks([])

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)

    ax.set_title("Paul Tol's 10-Color Palette", fontsize=12)
    plt.tight_layout()
    plt.show()

# updated to used the hue_column to set the hue values on 2025.02.28
def volcano_plot_sns_sinlge_comparison_generic(_df, l2fc_col='log2FoldChange',set_xlabel='log2fc model',xlimit=None,
                                                padj_col='padj', set_ylabel='-log10(padj)',ylimit=None,
                    title_text='volcano_plot',comparison_label=' Comparison',
                     hue_column=None,
                     log2FoldChange_threshold=.1,
                     figsize=(15, 10),legend_bbox_to_anchor=(1.15, 1),
                     label_top_features=False,feature_label_col='gene_names',n_top_features=50,
                     dot_size_shrink_factor=300
                     ):

    """
    Generate a volcano plot for a single differential expression comparison.

    This function creates a volcano plot using Seaborn and Matplotlib based on differential 
    expression data provided in a DataFrame. The x-axis displays log2 fold changes and the y-axis 
    displays the negative log10 of the adjusted p-values (padj). The plot visualizes significance 
    levels by categorizing data points (e.g., alpha=0.05, 0.1, 0.2) and optionally labels the top 
    features based on specified criteria.

    The function performs the following preprocessing steps:
      - Fills missing adjusted p-values with 1 and computes -log10(padj) to avoid log(0) issues.
      - Adds binary columns for significance at alpha levels of 0.2, 0.1, and 0.05, using a minimum 
        absolute log2 fold change threshold.
      - Combines these significance levels into a single categorical "Significance" column.
      - Adjusts out-of-range values for both axes (x: log2 fold change, y: -log10(padj)) based on 
        computed limits if not provided.
      - Optionally, labels the top features (genes) by plotting text labels for those with the lowest 
        adjusted p-values and extreme fold changes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing differential expression results. Must include columns for log2 fold change 
        (default 'log2FoldChange') and adjusted p-values (default 'padj'), among others.
    l2fc_col : str, optional
        Name of the column containing log2 fold change values. Default is 'log2FoldChange'.
    set_xlabel : str, optional
        Label for the x-axis. Default is 'log2fc model'.
    xlimit : float, optional
        Maximum absolute value for the x-axis. If None, it is computed from the data.
    padj_col : str, optional
        Name of the column containing adjusted p-values. Default is 'padj'.
    set_ylabel : str, optional
        Label for the y-axis. Default is '-log10(padj)'.
    ylimit : float, optional
        Maximum value for the y-axis. If None, it is computed from the data.
    title_text : str, optional
        Main title text for the plot. Default is 'volcano_plot'.
    comparison_label : str, optional
        Additional label text to describe the comparison (e.g., dataset or test used). Default is 'DeSeq2 Comparison'.
    hue_column : str, optional
        Column name to use for coloring the data points. If None, the function uses a default significance column.
    log2FoldChange_threshold : float, optional
        Minimum absolute log2 fold change for considering a feature significant when labeling. Default is 0.1.
    figsize : tuple, optional
        Figure size for the plot. Default is (15, 10).
    legend_bbox_to_anchor : tuple, optional
        Bounding box coordinates to anchor the legend. Default is (1.15, 1).
    label_top_features : bool, optional
        Whether to label the top features (e.g., genes) on the plot based on significance. Default is False.
    feature_label_col : str, optional
        Column name in `df` to use for feature labels if `label_top_features` is True. Default is 'gene_names'.
    n_top_features : int, optional
        Number of top features to label on the plot. Default is 50.
    dot_size_shrink_factor : int, optional
        Factor to shrink the dot size based on the number of data points. Default
        is 300.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The matplotlib Axes object containing the volcano plot.

    Notes
    -----
    - The function computes additional columns such as '-log10(padj)', significance flags at various alpha levels,
      and a 'Marker' column to distinguish data points that exceed defined axis limits.
    - Vertical dashed lines are drawn at ±log2FoldChange_threshold to highlight the minimum fold change considered significant.
    - The function handles two plotting cases: one where features are labeled without a hue column and one where a hue
      column is provided. In the latter case, the top features are labeled based on a filtered and sorted subset of the data.
    - Out-of-range values for '-log10(padj)' and log2 fold change are clipped to specified limits to improve plot clarity.
    """

    import seaborn as sns
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt

        # Get the tab10 palette
    tab10_palette = sns.color_palette("tab10")

    # Move gray to the first position
    #custom_palette = [tab10_palette[7]] + tab10_palette[:7] + tab10_palette[8:]
    significance_custom_palette = [tab10_palette[7]] + tab10_palette[:3] #+ tab10_palette[8:]
    hue_platte_custom_palette = [tab10_palette[7]] + tab10_palette[:6] + tab10_palette[8:]

    df = _df.copy()

    print(df.shape)

    # if hue_column is None set to 'Significance'
    if hue_column is None:
        hue_value='Significance'
    else:
        hue_value=hue_column

    required_columns = {l2fc_col, padj_col, }
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame is missing one of the required columns: {required_columns}")

    ###### Prepare df by adding coluomns for '-log10(padj)' and signifigcance level and adjusting out of range values
    # Replace NaN values with a specified value, for example, 1
    df[padj_col] = df[padj_col].fillna(1)
    # Prepare data by adjusting p-values to avoid log(0) issues
    df['-log10(padj)'] = -np.log10(df[padj_col].replace(0, np.nextafter(0, 1)))
    # Assuming df[padj_col] contains the adjusted p-values
    # Add column for alpha=0.2 significance level
    df['alpha=0.2'] = ((df[padj_col] < 0.2) & (abs(df[l2fc_col])>=log2FoldChange_threshold))
    # Add column for alpha=0.1 significance level
    df['alpha=0.1'] = ((df[padj_col] < 0.1) & (abs(df[l2fc_col])>=log2FoldChange_threshold))
    # Add column for alpha=0.05 significance level
    df['alpha=0.05'] = ((df[padj_col] < 0.05) & (abs(df[l2fc_col])>=log2FoldChange_threshold))

    # add column for signifigcance hue
    # first combine the alpha columns into one column labeled Signifigcance
    #df['Significance'] = 'alpha>0.2'
    df['Significance'] = 'Not Significant'
    df.loc[df['alpha=0.2'],'Significance'] = 'alpha=0.2'
    df.loc[df['alpha=0.1'],'Significance'] = 'alpha=0.1'
    df.loc[df['alpha=0.05'],'Significance'] = 'alpha=0.05'
    df['Significance'] = df['Significance'].astype('category')



    ######  adjusting out of range values and changing dot type if out of range


    ##### #####  set limits 
    # set ylimit if none to  to 99 percentile of ['-log10(padj)']
    if not ylimit:
        ylimit = df[(df[padj_col]<0.05)&(df[l2fc_col].abs()>log2FoldChange_threshold)]['-log10(padj)'].quantile(0.99)
        if np.isnan(ylimit):
            ylimit=df['-log10(padj)'].quantile(0.99)
    # set xlimit if none to 99 percentile of abs(x)
    if not xlimit:
        xlimit = df[(df[padj_col]<0.05)&(df[l2fc_col].abs()>log2FoldChange_threshold)][l2fc_col].abs().quantile(0.99)
    # if xlimit is nan set to quantile(0.99)
        if np.isnan(xlimit):
            xlimit=df[l2fc_col].abs().quantile(0.99)


    # add 'Marker' column for out of range data points with '-log10(padj)' > ylimit or abs(l2fc_col) > xlimit value of 'In_Range' or 'Out_of_Range'
    df['Marker'] = 'In_Range'
    df.loc[df['-log10(padj)']>=ylimit,'Marker'] = 'Out_of_Range'
    #  abs(l2fc_col) 
    df.loc[abs(df[l2fc_col])>=xlimit,'Marker'] = 'Out_of_Range'

    # order the categories
    # Ensure the required categories are present
    df['Marker'] = df['Marker'].astype('category')
    required_range_cats = {'In_Range', 'Out_of_Range'}
    if not required_range_cats.issubset(df['Marker'].cat.categories):
        df['Marker'] = df['Marker'].cat.set_categories(['In_Range', 'Out_of_Range'])


    # replace values in the -log10(padj) column that above the ylimit with the ylimit
    if ylimit:
        df['-log10(padj)'] = df['-log10(padj)'].apply(lambda x: (ylimit*0.99) if x>=ylimit else x)
    else:
        ylimit = df['-log10(padj)'].max()
    # replace values in the log2FoldChange column that above or below the xlimit with the xlimit
    if xlimit:
        df[l2fc_col] = df[l2fc_col].apply(lambda x: (xlimit*0.99) if x>=xlimit else x)
        df[l2fc_col] = df[l2fc_col].apply(lambda x: (-xlimit*0.99)  if x<=-xlimit else x)
    else:
        xlimit = max(abs(df[l2fc_col].min()), df[l2fc_col].max())

    ### set the marker size relative to number of dots
    rel_size=df.shape[0]/dot_size_shrink_factor


    if label_top_features and hue_column is None:
        fig, ax = plt.subplots(figsize=figsize)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', hue=hue_value, style='Marker', 
                            palette=significance_custom_palette,sizes=rel_size,  s=rel_size, 
                            ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--',label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend( 
            bbox_to_anchor=legend_bbox_to_anchor, 
            loc=1, 
            borderaxespad=0.05)
        #label top genes by padj
        for line in range(0,n_top_features):
            p.text(df.sort_values(by=padj_col)[l2fc_col].to_list()[line],df.sort_values(by=padj_col)['-log10(padj)'].to_list()[line],
                   df.sort_values(by=padj_col)[feature_label_col].to_list()[line],
                      horizontalalignment='left', size='small', color='black')
        #label top genes by neg l2fc
        for line in range(0,int(n_top_features/2)):
            p.text(df.sort_values(by=l2fc_col)[l2fc_col].to_list()[line],df.sort_values(by=l2fc_col)['-log10(padj)'].to_list()[line],
                   df.sort_values(by=l2fc_col)[feature_label_col].to_list()[line],
                      horizontalalignment='left', size='small', color='black')
        #label top genes by pos l2fc
        for line in range(0,int(n_top_features/2)):
            p.text(df.sort_values(by=l2fc_col,ascending=False)[l2fc_col].to_list()[line],df.sort_values(by=l2fc_col,ascending=False)['-log10(padj)'].to_list()[line],
                   df.sort_values(by=l2fc_col,ascending=False)[feature_label_col].to_list()[line],
                      horizontalalignment='left', size='small', color='black')
    
    elif label_top_features and hue_column is not None:
        fig, ax = plt.subplots(figsize=figsize)
        # plot once without hue to plot all the dots
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', #hue=hue_value,
                             style='Marker', color='gray',s=rel_size/2, alpha=0.5,ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--',label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend( #title=facet_col,
            bbox_to_anchor=legend_bbox_to_anchor, 
            loc=1, 
            borderaxespad=0.05)
        # plot once without hue to plot the dots with values in the hue column
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', hue=hue_value, style='Marker', 
                            palette=hue_platte_custom_palette[1:],sizes=(rel_size),  s=rel_size,  ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--',label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend( #title=facet_col,
            bbox_to_anchor=legend_bbox_to_anchor, 
            loc=1, 
            borderaxespad=0.05)
        if hue_column is not None:
            # Filter the DataFrame to only rows where the hue_column is not missing
            filtered_df = df[df[hue_column].notna()].sort_values(by=padj_col)
            top_features = filtered_df.head(n_top_features)

        ## only label top features if hue_column is not None
        #label top genes by padj
        n_top_features=min(n_top_features,filtered_df.shape[0])
        for line in range(0,n_top_features):
            p.text(filtered_df.sort_values(by=padj_col)[l2fc_col].to_list()[line],filtered_df.sort_values(by=padj_col)['-log10(padj)'].to_list()[line],
                   filtered_df.sort_values(by=padj_col)[feature_label_col].to_list()[line],
                      horizontalalignment='left', size='small', color='black')
        #label top genes by neg l2fc
        for line in range(0,int(n_top_features/2)):
            p.text(filtered_df.sort_values(by=l2fc_col)[l2fc_col].to_list()[line],filtered_df.sort_values(by=l2fc_col)['-log10(padj)'].to_list()[line],
                   filtered_df.sort_values(by=l2fc_col)[feature_label_col].to_list()[line],
                      horizontalalignment='left', size='small', color='black')
        #label top genes by pos l2fc
        for line in range(0,int(n_top_features/2)):
            p.text(filtered_df.sort_values(by=l2fc_col,ascending=False)[l2fc_col].to_list()[line],filtered_df.sort_values(by=l2fc_col,ascending=False)['-log10(padj)'].to_list()[line],
                   filtered_df.sort_values(by=l2fc_col,ascending=False)[feature_label_col].to_list()[line],
                      horizontalalignment='left', size='small', color='black')

    else:
        fig, ax = plt.subplots(figsize=figsize)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)',hue=hue_value, style='Marker', 
                             palette=significance_custom_palette, s=rel_size,  
                            ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--',label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--',)
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend( )
        # move legend
        plt.legend(bbox_to_anchor=legend_bbox_to_anchor, 
            loc=1, 
            borderaxespad=0.05)

    return p



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



def l2fc_pvalue_dotplot(
    diff_tests,
    metab_list,
    index_column='CHEM_ID',
    analyte_label='Metabolite',
    Timepoint_comparison='Drug_vs_Vehicle_30hr',
    pval_col='ttest_rel_nested_pvals_corrected_nested_predose_baseline_drug_vehicle',
    l2fc_col='ttest_rel_nested_mean_paired_l2fcfc_nested_predose_baseline_drug_vehicle',
    pval_label='p-value',
    x_axis_label='log2fc ((target/con)/(ref/con))',
    sort_x_axis=False,
    pvalue_cutoff=0.2,
    sizes=(20, 2000),
    figsize=(6,10),
    plot_title='Drug_vs_Vehicle_30hr l2fc ((target/con)/(ref/con))',
    savefig=False,
    file_name='test_plot.png'
):
    """
    Create a ring-overlay dot plot of selected metabolites from 'diff_tests'.

    Parameters
    ----------
    diff_tests : pd.DataFrame
        DataFrame containing the differential test results. Must have columns:
        ['Timepoint', 'CHEM_ID', 'CHEMICAL_NAME', <pval_col>, <l2fc_col>, etc.]
    metab_list : list
        List of CHEM_ID values to include in the plot.
    index_column : str
        Column name for CHEM_ID (default 'CHEM_ID').
    analyte_label : str
        Column name to use for labeling the y-axis (default 'Metabolite').
    Timepoint_comparison : str
        Value in 'Timepoint' to select for plotting (default 'Drug_vs_Vehicle_30hr').
    pval_col : str
        Column containing the p-values (default 'ttest_rel_nested_pvals_corrected_nested_predose_baseline_drug_vehicle').
    l2fc_col : str
        Column containing the log2 fold change (default 'ttest_rel_nested_mean_paired_l2fcfc_nested_predose_baseline_drug_vehicle').
    pval_label : str
        Label to use in the plot for p-value axis (default 'p-value').
    x_axis_label : str
        X-axis label (default 'log2fc ((target/con)/(ref/con))').
    sort_x_axis : bool
        Whether to sort the x-axis by the x_axis_label (default False).
    pvalue_cutoff : float
        Numeric cutoff for ring overlay (default 0.2).
    figsize : tuple
        Figure size (default (6,10)).
    plot_title : str
        Title string (default 'Drug_vs_Vehicle_30hr l2fc ((target/con)/(ref/con))').
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
    columns2keep = ['Timepoint', 'CHEM_ID', 'CHEMICAL_NAME', analyte_label, pval_col, l2fc_col]
    columns2keep_labels = ['Timepoint', 'CHEM_ID', 'CHEMICAL_NAME', analyte_label, pval_label, x_axis_label]

    # 1) Copy and prepare DataFrame
    df = diff_tests.copy()
    df[index_column] = df[index_column].astype(str)
    # Make truncated metabolite name (40 chars)
    df[analyte_label] = df['CHEMICAL_NAME'].astype(str).str[:40]

    # 2) Filter by timepoint
    df = df[df['Timepoint'] == Timepoint_comparison].copy()

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

    # 5) Filter for desired metabolite list
    df = df[df[index_column].isin(metab_list)].copy()
    # re order by index_list
    df[index_column] = pd.Categorical(df[index_column], categories=metab_list, ordered=True)
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
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel(analyte_label, fontsize=12)

    # Legend adjustments
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title=f"-log10 {pval_label}",
        markerscale=0.4,    # shrink markers in legend
        labelspacing=1.2,
        borderpad=1.2
    )
    plt.tight_layout()

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
    

l2fc_pvalue_dotplot(
    diff_tests=diff_tests_res_df,
    metab_list=metab_list,
    index_column='CHEM_ID',
    analyte_label='Metabolite',
    Timepoint_comparison='Drug_vs_Vehicle_',
    pval_col='ttest_rel_nested_pvals_corrected_nested_predose_baseline_drug_vehicle',
    l2fc_col='ttest_rel_nested_mean_paired_l2fcfc_nested_predose_baseline_drug_vehicle',
    pval_label='p-value',
    x_axis_label='log2fc ((target/con)/(ref/con))',
    sort_x_axis=False,
    pvalue_cutoff=0.2,
    figsize=(6, 10),
    plot_title='Drug_vs_Vehicle l2fc ((target/con)/(ref/con))',
    savefig=False,
    file_name='example_dotplot.png'
)
'''
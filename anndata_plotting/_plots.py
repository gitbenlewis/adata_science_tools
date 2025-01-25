
def volcano_plot_sns_sinlge_comparison_generic(df, l2fc_col='log2FoldChange',set_xlabel='log2fc Deseq2 model',xlimit=None,
                                                padj_col='padj', set_ylabel='-log10(padj)',ylimit=None,
                    title_text='volcano_plot',comparison_label='DeSeq2 Comparison',
                     facet_col=None,dot_color=None,
                     sharex=True,sharey=True,
                     log2FoldChange_threshold=.1,
                     figsize=(15, 10),
                     label_top_features=False,feature_label_col='gene_names',n_top_features=50
                     ):

    """
    Create a volcano plot using the given DataFrame.
    def volcano_plot_sns_sinlge_comparison_generic(df, l2fc_col='log2FoldChange', padj_col='padj', 
                        title_text='volcano_plot',comparison_label='DeSeq2 Comparison',
                        set_xlabel='log2fc Deseq2 model',set_ylabel='-log10(padj)',
                     facet_col=None,dot_color=None,
                     sharex=True,sharey=True,ylimit=None,xlimit=None,
                     log2FoldChange_threshold=1,
                     figsize=(15, 10),
                     label_top_features=False,feature_label_col='gene_names',n_top_features=50):
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
    custom_palette = [tab10_palette[7]] + tab10_palette[:3] #+ tab10_palette[8:]

    print(df.shape)

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
    rel_size=df.shape[0]/333


    if label_top_features:
        fig, ax = plt.subplots(figsize=figsize)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)', hue='Significance', style='Marker', 
                            palette=custom_palette,sizes=(rel_size),  s=rel_size, 
                            ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--',label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--')
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend( #title=facet_col,
            bbox_to_anchor=(1.15, 1), 
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

    else:
        fig, ax = plt.subplots(figsize=figsize)
        p = sns.scatterplot(data=df, x=l2fc_col, y='-log10(padj)',hue='Significance', style='Marker', 
                             palette=custom_palette, s=rel_size,  
                            ax=ax)
        p.set(xlim=(-xlimit, xlimit), ylim=(0, ylimit))
        p.set_title(f'{title_text}\n{comparison_label}\n\n')
        p.axvline(x=log2FoldChange_threshold, color='gray', linestyle='--',label=f'log2fc>|{log2FoldChange_threshold}| ')
        p.axvline(x=-log2FoldChange_threshold, color='gray', linestyle='--',)
        p.set_xlabel(set_xlabel)
        p.set_ylabel(set_ylabel)
        p.legend( #title=facet_col
            )
        # move legend
        plt.legend(bbox_to_anchor=(1.15, 1), 
            loc=1, 
            borderaxespad=0.05)

    return p
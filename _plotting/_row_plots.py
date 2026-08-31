from . import palettes
import matplotlib.pyplot as plt
import seaborn as sns

def plot_columns(
    df,
    columns2plot,
    columns2plot_titles,
    y_groupby,
    figsize,
    sharex,
    sharey,
    swarm_size=10,
    suptitle_fontsize=40,
    subplot_title_fontsize=30,
    y_label_fontsize=30,
    y_tick_label_fontsize=24,
):
    '''
    df: dataframe
    columns2plot: list of columns to plot
    columns2plot_titles: list of titles for each column
    y_groupby: column to group by
    figsize: tuple of figure size
    sharex: bool
    sharey: bool
    swarm_size: marker size for the overlaid sample swarm
    suptitle_fontsize: font size for the figure title
    subplot_title_fontsize: font size for each metric title
    y_label_fontsize: font size for the grouping-axis label
    y_tick_label_fontsize: font size for grouping-axis tick labels
    '''
    nplots=len(columns2plot)
    fig, axes = plt.subplots(1, nplots, sharex=sharex,sharey=sharey, figsize=figsize)
    fig.suptitle(f'Grouped by {y_groupby}\n', fontsize=suptitle_fontsize)
    for i in range(0,nplots):
        ax=axes[i]
        column2plot=columns2plot[i]
        g=sns.barplot(ax=ax,  y=y_groupby,x=column2plot, data=df,
                errorbar=('ci', 95),capsize=.5, errcolor="0.2",linewidth=3, edgecolor="0", palette=palettes.godsnot_102,)
        ax.set_title(columns2plot_titles[i], fontsize=subplot_title_fontsize)
        ax.set_ylabel(f' {y_groupby}', fontsize=y_label_fontsize)
        ax.yaxis.set_tick_params(labelsize=y_tick_label_fontsize)
        g=  sns.swarmplot(ax=ax,data=df,x=column2plot, y=y_groupby,  size=swarm_size,color=".25")
        if i>0:
            ax.set_yticks([])
            ax.set_ylabel(None)
        plt.tight_layout()

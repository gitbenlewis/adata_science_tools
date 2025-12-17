'''  _venn_plots.py '''
# module at /home/ubuntu/projects/gitbenlewis/adata_science_tools/anndata_plotting/_venn_plots.py
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import pandas as pd

def venn_plot_2list(list1, list2,  set_label_list, plot_title,show_plot=True,return_df=True):

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    from matplotlib_venn import venn3

    set1 = set(list1)
    set2 = set(list2)
    labels_with_totals=[f'{set_label_list[0]}\nTot:{len(set1)}\n', 
                        f'{set_label_list[1]}\nTot:{len(set2)}\n', 
                        ]

    # Calculate intersections and unique elements
    only1 = set1 - set2 
    only2 = set2 - set1 
    set1_set2 = set1 & set2 
    # Create a dictionary with overlapping set information
    overlap_dict = {
        'Set Combination': ['Only Set 1', 'Only Set 2', 
                            'Set 1 & Set 2',
                            ],
        'Elements': [str(list(only1)), str(list(only2)), 
                     str(list(set1_set2)),
                    ]
    }

    # Convert the dictionary to DataFrame
    overlap_df = pd.DataFrame(overlap_dict)


    if show_plot:
            # Create the Venn diagram
        venn2([set1, set2, ], set_labels=labels_with_totals)
        # font size on title
        plt.title(f'{plot_title}\n\n', fontsize=16)
        # Display the plot
        plt.show()
    if return_df:
        return overlap_df

def venn_plot_3list(list1, list2, list3, set_label_list, plot_title,show_plot=False,return_df=False):

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2
    from matplotlib_venn import venn3

    set1 = set(list1)
    set2 = set(list2)
    set3 = set(list3)
    labels_with_totals=[f'{set_label_list[0]}\nTot:{len(set1)}\n', 
                        f'{set_label_list[1]}\nTot:{len(set2)}\n', 
                        f'{set_label_list[2]}\nTot:{len(set3)}\n']

    # Calculate intersections and unique elements
    only1 = set1 - set2 - set3
    only2 = set2 - set1 - set3
    only3 = set3 - set1 - set2
    set1_set2 = set1 & set2 - set3
    set1_set3 = set1 & set3 - set2
    set2_set3 = set2 & set3 - set1
    set1_set2_set3 = set1 & set2 & set3
    # Create a dictionary with overlapping set information
    overlap_dict = {
        'Set Combination': ['Only Set 1', 'Only Set 2', 'Only Set 3',
                            'Set 1 & Set 2', 'Set 1 & Set 3', 'Set 2 & Set 3',
                            'Set 1 & Set 2 & Set 3'],
        'Elements': [list(only1), list(only2), list(only3),
                     list(set1_set2), list(set1_set3), list(set2_set3),
                     list(set1_set2_set3)]
    }

    # Convert the dictionary to DataFrame
    overlap_df = pd.DataFrame(overlap_dict)


    if show_plot:
            # Create the Venn diagram
        venn3([set1, set2, set3], set_labels=labels_with_totals)
        # font size on title
        plt.title(f'{plot_title}\n\n', fontsize=16)
        # Display the plot
        plt.show()
    if return_df:
        return overlap_df


def geneset_enrichment_venn(universe, geneset, hits,
                            dataset_label="DEGs",
                            geneset_label="GeneSet",
                            plot_title="",
                            shift_overlap=True,
                            shift_overlap_labely=0.1):
    import numpy as np
    from scipy.stats import hypergeom
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    universe = set(universe)
    geneset = set(geneset) & universe
    hits = set(hits) & universe

    overlap = hits & geneset

    M = len(universe)          # universe size
    n = len(geneset)           # geneset size
    N = len(hits)              # hits size
    x = len(overlap)           # overlap size

    rv = hypergeom(M, n, N)

    # Enrichment p-value: P(X >= x)
    pvalue = rv.sf(x - 1)

    print({
        "universe_M": M,
        "geneset_n": n,
        "hits_N": N,
        "overlap_x": x,
        "p_enrichment": pvalue
    })

    venn = venn2([hits, geneset],
                 set_labels=(f"{dataset_label} (N={N})",
                             f"{geneset_label} (n={n})"))

    # Overlap label
    lab = venn.get_label_by_id("11")
    if lab is not None:
        lab.set_text(f"overlap={x}\np={pvalue:.2e}")
        if shift_overlap:
            lab.set_y(shift_overlap_labely)

    plt.title(
        f"{plot_title}\nUniverse M={M} | {dataset_label} N={N} | {geneset_label} n={n} | overlap x={x}",
        fontsize=12
    )
    plt.show()

    return {"M": M, "n": n, "N": N, "x": x, "p_enrichment": pvalue, "overlap": overlap}


def geneset_enrichemnt_ol_ven_M_n_N_x(
    M_set, n_set, N_set,
    dataset_label="dataset_label",
    geneset_label="geneset_label",
    plot_title="plot_title",
    shift_overlap=True,
    shift_overlap_labely=0.1
        ):
            '''
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
            ' Total detected genes =', M,
            ' Total GeneSet genes =', n,
            ' Total GEX hit genes =', N,
            ' GEX with GeneSet =', x,
            ' hypergeometric p-value =', pvalue
            '''
            import numpy as np
            from scipy.stats import hypergeom
            import matplotlib.pyplot as plt
            from matplotlib_venn import venn2

            # Cast to sets
            M_set, n_set, N_set = set(M_set), set(n_set), set(N_set)

            # Overlap
            x_set = N_set.intersection(n_set)

            # Counts
            M = len(M_set)
            n = len(n_set)
            N = len(N_set)
            x = len(x_set)

            rv = hypergeom(M, n, N)

            # Correct enrichment p-value: P(X >= x)
            pvalue = rv.sf(x - 1)

            print([
                ' Total detected genes =', M,
                ' Total GeneSet genes =', n,
                ' Total GEX hit genes =', N,
                ' GEX with GeneSet =', x,
                ' hypergeometric p-value =', pvalue
            ])

            venn = venn2(
                [N_set, n_set],
                set_labels=(
                    f'Total DEGs=({N})\n',
                    f'Total GeneSet= {n}\n'
                    f'Overlap={x}\n'
                    f'pvalue={pvalue:.5e}'
                )
            )

            # Shift overlap label if requested
            if shift_overlap:
                overlap_label = venn.get_label_by_id('11')
                if overlap_label:
                    overlap_label.set_y(shift_overlap_labely)

            plt.title(
                f'{plot_title}\n'
                f'{M} total genes in the universe set.\n'
                f'DEGs={N}\n'
                f'Total GeneSet genes={n}\n'
                f'Overlap of DEG with GeneSet={x}\n',
                fontsize=16
            )

            plt.show()
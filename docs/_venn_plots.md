# `_venn_plots`

Venn and overlap-enrichment helpers from `_plotting/_venn_plots.py`.

Important package note: this module exists in `_plotting/`, but it is not currently re-exported by `_plotting/__init__.py`. Use a direct submodule import when needed.

```python
from adata_science_tools._plotting._venn_plots import venn_plot_2list
```

## Public functions

- `venn_plot_2list`
- `venn_plot_3list`
- `geneset_enrichment_venn`
- `geneset_enrichemnt_ol_ven_M_n_N_x`

## `venn_plot_2list`

Creates a two-set Venn diagram and optionally returns a summary `DataFrame`.

```python
overlap_df = venn_plot_2list(
    list1=genes_a,
    list2=genes_b,
    set_label_list=["A", "B"],
    plot_title="Overlap of A and B",
    show_plot=True,
    return_df=True,
)
```

Important behavior:

- Inputs are converted to Python sets before overlap calculation.
- The returned `DataFrame` summarizes unique and shared members as stringified lists.
- `show_plot=False` suppresses the Matplotlib rendering.

## `venn_plot_3list`

Creates a three-set Venn diagram and optionally returns an overlap summary `DataFrame`.

Important behavior:

- Inputs are converted to sets.
- The returned `DataFrame` keeps the `Elements` column as Python lists rather than strings.
- The default arguments are `show_plot=False` and `return_df=False`.

## `geneset_enrichment_venn`

Computes a hypergeometric enrichment p-value and displays a Venn diagram for:

- `universe`
- `geneset`
- `hits`

The function returns a dict with:

- `M`
- `n`
- `N`
- `x`
- `p_enrichment`
- `overlap`

Important behavior:

- All three inputs are intersected with the supplied universe.
- The overlap label is rewritten to include the overlap count and enrichment p-value.
- The function always shows the plot.

## `geneset_enrichemnt_ol_ven_M_n_N_x`

This legacy helper also computes a hypergeometric enrichment display from three sets.

Important behavior:

- The function name is misspelled in the source and docs to match the current API.
- It prints the universe and overlap statistics.
- It shows the plot.
- The current implementation does not return a value.

## Coverage note

This module is documented from source code rather than dedicated tests.

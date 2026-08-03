# `_venn_plots`

Public Venn-diagram and overlap-enrichment helpers from
`_plotting/_venn_plots.py`.

```python
from adata_science_tools import venn_plot_2list
```

## Public functions

- `venn_plot_2list`
- `venn_plot_3list`
- `geneset_enrichment_venn`
- `geneset_enrichemnt_ol_ven_M_n_N_x`

## `venn_plot_2list`

Creates a two-set Venn diagram and optionally returns a summary `DataFrame`.

```python
def venn_plot_2list(
    list1,
    list2,
    set_label_list,
    plot_title,
    show_plot=True,
    return_df=True,
):
```

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

<img src="assets/plotting_gallery/venn_plot_2list__two_set_overlap.png" alt="Two feature-set overlap" width="720">

*`two_set_overlap` — Fixed synthetic identifiers demonstrate the two exclusive regions, shared region, and deduplicated set totals. [Data provenance](plotting_gallery.md#data-and-analysis-provenance).*

Important behavior:

- Inputs are converted to Python sets before overlap calculation.
- Duplicate input values therefore count once.
- The returned `DataFrame` summarizes the two exclusive regions and the shared
  region. Its `Elements` values remain stringified lists for compatibility.
- Members in the returned regions are ordered deterministically.
- `show_plot=False` suppresses rendering, while `return_df=False` suppresses
  the table return.

## `venn_plot_3list`

Creates a three-set Venn diagram and optionally returns a summary `DataFrame`
for all seven exclusive regions.

```python
def venn_plot_3list(
    list1,
    list2,
    list3,
    set_label_list,
    plot_title,
    show_plot=False,
    return_df=False,
):
```

<img src="assets/plotting_gallery/venn_plot_3list__three_set_overlap.png" alt="Three feature-set overlap" width="720">

*`three_set_overlap` — Fixed synthetic identifiers populate every exclusive two-set and three-set region. [Data provenance](plotting_gallery.md#data-and-analysis-provenance).*

Important behavior:

- Inputs are converted to sets and duplicate values count once.
- The returned `Elements` cells are deterministically ordered Python lists.
- The defaults preserve the existing non-rendering, no-return behavior;
  request `show_plot=True` and/or `return_df=True` explicitly.

## `geneset_enrichment_venn`

Computes and plots an upper-tail hypergeometric enrichment test for the overlap
between a selected dataset and a gene set.

```python
def geneset_enrichment_venn(
    universe,
    geneset,
    hits,
    dataset_label="DEGs",
    geneset_label="GeneSet",
    plot_title="",
    shift_overlap=True,
    shift_overlap_labely=0.1,
):
```

<img src="assets/plotting_gallery/geneset_enrichment_venn__universe_filtered.png" alt="Gene-set enrichment overlap" width="720">

*`universe_filtered` — Fixed synthetic sets include out-of-universe identifiers to demonstrate filtering before the library-derived hypergeometric test. [Data and analysis provenance](plotting_gallery.md#data-and-analysis-provenance).*

Important behavior:

- `geneset` and `hits` are intersected with `universe` before counts are
  calculated.
- The returned `M`, `n`, `N`, and `x` values are the universe, filtered gene
  set, filtered hit set, and overlap sizes, respectively.
- `p_enrichment` is `P(X >= x)`, calculated as
  `scipy.stats.hypergeom(M, n, N).sf(x - 1)`.
- `overlap` is returned as a Python set.
- The overlap label includes `x` and the enrichment p-value. The function
  prints the same statistics and always calls `plt.show()`.

## `geneset_enrichemnt_ol_ven_M_n_N_x`

This misspelled legacy name is retained for compatibility. New code should use
`geneset_enrichment_venn`, which explicitly filters its inputs to the supplied
universe and returns the calculated statistics.

```python
def geneset_enrichemnt_ol_ven_M_n_N_x(
    M_set,
    n_set,
    N_set,
    dataset_label="dataset_label",
    geneset_label="geneset_label",
    plot_title="plot_title",
    shift_overlap=True,
    shift_overlap_labely=0.1,
):
```

<img src="assets/plotting_gallery/geneset_enrichemnt_ol_ven_M_n_N_x__replacement_smoke.png" alt="Legacy gene-set enrichment overlap" width="720">

*`replacement_smoke` — Compatibility rendering for the legacy API; use `geneset_enrichment_venn` for new analyses. [Data and analysis provenance](plotting_gallery.md#data-and-analysis-provenance).*

Important legacy behavior:

- Callers must ensure `n_set` and `N_set` are subsets of `M_set`; this helper
  does not filter them to its universe before constructing the hypergeometric
  distribution.
- `dataset_label` and `geneset_label` remain accepted for signature
  compatibility but are not used in the rendered labels.
- The helper prints its statistics, always calls `plt.show()`, and returns
  `None`.

## Regression coverage

`tests/test_venn_plots.py` covers public exports and signatures, deterministic
two-set and three-set regions, optional rendering and returns, universe
filtering, upper-tail p-values, missing zero-overlap labels, and the legacy
contract. The deterministic plotting gallery exercises all four renderers.

# Optional feature-label fallback

Pass `feature_label_fallback="alternate_column"` to resolve missing display labels from another column of the **same feature table**. The default, `None`, retains the previous label path. All existing arguments retain their names, order, and defaults; the new argument is appended.

A preferred value is missing when null (None, pd.NA, NaN), empty, whitespace-only, or the case-insensitive literal "nan" (including surrounding whitespace). Usable preferred values are preserved exactly, including surrounding whitespace. Fallback values use the same missingness rule. If both are missing, the original preferred value is retained and the plot's existing missing-label behavior applies. Existing truncation limits still apply.

An explicitly configured fallback column must exist, even when preferred labels are valid. If the preferred column is disabled or absent, its previous index/error behavior remains in effect; fallback does not replace feature IDs. Resolution does not modify input tables or AnnData. Numeric effects, p-values, thresholds, ordering, and selection are not derived from resolved display text. Ranked volcano ties still use original labels for backward-compatible selection. Forest labels are resolved rowwise and must still agree across grouped rows for each ID. A preferred label in one group and a different fallback in another raises the existing conflicting-label error; choose a consistent fallback for grouped results.

## Supported functions and source tables

| Functions | Preferred-label argument | Fallback source |
| --- | --- | --- |
| volcano_plot_generic; volcano_plot_sns_single_comparison_generic (legacy) | feature_label_col | input _df |
| forest | feature_label_col | var_df or adata.var |
| barh_column; l2fc_dotplot_single; l2fc_dotplot_column | feature_label_vars_col | var_df, otherwise adata.var |
| barh_l2fc_dotplot_column; barh_dotplot_dotplot_column; barh_dotplot_dotplot_dotplot_column; barh_4X_dotplot_column | feature_label_vars_col | var_df, otherwise adata.var |
| datapoints_effect_panels_column | feature_label_vars_col | var_df, otherwise adata.var (not expression or observation tables) |
| plot_column_of_bar_h_2groups_GEX_adata; plot_column_of_bar_h_2groups_with_l2fc_dotplot_GEX_adata (legacy) | feature_label_vars_col | var_df, otherwise adata.var |
| timeseries_paired_datapoints; plot_paired_point_anndata (legacy) | feature_name_label_col | adata.var |
| l2fc_pvalue_dotplot_protein_metabolite; l2fc_pvalue_dotplot_gex (legacy) | analyte_label_column | diff_tests |

Functions without a separate feature-display-label column are unchanged. For example, vbar_l2fc_dotplot_column uses feature_column as identity; meta_forest labels study/summary rows, not feature metadata. Generic category, axis, enrichment-term, and observation labels are not feature-label fallbacks.

## Synthetic example

```python
import pandas as pd
import adata_science_tools as adtl

results = pd.DataFrame({
    "gene_names": ["GENE_A", " ", None],
    "stable_id": ["synthetic_a", "synthetic_b", "synthetic_c"],
    "log2FoldChange": [-1.2, 0.8, 1.5],
    "pvalue": [0.001, 0.02, 0.005],
})
ax = adtl.volcano_plot_generic(
    results, feature_label_col="gene_names",
    feature_label_fallback="stable_id", label_top_features=True,
    label_layout="ranked_columns", n_top_features=3,
)
```

The labels are GENE_A, synthetic_b, and synthetic_c. The source table, selected rows, effects, and p-values remain unchanged.

## Legacy plotting limits

The two legacy analyte dotplots retain their original categorical coordinates and row eligibility. Existing overlapping original labels remain overlapping, with distinct resolved labels joined by " / " on their shared row; fallback does not add, remove, or move points. On pandas versions where string conversion retains nulls, seaborn may omit null-label rows as before. Prefer l2fc_dotplot_single for separate feature-ID-based rows. Legacy expression bar bootstrap intervals remain stochastic; tests fix their RNG without altering the library.

# `_IO`

Core I/O helpers for saving `AnnData` objects and converting them into analysis-ready `DataFrame` tables.

This module provides:

- `save_dataset`
- `make_df_obs_adataX`

These helpers are used by higher-level workflows such as model fitting and expectation correction. In particular, [`_model_fit.md`](_model_fit.md) depends on `make_df_obs_adataX(...)`, and [`_expectation_based_covar_correction.md`](_expectation_based_covar_correction.md) uses `save_dataset(...)` in its wrapper workflow.

## `save_dataset`

`save_dataset(...)` writes one `AnnData` object to a bundle of files with a shared basename.

```python
import adata_science_tools as adtl

adtl.save_dataset(
    adata,
    "results/corrected_dataset.h5ad",
)
```

To skip `.obsm` CSV exports or save only selected `.obsm` keys:

```python
adtl.save_dataset(
    adata,
    "results/corrected_dataset.h5ad",
    save_obsm=False,
)

adtl.save_dataset(
    adata,
    "results/corrected_dataset.h5ad",
    obsm_keys=["ref_values", "target_values"],
)
```

### Output files

If the output path is `results/corrected_dataset.h5ad`, the helper writes:

- `results/corrected_dataset.h5ad`
- `results/corrected_dataset.obs.csv`
- `results/corrected_dataset.var.csv`
- `results/corrected_dataset.X.csv`
- one CSV per layer, named like `results/corrected_dataset.layer.<layer_name>.csv`
- one CSV per saved `.obsm` table, named like `results/corrected_dataset.obsm.<obsm_key>.csv`

If the path does not end in `.h5ad`, the helper treats it as the basename and still writes the same set of files with `.h5ad` and CSV suffixes.

### Important behavior

- Parent directories are created automatically.
- `adata.X` is converted to dense before writing the `.X.csv` file if needed.
- Each layer is also written to CSV.
- Layer names containing `/` are normalized to `_` in the layer CSV filenames.
- By default, each 2D `.obsm` value aligned to observations is written to CSV.
- `.obsm` names containing `/` are normalized to `_` in CSV filenames.
- Requested missing `.obsm` keys raise `KeyError`.
- Sanitized `.obsm` filename collisions raise `ValueError` before files are written.
- Unsupported `.obsm` values, such as arrays that are not 2D, are logged and skipped.
- The helper always writes the `.h5ad` plus CSV bundle together. There is no flag to save only one of those outputs.

### `.obsm` table exports

For `.obsm` entries that are already `pandas.DataFrame` objects,
`save_dataset(...)` preserves the DataFrame index and columns.

For array-like or sparse 2D `.obsm` entries, the helper uses `adata.obs_names`
as the row index. If the number of `.obsm` columns equals `adata.n_vars`, it
uses `adata.var_names` as column labels; otherwise it writes deterministic
columns named `dim_0`, `dim_1`, and so on.

This is useful for source-value tables created by
`ref_vs_target_adata(save_source_values_obsm=True)`, which stores paired
reference and target source matrices in `.obsm["ref_values"]` and
`.obsm["target_values"]` by default:

```python
post_minus_pre = adtl.ref_vs_target_adata(
    adata,
    pair_by_key="Subject_ID",
    save_source_values_obsm=True,
)

adtl.save_dataset(post_minus_pre, "results/post_minus_pre.h5ad")
```

This writes:

- `results/post_minus_pre.obsm.ref_values.csv`
- `results/post_minus_pre.obsm.target_values.csv`

## `make_df_obs_adataX`

`make_df_obs_adataX(...)` builds a `pandas.DataFrame` from `AnnData` expression data and, optionally, prepends `adata.obs`.

```python
df = adtl.make_df_obs_adataX(
    adata,
    layer="pgml",
    include_obs=True,
)
```

Typical uses:

- build a combined `obs_X_df` for OLS or MixedLM fitting,
- inspect an expression matrix together with observation metadata,
- choose alternate feature labels or a different observation index.

## Data source selection

Current precedence is:

- `adata.raw.X` when `use_raw=True` and `layer is None`,
- `adata.raw.layers[layer]` when `use_raw=True` and `layer` is provided,
- `adata.layers[layer]` when `layer` is present on the main object,
- otherwise `adata.X`.

Implementation note:

- The function prints which source it used rather than logging through a `Logger`.

Practical caution:

- In standard `AnnData`, `adata.raw` typically exposes `X` and `var`, but not arbitrary layers. The current implementation still attempts `adata.raw.layers[layer]` when both `use_raw=True` and `layer` are supplied, so the safest raw-data usage is `use_raw=True` with `layer=None`.

## Column labeling

By default, expression columns are labeled with `adata.var_names`.

You can override that with `varcolumns`:

- `None`: use `adata.var_names`
- `str`: use one column from `adata.var`
- `list` of length `1`: same as a single-string column selection
- `list` of length `2` or more: build a `pandas.MultiIndex` from multiple `adata.var` columns

Example:

```python
df = adtl.make_df_obs_adataX(
    adata,
    varcolumns=["feature_class", "gene_name"],
    include_obs=False,
)
```

## Row indexing

By default, the DataFrame index is `adata.obs_names`.

If `index` is provided, it is interpreted as a column name in `adata.obs` and that column becomes the DataFrame index.

Example:

```python
df = adtl.make_df_obs_adataX(
    adata,
    index="SubjectID",
    include_obs=True,
)
```

## `include_obs`

When `include_obs=True`, the function concatenates `adata.obs` in front of the expression matrix columns.

When `include_obs=False`, the result contains only the expression values.

This distinction matters in [`_model_fit.md`](_model_fit.md), where the fit wrappers build a combined `obs_X_df` for downstream formula-based model fitting.

## Dense conversion and scale

If the selected matrix is sparse, the helper converts it to dense with `toarray()` before building the DataFrame.

That is convenient for downstream formula-based modeling, but it can increase memory use substantially on large matrices.

## Current limitations and caveats

The docs for this module are based on the current implementation rather than direct `_io` regression tests in this repo.

Important current caveats:

- The default `varcolumns` path is based on `adata.var_names` even when `use_raw=True`.
- If `adata.raw.var_names` differ from `adata.var_names`, column labels may not match the raw matrix automatically.
- The helper imports `AnnData` locally but does not enforce the input type beyond expected attribute access.

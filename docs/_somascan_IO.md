# `_somascan_IO`

SomaScan-specific import and cleanup helpers for working with `.adat` files and related observation metadata.

This module provides:

- `read_adat_2_AnnData`
- `soma_fill_sampletype_obs_values`
- `soma_make_adata_index_unique_by_merge`
- `make_df_obs_adataX_soma`

These functions are parser- and dataset-specific. Unlike the `_tools` module docs, this page is based on the current code rather than dedicated regression tests in `tests/`.

## `read_adat_2_AnnData`

`read_adat_2_AnnData(...)` parses a SomaLogic `.adat` file into an `AnnData` object.

### Full signature

```python
def read_adat_2_AnnData(path_or_buf: Union[str, TextIO]) -> AnnData:
```

```python
import adata_science_tools as adtl

adata = adtl.read_adat_2_AnnData("input/example.adat")
```

### Accepted inputs

- a filesystem path as `str`
- an already-open text buffer

### Returned structure

The parser constructs:

- `adata.X` from the RFU matrix
- `adata.obs` from row metadata
- `adata.var` from column metadata
- `adata.uns` from header metadata

Important behavior:

- all values in `adata.uns` are coerced to strings before return;
- this is done to avoid issues when writing the object back to `.h5ad`.

### Naming caveat

The function does not automatically set `adata.obs_names` and `adata.var_names` to SomaScan identifiers such as `SampleId` or `SeqId`.

The example comments in the source show the intended next steps:

- choose identifier columns,
- assign them to `obs_names` and `var_names`,
- make them unique if necessary.

## `soma_fill_sampletype_obs_values`

`soma_fill_sampletype_obs_values(...)` copies values from one donor obs column into one or more target obs columns for specific sample types.

### Full signature

```python
def soma_fill_sampletype_obs_values(
    adata: AnnData,
    donor_obs_column: str = 'SampleType',
    donor_obs_col_values_to_paste: list[str] | None = None,
    obs_columns_toFix: list[str] | None = None,
    make_copy: bool = False
):
```

```python
adata = adtl.soma_fill_sampletype_obs_values(
    adata,
    donor_obs_column="SampleType",
    donor_obs_col_values_to_paste=["QC", "Buffer", "Calibrator"],
    obs_columns_toFix=["AliquotingNotes", "AssayNotes", "TimePoint"],
    make_copy=True,
)
```

### Default behavior

- `donor_obs_column` defaults to `SampleType`
- `donor_obs_col_values_to_paste` defaults to `["QC", "Buffer", "Calibrator"]`
- `make_copy=False` modifies the original `AnnData`

### Important behavior

- if `make_copy=True`, the function returns a modified copy;
- if the donor column is missing, it raises `KeyError`;
- if no target columns are provided, it prints a note and returns without changes;
- if requested target columns are missing from `adata.obs`, it filters them down and prints a note;
- if no rows match the donor values, it prints a note and returns without changes;
- matched donor values are broadcast across all requested target columns.

The function uses `print(...)` for its status notes rather than structured logging.

## `soma_make_adata_index_unique_by_merge`

`soma_make_adata_index_unique_by_merge(...)` makes `adata.obs_names` more unique by appending another obs column value to selected rows.

### Full signature

```python
def soma_make_adata_index_unique_by_merge(
    adata: AnnData,
    donor_obs_column: str = 'Barcode2d',
    mask: pd.Series | None = None,
    duplicates_index_only: bool = True,
    ensure_global_unique: bool = False,
    make_copy: bool = False,
) -> AnnData:
```

```python
adata = adtl.soma_make_adata_index_unique_by_merge(
    adata,
    donor_obs_column="Barcode2d",
    duplicates_index_only=True,
    ensure_global_unique=True,
    make_copy=True,
)
```

### How it works

- start from the current `obs_names`
- pick rows using `mask`, or all rows if `mask is None`
- optionally restrict to duplicate index values when `duplicates_index_only=True`
- replace each selected name with `<old_name>_<donor_value>`
- optionally call `obs_names_make_unique()` when `ensure_global_unique=True`

### Important behavior

- `duplicates_index_only=True` is the default, so the helper only rewrites duplicated names unless you disable that behavior;
- `make_copy=False` modifies the original object in place;
- `mask` is intersected with the duplicate-name mask when `duplicates_index_only=True`.

## `make_df_obs_adataX_soma`

`make_df_obs_adataX_soma(...)` is a SomaScan-specific variant of the generic DataFrame helper from [`_IO.md`](_IO.md).

### Full signature

```python
def make_df_obs_adataX_soma(adata,layer=None,index=None,varcolumns=None,include_obs=True):
```

```python
df = adtl.make_df_obs_adataX_soma(
    adata,
    layer=None,
    include_obs=True,
)
```

It supports:

- choosing `adata.X` or a named layer,
- using `adata.obs_names` or an obs column as the row index,
- selecting one or more `adata.var` columns as feature labels,
- concatenating `adata.obs` in front of the expression matrix.

### Differences from `make_df_obs_adataX`

- no `use_raw` option
- no sparse-to-dense guard before building the `DataFrame`
- otherwise similar `varcolumns` handling, including MultiIndex output when multiple var columns are supplied

## Practical guidance

- Use `read_adat_2_AnnData(...)` as the initial import step for `.adat` files.
- Set and clean `obs_names` and `var_names` explicitly after import.
- Use `soma_fill_sampletype_obs_values(...)` to patch metadata columns for special sample types such as QC or buffer rows.
- Use `soma_make_adata_index_unique_by_merge(...)` when imported sample identifiers are duplicated.
- Prefer the generic helper in [`_IO.md`](_IO.md) unless you specifically need the SomaScan-flavored DataFrame export path.

## Current caveats

This module has no direct regression tests in `tests/`, so the page documents current code behavior and known practical caveats:

- `read_adat_2_AnnData(...)` depends on the external `somadata` package;
- `read_adat_2_AnnData(...)` leaves naming decisions to the caller after import;
- `make_df_obs_adataX_soma(...)` does not perform the sparse-to-dense conversion guard used in the generic `_IO` helper.

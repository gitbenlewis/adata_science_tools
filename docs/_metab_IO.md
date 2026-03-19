# `_metab_IO`

Metabolon Excel ingestion and export helpers for building `AnnData` objects from metabolomics workbooks.

This module currently exposes one public function:

- `metabolon_excel_2_adata_h5ad_csv`

It is a high-level parser that:

- reads assay, sample-metadata, and chemical-annotation sheets from an Excel workbook,
- constructs an `AnnData`,
- adds multiple layers from additional sheets,
- optionally merges external obs and var metadata,
- optionally writes output datasets to `.h5ad` plus CSV bundles.

This page is based on the current implementation in [\_metab_IO.py](_io/_metab_IO.py). There do not appear to be direct `_io` regression tests for this function in this repo.

## Basic parsing workflow

```python
import logging
from pathlib import Path

import adata_science_tools as adtl

logger = logging.getLogger("metab_io")

adata = adtl.metabolon_excel_2_adata_h5ad_csv(
    metabolon_excel_file="input/metabolon.xlsx",
    output_dir=Path("results"),
    output_filename="dataset.metab",
    logger=logger,
)
```

By default, the function reads these workbook sheets:

- assay data from `Batch-norm Imputed Data`
- observation metadata from `Sample Meta Data`
- variable metadata from `Chemical Annotation`

The default var index column is:

- `CHEM_ID`

## Layer creation

The parser also tries to add layers from these sheet names by default:

- `Volume-normalized Data`
- `Log Transformed Data`
- `Batch-normalized Data`
- `Batch-norm Imputed Data`
- `Peak Area Data`

Layer keys are normalized by lowercasing and replacing spaces and hyphens with underscores.

Examples:

- `Volume-normalized Data` becomes `volume_normalized_data`
- `Batch-norm Imputed Data` becomes `batch_norm_imputed_data`

Important behavior:

- each layer sheet is aligned to the parsed obs and var indexes before assignment;
- if a layer sheet cannot be loaded or aligned, the function prints a skip message and continues.

## Returned `AnnData`

The base parsed object uses:

- `assay_data.values` as `adata.X`
- the sample metadata sheet as `adata.obs`
- the chemical annotation sheet as `adata.var`

Additional behavior:

- the first column of the obs metadata sheet becomes the obs index;
- `index_col_for_var_metadata` becomes the var index;
- both indexes are cast to strings;
- assay columns are stripped of surrounding whitespace;
- an `adata.obs["metab_data_table_order"]` column is added with 1-based row order;
- object-typed obs columns are converted to strings.

## Save behavior

The function name and parameters suggest separate control over `.h5ad` and CSV outputs, but the current implementation behaves more narrowly.

### Current save path behavior

When `save_h5ad=True`, the function calls an internal `_save_dataset(...)` helper that writes:

- `<output_dir>/<output_filename>.h5ad`
- `<output_dir>/<output_filename>.obs.csv`
- `<output_dir>/<output_filename>.var.csv`
- `<output_dir>/<output_filename>.X.csv`
- one CSV for each layer

In other words:

- the current save path writes both `.h5ad` and CSV exports together;
- `also_save_csvs` is accepted by the public function, but it is not currently used to change save behavior.

The same pattern applies to the merged-metadata output path:

- `save_plus_metadata_h5ad=True` writes the plus-metadata `.h5ad` and CSV bundle;
- `also_plus_metadata_save_csvs` is currently accepted but not used to gate those CSV saves.

## External metadata merge

Set `merge_external_metadata=True` to enable optional obs and var metadata merges.

### Obs merge path

Obs metadata can be merged when all of these are provided:

- `external_obs_metadata_2_merge_file`
- `merge_key_in_external_obs_metadata`
- `merge_key_in_raw_obs_metadata`

Optional behavior:

- if `column_in_metadata_to_set_as_index` is provided, the merged obs table is reindexed to that column;
- object-typed obs columns are converted to strings after merge.

### Var merge path

Var metadata can be merged when:

- `external_var_metadata_2_merge_file` is provided
- `merge_key_in_external_var_metadata` is provided

Optional behavior:

- `columns_in_external_var_metadata_to_use` limits the imported external var columns;
- duplicate external var keys are dropped by keeping the first occurrence;
- merged object-typed var columns are converted to strings.

Current implementation note:

- `merge_key_in_raw_var_metadata` is accepted in the public signature but is not currently used in the var-merge call.

## Example with external metadata

```python
adata = adtl.metabolon_excel_2_adata_h5ad_csv(
    metabolon_excel_file="input/metabolon.xlsx",
    output_dir=Path("results"),
    output_filename="dataset.metab",
    logger=logger,
    merge_external_metadata=True,
    external_obs_metadata_2_merge_file="input/obs_metadata.csv",
    merge_key_in_external_obs_metadata="sample_id",
    merge_key_in_raw_obs_metadata="SAMPLE_NAME",
    column_in_metadata_to_set_as_index="sample_id",
    external_var_metadata_2_merge_file="input/var_metadata.csv",
    merge_key_in_external_var_metadata="CHEM_ID",
    columns_in_external_var_metadata_to_use=["pathway", "super_pathway"],
    save_plus_metadata_h5ad=True,
    plus_metadata_file_name="dataset.plus_metadata",
)
```

## Current implementation caveats

This function is useful, but the current implementation has a few practical constraints that the docs should make explicit.

### Logger expectations

The save and merge-enabled paths call `logger.info(...)` directly without guarding for `logger is None`.

Practical guidance:

- pass a real `Logger` whenever you enable `save_h5ad`, `save_plus_metadata_h5ad`, or metadata-merge logging paths.

### Output directory expectations

The current save paths join filenames under `output_dir`, so `output_dir` should be provided whenever saving is enabled.

Practical guidance:

- because `save_h5ad=True` is the current default, the safest explicit usage is to pass both `output_dir` and `logger`, or to disable saving when you only want the returned `AnnData`.

### Metadata-merge branch dependency

The merge branch initializes the working `adata` inside the external-obs merge path.

Practical guidance:

- the safest merge workflow is to provide the obs-merge inputs when `merge_external_metadata=True`, especially if you also want var metadata merge.

### No direct `_io` tests

There are no direct regression tests for this module in `tests/`, so this page documents the current code path and current caveats rather than a separately test-locked contract.

## When to use this module

- Use this function when your source data is a Metabolon Excel workbook with the expected assay and metadata sheets.
- Use [`_IO.md`](_IO.md) for the generic dataset-save helpers shared across the rest of the package.
- Use [`_model_fit.md`](_model_fit.md) and [`_expectation_based_covar_correction.md`](_expectation_based_covar_correction.md) once the parsed `AnnData` is ready for analysis.

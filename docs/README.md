# adata_science_tools docs

This directory holds module-level documentation for public `adata_science_tools` APIs.

## Available pages

- [`_IO.md`](_IO.md): core dataset save and AnnData-to-DataFrame helpers used across the package.
- [`_diff_test.md`](_diff_test.md): differential testing across independent, paired, and nested paired group comparisons, including CSV-backed input workflows.
- [`_expectation_based_covar_correction.md`](_expectation_based_covar_correction.md): expectation-model fitting, artifact export, prediction, regression-based correction, and residual or ratio transforms for `AnnData` objects.
- [`_metab_IO.md`](_metab_IO.md): Metabolon Excel ingestion, layer creation, optional metadata merge, and dataset export behavior.
- [`_model_fit.md`](_model_fit.md): OLS and MixedLM model-fitting APIs, summary-table schemas, filtering support, and model-spec sidecar behavior.
- [`_somascan_IO.md`](_somascan_IO.md): SomaScan `.adat` ingestion, sample-type cleanup, index normalization, and SomaScan-specific DataFrame export helpers.

## Notes

- These pages document the current implementation in `adata_science_tools`, not an earlier design draft.
- The first source of truth for behavior is the code in `_tools/`, `_io/`, and the tests in `tests/`.

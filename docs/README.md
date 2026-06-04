# adata_science_tools docs

This directory holds module-level documentation for public `adata_science_tools` APIs.

## Available pages

- [`_IO.md`](_IO.md): core dataset save and AnnData-to-DataFrame helpers used across the package.
- [`_column_plots.md`](_column_plots.md): horizontal bar plots, log2 fold-change dotplots, and multi-panel composite figure builders.
- [`_corr_dotplots.md`](_corr_dotplots.md): correlation scatterplots, subgroup fit overlays, and rank-comparison helpers.
- [`_diff_test.md`](_diff_test.md): differential testing across independent, paired, and nested paired group comparisons, including CSV-backed input workflows.
- [`_expectation_based_covar_correction.md`](_expectation_based_covar_correction.md): expectation-model fitting, artifact export, prediction, regression-based correction, and residual or ratio transforms for `AnnData` objects.
- [`_histograms.md`](_histograms.md): AnnData and wide-DataFrame histogram plotting with obs/var filters and subgroup overlays.
- [`_metab_IO.md`](_metab_IO.md): Metabolon Excel ingestion, layer creation, optional metadata merge, and dataset export behavior.
- [`_model_fit.md`](_model_fit.md): OLS and MixedLM model-fitting APIs, summary-table schemas, filtering support, and model-spec sidecar behavior.
- [`_paired_datapoints.md`](_paired_datapoints.md): paired ref/target or Pre/Post datapoint plots with obs/var filters, grouping, and a synthetic example image.
- [`_plots.md`](_plots.md): volcano plots, QQ plots for p-values, and paired time-series datapoint plots.
- [`_plots_depreciated.md`](_plots_depreciated.md): legacy plotting APIs kept for backward compatibility, with preferred modern replacements.
- [`_preprocessing.md`](_preprocessing.md): row-wise AnnData preprocessing helpers, including paired target-vs-reference transforms.
- [`_row_plots.md`](_row_plots.md): grouped bar-plus-swarm row layouts from plain `DataFrame` input.
- [`_simulate_data.md`](_simulate_data.md): covariate-driven dataset simulation helpers plus the config-driven example workflow used in `example_simulated_data/`.
- [`_somascan_IO.md`](_somascan_IO.md): SomaScan `.adat` ingestion, sample-type cleanup, index normalization, and SomaScan-specific DataFrame export helpers.
- [`_utils.md`](_utils.md): plotting helper functions for previewing and optionally saving color palettes.
- [`_venn_plots.md`](_venn_plots.md): Venn and enrichment-overlap utilities in the package directory that are not currently re-exported at the package top level.
- [`palettes.md`](palettes.md): named palette constants exposed as `adata_science_tools.palettes`.

## Notes

- These pages document the current implementation in `adata_science_tools`, not an earlier design draft.
- The first source of truth for behavior is the code in `_tools/`, `_io/`, `_simulate_data/`, and the tests in `tests/`.
- Within `_plotting`, direct regression coverage currently exists for `_corr_dotplots.py`, `_histograms.py`, and `paired_datapoints()`; the other plotting pages are based on current code and repo example usage.

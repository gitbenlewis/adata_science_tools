# adata_science_tools

## Documentation

- Published docs: [gitbenlewis.github.io/adata_science_tools](https://gitbenlewis.github.io/adata_science_tools/)
- Source docs: [docs/README.md](docs/README.md)
- Plotting gallery: [deterministic renderer examples](docs/plotting_gallery.md)
- Start with [simulation helpers](docs/_simulate_data.md) and [correlation dotplots](docs/_corr_dotplots.md).

Data-science tools that operate on `AnnData` objects.

## Setup

### Clone the repository

```bash
git clone https://github.com/gitbenlewis/adata_science_tools.git
cd adata_science_tools
```

### Create the conda environment

```bash
conda env create -f config/env_not_base.yaml -n not_base
conda activate not_base
```

If `not_base` already exists, update it in place instead of deleting it:

```bash
conda env update -f config/env_not_base.yaml -n not_base
```

### Run the self-contained plotting gallery

```bash
bash scripts/000_generate_plotting_gallery.bash
```

This regenerates the deterministic PNG catalog in
[`docs/assets/plotting_gallery`](docs/assets/plotting_gallery) and writes its run
log under `scripts/logs/`.

## Simulated data example

The repo includes a small config-driven simulated-data workflow in
`example_simulated_data/`. Its
[`config.yaml`](example_simulated_data/config/config.yaml) uses repository-relative
output paths, so the commands below are portable when run from the repository
root. Outputs resolve under `example_simulated_data/results/`.

```bash
conda activate not_base
python example_simulated_data/scripts/simulate_1_var_covar_age.py
python example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py
```

The default parameters simulate one feature, `simulated_feature`, from `Age`
and `case_control`, add residual `y` variance so the points are not constrained
to exact fit lines, and write both a baseline `AnnData` bundle and a dotplot.

The main example knobs are `beta` or `beta_age`, `beta_case_control`, `case_control_prob`, and `residual_stdev`.

See [`docs/_simulate_data.md`](docs/_simulate_data.md) for the simulation API and config details, and [`docs/_corr_dotplots.md`](docs/_corr_dotplots.md) for the plotting API.

Committed output snapshots: [baseline.h5ad](example_simulated_data/results/simulate_1_var_covar_age/baseline/baseline.h5ad) and [baseline.png](example_simulated_data/results/plot_dotplot_simulate_1_var_covar_age/baseline/baseline.png).

![baseline simulated dotplot](example_simulated_data/results/plot_dotplot_simulate_1_var_covar_age/baseline/baseline.png)

## Paired datapoints examples

[`adtl.paired_datapoints()`](docs/_paired_datapoints.md) can add a third x-axis
position for either the signed post-baseline difference or
`log2(post / baseline)`. The secondary y-axis is symmetric around zero in both
modes.

### Raw post-baseline difference

![Paired datapoints with raw post-baseline differences](docs/assets/plotting_gallery/paired_datapoints__difference_axis.png)

Opt-in legend metrics can summarize the finite post-filter values at baseline,
post, and the raw pairwise `post - baseline` position.

![Paired datapoints with raw post-baseline differences and per-position summary legend](docs/assets/plotting_gallery/paired_datapoints__difference_summary_legend.png)

### Post-over-baseline log2 fold change

![Paired datapoints with post-over-baseline log2 fold changes](docs/assets/plotting_gallery/paired_datapoints__log2fc_axis.png)

The log2FC summary uses valid pairwise `log2(post / baseline)` values rather
than a fold change calculated from the endpoint means.

![Paired datapoints with post-over-baseline log2 fold changes and per-position summary legend](docs/assets/plotting_gallery/paired_datapoints__log2fc_summary_legend.png)

## Example plots from `example_PMID_33969320`

These are committed output snapshots from a dataset-specific workflow, not a
fresh-clone runnable example. The workflow expects external study data and a
`code_library` checkout that are not included in this repository.

### Column plot

API: [`adtl.datapoints_effect_panels_column()`](docs/_column_plots.md#datapoints_effect_panels_column)
([source](_plotting/_column_plots.py))

[View plot file](example_PMID_33969320/results/figures/diff_datapoint_plots/COVID_over_NOT_D0_barh_l2fc_dotplot_FDR.png)
![COVID versus non-COVID day-zero distributions with a log2 fold-change effect panel](example_PMID_33969320/results/figures/diff_datapoint_plots/COVID_over_NOT_D0_barh_l2fc_dotplot_FDR.png)

### Volcano plot

API: [`adtl.volcano_plot_generic()`](docs/_plots.md#volcano_plot_generic)

[View plot file](example_PMID_33969320/results/figures/volcano_plots/COVID_over_NOT_D0_volcano_FDR.png)
![COVID versus non-COVID day-zero volcano plot](example_PMID_33969320/results/figures/volcano_plots/COVID_over_NOT_D0_volcano_FDR.png)

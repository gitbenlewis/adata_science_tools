# Flask web app

The optional web app provides code-free dataset loading, plotting, statistics,
and downloads through a browser. It calls this repository's existing scientific
functions. Start locally with one command after installing the dependencies.
Public deployment to `adata-science-tools.com` is supported by the application
structure but is **not configured or deployed by this change**.

## Local quick start (macOS or Linux)

From the repository root, create the scientific environment as described in the
[README](../README.md), then run:

```bash
conda activate not_base
python -m pip install -r config/requirements-web.txt
python scripts/run_web.py
```

Open **http://127.0.0.1:5000**. The launcher starts both the Flask development
server and one analysis worker. It binds only to localhost. Press Ctrl+C to stop
both. If port 5000 is occupied, use `python scripts/run_web.py --port 5055`.
The scientific environment must already be installed; the web requirements add
Flask and its dependencies, not a second scientific environment.

The **Tutorial** tab introduces the AnnData structure with the official diagram
bundled locally, explains X/obs/var and additional fields, and walks through the
upload-to-results workflow. The diagram's source and license are linked beneath it.

Under **Datasets**, select **Open demo dataset** for a deterministic synthetic study, or upload your
own data. Then choose **Analysis studio**, select features and a method, fill in
its controls, and select **Run analysis**. Results appear under **Results &
downloads**. The first three features are selected initially; **All features**
explicitly selects the full feature set. No Python code is required in the GUI.

Uploads and results are stored under `instance/web/`, which is ignored by Git.
Use `--data-dir /absolute/path` to keep them elsewhere. Reuse that path for the
server, worker, and account-management commands. The worker removes datasets
and their results 24 hours after upload, checking once per minute; active jobs
finish first. Stopping the worker pauses expiry cleanup. A browser refresh does
not cancel a queued or running job. An interrupted running job is marked failed
when the worker restarts, and can be submitted again.

The current worker uses POSIX file locking and is tested on macOS. Use Linux or
WSL for Windows. Native Windows worker support is not implemented.

## Analysis pipelines

**05 Analysis pipelines** offers three prebuilt workflows using the existing adapters:

| Workflow | Steps |
|---|---|
| Explore groups | Histograms, grouped datapoints, grouped feature means |
| Compare independent groups | Grouped datapoints, grouped means, Welch's t-test or Mann–Whitney U |
| Compare paired samples | Paired datapoints, grouped means, paired t-test or Wilcoxon signed-rank |

Set the matrix, features, observation filter, and metadata conversions in
**Analysis studio**, then review the selection in **Analysis pipelines**. Choose
the group column, comparison groups, and pairing column where required. No
normalization or automatic model selection is performed. Choose a test that
matches the experimental design and measurement scale.

Plots use at most the first 24 selected features; means and statistical tests use
the full selection, including that feature set for FDR correction. Descriptive
group plots and means include all groups retained by the observation filter;
tests compare the selected reference and target groups only.

**Run pipeline** queues all three jobs atomically: insufficient quota queues none.
Each step runs independently and has its own status, downloads, and analysis
record in **Results & downloads**, labeled with a shared pipeline run ID and step
number. A failed step does not cancel the others. Runs persist across page reloads
and use the same worker timeout, ownership checks, and retention as other analyses.

## Input formats

### H5AD

Upload one `.h5ad` file with a numeric `X` matrix and unique, nonempty observation
and feature identifiers. The original file retains its layers, raw matrix,
embeddings, and annotations. The studio offers `X`, each layer, and `raw` when
present. Unsupported linked/virtual HDF5 data and oversized expanded arrays are
rejected during import.

### CSV bundle

Upload all three files into their named fields. Their basenames may differ from
these examples. The layout matches [`save_dataset()`](_IO.md) exports:

| File | Rows | Columns |
|---|---|---|
| `adata.X.csv` | Observation identifiers in first column | Feature identifiers in header; numeric matrix values |
| `adata.obs.csv` | Same observation identifiers in first column | Observation metadata |
| `adata.var.csv` | Same feature identifiers in first column | Feature metadata; may contain zero metadata columns |

Metadata is aligned to the X matrix by identifier. Ordering may differ, but the
identifier sets must match exactly. Duplicate IDs, duplicate headers, nonnumeric
matrix values, and mismatches produce errors. No automatic transpose, renaming,
normalization, or dropping of unmatched samples is performed.

Identifiers and metadata strings preserve leading zeros and literal strings
such as `NA`. Blank cells are missing values. X numeric values are parsed as
numbers. CSV metadata starts as text to avoid silently changing categorical
codes. In **Metadata types**, explicitly choose numeric or categorical columns
for an analysis. Existing model APIs also coerce numeric-looking predictors;
this behavior is preserved and reported. The three CSVs cannot restore layers,
embeddings, `uns`, or all original metadata types; prefer H5AD to retain them.

## Available GUI analyses

| GUI method | Package API | Inputs / notes |
|---|---|---|
| Histograms | `adata_histograms()` | Selected features; optional group overlays |
| Grouped datapoints | `datapoints()` | Box, violin, or points; optional observation grouping |
| Paired datapoints | `paired_datapoints()` | Explicit reference, target, and subject; optional difference or log2FC panel |
| Correlation | `corr_dotplot()` | Numeric metadata or features; Pearson or Spearman |
| Longitudinal trajectories | `longitudinal_trajectories()` | One feature, subject ID, visit column, explicit comma-separated visit order |
| Category composition | `category_composition()` | Two observation categories; counts, fractions, or percentages |
| Differential testing | `diff_test()` | Welch's independent t-test, Mann–Whitney U, paired t-test, or Wilcoxon signed-rank |
| Linear models | `fit_smf_ols_models_and_summarize_adata()` | Selected predictors and features |
| Mixed-effects models | `fit_smf_mixedlm_models_and_summarize_adata()` | Predictors, random-intercept grouping column, REML or ML |
| Grouped means | `average_feature_expression()` | Group column; no automatic transformation |
| Volcano | `volcano_plot_generic()` | Explicit effect and p-value/FDR columns and thresholds |
| QQ | `qqplot()` | Explicit p-value/FDR column; use unadjusted p-values for conventional diagnostic interpretation |
| Distributions + effects | `datapoints_effect_panels_column()` | Observation groups plus precomputed feature effects and p-values |
| Forest | `forest()` | Precomputed estimate and confidence-bound columns |
| Export selection | AnnData H5AD writer | Selected matrix becomes X with selected obs/var annotations |

Result plots accept feature-level statistics from `adata.var` or a completed
differential-test/OLS/mixed-model run for the same dataset. Select **Plot these
results** on a statistical result to connect that table to the plotting form.
These renderers do not estimate effects or confidence bounds from observations.
When comparing plots with prior results, keep matrix, filters, grouping, and
metadata conversions consistent with the original run; its exact settings are
shown and downloadable. Result plots respect the selected feature identifiers.

The GUI is an explicit subset of the package APIs. Nested differential tests,
expectation correction, enrichment-table uploads, meta-analysis renderers,
deprecated renderers, and every advanced plotting keyword are not exposed yet.
They remain available through Python. Adding an adapter and form definition in
[`web/analysis.py`](../web/analysis.py) extends the GUI without altering the
scientific implementation.

## Scientific behavior and downloads

All runs operate on a separate selection. The input dataset stays unchanged.
The selected matrix becomes the adapter's working X, so a layer/raw selection
is explicit in the provenance. Only selected matrix values and obs/var are
copied; unused layers, raw, and embeddings are not copied into each analysis or
the **Export selected dataset** download.

The package's tests, missing-data handling, and FDR calculations are preserved.
Feature selection changes the multiple-testing scope. `diff_test()` may exclude
all-zero features according to its existing behavior. Paired analyses require
unique, nonmissing subject IDs within each compared group. Complete/unmatched
pair counts are recorded; the package controls its existing alignment behavior.
Group sizes describe the selected groups, not each feature's finite-value count.

Download figures as PNG, SVG, or PDF; tables as CSV; and `analysis.json` for
parameters, matrix/filter selection, input fingerprints, versions, warnings,
and Git revision/dirty status when available. `reproduce.py` replays a run using
the same checkout, environment, `analysis.json`, and original input dataset
named `data.h5ad`. For CSV inputs, first reconstruct it with
`web.data.load_csv_bundle()`. Reproduction is optional; GUI users need no code.

## Optional username/password login

Login is available but not required by default. Without login, access belongs
to a signed browser session cookie. Clearing that cookie loses access to the
anonymous workspace; its data still expires normally. Signing in opens the
account's workspace and does not transfer anonymous uploads. Accounts can access
their retained runs after signing out and back in.

Create an account from the repository root:

```bash
python scripts/run_web.py --create-user scientist
```

The terminal securely prompts twice for a password of at least 12 characters.
Passwords are stored using Werkzeug's password hashing, never plaintext. The
same command can reset an existing password after confirmation. There is no
public account registration or email password-reset service in this version.

To require login before opening the workspace:

```bash
python scripts/run_web.py --require-login
```

Or set `ADTL_WEB_AUTH_REQUIRED=true` for the server environment. Sign out uses a
CSRF-protected POST. All dataset, job, and download requests check ownership.
Login attempts are limited to five per 15 minutes per remote IP or username.
Additional edge rate limiting is needed for public hosting. If a proxy forwards
all traffic from one IP, configure a trusted proxy boundary before enabling
real-client-IP handling; the app deliberately does not trust arbitrary
`X-Forwarded-For` headers.

## Configuration and resource limits

Environment variables use the `ADTL_WEB_` prefix. Booleans and numbers use JSON
syntax (`true`, `false`, integer bytes). `TRUSTED_HOSTS` is a JSON list.

| Variable suffix | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `instance/web` | Private filesystem directory shared by server and worker |
| `AUTH_REQUIRED` | `false` | Require a login for dataset/analysis access |
| `PUBLIC_MODE` | `false` | Require configured secret/hosts; use secure cookies and HSTS |
| `SECRET_KEY` | Generated local file | Stable signing secret; explicitly required in public mode |
| `TRUSTED_HOSTS` | Localhost addresses | Accepted Host headers |
| `MAX_CONTENT_LENGTH` | `104857600` | 100 MiB for the whole upload request, including all CSVs |
| `MAX_IMPORT_BYTES` | `536870912` | HDF5 expanded fixed-size array estimate; not a hard memory limit |
| `MAX_DENSE_BYTES` | `134217728` | 128 MiB estimated selected dense matrix; not total process RAM |
| `JOB_TIMEOUT_SECONDS` | `300` | Wall-clock limit per import or analysis child process |
| `MAX_DATASETS` | `3` | Datasets per anonymous session/account |
| `MAX_TOTAL_DATASETS` | `30` | Datasets across this installation |
| `MAX_JOBS` | `20` | Retained jobs per dataset, including its import |
| `MAX_QUEUED_JOBS` | `30` | Global queued/running jobs |
| `RETENTION_HOURS` | `24` | Dataset and result lifetime from upload |

The worker runs one job at a time and terminates timed-out child processes.
It is suitable for a small single-host service, not a distributed queue. Limits
must be sized to the host and workload. CSV parsing, variable-length HDF5 strings,
full H5AD loading, scientific intermediates, and figure rendering can consume
more memory than the matrix estimate. Use operating-system/container memory and
disk limits for public hosting. No large-dataset throughput benchmark has been
performed; passing small test fixtures is not a capacity guarantee.

## Deployment path for adata-science-tools.com

Keep the local launcher as the easy desktop entry point. For a hosted service,
run the Flask factory under a production WSGI server, with the analysis worker
as a separate supervised process. Flask is the only direct web dependency here;
install the production server you choose in the deployment environment.

Example environment (replace paths; generate and protect a real random secret):

```bash
export ADTL_WEB_PUBLIC_MODE=true
export ADTL_WEB_DATA_ROOT=/srv/adata-science-tools/private-data
export ADTL_WEB_TRUSTED_HOSTS='["adata-science-tools.com","www.adata-science-tools.com"]'
export ADTL_WEB_AUTH_REQUIRED=true
# Set ADTL_WEB_SECRET_KEY through the host's protected secret configuration.
```

Set `AUTH_REQUIRED=false` only when intentionally providing anonymous analysis.
An optional-login public site still isolates each anonymous browser session.

From the parent directory of the checkout, an example Gunicorn command is:

```bash
gunicorn --bind 127.0.0.1:8000 --workers 2 'adata_science_tools.web:create_app()'
```

Run the worker from the repository root with the same environment:

```bash
python scripts/run_web.py --worker
```

Before enabling public traffic, configure DNS for your host, HTTPS termination,
forwarding of the intended Host header, upload/body timeouts, edge request-rate
limits, a filesystem quota, worker memory/CPU limits, process supervision, and a
published data-retention/privacy policy. Keep the storage directory outside the
web server's document root. Uploaded H5AD files are untrusted binary inputs;
run workers as an unprivileged account/container without host credentials and
keep scientific parsing dependencies patched. Complete deployment-specific
security and load testing before opening anonymous uploads.

These are deployment steps, not services provisioned by the repository. No DNS,
domain purchase, TLS certificate, hosting account, or public exposure is created
by the local launcher. See the official [Flask deployment guidance](https://flask.palletsprojects.com/en/stable/deploying/)
and [security guidance](https://flask.palletsprojects.com/en/stable/web-security/).

## Tests

With the scientific environment and optional web dependency installed:

```bash
MPLBACKEND=Agg python -m pytest tests/test_web_data.py tests/test_web_analysis.py tests/test_web_app.py tests/test_web_pipelines.py
```

Coverage includes CSV alignment and identifiers, H5AD validation, sparse/layer/raw
selection, scientific API parity, each exposed renderer, complete upload/run/
download flows, ownership checks, CSRF, login throttling, and retention cleanup.
The Flask-specific suite skips when the optional Flask dependency is absent.

Browser-state regression tests use Node.js's built-in test runner, with no npm
dependencies (Node.js is not needed to run the web app):

```bash
node --test tests/web_state.test.cjs
```

These tests control response order to check dataset switching and background
polling, block submissions during loading, and verify that pipeline settings
survive navigation back from the studio.

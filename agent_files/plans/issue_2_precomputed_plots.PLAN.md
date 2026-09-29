# Issue 2: Precomputed enrichment and coordinate plots

## Status and scope

Status: complete. Updated: 2026-09-29. Repository: `adata_science_tools`.
Issue: https://github.com/gitbenlewis/adata_science_tools/issues/2
The user approved items 2–5: implementation, tests, docs, three synthetic gallery
examples, and this plan. The user subsequently authorized commit and push.
No dependency installation or separate GitHub issue action is requested.

Provide `enrichment_dotplot()` and `coordinate_scatter()` with `(fig, ax, plotted)`
returns. Do not fit models, calculate enrichment/correlation, transform source
coordinates, mutate inputs, change existing APIs, or automatically select terms.
Use the existing order/palette/reference helpers and synthetic fixtures only.

## Findings and design decisions

Current local/remote main is `3f3449b`; the worktree started clean.
`corr_dotplot()` calls `_compute_corr_and_fit` before display switches.
`l2fc_dotplot_single()` clips significance to a tiny positive value and couples
color/area to its logarithm, so it does not meet the requested contract.
Baseline: 79 tests and 58 subtests passed for correlation, tabular, and column
renderers in the existing `not_base` Python environment with Agg.

1. Comparison plots use stable term IDs, ordered category offsets, constant
   area, comparison color/shape, strict significance < cutoff, and a zero line.
   Missing/nonfinite scores or significance omit the point without removing its
   slot. Duplicate term/comparison keys and significance outside [0, 1] fail.
2. Returned rows retain source order, index, and original columns/values. Audit
   columns describe coordinates/status/encodings; missing term/comparison records
   are reported in `plotted.attrs['missing_combinations']`, so absent combinations
   do not require fabricating source rows. Explicit orders must include observed
   categories; extra requested categories reserve empty slots.
3. Bubble plots require explicit numeric x/color/area columns (x may use score).
   Caller-supplied linear Normalize objects with fixed bounds support common panel scales.
   Marker sizes are areas in points squared, with linear interpolation over a
   documented range. Return display encodings, clipping flags, and scale metadata.
   Optional negative-log10 color display requires an explicit positive floor;
   preserve source zeros and label censored color values in the legend/colorbar.
4. Coordinate scatter uses only supplied finite numeric coordinates. Missing hue
   or nonfinite coordinates exclude a point with an audit reason. Empty inputs
   yield an empty axes/table; singleton/constant groups remain valid. User labels
   are used verbatim, without inferring explained variance.
5. Supplied axes retain ownership: no closing, global layout, or show call on
   caller-owned figures. Newly created figures follow show/save conventions.
   Use explicit limits, legend placement and font controls; wrap long term labels
   for display only. No unresolved scientific questions or new dependencies.

## Phases and proposed files

1. Complete: add coordinate renderer to `_plotting/_tabular_plots.py`, new
   `_plotting/_enrichment.py`, exports, `tests/test_coordinate_scatter.py`, and
   `tests/test_enrichment_dotplot.py`. Verify numerical artists, audit returns,
   missing slots, duplicates/labels, threshold boundary, empirical zeros, common
   normalization, and no calls to regression/correlation. Keep checks specific
   to reachable failures and avoid redundant helpers/tests.
2. Complete: add `docs/_enrichment.md`, update tabular docs and docs index, and
   register three cases in gallery generator/manifest/catalog and gallery tests.
   Generate three PNGs and inspect long labels and legends visually.
3. Complete: run focused existing/new suites and the full suite; check input and
   supplied-axis ownership, deterministic renders, docs links and exports, and
   review the final diff. Finish when all issue contracts have evidence.

## Risks and acceptance

Term alignment and missingness must not manufacture zero effects or classify
unknown significance as nonsignificant. Display floors must never overwrite
empirical values. Shared normalization must use identical bounds across panels;
independent defaults are explicitly labeled as per-call scales. Existing API
behavior must remain unchanged. Validation should target these actual risks.
Tests use Agg and the existing environment; no cross-version claim is implied.

## Implementation scratchpad

### 2026-09-29 — Approved start

Read issue #2, verified current main and the gaps above, and ran the baseline.
The first baseline command used a singular test filename and collected no tests;
the corrected command passed 79 tests and 58 subtests. Saved the approved design
and concrete omission/metadata policies before writing the renderers.

### 2026-09-29 — Core renderers verified

Added the renderers and 15 focused tests; all passed (5 subtests, 4.93 seconds).
Tests cover source values/order/index, exact artists, cutoff equality, empirical
zeros, nonfinite omissions, duplicate keys/labels, absent terms, shared scales,
axes ownership/saving, empty/singleton/constant coordinates, and visible layout.
The scatter tests patch regression/correlation calls to fail if reached.
Bubble comparisons, when supplied, use comparison-specific shapes so the numeric
color encoding does not hide comparison identity. Censoring is also labeled on
the colorbar when present. No fitting or scientific calculations were introduced.

### 2026-09-29 — Gallery and documentation

Added three deterministic synthetic gallery cases, their manifest entries, PNGs,
API documentation, and catalog links. Visually inspected all three PNGs: wrapped
term labels, the outside comparison legend, the bubble size legend and censored
color label, and coincident coordinate rows are readable. Gallery coverage is now
47 renderers and 68 cases. Normalization accepts linear Normalize with fixed
bounds; logarithmic color must use the explicit floor path, preventing LogNorm
from masking empirical zeros silently. Existing renderer implementations and
assets remain unchanged. Focused gallery/docs checks and final review are running.

### 2026-09-29 — Focused verification and review

Focused pytest for coordinate, enrichment, gallery, and documentation passed:
43 tests, 242 subtests, 16 warnings from existing gallery paths (42.17 seconds).
All three documented example blocks executed and both new signatures matched
source ASTs. Repeated gallery cases produced identical bytes. Manual nullable
numeric/string and categorical-term checks passed for both enrichment modes,
including empty tables; original values and dtypes were retained. Final review
used the requested Karpathy and bioinformatics engineering skills: no new
production helper layer, dependency, or unrelated refactor. Checks protect term
alignment, source columns, significance domains, and explicit display encodings.
The full regression suite is now running. `git diff --check` passes.

### 2026-09-29 — Completed

Full regression command:
`PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/adtl-issue2-mpl MPLBACKEND=Agg /Users/ben/miniconda3/envs/not_base/bin/python -m pytest -q`
passed 622 tests and 620 subtests in 83.54 seconds. The 178 warnings came from
existing test paths; neither new renderer produced warnings. This includes the
existing correlation, tabular, column, export, gallery, and documentation suites.
The final diff has no whitespace errors and changes no tracked gallery PNGs;
only the three new PNGs were added. No new dependencies or existing public
signature/default changes were introduced. All approved phases are complete.

Verification is limited to the installed not_base environment and Agg backend;
a dependency-version matrix was not run. The complete historical gallery was
not regenerated because its renderers/assets were unchanged; the new cases and
focused existing cases were rendered deterministically and inspected as noted
above. No unresolved implementation questions remain. Changes are local and
uncommitted; commit, push, and GitHub issue closure were not performed.

### 2026-09-29 — Publication authorized

The user requested commit and push after completion. Verified only the approved
issue #2 files are changed, with no subsequent code changes since the full suite
passed. Publication targets the existing `main` branch at `origin`. Check the
working tree, staged index, and outgoing history against the 100 MiB blob limit,
then commit and push normally. Remote SHA verification follows the push.

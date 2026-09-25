# Issue 1: Horizontal datapoint plots

## Status and objective

Status: implementation and simplicity review complete; ready for authorized publication. Updated: 2026-09-25.
Repository: `adata_science_tools`; no related repositories.
Issue: https://github.com/gitbenlewis/adata_science_tools/issues/1

Provide native horizontal rendering in `datapoints()` and `paired_datapoints()`,
median-only ticks for grouped observations, and synthetic documentation/gallery
examples. Preserve default vertical rendering and `(fig, axes, plot_df)` returns.

## Authorization and scope

The user approved implementation, tests, documentation, and gallery changes with
`approve_2_through_4` in this session, then requested this persistent plan.
The user subsequently requested a simplicity review followed by a push;
review cleanup, committing, and pushing to the existing upstream are authorized.
PR creation and issue closure are not requested.
Use only synthetic data and generic labels in all artifacts. No new dependencies.
Preserve scientific calculations, filtering, pairing, slope colors, seeded jitter,
and returned long-form data. Keep y limits, y scale, axis labels, and sharing tied
to their physical axes. Horizontal paired-difference axes are out of scope and
must raise a clear ValueError.

## Open questions and current best guesses

1. Resolved: the user approved adding styled `x_reference_lines` to
   `paired_datapoints()`. The issue asks for this only in `datapoints()`; the user
   asked whether paired plots need the same update. Recommendation: add the
   keyword using the existing reference-line helper. This changes the public
   API beyond the initial scope; the user explicitly approved implementation,
   tests, and documentation for this extension.
2. Median ticks reuse `boxplot_width` for their categorical span; include only
   finite values among `summary_included` rows and omit groups with fewer than
   two qualifying values. Existing box/violin summary behavior stays unchanged.
3. Group annotations rotate with the distribution: metric annotations use the
   numeric x value in horizontal mode; `axes_top`/`axes_bottom` map to the right/
   left ends of that axis. Document and test this behavior.
4. Explicit physical y limits take precedence over automatic top-first category
   ordering; callers can adjust returned axes directly for numeric x limits,
   scaling, typography, or other layout needs.

## Read-only findings

Local HEAD and remote main both equal `3b205081e9c7d81bafa3713c84410181e3a42ac3`.
The worktree was clean before implementation. Both functions render in
`_plotting/_datapoints.py`. `_plotting/_utils.py` already supports styled x/y
reference lines. Category jitter and pairing are computed before rendering,
so artist coordinates can be swapped without modifying the returned data.
Baseline: 119 tests and 67 subtests passed for the two datapoint test modules.

## Ordered phases and verification

1. Complete: implement orientation validation, coordinate selection, horizontal
   distributions, category ticks, labels, annotations, zero/reference lines, and
   median ticks. Complete when focused tests prove the requested geometry and
   unchanged default/data behavior.
2. Complete: add synthetic horizontal examples to both function docs, the gallery
   generator, manifest, index, and two PNG assets. Complete when each renders
   legibly and repeated generation produces identical files.
3. Complete: run the focused datapoint/gallery tests and full suite, inspect the
   diff, and check existing vertical gallery assets for unchanged rendering.
   Complete when failures are resolved and all success criteria have evidence.

## Proposed diff by file

1. `_plotting/_datapoints.py`: keyword-only orientation in both functions;
   rendering-only coordinate changes; grouped median ticks; x reference lines
   in both renderers;
   reject horizontal paired-difference plots.
2. `tests/test_datapoints.py`, `tests/test_paired_datapoints.py`: regression tests
   for ordering, coordinates, summaries, physical axes, colors, distributions,
   invalid options, paired endpoints, determinism, and unchanged plot data.
3. `docs/_datapoints.md`, `docs/_paired_datapoints.md`: signatures, behavior, and
   runnable synthetic horizontal examples.
4. `example_plotting_gallery/generate_gallery.py`,
   `example_plotting_gallery/manifest.py`, `docs/plotting_gallery.md`,
   `tests/test_plotting_gallery.py`: register and verify the new gallery cases.
5. `docs/assets/plotting_gallery/datapoints__horizontal_medians.png` and
   `docs/assets/plotting_gallery/paired_datapoints__horizontal_pairs.png`: new images.
6. This plan: progress, decisions, and verification evidence.

## Success criteria and risk controls

1. The issue's grouped target call renders numeric x values and ordered y groups,
   with first category at top, categorical jitter, median ticks, and a zero line.
2. Paired plots connect the same complete pairs with unchanged styling and data;
   incomplete pairs retain existing handling.
3. Default vertical outputs remain unchanged; explicit vertical matches default;
   repeated seeded rendering is deterministic in either orientation.
4. Median ticks use only eligible finite data and omit singletons/empty groups.
5. Styled reference lines and physical labels/limits/scales are tested, including
   deduplication with zero lines and legend handling.
6. New examples use synthetic data, render legibly, and regenerate identically.
7. Focused and full tests pass, `git diff --check` is clean, and the diff stays in
   scope. Shared category axes must not invert twice across panels.

## Final verification and remaining boundaries

All seven success criteria are met. Geometry, finite medians, singleton/empty
omission, colors, physical-axis settings, reference legends, invalid combinations,
and data preservation have focused regression coverage. Both synthetic examples
execute; their documented signatures match the implementation. Visual inspection
confirmed readable images, and all 11 datapoint gallery assets regenerated
byte-for-byte, including nine unchanged vertical examples.

Pre-review full-suite command in the existing `not_base` environment:
`PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/adtl-issue1-mpl MPLBACKEND=Agg python -m pytest -q -p no:cacheprovider`.
Result: **606 tests and 612 subtests passed in 74.49 seconds**.
`git diff --check` passed. Only the listed source, tests, documentation, gallery,
images, and this plan changed.

The suite reports 178 warnings, including eight pending-deprecation warnings
from the older Matplotlib `vert=False` API used for horizontal boxes/violins.
That keyword preserves support for older Matplotlib; default vertical calls do
not receive it. Cross-version environments were not run, and no dependencies
were installed. Horizontal paired-difference axes remain explicitly unsupported
as requested. There are no unresolved questions or required implementation
steps. At the initial implementation completion, changes were local. The review
and publication follow-up is recorded below.

## Implementation scratchpad

### 2026-09-25 — Inspection and baseline

Read the issue, current source, docs, gallery structure, and applicable guidance.
Verified current main and clean worktree. Ran the existing datapoint suites with
the `not_base` Python environment, Agg backend, and a temporary MPLCONFIGDIR:
119 passed, 67 subtests passed in 31.24 seconds.

### 2026-09-25 — Rendering implementation started

After approval, added orientation keywords/validation, grouped median ticks,
grouped x-reference lines, physical labels, rotated annotations, and coordinate
selection in `_plotting/_datapoints.py`. The paired renderer still needs coordinate
and distribution changes. No post-edit tests have run yet. Created this plan at
the user's request; paired x-reference lines are pending their scope choice.

### 2026-09-25 — Scope extension approved

The user approved `x_reference_lines` for `paired_datapoints()`. Added the keyword
and reused the existing line validation/drawing helpers, including legend handles.
Completed paired coordinate/distribution rotation and physical labels. Next: add
focused geometry and compatibility tests, then run phase 1 verification.

### 2026-09-25 — Phase 1 verified

Added focused tests for coordinates, categorical jitter, shared-axis inversion,
subset/marker/side colors, finite summary medians and singleton omission,
physical labels/scales/limits, box/violin geometry, paired endpoints, invalid
options, deterministic rendering, and unchanged plot data/default pixels.
The first run exposed a missing figure-level reference legend and a test-only
NumPy masked-array/pandas subtraction mismatch. Fixed both. The two test modules
now pass: 133 tests, 89 subtests in 15.39 seconds. Eight pending-deprecation
warnings come from horizontal box/violin `vert=False`, retained for older
Matplotlib compatibility; default vertical calls do not receive the keyword.
Phase 1 is complete. Next: synthetic documentation and gallery examples.

### 2026-09-25 — Phase 2 verified; integration checks started

Added both horizontal gallery cases, synchronized signatures and physical-axis
semantics in the docs, and updated manifest/index/gallery tests to 65 cases.
Generated and visually inspected the two new PNGs: labels, medians, references,
and pair connectors are legible. Regenerated all 11 datapoint gallery images in
a temporary directory: every file matches the repository asset byte-for-byte,
including all nine existing vertical examples and both new horizontal examples.
Focused datapoint, update, paired, and gallery suites passed: 177 tests and
201 subtests in 47.81 seconds. Added two further edge-case regressions for
unobserved categories/rotated annotations and reference legends with vertical
secondary axes; these will be included in the full run now in progress.
`git diff --check` passes. Next: verify documented signatures/examples, full
suite, final diff review, and close out this plan.

### 2026-09-25 — Full-suite documentation correction

Both documented signatures exactly match the implementation; both new examples
execute and render successfully. The full run passed 606 tests and 610 subtests
but failed two documentation subtests: the catalog expects a 520-pixel display
width for images with these aspect ratios, whereas the new entries used 700.
Corrected only those two thumbnail widths. No rendering/scientific failures were
reported. Rerunning the complete suite after the documentation correction.

### 2026-09-25 — Complete

The final full-suite run passed: 606 tests, 612 subtests, 178 warnings in 74.49
seconds. The two catalog-width failures are resolved. Confirmed the scoped diff
and recorded all success-criterion evidence and remaining compatibility limits
above. No required implementation work remains.

### 2026-09-25 — Simplicity review and publication requested

Reviewed the source, tests, synthetic examples, and docs using the explicitly
requested Karpathy and bioinformatics engineering skills. No unnecessary
production helper or guard was found: orientation validation, rejection of
horizontal paired-difference axes, and omission of medians for fewer than two
finite values implement the issue requirements. Existing reference validation
is reused rather than adding a new schema or validator.

Trimmed repeated invalid-type variants and duplicate shared reference-schema
checks from the new tests. Kept the issue-required invalid-orientation and
unsupported-combination checks. Changed the new artist/gallery comparisons to
`zip(..., strict=True)` so missing/extra artists or images cannot silently escape
the coordinate/determinism assertions. No scientific or rendering code changed.
Next: run affected tests, verify Git blob sizes and remote state, commit the
scoped files, push, and confirm remote/local SHA agreement.

### 2026-09-25 — Review verified; ready to publish

The affected suites passed after cleanup: 157 tests and 178 subtests in 46.93
seconds, with 24 warnings. Command: `python -m pytest -q -p no:cacheprovider
tests/test_datapoints.py tests/test_paired_datapoints.py tests/test_plotting_gallery.py`
using the same `not_base` environment and Agg configuration. The full suite was
not repeated because the follow-up changed tests and this plan only; its earlier
606-test pass remains the production-code verification. No unresolved review
findings remain.

Fetched origin and verified main has no incoming or outgoing commits before
publication. Working-tree and existing-index size scans found no blobs above
100 MiB. Both new horizontal PNGs are included in the scoped commit. Next Git
steps are staged/outgoing-blob checks, commit, push to origin/main, and remote
SHA/clean-worktree verification; the publication result is reported in the task.

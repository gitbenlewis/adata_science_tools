# Plan: Multi-Operation `ref_vs_target_adata` Layers

1. GUIDELINES: Apply the reviewed rules from `codex_session_start_SKILL.md` and `andrej-karpathy-skills`: no edits before approval, minimal diff, preserve numerical formulas, no new dependencies, no unrelated refactors.

2. SUMMARY: Extend `ref_vs_target_adata()` so `opperation_flavor` and corrected alias `operation_flavor` accept either one operation string or a sequence of operation strings.

3. API: Keep the parameter list unchanged; only update the type hint/docstring to document `str | Sequence[str]` behavior.

4. BEHAVIOR: Preserve current single-string behavior exactly, including existing `layers_to_compute` layer names and `.X` selection.

5. BEHAVIOR: In operation-list mode, normalize aliases to canonical operation names, compute every requested operation, store each computed matrix in `result.layers`, and set `result.X` to the first requested operation for the selected base source.

6. LAYER NAMES: If one source is computed, use operation names directly, e.g. `subtraction`, `relative_change_pct`, `relative_change_l2fc`; if multiple sources are computed, use `source__operation`, with `.X` labeled as `X`, e.g. `X__subtraction`, `alt__relative_change_l2fc`.

7. BASE SELECTION: Keep existing `base_layer` semantics for choosing the source used by `.X`; use the first canonical operation as the base operation, so the example list sets `.X` to `subtraction`.

8. IMPLEMENTATION: Refactor the existing operation loop locally so paired target/ref source matrices and bounds are computed once per source, then reused for each operation; keep formulas unchanged for subtraction, percent change, fold change, and log2 fold change.

9. VALIDATION: Raise `ValueError` for an empty operation list, unsupported operation entries, duplicate canonical operations after alias normalization, or generated layer-key collisions.

10. METADATA: Keep existing scalar metadata fields compatible by setting `operation_flavor`, `ref_vs_target_operation`, and `var["ref_vs_target_operation"]` to the base operation; add list/mapping metadata in `.uns["ref_vs_target_adata"]` for `operation_flavors`, generated operation layer keys, and the selected base operation layer.

11. SOURCE VALUES: Keep `save_source_values_obsm=True` tied to the base source values before operation computation; no per-operation `.obsm` outputs.

12. DIFF: Modify `_preprocessing/_adata_row_operations.py` for normalization, cross-product computation, result layer assignment, and metadata.

13. DIFF: Update `tests/test_adata_row_operations.py` with focused tests for operation-list layers, `.X` base operation selection, cross-product layer naming with `layers_to_compute`, alias-list support, and duplicate alias rejection.

14. DIFF: Update `docs/_preprocessing.md` with the new operation-list call pattern, layer naming rules, `.X` selection rule, and metadata note.

15. TEST PLAN: Run `conda run -n not_base python -m pytest tests/test_adata_row_operations.py -q`; baseline currently passes there with `18 passed, 3 subtests passed`.

16. ASSUMPTION: Do not change statistical formulas, epsilon defaults, bounds behavior, pair matching, obs/var merge behavior, or existing single-operation outputs.

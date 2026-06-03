# Plan: Add `ref_vs_target_adata` for paired Pre/Post AnnData transforms

1. **Title**: Add `ref_vs_target_adata` for paired Pre/Post AnnData transforms.

2. **Full Function Signature**:

```python
def ref_vs_target_adata(
    adata: ad.AnnData,
    groupby_key: str = "Pre_or_Post_obs_col",
    groupby_key_target_value: str = "Post",
    groupby_key_ref_value: str = "Pre",
    opperation_flavor: str = "subtraction",
    obs_dfs: str = "merge",
    ref_obs_suffix: str = ".src_pre",
    target_obs_suffix: str = ".src_post",
    keep_var_df: bool = True,
    **params,
) -> ad.AnnData | tuple[ad.AnnData, pd.DataFrame]:
```

3. **Diff Summary**: Update `_preprocessing/_adata_row_operations.py` with one new public function. Add focused tests in `tests/test_adata_row_operations.py`. Save this plan as `plans/ref_vs_target_adata_PLAN.md`.

4. **Config Params**: Support `pair_by_key`, `layer`, `layers_to_compute`, `base_layer`, `epsilon=1e-9`, `target_min_value`, `target_max_value`, `ref_min_value`, `ref_max_value`, `merge_shared_obs_cols=False`, `return_df=False`, `allow_unused_params=False`, `logger=None`, and `log_level="INFO"` through `**params`.

5. **Pairing Behavior**: Require `pair_by_key`. Raise on missing pair IDs in selected Pre/Post observations. Raise on duplicate pair IDs within either selected group. Use only overlapping pair IDs; log and store dropped target-only/ref-only IDs in `.uns`. Raise if no overlapping pairs exist.

6. **Operation Behavior**: Compute target minus ref, so defaults mean `Post - Pre`. Implement `subtraction`, `relative_change_pct`, `relative_change_fc`, and `relative_change_l2fc`. Relative operations use configurable `epsilon`.

7. **LOD/Bounds Behavior**: Optional `target_min_value`, `target_max_value`, `ref_min_value`, and `ref_max_value` clamp valid paired values before computing all operations. Values below min are replaced by min; values above max are replaced by max. This supports LOD flooring, e.g. `target_min_value=0.5` treats any valid Post value below `0.5` as `0.5`.

8. **Data Sources**: Support `.X` plus selected layers. If `layers_to_compute` is omitted, compute the requested `layer` or `.X`. If multiple layers are requested, compute each layer and choose `.X` from `base_layer`, defaulting to the first requested source. Do not add raw support in v1.

9. **Returned AnnData**: Return one observation per matched pair, indexed by stringified `pair_by_key`. Store `ref_obs_name`, `target_obs_name`, pair/order metadata, operation metadata, and source group labels in `.obs`/`.uns`. When `return_df=True`, return `(result_adata, result_df)`.

10. **Obs Metadata**: For `obs_dfs="merge"`, include pair key plus both ref and target `.obs` columns with `ref_obs_suffix` and `target_obs_suffix`. If `merge_shared_obs_cols=True`, collapse columns whose ref/target values are identical for every retained pair. Keep `obs_dfs="keep_ref"` and `"keep_target"` as minimal provenance-preserving alternatives.

11. **Var Metadata**: If `keep_var_df=True`, copy `adata.var` and add operation metadata columns. If `keep_var_df=False`, create a same-index `.var` with only generated operation metadata. Do not alter var names or feature ordering.

12. **Tests**: Add deterministic `unittest` coverage for shuffled Pre/Post pairing, target-minus-ref math, `.X` plus layer computation, obs merge/keep modes, var metadata, relative operations with epsilon and LOD clamping, configurable return, typo/corrected operation aliases, dropped unmatched pairs, missing pair IDs, duplicate pair IDs, and no-overlap errors.

13. **Verification**: Run `python -m unittest tests.test_adata_row_operations -v`, then `python -m unittest discover -s tests -v`. Accept only if the new targeted tests and existing suite pass without changing statistical models, thresholds, public signatures outside the new function, or adding dependencies.

14. **Assumptions**: The implementation will be a standalone function only. Unmatched pairs are dropped with provenance, while missing pair IDs and duplicate pair IDs are errors. Bounds are clamping controls, not filtering or missingness controls.

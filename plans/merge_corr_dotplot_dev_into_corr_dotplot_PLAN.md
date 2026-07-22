# Plan: Promote `corr_dotplot_dev()` Functionality into `corr_dotplot()`

## Objective and constraints

1. Make `corr_dotplot()` the single implementation and supported entry point for both the existing correlation scatter plot and the optional marginal-histogram layouts, so repository callers no longer need `corr_dotplot_dev()`.

2. Start the implementation by copying the current `corr_dotplot_dev()` signature and body onto `corr_dotplot()`. Treat that copied implementation as the initial source of truth before selectively restoring only important `corr_dotplot()`-specific compatibility behavior.

3. Preserve the existing data assembly, filtering order, correlation methods, regression calculations, subgroup ordering, palettes, fit/stat formatting, and returned overall `fit`, `corr_value`, and `corr_pvalue`. Do not change statistical or biological behavior.

4. Add no external dependencies and avoid unrelated refactoring. The final module should contain one full implementation, not two large near-identical functions or a new abstraction that merely relocates the duplication.

## Verified current state

5. Both functions live in `_plotting/_corr_dotplots.py`. Their input assembly, filtering, overall statistics, scatter layer, subgroup fits, palette behavior, and most legend/footer code are currently duplicated.

6. `corr_dotplot_dev()` adds these public controls that must move to `corr_dotplot()` with their current names and defaults: `show_x_marginal_hist`, `show_y_marginal_hist`, `x_marginal_hist_bins`, `y_marginal_hist_bins`, `x_marginal_hist_fill`, `x_marginal_hist_KDE`, `y_marginal_hist_fill`, `y_marginal_hist_KDE`, `show_all_obs_x_hist`, `show_all_obs_y_hist`, `x_marginal_hist_height_ratio`, and `y_marginal_hist_width_ratio`.

7. The dev-only implementation also owns the four composite layouts, shared axes, subgroup and all-observation histograms, common per-axis bin ranges, title movement to the x marginal, marginal-aware legend placement, marginal-aware footer spacing, main-axis limit restoration, axes-dictionary return, and fallback to all-data overlays when no valid subset values remain.

8. The important existing `corr_dotplot()` compatibility behavior is its return contract: without marginal plots it returns a single `matplotlib.axes.Axes`. `spearman_cor_dotplot()` and existing callers inherit that contract.

9. The focused baseline is clean: `python tests/test_corr_dotplots.py` ran 22 tests successfully on 2026-07-22. The only output was existing seaborn warnings about palettes containing more colors than required.

## Implementation sequence

10. In `_plotting/_corr_dotplots.py`, first replace the current `corr_dotplot()` signature and body with a direct copy of the current `corr_dotplot_dev()` signature and body, changing only the function name. This deliberately makes promotion happen before compatibility edits, which keeps the migration mechanically reviewable.

11. Restore the complete stable `corr_dotplot()` parameter documentation, then extend it with the marginal parameters, layout behavior, conditional return contract, and empty-subset fallback. Keep the existing notes about observation/feature name collisions and overall returned statistics.

12. Keep the copied dev layout implementation intact: use one main axes with no marginals, a 2x1 grid for x-only, a 1x2 grid for y-only, and a 2x2 grid with the top-right slot unused when both marginals are enabled.

13. Keep every copied marginal behavior intact: draw from the already-filtered `working_df`; use `subset_key` rather than `hue` for grouped marginals; use `subset_palette` for subgroup marginal colors; preserve current fill, KDE, alpha, bin, count-axis, shared-axis, title, legend, and footer behavior; and restore the main scatter limits after shared marginal plotting.

14. Add back the important stable return behavior after the copy. When both `show_x_marginal_hist` and `show_y_marginal_hist` are `False`, return the existing single main `Axes`. When either marginal is enabled, return `{"main": axes, "x_marginal": axes_x_marginal, "y_marginal": axes_y_marginal}`. The other four tuple elements remain unchanged in both modes.

15. Do not restore the old stable empty-subset display behavior. Promote the dev fallback into `corr_dotplot()` so an all-null or otherwise empty `subset_key` still shows the all-data fit and statistics, and enabled marginals show the all-data distributions. This is an intentional visible edge-case change, not a change to the returned overall statistics.

16. Do not restore `plt.subplots()` solely for the no-marginal path. The copied `plt.figure()` plus `add_subplot()` path produces the required single axes and avoids maintaining two figure-construction flows; existing no-marginal regression tests will verify that this is behaviorally acceptable.

17. Keep `spearman_cor_dotplot()` as the existing thin wrapper around `corr_dotplot()`. With no marginal flags it must continue returning a single `Axes`; when marginal flags are forwarded through `**kwargs`, it should naturally return the axes dictionary while still forcing `method="spearman"`.

18. After `corr_dotplot()` is verified, replace the duplicated `corr_dotplot_dev()` body with a small compatibility wrapper around `corr_dotplot()`. The wrapper should retain the old dev contract by normalizing a no-marginal single `Axes` result into an axes dictionary; marginal calls will already receive the dictionary from `corr_dotplot()`.

19. Mark `corr_dotplot_dev()` as deprecated in its docstring and emit a standard-library `DeprecationWarning` with `stacklevel=2`. Do not remove it in this change, because preserving the existing public name gives downstream users a migration window while this repository stops calling it.

20. Keep helper functions such as `_compute_corr_and_fit()`, `_plot_fit_line()`, and the legend/stat formatting helpers unchanged unless the promotion itself makes a narrowly scoped adjustment necessary.

## Tests and verification

21. In `tests/test_corr_dotplots.py`, retain all existing stable `corr_dotplot()` regression tests unchanged where possible. They must continue covering collision renaming, styling, hidden stats, subgroup fits, unavailable fits, independent legends, method-specific titles, palette separation, numeric subgroup order, and the single-`Axes` no-marginal return.

22. Retarget the dev marginal tests to `corr_dotplot()` and rename them accordingly. For the layout matrix, assert a single `Axes` for the no-marginal case and the three-key axes dictionary for x-only, y-only, and both-marginal cases.

23. Preserve focused coverage for grouped marginals, all-observation overlays, filtered-data parity, integer and explicit-edge bins, independent fill/KDE toggles, title ownership and `axes_title_y`, legend coexistence and custom anchors, separate hue/subset palettes, numeric subgroup ordering, footer non-overlap, `show=False`, and main-axis limits.

24. Add explicit assertions that custom marginal height and width ratios change the relative panel sizes, so all promoted public marginal parameters have regression coverage.

25. Retarget the empty-subset fallback test to `corr_dotplot()` and cover both a marginal-enabled call and a no-marginal call. Assert the all-data fit line, legend title, footer text, returned overall correlation, and all-data marginals where enabled.

26. Add a small compatibility test for `corr_dotplot_dev()` that checks the deprecation warning, verifies its no-marginal axes-dictionary normalization, and confirms its numerical outputs match `corr_dotplot()` for the same data. Avoid keeping a second copy of the full marginal test matrix for the wrapper.

27. Extend the existing `spearman_cor_dotplot()` forwarding test with one marginal-enabled call, confirming the axes dictionary and Spearman statistics without duplicating layout tests.

28. Run `python tests/test_corr_dotplots.py` after the implementation, then run the repository-wide test command supported by the environment. If no project-level runner is configured, run `python -m unittest discover -s tests` and report any unrelated failures separately.

## Documentation and repository migration

29. In `docs/_corr_dotplots.md`, move the marginal signature, example, return shape, and behavioral notes into the primary `corr_dotplot()` section. Explain the conditional axes return clearly and reduce the `corr_dotplot_dev()` section to a deprecation/migration note.

30. In `example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py`, replace `adtl.corr_dotplot_dev(...)` with `adtl.corr_dotplot(...)`. Its configured defaults enable both marginals, so its existing `axes.keys()` logging remains valid.

31. Leave `example_simulated_data/config/config.yaml` unchanged because its marginal option names and values remain valid after promotion.

32. Leave historical files under `plans/` unchanged. They document how the current behavior arose and are not runtime call sites.

## Proposed diff by file

33. `_plotting/_corr_dotplots.py`: promote the dev signature/body into `corr_dotplot()`, restore the important conditional return compatibility, retain the dev empty-subset fallback, update the canonical docstring, and reduce `corr_dotplot_dev()` to a deprecated compatibility wrapper.

34. `tests/test_corr_dotplots.py`: move marginal behavior coverage to `corr_dotplot()`, adjust return-shape expectations by layout, add ratio/fallback/Spearman coverage, and keep one focused dev-wrapper compatibility test.

35. `docs/_corr_dotplots.md`: document one canonical API and the compatibility wrapper.

36. `example_simulated_data/scripts/plot_dotplot_simulate_1_var_covar_age.py`: migrate the sole non-test, non-documentation dev call site to `corr_dotplot()`.

## Success criteria

37. No repository runtime or example code calls `corr_dotplot_dev()`; only the compatibility definition, its focused test, documentation migration note, and historical plans may still mention it.

38. Every plotting option currently accepted by `corr_dotplot_dev()` is accepted by `corr_dotplot()` with the same default and produces the same marginal plots, titles, legends, colors, filtered observations, and overall statistics.

39. Existing `corr_dotplot()` and `spearman_cor_dotplot()` calls that do not enable marginals continue to receive a single `Axes` and pass their current regression tests.

40. Marginal-enabled `corr_dotplot()` calls receive the three-key axes dictionary, allowing the current example to switch function names without any other code change.

41. The implementation contains only one full correlation-dotplot code path, all focused and repository-wide tests pass, and any commands not run or remaining risks are reported.

## Implementation scratchpad

42. Status: implementation completed on 2026-07-22.

43. Baseline command: `python tests/test_corr_dotplots.py`.

44. Baseline result: 22 tests passed in 4.101 seconds on 2026-07-22; existing seaborn oversized-palette warnings were observed.

45. Locked starting approach: copy `corr_dotplot_dev()` onto `corr_dotplot()` first, then restore only the important stable compatibility behavior described above.

46. Implementation result: promoted the dev layout, marginal rendering, empty-subset fallback, and marginal-aware title/legend/footer behavior into `corr_dotplot()` and removed the duplicated implementation.

47. Compatibility result: `corr_dotplot()` returns one `Axes` with no marginals and the three-key axes dictionary with either marginal enabled. `corr_dotplot_dev()` retains its exact signature, emits `DeprecationWarning`, forwards to `corr_dotplot()`, and normalizes its return to the historical axes dictionary.

48. Migration result: all marginal tests now exercise `corr_dotplot()`; the simulated-data example calls `corr_dotplot()`; the docs present one canonical API and a short dev-wrapper migration note.

49. Focused verification: `python tests/test_corr_dotplots.py` passed 25 tests in 4.782 seconds after adding coverage for conditional axes returns, no-marginal empty-subset fallback, explicit bins, subplot ratios, main-limit preservation, marginal Spearman forwarding, and wrapper compatibility.

50. Repository verification: `python -m unittest discover -s tests` passed 221 tests in 12.991 seconds.

51. Additional verification: Python compilation passed for the implementation, focused test module, and migrated example script; `corr_dotplot()` and `corr_dotplot_dev()` signatures compare equal via `inspect.signature`; `git diff --check` passed.

52. Deviations: none from the approved implementation sequence or behavior decisions.

53. Remaining observations: test output still contains the pre-existing seaborn oversized-palette warnings and small-sample scipy/statsmodels warnings; no new failures or unresolved verification risks remain.

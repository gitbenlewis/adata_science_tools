# Plan: Fix Numeric `subset_key` Ordering In Correlation Dotplots

1. Summary: Change both `corr_dotplot()` and `corr_dotplot_dev()` so non-categorical numeric `subset_key` values are processed in sorted numeric order, while categorical values keep category order and non-numeric values keep first-seen order.
2. Context reviewed: Prior subset-fit planning chose first-appearance order for non-categorical subsets. That behavior can diverge from seaborn's numeric hue legend ordering when the same numeric column is used for both `hue` and `subset_key`.
3. Implementation: In `_plotting/_corr_dotplots.py`, update only the two subset-value selection branches. Use sorted order only when `pd.api.types.is_numeric_dtype(non_null_subset)` is true. Do not change statistical calculations, filtering, palettes, signatures, or return values.
4. Tests: Add focused synthetic-data tests in `tests/test_corr_dotplots.py` for `corr_dotplot()` and `corr_dotplot_dev()` where numeric subgroup values appear first as `30, 10, 20`. Assert fit legend order and colors align with the sorted numeric hue legend order `10, 20, 30`.
5. Data privacy: Use only synthetic column names and synthetic numeric values in tests and plan text. Do not reference private project files, notebooks, patient data, or study-specific dataset details in the public repo change.
6. Verification: Run `python -m unittest /home/benjaminl/projects/gitbenlewis/adata_science_tools/tests/test_corr_dotplots.py` after edits.
7. Assumptions: Numeric categorical columns should still respect explicit categorical order, because that is caller-defined display intent. Numeric object strings should remain first-seen order, because seaborn treats them as categorical labels rather than numeric hue values.

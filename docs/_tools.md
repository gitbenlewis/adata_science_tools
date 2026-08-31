# `_tools`

This page covers the general utility implemented in
[`_tools/_tools.py`](../_tools/_tools.py) and re-exported as
`adata_science_tools.average_feature_expression`. Other public `_tools`
submodules have dedicated pages for [differential testing](_diff_test.md),
[expectation-based correction](_expectation_based_covar_correction.md), and
[model fitting](_model_fit.md).

## `average_feature_expression`

`average_feature_expression(...)` returns one mean-expression row per category
in an observation column.

### Full signature

```python
def average_feature_expression(
    adata,
    groupby_key,
    layer=None,
    use_raw=False,
    log1p=False,
    zscore=False,
    subtract_mean=True,
):
```

```python
import adata_science_tools as adtl

average_expression = adtl.average_feature_expression(
    adata,
    groupby_key="cell_type",
    layer="normalized",
)
```

### Data selection and transforms

- By default, values and feature names come from `adata.X` and
  `adata.var_names`.
- `layer="name"` selects `adata.layers["name"]`.
- `use_raw=True` selects `adata.raw.X` and `adata.raw.var_names`; it cannot be
  combined with `layer`.
- `log1p=True` transforms values before grouping.
- `zscore=True` scales features before grouping. `subtract_mean=True` is the
  runtime default and controls the explicit centering step used by this path;
  it has no effect when `zscore=False`.

### Group ordering and return value

`adata.obs[groupby_key]` must be categorical because the function reads its
declared category order. The returned `pandas.DataFrame` uses those categories
as its row index and the selected feature names as its columns. Declared
categories with no matching observations are retained and receive the result of
the mean over an empty group.

## Coverage note

This page documents current source behavior. There is no focused regression
test for `average_feature_expression(...)` in `tests/`.

# `palettes`

Named color palette constants from `_plotting/palettes.py`.

The plotting package exposes this module as `adata_science_tools.palettes`.

```python
import adata_science_tools as adtl

adtl.palettes.tol_colors
adtl.palettes.godsnot_102
```

## Main palette constants

- `tol_colors`: Paul Tol's 10-color palette
- `Iridescent`: extended sequential palette
- `vega_10`
- `vega_10_scanpy`
- `vega_20`
- `vega_20_scanpy`
- `default_20`
- `zeileis_28`
- `default_28`
- `godsnot_102`
- `default_102`

Several of these are copied or adapted from Scanpy and Matplotlib palette definitions.

## Typical usage

```python
import seaborn as sns
import adata_science_tools as adtl

sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="group",
    palette=adtl.palettes.godsnot_102,
)
```

## Internal helper

`_plot_color_cycle(...)` is an internal preview helper that renders all palette lists. It is mainly used in the module's `__main__` block.

## Coverage note

This module is a constants module; it does not appear to have direct regression tests.

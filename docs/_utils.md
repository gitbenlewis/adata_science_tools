# `_utils`

Small plotting utility helpers from `_plotting/_utils.py`.

This module contains convenience functions for previewing color palettes.

## Public entry points

- `show_tol_colors`
- `show_colors`

## `show_tol_colors`

Use `show_tol_colors(...)` to preview Paul Tol's 10-color palette, or a supplied list of colors, as a labeled bar strip.

### Full signature

```python
def show_tol_colors(colors=None):
```

```python
import adata_science_tools as adtl

adtl.show_tol_colors()
```

Important behavior:

- If `colors` is omitted, the function uses the built-in Tol palette.
- Each bar is labeled with its hex code.
- The function calls `plt.show()`.
- The current implementation does not return the figure.

## `show_colors`

Use `show_colors(...)` to preview an arbitrary color list and optionally save the figure.

### Full signature

```python
def show_colors(colors=None,
                title_text=' Color Palette',
                save_plot=False,
                save_file_dir=None,
                save_file_name='color_palette.png'):
```

```python
adtl.show_colors(
    colors=adtl.palettes.vega_10_scanpy,
    title_text="Scanpy Vega 10",
    save_plot=True,
    save_file_name="vega_10_scanpy.png",
)
```

Important behavior:

- `colors` should be a list-like set of color strings.
- The plot is shown with `plt.show()` before any optional save.
- When `save_plot=True`, the function writes to `save_file_name` or `save_file_dir / save_file_name`.
- `save_file_dir` is used with the `/` operator, so passing a `pathlib.Path` is the safest option.
- The function does not explicitly return the figure.

## Coverage note

This module is documented from current code. There do not appear to be dedicated regression tests for it.

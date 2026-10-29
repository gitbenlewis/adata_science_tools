"""Display-only extensions for the public volcano API."""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle


def prepare_rendering(table, effect_col, probability_col, labels):
    """Keep one positional metadata row per input, including invalid observations."""
    options = {} if labels is None else dict(labels)
    unknown = set(options) - {"ids", "column", "id_col", "display_col"}
    if unknown:
        raise ValueError(f"Unknown labels options: {sorted(unknown)}")
    if "ids" in options and "column" in options:
        raise ValueError("labels accepts either ids or column, not both")
    ids = table[options["id_col"]] if "id_col" in options else pd.Series(table.index, index=table.index)
    if labels is not None and (ids.isna().any() or ids.duplicated().any()):
        raise ValueError("Explicit labels require unique, nonmissing feature IDs")
    selected = np.zeros(len(table), dtype=bool)
    if "ids" in options:
        requested = list(options["ids"])
        missing = set(requested) - set(ids)
        if missing:
            raise ValueError(f"Unknown label feature IDs: {missing}")
        selected = ids.isin(requested).to_numpy()
    elif "column" in options:
        selector = table[options["column"]]
        if not pd.api.types.is_bool_dtype(selector.dtype):
            raise ValueError("Label selection column must have boolean dtype")
        selected = selector.fillna(False).to_numpy(dtype=bool)
    display = ids.astype(str).to_numpy()
    if "display_col" in options:
        names = table[options["display_col"]]
        missing = names.isna() | names.astype("string").str.strip().str.casefold().isin(["", "nan"])
        display = names.astype(object).where(~missing, ids).astype(str).to_numpy()
    x = pd.to_numeric(table[effect_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    probabilities = pd.to_numeric(table[probability_col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(x) & np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)
    y = np.full(len(table), np.nan)
    # Zero is valid: retain raw p=0 and use the smallest positive float only for display.
    y[valid] = -np.log10(np.maximum(probabilities[valid], np.nextafter(0., 1.)))
    return pd.DataFrame({"feature_id": ids.to_numpy(), "raw_x": x, "raw_probability": table[probability_col].to_numpy(copy=True),
                         "raw_y": y, "valid": valid, "selected": selected, "display_label": display}, index=table.index)


def draw_rendering(ax, table, metadata, *, hue, styles, overflow, legend, palette, size,
                   xlimit, ylimit, legend_defaults, extra_handles, extra_labels):
    """Draw each observation once; split fills retain both categories under clipping."""
    styles = {} if styles is None else dict(styles)
    overflow = {} if overflow is None else dict(overflow)
    legend = {} if legend is None else dict(legend)
    unknown = set(overflow) - {"enabled", "marker"}
    if unknown:
        raise ValueError(f"Unknown overflow options: {sorted(unknown)}")
    enabled = overflow.get("enabled", True)
    unknown = set(legend) - {"visible", "order", "ncols", "loc", "bbox_to_anchor", "title", "fontsize", "markerscale", "labelspacing", "columnspacing", "handletextpad", "borderaxespad"}
    if unknown:
        raise ValueError(f"Unknown legend options: {sorted(unknown)}")
    memberships = []
    for value in table[hue]:
        group = tuple(value) if isinstance(value, (list, tuple)) else (() if pd.isna(value) else (value,))
        if len(group) > 2 or len(set(group)) != len(group):
            raise ValueError("Each feature may have at most two distinct categories")
        memberships.append(group)
    if isinstance(table[hue].dtype, pd.CategoricalDtype):
        category_memberships = [tuple(c) if isinstance(c, (list, tuple)) else (c,)
                                for c in table[hue].cat.categories]
        observed = list(dict.fromkeys(c for group in category_memberships for c in group))
    else:
        observed = list(dict.fromkeys(item for group in memberships for item in group))
    # Baseline palette identities must not change with style/legend ordering.
    palette_order = observed + [c for c in styles if c not in observed]
    colors = {c: palette[i % len(palette)] for i, c in enumerate(palette_order)}
    order = list(styles) + [item for item in observed if item not in styles]
    legend_order = order
    if "order" in legend:
        requested = list(legend.pop("order"))
        if len(set(requested)) != len(requested) or set(requested) - set(order):
            raise ValueError("legend order must contain distinct known categories")
        legend_order = requested + [item for item in order if item not in requested]
    resolved = {}
    for category in order:
        style = dict(styles.get(category, {}))
        unknown = set(style) - {"color", "marker", "size", "alpha", "edgecolor", "edgewidth"}
        if unknown:
            raise ValueError(f"Unknown group style options: {sorted(unknown)}")
        resolved[category] = {"color": colors[category], "marker": "o", "size": size,
                              "alpha": 1., "edgecolor": "none", "edgewidth": 0., **style}
    raw_x, raw_y = metadata.raw_x.to_numpy(), metadata.raw_y.to_numpy()
    valid = metadata.valid.to_numpy()
    cx = valid & ((raw_x < -xlimit) | (raw_x > xlimit))
    cy = valid & ((raw_y < 0) | (raw_y > ylimit))
    metadata["clipped_x"], metadata["clipped_y"] = cx, cy
    metadata["display_x"] = np.clip(raw_x, -xlimit, xlimit) if enabled else raw_x
    metadata["display_y"] = np.clip(raw_y, 0, ylimit) if enabled else raw_y
    metadata["rendered"] = valid & (enabled | ~(cx | cy))
    metadata["categories"] = memberships
    markers = np.full(len(metadata), "", dtype=object)
    batches = {}
    for position in np.flatnonzero(metadata.rendered.to_numpy()):
        group = memberships[position]
        # The first category in caller order owns geometry; colors retain both memberships.
        group = tuple(sorted(group, key=order.index))
        style = resolved[group[0]] if group else {"color": "gray", "marker": "o", "size": size, "alpha": .5, "edgecolor": "none", "edgewidth": 0.}
        marker = overflow.get("marker", "D") if enabled and (cx[position] or cy[position]) else style["marker"]
        markers[position] = marker
        # Batch by category/style and clipping state; avoid one artist per feature.
        key = (group, bool(enabled and (cx[position] or cy[position])))
        if key in batches:
            batches[key][0].append(position)
            continue
        kwargs = dict(marker=marker, markersize=np.sqrt(style["size"]), alpha=style["alpha"],
                      markeredgecolor=style["edgecolor"], markeredgewidth=style["edgewidth"],
                      markerfacecolor=style["color"], linestyle="none", clip_on=False)
        if len(group) == 2:
            if MarkerStyle(marker, fillstyle="left").get_alt_path() is None:
                raise ValueError("Split categories require a half-fillable marker (e.g. o, s, D, ^)")
            kwargs.update(fillstyle="left", markerfacecoloralt=resolved[group[1]]["color"])
        batches[key] = ([position], kwargs)
    metadata["marker"] = markers
    for positions, kwargs in batches.values():
        artist = Line2D(metadata.display_x.to_numpy()[positions], metadata.display_y.to_numpy()[positions], **kwargs)
        artist.set_gid(f"volcano_point_{positions[0]}")
        ax.add_line(artist)
    handles = [Line2D([], [], linestyle="none", marker=resolved[c]["marker"],
                      markerfacecolor=resolved[c]["color"], markeredgecolor=resolved[c]["edgecolor"],
                      markeredgewidth=resolved[c]["edgewidth"], alpha=resolved[c]["alpha"],
                      markersize=np.sqrt(resolved[c]["size"])) for c in legend_order]
    names = [str(c) for c in legend_order]
    if enabled and (cx | cy).any():
        handles.append(Line2D([], [], color="gray", linestyle="none", marker=overflow.get("marker", "D")))
        names.append("Outside display limits")
    if legend.pop("visible", True):
        ax.legend(handles + extra_handles, names + extra_labels, **{**legend_defaults, **legend})

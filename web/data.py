"""Validated AnnData import and bounded GUI metadata."""

import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def read_csv(path, *, numeric=False):
    # Inspect the header before pandas can silently rename duplicate columns.
    with open(path, encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header[1:]) != len(set(header[1:])) or any(not h for h in header[1:]):
        raise ValueError(f"{Path(path).name}: missing or duplicate column names.")
    # Parse the ID as an ordinary text column first: pandas may infer a numeric
    # index even with dtype=str when index_col=0 is used directly.
    parsed = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame = parsed.iloc[:, 1:].copy()
    frame.index = pd.Index(parsed.iloc[:, 0].to_numpy(), name=header[0] or None)
    if not frame.index.is_unique or any(not v.strip() for v in frame.index):
        raise ValueError(f"{Path(path).name}: row identifiers must be unique and nonempty.")
    if numeric:
        try:
            frame = frame.replace("", np.nan).apply(pd.to_numeric, errors="raise")
        except ValueError as exc:
            raise ValueError("X must contain numeric values; blank matrix cells represent missing values.") from exc
    else:
        # Preserve strings and IDs exactly. Numeric metadata types are an explicit GUI choice.
        frame = frame.replace("", np.nan)
    return frame


def load_csv_bundle(directory):
    directory = Path(directory)
    x = read_csv(directory / "X.csv", numeric=True)
    obs = read_csv(directory / "obs.csv")
    var = read_csv(directory / "var.csv")
    if set(x.index) != set(obs.index):
        raise ValueError("X row identifiers must match obs row identifiers exactly.")
    if set(x.columns) != set(var.index):
        raise ValueError("X column identifiers must match var row identifiers exactly.")
    return ad.AnnData(x.to_numpy(), obs=obs.loc[x.index].copy(), var=var.loc[x.columns].copy())


def check_h5ad(path, max_bytes):
    """Reject linked files and oversized expanded arrays before AnnData reads them."""
    with h5py.File(path, "r") as handle:
        seen = set()
        size = 0

        def inspect(group):
            nonlocal size
            address = h5py.h5o.get_info(group.id).addr
            if address in seen:
                raise ValueError("H5AD hard-link aliases or cycles are not supported.")
            seen.add(address)
            for key in group:
                link = group.get(key, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError("H5AD linked files and symbolic links are not supported.")
                item = group[key]
                if isinstance(item, h5py.Group):
                    inspect(item)
                else:
                    if item.is_virtual or item.external:
                        raise ValueError("H5AD external or virtual datasets are not supported.")
                    size += item.size * max(item.dtype.itemsize, 8)
                    if size > max_bytes:
                        raise ValueError("Expanded H5AD arrays exceed the configured import limit.")
        inspect(handle)


def validate_adata(adata):
    if not adata.n_obs or not adata.n_vars or adata.X is None:
        raise ValueError("The dataset must have observations, features, and an X matrix.")
    for label, names in (("obs", adata.obs_names), ("var", adata.var_names)):
        if not names.is_unique or any(not str(n).strip() for n in names):
            raise ValueError(f"{label} identifiers must be unique and nonempty.")
    for label, frame in (("obs", adata.obs), ("var", adata.var)):
        if not frame.columns.is_unique or any(not isinstance(c, str) or not c for c in frame.columns):
            raise ValueError(f"{label} metadata column names must be unique nonempty strings.")
    if adata.X.dtype.kind not in "biuf":
        raise ValueError("X must be a real numeric matrix.")


def table_preview(frame, rows=20, columns=12):
    part = frame.iloc[:rows, :columns].copy()
    # pandas handles numpy scalars, categorical values, dates, NaN and infinity here.
    return json.loads(part.to_json(orient="split", date_format="iso"))


def metadata(adata):
    obs = {}
    for col in adata.obs:
        series = adata.obs[col]
        values = series.dropna().unique()
        obs[col] = {"dtype": str(series.dtype), "missing": int(series.isna().sum()),
                    "nunique": len(values), "values": [str(v) for v in values[:200]],
                    "numeric": bool(pd.api.types.is_numeric_dtype(series))}
    preview = adata.X[:10, :8]
    if sparse.issparse(preview):
        preview = preview.toarray()
    return {"n_obs": adata.n_obs, "n_vars": adata.n_vars, "sparse": sparse.issparse(adata.X),
            "dtype": str(adata.X.dtype), "layers": list(adata.layers),
            "raw": adata.raw is not None,
            "features": adata.var_names.tolist(),
            "raw_features": adata.raw.var_names.tolist() if adata.raw is not None else [],
            "obs_columns": obs, "var_columns": adata.var.columns.tolist(),
            "previews": {"obs": table_preview(adata.obs), "var": table_preview(adata.var),
                         "X": table_preview(pd.DataFrame(preview, index=adata.obs_names[:10],
                                                         columns=adata.var_names[:8]))}}


def feature_labels(var, column=""):
    """Resolve display text across the full matrix feature table, before selection."""
    if not var.index.is_unique:
        raise ValueError("Feature identifiers must be unique in the chosen matrix.")
    if not column:
        return dict(zip(var.index, var.index))
    if column not in var.columns:
        raise ValueError(f"Unknown feature label column: {column}")
    labels = var[column].astype("string")
    missing = labels.isna() | labels.str.strip().str.lower().isin(["", "nan"])
    labels = labels.mask(missing, pd.Series(var.index, index=var.index)).astype(str)
    duplicated = labels.duplicated(keep=False) & ~missing
    reserved = set(labels[~duplicated])
    used = set()
    resolved = {}
    for identifier, label in labels.items():
        candidate = f"{label} [{identifier}]" if duplicated.loc[identifier] else label
        # An annotation can itself look like a disambiguated label.
        while candidate in used or (duplicated.loc[identifier] and candidate in reserved):
            candidate += f" [{identifier}]"
        resolved[identifier] = candidate
        used.add(candidate)
    return resolved


def read_feature_labels(path, matrix="X", column=""):
    """Read only feature IDs and one annotation column; never load expression arrays."""
    with h5py.File(path, "r") as handle:
        if matrix == "raw":
            if "raw" not in handle or "var" not in handle["raw"]:
                raise ValueError("This dataset has no raw matrix.")
            group = handle["raw/var"]
        else:
            if matrix != "X" and not (matrix.startswith("layer:") and matrix[6:] in handle.get("layers", {})):
                raise ValueError("Unknown matrix source.")
            group = handle["var"]
        columns = list(group.attrs["column-order"])
        if column and column not in columns:
            raise ValueError(f"Unknown feature label column: {column}")
        index = pd.Index(ad.io.read_elem(group[group.attrs["_index"]]))
        var = pd.DataFrame(index=index)
        if column:
            var[column] = ad.io.read_elem(group[column])
        return {"columns": columns, "labels": feature_labels(var, column)}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def demo_dataset():
    rng = np.random.default_rng(42)
    subjects = np.repeat([f"P{i:02}" for i in range(1, 17)], 2)
    group = np.tile(["Reference", "Treatment"], 16)
    age = np.repeat(rng.integers(25, 70, 16), 2)
    x = rng.lognormal(2, 0.35, (32, 8))
    x[group == "Treatment", :3] *= 1.6
    obs = pd.DataFrame({"condition": pd.Categorical(group, categories=["Reference", "Treatment"]),
                        "subject": subjects, "age": age, "visit": np.tile([0, 1], 16)},
                       index=[f"sample_{i:02}" for i in range(32)])
    var = pd.DataFrame({"label": [f"Marker {i}" for i in range(1, 9)]},
                       index=[f"feature_{i}" for i in range(1, 9)])
    data = ad.AnnData(x, obs=obs, var=var)
    data.layers["log1p"] = np.log1p(x)
    return data

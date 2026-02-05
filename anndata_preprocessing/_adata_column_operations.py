# _adata_column_operations.py
# module at adata_science_tools/anndata_preprocessing/_adata_column_operations.py
# first update 2026-02-04

# module imports
import pandas as pd
import numpy as np
import anndata as ad
import logging


def compute_var_ratios_sums_diffs_adata(
    adata: ad.AnnData,
    derived_variables_csv_file: str = "derived_variables_csv_file.csv",
    numerator_var_names_col: str = "numerator_var_names",
    denominator_var_names_col: str = "denominator_var_names",
    new_var_names_col: str = "new_var_names",
    var_meta_data_cols_list: list[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    transform: str = "linear",
    return_new_adata_only: bool = False,
    logger: logging.Logger | None = None,
    log_level="INFO",
):
    """
    Compute derived variables (ratios, sums, and differences) from an AnnData object.

    This function reads a CSV that specifies numerator/denominator expressions and
    returns a new AnnData with derived variables only, or concatenates it to the
    input AnnData along the variables axis (features).

    Notes
    -----
    - If `use_raw=True`, values are read from `adata.raw.X`.
    - If `layer` is provided, values are read from `adata.layers[layer]`.
    - `transform="ln"` assumes the selected matrix is natural-log transformed. Sums
      and differences are performed in linear space, then converted back to ln.
    """
    import re

    log = logger or logging.getLogger(__name__)
    if log_level is not None:
        log.setLevel(log_level)
        if log.handlers:
            for handler in log.handlers:
                handler.setLevel(log_level)
        elif logger is None:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
            log.addHandler(handler)
            log.propagate = False
    # log the parameters use dont use C-style string formatting to avoid issues with non-string types
    log.info(
        "compute_var_ratios_sums_diffs_adata called with parameters: "
        f"derived_variables_csv_file={derived_variables_csv_file}, "
        f"numerator_var_names_col={numerator_var_names_col}, "
        f"denominator_var_names_col={denominator_var_names_col}, "
        f"new_var_names_col={new_var_names_col}, "
        f"var_meta_data_cols_list={var_meta_data_cols_list}, "
        f"layer={layer}, use_raw={use_raw}, transform={transform}, "
        f"return_new_adata_only={return_new_adata_only}"
    )
    if adata is None:
        raise ValueError("adata is required.")
    if use_raw and adata.raw is None:
        raise ValueError("use_raw=True but adata.raw is None.")
    if transform not in {"linear", "ln"}:
        raise ValueError("transform must be 'linear' or 'ln'.")
    if use_raw and layer is not None:
        log.warning("use_raw=True; ignoring layer=%s.", layer)

    if use_raw:
        base_X = adata.raw.X
        base_var_names = adata.raw.var_names
        if adata.var_names.equals(adata.raw.var_names):
            base_var = adata.var.copy()
        else:
            log.warning("adata.var_names does not match adata.raw.var_names; using adata.raw.var.")
            base_var = adata.raw.var.copy()
    else:
        if layer is not None:
            if layer not in adata.layers:
                raise KeyError(f"layer '{layer}' not found in adata.layers.")
            base_X = adata.layers[layer]
        else:
            base_X = adata.X
        base_var_names = adata.var_names
        base_var = adata.var.copy()

    derived_df = pd.read_csv(derived_variables_csv_file, dtype=str)
    required_cols = {numerator_var_names_col, new_var_names_col}
    missing_cols = required_cols.difference(derived_df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in CSV: {sorted(missing_cols)}")
    if denominator_var_names_col not in derived_df.columns:
        derived_df[denominator_var_names_col] = ""

    if var_meta_data_cols_list is None:
        var_meta_data_cols_list = []
    elif not isinstance(var_meta_data_cols_list, list):
        raise TypeError("var_meta_data_cols_list must be a list or None.")
    else:
        missing_meta_cols = [c for c in var_meta_data_cols_list if c not in derived_df.columns]
        if missing_meta_cols:
            raise KeyError(f"Missing metadata columns in CSV: {missing_meta_cols}")

    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        text = str(value).strip()
        return text == "" or text.lower() in {"nan", "none"}

    token_re = re.compile(r"[+-]?[^+-]+")

    def _parse_terms(expr: str) -> list[tuple[str, int]]:
        if _is_missing(expr):
            return []
        expr_str = str(expr).replace(" ", "")
        tokens = token_re.findall(expr_str)
        terms: list[tuple[str, int]] = []
        for token in tokens:
            if token in {"", "+", "-"}:
                continue
            sign = -1 if token[0] == "-" else 1
            name = token[1:] if token[0] in {"+", "-"} else token
            name = name.strip()
            if name:
                terms.append((name, sign))
        return terms

    base_var_names_str = pd.Index(base_var_names).astype(str)
    name_to_idx = {name: idx for idx, name in enumerate(base_var_names_str)}

    try:
        import scipy.sparse as sp
        is_sparse = sp.issparse(base_X)
    except Exception:
        is_sparse = False

    exp_cache: dict[int, np.ndarray] = {}

    def _get_col(idx: int) -> np.ndarray:
        if is_sparse:
            return np.asarray(base_X[:, idx].toarray()).ravel()
        return np.asarray(base_X[:, idx]).ravel()

    def _get_linear_col(idx: int) -> np.ndarray:
        col = _get_col(idx)
        if transform == "ln":
            if idx in exp_cache:
                return exp_cache[idx]
            col = np.exp(col)
            exp_cache[idx] = col
        return col

    def _compute_linear_expr(terms: list[tuple[str, int]]) -> np.ndarray:
        values = None
        for name, sign in terms:
            idx = name_to_idx[name]
            col = _get_linear_col(idx)
            if values is None:
                values = sign * col
            else:
                values = values + sign * col
        if values is None:
            values = np.zeros(adata.n_obs, dtype=float)
        return values

    def _safe_log(values: np.ndarray, label: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        valid = values > 0
        if not np.all(valid):
            log.warning("%s has %d non-positive values; setting to NaN for log.", label, np.sum(~valid))
        out = np.full(values.shape, np.nan, dtype=float)
        out[valid] = np.log(values[valid])
        return out

    new_cols: list[np.ndarray] = []
    new_var_names: list[str] = []
    new_var_meta_rows: list[dict] = []
    skipped_rows = 0

    for row_idx, row in derived_df.iterrows():
        new_var_name = row.get(new_var_names_col)
        if _is_missing(new_var_name):
            msg = f"Row {row_idx} skipped: missing {new_var_names_col}."
            log.warning(msg)
            print(msg)
            skipped_rows += 1
            continue
        new_var_name = str(new_var_name).strip()
        if new_var_name in base_var_names_str:
            msg = f"Row {row_idx} skipped: new var name '{new_var_name}' already exists in adata.var_names."
            log.warning(msg)
            print(msg)
            skipped_rows += 1
            continue
        if new_var_name in new_var_names:
            msg = f"Row {row_idx} skipped: duplicate new var name '{new_var_name}' in CSV."
            log.warning(msg)
            print(msg)
            skipped_rows += 1
            continue

        numerator_terms = _parse_terms(row.get(numerator_var_names_col))
        if not numerator_terms:
            msg = f"Row {row_idx} ({new_var_name}) skipped: empty {numerator_var_names_col}."
            log.warning(msg)
            print(msg)
            skipped_rows += 1
            continue

        denominator_terms = _parse_terms(row.get(denominator_var_names_col))

        missing_vars = [name for name, _ in numerator_terms if name not in name_to_idx]
        missing_vars.extend([name for name, _ in denominator_terms if name not in name_to_idx])
        if missing_vars:
            msg = (
                f"Row {row_idx} ({new_var_name}) skipped: missing var_names in adata: "
                f"{sorted(set(missing_vars))}"
            )
            log.warning(msg)
            print(msg)
            skipped_rows += 1
            continue

        numerator_lin = _compute_linear_expr(numerator_terms)

        if denominator_terms:
            denominator_lin = _compute_linear_expr(denominator_terms)
            if transform == "linear":
                with np.errstate(divide="ignore", invalid="ignore"):
                    derived_values = numerator_lin / denominator_lin
            else:
                num_log = _safe_log(numerator_lin, f"{new_var_name} numerator")
                den_log = _safe_log(denominator_lin, f"{new_var_name} denominator")
                derived_values = num_log - den_log
        else:
            if transform == "linear":
                derived_values = numerator_lin
            else:
                derived_values = _safe_log(numerator_lin, f"{new_var_name} numerator")

        new_cols.append(np.asarray(derived_values, dtype=float))
        new_var_names.append(new_var_name)
        meta_row = {col: row.get(col) for col in var_meta_data_cols_list}
        new_var_meta_rows.append(meta_row)

    if not new_cols:
        log.warning("No derived variables were added. Skipped rows: %d", skipped_rows)
        empty_var = pd.DataFrame(columns=var_meta_data_cols_list)
        derived_adata = ad.AnnData(
            X=np.zeros((adata.n_obs, 0)),
            obs=adata.obs.copy(),
            var=empty_var,
        )
        return derived_adata if return_new_adata_only else ad.concat([adata, derived_adata], axis="var", join="outer")

    new_block = np.column_stack(new_cols)
    if is_sparse:
        import scipy.sparse as sp
        new_block = sp.csr_matrix(new_block)
    derived_var = pd.DataFrame(new_var_meta_rows, index=new_var_names)
    for col in var_meta_data_cols_list:
        if col not in derived_var.columns:
            derived_var[col] = pd.NA
    derived_adata = ad.AnnData(
        X=new_block,
        obs=adata.obs.copy(),
        var=derived_var,
    )


    log.info(
        "Computed %d derived variables (skipped %d rows).",
        len(new_var_names),
        skipped_rows,
    )
    
    # convert all columns of type object to string
    for col in derived_adata.var.select_dtypes(include=['object']).columns:
        derived_adata.var[col] = derived_adata.var[col].astype(str)
    log.info(f"derived_adata.var dtypes after conversion: \n{derived_adata.var.dtypes}")

    if return_new_adata_only:
        return derived_adata
    else:
        concat_adata = ad.concat([adata, derived_adata], axis="var", join="outer", merge="same")
        log.info(f"concat_adata.var dtypes after concatenation: \n{concat_adata.var.dtypes}")
        for col in derived_adata.var.select_dtypes(include=['object']).columns:
            concat_adata.var[col] = concat_adata.var[col].astype(str)
        log.info(f"concat_adata.var dtypes after conversion: \n{concat_adata.var.dtypes}")
        return concat_adata 


def compute_var_ratios_sums_diffs_adata_multiple_layers(
    adata: ad.AnnData,
    layers_to_compute: list[str] | None = None,
    layers_transforms: list[str] | None = None,
    base_layer: str | None = None,
    **kwargs,
) -> ad.AnnData:
    """
    Compute derived variables for multiple layers in an AnnData object.

    This function calls `compute_var_ratios_sums_diffs_adata` for each specified layer,
    stores each result in the derived AnnData layers, and concatenates the derived
    features to the input AnnData along the variables axis (features).

    Parameters
    ----------
    adata : ad.AnnData
        The input AnnData object.
    layers_to_compute : list[str] | None, optional
        List of layer names to compute derived variables for. If None, uses adata.X.
    layers_transforms : list[str] | None, optional
        List of transforms corresponding to `layers_to_compute`. If None, uses "linear"
        for all layers. Must match length of `layers_to_compute` if provided.
    base_layer : str | None, optional
        Layer name to use as the base `.X` for concatenation. If None, uses the
        first entry in `layers_to_compute`.
    **kwargs
        Additional keyword arguments passed to `compute_var_ratios_sums_diffs_adata`.

    Returns
    -------
    ad.AnnData
        The AnnData object with derived variables added for each specified layer.
    """
    log = kwargs.get("logger") or logging.getLogger(__name__)
    if "layer" in kwargs:
        log.warning("layers_to_compute is provided; ignoring 'layer' in kwargs.")
        kwargs.pop("layer")
    if "return_new_adata_only" in kwargs:
        log.warning("return_new_adata_only is managed internally; ignoring in kwargs.")
        kwargs.pop("return_new_adata_only")

    if layers_to_compute is None:
        layers_to_compute = [None]
    if not isinstance(layers_to_compute, list) or len(layers_to_compute) == 0:
        raise ValueError("layers_to_compute must be a non-empty list or None.")
    if layers_transforms is None:
        layers_transforms = ["linear"] * len(layers_to_compute)
    if not isinstance(layers_transforms, list) or len(layers_transforms) != len(layers_to_compute):
        raise ValueError("layers_transforms must be a list matching length of layers_to_compute.")
    if base_layer is None:
        base_layer = layers_to_compute[0]
    if base_layer is not None and base_layer not in layers_to_compute:
        raise ValueError("base_layer must be included in layers_to_compute or None.")


    derived_base = None
    for layer, layer_transform in zip(layers_to_compute, layers_transforms):
        temp_adata = compute_var_ratios_sums_diffs_adata(
            adata=adata,
            layer=layer,
            transform=layer_transform,
            return_new_adata_only=True,
            **kwargs,
        )
        if derived_base is None:
            derived_base = temp_adata
            if layer is not None:
                derived_base.layers[layer] = derived_base.X
        else:
            if not temp_adata.var_names.equals(derived_base.var_names):
                raise ValueError("Derived var_names differ across layers; cannot combine.")
            if layer is not None:
                derived_base.layers[layer] = temp_adata.X

    if derived_base is None:
        return adata

    adata_for_concat = adata
    if base_layer is not None:
        if base_layer not in adata.layers:
            raise KeyError(f"base_layer '{base_layer}' not found in adata.layers.")
        adata_for_concat = adata.copy()
        adata_for_concat.X = adata.layers[base_layer]
        if base_layer not in derived_base.layers:
            raise KeyError(f"base_layer '{base_layer}' not found in derived layers.")
        derived_base.X = derived_base.layers[base_layer]

    concat_adata = ad.concat([adata_for_concat, derived_base], axis="var", join="outer", merge="same")
    # set concat_adata.var index name to var_names
    concat_adata.var.index.name = "var_names"
    log.info(f"concat_adata.var dtypes after concatenation: \n{concat_adata.var.dtypes}")
    for col in concat_adata.var.select_dtypes(include=['object']).columns:
        concat_adata.var[col] = concat_adata.var[col].astype(str)
    log.info(f"concat_adata.var dtypes after conversion: \n{concat_adata.var.dtypes}")

    return concat_adata

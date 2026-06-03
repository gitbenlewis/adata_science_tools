# _adata_row_operations.py
# module at adata_science_tools/_preprocessing/_adata_row_operations.py
'''Functions for performing row-wise operations on AnnData objects, such as computing paired means and differences between groups of observations.'''
# updated 2026-03-13 added CFG_filter_adata_by_obs() function to filter adata by obs column values specified in config dict keys. Returns filtered adata object. Compatible with yaml file generated config dict.
#



# module imports
import logging
from typing import Optional, Sequence

import pandas as pd
import numpy as np
import anndata as ad


def CFG_filter_adata_by_obs(
    adata: ad.AnnData,
    dataset_cfg: Optional[dict] = None,
    filter_obs_boolean_column: Optional[str] = None,
    filter_obs_column_key: Optional[str] = None,
    filter_obs_column_values_list: Optional[Sequence] = None,
    copy: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> ad.AnnData:
    """
    Filter anndata object by obs column values specified in config dict keys
    or by explicit filter args.

    Accepts either `dataset_cfg` (a dict taken from the YAML dataset block)
    or explicit args: filter_obs_boolean_column, filter_obs_column_key,
    filter_obs_column_values_list. If dataset_cfg is provided, explicit args
    are used as fallbacks.

    Returns filtered AnnData (copy by default).
    """
    logger = logger or logging.getLogger(__name__)

    # prefer dataset_cfg keys if provided
    if dataset_cfg:
        filter_obs_boolean_column = dataset_cfg.get("filter_obs_boolean_column", filter_obs_boolean_column)
        filter_obs_column_key = dataset_cfg.get("filter_obs_column_key", filter_obs_column_key)
        filter_obs_column_values_list = dataset_cfg.get("filter_obs_column_values_list", filter_obs_column_values_list)

    # nothing to do
    if filter_obs_boolean_column is None and (filter_obs_column_key is None or filter_obs_column_values_list is None):
        logger.info("CFG_filter_adata_by_obs: no filtering keys provided; returning original adata (copy=%s)", copy)
        return adata.copy() if copy else adata

    # start from the original adata; apply filters in sequence with AND semantics
    _adata = adata

    if filter_obs_boolean_column is not None:
        if filter_obs_boolean_column not in _adata.obs.columns:
            raise KeyError(f"filter_obs_boolean_column '{filter_obs_boolean_column}' not found in adata.obs")
        # coerce to boolean
        mask_bool = _adata.obs[filter_obs_boolean_column].astype(bool)
        _adata = _adata[mask_bool, :].copy() if copy else _adata[mask_bool, :]
        logger.info("Filtered adata by boolean column '%s': %d observations remain", filter_obs_boolean_column, _adata.n_obs)

    if filter_obs_column_key is not None and filter_obs_column_values_list is not None:
        if filter_obs_column_key not in _adata.obs.columns:
            raise KeyError(f"filter_obs_column_key '{filter_obs_column_key}' not found in adata.obs")
        vals = list(filter_obs_column_values_list)

        # Try numeric-aware matching first; fall back to string matching
        try:
            vals_num = [float(v) for v in vals]
            col_num = pd.to_numeric(_adata.obs[filter_obs_column_key], errors="coerce")
            if not col_num.isna().all():
                mask_vals = col_num.isin(vals_num)
                if mask_vals.any():
                    _adata = _adata[mask_vals, :].copy() if copy else _adata[mask_vals, :]
                    logger.info("Applied numeric value filter on column '%s' (kept %d obs)", filter_obs_column_key, _adata.n_obs)
                else:
                    # no numeric matches -> try string
                    raise ValueError("No numeric matches found; falling back to string matching")
            else:
                raise ValueError("Column numeric conversion yields all NaN; falling back to string matching")
        except Exception:
            mask_vals = _adata.obs[filter_obs_column_key].astype(str).isin([str(v) for v in vals])
            _adata = _adata[mask_vals, :].copy() if copy else _adata[mask_vals, :]
            logger.info("Applied string value filter on column '%s' (kept %d obs)", filter_obs_column_key, _adata.n_obs)

    return _adata


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
    """
    Compute paired target-vs-reference transforms for matched AnnData observations.

    Pairing is controlled by `pair_by_key` in `params`. The returned AnnData has
    one observation per overlapping pair ID, indexed by the stringified pair ID.
    """
    params = dict(params)
    operation_flavor = params.pop("operation_flavor", None)
    if operation_flavor is not None:
        opperation_flavor = operation_flavor

    pair_by_key = params.pop("pair_by_key", None)
    layer = params.pop("layer", None)
    layers_to_compute = params.pop("layers_to_compute", None)
    base_layer_provided = "base_layer" in params
    base_layer = params.pop("base_layer", None)
    epsilon = params.pop("epsilon", 1e-9)
    target_min_value = params.pop("target_min_value", None)
    target_max_value = params.pop("target_max_value", None)
    ref_min_value = params.pop("ref_min_value", None)
    ref_max_value = params.pop("ref_max_value", None)
    merge_shared_obs_cols = params.pop("merge_shared_obs_cols", False)
    return_df = params.pop("return_df", False)
    allow_unused_params = params.pop("allow_unused_params", False)
    logger = params.pop("logger", None)
    log_level = params.pop("log_level", "INFO")
    save_source_values_obsm = params.pop("save_source_values_obsm", False)
    target_values_obsm_key = params.pop("target_values_obsm_key", "target_values")
    ref_values_obsm_key = params.pop("ref_values_obsm_key", "ref_values")

    if params and not allow_unused_params:
        raise ValueError(f"Unused params: {sorted(params)}")

    log = logger or logging.getLogger(__name__)
    if log_level is not None:
        log.setLevel(log_level)

    if pair_by_key is None:
        raise ValueError("pair_by_key is required.")
    if groupby_key not in adata.obs.columns:
        raise KeyError(f"groupby_key '{groupby_key}' not found in adata.obs.")
    if pair_by_key not in adata.obs.columns:
        raise KeyError(f"pair_by_key '{pair_by_key}' not found in adata.obs.")

    operation_aliases = {
        "subtraction": "subtraction",
        "subtract": "subtraction",
        "difference": "subtraction",
        "diff": "subtraction",
        "target_minus_ref": "subtraction",
        "relative_change_pct": "relative_change_pct",
        "relative_change_percent": "relative_change_pct",
        "percent_change": "relative_change_pct",
        "pct": "relative_change_pct",
        "relative_change_fc": "relative_change_fc",
        "fold_change": "relative_change_fc",
        "foldchange": "relative_change_fc",
        "fc": "relative_change_fc",
        "relative_change_l2fc": "relative_change_l2fc",
        "log2_fold_change": "relative_change_l2fc",
        "log2fc": "relative_change_l2fc",
        "l2fc": "relative_change_l2fc",
    }
    requested_operation = str(opperation_flavor)
    operation = operation_aliases.get(requested_operation.lower())
    if operation is None:
        raise ValueError(f"Unsupported opperation_flavor '{opperation_flavor}'.")

    if layers_to_compute is None:
        sources_to_compute = [layer]
    elif isinstance(layers_to_compute, (str, bytes)):
        sources_to_compute = [layers_to_compute]
    else:
        sources_to_compute = list(layers_to_compute)
    if not sources_to_compute:
        raise ValueError("layers_to_compute must be a non-empty list or None.")
    if not base_layer_provided:
        base_layer = sources_to_compute[0]
    if base_layer not in sources_to_compute:
        raise ValueError("base_layer must be included in layers_to_compute or None.")
    for source in sources_to_compute:
        if source is not None and source not in adata.layers:
            raise KeyError(f"layer '{source}' not found in adata.layers.")

    target_mask = adata.obs[groupby_key] == groupby_key_target_value
    ref_mask = adata.obs[groupby_key] == groupby_key_ref_value
    target_obs = adata.obs.loc[target_mask].copy()
    ref_obs = adata.obs.loc[ref_mask].copy()

    target_missing_pair_ids = target_obs.index[target_obs[pair_by_key].isna()].tolist()
    ref_missing_pair_ids = ref_obs.index[ref_obs[pair_by_key].isna()].tolist()
    if target_missing_pair_ids or ref_missing_pair_ids:
        raise ValueError(
            "Missing pair IDs found in selected observations: "
            f"target={target_missing_pair_ids}, ref={ref_missing_pair_ids}"
        )

    target_pair_ids = target_obs[pair_by_key].astype(str)
    ref_pair_ids = ref_obs[pair_by_key].astype(str)
    target_duplicate_ids = sorted(target_pair_ids[target_pair_ids.duplicated(keep=False)].unique())
    ref_duplicate_ids = sorted(ref_pair_ids[ref_pair_ids.duplicated(keep=False)].unique())
    if target_duplicate_ids or ref_duplicate_ids:
        raise ValueError(
            "Duplicate pair IDs found in selected observations: "
            f"target={target_duplicate_ids}, ref={ref_duplicate_ids}"
        )

    target_pair_id_set = set(target_pair_ids.tolist())
    ref_pair_id_set = set(ref_pair_ids.tolist())
    matched_pair_ids = sorted(target_pair_id_set.intersection(ref_pair_id_set))
    if not matched_pair_ids:
        raise ValueError("No overlapping pair IDs found between target and reference observations.")

    dropped_target_only_pair_ids = sorted(target_pair_id_set.difference(ref_pair_id_set))
    dropped_ref_only_pair_ids = sorted(ref_pair_id_set.difference(target_pair_id_set))
    if dropped_target_only_pair_ids or dropped_ref_only_pair_ids:
        log.info(
            "Dropping unmatched pair IDs: target_only=%s, ref_only=%s",
            dropped_target_only_pair_ids,
            dropped_ref_only_pair_ids,
        )

    target_obs_name_by_pair = pd.Series(target_obs.index.to_numpy(), index=target_pair_ids.to_numpy())
    ref_obs_name_by_pair = pd.Series(ref_obs.index.to_numpy(), index=ref_pair_ids.to_numpy())
    target_obs_names = target_obs_name_by_pair.loc[matched_pair_ids].to_numpy()
    ref_obs_names = ref_obs_name_by_pair.loc[matched_pair_ids].to_numpy()
    target_positions = adata.obs_names.get_indexer(target_obs_names)
    ref_positions = adata.obs_names.get_indexer(ref_obs_names)

    try:
        import scipy.sparse as sp

        def _is_sparse(value):
            return sp.issparse(value)
    except Exception:
        def _is_sparse(value):
            return False

    def _as_dense_array(value) -> np.ndarray:
        if _is_sparse(value):
            return value.toarray()
        return np.asarray(value)

    bounds_requested = any(
        value is not None
        for value in (target_min_value, target_max_value, ref_min_value, ref_max_value)
    )

    def _apply_bounds(value, min_value, max_value):
        if min_value is None and max_value is None:
            return value
        bounded_value = _as_dense_array(value).astype(float, copy=True)
        if min_value is not None:
            bounded_value = np.maximum(bounded_value, min_value)
        if max_value is not None:
            bounded_value = np.minimum(bounded_value, max_value)
        return bounded_value

    def _get_source_matrix(source):
        if source is None:
            return adata.X
        return adata.layers[source]

    def _compute_operation(source):
        matrix = _get_source_matrix(source)
        target_values = matrix[target_positions, :].copy()
        ref_values = matrix[ref_positions, :].copy()
        if bounds_requested:
            target_values = _apply_bounds(target_values, target_min_value, target_max_value)
            ref_values = _apply_bounds(ref_values, ref_min_value, ref_max_value)

        if operation == "subtraction":
            return target_values - ref_values, target_values, ref_values

        target_values = _as_dense_array(target_values).astype(float, copy=False)
        ref_values = _as_dense_array(ref_values).astype(float, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            if operation == "relative_change_pct":
                result_values = ((target_values - ref_values) / (ref_values + epsilon)) * 100
            elif operation == "relative_change_fc":
                result_values = (target_values + epsilon) / (ref_values + epsilon)
            else:
                result_values = np.log2((target_values + epsilon) / (ref_values + epsilon))
        return result_values, target_values, ref_values

    layer_results = {}
    base_target_values = None
    base_ref_values = None
    for source in sources_to_compute:
        result_values, target_values, ref_values = _compute_operation(source)
        layer_results[source] = result_values
        if source == base_layer:
            base_target_values = target_values
            base_ref_values = ref_values

    matched_index = pd.Index(matched_pair_ids, name=pair_by_key)
    target_aligned_obs = adata.obs.loc[target_obs_names].copy()
    ref_aligned_obs = adata.obs.loc[ref_obs_names].copy()
    target_aligned_obs.index = matched_index
    ref_aligned_obs.index = matched_index

    if obs_dfs == "merge":
        result_obs_data = {}
        for column in ref_aligned_obs.columns:
            ref_values = ref_aligned_obs[column]
            target_values = target_aligned_obs[column]
            if merge_shared_obs_cols and ref_values.equals(target_values):
                result_obs_data[column] = ref_values.to_numpy()
            else:
                result_obs_data[f"{column}{ref_obs_suffix}"] = ref_values.to_numpy()
                result_obs_data[f"{column}{target_obs_suffix}"] = target_values.to_numpy()
        if pair_by_key not in result_obs_data:
            result_obs_data = {pair_by_key: matched_pair_ids, **result_obs_data}
        else:
            result_obs_data[pair_by_key] = matched_pair_ids
        result_obs = pd.DataFrame(result_obs_data, index=matched_index)
    elif obs_dfs == "keep_ref":
        result_obs = ref_aligned_obs.copy()
        result_obs[pair_by_key] = matched_pair_ids
    elif obs_dfs == "keep_target":
        result_obs = target_aligned_obs.copy()
        result_obs[pair_by_key] = matched_pair_ids
    else:
        raise ValueError("obs_dfs must be 'merge', 'keep_ref', or 'keep_target'.")

    result_obs["ref_obs_name"] = ref_obs_names
    result_obs["target_obs_name"] = target_obs_names
    result_obs["pair_order"] = np.arange(len(matched_pair_ids), dtype=int)
    result_obs["ref_groupby_value"] = groupby_key_ref_value
    result_obs["target_groupby_value"] = groupby_key_target_value
    result_obs["ref_vs_target_operation"] = operation

    result_var = adata.var.copy() if keep_var_df else pd.DataFrame(index=adata.var_names.copy())
    result_var["ref_vs_target_operation"] = operation
    result_var["ref_vs_target_groupby_key"] = groupby_key
    result_var["ref_vs_target_target_value"] = groupby_key_target_value
    result_var["ref_vs_target_ref_value"] = groupby_key_ref_value
    result_var["ref_vs_target_pair_by_key"] = pair_by_key

    def _source_label(source):
        return ".X" if source is None else str(source)

    result_adata = ad.AnnData(
        X=layer_results[base_layer],
        obs=result_obs,
        var=result_var,
    )
    for source in sources_to_compute:
        if source is not None:
            result_adata.layers[source] = layer_results[source]

    source_labels = [_source_label(source) for source in sources_to_compute]
    result_adata.uns["ref_vs_target_adata"] = {
        "pair_by_key": pair_by_key,
        "groupby_key": groupby_key,
        "groupby_key_target_value": groupby_key_target_value,
        "groupby_key_ref_value": groupby_key_ref_value,
        "requested_operation_flavor": requested_operation,
        "operation_flavor": operation,
        "obs_dfs": obs_dfs,
        "ref_obs_suffix": ref_obs_suffix,
        "target_obs_suffix": target_obs_suffix,
        "keep_var_df": keep_var_df,
        "layer": _source_label(layer),
        "layers_to_compute": source_labels,
        "base_layer": _source_label(base_layer),
        "epsilon": epsilon,
        "target_min_value": target_min_value,
        "target_max_value": target_max_value,
        "ref_min_value": ref_min_value,
        "ref_max_value": ref_max_value,
        "merge_shared_obs_cols": merge_shared_obs_cols,
        "n_target_obs": int(target_mask.sum()),
        "n_ref_obs": int(ref_mask.sum()),
        "n_matched_pairs": len(matched_pair_ids),
        "matched_pair_ids": matched_pair_ids,
        "dropped_target_only_pair_ids": dropped_target_only_pair_ids,
        "dropped_ref_only_pair_ids": dropped_ref_only_pair_ids,
        "save_source_values_obsm": save_source_values_obsm,
        "target_values_obsm_key": target_values_obsm_key if save_source_values_obsm else None,
        "ref_values_obsm_key": ref_values_obsm_key if save_source_values_obsm else None,
    }
    result_adata.uns["ref_vs_target_dropped_target_only_pair_ids"] = dropped_target_only_pair_ids
    result_adata.uns["ref_vs_target_dropped_ref_only_pair_ids"] = dropped_ref_only_pair_ids

    if save_source_values_obsm:
        result_adata.obsm[target_values_obsm_key] = pd.DataFrame(
            _as_dense_array(base_target_values),
            index=result_adata.obs_names.copy(),
            columns=result_adata.var_names.copy(),
        )
        result_adata.obsm[ref_values_obsm_key] = pd.DataFrame(
            _as_dense_array(base_ref_values),
            index=result_adata.obs_names.copy(),
            columns=result_adata.var_names.copy(),
        )

    if return_df:
        result_df = pd.DataFrame(
            _as_dense_array(layer_results[base_layer]),
            index=result_adata.obs_names.copy(),
            columns=result_adata.var_names.copy(),
        )
        return result_adata, result_df
    return result_adata


def compute_paired_mean_adata(
    adata,
    layer='RFU',
    pair_by_key='AnimalID_Tattoo',
    groupby_key='Treatment_unique',
    datapoint_1='drug78hr',
    datapoint_2='drug30hr',
    debug_mode=True,
    layers_to_compute=None,
    base_layer=None,
):
    """
    Compute the average of two selected groups in an AnnData object and return a new AnnData object.
    
    Parameters:
    - adata: AnnData object
    - layer: The layer to use (default: 'RFU'). If None, uses .X
    - layers_to_compute: Optional list of layers to compute the mean for.
      If provided, the `layer` argument is ignored.
    - base_layer: The layer to use as `.X` in the returned AnnData when
      `layers_to_compute` is provided. Defaults to the first entry in
      `layers_to_compute`. Use None to base on `.X`.
    - pair_by_key: The key in .obs to pair the data
    - groupby_key: The key in .obs used to select groups
    - datapoint_1: First group to compute the mean
    - datapoint_2: Second group to compute the mean
    - debug_mode: If True, prints debug information
    
    Returns:
    - adata_mean_datapoint_12: AnnData object with averaged values
    - mean_datapoint_12_df: DataFrame of averaged values
    """
    _adata = adata.copy()

    multi_layer_mode = layers_to_compute is not None
    if layers_to_compute is None:
        layers_to_compute = [layer]
    if not isinstance(layers_to_compute, list) or len(layers_to_compute) == 0:
        raise ValueError("layers_to_compute must be a non-empty list or None.")
    if base_layer is None:
        base_layer = layers_to_compute[0]
    if base_layer is not None and base_layer not in layers_to_compute:
        raise ValueError("base_layer must be included in layers_to_compute or None.")

    for layer_name in layers_to_compute:
        if layer_name is not None and layer_name not in _adata.layers:
            raise KeyError(f"layer '{layer_name}' not found in adata.layers.")
    
    # Boolean indexing to select cells in each group
    datapoint_1_idx = _adata.obs[groupby_key].isin([datapoint_1])
    datapoint_2_idx = _adata.obs[groupby_key].isin([datapoint_2])
    
    # Ensure `pair_by_key` is categorical for proper sorting
    if not isinstance(_adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
        _adata.obs[pair_by_key] = _adata.obs[pair_by_key].astype('category')

    order_1 = np.argsort(_adata.obs.loc[datapoint_1_idx, pair_by_key].cat.codes)
    order_2 = np.argsort(_adata.obs.loc[datapoint_2_idx, pair_by_key].cat.codes)

    obs_names1 = _adata[datapoint_1_idx].obs_names.to_numpy()[order_1]
    obs_names2 = _adata[datapoint_2_idx].obs_names.to_numpy()[order_2]
    obs_names = 'avg_' + obs_names1 + '_' + obs_names2

    def _get_layer_matrix(layer_name):
        if layer_name is None:
            return _adata.X
        return _adata.layers[layer_name]

    layer_results = {}
    for layer_name in layers_to_compute:
        X = _get_layer_matrix(layer_name)
        # Select the data matrix for each group
        data_datapoint_1 = X[datapoint_1_idx, :].copy()
        data_datapoint_2 = X[datapoint_2_idx, :].copy()

        if debug_mode and layer_name == base_layer:
            display('data_datapoint_1:', data_datapoint_1.shape, data_datapoint_1[:11,:2])
            display('data_datapoint_2:', data_datapoint_2.shape, data_datapoint_2[:11,:2])

        # Sort the data by `pair_by_key`
        data_datapoint_1_rel = data_datapoint_1[order_1]
        data_datapoint_2_rel = data_datapoint_2[order_2]

        if debug_mode and layer_name == base_layer:
            display('data_datapoint_1_rel:', data_datapoint_1_rel.shape, data_datapoint_1_rel[:11,:2])
            display('data_datapoint_2_rel:', data_datapoint_2_rel.shape, data_datapoint_2_rel[:11,:2])

        # Compute the mean
        mean_datapoint_12 = (data_datapoint_1_rel + data_datapoint_2_rel) / 2
        layer_results[layer_name] = mean_datapoint_12

        if debug_mode and layer_name == base_layer:
            display('mean_datapoint_12:', mean_datapoint_12.shape, mean_datapoint_12[:11,:2])

    base_matrix = layer_results[base_layer]

    # Convert base layer to DataFrame
    mean_datapoint_12_df = pd.DataFrame(base_matrix, columns=_adata.var.index)
    mean_datapoint_12_df['obs_names1'] = obs_names1
    mean_datapoint_12_df['obs_names2'] = obs_names2
    mean_datapoint_12_df['obs_names'] = obs_names
    mean_datapoint_12_df.set_index('obs_names', inplace=True)
    mean_datapoint_12_df.drop(columns=['obs_names1', 'obs_names2'], inplace=True)

    if debug_mode:
        display(mean_datapoint_12_df.head(5))

    # Create new AnnData object
    adata_mean_datapoint_12 = ad.AnnData(X=mean_datapoint_12_df)
    adata_mean_datapoint_12.var_names = _adata.var_names
    adata_mean_datapoint_12.var = _adata.var.copy()

    if multi_layer_mode:
        for layer_name in layers_to_compute:
            if layer_name is None:
                continue
            adata_mean_datapoint_12.layers[layer_name] = layer_results[layer_name]

    if debug_mode:
        display(adata_mean_datapoint_12[:11,:2])
    
    return adata_mean_datapoint_12, mean_datapoint_12_df


def compute_paired_difference_adata(
    adata,
    layer='RFU',
    pair_by_key='AnimalID_Tattoo',
    groupby_key='Treatment_unique',
    datapoint_1='drug78hr',
    datapoint_2='drug30hr',
    debug_mode=True,
    layers_to_compute=None,
    base_layer=None,
):
    """
    Compute the paired difference (datapoint_1 - datapoint_2) in an AnnData object and return a new AnnData object.
    
    Parameters:
    - adata: AnnData object
    - layer: The layer to use (default: 'RFU'). If None, uses .X
    - layers_to_compute: Optional list of layers to compute the difference for.
      If provided, the `layer` argument is ignored.
    - base_layer: The layer to use as `.X` in the returned AnnData when
      `layers_to_compute` is provided. Defaults to the first entry in
      `layers_to_compute`. Use None to base on `.X`.
    - pair_by_key: The key in .obs to pair the data
    - groupby_key: The key in .obs used to select groups
    - datapoint_1: Group treated as minuend
    - datapoint_2: Group treated as subtrahend
    - debug_mode: If True, prints debug information
    
    Returns:
    - adata_diff_datapoint_12: AnnData object with difference values
    - diff_datapoint_12_df: DataFrame of difference values
    """
    _adata = adata.copy()

    multi_layer_mode = layers_to_compute is not None
    if layers_to_compute is None:
        layers_to_compute = [layer]
    if not isinstance(layers_to_compute, list) or len(layers_to_compute) == 0:
        raise ValueError("layers_to_compute must be a non-empty list or None.")
    if base_layer is None:
        base_layer = layers_to_compute[0]
    if base_layer is not None and base_layer not in layers_to_compute:
        raise ValueError("base_layer must be included in layers_to_compute or None.")

    for layer_name in layers_to_compute:
        if layer_name is not None and layer_name not in _adata.layers:
            raise KeyError(f"layer '{layer_name}' not found in adata.layers.")
    
    datapoint_1_idx = _adata.obs[groupby_key].isin([datapoint_1])
    datapoint_2_idx = _adata.obs[groupby_key].isin([datapoint_2])
    
    if not isinstance(_adata.obs[pair_by_key].dtype, pd.CategoricalDtype):
        _adata.obs[pair_by_key] = _adata.obs[pair_by_key].astype('category')

    order_1 = np.argsort(_adata.obs.loc[datapoint_1_idx, pair_by_key].cat.codes)
    order_2 = np.argsort(_adata.obs.loc[datapoint_2_idx, pair_by_key].cat.codes)

    obs_names1 = _adata[datapoint_1_idx].obs_names.to_numpy()[order_1]
    obs_names2 = _adata[datapoint_2_idx].obs_names.to_numpy()[order_2]
    obs_names = 'diff_' + obs_names1 + '_' + obs_names2

    def _get_layer_matrix(layer_name):
        if layer_name is None:
            return _adata.X
        return _adata.layers[layer_name]

    layer_results = {}
    for layer_name in layers_to_compute:
        X = _get_layer_matrix(layer_name)
        data_datapoint_1 = X[datapoint_1_idx, :].copy()
        data_datapoint_2 = X[datapoint_2_idx, :].copy()

        if debug_mode and layer_name == base_layer:
            display('data_datapoint_1:', data_datapoint_1.shape, data_datapoint_1[:11,:2])
            display('data_datapoint_2:', data_datapoint_2.shape, data_datapoint_2[:11,:2])

        data_datapoint_1_rel = data_datapoint_1[order_1]
        data_datapoint_2_rel = data_datapoint_2[order_2]

        if debug_mode and layer_name == base_layer:
            display('data_datapoint_1_rel:', data_datapoint_1_rel.shape, data_datapoint_1_rel[:11,:2])
            display('data_datapoint_2_rel:', data_datapoint_2_rel.shape, data_datapoint_2_rel[:11,:2])

        diff_datapoint_12 = data_datapoint_1_rel - data_datapoint_2_rel
        layer_results[layer_name] = diff_datapoint_12

        if debug_mode and layer_name == base_layer:
            display('diff_datapoint_12:', diff_datapoint_12.shape, diff_datapoint_12[:11,:2])

    base_matrix = layer_results[base_layer]

    diff_datapoint_12_df = pd.DataFrame(base_matrix, columns=_adata.var.index)
    diff_datapoint_12_df['obs_names1'] = obs_names1
    diff_datapoint_12_df['obs_names2'] = obs_names2
    diff_datapoint_12_df['obs_names'] = obs_names
    diff_datapoint_12_df.set_index('obs_names', inplace=True)
    diff_datapoint_12_df.drop(columns=['obs_names1', 'obs_names2'], inplace=True)

    if debug_mode:
        display(diff_datapoint_12_df.head(5))

    adata_diff_datapoint_12 = ad.AnnData(X=diff_datapoint_12_df)
    adata_diff_datapoint_12.var_names = _adata.var_names
    adata_diff_datapoint_12.var = _adata.var.copy()

    if multi_layer_mode:
        for layer_name in layers_to_compute:
            if layer_name is None:
                continue
            adata_diff_datapoint_12.layers[layer_name] = layer_results[layer_name]

    if debug_mode:
        display(adata_diff_datapoint_12[:11,:2])
    
    return adata_diff_datapoint_12, diff_datapoint_12_df

'''_adata = adata.copy()
adata_mean_datapoint_12, mean_datapoint_12_df = compute_mean_adata(
    adata=_adata,
    layer='Volume_norm',
    pair_by_key='AnimalID_Tattoo',
    groupby_key='Treatment_unique',
    datapoint_1='drug78hr',
    datapoint_2='drug30hr',
    debug_mode=False
)
display(adata_mean_datapoint_12[:11,:2], adata_mean_datapoint_12.obs.head(5))
display(mean_datapoint_12_df.head(5))'''

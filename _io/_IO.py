
import logging
from pathlib import Path
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd


def save_dataset(
    _adata: ad.AnnData,
    output_path: str | Path,
    logger: logging.Logger | None = None,
    save_obsm: bool = True,
    obsm_keys: Sequence[str] | None = None,
) -> None:
    """Save a parsed dataset as h5ad plus CSV exports."""
    logger = logger or logging.getLogger(__name__)
    output_path = Path(output_path)
    output_base_path = output_path.with_suffix("") if output_path.suffix == ".h5ad" else output_path
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    h5ad_path = Path(f"{output_base_path}.h5ad")
    obs_csv_path = Path(f"{output_base_path}.obs.csv")
    var_csv_path = Path(f"{output_base_path}.var.csv")
    x_csv_path = Path(f"{output_base_path}.X.csv")
    if save_obsm:
        if obsm_keys is None:
            keys_to_save = list(_adata.obsm.keys())
        elif isinstance(obsm_keys, (str, bytes)):
            keys_to_save = [str(obsm_keys)]
        else:
            keys_to_save = list(obsm_keys)
        missing_obsm_keys = [key for key in keys_to_save if key not in _adata.obsm]
        if missing_obsm_keys:
            raise KeyError(f"obsm key(s) not found: {missing_obsm_keys}")

        safe_obsm_names = {key: str(key).replace("/", "_") for key in keys_to_save}
        safe_name_counts = {}
        for safe_name in safe_obsm_names.values():
            safe_name_counts[safe_name] = safe_name_counts.get(safe_name, 0) + 1
        duplicated_safe_names = sorted(
            safe_name
            for safe_name, count in safe_name_counts.items()
            if count > 1
        )
        if duplicated_safe_names:
            raise ValueError(f"Sanitized obsm filenames collide: {duplicated_safe_names}")
    else:
        keys_to_save = []
        safe_obsm_names = {}

    logger.info("Saving adata to %s", h5ad_path)
    _adata.write_h5ad(h5ad_path)
    logger.info("Saving adata.obs to %s", obs_csv_path)
    _adata.obs.to_csv(obs_csv_path)
    logger.info("Saving adata.var to %s", var_csv_path)
    _adata.var.to_csv(var_csv_path)
    logger.info("Saving adata.X to %s", x_csv_path)
    x_data = _adata.X.toarray() if hasattr(_adata.X, "toarray") else _adata.X
    pd.DataFrame(x_data, index=_adata.obs_names, columns=_adata.var_names).to_csv(x_csv_path)
    for layer_name, layer_data in _adata.layers.items():
        safe_name = str(layer_name).replace("/", "_")
        layer_csv_path = Path(f"{output_base_path}.layer.{safe_name}.csv")
        logger.info("Saving adata.layers['%s'] to %s", layer_name, layer_csv_path)
        layer_values = layer_data.toarray() if hasattr(layer_data, "toarray") else layer_data
        pd.DataFrame(layer_values, index=_adata.obs_names, columns=_adata.var_names).to_csv(layer_csv_path)
    if save_obsm:
        for obsm_key in keys_to_save:
            obsm_value = _adata.obsm[obsm_key]
            if isinstance(obsm_value, pd.DataFrame):
                obsm_df = obsm_value
            else:
                obsm_values = obsm_value.toarray() if hasattr(obsm_value, "toarray") else np.asarray(obsm_value)
                if obsm_values.ndim != 2:
                    logger.warning(
                        "Skipping adata.obsm['%s'] because it is not 2D; shape=%s",
                        obsm_key,
                        getattr(obsm_values, "shape", None),
                    )
                    continue
                if obsm_values.shape[0] != _adata.n_obs:
                    logger.warning(
                        "Skipping adata.obsm['%s'] because row count %d does not match adata.n_obs %d",
                        obsm_key,
                        obsm_values.shape[0],
                        _adata.n_obs,
                    )
                    continue
                if obsm_values.shape[1] == _adata.n_vars:
                    obsm_columns = _adata.var_names
                else:
                    obsm_columns = [f"dim_{idx}" for idx in range(obsm_values.shape[1])]
                obsm_df = pd.DataFrame(obsm_values, index=_adata.obs_names, columns=obsm_columns)

            if obsm_df.ndim != 2:
                logger.warning(
                    "Skipping adata.obsm['%s'] because it is not 2D; shape=%s",
                    obsm_key,
                    getattr(obsm_df, "shape", None),
                )
                continue
            if obsm_df.shape[0] != _adata.n_obs:
                logger.warning(
                    "Skipping adata.obsm['%s'] because row count %d does not match adata.n_obs %d",
                    obsm_key,
                    obsm_df.shape[0],
                    _adata.n_obs,
                )
                continue

            obsm_csv_path = Path(f"{output_base_path}.obsm.{safe_obsm_names[obsm_key]}.csv")
            logger.info("Saving adata.obsm['%s'] to %s", obsm_key, obsm_csv_path)
            obsm_df.to_csv(obsm_csv_path)
    logger.info("Saved dataset to directory: %s", h5ad_path.parent)
    logger.info("Saved dataset with base filename: %s", h5ad_path.stem)


_save_dataset = save_dataset


############  
def make_df_obs_adataX(
    adata,
    layer: str | None = None,
    index: str | None = None,
    varcolumns: list[str] | str | None = None,
    include_obs: bool = True,
    use_raw: bool = False
):
    """
    
    Build a :class:`pandas.DataFrame` from an :class:`~anndata.AnnData` object.

    The function pulls an expression matrix from either
    ``adata.X`` / ``adata.layers`` (default) or ``adata.raw``
    (when *use_raw=True*) and optionally concatenates cell metadata
    (``adata.obs``) so that downstream analyses can be done with a single
    DataFrame.

    Parameters
    ----------
    adata
        Annotated data object to convert.
    layer
        Name of the layer to use instead of the main matrix.  
        Ignored if *layer=None*.
    index
        Column in ``adata.obs`` that should become the DataFrame's index.
    varcolumns
        Gene/feature labels.  
        * ``None`` - use ``.var_names`` that correspond to the chosen matrix.  
        * ``str`` - use that column in ``.var``.  
        * ``list`` -  
          - one element → as above;  
          - ≥2 elements → build a :class:`pandas.MultiIndex`.
    include_obs
        If *True*, prepend ``adata.obs`` to the expression table.
    use_raw
        If *True*, pull expression values (and associated ``.var`` table) from
        ``adata.raw`` instead of the main object.

    Returns
    -------
    pandas.DataFrame
        * ``shape = (n_obs, n_obs_meta + n_vars)`` when *include_obs=True*  
        * ``shape = (n_obs, n_vars)`` when *include_obs=False*

    Notes
    -----
    If the matrix is sparse the helper converts it to dense with
    ``toarray()``, trading memory for convenience.  For very large data
    sets consider:

    ```python
    from pandas.api.extensions import SparseDtype
    df = pd.DataFrame.sparse.from_spmatrix(X, index=idx, columns=vars)
    ```
    to keep the DataFrame itself sparse.
    """

    # ──────────────────────────────────────────────────────────────
    import pandas as pd
    from anndata import AnnData
    # Set up feature (variable) columns
    if varcolumns is None:
        varcolumns = adata.var_names
    elif isinstance(varcolumns, str):
        varcolumns = adata.var[varcolumns]
    elif isinstance(varcolumns, list):
        if len(varcolumns)==1:
            varcolumns = adata.var[varcolumns[0]]
        else:
            varcolumns = adata.var[varcolumns]
            varcolumns = pd.MultiIndex.from_arrays(varcolumns.values.T, names=varcolumns.columns)  
    # Set up the index
    index=adata.obs_names if index is None else adata.obs[index]
    # handle if use_raw =True 
    if use_raw and adata.raw is not None:
        X = adata.raw.X if layer is None else adata.raw.layers[layer] # Use the raw or raw.layer 
        print(f'Using raw data from adata.raw.{layer}.' if layer else 'Using raw data from adata.raw.X.')
    elif layer is not None and layer in adata.layers: 
        X = adata.layers[layer] # Use the specified layer 
        print(f'Using data from adata.layers.{layer}.')
    else:
        X = adata.X # Use the main data matrix
        print('Using data from adata.X.')
    
    if hasattr(X, "toarray"):  # Convert sparse matrix to dense if necessary
        X = X.toarray()

    df_adataX=pd.DataFrame(X,columns=varcolumns,index =index  )

    if include_obs:
        df_obs=adata.obs
        df_obs_adataX= pd.concat([df_obs,df_adataX], axis=1)
        return df_obs_adataX
    return df_adataX

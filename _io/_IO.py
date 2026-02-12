
import pandas as pd
from anndata import AnnData
from typing import Optional, Union, List


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
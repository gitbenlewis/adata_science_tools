# gseapy_post_process_helpers.py
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_PMID_33969320/code_library/gseapy_post_process_helpers.py
import os
import logging
import gseapy as gp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from gseapy import Msigdb

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def get_pvalue_filtered_term_list(gseapy_output_file, pvalue_column='fdr', pvalue_threshold=0.1,keeptop_n_terms=None, NES_column='nes',pos_NES_only= False, neg_NES_only= False):
    """
    Function to get the list of terms that have a p-value < 0.1 from the GSEAPY output file.
    Parameters:
    - gseapy_output_file (str): Path to the GSEAPY output file.
    - pvalue_column (str): Name of the p-value column (default is 'fdr'). pval or fdr
    - pvalue_threshold (float): p-value threshold (default is 0.1).
    - keeptop_n_terms (int): If specified, keep only the top n terms after filtering.
    - pos_NES_only (bool): If True, filter to keep only terms with NES >= 0.
    - neg_NES_only (bool): If True, filter to keep only terms with NES <= 0.
    Returns:
    list: List of terms with p-value < pvalue_threshold (default is 0.1).
    """
    import pandas as pd
    gseapy_df = pd.read_csv(gseapy_output_file, index_col='Term')
    if pos_NES_only:
        gseapy_df = gseapy_df[gseapy_df[NES_column] > 0]
    if neg_NES_only:
        gseapy_df = gseapy_df[gseapy_df[NES_column] < 0]
    pvalue_filtered_terms = gseapy_df[gseapy_df[pvalue_column] < pvalue_threshold]
    # sort by p-value column
    pvalue_filtered_terms = pvalue_filtered_terms.sort_values(by=pvalue_column, ascending=True)
    if keeptop_n_terms is not None:
        pvalue_filtered_terms = pvalue_filtered_terms.head(keeptop_n_terms)
    filtered_terms= pvalue_filtered_terms.index.tolist()
    return filtered_terms


def get_pvalue_filtered_term_list(gseapy_output_file, pvalue_column='fdr', pvalue_threshold=0.1,keeptop_n_terms=None, NES_column='nes',pos_NES_only= False, neg_NES_only= False):
    """
    Function to get the list of terms that have a p-value < 0.1 from the GSEAPY output file.
    Parameters:
    - gseapy_output_file (str): Path to the GSEAPY output file.
    - pvalue_column (str): Name of the p-value column (default is 'fdr'). pval or fdr
    - pvalue_threshold (float): p-value threshold (default is 0.1).
    - keeptop_n_terms (int): If specified, keep only the top n terms after filtering.
    - pos_NES_only (bool): If True, filter to keep only terms with NES >= 0.
    - neg_NES_only (bool): If True, filter to keep only terms with NES <= 0.
    Returns:
    list: List of terms with p-value < pvalue_threshold (default is 0.1).
    """
    import pandas as pd
    gseapy_df = pd.read_csv(gseapy_output_file, index_col='Term')
    if pos_NES_only:
        gseapy_df = gseapy_df[gseapy_df[NES_column] > 0]
    if neg_NES_only:
        gseapy_df = gseapy_df[gseapy_df[NES_column] < 0]
    pvalue_filtered_terms = gseapy_df[gseapy_df[pvalue_column] < pvalue_threshold]
    # sort by p-value column
    pvalue_filtered_terms = pvalue_filtered_terms.sort_values(by=pvalue_column, ascending=True)
    if keeptop_n_terms is not None:
        pvalue_filtered_terms = pvalue_filtered_terms.head(keeptop_n_terms)
    filtered_terms= pvalue_filtered_terms.index.tolist()
    return filtered_terms

import pandas as pd
def get_leading_edge_genes(gseapy_output_file, term_of_interest,gene_list_col='lead_genes'):
    """
    Function to get the leading edge genes for a given term from a gseapy_wrapper output file.
    Parameters:
    gseapy_output_file (str): Path to the gseapy output file.
    term_of_interest (str): The term for which to get the leading edge genes.
    Returns:
    list: List of leading edge genes for the specified term.
    """
    gseapy_df = pd.read_csv(gseapy_output_file, index_col='Term')
    leading_edge_genes = gseapy_df.loc[term_of_interest, gene_list_col].split(';')
    return leading_edge_genes

def make_dictionary_of_leading_edge_genes(gseapy_output_file, terms_of_interest_list=None,gene_list_col='lead_genes'):
    """
    Function to create a dictionary of leading edge genes for multiple terms.
    Parameters:
    gseapy_output_file (str): Path to the gseapy output file.
    terms_of_interest (list): List of terms for which to get the leading edge genes.
    If None, all terms in the file will be used.
    Returns:
    dict: Dictionary with terms as keys and lists of leading edge genes as values.
    """
    gseapy_df = pd.read_csv(gseapy_output_file, index_col='Term')
    if terms_of_interest_list is None:
        terms_of_interest_list = gseapy_df.index.tolist()
    leading_edge_dict = {}
    for term in terms_of_interest_list:
        leading_edge_dict[term] = get_leading_edge_genes(gseapy_output_file, term,gene_list_col=gene_list_col)
    return leading_edge_dict

from collections import Counter, defaultdict
from typing import Mapping, Iterable, Optional, List, Dict, Set, Union

def most_popular_genes(
    term_genes_dict: Mapping[str, Iterable[str]],
    *,
    count_by: str = "terms",              # "terms" or "occurrences"
    case_sensitive: bool = False,         # default normalizes to UPPER (useful for HGNC symbols)
    dedupe_within_term: bool = False,     # only used when count_by="occurrences"
    min_count: int = 1,
    top_n: Optional[int] = None,
    include_terms: bool = True,
    as_gene_count_terms_df: bool = False,
    expanded_terms_onehot: bool = False,  # if True, return a DataFrame with one-hot encoded terms
    rename: Optional[Mapping[str, str]] = None,  # e.g., {"PARK2": "PRKN"}
):
    """
    Compute gene 'popularity' across a dict of {term: iterable of gene symbols}.

    Parameters
    ----------
    term_genes_dict : Mapping[str, Iterable[str]]
        Map from term name to a list/iterable of gene names.
    count_by : {"terms","occurrences"}, default="terms"
        - "terms": count the number of DISTINCT terms each gene occurs in.
        - "occurrences": count total appearances across all lists.
    case_sensitive : bool, default=False
        If False, normalize names by upper-casing and stripping whitespace.
    dedupe_within_term : bool, default=False
        Only used when count_by="occurrences". If True, duplicates within a term
        won't inflate the occurrence count. For "terms", duplicates never inflate.
    min_count : int, default=1
        Keep only genes with count >= min_count.
    top_n : Optional[int], default=None
        If provided, return only the top N genes (ties broken alphabetically).
    include_terms : bool, default=True
        Include the sorted list of terms in which each gene appears.
    as_gene_count_terms_df : bool, default=False
        If True, return a pandas.DataFrame with columns ["gene","count","terms"].
    expanded_terms_onehot : bool, default=False
        If True, return a DataFrame with one-hot encoded terms as columns.
    rename : Optional[Mapping[str,str]], default=None
        Mapping of normalized names to preferred names (handles synonyms).
        If case_sensitive=False, keys are upper-cased internally.

    Returns
    -------
    List[dict] or pandas.DataFrame
        Sorted by count desc, gene asc. Each row has at least {"gene","count"}
        and optionally {"terms"} if include_terms=True.
    """
    if count_by not in {"terms", "occurrences"}:
        raise ValueError("count_by must be 'terms' or 'occurrences'")

    # Normalize rename map keys if needed so lookups are consistent
    _rename = None
    if rename is not None:
        if case_sensitive:
            _rename = dict(rename)
        else:
            _rename = {str(k).strip().upper(): v for k, v in rename.items() if str(k).strip()}

    def norm(x: Union[str, None]) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s if case_sensitive else s.upper()
        if _rename:
            return _rename.get(s, s)
        return s

    gene_to_terms: Dict[str, Set[str]] = defaultdict(set)
    occ = Counter()

    for term, genes in term_genes_dict.items():
        if not genes:
            continue
        term_name = str(term)
        seen_in_this_term: Set[str] = set()
        for g in genes:
            gn = norm(g)
            if gn is None:
                continue
            gene_to_terms[gn].add(term_name)
            if count_by == "occurrences":
                if dedupe_within_term and gn in seen_in_this_term:
                    continue
                seen_in_this_term.add(gn)
                occ[gn] += 1

    if count_by == "terms":
        counts = Counter({g: len(terms) for g, terms in gene_to_terms.items()})
    else:
        counts = occ

    items = [(g, c) for g, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if top_n is not None:
        items = items[:top_n]

    if as_gene_count_terms_df:
        import pandas as pd
        data = {
            "gene": [g for g, _ in items],
            "count": [c for _, c in items],
        }
        if include_terms:
            data["terms"] = [sorted(gene_to_terms[g]) for g, _ in items]
        df = pd.DataFrame(data)
        # If expanded_terms_onehot is requested, do one-hot encoding of terms
        if expanded_terms_onehot and include_terms:
            # Get all unique terms
            all_terms = set()
            for terms_list in df["terms"]:
                all_terms.update(terms_list)
            all_terms = sorted(all_terms)
            # Build one-hot columns
            onehot = []
            for terms_list in df["terms"]:
                row = {term: int(term in terms_list) for term in all_terms}
                onehot.append(row)
            onehot_df = pd.DataFrame(onehot, index=df.index)
            df = pd.concat([df, onehot_df], axis=1)
        return df
    else:
        result = []
        for g, c in items:
            row = {"gene": g, "count": c}
            if include_terms:
                row["terms"] = sorted(gene_to_terms[g])
            result.append(row)
        return result

    

# === Network analysis helpers built on top of most_popular_genes() ===
from typing import Tuple

def onehot_from_most_popular_df(df) -> pd.DataFrame:
    """
    Take the DataFrame returned by most_popular_genes(..., as_gene_count_terms_df=True, expanded_terms_onehot=True)
    and return a clean binary one-hot matrix with genes as index and terms as columns.
    Drops the ['count','terms'] columns if present.
    """
    if 'gene' in df.columns:
        df = df.copy()
        df = df.set_index('gene')
    # Drop non one-hot columns if present
    cols_to_drop = [c for c in ['count', 'terms'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    # Ensure strictly binary (0/1)
    return (df > 0).astype(int)


def _similarity_from_onehot(X, metric: str, mode: str):
    """
    Compute a sparse similarity/adjacency matrix from a binary one-hot matrix X.
    Parameters
    ----------
    X : scipy.sparse.csr_matrix (genes x terms)
    metric : {'cooccurrence','cosine','jaccard','pearson'}
    mode   : {'gene','term'} — similarity over rows ('gene') or columns ('term')
    Returns
    -------
    W : scipy.sparse.csr_matrix  (n_nodes x n_nodes)
    sizes : np.ndarray           (per-node set sizes)
    """
    import numpy as np
    import scipy.sparse as sp

    if mode not in {"gene", "term"}:
        raise ValueError("mode must be 'gene' or 'term'")

    X = X.tocsr()
    if mode == 'gene':
        sizes = np.asarray(X.sum(axis=1)).ravel()
        A_counts = (X @ X.T).tocsr()
    else:
        sizes = np.asarray(X.sum(axis=0)).ravel()
        A_counts = (X.T @ X).tocsr()

    # zero the diagonal
    A_counts.setdiag(0)
    A_counts.eliminate_zeros()

    metric = metric.lower()
    if metric == 'cooccurrence':
        W = A_counts
    elif metric == 'cosine':
        inv_norm = 1.0 / np.sqrt(np.maximum(sizes, 1))
        Dinv = sp.diags(inv_norm)
        W = (Dinv @ A_counts @ Dinv).tocsr()
    elif metric == 'jaccard':
        # w_ij = |A∩B| / |A∪B|
        C = A_counts.tocoo(copy=True)
        i, j, inter = C.row, C.col, C.data
        union = sizes[i] + sizes[j] - inter
        data = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
        W = sp.csr_matrix((data, (i, j)), shape=C.shape)
    elif metric == 'pearson':
        # For binary data, Pearson corr == phi coefficient. Use dense corr on the smaller axis.
        # Guard: only use when matrix is reasonably small.
        import numpy as np
        if mode == 'gene':
            M = X.toarray()
            mat = np.corrcoef(M)
        else:
            M = X.T.toarray()
            mat = np.corrcoef(M)
        np.fill_diagonal(mat, 0.0)
        mat = np.where(mat > 0, mat, 0.0)  # keep positive similarity only
        W = sp.csr_matrix(mat)
    else:
        raise ValueError("metric must be one of {'cooccurrence','cosine','jaccard','pearson'}")

    return W.tocsr(), sizes


def build_network_from_onehot(
    onehot_df: pd.DataFrame,
    *,
    mode: str = 'gene',
    metric: str = 'cosine',
    min_weight: float | int | None = None,
    top_k: int | None = None,
    keep_isolates: bool = False,
) -> Tuple["nx.Graph", pd.DataFrame, pd.DataFrame]:
    """
    Build a weighted NetworkX graph from a one-hot matrix (genes x terms).

    Parameters
    ----------
    onehot_df : pd.DataFrame
        Binary matrix with genes as rows and terms as columns.
    mode : {'gene','term'}
        Build a gene–gene network (shared terms) or term–term network (shared genes).
    metric : {'cooccurrence','cosine','jaccard','pearson'}
        Edge weight metric. For guidance: cooccurrence (ints), cosine∈[0,1], jaccard∈[0,1].
    min_weight : float|int|None
        Drop edges with weight < min_weight. Choose e.g. 2 for cooccurrence, 0.2–0.5 for cosine/jaccard.
    top_k : int|None
        Keep only the top-k strongest neighbors per node (applied after thresholding).
    keep_isolates : bool
        If False, remove nodes with degree 0.

    Returns
    -------
    G : nx.Graph
    nodes_df : pd.DataFrame with node metrics (degree, strength, centralities, community, size)
    edges_df : pd.DataFrame with edges (source, target, weight)
    """
    import numpy as np
    import pandas as pd
    import networkx as nx
    import scipy.sparse as sp
    from networkx.algorithms.community import greedy_modularity_communities

    # Ensure binary CSR matrix
    X = sp.csr_matrix(onehot_df.values.astype(int))

    W, sizes = _similarity_from_onehot(X, metric=metric, mode=mode)

    # Threshold by min_weight if requested
    if min_weight is not None:
        W = W.tocsr()
        mask = W.data >= float(min_weight)
        W.data = W.data * mask
        W.eliminate_zeros()

    # Sparsify by top-k per row
    if top_k is not None and top_k > 0:
        W = W.tolil()
        for r in range(W.shape[0]):
            row_idx = W.rows[r]
            row_vals = W.data[r]
            if len(row_vals) > top_k:
                keep_idx = np.argsort(row_vals)[-top_k:]
                keep_cols = {row_idx[i] for i in keep_idx}
                W.rows[r] = [c for c in row_idx if c in keep_cols]
                W.data[r] = [v for c, v in zip(row_idx, row_vals) if c in keep_cols]
        W = W.tocsr()

    # Build graph
    if mode == 'gene':
        node_names = list(onehot_df.index)
        size_label = 'n_terms'
    else:
        node_names = list(onehot_df.columns)
        size_label = 'n_genes'

    G = nx.Graph()
    for idx, name in enumerate(node_names):
        G.add_node(name, **{size_label: int(sizes[idx])})

    coo = W.tocoo()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i < j and w > 0:
            G.add_edge(node_names[i], node_names[j], weight=float(w))

    if not keep_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))

    # Compute communities (greedy modularity supports 'weight')
    try:
        comms = list(greedy_modularity_communities(G, weight='weight'))
        node2comm = {}
        for cid, nodes in enumerate(comms, start=1):
            for n in nodes:
                node2comm[n] = cid
    except Exception:
        node2comm = {n: None for n in G.nodes()}

    # Centrality metrics
    degree = dict(G.degree())
    strength = dict(G.degree(weight='weight'))
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in G.nodes()}
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    nodes_df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree': [degree[n] for n in G.nodes()],
        'strength': [strength[n] for n in G.nodes()],
        'betweenness': [betweenness[n] for n in G.nodes()],
        'eigenvector': [eigenvector[n] for n in G.nodes()],
        'community': [node2comm[n] for n in G.nodes()],
        size_label: [G.nodes[n][size_label] for n in G.nodes()],
    }).sort_values(['community','strength','degree'], ascending=[True, False, False])

    edges_df = pd.DataFrame([(u, v, d.get('weight', 1.0)) for u, v, d in G.edges(data=True)],
                            columns=['source', 'target', 'weight'])

    return G, nodes_df, edges_df


def network_from_most_popular_df(
    df: pd.DataFrame,
    *,
    mode: str = 'gene',
    metric: str = 'cosine',
    min_weight: float | int | None = None,
    top_k: int | None = None,
    keep_isolates: bool = False,
):
    """
    Convenience wrapper: feed the DataFrame returned by most_popular_genes(..., as_gene_count_terms_df=True,
    expanded_terms_onehot=True) and directly obtain a graph plus tables.

    Example
    -------
    >>> df = most_popular_genes(term2genes, as_gene_count_terms_df=True, expanded_terms_onehot=True)
    >>> G, nodes, edges = network_from_most_popular_df(df, mode='gene', metric='cosine', min_weight=0.3, top_k=10)
    """
    import pandas as pd
    onehot = onehot_from_most_popular_df(df)
    return build_network_from_onehot(onehot, mode=mode, metric=metric, min_weight=min_weight, top_k=top_k, keep_isolates=keep_isolates)


def export_graphml(G, path: str) -> str:
    """Export NetworkX graph to GraphML (Cytoscape/Gephi friendly) and return the path."""
    import os
    import networkx as nx
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    nx.write_graphml(G, path)
    return path
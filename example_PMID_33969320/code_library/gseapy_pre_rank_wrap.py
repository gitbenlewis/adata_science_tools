# gseapy_pre_rank_wrap.py
# version 2
# updated 20260126 to add return_single_pre_res option
# module at projects/gitbenlewis/run_GSEApy_wrapper/gseapy_pre_rank_wrap.py
'''
basic_set_of_enrichr_term_collections=[
'GO_Biological_Process_2025','GO_Cellular_Component_2025','GO_Molecular_Function_2025',
'KEGG_2021_Human','Reactome_2022', 'Reactome_Pathways_2024',
'WikiPathways_2024_Human', 'WikiPathways_2024_Mouse',
'MSigDB_Computational', 'MSigDB_Hallmark_2020',
'CellMarker_2024',
'ARCHS4_TFs_Coexp', 'ARCHS4_Tissues',
'Disease_Perturbations_from_GEO_down','Disease_Perturbations_from_GEO_up',
            ]

run_gseapy_prerank_multiple_term_collections(
    data_csv_path=deseq2_results_file,
    list_of_term_collections=basic_set_of_enrichr_term_collections,
    rank_metric_calc_flavor='signed_neg_log10_pvalue',
    l2fc_column_name= 'log2FoldChange',
    pvalue_column_name='padj',
    permutation_num=1000,
    threads=32,
    seed=6,
    file_base_name='HOM_over_WT',
    results_dir='./',
    GSEA_run_filename_tag='GSEA_prerank_run_multi',
    )
'''
 
# config

####################################


# module imports
import os
import numpy as np
import pandas as pd
import gseapy as gp
import gseapy.msigdb as gmsigdb
from typing import Mapping, Sequence, Union, Optional

import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

#####################################

def process_input_table(
        input_df:pd.DataFrame = None,
        feature_name_column_name: str ='feature_name',
        make_feature_names_unique: bool = True,
        keep_first_occurrence_for_rows_with_multiple_feature_names: bool = False,
        explode_rows_with_multiple_feature_names: bool = False,
        rows_with_multiple_feature_names_delimiter: Optional[str] = None,
        rank_metric_calc_flavor='signed_neg_log10_pvalue', # 'use_column_key' or 'signed_neg_log10_pvalue'
        existing_rank_metric_column_name: Optional[str] | None = None,
        new_rank_metric_column_name: Optional[str]  = 'rank',
        l2fc_column_name: Optional[str] | None = 'log2FoldChange',
        pvalue_column_name: Optional[str] | None ='pvalue',
        save_table_output: bool = True,
        results_dir: str ='./',
        file_base_name: str ='datasetID',
        GSEA_run_filename_tag: str ='GSEA_prerank_run',
        )->pd.DataFrame:
    '''
    '''
    _df = input_df.copy()
    # ensure feature_name_column_name column is string
    _df[feature_name_column_name] = _df[feature_name_column_name].astype(str)
    # handle rows with multiple feature names
    if keep_first_occurrence_for_rows_with_multiple_feature_names and rows_with_multiple_feature_names_delimiter is not None:
        _df[feature_name_column_name] = _df[feature_name_column_name].str.split(rows_with_multiple_feature_names_delimiter).str[0]
    # explode multiple feature names into multiple rows if needed
    if explode_rows_with_multiple_feature_names and rows_with_multiple_feature_names_delimiter is not None:
        _df[feature_name_column_name] = _df[feature_name_column_name].str.split(rows_with_multiple_feature_names_delimiter)
        _df = _df.explode(feature_name_column_name, ignore_index=True)
    # calculate the ranking metric
    if rank_metric_calc_flavor=='signed_neg_log10_pvalue': 
        _df = _df.dropna(subset=[feature_name_column_name, l2fc_column_name, pvalue_column_name])
        _df['neg_log10_pvalue'] = -np.log10(_df[pvalue_column_name].replace(0, np.nextafter(0, 1)))
        _df[new_rank_metric_column_name]= _df[l2fc_column_name].apply(np.sign) * _df['neg_log10_pvalue']
        ranking_series = _df.set_index(feature_name_column_name)[new_rank_metric_column_name].sort_values(ascending=False)
    elif rank_metric_calc_flavor=='use_column_key':
        if existing_rank_metric_column_name is None:
            raise ValueError("existing_rank_metric_column_name must be provided when using 'use_column_key' flavor.")
        ranking_series = _df.set_index(feature_name_column_name)[existing_rank_metric_column_name].sort_values(ascending=False)
    else:
        raise ValueError(f"Unknown rank_metric_calc_flavor: {rank_metric_calc_flavor}")
    # make the rnk_df from the ranking series
    rnk_df = ranking_series.reset_index().rename(columns={ranking_series.index.name:'feature_name', ranking_series.name:new_rank_metric_column_name})
    # make feature names unique by keeping the highest absolute values rank value
    if make_feature_names_unique:
        rnk_df['absolute_value_rnk'] = rnk_df[new_rank_metric_column_name].abs()
        rnk_df = rnk_df.sort_values(by=['absolute_value_rnk'], ascending=False).drop_duplicates(subset=['feature_name'], keep='first').drop(columns=['absolute_value_rnk'])
        # re-sort by rank metric
        rnk_df = rnk_df.sort_values(by=[new_rank_metric_column_name], ascending=False)
    # save the rnk_df if needed
    if save_table_output:
        out_file = os.path.join(results_dir, f"{file_base_name}.{GSEA_run_filename_tag}.rnk_df.csv")
        rnk_df.to_csv(out_file, index=False)
        LOGGER.info(f"Ranked DataFrame saved at {out_file}")
    return rnk_df


def run_single_gsea_prerank(
        rnk_df,
        single_gene_set_term_collection,
        collection_name: Optional[str] | None = None,
        min_size=10,
        max_size=1000,
        permutation_num=100,
        threads=32,
        seed=6,
        verbose=True,
        **kwargs,
        ):
    single_pre_res = gp.prerank(
        rnk=rnk_df,
        gene_sets=single_gene_set_term_collection,
        min_size=min_size,
        max_size=max_size,
        threads=threads,
        permutation_num=permutation_num,
        #outdir='./',
        seed=seed,
        verbose=verbose,
    )
    # Attach term collection information to each result in the dictionary.
    for key, value in single_pre_res.results.items():
        if isinstance(value, dict):
            single_pre_res.results[key]['TermCollection'] = collection_name if collection_name is not None else single_gene_set_term_collection
    # **Update the res2d DataFrame to include 'TermCollection'.**
    #single_pre_res.res2d['TermCollection'] = collection_name if collection_name is not None else single_gene_set_term_collection
    return single_pre_res



def make_results_dataframe(
        single_pre_res,
        collection_name: Optional[str] | None = None,
        results_table_min_genes_matched: int = 10,
        save_table_output: bool = True,
        results_dir: str ='./',
        file_base_name: str ='datasetID',
        GSEA_run_filename_tag: str ='GSEA_prerank_run',
        ):
    """Convert the pre_res.results into a tidy DataFrame and save if needed.
    The 'collection_name' parameter is used to annotate the results.
    returns: pd.DataFrame with columns 'TermCollection', 'Term', 'name', 'es', 'nes', 'pval', 'fdr', 'fwerp'

    """
    if collection_name is None:
        collection_name = getattr(single_pre_res, 'gene_sets', 'Unknown_TermCollection')
    df = pd.DataFrame()
    try:
        df = pd.DataFrame(single_pre_res.results).T
        # Filter for gene sets with enough matched genes.
        df = df[df['matched_genes'].str.split(';').str.len() >= results_table_min_genes_matched].copy()
        # Add custom columns based on Term and collection.
        df['Term'] = df.index
        df['TermCollection'] = collection_name
        df['TermCollection_Term'] = collection_name + '__' + df['Term']
        if 'tag %' in df.columns:
            df['leadhits_total'] = df['tag %'].apply(lambda x: x.split('/')[0] + '_' + x.split('/')[1] if isinstance(x, str) else x)
        # Rearrange columns.
        front_cols = ['TermCollection', 'Term', 'name', 'es', 'nes', 'pval', 'fdr', 'fwerp']
        if 'leadhits_total' in df.columns:
            front_cols.append('leadhits_total')
        remaining = [col for col in df.columns if col not in front_cols]
        df = df[front_cols + remaining]
        # Ensure numeric columns are floats.
        for col in ['es', 'nes', 'pval', 'fdr', 'fwerp']:
            df[col] = df[col].astype(float)
        df = df.sort_values(by=['TermCollection', 'fdr'], ascending=[True, True])
        if save_table_output:
            out_file = os.path.join(results_dir, f"{file_base_name}.{GSEA_run_filename_tag}.{collection_name}.csv")
            df.drop(columns=['RES'], inplace=True, errors='ignore')
            df.to_csv(out_file, index=False, )# quoting=csv.QUOTE_MINIMAL
            LOGGER.info(f"Results saved for {collection_name} at {out_file}")
    except Exception as e:
        LOGGER.exception("Error making results DataFrame for %s", collection_name)
    return df


def run_gseapy_prerank_multiple_term_collections(
        ## for input data
        data_df: Optional[pd.DataFrame] = None,
        data_csv_path: Optional[Union[str, os.PathLike]] = None,
        data_csv_path_sepstr: str =',',
        ##   process_input_table() # for preparing the ranking metric
        feature_name_column_name: str ='feature_name',
        make_feature_names_unique: bool = True,
        keep_first_occurrence_for_rows_with_multiple_feature_names: bool = False,
        explode_rows_with_multiple_feature_names: bool = False,
        rows_with_multiple_feature_names_delimiter: Optional[str] = None,
        rank_metric_calc_flavor: str ='signed_neg_log10_pvalue', # 'use_column_key' or 'signed_neg_log10_pvalue'
        existing_rank_metric_column_name: Optional[str] | None = None,
        new_rank_metric_column_name: Optional[str] | None = 'rank',
        l2fc_column_name: Optional[str] | None = 'log2FoldChange',
        pvalue_column_name: Optional[str] | None ='padj',
        ## for GSEApy prerank run_single_gsea_prerank() and make_results_dataframe() and main loop through term families
        list_of_term_collections: Union[str, Sequence[str],None] = None,
        list_of_term_collection_names: Union[str, Sequence[str],None] = None,
        min_size: int =10,
        max_size: int =1000,
        permutation_num: int = 100,
        results_table_min_genes_matched: int = 0,
        threads: int = 4,
        seed: int = 6,
        verbose: bool = True,
        save_table_output: bool = True,
        results_dir: str ='./',
        file_base_name: str ='datasetID',
        GSEA_run_filename_tag: str ='GSEA_prerank_run',
        return_single_pre_res: bool = False,
        )-> None:
        '''Run GSEApy prerank analysis on the given data.
        Either data_df or data_csv_path must be provided.
        '''   
        ## input verification and loading  
        if data_df is None and data_csv_path is None:
            raise ValueError("Either data_df or data_csv_path must be provided.")
        if data_df is None:
            data_df = pd.read_csv(data_csv_path, sep=data_csv_path_sepstr)
        if list_of_term_collections is None:
            raise ValueError("list_of_term_collections must be provided.")
        if isinstance(list_of_term_collections, str):
            list_of_term_collections = [list_of_term_collections]
        # log all parameters for process input table
        LOGGER.info(f"Running GSEApy prerank with parameters:\n"
                    f"feature_name_column_name: {feature_name_column_name}\n"
                    f"make_feature_names_unique: {make_feature_names_unique}\n"
                    f"keep_first_occurrence_for_rows_with_multiple_feature_names: {keep_first_occurrence_for_rows_with_multiple_feature_names}\n"
                    f"explode_rows_with_multiple_feature_names: {explode_rows_with_multiple_feature_names}\n"
                    f"rows_with_multiple_feature_names_delimiter: {rows_with_multiple_feature_names_delimiter}\n"
                    f"rank_metric_calc_flavor: {rank_metric_calc_flavor}\n"
                    f"existing_rank_metric_column_name: {existing_rank_metric_column_name}\n"
                    f"new_rank_metric_column_name: {new_rank_metric_column_name}\n"
                    f"l2fc_column_name: {l2fc_column_name}\n"
                    f"pvalue_column_name: {pvalue_column_name}\n"
                    f"save_table_output: {save_table_output}\n"
                    f"results_dir: {results_dir}\n"
                    f"file_base_name: {file_base_name}\n"
                    f"GSEA_run_filename_tag: {GSEA_run_filename_tag}\n"
                    )
        
        rnk_df = process_input_table(
            input_df=data_df,
            feature_name_column_name=feature_name_column_name,
            make_feature_names_unique=make_feature_names_unique,
            keep_first_occurrence_for_rows_with_multiple_feature_names=keep_first_occurrence_for_rows_with_multiple_feature_names,
            explode_rows_with_multiple_feature_names=explode_rows_with_multiple_feature_names,
            rows_with_multiple_feature_names_delimiter=rows_with_multiple_feature_names_delimiter,
            rank_metric_calc_flavor=rank_metric_calc_flavor,
            existing_rank_metric_column_name=existing_rank_metric_column_name,
            new_rank_metric_column_name=new_rank_metric_column_name,
            l2fc_column_name=l2fc_column_name,
            pvalue_column_name=pvalue_column_name,
            save_table_output=save_table_output,
            results_dir=results_dir,
            file_base_name=file_base_name,
            GSEA_run_filename_tag=GSEA_run_filename_tag,
        )

        # zip term collection names if provided
        if list_of_term_collection_names is not None:
            if isinstance(list_of_term_collection_names, str):
                list_of_term_collection_names = [list_of_term_collection_names]
            if len(list_of_term_collection_names) != len(list_of_term_collections):
                raise ValueError("Length of list_of_term_collection_names must match length of list_of_term_collections.")
            term_collections_with_names = zip(list_of_term_collections, list_of_term_collection_names)
        else:
            term_collections_with_names = [(tc, None) for tc in list_of_term_collections]
        # main loop through term collections
        for single_term_collection, collection_name in term_collections_with_names:
            LOGGER.info(f"\nRunning GSEApy prerank for term collection: {single_term_collection}")
            # log all parameters for run_single_gsea_prerank
            LOGGER.info(f"Parameters for run_single_gsea_prerank:\n"
                        f"rnk_df shape: {rnk_df.shape}\n"
                        f"single_term_collection: {single_term_collection}\n"
                        f"collection_name: {collection_name}\n"
                        f"permutation_num: {permutation_num}\n"
                        f"min_size: {min_size}\n"
                        f"max_size: {max_size}\n"
                        f"threads: {threads}\n"
                        f"seed: {seed}\n"
                        f"verbose: {verbose}\n"
                        )
            single_pre_res = run_single_gsea_prerank(
                rnk_df,
                single_term_collection,
                collection_name,
                permutation_num=permutation_num,
                min_size=min_size,
                max_size=max_size,
                threads=threads,
                seed=seed,
                verbose=verbose,
            )
            # log all parameters for make_results_dataframe
            LOGGER.info(f"Parameters for make_results_dataframe:\n"
                        f"collection_name: {collection_name}\n"
                        f"results_table_min_genes_matched: {results_table_min_genes_matched}\n"
                        f"save_table_output: {save_table_output}\n"
                        f"results_dir: {results_dir}\n"
                        f"file_base_name: {file_base_name}\n"
                        f"GSEA_run_filename_tag: {GSEA_run_filename_tag}\n"
                        )
            results_df = make_results_dataframe(
                single_pre_res, collection_name,
                results_table_min_genes_matched=results_table_min_genes_matched,
                save_table_output=save_table_output,
                results_dir=results_dir,
                file_base_name=file_base_name,
                GSEA_run_filename_tag=GSEA_run_filename_tag,)
            LOGGER.info(f"Processed term collection: {single_term_collection} with {results_df.shape[0]} significant terms with at least {results_table_min_genes_matched} matched genes.")
        # end of main loop through term collections
        LOGGER.info("GSEApy prerank analysis completed for all term collections.")
        
        if return_single_pre_res:
            return single_pre_res
        return


import ast
import pandas as pd

def geneset_table_with_multiple_feature_label_split_to_gseapy_gmt(
    geneset_table_with_multiple_feature_csv_path: str,
    gene_set_column_key: str = "CHEM_ID",
    save_output: bool = False,
    output_gmt_path: str | None = None,
) -> pd.DataFrame:
    """
    Convert a pathways CSV into a GMT-style DataFrame (first col = pathway, remaining = gene-set ids).
    gene_set_column_key sets which column contains the list-like ids (default: CHEM_ID).
    """
    geneset_table_df = pd.read_csv(geneset_table_with_multiple_feature_csv_path)
    required_cols = {"pathway", gene_set_column_key}
    if not required_cols.issubset(geneset_table_df.columns):
        raise ValueError(f"Input file must contain columns {required_cols}")

    def parse_ids(val):
        if isinstance(val, (list, tuple, set)):
            return [str(v) for v in val]
        if pd.isna(val):
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return [str(val)]

    id_lists = geneset_table_df[gene_set_column_key].apply(parse_ids)
    split_cols = id_lists.apply(lambda ids: "\t".join(ids)).str.split("\t", expand=True)
    gmt_df = geneset_table_df[["pathway"]].copy()
    gmt_df.insert(1, "description", "")  # gseapy expects name, description, then genes
    gmt_df = gmt_df.join(split_cols)
    gmt_df.columns = ["" for _ in gmt_df.columns]

    if save_output and output_gmt_path:
        gmt_df.to_csv(output_gmt_path, sep="\t", index=False, header=False)
    return gmt_df


def metabolon_chemical_annotation_to_pathway_table(
    metabolon_excel_path: str,
    sheet_name: str = "Chemical Annotation",
    groupby_col: str = "SUB_PATHWAY",
    columns_to_include: Optional[Sequence[str]] = None,
    save_output: bool = False,
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse a metabolon Excel file's Chemical Annotation tab into a pathway table.
    Returns a DataFrame with one row per pathway and list-valued columns of IDs/names.
    """
    if columns_to_include is None:
        columns_to_include = [
            "CHEMICAL_NAME",
            "CHEM_ID",
            "LIB_ID",
            "COMP_ID",
            "CHRO_LIB_ENTRY_ID",
            "CAS",
            "CHEMSPIDER",
            "HMDB",
            "KEGG",
            "PUBCHEM",
            "PLATFORM",
        ]

    chem_ann_df = pd.read_excel(metabolon_excel_path, sheet_name=sheet_name)

    required_cols = [groupby_col, *columns_to_include]
    missing = [c for c in required_cols if c not in chem_ann_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in Chemical Annotation: {missing}")

    chem_ann_df[columns_to_include] = chem_ann_df[columns_to_include].astype(str)
    res_df = (
        chem_ann_df.groupby(groupby_col)[columns_to_include]
        .agg(list)
        .reset_index()
        .rename(columns={groupby_col: "pathway"})
    )

    if save_output and output_csv_path:
        res_df.to_csv(output_csv_path, index=False)

    return res_df


def _write_gmt_dict(
    geneset_dict: Mapping[str, Sequence[str]],
    output_gmt_path: str,
    description: str = "",
) -> None:
    output_dir = os.path.dirname(output_gmt_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_gmt_path, "w") as handle:
        for term in sorted(geneset_dict.keys()):
            genes = [str(g) for g in geneset_dict.get(term, [])]
            line = "\t".join([term, description, *genes])
            handle.write(f"{line}\n")


def _load_homolog_mapping(
    mapping_file: str,
    source_col: str,
    target_col: str,
) -> dict[str, str]:
    mapping_df = pd.read_csv(mapping_file, dtype=str)
    required_cols = {source_col, target_col}
    if not required_cols.issubset(mapping_df.columns):
        raise ValueError(f"Mapping file must contain columns {required_cols}")
    mapping_df = mapping_df[[source_col, target_col]].dropna()
    mapping_df = mapping_df[(mapping_df[source_col] != "") & (mapping_df[target_col] != "")]
    return dict(zip(mapping_df[source_col], mapping_df[target_col]))


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _ensure_parent_dir(path: str) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def make_gmts_from_enrichr_libraries(
    list_of_term_collections: Sequence[str],
    outdir: str,
    organism: str = "Human",
    max_size=20000,
    min_size=0

) -> list[str]:
    """
    Download Enrichr libraries and write one GMT per collection.
    Files are named <collection>.symbols.gmt.
    """
    if not list_of_term_collections:
        raise ValueError("list_of_term_collections must be provided.")
    os.makedirs(outdir, exist_ok=True)
    output_files = []
    for collection in list_of_term_collections:
        geneset_dict = gp.get_library(collection, organism=organism,max_size = max_size, min_size = min_size)
        output_gmt_path = os.path.join(outdir, f"{collection}.gmt")
        _write_gmt_dict(geneset_dict, output_gmt_path, description=collection)
        LOGGER.info("Saved Enrichr GMT for %s at %s", collection, output_gmt_path)
        output_files.append(output_gmt_path)
    return output_files


def make_gmts_from_Msigdb(
    dbver: str = "2025.1.Mm",
    list_of_Msigdb_categories: Optional[Sequence[str]] = None,
    outdir: str = "./",
) -> list[str]:
    """
    Download MSigDB collections and write one GMT per category.
    Files are named <category>.v<dbver>.symbols.gmt.
    """
    os.makedirs(outdir, exist_ok=True)
    if list_of_Msigdb_categories is None:
        list_of_Msigdb_categories = gmsigdb.Msigdb.list_category(dbver)
        if not list_of_Msigdb_categories:
            raise ValueError(f"No MSigDB categories found for dbver={dbver}")
    output_files = []
    for category in list_of_Msigdb_categories:
        geneset_dict = gmsigdb.Msigdb.get_gmt(category=category, dbver=dbver, entrez=False)
        if geneset_dict is None:
            LOGGER.info("No GMT returned for category %s (dbver=%s)", category, dbver)
            continue
        output_gmt_path = os.path.join(outdir, f"{category}.gmt")
        _write_gmt_dict(geneset_dict, output_gmt_path, description=f"{category}.v{dbver}")
        LOGGER.info("Saved MSigDB GMT for %s at %s", category, output_gmt_path)
        output_files.append(output_gmt_path)
    return output_files


def convert_gmt_homologs(
    gmt_input_file: str,
    gmt_out_put_file: str,
    covert2organism: str = "Mouse",
    homolog_mapping_h2m_file: Optional[str] = None,
    source_index_col: str = "gene_name",
    mouse_gene_name_col: str = "mouse_gene_name",
    homolog_mapping_m2h_file: Optional[str] = None,
    human_gene_name_col: str = "human_gene_name",
) -> str:
    """
    Convert a GMT file between human and mouse using one-to-one symbol mappings.
    For Mouse conversion: map source_index_col -> mouse_gene_name_col using homolog_mapping_h2m_file.
    For Human conversion: map mouse_gene_name_col -> human_gene_name_col using homolog_mapping_m2h_file.
    """
    target = covert2organism.strip().lower()
    if target == "mouse":
        if homolog_mapping_h2m_file is None:
            raise ValueError("homolog_mapping_h2m_file must be provided for Mouse conversion.")
        mapping = _load_homolog_mapping(homolog_mapping_h2m_file, source_index_col, mouse_gene_name_col)
    elif target == "human":
        if homolog_mapping_m2h_file is None:
            raise ValueError("homolog_mapping_m2h_file must be provided for Human conversion.")
        mapping = _load_homolog_mapping(homolog_mapping_m2h_file, source_index_col, human_gene_name_col)
    else:
        raise ValueError("covert2organism must be 'Mouse' or 'Human'.")

    _ensure_parent_dir(gmt_out_put_file)
    with open(gmt_input_file, "r") as in_handle, open(gmt_out_put_file, "w") as out_handle:
        for line in in_handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            term = parts[0] if len(parts) > 0 else ""
            description = parts[1] if len(parts) > 1 else ""
            genes = parts[2:] if len(parts) > 2 else []
            mapped_genes = [mapping[g] for g in genes if g in mapping]
            mapped_genes = _dedupe_preserve_order(mapped_genes)
            out_handle.write("\t".join([term, description, *mapped_genes]) + "\n")
    LOGGER.info("Converted GMT %s -> %s (target=%s)", gmt_input_file, gmt_out_put_file, covert2organism)
    return gmt_out_put_file


def convert_gmt_homologs_directory(
    gmt_input_directory: Optional[str] = None,
    gmt_output_directory: Optional[str] = None,
    gmt_input_file: Optional[str] = None,
    gmt_out_put_file: Optional[str] = None,
    covert2organism: str = "Mouse",
    homolog_mapping_h2m_file: Optional[str] = None,
    source_index_col: str = "gene_name",
    mouse_gene_name_col: str = "mouse_gene_name",
    homolog_mapping_m2h_file: Optional[str] = None,
    human_gene_name_col: str = "human_gene_name",
) -> list[str]:
    """
    Convert all GMT files in a directory (or one file) using convert_gmt_homologs().
    Output filenames append .h2m.gmt or .m2h.gmt.
    """
    target = covert2organism.strip().lower()
    if target == "mouse":
        suffix = ".h2m.gmt"
    elif target == "human":
        suffix = ".m2h.gmt"
    else:
        raise ValueError("covert2organism must be 'Mouse' or 'Human'.")

    if gmt_input_file is not None:
        if gmt_out_put_file is None:
            base_name = os.path.basename(gmt_input_file)
            base_stem = base_name[:-4] if base_name.lower().endswith(".gmt") else base_name
            out_dir = gmt_output_directory or os.path.dirname(gmt_input_file)
            gmt_out_put_file = os.path.join(out_dir, f"{base_stem}{suffix}")
        input_files = [gmt_input_file]
        output_files = [gmt_out_put_file]
    else:
        if gmt_input_directory is None:
            raise ValueError("gmt_input_directory must be provided when gmt_input_file is not set.")
        if gmt_output_directory is None:
            raise ValueError("gmt_output_directory must be provided when converting a directory.")
        input_files = sorted(
            os.path.join(gmt_input_directory, f)
            for f in os.listdir(gmt_input_directory)
            if f.lower().endswith(".gmt")
        )
        output_files = []
        for input_file in input_files:
            base_name = os.path.basename(input_file)
            base_stem = base_name[:-4] if base_name.lower().endswith(".gmt") else base_name
            output_files.append(os.path.join(gmt_output_directory, f"{base_stem}{suffix}"))

    if not input_files:
        LOGGER.info("No GMT files found to convert in %s", gmt_input_directory)
        return []
    _ensure_parent_dir(output_files[0])
    converted_files = []
    for in_file, out_file in zip(input_files, output_files):
        converted_files.append(
            convert_gmt_homologs(
                gmt_input_file=in_file,
                gmt_out_put_file=out_file,
                covert2organism=covert2organism,
                homolog_mapping_h2m_file=homolog_mapping_h2m_file,
                source_index_col=source_index_col,
                mouse_gene_name_col=mouse_gene_name_col,
                homolog_mapping_m2h_file=homolog_mapping_m2h_file,
                human_gene_name_col=human_gene_name_col,
            )
        )
    return converted_files









############### end of module functions some usefule lists below ###############

basic_set_of_enrichr_term_collections=[
'GO_Biological_Process_2025','GO_Cellular_Component_2025','GO_Molecular_Function_2025',
'KEGG_2021_Human','Reactome_2022', 'Reactome_Pathways_2024',
'WikiPathways_2024_Human', 'WikiPathways_2024_Mouse',
'MSigDB_Computational', 'MSigDB_Hallmark_2020',
'CellMarker_2024',
'ARCHS4_TFs_Coexp', 'ARCHS4_Tissues',
'Disease_Perturbations_from_GEO_down','Disease_Perturbations_from_GEO_up',
            ]
expansion_set_of_enrichr_term_collections=[
'ARCHS4_Cell-lines','ARCHS4_IDG_Coexp','ARCHS4_Kinases_Coexp','ARCHS4_TFs_Coexp', 'ARCHS4_Tissues',
'Tabula_Sapiens', 'Tabula_Muris',
'MGI_Mammalian_Phenotype_Level_4_2021','Human_Phenotype_Ontology',
 'GWAS_Catalog_2025','UK_Biobank_GWAS_v1', 'OMIM_Disease',
'MSigDB_Computational', 'MSigDB_Hallmark_2020', 'MSigDB_Oncogenic_Signatures',
'Kinase_Perturbations_from_GEO_down','Kinase_Perturbations_from_GEO_up',
'Disease_Perturbations_from_GEO_down', 'Disease_Perturbations_from_GEO_up', 
 'Aging_Perturbations_from_GEO_down','Aging_Perturbations_from_GEO_up',
'Gene_Perturbations_from_GEO_down','Gene_Perturbations_from_GEO_up',
'TF-LOF_Expression_from_GEO',
'Drug_Perturbations_from_GEO_down', 'Drug_Perturbations_from_GEO_up', 
'RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO', 
            ]
#names = gp.get_library_name()
#names[:]
# comment out version of earlier years 
all_enrichr_libraries_2025_sin_archive = [
'ARCHS4_Cell-lines',
 'ARCHS4_IDG_Coexp',
 'ARCHS4_Kinases_Coexp',
 'ARCHS4_TFs_Coexp',
 'ARCHS4_Tissues',
 'Achilles_fitness_decrease',
 'Achilles_fitness_increase',
 'Aging_Perturbations_from_GEO_down',
 'Aging_Perturbations_from_GEO_up',
 'Allen_Brain_Atlas_10x_scRNA_2021',
 'Allen_Brain_Atlas_down',
 'Allen_Brain_Atlas_up',
 'Azimuth_2023',
 'Azimuth_Cell_Types_2021',
 #'BioCarta_2013',
 #'BioCarta_2015',
 'BioCarta_2016',
 'BioPlanet_2019',
 'BioPlex_2017',
 'CCLE_Proteomics_2020',
 'CM4AI_U2OS_Protein_Localization_Assemblies',
 'COMPARTMENTS_Curated_2025',
 'COMPARTMENTS_Experimental_2025',
 'CORUM',
 'COVID-19_Related_Gene_Sets',
 'COVID-19_Related_Gene_Sets_2021',
 'Cancer_Cell_Line_Encyclopedia',
 'CellMarker_2024',
 'CellMarker_Augmented_2021',
 #'ChEA_2013',
 #'ChEA_2015',
 #'ChEA_2016',
 'ChEA_2022',
 'Chromosome_Location',
 'Chromosome_Location_hg19',
 #'ClinVar_2019',
 'ClinVar_2025',
 'DGIdb_Drug_Targets_2024',
 'DSigDB',
 'Data_Acquisition_Method_Most_Popular_Genes',
 'DepMap_CRISPR_GeneDependency_CellLines_2023',
 'DepMap_WG_CRISPR_Screens_Broad_CellLines_2019',
 'DepMap_WG_CRISPR_Screens_Sanger_CellLines_2019',
 'Descartes_Cell_Types_and_Tissue_2021',
 'Diabetes_Perturbations_GEO_2022',
 'DisGeNET',
 'Disease_Perturbations_from_GEO_down',
 'Disease_Perturbations_from_GEO_up',
 'Disease_Signatures_from_GEO_down_2014',
 'Disease_Signatures_from_GEO_up_2014',
 'DrugMatrix',
 'Drug_Perturbations_from_GEO_2014',
 'Drug_Perturbations_from_GEO_down',
 'Drug_Perturbations_from_GEO_up',
 #'ENCODE_Histone_Modifications_2013',
 'ENCODE_Histone_Modifications_2015',
 #'ENCODE_TF_ChIP-seq_2014',
 'ENCODE_TF_ChIP-seq_2015',
 'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X',
 'ESCAPE',
 'Elsevier_Pathway_Collection',
 'Enrichr_Libraries_Most_Popular_Genes',
 'Enrichr_Submissions_TF-Gene_Coocurrence',
 'Enrichr_Users_Contributed_Lists_2020',
 'Epigenomics_Roadmap_HM_ChIP-seq',
 'FANTOM6_lncRNA_KD_DEGs',
 #'GO_Biological_Process_2021',
 #'GO_Biological_Process_2023',
 'GO_Biological_Process_2025',
 #'GO_Cellular_Component_2021',
 #'GO_Cellular_Component_2023',
 'GO_Cellular_Component_2025',
 #'GO_Molecular_Function_2021',
 #'GO_Molecular_Function_2023',
 'GO_Molecular_Function_2025',
 'GTEx_Aging_Signatures_2021',
 'GTEx_Tissue_Expression_Down',
 'GTEx_Tissue_Expression_Up',
 'GTEx_Tissues_V8_2023',
 #'GWAS_Catalog_2019',
 #'GWAS_Catalog_2023',
 'GWAS_Catalog_2025',
 'GeDiPNet_2023',
 'GeneSigDB',
 'Gene_Perturbations_from_GEO_down',
 'Gene_Perturbations_from_GEO_up',
 'Genes_Associated_with_NIH_Grants',
 'Genome_Browser_PWMs',
 'GlyGen_Glycosylated_Proteins_2022',
 'HDSigDB_Human_2021',
 'HDSigDB_Mouse_2021',
 'HMDB_Metabolites',
 'HMS_LINCS_KinomeScan',
 'HomoloGene',
 'HuBMAP_ASCT_plus_B_augmented_w_RNAseq_Coexpression',
 'HuBMAP_ASCTplusB_augmented_2022',
 'HumanCyc_2015',
 'HumanCyc_2016',
 'Human_Gene_Atlas',
 'Human_Phenotype_Ontology',
 'IDG_Drug_Targets_2022',
 'InterPro_Domains_2019',
 'JASPAR_PWM_Human_2025',
 'JASPAR_PWM_Mouse_2025',
 'Jensen_COMPARTMENTS',
 'Jensen_DISEASES',
 'Jensen_DISEASES_Curated_2025',
 'Jensen_DISEASES_Experimental_2025',
 'Jensen_TISSUES',
# 'KEA_2013',
 'KEA_2015',
 #'KEGG_2013',
 #'KEGG_2015',
 'KEGG_2016',
 #'KEGG_2019_Human',
 'KEGG_2019_Mouse',
 'KEGG_2021_Human',
 'KOMP2_Mouse_Phenotypes_2022',
 'Kinase_Perturbations_from_GEO_down',
 'Kinase_Perturbations_from_GEO_up',
 'L1000_Kinase_and_GPCR_Perturbations_down',
 'L1000_Kinase_and_GPCR_Perturbations_up',
 'LINCS_L1000_CRISPR_KO_Consensus_Sigs',
 'LINCS_L1000_Chem_Pert_Consensus_Sigs',
 'LINCS_L1000_Chem_Pert_down',
 'LINCS_L1000_Chem_Pert_up',
 'LINCS_L1000_Ligand_Perturbations_down',
 'LINCS_L1000_Ligand_Perturbations_up',
 'Ligand_Perturbations_from_GEO_down',
 'Ligand_Perturbations_from_GEO_up',
 'MAGMA_Drugs_and_Diseases',
 'MAGNET_2023',
 'MCF7_Perturbations_from_GEO_down',
 'MCF7_Perturbations_from_GEO_up',
 #'MGI_Mammalian_Phenotype_Level_4_2021',
 'MGI_Mammalian_Phenotype_Level_4_2024',
 'MSigDB_Computational',
 'MSigDB_Hallmark_2020',
 'MSigDB_Oncogenic_Signatures',
 'Metabolomics_Workbench_Metabolites_2022',
 'Microbe_Perturbations_from_GEO_down',
 'Microbe_Perturbations_from_GEO_up',
 'MoTrPAC_2023',
 'Mouse_Gene_Atlas',
 'NCI-60_Cancer_Cell_Lines',
 'NCI-Nature_2016',
 'NIBR_DRUGseq_2025_down',
 'NIBR_DRUGseq_2025_up',
 'NURSA_Human_Endogenous_Complexome',
 'OMIM_Disease',
 'OMIM_Expanded',
 'Old_CMAP_down',
 'Old_CMAP_up',
 'Orphanet_Augmented_2021',
 'PFOCR_Pathways_2023',
 'PPI_Hub_Proteins',
 'PanglaoDB_Augmented_2021',
 #'Panther_2015',
 'Panther_2016',
 'PerturbAtlas',
 'PerturbAtlas_MouseGenePerturbationSigs',
 'PerturbSeq_ReplogleK562',
 'PerturbSeq_ReplogleRPE1',
 'Pfam_Domains_2019',
 'Pfam_InterPro_Domains',
 'PheWeb_2019',
 'PhenGenI_Association_2021',
 'Phosphatase_Substrates_from_DEPOD',
 'ProteomicsDB_2020',
 'Proteomics_Drug_Atlas_2023',
 'RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO',
 'Rare_Diseases_AutoRIF_ARCHS4_Predictions',
 'Rare_Diseases_AutoRIF_Gene_Lists',
 'Rare_Diseases_GeneRIF_ARCHS4_Predictions',
 'Rare_Diseases_GeneRIF_Gene_Lists',
 'Reactome_2022',
 'Reactome_Pathways_2024',
 'RummaGEO_DrugPerturbations_2025',
 'RummaGEO_GenePerturbations_2025',
 'Rummagene_kinases',
 'Rummagene_signatures',
 'Rummagene_transcription_factors',
 'SILAC_Phosphoproteomics',
 'Sciplex_Drug_Perturbation_Signatures_2025',
 'SubCell_BarCode',
 'SynGO_2022',
 'SynGO_2024',
 'SysMyo_Muscle_Gene_Sets',
 'TF-LOF_Expression_from_GEO',
 'TF_Perturbations_Followed_by_Expression',
 'TG_GATES_2020',
 'TISSUES_Curated_2025',
 'TISSUES_Experimental_2025',
 'TRANSFAC_and_JASPAR_PWMs',
 'TRRUST_Transcription_Factors_2019',
 'Table_Mining_of_CRISPR_Studies',
 'Tabula_Muris',
 'Tabula_Sapiens',
 'TargetScan_microRNA',
 'TargetScan_microRNA_2017',
 #'The_Kinase_Library_2023',
 'The_Kinase_Library_2024',
 'Tissue_Protein_Expression_from_Human_Proteome_Map',
 'Tissue_Protein_Expression_from_ProteomicsDB',
 'Transcription_Factor_PPIs',
 'UK_Biobank_GWAS_v1',
 'Virus-Host_PPI_P-HIPSTer_2020',
 'VirusMINT',
 'Virus_Perturbations_from_GEO_down',
 'Virus_Perturbations_from_GEO_up',
 #'WikiPathway_2021_Human',
 #'WikiPathway_2023_Human',
 #'WikiPathways_2013',
 #'WikiPathways_2015',
 'WikiPathways_2016',
 #'WikiPathways_2019_Human',
 #'WikiPathways_2019_Mouse',
 'WikiPathways_2024_Human',
 'WikiPathways_2024_Mouse',
 'dbGaP',
 'huMAP',
 'lncHUB_lncRNA_Co-Expression',
 'miRTarBase_2017'
]
#!/usr/bin/env python3
"""script doc string."""
# /home/ubuntu/projects/gitbenlewis/adata_science_tools/example_PMID_33969320/scripts/py0_download_and_parse_input_files_PMID_33969320.py

import sys
import os
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import logging
import shutil
import urllib.request
import yaml

 # CFG Configuration
####################################
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIG_YAML_PATH = REPO_ROOT / "config" / "config.yaml"
with REPO_CONFIG_YAML_PATH.open() as f:
    CFG = yaml.safe_load(f)

# out and log path
DOWNLOAD_AND_PARSE_OLINK_PARAMS = CFG["py0_download_and_parse_input_files_PMID_33969320_params"]
DOWNLOAD_PARAMS= DOWNLOAD_AND_PARSE_OLINK_PARAMS["download_input_files_params"]
PARSE_OLINK_PARAMS= DOWNLOAD_AND_PARSE_OLINK_PARAMS["parse_olink_xlsx_file_2_adata_params"]
PARSE_SOMALOGIC_PARAMS= DOWNLOAD_AND_PARSE_OLINK_PARAMS["parse_somalogic_xlsx_file_2_adata_params"]
DOWNLOAD_DIR = Path(DOWNLOAD_AND_PARSE_OLINK_PARAMS["download_input_files_params"]["repo_or_project_dir"])
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = DOWNLOAD_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------- Configuration ----------------------------------------------------

#### start #### log file setup
# ---------- logging setup ----------
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_BASE_NAME = Path(__file__).stem
LOG_FILENAME = f"{SCRIPT_BASE_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"
RESULTS_LOG_FILE = LOG_DIR / LOG_FILENAME
SCRIPT_LOG_DIR = Path(__file__).resolve().parent / "logs"
SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_LOG_FILE = SCRIPT_LOG_DIR / LOG_FILENAME
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_LOG_FILE),
        logging.FileHandler(SCRIPT_LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Logging to %s", RESULTS_LOG_FILE)
LOGGER.info("Logging to %s", SCRIPT_LOG_FILE)
logging.captureWarnings(True)
logging.getLogger("py.warnings").propagate = True
# ---------- logging setup ----------

# dataclass G() ---------------------------------------------------------------------
# save all the global variable in a dataclass G
from dataclasses import dataclass
@dataclass
class G():
    '''Class to hold global variables'''
    WRITE_DIR='/home/ubuntu/write/'
    GITBENLEWIS_REPO_PARENT_DIR='/home/ubuntu/projects/gitbenlewis/'
    SCRIPTS_DIR='../scripts/'
    CONFIG_DIR='../config/'
    RESULTS_DIR='../results/'
    WRITE_CACHE=False
    ##### input files   
    
    # control variables to change
    SAVE_OUTPUT=True
    SAVE_OUTPUT_FIGURES=True
# ------------- dataclass G()  --------------------------------------------------------

########## import custom code libraries ################################################
import sys
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
# Use the current working directory instead of __file__
#REPO_ROOT = Path(os.getcwd()).resolve().parent.parent
print(f"REPO_ROOT set to: {str(REPO_ROOT)}")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from code_library import adata_science_tools as adtl
print(f"Using adtl from {adtl.__file__}")
from code_library import gseapy_pre_rank_wrap as gprw
print(f"Using gprw from {gprw.__file__}")
from code_library import gseapy_dot_plots as gdp
print(f"Using gdp from {gdp.__file__}")

########################################################## import custom code libraries ################################################
 
# ----------------- Configuration ----------------------------------------------------


# download from a public source
# Longitudinal proteomic analysis of severe COVID-19 reveals survival-associated signatures, tissue-specific cell death, and cell-cell interactions


def download_file(url: str, destination: Path, overwrite: bool = True) -> None:
    import urllib.request
    import shutil
    """Download a file from URL to destination, overwriting if it exists."""
    if destination.exists() and not overwrite:
        LOGGER.info("File already exists and overwrite is False: %s", destination)
        return
    if destination.exists():
        LOGGER.info("Overwriting existing file: %s", destination)
    LOGGER.info("Downloading %s -> %s", url, destination)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    LOGGER.info("Downloaded %s (%d bytes)", destination.name, destination.stat().st_size)


def download_input_files(overwrite: bool = True) -> None:
    """Download patient metadata and Olink data from configured URLs."""
    patient_url = DOWNLOAD_PARAMS.get("observation_metadata_url","https://data.mendeley.com/public-files/datasets/nf853r8xsj/files/e81b51ed-8f29-4f2b-bccc-5b3364341e92/file_downloaded")
    patient_filename = DOWNLOAD_PARAMS.get("observation_metadata_filename","Clinical_Metadata.xlsx")
    olink_url = DOWNLOAD_PARAMS.get("olink_data_url","https://data.mendeley.com/public-files/datasets/nf853r8xsj/files/430eabd8-9b62-4e5f-b7c4-75ac26567c2f/file_downloaded")
    olink_filename = DOWNLOAD_PARAMS.get("olink_data_filename","Olink_Proteomics.xlsx")
    olink_var_metadata_url = DOWNLOAD_PARAMS.get("olink_var_metadata_url","https://data.mendeley.com/public-files/datasets/nf853r8xsj/files/5cf61188-4534-4e86-9114-72a833bfeddd/file_downloaded")
    olink_var_metadata_filename = DOWNLOAD_PARAMS.get("olink_var_metadata_filename","Supplemental-Table-2-Olink-Assays-NPX-values_v2.xlsx")
    somalogic_data_url = DOWNLOAD_PARAMS.get("somalogic_data_url","https://data.mendeley.com/public-files/datasets/nf853r8xsj/files/1bc0ffe4-13c1-4809-927d-07dcffe11c15/file_downloaded")
    somalogic_data_filename = DOWNLOAD_PARAMS.get("somalogic_data_filename","Somalogic_Proteomics.xlsx")
    download_file(patient_url, DOWNLOAD_DIR / patient_filename, overwrite=overwrite)
    download_file(olink_url, DOWNLOAD_DIR / olink_filename, overwrite=overwrite)
    download_file(olink_var_metadata_url, DOWNLOAD_DIR / olink_var_metadata_filename, overwrite=overwrite)
    download_file(somalogic_data_url, DOWNLOAD_DIR / somalogic_data_filename, overwrite=overwrite)


def map_uniprot_ids_to_gene_names(uniprot_ids, logger: logging.Logger | None = None) -> list[str]:
    """Map UniProt IDs to gene names using UniProt ID mapping API.

    Rules:
    - If a row has multiple UniProt IDs (comma-separated), try them in order.
    - If no mapping is found, fall back to the UniProt ID for that row.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    def split_candidates(raw_id: str):
        if raw_id is None:
            return []
        text = str(raw_id).strip()
        if not text:
            return []
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        parts = [part.strip() for part in text.split(",")]
        return [part for part in parts if part]

    def normalize_id(candidate: str):
        """Return candidate plus normalized variants (strip .1 or -2 suffixes)."""
        variants = [candidate]
        if "." in candidate:
            variants.append(candidate.split(".", 1)[0])
        if "-" in candidate:
            variants.append(candidate.split("-", 1)[0])
        # preserve order, remove duplicates
        seen = set()
        ordered = []
        for item in variants:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def submit_job(ids: list[str]) -> str | None:
        import json
        from urllib.parse import urlencode

        data = urlencode({"from": "UniProtKB_AC-ID", "to": "Gene_Name", "ids": ",".join(ids)}).encode("utf-8")
        request = urllib.request.Request(
            "https://rest.uniprot.org/idmapping/run",
            data=data,
            method="POST",
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        return payload.get("jobId")

    def wait_for_job(job_id: str, timeout_s: float = 300.0, sleep_s: float = 3.0) -> None:
        import json
        import time

        status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
        start = time.time()
        while True:
            with urllib.request.urlopen(status_url) as response:
                status = json.load(response)
            if "results" in status or "failedIds" in status or status.get("jobStatus") == "FINISHED":
                return
            if status.get("jobStatus") == "FAILED":
                raise RuntimeError(f"UniProt ID mapping failed: {status}")
            if time.time() - start > timeout_s:
                raise TimeoutError("Timed out waiting for UniProt ID mapping job")
            time.sleep(sleep_s)

    def fetch_all_results(job_id: str) -> list[dict]:
        """Fetch all results for a job via UniProt stream endpoint (no paging loss)."""
        stream_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}?format=tsv"
        request = urllib.request.Request(
            stream_url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept": "text/tab-separated-values",
            },
        )
        with urllib.request.urlopen(request) as response:
            text = response.read().decode("utf-8", errors="replace")

        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []

        header = lines[0].split("\t")
        from_idx = header.index("From") if "From" in header else 0
        to_idx = header.index("To") if "To" in header else 1
        results: list[dict] = []
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) <= max(from_idx, to_idx):
                continue
            results.append({"from": cols[from_idx], "to": cols[to_idx]})
        return results

    raw_ids = [str(x) for x in uniprot_ids]
    candidates_per_row = [split_candidates(raw_id) for raw_id in raw_ids]
    unique_ids = sorted({candidate for candidates in candidates_per_row for candidate in candidates})

    mapping: dict[str, str] = {}
    if unique_ids:
        try:
            # Chunk IDs to avoid missing large results
            chunk_size = 500
            for i in range(0, len(unique_ids), chunk_size):
                chunk = unique_ids[i : i + chunk_size]
                job_id = submit_job(chunk)
                if not job_id:
                    logger.warning("UniProt ID mapping did not return a job ID for chunk %d-%d.", i, i + chunk_size)
                    continue
                wait_for_job(job_id)
                results = fetch_all_results(job_id)
                for entry in results:
                    from_id = entry.get("from")
                    to_id = entry.get("to")
                    if not from_id or to_id is None:
                        continue
                    if isinstance(to_id, list):
                        if to_id:
                            mapping[from_id] = to_id[0]
                    else:
                        mapping[from_id] = to_id
        except Exception as exc:
            logger.warning("UniProt ID mapping failed; falling back to UniProt IDs. Error: %s", exc)
            mapping = {}

    gene_names: list[str] = []
    mapped_count = 0
    for raw_id, candidates in zip(raw_ids, candidates_per_row, strict=True):
        if not candidates:
            gene_names.append(raw_id)
            continue
        resolved = None
        for candidate in candidates:
            for variant in normalize_id(candidate):
                mapped = mapping.get(variant)
                if mapped:
                    resolved = mapped
                    break
            if resolved:
                break
        if resolved is not None:
            mapped_count += 1
            gene_names.append(resolved)
        else:
            gene_names.append(candidates[0])
    logger.info("UniProt mapping resolved %d/%d IDs", mapped_count, len(gene_names))
    return gene_names


def parse_olink_xlsx_file_2_adata(  logger: logging.Logger):
    ''' 
    '''
    import pandas as pd
    import anndata
    import pathlib 
    from pathlib import Path
    # 1a) read the var metadata files 
    VAR_METADATA_DF = pd.read_excel(
        PARSE_OLINK_PARAMS.get("olink_var_data_xlsx_path"),
        sheet_name= PARSE_OLINK_PARAMS.get("olink_data_adata_var_sheet_name_1","2A-Olink-Assay"),
        header=1,
        dtype=str)
    # set index to "OlinkID" with drop=False to keep it as a column as well
    VAR_METADATA_DF.set_index("OlinkID", inplace=True, drop=False)
    logger.info(f"VAR_METADATA_DF read with shape {VAR_METADATA_DF.shape} and index set to OlinkID")
    # 1b) read the 2nd var metadata files 
    VAR_METADATA_2_DF = pd.read_excel(
        PARSE_OLINK_PARAMS.get("olink_var_data_xlsx_path"),
        sheet_name= PARSE_OLINK_PARAMS.get("olink_data_adata_var_sheet_name_2","2B Olink proteins"),
        header=1,
        dtype=str)
    logger.info(f"VAR_METADATA_2_DF read with shape {VAR_METADATA_2_DF.shape}") 
    # 2) read the obs metadata file
    OBS_METADATA_DF = pd.read_excel(
        PARSE_OLINK_PARAMS.get("olink_obs_data_xlsx_path"),
        sheet_name= PARSE_OLINK_PARAMS.get("olink_obs_data_xlsx_sheet_name","Subject-level metadata"),
        #header=1,
        index_col=0,
        dtype=str)
    # set index to str
    OBS_METADATA_DF.index = OBS_METADATA_DF.index.astype(str)
    logger.info(f"OBS_METADATA_DF read with shape {OBS_METADATA_DF.shape}")
    # 3) read the data matrix file
    X_DF = pd.read_excel(
        PARSE_OLINK_PARAMS.get("olink_X_data_xlsx_path"),
        sheet_name= PARSE_OLINK_PARAMS.get("olink_data_matrix_xlsx_sheet_name","Olink Proteomics"),
        header=0,
        index_col=0,
        )
    import_obs_df=X_DF[['Day']].copy()
    import_obs_df['Day'] = import_obs_df['Day'].astype(str)
    # drop the "Day" column from X_DF since it's not part of the assay data
    X_DF.drop(columns=['Day'], inplace=True)
    # rename index to obs_names
    X_DF.index.name = 'obs_names'
    logger.info(f"Data matrix X_DF read with shape {X_DF.shape} ")
    # 4) make anndata object with assay data only, no metadata yet
    adata = anndata.AnnData(X=X_DF, obs=import_obs_df)
    # name var index to var_names and obs index to obs_names
    adata.var_names.name = 'var_names'
    adata.obs_names.name = 'obs_names'
    logger.info(f"adata created with shape {adata.shape} and obs_names and var_names set")
    # 5) parse obs_names to generate a adata.obs['Public ID'] column
    # use this to merge with the obs metadata later
    adata.obs['Public ID'] = adata.obs_names.str.split('_').str[0].astype(str)
    # strip leading 0s from Public ID to match the format in the obs metadata sheet
    adata.obs['Public ID'] = adata.obs['Public ID'].str.lstrip('0')
    # 6) merge the obs metadata with the adata.obs dataframe using the Public ID column
    adata.obs = adata.obs.merge(OBS_METADATA_DF,
                                left_on='Public ID',
                                right_index=True,
                                #right_on='Public ID',# left_index=True,
                                how='left')
    logger.info(f"adata.obs merged with OBS_METADATA_DF, resulting shape: {adata.obs.shape}")   
    # 7) merge the var metadata from VAR_METADATA_DF with the adata.var dataframe using the 'OlinkID'
    VAR_METADATA_DF=VAR_METADATA_DF[PARSE_OLINK_PARAMS.get("olink_data_adata_var_sheet_name_1_columns_to_use")].copy()
    adata.var=adata.var.merge(VAR_METADATA_DF ,
    left_index=True,
    right_index=True,
    how='left')

    logger.info(f"adata.var merged with VAR_METADATA_DF, resulting shape: {adata.var.shape}")
    # 8) merge the var metadata from VAR_METADATA_DF_2 with the adata.var dataframe using the 'Uniprot' and 'UniProt' columns
    # first set the index of VAR_METADATA_DF_2 to 'Uniprot'
    adata.var=adata.var.merge(VAR_METADATA_2_DF.drop_duplicates(subset=["Uniprot"]).set_index("Uniprot"),
                                left_on='UniProt',
                                right_index=True,
                                how='left',
                                 validate="m:1",
                                )
    logger.info(f"adata.var merged with VAR_METADATA_DF_2, resulting shape: {adata.var.shape}")
    logger.info(f"final adata : {adata}")
    return adata

def parse_somalogic_xlsx_file_2_adata(  logger: logging.Logger):
    ''' 
    '''
    import pandas as pd
    import anndata
    import pathlib 
    from pathlib import Path
    # 1a) read the var metadata files 
    # 2) read the obs metadata file
    OBS_METADATA_DF = pd.read_excel(
        PARSE_SOMALOGIC_PARAMS.get("somalogic_obs_data_xlsx_path"),
        sheet_name= PARSE_SOMALOGIC_PARAMS.get("somalogic_data_adata_obs_sheet_name","Subject-level metadata"),
        #header=1,
        index_col=0,
        dtype=str)
    # set index to str
    OBS_METADATA_DF.index = OBS_METADATA_DF.index.astype(str)
    #display(OBS_METADATA_DF.head())
    logger.info(f"OBS_METADATA_DF read with shape {OBS_METADATA_DF.shape}")
    # 3) read the data matrix file
    X_DF = pd.read_excel(
        PARSE_SOMALOGIC_PARAMS.get("somalogic_X_data_xlsx_path"),
        sheet_name= PARSE_SOMALOGIC_PARAMS.get("somalogic_data_adata_X_sheet_name","Sheet1"),
        header=0,
       # index_col=0,
        )
    # add letter 'D' to front of day values
    X_DF['day'] = 'D' + X_DF['day'].astype(str)
    # merge the 'Public ID' with 'day' column to create a unique identifier for merging with the obs metadata later
    X_DF['obs_names'] = X_DF['Public'].astype(str) + '_' + X_DF['day'].astype(str)
    X_DF.set_index('obs_names', inplace=True)
    #display(X_DF.head())
    import_obs_df=X_DF[['Public','day','sample_barcode']].copy()
    import_obs_df.columns=['Public ID','Day','sample barcode']
    # drop the "day" column from X_DF since it's not part of the assay data
    X_DF.drop(columns=['Public','day','sample_barcode'], inplace=True)
    # rename index to obs_names
    #X_DF.index.name = 'obs_names'
    logger.info(f"Data matrix X_DF read with shape {X_DF.shape} ")
    # 4) make anndata object with assay data only, no metadata yet
    adata = anndata.AnnData(X=X_DF, obs=import_obs_df)
    # name var index to var_names and obs index to obs_names
    adata.var_names.name = 'var_names'
    adata.obs_names.name = 'obs_names'
    logger.info(f"adata created with shape {adata.shape} and obs_names and var_names set")
    # 5) parse obs_names to generate a adata.obs['Public ID'] column
    # use this to merge with the obs metadata later
    adata.obs['Public ID'] = adata.obs_names.str.split('_').str[0].astype(str)
    # strip leading 0s from Public ID to match the format in the obs metadata sheet
    adata.obs['Public ID'] = adata.obs['Public ID'].str.lstrip('0')
    # 6) merge the obs metadata with the adata.obs dataframe using the Public ID column
    adata.obs = adata.obs.merge(OBS_METADATA_DF,
                                left_on='Public ID',
                                right_index=True,
                                #right_on='Public ID',# left_index=True,
                                how='left')
    logger.info(f"adata.obs merged with OBS_METADATA_DF, resulting shape: {adata.obs.shape}")   
 
    #display(adata.obs.head())

    logger.info(f"final adata : {adata}")
    return adata


def save_adata_dataset(
    _adata,
    output_path,
    save_X_obs_var_csv=True,
    save_layers_as_csv=False,
    logger: logging.Logger | None = None,
):
    """Save dataset helper function."""
    if logger is None:
        logger = logging.getLogger(__name__)

    H5AD_PATH = Path(f"{output_path}.h5ad")
    logger.info(f"Saving adata to {H5AD_PATH}")
    _adata.write_h5ad(H5AD_PATH)

    if save_X_obs_var_csv:
        OBS_CSV_PATH = Path(f"{output_path}.obs.csv")
        VAR_CSV_PATH = Path(f"{output_path}.var.csv")
        X_CSV_PATH = Path(f"{output_path}.X.csv")
        logger.info(f"Saving adata.obs to {OBS_CSV_PATH}")
        _adata.obs.to_csv(OBS_CSV_PATH)
        logger.info(f"Saving adata.var to {VAR_CSV_PATH}")
        _adata.var.to_csv(VAR_CSV_PATH)
        logger.info(f"Saving adata.X to {X_CSV_PATH}")
        pd.DataFrame(_adata.X, index=_adata.obs_names, columns=_adata.var_names).to_csv(X_CSV_PATH)

    if save_layers_as_csv:
        for layer_name, layer_data in _adata.layers.items():
            safe_name = str(layer_name).replace("/", "_")
            layer_csv_path = Path(f"{output_path}.layer.{safe_name}.csv")
            logger.info(f"Saving adata.layers['{layer_name}'] to {layer_csv_path}")
            if hasattr(layer_data, "toarray"):
                layer_data = layer_data.toarray()
            pd.DataFrame(layer_data, index=_adata.obs_names, columns=_adata.var_names).to_csv(layer_csv_path)

    output_dir = H5AD_PATH.parent
    logger.info(f"Saved dataset to directory: {output_dir}")
    basename = H5AD_PATH.stem
    logger.info(f"Saved dataset with base filename: {basename}")
if __name__ == "__main__":
    #### do the downloads
    LOGGER.info("Starting download of input files...")
    download_input_files(overwrite=DOWNLOAD_PARAMS.get("overwrite_existing_files", False))
    LOGGER.info("Finished downloading input files.")
    LOGGER.info("Starting parsing of Olink XLSX file to anndata...")
    adata=parse_olink_xlsx_file_2_adata(logger=LOGGER)
    if PARSE_OLINK_PARAMS.get("run_uniprot_2_gene_name_mapping", False):
        LOGGER.info("Mapping UniProt IDs to gene names for olink dataset...")
        adata.var["gene_name"] = map_uniprot_ids_to_gene_names(adata.var['UniProt'], logger=LOGGER)
        LOGGER.info("Added gene_name column to adata.var")
        if PARSE_OLINK_PARAMS.get("save_uniprot_2_gene_name_mapping", False):
            mapping_df = pd.DataFrame({
                "var_name": adata.var_names,
                'UniProt': adata.var['UniProt'],
                "gene_name": adata.var["gene_name"]
            })
            mapping_output_path = PARSE_OLINK_PARAMS.get("uniprot_2_gene_name_mapping_output_path", OUTPUT_DIR / "olink_uniprot_to_gene_name_mapping.csv")
            LOGGER.info(f"Saving UniProt to gene name mapping to {mapping_output_path}")
            mapping_df.to_csv(mapping_output_path, index=False)
    LOGGER.info(f"adata \n{adata}")
    LOGGER.info("Finished parsing Olink XLSX file to anndata.")
    LOGGER.info("Saving adata to %s", PARSE_OLINK_PARAMS.get("adata_parsed_data_output_path"))
    save_adata_dataset(adata, 
                      output_path=PARSE_OLINK_PARAMS.get("adata_parsed_data_output_path"),
                      save_X_obs_var_csv=PARSE_OLINK_PARAMS.get("save_X_obs_var_csv", True),
                      save_layers_as_csv=PARSE_OLINK_PARAMS.get("save_layers_as_csv", False),
                      logger=LOGGER)
    LOGGER.info("Finished saving olink adata dataset.")

    LOGGER.info("Starting parsing of Somalogic XLSX file to anndata...")
    adata_soma=parse_somalogic_xlsx_file_2_adata(logger=LOGGER)
    
    if PARSE_SOMALOGIC_PARAMS.get("run_uniprot_2_gene_name_mapping", False):
        LOGGER.info("Mapping UniProt IDs to gene names for somalogic dataset...")
        adata_soma.var["gene_name"] = map_uniprot_ids_to_gene_names(adata_soma.var_names, logger=LOGGER)
        LOGGER.info("Added gene_name column to adata_soma.var")
        if PARSE_SOMALOGIC_PARAMS.get("save_uniprot_2_gene_name_mapping", False):
            mapping_df = pd.DataFrame({
                "var_name": adata_soma.var_names,
                "gene_name": adata_soma.var["gene_name"]
            })
            mapping_output_path = PARSE_SOMALOGIC_PARAMS.get("uniprot_2_gene_name_mapping_output_path", OUTPUT_DIR / "somalogic_uniprot_to_gene_name_mapping.csv")
            LOGGER.info(f"Saving UniProt to gene name mapping to {mapping_output_path}")
            mapping_df.to_csv(mapping_output_path, index=False)
    # 7) merge the var metadata if merge_var_metadata_from_csv is True
    if PARSE_SOMALOGIC_PARAMS.get("merge_var_metadata_from_csv", False):
        LOGGER.info("Merging var metadata from CSV: %s", PARSE_SOMALOGIC_PARAMS.get("somalogic_var_data_csv_path"))
        VAR_METADATA_DF = pd.read_csv(PARSE_SOMALOGIC_PARAMS.get("somalogic_var_data_csv_path"))
        VAR_METADATA_DF.set_index("var_name", inplace=True)
        adata.var=adata.var.merge(VAR_METADATA_DF ,
        left_index=True,
        right_index=True,
        how='left')
        LOGGER.info(f"adata.var merged with VAR_METADATA_DF, resulting shape: {adata.var.shape}")
        LOGGER.info("Finished merging var metadata from CSV.")
    LOGGER.info(f"adata_soma \n{adata_soma}")
    LOGGER.info("Finished parsing Somalogic XLSX file to anndata.")
    LOGGER.info("Saving adata_soma to %s", PARSE_SOMALOGIC_PARAMS.get("adata_parsed_data_output_path"))
    save_adata_dataset(adata_soma, 
                      output_path=PARSE_SOMALOGIC_PARAMS.get("adata_parsed_data_output_path"),
                      save_X_obs_var_csv=PARSE_SOMALOGIC_PARAMS.get("save_X_obs_var_csv", True),
                      save_layers_as_csv=PARSE_SOMALOGIC_PARAMS.get("save_layers_as_csv", False),
                      logger=LOGGER)
    LOGGER.info("Finished saving somalogic adata dataset.")

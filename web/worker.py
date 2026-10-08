"""One supervisor with bounded, separate processes for imports and analyses."""

import json
import os
import signal
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from .store import Store


def execute_job(root, job_id, max_dense_bytes, max_import_bytes):
    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from .analysis import run_analysis
    from .data import check_h5ad, demo_dataset, load_csv_bundle, metadata, read_csv, sha256, table_preview, validate_adata

    store = Store(root)
    job = store.job(job_id)
    request = json.loads(job["request"])
    directory = store.directory(job["dataset_id"])
    output_dir = directory / "jobs" / job_id
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        if request["kind"] == "import":
            format_name = request["format"]
            inputs = {}
            if format_name == "h5ad":
                source = directory / "upload.h5ad"
                inputs["h5ad"] = sha256(source)
                check_h5ad(source, max_import_bytes)
                data = ad.read_h5ad(source)
            elif format_name == "csv":
                inputs = {name: sha256(directory / name) for name in ("X.csv", "obs.csv", "var.csv")}
                data = load_csv_bundle(directory)
            elif format_name == "covid":
                from .examples import COVID_DIRECTORY
                source = COVID_DIRECTORY / "covid_proteomics.h5ad"
                inputs["h5ad"] = sha256(source)
                check_h5ad(source, max_import_bytes)
                data = ad.read_h5ad(source)
            else:
                data = demo_dataset()
            validate_adata(data)
            if data.n_vars > 50000 or data.n_obs > 1000000:
                raise ValueError("This service supports up to 50,000 features and 1,000,000 observations per dataset.")
            info = metadata(data)
            info["input_sha256"] = inputs
            info["format"] = format_name
            if format_name == "h5ad":
                source.replace(directory / "data.h5ad")
            else:
                data.write_h5ad(directory / "data.h5ad", compression="gzip")
                for name in ("X.csv", "obs.csv", "var.csv"):
                    (directory / name).unlink(missing_ok=True)
            info["dataset_sha256"] = sha256(directory / "data.h5ad")
            with store.connect() as db:
                db.execute("UPDATE datasets SET status='ready',metadata=? WHERE id=?",
                           (json.dumps(info, allow_nan=False), job["dataset_id"]))
            store.finish(job_id, {"summary": "Dataset loaded and validated."})
            return

        params = request["parameters"]
        data = ad.read_h5ad(directory / "data.h5ad")
        table = None
        if params.get("source_job"):
            source = store.job(params["source_job"])
            if not source or source["dataset_id"] != job["dataset_id"] or source["status"] != "complete":
                raise ValueError("The result source is not available for this dataset.")
            table_path = directory / "jobs" / source["id"] / "results.csv"
            table = read_csv(table_path)
            for col in table:
                numeric = pd.to_numeric(table[col], errors="coerce")
                if numeric.notna().sum() == table[col].notna().sum():
                    table[col] = numeric
        result = run_analysis(data, params, table, max_dense_bytes)
        files = []
        if result["figure"] is not None:
            for extension in ("png", "svg", "pdf"):
                filename = f"figure.{extension}"
                # Fix metadata timestamps; scientific jitter seeds remain deterministic.
                meta = {"Date": None} if extension == "svg" else {"CreationDate": None, "ModDate": None} if extension == "pdf" else {}
                with matplotlib.rc_context({"svg.hashsalt": "adata-science-tools"}):
                    result["figure"].savefig(output_dir / filename, dpi=140, bbox_inches="tight", metadata=meta)
                files.append(filename)
        preview = None
        columns = []
        if result["table"] is not None:
            result["table"].to_csv(output_dir / "results.csv")
            files.append("results.csv")
            preview = table_preview(result["table"], rows=30)
            columns = result["table"].columns.tolist()
        if result["selected"] is not None:
            result["selected"].write_h5ad(output_dir / "selection.h5ad", compression="gzip")
            files.append("selection.h5ad")
        source_info = json.loads(store.dataset(job["dataset_id"])["metadata"])
        source_run = json.loads(source["request"]) if params.get("source_job") else None
        if source_run is not None:
            source_params = source_run["parameters"]
            selection_keys = ("matrix", "filter_column", "filter_values", "numeric_columns", "categorical_columns")
            changed = [key for key in selection_keys if params.get(key) != source_params.get(key)]
            if params["operation"] == "effects" and source_params["operation"] == "diff_test":
                changed += [key for key in ("group", "reference", "target") if params.get(key) != source_params.get(key)]
            if changed:
                result["warnings"].append("Current selection differs from the statistical source run: " + ", ".join(changed) + ". Effects and p-values still come from the original run.")
        provenance = {"parameters": params, "function": result["function"], "function_kwargs": result["kwargs"],
                      "selection": result["summary"], "warnings": result["warnings"], "source_run": source_run,
                      "dataset_sha256": source_info["dataset_sha256"], "input_sha256": source_info["input_sha256"],
                      "versions": {name: version(name) for name in ("anndata", "numpy", "pandas", "scipy", "matplotlib", "statsmodels", "Flask")}}
        if request.get("pipeline"):
            provenance["pipeline"] = request["pipeline"]
        # Record the exact checkout when available; source archives need no Git installation.
        try:
            provenance["git_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
            provenance["git_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            provenance["git_revision"] = None
        (output_dir / "analysis.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
        (output_dir / "reproduce.py").write_text(
            '# Run from the same checkout/environment with your original dataset and analysis.json.\n'
            'import json\nimport anndata as ad\nfrom adata_science_tools.web.analysis import run_analysis\n'
            'spec = json.load(open("analysis.json"))\ndata = ad.read_h5ad("data.h5ad")\n'
            '# For CSV inputs, first use load_csv_bundle() on X.csv, obs.csv, var.csv.\n'
            'source = None\n'
            'if spec["source_run"] is not None:\n'
            '    source = run_analysis(data, spec["source_run"]["parameters"])["table"]\n'
            'result = run_analysis(data, spec["parameters"], result_table=source)\n'
            'if result["table"] is not None:\n    result["table"].to_csv("reproduced.csv")\n'
            'if result["figure"] is not None:\n    result["figure"].savefig("reproduced.png", dpi=140, bbox_inches="tight")\n',
            encoding="utf-8")
        files += ["analysis.json", "reproduce.py"]
        store.finish(job_id, {"files": files, "preview": preview, "columns": columns,
                              "summary": result["summary"], "warnings": result["warnings"]})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        error = str(exc)[:500] if isinstance(exc, (ValueError, KeyError, TypeError)) else "Analysis failed. Check the selected inputs; details are available in the server job log."
        store.finish(job_id, error=error)
        if request["kind"] == "import":
            with store.connect() as db:
                db.execute("UPDATE datasets SET status='failed' WHERE id=?", (job["dataset_id"],))
    finally:
        plt.close("all")


def serve_worker(config):
    # A single supervisor avoids simultaneous Matplotlib state and uncontrolled memory use.
    import fcntl
    store = Store(config["DATA_ROOT"])
    lock = open(store.root / "worker.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("An analysis worker already uses this data directory.") from exc
    with store.connect() as db:
        db.execute("UPDATE jobs SET status='failed',error='Worker restarted; submit this analysis again.' WHERE status='running'")
        db.execute("UPDATE datasets SET status='failed' WHERE status='queued' AND id IN "
                   "(SELECT dataset_id FROM jobs WHERE status='failed' AND json_extract(request,'$.kind')='import')")
    stop = False

    def terminate(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_web.py"
    last_cleanup = 0
    try:
        while not stop:
            if time.time() - last_cleanup > 60:
                store.cleanup(config["RETENTION_HOURS"])
                last_cleanup = time.time()
            (store.root / "worker.heartbeat").touch()
            job = store.claim()
            if not job:
                time.sleep(0.5)
                continue
            folder = store.directory(job["dataset_id"]) / "jobs" / job["id"]
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       ADTL_WEB_DATA_ROOT=str(store.root), ADTL_WEB_MAX_DENSE_BYTES=str(config["MAX_DENSE_BYTES"]),
                       ADTL_WEB_MAX_IMPORT_BYTES=str(config["MAX_IMPORT_BYTES"]))
            with open(folder / "worker.log", "w") as log:
                child = subprocess.Popen([sys.executable, str(launcher), "--execute-job", job["id"]],
                                         stdout=log, stderr=log, env=env)
                deadline = time.monotonic() + config["JOB_TIMEOUT_SECONDS"]
                while child.poll() is None and not stop and time.monotonic() < deadline:
                    (store.root / "worker.heartbeat").touch()
                    time.sleep(0.5)
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    store.finish(job["id"], error="Worker stopped or job exceeded the configured time limit. Reduce the selection and try again.")
                else:
                    result = store.job(job["id"])
                    if result and result["status"] == "running":
                        store.finish(job["id"], error="The analysis process exited before producing a result.")
                # A completed dataset can be deleted while its child process exits.
                result = store.job(job["id"])
                if result and result["status"] == "failed" and json.loads(job["request"])["kind"] == "import":
                    with store.connect() as db:
                        db.execute("UPDATE datasets SET status='failed' WHERE id=?", (job["dataset_id"],))
    finally:
        (store.root / "worker.heartbeat").unlink(missing_ok=True)
        lock.close()

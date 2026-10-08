"""HTTP interface, anonymous session isolation, and optional password login."""

import hashlib
import hmac
import json
import os
import secrets
import time
from datetime import timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from .store import Store
from .examples import COVID_PRESETS


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        DATA_ROOT=str(Path(__file__).resolve().parents[1] / "instance" / "web"),
        MAX_CONTENT_LENGTH=100 * 1024**2, MAX_FORM_MEMORY_SIZE=128 * 1024, MAX_FORM_PARTS=12,
        MAX_DENSE_BYTES=128 * 1024**2, MAX_IMPORT_BYTES=512 * 1024**2,
        MAX_DATASETS=3, MAX_TOTAL_DATASETS=30, MAX_JOBS=20, MAX_QUEUED_JOBS=30,
        JOB_TIMEOUT_SECONDS=300, RETENTION_HOURS=24, AUTH_REQUIRED=False, PUBLIC_MODE=False,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
    )
    app.config.from_prefixed_env(prefix="ADTL_WEB")
    app.config.update(config or {})
    store = Store(app.config["DATA_ROOT"])
    app.extensions["store"] = store
    if app.config["PUBLIC_MODE"]:
        if not app.config.get("SECRET_KEY") or len(app.config["SECRET_KEY"]) < 32:
            raise ValueError("Public mode requires ADTL_WEB_SECRET_KEY with at least 32 characters.")
        app.config["SESSION_COOKIE_SECURE"] = True
        if app.config["TRUSTED_HOSTS"] == ["localhost", "127.0.0.1", "[::1]"]:
            raise ValueError("Public mode requires ADTL_WEB_TRUSTED_HOSTS for the deployed domain.")
    elif not app.config.get("SECRET_KEY"):
        key_path = store.root / "session.key"
        try:
            with key_path.open("x") as handle:
                os.chmod(key_path, 0o600)
                handle.write(secrets.token_hex(32))
        except FileExistsError:
            pass
        app.config["SECRET_KEY"] = key_path.read_text().strip()

    def owner():
        return "user:" + session["user_id"] if session.get("user_id") else "anon:" + session["anonymous_id"]

    def owned_dataset(dataset_id):
        data = store.dataset(dataset_id, owner())
        if data is None:
            abort(404)
        return data

    @app.before_request
    def protect_request():
        session.permanent = True
        if "anonymous_id" not in session:
            session["anonymous_id"] = secrets.token_hex(24)
            session["csrf"] = secrets.token_hex(32)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            token = request.headers.get("X-CSRF-Token") or request.form.get("csrf", "")
            if not hmac.compare_digest(str(token), session["csrf"]):
                abort(400, "The form expired. Reload the page and try again.")
        if app.config["AUTH_REQUIRED"] and not session.get("user_id") and request.endpoint not in {"login", "static", "health"}:
            if request.path.startswith("/api/"):
                abort(401, "Sign in to use the application.")
            return redirect(url_for("login"))

    @app.after_request
    def response_headers(response):
        response.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                                 "Referrer-Policy": "same-origin", "Cache-Control": "no-store",
                                 "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'"})
        if app.config["PUBLIC_MODE"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/"):
            return jsonify(error=error.description), error.code
        return render_template("error.html", message=error.description), error.code

    @app.errorhandler(ValueError)
    def bad_input(error):
        if request.path.startswith("/api/"):
            return jsonify(error=str(error)), 400
        return render_template("error.html", message=str(error)), 400

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        return render_template("index.html", username=session.get("username"),
                               public_mode=app.config["PUBLIC_MODE"], retention=app.config["RETENTION_HOURS"],
                               upload_mb=app.config["MAX_CONTENT_LENGTH"] // 1024**2)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        status = 200
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            identities = [hmac.new(app.secret_key.encode(), value.encode(), hashlib.sha256).hexdigest()
                          for value in ("ip:" + (request.remote_addr or "unknown"), "user:" + username)]
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("DELETE FROM login_attempts WHERE created<?", (time.time() - 900,))
                count = max(db.execute("SELECT count(*) FROM login_attempts WHERE identity=?", (identity,)).fetchone()[0]
                            for identity in identities)
                if count >= 5:
                    error, status = "Too many sign-in attempts. Try again in 15 minutes.", 429
                else:
                    db.executemany("INSERT INTO login_attempts VALUES(?,?)", [(identity, time.time()) for identity in identities])
                    user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            if not error:
                # A fixed dummy hash keeps unknown usernames on the password-hash path too.
                password_hash = user["password_hash"] if user else app.config["DUMMY_PASSWORD_HASH"]
                valid = len(password) <= 1024 and check_password_hash(password_hash, password)
                if user and valid:
                    session.clear()
                    session.update(user_id=user["id"], username=user["username"],
                                   anonymous_id=secrets.token_hex(24), csrf=secrets.token_hex(32))
                    with store.connect() as db:
                        db.executemany("DELETE FROM login_attempts WHERE identity=?", [(v,) for v in identities])
                    return redirect(url_for("index"))
                error, status = "Incorrect username or password.", 401
        return render_template("login.html", error=error, required=app.config["AUTH_REQUIRED"]), status

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login") if app.config["AUTH_REQUIRED"] else url_for("index"))

    app.config["DUMMY_PASSWORD_HASH"] = generate_password_hash(secrets.token_hex(32))

    @app.get("/api/state")
    def state():
        from .analysis import CATALOG
        from .pipelines import PIPELINES
        heartbeat = store.root / "worker.heartbeat"
        alive = heartbeat.exists() and time.time() - heartbeat.stat().st_mtime < 15
        datasets = [{k: d[k] for k in ("id", "name", "created", "status")} for d in store.datasets(owner())]
        return jsonify(datasets=datasets, catalog=CATALOG, pipelines=PIPELINES, covid_presets=COVID_PRESETS, worker_running=alive, username=session.get("username"))

    @app.post("/api/datasets")
    def upload():
        format_name = request.form.get("format")
        if format_name not in {"h5ad", "csv", "demo", "covid"}:
            raise ValueError("Choose H5AD, CSV bundle, or an example dataset.")
        required = {"h5ad": ["h5ad"], "csv": ["X", "obs", "var"], "demo": [], "covid": []}[format_name]
        if any(key not in request.files or not request.files[key].filename for key in required):
            raise ValueError("Upload all required files before loading the dataset.")
        name = request.form.get("name", "").strip() or ({"demo": "Synthetic paired study", "covid": "COVID proteomics · PMID 33969320"}.get(format_name, "Uploaded dataset"))
        dataset_id = store.create_dataset(owner(), name, app.config["MAX_DATASETS"], app.config["MAX_TOTAL_DATASETS"])
        try:
            for key in required:
                expected = ".h5ad" if key == "h5ad" else ".csv"
                if not request.files[key].filename.lower().endswith(expected):
                    raise ValueError(f"{key} must be a {expected} file.")
                filename = "upload.h5ad" if key == "h5ad" else key + ".csv"
                request.files[key].save(store.directory(dataset_id) / filename)
            job_id = store.enqueue(dataset_id, {"kind": "import", "format": format_name},
                                   app.config["MAX_JOBS"], app.config["MAX_QUEUED_JOBS"])
        except Exception:
            store.delete_dataset(dataset_id)
            raise
        return jsonify(dataset_id=dataset_id, job_id=job_id), 202

    @app.get("/api/datasets/<dataset_id>")
    def dataset_detail(dataset_id):
        data = owned_dataset(dataset_id)
        data["metadata"] = json.loads(data["metadata"])
        data.pop("owner")
        jobs = store.jobs(dataset_id)
        for job in jobs:
            job["request"] = json.loads(job["request"])
            job["result"] = json.loads(job["result"])
        return jsonify(dataset=data, jobs=jobs)

    @app.delete("/api/datasets/<dataset_id>")
    def delete_dataset(dataset_id):
        owned_dataset(dataset_id)
        store.delete_dataset(dataset_id)
        return {"deleted": True}

    @app.post("/api/datasets/<dataset_id>/jobs")
    def submit(dataset_id):
        from .analysis import validate_request
        data = owned_dataset(dataset_id)
        if data["status"] != "ready":
            raise ValueError("Wait for dataset validation to finish.")
        payload = validate_request(request.get_json())
        if payload.get("source_job"):
            source = store.job(payload["source_job"], owner())
            if not source or source["dataset_id"] != dataset_id or source["status"] != "complete":
                raise ValueError("Choose completed results from this dataset.")
            source_request = json.loads(source["request"])
            if source_request.get("parameters", {}).get("operation") not in {"diff_test", "ols", "mixedlm"}:
                raise ValueError("Choose differential-test or model results as the source.")
            if "results.csv" not in json.loads(source["result"]).get("files", []):
                raise ValueError("The selected run has no result table.")
        job_id = store.enqueue(dataset_id, {"kind": "analysis", "parameters": payload},
                               app.config["MAX_JOBS"], app.config["MAX_QUEUED_JOBS"])
        return jsonify(job_id=job_id), 202

    @app.post("/api/datasets/<dataset_id>/pipelines")
    def submit_pipeline(dataset_id):
        from .pipelines import PIPELINES, build_pipeline
        data = owned_dataset(dataset_id)
        if data["status"] != "ready":
            raise ValueError("Wait for dataset validation to finish.")
        payload = request.get_json()
        steps = build_pipeline(payload)
        job_ids = store.enqueue_pipeline(dataset_id, PIPELINES[payload["pipeline"]]["label"], steps,
                                         app.config["MAX_JOBS"], app.config["MAX_QUEUED_JOBS"])
        return jsonify(job_ids=job_ids), 202

    @app.get("/api/jobs/<job_id>/files/<filename>")
    def download(job_id, filename):
        job = store.job(job_id, owner())
        if not job or job["status"] != "complete":
            abort(404)
        result = json.loads(job["result"])
        if filename not in result.get("files", []):
            abort(404)
        path = store.directory(job["dataset_id"]) / "jobs" / job["id"] / filename
        return send_file(path, as_attachment=filename != "figure.png", download_name=filename)

    return app

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import importlib.util
if importlib.util.find_spec("flask") is None:
    raise unittest.SkipTest("Install config/requirements-web.txt to test the optional web app.")

from werkzeug.security import generate_password_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adata_science_tools.web import create_app
from adata_science_tools.web.worker import execute_job


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app({"TESTING": True, "DATA_ROOT": self.temp.name, "SECRET_KEY": "test-only-secret", "MAX_TOTAL_DATASETS": 10})
        self.client = self.app.test_client()
        self.client.get("/")
        self.store = self.app.extensions["store"]

    def tearDown(self):
        self.temp.cleanup()

    def headers(self, client=None):
        with (client or self.client).session_transaction() as session:
            return {"X-CSRF-Token": session["csrf"]}

    def demo(self):
        response = self.client.post("/api/datasets", data={"format": "demo"}, headers=self.headers())
        self.assertEqual(response.status_code, 202, response.json)
        job = self.store.claim()
        execute_job(self.temp.name, job["id"], self.app.config["MAX_DENSE_BYTES"], self.app.config["MAX_IMPORT_BYTES"])
        self.assertEqual(self.store.job(job["id"])["status"], "complete", self.store.job(job["id"])["error"])
        return response.json["dataset_id"]

    def test_upload_analysis_download_and_session_isolation(self):
        dataset_id = self.demo()
        response = self.client.post(f"/api/datasets/{dataset_id}/jobs", headers=self.headers(),
                                   json={"operation": "histogram", "features": ["feature_1"], "bins": 12})
        self.assertEqual(response.status_code, 202)
        job = self.store.claim()
        execute_job(self.temp.name, job["id"], self.app.config["MAX_DENSE_BYTES"], self.app.config["MAX_IMPORT_BYTES"])
        self.assertEqual(self.store.job(job["id"])["status"], "complete", self.store.job(job["id"])["error"])
        result = self.client.get(f"/api/jobs/{job['id']}/files/figure.png")
        self.assertEqual(result.status_code, 200)
        self.assertTrue(result.data.startswith(b"\x89PNG"))
        result.close()
        record = self.client.get(f"/api/jobs/{job['id']}/files/analysis.json")
        self.assertEqual(record.json["parameters"]["operation"], "histogram")
        record.close()
        stranger = self.app.test_client()
        stranger.get("/")
        self.assertEqual(stranger.get(f"/api/datasets/{dataset_id}").status_code, 404)
        self.assertEqual(stranger.get(f"/api/jobs/{job['id']}/files/figure.png").status_code, 404)
        self.assertEqual(stranger.delete(f"/api/datasets/{dataset_id}", headers=self.headers(stranger)).status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/files/worker.log").status_code, 404)

    def test_csv_h5ad_uploads_and_failed_import(self):
        files = {"format": "csv", "X": (io.BytesIO(b",a,b\ns1,1,2\ns2,3,4\n"), "adata.X.csv"),
                 "obs": (io.BytesIO(b",group\ns2,B\ns1,A\n"), "adata.obs.csv"),
                 "var": (io.BytesIO(b",label\nb,B\na,A\n"), "adata.var.csv")}
        response = self.client.post("/api/datasets", data=files, headers=self.headers())
        self.assertEqual(response.status_code, 202)
        job = self.store.claim()
        execute_job(self.temp.name, job["id"], 1024**2, 1024**2)
        self.assertEqual(self.store.job(job["id"])["status"], "complete")
        path = self.store.directory(response.json["dataset_id"]) / "data.h5ad"
        response = self.client.post("/api/datasets", data={"format": "h5ad", "h5ad": (io.BytesIO(path.read_bytes()), "dataset.h5ad")}, headers=self.headers())
        job = self.store.claim()
        execute_job(self.temp.name, job["id"], 1024**2, 1024**2)
        self.assertEqual(self.store.job(job["id"])["status"], "complete")
        response = self.client.post("/api/datasets", data={"format": "h5ad", "h5ad": (io.BytesIO(b"not hdf5"), "bad.h5ad")}, headers=self.headers())
        job = self.store.claim()
        execute_job(self.temp.name, job["id"], 1024**2, 1024**2)
        self.assertEqual(self.store.job(job["id"])["status"], "failed")
        self.assertEqual(self.store.dataset(response.json["dataset_id"])["status"], "failed")

    def test_pipelines_execute_and_record_each_step(self):
        dataset_id = self.demo()
        self.assertEqual(set(self.client.get("/api/state").json["pipelines"]), {"explore", "independent", "paired"})
        for pipeline in ("explore", "independent", "paired"):
            with self.subTest(pipeline=pipeline):
                payload = {"pipeline": pipeline, "features": ["feature_1", "feature_2"],
                           "matrix": "layer:log1p", "group": "condition"}
                if pipeline != "explore":
                    payload.update(reference="Reference", target="Treatment")
                if pipeline == "paired":
                    payload.update(pair="subject", test="ttest_rel")
                response = self.client.post(f"/api/datasets/{dataset_id}/pipelines", json=payload, headers=self.headers())
                self.assertEqual(response.status_code, 202, response.json)
                ids = response.json["job_ids"]
                self.assertEqual(len(ids), 3)
                run_ids = set()
                for step, job_id in enumerate(ids, 1):
                    job = self.store.claim()
                    self.assertEqual(job["id"], job_id)
                    execute_job(self.temp.name, job_id, 1024**2, 1024**2)
                    result = self.store.job(job_id)
                    self.assertEqual(result["status"], "complete", result["error"])
                    with self.client.get(f"/api/jobs/{job_id}/files/analysis.json") as record:
                        self.assertEqual(record.json["pipeline"]["step"], step)
                        self.assertEqual(record.json["selection"]["matrix"], "layer:log1p")
                        self.assertEqual(record.json["selection"]["n_vars"], 2)
                        run_ids.add(record.json["pipeline"]["id"])
                self.assertEqual(len(run_ids), 1)
        # A new page request retains all pipeline steps in the persisted dataset history.
        self.client.get("/")
        jobs = self.client.get(f"/api/datasets/{dataset_id}").json["jobs"]
        self.assertEqual(sum("pipeline" in job["request"] for job in jobs), 9)

    def test_pipeline_submission_is_atomic_and_owned(self):
        dataset_id = self.demo()
        url = f"/api/datasets/{dataset_id}/pipelines"
        payload = {"pipeline": "explore", "features": ["feature_1"], "group": "condition"}
        self.assertEqual(self.client.post(url, json=payload).status_code, 400)
        stranger = self.app.test_client()
        stranger.get("/")
        self.assertEqual(stranger.post(url, json=payload, headers=self.headers(stranger)).status_code, 404)
        original = len(self.store.jobs(dataset_id))
        for config_key, limit in (("MAX_JOBS", original + 2), ("MAX_QUEUED_JOBS", 2)):
            previous = self.app.config[config_key]
            self.app.config[config_key] = limit
            response = self.client.post(url, json=payload, headers=self.headers())
            self.assertEqual(response.status_code, 400, response.json)
            self.assertEqual(len(self.store.jobs(dataset_id)), original)
            self.app.config[config_key] = previous

    def test_auth_required_login_logout_and_throttle(self):
        with self.store.connect() as db:
            db.execute("INSERT INTO users VALUES(?,?,?)", ("user1", "scientist", generate_password_hash("long-test-password")))
        self.app.config["AUTH_REQUIRED"] = True
        self.assertEqual(self.client.get("/").status_code, 302)
        self.assertEqual(self.client.get("/api/state").status_code, 401)
        for _ in range(5):
            response = self.client.post("/login", data={"username": "unknown", "password": "incorrect"}, headers=self.headers())
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.post("/login", data={"username": "scientist", "password": "long-test-password"}, headers=self.headers()).status_code, 429)
        with self.store.connect() as db:
            db.execute("DELETE FROM login_attempts")
        response = self.client.post("/login", data={"username": "scientist", "password": "long-test-password"}, headers=self.headers())
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get("/api/state").status_code, 200)
        self.demo()
        self.client.post("/logout", headers=self.headers())
        self.assertEqual(self.client.get("/api/state").status_code, 401)

    def test_csrf_upload_limit_public_settings_and_cleanup(self):
        self.assertEqual(self.client.post("/api/datasets", data={"format": "demo"}).status_code, 400)
        self.app.config["MAX_CONTENT_LENGTH"] = 20
        response = self.client.post("/api/datasets", data={"format": "h5ad", "h5ad": (io.BytesIO(b"x" * 100), "data.h5ad")}, headers=self.headers())
        self.assertEqual(response.status_code, 413)
        self.app.config["MAX_CONTENT_LENGTH"] = 1024**2
        dataset_id = self.demo()
        with self.store.connect() as db:
            db.execute("UPDATE datasets SET created=0 WHERE id=?", (dataset_id,))
        self.store.cleanup(24)
        self.assertIsNone(self.store.dataset(dataset_id))
        self.assertFalse(self.store.directory(dataset_id).exists())
        with self.assertRaisesRegex(ValueError, "SECRET_KEY"):
            create_app({"PUBLIC_MODE": True, "DATA_ROOT": self.temp.name})
        with self.assertRaisesRegex(ValueError, "TRUSTED_HOSTS"):
            create_app({"PUBLIC_MODE": True, "DATA_ROOT": self.temp.name, "SECRET_KEY": "a" * 64})
        response = self.client.get("/")
        self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")


    def test_covid_downloads_and_import_are_available(self):
        from adata_science_tools.web.examples import COVID_DIRECTORY, COVID_PRESETS
        from adata_science_tools.web.data import sha256
        response = self.client.get("/")
        self.assertIn(b"Open COVID example", response.data)
        self.assertEqual(self.client.get("/api/state").json["covid_presets"], COVID_PRESETS)
        for filename in ("covid_proteomics.h5ad", "X.csv", "obs.csv", "var.csv", "histogram.png", "datapoints.png"):
            with self.client.get("/static/examples/covid_proteomics/" + filename) as response:
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, (COVID_DIRECTORY / filename).read_bytes())
        response = self.client.post("/api/datasets", data={"format": "covid"}, headers=self.headers())
        self.assertEqual(response.status_code, 202)
        job = self.store.claim()
        execute_job(self.temp.name, job["id"], self.app.config["MAX_DENSE_BYTES"], self.app.config["MAX_IMPORT_BYTES"])
        self.assertEqual(self.store.job(job["id"])["status"], "complete")
        dataset = self.client.get("/api/datasets/" + response.json["dataset_id"]).json["dataset"]
        self.assertEqual(dataset["metadata"]["format"], "covid")
        self.assertEqual(dataset["metadata"]["n_obs"], 784)
        self.assertEqual(dataset["metadata"]["n_vars"], 1429)
        self.assertEqual(dataset["metadata"]["input_sha256"]["h5ad"], sha256(COVID_DIRECTORY / "covid_proteomics.h5ad"))
        self.assertTrue((COVID_DIRECTORY / "covid_proteomics.h5ad").exists())


if __name__ == "__main__":
    unittest.main()

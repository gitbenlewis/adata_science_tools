#!/usr/bin/env python3
"""Exercise a running native or Docker web app using only Python's standard library."""

import argparse
import http.cookiejar
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class Client:
    def __init__(self, url, session=None):
        self.url = url.rstrip("/")
        self.jar = http.cookiejar.MozillaCookieJar(str(session) if session else None)
        if session and session.exists():
            self.jar.load(ignore_discard=True, ignore_expires=True)
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.csrf = ""

    def request(self, path, data=None, method=None, content_type=None):
        headers = {"X-CSRF-Token": self.csrf}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        with self.opener.open(urllib.request.Request(self.url + path, data=data, method=method, headers=headers), timeout=30) as response:
            return response.read()

    def json(self, path, **kwargs):
        return json.loads(self.request(path, **kwargs))

    def page(self, path="/"):
        page = self.request(path).decode()
        self.csrf = re.search(r'name="csrf-token" content="([^"]+)"', page).group(1)
        return page

    def upload(self, format_name, files=None):
        boundary = uuid.uuid4().hex
        parts = []
        for key, value in {"format": format_name, "name": "Runtime smoke test"}.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, (filename, data) in (files or {}).items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return self.json("/api/datasets", data=b"".join(parts), content_type=f"multipart/form-data; boundary={boundary}")

    def wait(self, dataset_id, job_ids):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            detail = self.json(f"/api/datasets/{dataset_id}")
            jobs = [job for job in detail["jobs"] if job["id"] in job_ids]
            for job in jobs:
                assert job["status"] != "failed", job["error"]
            if len(jobs) == len(job_ids) and all(job["status"] == "complete" for job in jobs):
                return jobs
            time.sleep(0.5)
        raise TimeoutError("Worker did not finish the smoke-test jobs within 300 seconds.")


def smoke(client, keep=False):
    created = []
    try:
        assert client.json("/health")["status"] == "ok"
        page = client.page()
        assert "nav-pipelines" in page and "nav-tutorial" in page
        assert b"<svg" in client.request("/static/images/anndata_schema.svg")
        assert client.json("/api/state")["worker_running"], "Start the analysis worker first."
        uploaded = client.upload("demo")
        dataset_id = uploaded["dataset_id"]
        created.append(dataset_id)
        client.wait(dataset_id, [uploaded["job_id"]])
        print("Demo import passed.", flush=True)

        response = client.json(f"/api/datasets/{dataset_id}/pipelines", data={
            "pipeline": "paired", "features": ["feature_1", "feature_2"], "matrix": "layer:log1p",
            "group": "condition", "reference": "Reference", "target": "Treatment",
            "pair": "subject", "test": "ttest_rel",
        })
        jobs = client.wait(dataset_id, response["job_ids"])
        assert len(jobs) == 3
        for job in jobs:
            base = f"/api/jobs/{job['id']}/files/"
            record = client.json(base + "analysis.json")
            assert record["selection"]["matrix"] == "layer:log1p"
            assert record["pipeline"]["total"] == 3
            assert client.request(base + "results.csv")
            if job["request"]["parameters"]["operation"] == "paired":
                assert client.request(base + "figure.png").startswith(b"\x89PNG")
        print("Three-step pipeline, figures, tables, and provenance passed.", flush=True)

        exported = client.json(f"/api/datasets/{dataset_id}/jobs", data={
            "operation": "export", "features": ["feature_1", "feature_2"], "matrix": "X"})
        client.wait(dataset_id, [exported["job_id"]])
        h5ad = client.request(f"/api/jobs/{exported['job_id']}/files/selection.h5ad")
        files_by_format = {
            "h5ad": {"h5ad": ("smoke.h5ad", h5ad)},
            "csv": {"X": ("adata.X.csv", b",gene1,gene2\ns1,1,2\ns2,3,4\n"),
                    "obs": ("adata.obs.csv", b",group\ns2,B\ns1,A\n"),
                    "var": ("adata.var.csv", b",label\ngene2,two\ngene1,one\n")},
        }
        for format_name, files in files_by_format.items():
            uploaded = client.upload(format_name, files)
            created.append(uploaded["dataset_id"])
            client.wait(uploaded["dataset_id"], [uploaded["job_id"]])
            print(f"{format_name.upper()} upload passed.", flush=True)
        stranger = Client(client.url)
        stranger.page()
        try:
            stranger.request(f"/api/datasets/{dataset_id}")
        except urllib.error.HTTPError as error:
            assert error.code in {401, 404}
        else:
            raise AssertionError("Another session could access the test dataset.")
        print("Session isolation passed.", flush=True)
        return created
    finally:
        if not keep:
            for dataset_id in created:
                try:
                    client.request(f"/api/datasets/{dataset_id}", method="DELETE")
                except urllib.error.HTTPError as error:
                    print(f"Test dataset {dataset_id} could not be removed: HTTP {error.code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--username")
    parser.add_argument("--session", type=Path, help="Private cookie file for checking persistence after a restart")
    parser.add_argument("--keep", action="store_true", help="Retain the three test datasets for restart checks")
    parser.add_argument("--verify-dataset", help="Verify a retained dataset using --session instead of creating new data")
    args = parser.parse_args()
    client = Client(args.url, args.session)
    if args.username:
        password = os.environ.get("ADTL_SMOKE_PASSWORD")
        if not password:
            parser.error("Set ADTL_SMOKE_PASSWORD when using --username.")
        client.page("/login")
        client.request("/login", data=urllib.parse.urlencode({"username": args.username, "password": password}).encode(),
                       content_type="application/x-www-form-urlencoded")
    if args.verify_dataset:
        detail = client.json(f"/api/datasets/{args.verify_dataset}")
        assert detail["dataset"]["status"] == "ready"
        assert detail["jobs"] and all(job["status"] == "complete" for job in detail["jobs"])
        print("Dataset, completed jobs, and session survived the restart.")
    else:
        ids = smoke(client, keep=args.keep)
        if args.keep:
            print("Retained dataset IDs:", ", ".join(ids))
    if args.session:
        args.session.touch(mode=0o600, exist_ok=True)
        args.session.chmod(0o600)
        client.jar.save(ignore_discard=True, ignore_expires=True)


if __name__ == "__main__":
    main()

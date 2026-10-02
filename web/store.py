"""Disk-backed datasets and a small, single-worker SQLite job queue."""

import json
import secrets
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL,
                    created REAL NOT NULL, status TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL
                        REFERENCES datasets(id) ON DELETE CASCADE,
                    created REAL NOT NULL, status TEXT NOT NULL,
                    request TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS login_attempts (
                    identity TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS dataset_owner ON datasets(owner);
                CREATE INDEX IF NOT EXISTS job_dataset ON jobs(dataset_id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "state.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def directory(self, dataset_id):
        # Only opaque server-generated IDs ever become path components.
        if len(dataset_id) != 32 or any(c not in "0123456789abcdef" for c in dataset_id):
            raise ValueError("Invalid dataset identifier.")
        return self.root / dataset_id

    def create_dataset(self, owner, name, max_datasets, max_total_datasets):
        dataset_id = secrets.token_hex(16)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM datasets WHERE owner=?", (owner,)).fetchone()[0]
            total = db.execute("SELECT count(*) FROM datasets").fetchone()[0]
            if count >= max_datasets or total >= max_total_datasets:
                raise ValueError("Dataset limit reached. Delete an old dataset or try again later.")
            db.execute("INSERT INTO datasets(id,owner,name,created,status) VALUES(?,?,?,?,?)",
                       (dataset_id, owner, name[:160], time.time(), "uploading"))
        self.directory(dataset_id).mkdir(mode=0o700)
        return dataset_id

    def dataset(self, dataset_id, owner=None):
        with self.connect() as db:
            sql = "SELECT * FROM datasets WHERE id=?"
            args = [dataset_id]
            if owner is not None:
                sql += " AND owner=?"
                args.append(owner)
            row = db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def datasets(self, owner):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM datasets WHERE owner=? ORDER BY created DESC", (owner,))]

    def enqueue(self, dataset_id, payload, limit=20, global_limit=50):
        job_id = secrets.token_hex(16)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            busy = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= limit or busy >= global_limit:
                raise ValueError("Analysis limit reached. Delete an old dataset or wait for running jobs.")
            db.execute("INSERT INTO jobs(id,dataset_id,created,status,request) VALUES(?,?,?,?,?)",
                       (job_id, dataset_id, time.time(), "queued", json.dumps(payload, allow_nan=False)))
            if payload["kind"] == "import":
                db.execute("UPDATE datasets SET status='queued' WHERE id=?", (dataset_id,))
        return job_id

    def job(self, job_id, owner=None):
        with self.connect() as db:
            sql = "SELECT j.* FROM jobs j JOIN datasets d ON j.dataset_id=d.id WHERE j.id=?"
            args = [job_id]
            if owner is not None:
                sql += " AND d.owner=?"
                args.append(owner)
            row = db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def enqueue_pipeline(self, dataset_id, name, steps, limit=20, global_limit=50):
        pipeline_id = secrets.token_hex(16)
        ids = [secrets.token_hex(16) for _ in steps]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            busy = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count + len(steps) > limit or busy + len(steps) > global_limit:
                raise ValueError("Not enough analysis slots for the full pipeline. Wait for running jobs or use a new dataset.")
            for index, (job_id, params) in enumerate(zip(ids, steps), 1):
                payload = {"kind": "analysis", "parameters": params,
                           "pipeline": {"id": pipeline_id, "name": name, "step": index, "total": len(steps)}}
                db.execute("INSERT INTO jobs(id,dataset_id,created,status,request) VALUES(?,?,?,?,?)",
                           (job_id, dataset_id, time.time(), "queued", json.dumps(payload, allow_nan=False)))
        return ids

    def jobs(self, dataset_id):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM jobs WHERE dataset_id=? ORDER BY created DESC", (dataset_id,))]

    def finish(self, job_id, result=None, error=""):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,result=?,error=? WHERE id=?",
                       ("failed" if error else "complete", json.dumps(result or {}, allow_nan=False), error, job_id))

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row:
                db.execute("UPDATE jobs SET status='running' WHERE id=?", (row["id"],))
                return dict(row)

    def delete_dataset(self, dataset_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            busy = db.execute("SELECT 1 FROM jobs WHERE dataset_id=? AND status IN ('queued','running')",
                              (dataset_id,)).fetchone()
            if busy:
                raise ValueError("Wait for this dataset's queued or running jobs before deleting it.")
            db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        shutil.rmtree(self.directory(dataset_id), ignore_errors=True)

    def cleanup(self, retention_hours):
        with self.connect() as db:
            rows = db.execute("SELECT id FROM datasets WHERE created<?",
                              (time.time() - retention_hours * 3600,)).fetchall()
            db.execute("DELETE FROM login_attempts WHERE created<?", (time.time() - 900,))
        for row in rows:
            try:
                self.delete_dataset(row["id"])
            except ValueError:
                pass

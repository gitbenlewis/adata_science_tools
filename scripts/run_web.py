#!/usr/bin/env python3
"""Start the local GUI and worker, or manage optional login accounts."""

import argparse
import getpass
import os
from pathlib import Path
import secrets
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--require-login", action="store_true")
    parser.add_argument("--create-user", metavar="USERNAME", help="Create or reset a login account; prompts securely for its password")
    parser.add_argument("--worker", action="store_true", help="Run only the analysis worker (for a hosted deployment)")
    parser.add_argument("--execute-job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.data_dir:
        os.environ["ADTL_WEB_DATA_ROOT"] = str(args.data_dir.resolve())
    if args.require_login:
        os.environ["ADTL_WEB_AUTH_REQUIRED"] = "true"
    try:
        import flask  # noqa: F401
    except ImportError:
        parser.exit(1, "Install the optional web dependency first:\n  python -m pip install -r config/requirements-web.txt\n")
    from adata_science_tools.web import create_app
    app = create_app()
    store = app.extensions["store"]
    if args.create_user:
        from werkzeug.security import generate_password_hash
        username = args.create_user.strip()
        if not username or len(username) > 80:
            parser.error("Use a username between 1 and 80 characters.")
        password = getpass.getpass("Password (at least 12 characters): ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm or not 12 <= len(password) <= 1024:
            parser.error("Passwords must match and contain 12–1024 characters.")
        with store.connect() as db:
            existing = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if existing:
                answer = input(f"Reset the password for {username}? [y/N] ")
                if answer.lower() != "y":
                    return
                db.execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(password), existing["id"]))
            else:
                db.execute("INSERT INTO users VALUES(?,?,?)", (secrets.token_hex(16), username, generate_password_hash(password)))
        print(f"Account saved: {username}")
        return
    if args.execute_job:
        from adata_science_tools.web.worker import execute_job
        execute_job(store.root, args.execute_job, app.config["MAX_DENSE_BYTES"], app.config["MAX_IMPORT_BYTES"])
        return
    if args.worker:
        from adata_science_tools.web.worker import serve_worker
        serve_worker(app.config)
        return
    if app.config["PUBLIC_MODE"]:
        parser.error("The local launcher binds to localhost. Use a production WSGI server and --worker for public mode; see docs/web.md.")
    if app.config["AUTH_REQUIRED"]:
        with store.connect() as db:
            if not db.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                parser.exit(1, "Create a login account first:\n  python scripts/run_web.py --create-user YOUR_USERNAME\nUse the same --data-dir if specified.\n")
    env = dict(os.environ, ADTL_WEB_DATA_ROOT=str(store.root))
    worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], env=env)
    print(f"\nAnnData Science Tools: http://127.0.0.1:{args.port}\nData directory: {store.root}\nPress Ctrl+C to stop the app and worker.\n", flush=True)
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


if __name__ == "__main__":
    main()

import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adata_science_tools.web.store import Store
from adata_science_tools.web.worker import serve_worker


class WebWorkerTests(unittest.TestCase):
    def test_completed_dataset_deleted_before_child_exit_does_not_stop_worker(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            dataset_id = store.create_dataset("owner", "test", 3, 30)
            job_id = store.enqueue(dataset_id, {"kind": "import", "format": "demo"})
            handlers = {}

            def launch(*args, **kwargs):
                store.finish(job_id)
                store.delete_dataset(dataset_id)
                return Mock(poll=Mock(return_value=0))

            def stop_when_idle(seconds):
                handlers[signal.SIGTERM](signal.SIGTERM, None)

            with patch("adata_science_tools.web.worker.signal.signal", side_effect=lambda sig, fn: handlers.update({sig: fn})), \
                 patch("adata_science_tools.web.worker.subprocess.Popen", side_effect=launch), \
                 patch("adata_science_tools.web.worker.time.sleep", side_effect=stop_when_idle) as sleep:
                serve_worker(dict(DATA_ROOT=root, RETENTION_HOURS=24, MAX_DENSE_BYTES=1000000,
                                  MAX_IMPORT_BYTES=1000000, JOB_TIMEOUT_SECONDS=30))
            sleep.assert_called_once_with(0.5)
            self.assertIsNone(store.job(job_id))
            self.assertFalse((Path(root) / "worker.heartbeat").exists())


if __name__ == "__main__":
    unittest.main()

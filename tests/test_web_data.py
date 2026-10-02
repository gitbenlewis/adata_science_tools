import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adata_science_tools.web.data import check_h5ad, load_csv_bundle, metadata, read_csv, validate_adata
from adata_science_tools.web.analysis import prepare_selection


class WebDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_bundle(self):
        (self.root / "X.csv").write_text(',001,NA\n0002,1,2\n0001,3,\n')
        (self.root / "obs.csv").write_text(',group,age\n0001,NA,44\n0002,control,32\n')
        (self.root / "var.csv").write_text(',label\nNA,B\n001,A\n')

    def test_csv_alignment_preserves_ids_and_metadata_strings(self):
        self.write_bundle()
        data = load_csv_bundle(self.root)
        self.assertEqual(data.obs_names.tolist(), ["0002", "0001"])
        self.assertEqual(data.var_names.tolist(), ["001", "NA"])
        self.assertEqual(data.obs.group.tolist(), ["control", "NA"])
        self.assertEqual(data.obs.age.tolist(), ["32", "44"])
        self.assertEqual(data.var.label.tolist(), ["A", "B"])
        np.testing.assert_allclose(data.X, [[1, 2], [3, np.nan]], equal_nan=True)
        data.write_h5ad(self.root / "data.h5ad")
        reread = ad.read_h5ad(self.root / "data.h5ad")
        np.testing.assert_allclose(data.X, reread.X, equal_nan=True)
        assert_frame_equal(data.obs, reread.obs, check_categorical=False)

    def test_rejects_duplicate_headers_ids_mismatch_and_nonnumeric_matrix(self):
        for name, content, match in [
            ("X.csv", ',001,001\n0002,1,2\n0001,3,4\n', "duplicate"),
            ("obs.csv", ',group\n0001,a\n0001,b\n', "unique"),
            ("obs.csv", ',group\n0001,a\nmissing,b\n', "match"),
            ("X.csv", ',001,NA\n0002,foo,2\n0001,3,4\n', "numeric"),
        ]:
            with self.subTest(name=name, content=content):
                self.write_bundle()
                (self.root / name).write_text(content)
                with self.assertRaisesRegex(ValueError, match):
                    load_csv_bundle(self.root)

    def test_empty_feature_metadata_is_supported(self):
        self.write_bundle()
        (self.root / "var.csv").write_text('""\n001\nNA\n')
        self.assertEqual(load_csv_bundle(self.root).shape, (2, 2))

    def test_h5ad_rejects_external_links_and_expansion(self):
        with h5py.File(self.root / "linked.h5ad", "w") as handle:
            handle["outside"] = h5py.ExternalLink("/tmp/other.h5ad", "/")
        with self.assertRaisesRegex(ValueError, "linked"):
            check_h5ad(self.root / "linked.h5ad", 1024)
        with h5py.File(self.root / "big.h5ad", "w") as handle:
            handle.create_dataset("X", shape=(1000, 1000), dtype="float64", compression="gzip")
        with self.assertRaisesRegex(ValueError, "Expanded"):
            check_h5ad(self.root / "big.h5ad", 1024)

    def test_sparse_layer_raw_selection_and_no_mutation(self):
        data = ad.AnnData(sparse.csr_matrix([[1., 2.], [3., 4.]]),
                          obs=pd.DataFrame({"group": ["A", "B"], "age": ["20", "30"]}, index=["s1", "s2"]),
                          var=pd.DataFrame(index=["f1", "f2"]))
        data.layers["double"] = data.X * 2
        data.raw = data.copy()
        for matrix, expected in [("X", 3), ("layer:double", 6), ("raw", 3)]:
            params = {"operation": "export", "matrix": matrix, "features": ["f1"],
                      "filter_column": "group", "filter_values": ["B"], "numeric_columns": ["age"]}
            work, summary = prepare_selection(data, params)
            self.assertTrue(sparse.issparse(work.X))
            self.assertEqual(work.X[0, 0], expected)
            self.assertEqual(summary["n_obs"], 1)
            self.assertEqual(work.obs.age.iloc[0], 30)
        self.assertEqual(data.obs.age.iloc[0], "20")
        self.assertEqual(data.shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "budget"):
            prepare_selection(data, {"operation": "export"}, max_dense_bytes=1)
        self.assertEqual(metadata(data)["n_vars"], 2)


if __name__ == "__main__":
    unittest.main()

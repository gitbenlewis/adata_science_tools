import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl


class SaveDatasetTests(unittest.TestCase):
    def make_adata(self):
        obs = pd.DataFrame({"group": ["a", "b", "c"]}, index=["s1", "s2", "s3"])
        var = pd.DataFrame({"feature_type": ["protein", "rna"]}, index=["geneA", "geneB"])
        adata = ad.AnnData(
            X=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            obs=obs,
            var=var,
        )
        adata.layers["scaled"] = adata.X + 10.0
        adata.obsm["source_values"] = pd.DataFrame(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            index=adata.obs_names,
            columns=["src_a", "src_b"],
        )
        adata.obsm["embedding"] = np.array(
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3], [2.1, 2.2, 2.3]]
        )
        adata.obsm["feature_width"] = np.array(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        )
        adata.obsm["sparse_embedding"] = sp.csr_matrix(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]
        )
        return adata

    def test_save_dataset_writes_default_obsm_tables(self):
        adata = self.make_adata()
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "dataset.h5ad"

            adtl.save_dataset(adata, base_path)

            self.assertTrue((Path(tmpdir) / "dataset.h5ad").exists())
            self.assertTrue((Path(tmpdir) / "dataset.obs.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.var.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.X.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.layer.scaled.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.obsm.source_values.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.obsm.embedding.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.obsm.feature_width.csv").exists())
            self.assertTrue((Path(tmpdir) / "dataset.obsm.sparse_embedding.csv").exists())

            source_values = pd.read_csv(Path(tmpdir) / "dataset.obsm.source_values.csv", index_col=0)
            assert_frame_equal(source_values, adata.obsm["source_values"])

            embedding = pd.read_csv(Path(tmpdir) / "dataset.obsm.embedding.csv", index_col=0)
            self.assertEqual(embedding.columns.tolist(), ["dim_0", "dim_1", "dim_2"])
            assert_allclose(embedding.to_numpy(), adata.obsm["embedding"], atol=1e-8, rtol=1e-8)

            feature_width = pd.read_csv(Path(tmpdir) / "dataset.obsm.feature_width.csv", index_col=0)
            self.assertEqual(feature_width.columns.tolist(), ["geneA", "geneB"])
            assert_allclose(feature_width.to_numpy(), adata.obsm["feature_width"], atol=1e-8, rtol=1e-8)

            sparse_embedding = pd.read_csv(Path(tmpdir) / "dataset.obsm.sparse_embedding.csv", index_col=0)
            self.assertEqual(sparse_embedding.columns.tolist(), ["geneA", "geneB"])
            assert_allclose(
                sparse_embedding.to_numpy(),
                adata.obsm["sparse_embedding"].toarray(),
                atol=1e-8,
                rtol=1e-8,
            )

    def test_save_dataset_can_skip_or_select_obsm_keys(self):
        adata = self.make_adata()
        with tempfile.TemporaryDirectory() as tmpdir:
            no_obsm_base = Path(tmpdir) / "no_obsm"
            selected_base = Path(tmpdir) / "selected"

            adtl.save_dataset(adata, no_obsm_base, save_obsm=False)
            adtl.save_dataset(adata, selected_base, obsm_keys=["embedding"])

            self.assertFalse((Path(tmpdir) / "no_obsm.obsm.embedding.csv").exists())
            self.assertTrue((Path(tmpdir) / "selected.obsm.embedding.csv").exists())
            self.assertFalse((Path(tmpdir) / "selected.obsm.source_values.csv").exists())

    def test_save_dataset_validates_requested_obsm_keys_and_safe_name_collisions(self):
        adata = self.make_adata()
        adata.obsm["a/b"] = np.ones((adata.n_obs, 1))
        adata.obsm["a_b"] = np.ones((adata.n_obs, 1)) * 2

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(KeyError, "obsm key"):
                adtl.save_dataset(adata, Path(tmpdir) / "missing", obsm_keys=["missing"])

            with self.assertRaisesRegex(ValueError, "collide"):
                adtl.save_dataset(adata, Path(tmpdir) / "collision", obsm_keys=["a/b", "a_b"])

    def test_save_dataset_warns_and_skips_unsupported_obsm_values(self):
        adata = self.make_adata()
        adata.obsm["tensor"] = np.zeros((adata.n_obs, 2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertLogs("adata_science_tools._io._IO", level="WARNING") as logs:
                adtl.save_dataset(adata, Path(tmpdir) / "dataset", obsm_keys=["tensor"])

            self.assertIn("not 2D", "\n".join(logs.output))
            self.assertFalse((Path(tmpdir) / "dataset.obsm.tensor.csv").exists())

    def test_save_dataset_exports_ref_vs_target_source_values_obsm(self):
        obs = pd.DataFrame(
            {
                "Pre_or_Post_obs_col": ["Pre", "Post", "Pre", "Post"],
                "Subject_ID": ["A", "A", "B", "B"],
            },
            index=["A_pre", "A_post", "B_pre", "B_post"],
        )
        var = pd.DataFrame(index=["feature_1", "feature_2"])
        source_adata = ad.AnnData(
            X=np.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                ]
            ),
            obs=obs,
            var=var,
        )

        result = adtl.ref_vs_target_adata(
            source_adata,
            pair_by_key="Subject_ID",
            save_source_values_obsm=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            adtl.save_dataset(result, Path(tmpdir) / "post_minus_pre")

            ref_values = pd.read_csv(Path(tmpdir) / "post_minus_pre.obsm.pre_values.csv", index_col=0)
            target_values = pd.read_csv(Path(tmpdir) / "post_minus_pre.obsm.post_values.csv", index_col=0)

            assert_frame_equal(ref_values, result.obsm["pre_values"])
            assert_frame_equal(target_values, result.obsm["post_values"])


if __name__ == "__main__":
    unittest.main()

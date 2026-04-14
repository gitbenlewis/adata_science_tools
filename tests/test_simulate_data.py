import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools._simulate_data as simdata


class SimulateDataTests(unittest.TestCase):
    def test_subpackage_exports(self):
        self.assertTrue(hasattr(simdata, "sim_observations_covars"))
        self.assertTrue(hasattr(simdata, "sim_covar_dependent_features"))
        self.assertTrue(hasattr(simdata, "sim_covar_dependent_dataset"))

    def test_sim_observations_covars_is_seeded_and_supports_binomial_alias(self):
        obs_kwargs = {
            "obs_key_list": ["Age", "gender"],
            "obs_covar_dist_params": {
                "Age": {"dist": "normal", "mean": 10, "stdev": 5},
                "gender": {"dist": "bionomial", "prob": 0.5},
            },
            "n_obs": 5,
            "obs_names_prefix": "sample_",
        }
        obs_df_a = simdata.sim_observations_covars(random_seed=7, **obs_kwargs)
        obs_df_b = simdata.sim_observations_covars(random_seed=7, **obs_kwargs)
        pd.testing.assert_frame_equal(obs_df_a, obs_df_b)
        self.assertEqual(obs_df_a.index.tolist(), [f"sample_{idx}" for idx in range(1, 6)])
        self.assertTrue(np.issubdtype(obs_df_a["Age"].dtype, np.floating))
        self.assertTrue(np.issubdtype(obs_df_a["gender"].dtype, np.integer))
        self.assertTrue(set(obs_df_a["gender"].tolist()).issubset({0, 1}))

    def test_sim_covar_dependent_features_generates_expected_matrix_and_var_df(self):
        obs_df = pd.DataFrame(
            {
                "Age": [20.0, 40.0, 60.0],
                "gender": [0, 1, 1],
            },
            index=["obs_1", "obs_2", "obs_3"],
        )
        X, var_df, returned_obs_df, adata = simdata.sim_covar_dependent_features(
            obs_df=obs_df,
            var_names=["feature_a", "feature_b"],
            betas=[[0.1, 5.0], [0.2, -1.0]],
            yints=[1.0, 3.0],
            also_return_adata=True,
            save_adata_dataset=False,
        )
        expected = np.array(
            [
                [3.0, 7.0],
                [10.0, 10.0],
                [12.0, 14.0],
            ]
        )
        assert_allclose(X, expected, atol=1e-8, rtol=1e-8)
        pd.testing.assert_frame_equal(returned_obs_df, obs_df)
        self.assertEqual(var_df.index.tolist(), ["feature_a", "feature_b"])
        self.assertEqual(
            var_df.columns.tolist(),
            [
                "yint",
                "beta_Age",
                "beta_gender",
                "residual_dist",
                "residual_mean",
                "residual_stdev",
            ],
        )
        self.assertEqual(var_df.loc["feature_a", "beta_gender"], 5.0)
        self.assertEqual(var_df.loc["feature_b", "yint"], 3.0)
        self.assertEqual(var_df.loc["feature_a", "residual_dist"], "normal")
        self.assertEqual(var_df.loc["feature_a", "residual_mean"], 0.0)
        self.assertEqual(var_df.loc["feature_a", "residual_stdev"], 0.0)
        self.assertIsNotNone(adata)
        self.assertEqual(adata.obs_names.tolist(), obs_df.index.tolist())
        self.assertEqual(adata.var_names.tolist(), ["feature_a", "feature_b"])
        assert_allclose(adata.X, expected, atol=1e-8, rtol=1e-8)
        assert_allclose(adata.layers["linear_mean"], expected, atol=1e-8, rtol=1e-8)
        assert_allclose(adata.layers["residual"], np.zeros_like(expected), atol=1e-8, rtol=1e-8)

    def test_sim_covar_dependent_features_broadcasts_1d_betas_and_scalar_yint(self):
        obs_df = pd.DataFrame(
            {
                "Age": [1.0, 2.0],
                "gender": [0, 1],
            },
            index=["obs_1", "obs_2"],
        )
        X, var_df, _, adata = simdata.sim_covar_dependent_features(
            obs_df=obs_df,
            var_names=["feature_a", "feature_b", "feature_c"],
            betas=[2.0, 3.0],
            yints=4.0,
            also_return_adata=False,
            save_adata_dataset=False,
        )
        expected = np.array(
            [
                [6.0, 6.0, 6.0],
                [11.0, 11.0, 11.0],
            ]
        )
        assert_allclose(X, expected, atol=1e-8, rtol=1e-8)
        self.assertTrue((var_df["yint"] == 4.0).all())
        self.assertIsNone(adata)

    def test_sim_covar_dependent_features_supports_seeded_residual_noise_and_layers(self):
        obs_df = pd.DataFrame(
            {
                "Age": [20.0, 40.0, 60.0],
                "gender": [0, 1, 1],
            },
            index=["obs_1", "obs_2", "obs_3"],
        )
        kwargs = {
            "obs_df": obs_df,
            "var_names": ["feature_a", "feature_b"],
            "betas": [[0.1, 5.0], [0.2, -1.0]],
            "yints": [1.0, 3.0],
            "residual_mean": 0.5,
            "residual_stdev": 0.25,
            "random_seed": 13,
            "also_return_adata": True,
            "save_adata_dataset": False,
        }
        X_a, var_df_a, _, adata_a = simdata.sim_covar_dependent_features(**kwargs)
        X_b, var_df_b, _, adata_b = simdata.sim_covar_dependent_features(**kwargs)
        expected_linear_mean = np.array(
            [
                [3.0, 7.0],
                [10.0, 10.0],
                [12.0, 14.0],
            ]
        )
        assert_allclose(X_a, X_b, atol=1e-8, rtol=1e-8)
        pd.testing.assert_frame_equal(var_df_a, var_df_b)
        self.assertIsNotNone(adata_a)
        self.assertIsNotNone(adata_b)
        assert_allclose(adata_a.layers["linear_mean"], expected_linear_mean, atol=1e-8, rtol=1e-8)
        assert_allclose(adata_a.layers["residual"], X_a - expected_linear_mean, atol=1e-8, rtol=1e-8)
        assert_allclose(adata_b.layers["residual"], X_b - expected_linear_mean, atol=1e-8, rtol=1e-8)
        self.assertFalse(np.allclose(adata_a.layers["residual"], 0.0))
        self.assertTrue((var_df_a["residual_dist"] == "normal").all())
        self.assertTrue((var_df_a["residual_mean"] == 0.5).all())
        self.assertTrue((var_df_a["residual_stdev"] == 0.25).all())

    def test_sim_covar_dependent_dataset_wrapper_saves_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            obs_csv_path = temp_path / "obs_metadata"
            dataset_output_path = temp_path / "simulated_dataset"
            X, var_df, obs_df, adata = simdata.sim_covar_dependent_dataset(
                obs_key_list=["Age", "gender"],
                obs_covar_dist_params={
                    "Age": {"dist": "normal", "mean": 50, "stdev": 2},
                    "gender": {"dist": "binomial", "prob": 0.25},
                },
                n_obs=4,
                obs_names_prefix="cell_",
                save_obs_df=True,
                save_obs_df_path=obs_csv_path,
                random_seed=11,
                var_names=["feature_a"],
                betas=[0.5, 2.0],
                yints=1.5,
                residual_mean=0.0,
                residual_stdev=0.75,
                also_return_adata=True,
                save_adata_dataset=True,
                output_path=dataset_output_path,
            )
            self.assertEqual(X.shape, (4, 1))
            self.assertEqual(var_df.shape, (1, 6))
            self.assertEqual(obs_df.index.tolist(), [f"cell_{idx}" for idx in range(1, 5)])
            self.assertIsNotNone(adata)
            self.assertTrue((temp_path / "obs_metadata.csv").exists())
            self.assertTrue((temp_path / "simulated_dataset.h5ad").exists())
            self.assertTrue((temp_path / "simulated_dataset.obs.csv").exists())
            self.assertTrue((temp_path / "simulated_dataset.var.csv").exists())
            self.assertTrue((temp_path / "simulated_dataset.X.csv").exists())
            self.assertTrue((temp_path / "simulated_dataset.layer.linear_mean.csv").exists())
            self.assertTrue((temp_path / "simulated_dataset.layer.residual.csv").exists())
            self.assertIn("linear_mean", adata.layers)
            self.assertIn("residual", adata.layers)
            self.assertFalse(np.allclose(adata.layers["residual"], 0.0))
            assert_allclose(adata.X, adata.layers["linear_mean"] + adata.layers["residual"])

    def test_validation_errors_are_raised_for_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "must contain at least one covariate"):
            simdata.sim_observations_covars(obs_key_list=[], save_obs_df=False)

        with self.assertRaisesRegex(ValueError, "missing a distribution spec"):
            simdata.sim_observations_covars(
                obs_key_list=["Age", "gender"],
                obs_covar_dist_params={"Age": {"dist": "normal", "mean": 10, "stdev": 5}},
            )

        with self.assertRaisesRegex(ValueError, "prob must be between 0 and 1"):
            simdata.sim_observations_covars(
                obs_key_list=["gender"],
                obs_covar_dist_params={"gender": {"dist": "binomial", "prob": 1.2}},
            )

        with self.assertRaisesRegex(ValueError, "stdev must be >= 0"):
            simdata.sim_observations_covars(
                obs_key_list=["Age"],
                obs_covar_dist_params={"Age": {"dist": "normal", "mean": 10, "stdev": -1}},
            )

        bad_obs_df = pd.DataFrame({"Age": [10, "bad"], "gender": [0, 1]})
        with self.assertRaisesRegex(TypeError, "non-numeric or missing predictor values"):
            simdata.sim_covar_dependent_features(
                obs_df=bad_obs_df,
                var_names=["feature_a"],
                betas=[0.1, 2.0],
                save_adata_dataset=False,
            )

        valid_obs_df = pd.DataFrame({"Age": [10.0, 20.0], "gender": [0, 1]})
        with self.assertRaisesRegex(ValueError, "duplicate feature names"):
            simdata.sim_covar_dependent_features(
                obs_df=valid_obs_df,
                var_names=["feature_a", "feature_a"],
                betas=[[0.1, 2.0], [0.2, 3.0]],
                save_adata_dataset=False,
            )

        with self.assertRaisesRegex(ValueError, "1D betas must have length 2"):
            simdata.sim_covar_dependent_features(
                obs_df=valid_obs_df,
                var_names=["feature_a"],
                betas=[0.1],
                save_adata_dataset=False,
            )

        with self.assertRaisesRegex(ValueError, "1D yints must have length 2"):
            simdata.sim_covar_dependent_features(
                obs_df=valid_obs_df,
                var_names=["feature_a", "feature_b"],
                betas=[0.1, 2.0],
                yints=[1.0],
                save_adata_dataset=False,
            )

        with self.assertRaisesRegex(ValueError, "Unsupported residual_dist"):
            simdata.sim_covar_dependent_features(
                obs_df=valid_obs_df,
                var_names=["feature_a"],
                betas=[0.1, 2.0],
                residual_dist="laplace",
                save_adata_dataset=False,
            )

        with self.assertRaisesRegex(ValueError, "residual_stdev must be >= 0"):
            simdata.sim_covar_dependent_features(
                obs_df=valid_obs_df,
                var_names=["feature_a"],
                betas=[0.1, 2.0],
                residual_stdev=-0.1,
                save_adata_dataset=False,
            )


if __name__ == "__main__":
    unittest.main()

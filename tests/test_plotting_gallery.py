import importlib
import inspect
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import adata_science_tools as adtl
gallery_module = importlib.import_module(
    "adata_science_tools.example_plotting_gallery.generate_gallery"
)
from adata_science_tools.example_plotting_gallery.generate_gallery import (
    generate_gallery,
)
from adata_science_tools.example_plotting_gallery.manifest import (
    EXCLUDED_PUBLIC_CALLABLES,
    RENDERER_MANIFEST,
    RENDERER_NAMES,
    validate_manifest,
)
from adata_science_tools.example_plotting_gallery.simulated_data import (
    make_independent_group_adata,
    make_ols_model_results,
    make_residual_diagnostic_frame,
    run_independent_diff_test,
)


class PlottingGalleryTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_manifest_exactly_covers_exported_plotting_functions(self):
        validate_manifest()
        public_functions = {
            name
            for name, value in vars(adtl.pl).items()
            if not name.startswith("_") and inspect.isfunction(value)
        }

        self.assertEqual(
            public_functions,
            RENDERER_NAMES | set(EXCLUDED_PUBLIC_CALLABLES),
        )
        self.assertEqual(len(RENDERER_MANIFEST), 44)
        self.assertEqual(
            sum(len(spec.cases) for spec in RENDERER_MANIFEST),
            51,
        )
        for spec in RENDERER_MANIFEST:
            renderer = getattr(adtl.pl, spec.name)
            self.assertEqual(
                renderer.__module__,
                f"adata_science_tools.{spec.module}",
            )

    def test_committed_assets_exactly_match_manifest_cases(self):
        declared_assets = {
            case.asset
            for spec in RENDERER_MANIFEST
            for case in spec.cases
        }
        asset_directory = REPO_ROOT / "docs" / "assets" / "plotting_gallery"
        committed_assets = {
            path.name for path in asset_directory.glob("*.png")
        }

        self.assertEqual(committed_assets, declared_assets)
        for asset_name in committed_assets:
            asset_path = asset_directory / asset_name
            self.assertGreater(asset_path.stat().st_size, 0)
            self.assertEqual(asset_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_gallery_assets_are_embedded_once_in_their_documentation_pages(self):
        docs_dir = REPO_ROOT / "docs"
        renderer_doc_overrides = {
            "paired_datapoints": "_paired_datapoints.md",
        }
        doc_names = {
            f"{spec.module.rsplit('.', 1)[-1]}.md"
            for spec in RENDERER_MANIFEST
        } | set(renderer_doc_overrides.values())
        documentation = {
            name: (docs_dir / name).read_text(encoding="utf-8")
            for name in sorted(doc_names)
        }

        for spec in RENDERER_MANIFEST:
            expected_doc = renderer_doc_overrides.get(
                spec.name,
                f"{spec.module.rsplit('.', 1)[-1]}.md",
            )
            for case in spec.cases:
                image_tag = (
                    f'<img src="assets/plotting_gallery/{case.asset}"'
                )
                occurrences = {
                    doc_name: markdown.count(image_tag)
                    for doc_name, markdown in documentation.items()
                    if image_tag in markdown
                }

                with self.subTest(renderer=spec.name, case=case.case_id):
                    self.assertEqual(occurrences, {expected_doc: 1})

    def test_independent_builder_is_balanced_and_deterministic(self):
        first = make_independent_group_adata(random_seed=2026)
        second = make_independent_group_adata(random_seed=2026)

        np.testing.assert_allclose(first.X, second.X)
        pd.testing.assert_frame_equal(first.obs, second.obs)
        pd.testing.assert_frame_equal(first.var, second.var)
        self.assertEqual(
            first.obs["condition"].value_counts(sort=False).to_dict(),
            {"control": 24, "case": 24},
        )
        self.assertEqual(
            first.var["truth_class"].tolist(),
            ["positive", "negative", "null", "constant", "all_zero"],
        )

    def test_synthetic_response_fixture_has_invariant_sample_metadata(self):
        expression = pd.read_csv(
            REPO_ROOT
            / "example_plotting_gallery"
            / "data"
            / "synthetic_expression.csv"
        )
        metadata_cardinality = expression.groupby("sample_id")[[
            "response_group",
            "subtype",
            "cohort",
        ]].nunique()

        self.assertTrue(
            metadata_cardinality.eq(1).all().all(),
            metadata_cardinality.loc[~metadata_cardinality.eq(1).all(axis=1)],
        )

    def test_diff_results_are_left_joined_without_matrix_misalignment(self):
        source = make_independent_group_adata(random_seed=2026)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original, annotated, results = run_independent_diff_test(source)

        self.assertTrue(original.var_names.equals(annotated.var_names))
        np.testing.assert_allclose(original.X, annotated.X)
        self.assertNotIn("feature_all_zero", results.index)
        self.assertTrue(
            pd.isna(annotated.var.loc["feature_all_zero", "effect"])
        )
        pd.testing.assert_series_equal(
            results["effect"],
            results["l2fc_case_vs_control"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            results["pvalue"],
            results["ttest_ind_pvals_case_vs_control"],
            check_names=False,
        )

    def test_library_derived_ols_and_residual_inputs_are_auditable(self):
        source = make_independent_group_adata(random_seed=2026)
        ols_results = make_ols_model_results(source)
        residuals = make_residual_diagnostic_frame(source)

        self.assertEqual(
            ols_results.index.tolist(),
            ["feature_positive", "feature_negative", "feature_null"],
        )
        self.assertIn(
            "gallery_ols_Coef_condition_indicator",
            ols_results.columns,
        )
        self.assertIn(
            "gallery_ols_CI_low_condition_indicator",
            ols_results.columns,
        )
        self.assertEqual(
            set(residuals.columns),
            {
                "sample",
                "feature",
                "feature_label",
                "group",
                "observed",
                "expected",
                "residual",
                "model_name",
            },
        )
        np.testing.assert_allclose(
            residuals["residual"],
            residuals["observed"] - residuals["expected"],
        )

    def test_selected_gallery_generation_writes_declared_pngs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = generate_gallery(
                temporary_directory,
                renderer_names=[
                    "category_composition",
                    "geneset_enrichemnt_ol_ven_M_n_N_x",
                    "geneset_enrichment_venn",
                    "plot_heatmap",
                    "spearman_cor_dotplot_2",
                    "vbar_l2fc_dotplot_column",
                    "venn_plot_2list",
                    "venn_plot_3list",
                ],
            )

            self.assertEqual(
                {path.name for path in generated},
                {
                    "category_composition__percent_annotated.png",
                    "geneset_enrichemnt_ol_ven_M_n_N_x__replacement_smoke.png",
                    "geneset_enrichment_venn__universe_filtered.png",
                    "plot_heatmap__clustered.png",
                    "plot_heatmap__fixed_order.png",
                    "spearman_cor_dotplot_2__dual_hue.png",
                    "vbar_l2fc_dotplot_column__synthetic_response_panel.png",
                    "venn_plot_2list__two_set_overlap.png",
                    "venn_plot_3list__three_set_overlap.png",
                },
            )
            for path in generated:
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_config_driven_cli_generates_selected_asset_and_log(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_dir = temporary_path / "assets"
            log_dir = temporary_path / "logs"
            config_path = temporary_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "plotting_gallery_params:",
                        f"  output_dir: {output_dir}",
                        f"  log_dir: {log_dir}",
                        "  renderer_names:",
                        "    - category_composition",
                        "  case_ids:",
                        "    - percent_annotated",
                        "  continue_on_error: false",
                    ]
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPO_PARENT)
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(temporary_path / "matplotlib")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "adata_science_tools.example_plotting_gallery.generate_gallery",
                    "--config",
                    str(config_path),
                ],
                cwd=REPO_PARENT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output_dir.glob("*.png")},
                {"category_composition__percent_annotated.png"},
            )
            log_paths = list(log_dir.glob("generate_gallery_*.log"))
            self.assertEqual(len(log_paths), 1)
            self.assertIn(
                "Generated 1 gallery cases",
                log_paths[0].read_text(encoding="utf-8"),
            )

    def test_seeded_gallery_generation_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first = generate_gallery(
                    Path(temporary_directory) / "first",
                    renderer_names=["barh_column"],
                )[0]
                second = generate_gallery(
                    Path(temporary_directory) / "second",
                    renderer_names=["barh_column"],
                )[0]

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_failed_generation_preserves_previous_asset_atomically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_path = (
                Path(temporary_directory)
                / "category_composition__percent_annotated.png"
            )
            previous_content = b"previous gallery asset"
            asset_path.write_bytes(previous_content)

            with mock.patch.object(
                gallery_module,
                "_invoke_case",
                side_effect=RuntimeError("forced renderer failure"),
            ):
                with self.assertRaisesRegex(
                    gallery_module.GalleryGenerationError,
                    "forced renderer failure",
                ):
                    generate_gallery(
                        temporary_directory,
                        renderer_names=["category_composition"],
                    )

            self.assertEqual(asset_path.read_bytes(), previous_content)
            self.assertEqual(
                list(Path(temporary_directory).glob(".*.png")),
                [],
            )


if __name__ == "__main__":
    unittest.main()

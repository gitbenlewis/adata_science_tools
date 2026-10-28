"""Public root/tl entrypoints must expose the same statistical callables."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl


def test_statistical_inference_public_exports():
    for name in ("paired_ratio_inference", "estimate_right_censored", "summarize_km",
                 "compare_survival", "SurvivalInputError"):
        assert callable(getattr(adtl, name))
        assert getattr(adtl, name) is getattr(adtl.tl, name)

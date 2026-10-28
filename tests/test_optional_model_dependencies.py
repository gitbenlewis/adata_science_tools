"""Exercise optional dependency boundaries in isolated Python processes."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_PARENT = Path(__file__).resolve().parents[2]
BLOCK_IMPORTS = """
import importlib.abc
import sys

blocked = sys.argv[1].split(',')
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('Blocked ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockImports())
import adata_science_tools as adtl
"""


class OptionalModelDependencyTests(unittest.TestCase):
    def run_isolated(self, blocked, code):
        result = subprocess.run(
            [sys.executable, "-c", BLOCK_IMPORTS + code, blocked],
            cwd=REPO_PARENT,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_plotting_without_model_dependencies(self):
        for blocked in ("statsmodels", "patsy", "yaml", "statsmodels,patsy,yaml"):
            with self.subTest(blocked=blocked):
                self.run_isolated(blocked, """
import pandas as pd
import matplotlib.pyplot as plt
assert callable(adtl.calculate_expectations)
assert callable(adtl.fit_smf_ols_models_and_summarize_wide)
fig, _ = adtl.adata_histograms(
    df=pd.DataFrame({'a': [1., 2., 3.]}), var_names=['a'], show=False)
fig.canvas.draw()
plt.close(fig)
""")

    def test_fitting_requires_model_dependencies(self):
        for blocked in ("statsmodels", "patsy"):
            with self.subTest(blocked=blocked):
                self.run_isolated(blocked, """
import pandas as pd
try:
    adtl.fit_smf_ols_models_and_summarize_wide(
        pd.DataFrame({'x': [0., 1., 2.], 'y': [1., 3., 5.]}),
        feature_columns=['y'], predictors=['x'])
except ModuleNotFoundError as error:
    assert error.name.split('.')[0] in blocked
else:
    raise AssertionError('Fitting silently accepted a missing dependency')
""")

    def test_in_memory_expectations_without_yaml(self):
        self.run_isolated("yaml", """
import anndata as ad
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
adata = ad.AnnData(
    X=np.array([[1.], [3.], [5.], [7.], [9.]]),
    obs=pd.DataFrame({'x': [0., 1., 2., 3., 4.]}, index=list('abcde')),
    var=pd.DataFrame(index=['feature']))
coefficients = adtl.calculate_expectations(adata, predictors=['x'])
prediction = adtl.predict_expectation(adata, coefficients)
np.testing.assert_allclose(prediction, adata.X, atol=1e-10)
with tempfile.TemporaryDirectory() as directory:
    destination = Path(directory) / 'model.csv'
    try:
        adtl.save_expectation_model_files(coefficients, destination)
    except ModuleNotFoundError as error:
        assert error.name == 'yaml'
    else:
        raise AssertionError('YAML serialization accepted a missing dependency')
    assert not destination.exists()
""")

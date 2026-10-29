"""Synthetic public-API paired panels; optional output directory argv[1]."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl

output = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
data = pd.DataFrame({'visit': ['Before', 'After'] * 12,
 'subject': np.repeat(np.arange(12), 2),
 'score': rng.uniform(1, 3, 24), 'feature_a': rng.uniform(2, 8, 24),
 'feature_b': rng.uniform(10, 20, 24)})
# Illustrative strings, not estimates from these random observations.
annotations = {'defaults': {'fontsize': 10}, **{name: {
 'text': 'Supplied change summary\nIllustrative CI [−0.25, 0.01]'}
 for name in ['score', 'feature_a', 'feature_b']}}
fig, axes, plotted = adtl.paired_datapoints(
 df=data, var_names=['score', 'feature_a', 'feature_b'],
 groupby_key='visit', groupby_key_ref_value='Before',
 groupby_key_target_value='After', pair_by_key='subject',
 connect_lines=True, line_color='0.65', line_alpha=.5,
 point_color_by_side=True, ref_point_color='#44AA99', target_point_color='#332288',
 jitter_amount=0, boxplot=False, ncols=3, figsize=(12, 5), sharey=False,
 panel_annotations=annotations,
 ylabel={'score': 'Composite score', 'feature_a': 'Measurement', 'feature_b': 'Measurement'},
 ylims={name: (0, None) for name in ['score', 'feature_a', 'feature_b']},
 subplot_title_style={'color': '#183B4E', 'fontweight': 'bold'},
 xtick_label_style={'rotation': 0, 'ha': 'center'}, show=False)
for extension in ['png', 'pdf', 'svg']:
 fig.savefig(output / ('paired_scores.' + extension), dpi=150, bbox_inches='tight')

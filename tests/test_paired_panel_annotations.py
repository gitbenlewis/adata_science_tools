"""Public API regression tests for annotated paired panels."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import adata_science_tools as adtl


def plot(**kwargs):
    df = pd.DataFrame({'visit': ['Before', 'After'] * 4,
                       'subject': np.repeat(np.arange(4), 2),
                       'cohort': ['X'] * 4 + ['Y'] * 4,
                       **{name: np.arange(8.) * scale + 1 for name, scale in
                          [('a', 1), ('b', 10), ('c', 100)]}})
    options = dict(df=df, var_names=['c', 'a', 'b'], groupby_key='visit',
                   groupby_key_ref_value='Before', groupby_key_target_value='After',
                   pair_by_key='subject', jitter_amount=0, boxplot=False,
                   ncols=2, show=False)
    options.update(kwargs)
    return adtl.paired_datapoints(**options)


@pytest.mark.parametrize('sharey', [False, True, 'row', 'col'])
@pytest.mark.parametrize('difference', [False, True])
def test_annotations_preserve_data_and_clear_titles(sharey, difference, tmp_path):
    _, _, before = plot(sharey=sharey, show_paired_difference=difference)
    fig, axes, after = plot(sharey=sharey, show_paired_difference=difference,
        panel_annotations={'defaults': {'fontsize': 11},
                           'a': {'text': 'Supplied change\n95% CI [1, 2]\nThird line'}},
        ylabel={'a': 'Measurement'},
        subplot_title_style={'color': '#183B4E', 'panels': {'a': {'fontweight': 'bold'}}},
        xtick_label_style={'rotation': 0, 'ha': 'center'})
    pd.testing.assert_frame_equal(before, after)
    assert list(axes) == ['c', 'a', 'b']
    assert not fig.axes[3].get_visible()
    assert axes['a'].get_ylabel() == 'Measurement'
    assert axes['b'].get_ylabel() == 'value'
    assert axes['a'].title.get_fontweight() == 'bold'
    assert axes['a'].get_xticklabels()[0].get_rotation() == 0
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = axes['a'].texts[0]
    assert text.get_text().count('\n') == 2
    bounds = text.get_window_extent(renderer)
    assert bounds.y0 > axes['a'].bbox.y1
    assert bounds.y1 < axes['a'].title.get_window_extent(renderer).y0
    for extension in ('png', 'pdf', 'svg'):
        path = tmp_path / ('panels.' + extension)
        fig.savefig(path, bbox_inches='tight')
        assert path.stat().st_size > 1000


@pytest.mark.parametrize('option', ['ylabel', 'ylims', 'panel_annotations'])
def test_unknown_panel_keys(option):
    with pytest.raises(ValueError, match='Unknown panel'):
        plot(**{option: {'missing': None}})


def test_limits_and_shared_conflicts():
    _, axes, values = plot(ylims={'a': (0, None), 'c': (0, 5)})
    assert axes['a'].get_ylim()[0] == 0
    assert axes['a'].get_ylim()[1] > 8
    assert axes['c'].get_ylim() == (0, 5)
    assert values.loc[values.panel == 'c', 'value'].max() > 5
    with pytest.raises(ValueError, match='Conflicting'):
        plot(sharey=True, ylims={'a': (0, None), 'b': (1, None)})


def test_facet_keys_and_defaults():
    fig, axes, _ = plot(var_names=['a'], subplot_by_obs_key='cohort',
                       panel_annotations={'X': {'text': 'X summary'}}, ylabel='Shared')
    assert list(axes) == ['X', 'Y']
    assert axes['X'].texts[0].get_text() == 'X summary'
    assert all(ax.get_ylabel() == 'Shared' for ax in axes.values())
    with pytest.raises(ValueError, match='Unknown panel'):
        plot(var_names=['a'], subplot_by_obs_key='cohort', panel_annotations={'a': {'text': 'wrong'}})


@pytest.mark.parametrize('mode,joined', [('row', ('c', 'a')), ('col', ('c', 'b'))])
def test_difference_sharing(mode, joined):
    fig, axes, _ = plot(sharey=mode, show_paired_difference=True)
    twins = {ax.get_label().replace('__paired_difference', ''): ax
             for ax in fig.axes if ax.get_label().endswith('__paired_difference')}
    first, second = joined
    assert twins[first].get_shared_y_axes().joined(twins[first], twins[second])
    other = ({'a', 'b', 'c'} - set(joined)).pop()
    assert not twins[first].get_shared_y_axes().joined(twins[first], twins[other])


def test_partial_shared_limit_includes_later_panel():
    _, axes, _ = plot(var_names=['a', 'b', 'c'], sharey=True, ylims={'a': (0, None)})
    assert all(ax.get_ylim()[1] > 701 for ax in axes.values())


def test_annotation_position_validation_and_explicit_title_style():
    with pytest.raises(ValueError, match='placement'):
        plot(panel_annotations={'a': {'text': 'summary', 'ha': 'left'}})
    _, axes, _ = plot(subplot_title_y=.9,
                      subplot_title_style={'pad': 80, 'fontweight': 'bold'},
                      panel_annotations={'a': {'text': 'summary'}})
    assert axes['a'].title.get_position()[1] == .9
    assert axes['a'].title.get_fontweight() == 'bold'

"""Explicit GUI-to-package adapters; no executable formulas or arbitrary kwargs."""

import logging
import math
import warnings

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .. import _plotting as pl
from .. import _tools as tl
from .data import feature_labels


def field(key, label, kind="obs", choices=None, default=None, required=False):
    return dict(key=key, label=label, kind=kind, choices=choices, default=default, required=required)


GROUP = field("group", "Group column", required=True)
PAIR = field("pair", "Subject / pairing column", required=True)
REF = field("reference", "Reference group", "level", required=True)
TARGET = field("target", "Target group", "level", required=True)
PVALUE = field("pvalue", "P-value or FDR column", "table", required=True)
EFFECT = field("effect", "Effect column", "table", required=True)
CATALOG = {
    "histogram": {"label": "Histograms", "category": "Explore", "function": "adata_histograms",
                  "fields": [field("group", "Color by"), field("bins", "Bins", "number", default=25)]},
    "datapoints": {"label": "Grouped datapoints", "category": "Explore", "function": "datapoints",
                   "fields": [field("group", "Group column"), field("distribution", "Distribution", "choice",
                                   ["box", "violin", "points"], "box")]},
    "paired": {"label": "Paired datapoints", "category": "Explore", "function": "paired_datapoints",
               "fields": [GROUP, REF, TARGET, PAIR, field("difference", "Difference panel", "choice",
                                                       ["none", "difference", "log2fc"], "none")]},
    "correlation": {"label": "Correlation plot", "category": "Explore", "function": "corr_dotplot",
                    "fields": [field("x", "X variable", "variable", required=True),
                               field("y", "Y variable", "variable", required=True),
                               field("group", "Color by"), field("method", "Correlation", "choice",
                                                                ["pearson", "spearman"], "pearson")]},
    "longitudinal": {"label": "Longitudinal trajectories", "category": "Explore", "function": "longitudinal_trajectories",
                     "fields": [field("x", "Time / visit column", required=True), PAIR,
                                field("y", "Response feature", "feature", required=True),
                                field("group", "Color by"), field("order", "Visit order (comma separated)", "text", required=True)]},
    "composition": {"label": "Category composition", "category": "Explore", "function": "category_composition",
                    "fields": [field("x", "X category", required=True), field("group", "Category", required=True),
                               field("normalize", "Scale", "choice", ["count", "fraction", "percent"], "count")]},
    "diff_test": {"label": "Differential testing", "category": "Statistics", "function": "diff_test",
                  "fields": [GROUP, REF, TARGET, field("pair", "Pairing column (paired tests)"),
                             field("test", "Test", "choice", ["ttest_ind", "mannwhitneyu", "ttest_rel", "WilcoxonSigned"], "ttest_ind")]},
    "ols": {"label": "Linear models (OLS)", "category": "Statistics", "function": "fit_smf_ols_models_and_summarize_adata",
            "fields": [field("predictors", "Predictor columns", "multi_obs", required=True)]},
    "mixedlm": {"label": "Mixed-effects models", "category": "Statistics", "function": "fit_smf_mixedlm_models_and_summarize_adata",
                "fields": [field("predictors", "Predictor columns", "multi_obs", required=True),
                           field("group", "Random-intercept grouping column", required=True),
                           field("reml", "Estimation", "choice", ["REML", "ML"], "REML")]},
    "average": {"label": "Grouped feature means", "category": "Statistics", "function": "average_feature_expression",
                "fields": [GROUP]},
    "volcano": {"label": "Volcano plot", "category": "Results", "function": "volcano_plot_generic",
                "fields": [EFFECT, PVALUE, field("cutoff", "P-value / FDR threshold", "number", default=0.05),
                           field("effect_cutoff", "Absolute effect threshold", "number", default=0.1),
                           field("point_size", "Point area (points²)", "number", default=40)]},
    "qq": {"label": "P-value QQ plot", "category": "Results", "function": "qqplot", "fields": [PVALUE]},
    "effects": {"label": "Distributions + effects", "category": "Results", "function": "datapoints_effect_panels_column",
                "fields": [GROUP, REF, TARGET, EFFECT, PVALUE,
                           field("distribution", "Distribution", "choice", ["box", "violin", "bar"], "box")]},
    "forest": {"label": "Forest plot", "category": "Results", "function": "forest",
               "fields": [EFFECT, field("ci_low", "Lower confidence bound", "table", required=True),
                          field("ci_high", "Upper confidence bound", "table", required=True)]},
    "export": {"label": "Export selected dataset", "category": "Data", "function": None, "fields": []},
}


def validate_request(payload):
    if not isinstance(payload, dict) or payload.get("operation") not in CATALOG:
        raise ValueError("Choose a supported analysis.")
    spec = CATALOG[payload["operation"]]
    allowed = {"operation", "features", "matrix", "filter_column", "filter_values", "numeric_columns",
               "categorical_columns", "source_job", "title", "palette", "yscale", "feature_label_column"}
    allowed.update(f["key"] for f in spec["fields"])
    if set(payload) - allowed:
        raise ValueError("Unrecognized analysis parameters.")
    for key in ("features", "filter_values", "numeric_columns", "categorical_columns", "predictors"):
        if key in payload and (not isinstance(payload[key], list) or
                               any(not isinstance(v, str) for v in payload[key])):
            raise ValueError(f"{key} must be a list of names.")
    for key, value in payload.items():
        if key not in {"features", "filter_values", "numeric_columns", "categorical_columns", "predictors"}:
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                raise ValueError(f"Invalid value for {key}.")
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f"{key} is too long.")
    if "feature_label_column" in payload and not isinstance(payload["feature_label_column"], str):
        raise ValueError("feature_label_column must be a column name.")
    for item in spec["fields"]:
        value = payload.get(item["key"], item["default"])
        if item["required"] and (value is None or value == "" or value == []):
            raise ValueError(f"Select {item['label']}.")
        if item["choices"] and value not in item["choices"]:
            raise ValueError(f"Invalid {item['label']}.")
        if item["kind"] == "number":
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{item['label']} must be numeric.") from exc
            if not math.isfinite(number):
                raise ValueError(f"{item['label']} must be finite.")
        payload[item["key"]] = value if value is not None else ""
    if payload.get("palette", "colorblind") not in {"colorblind", "deep", "Set2", "viridis"}:
        raise ValueError("Invalid palette.")
    if payload.get("yscale", "linear") not in {"linear", "log", "symlog"}:
        raise ValueError("Invalid y-axis scale.")
    return payload


def prepare_selection(adata, params, max_dense_bytes=256 * 1024**2):
    obs = adata.obs.copy()
    for column in params.get("numeric_columns", []):
        if column not in obs:
            raise ValueError(f"Unknown metadata column: {column}")
        obs[column] = pd.to_numeric(obs[column], errors="raise")
    for column in params.get("categorical_columns", []):
        if column not in obs:
            raise ValueError(f"Unknown metadata column: {column}")
        obs[column] = obs[column].astype("category")
    mask = np.ones(len(obs), dtype=bool)
    if params.get("filter_column"):
        column = params["filter_column"]
        if column not in obs or not params.get("filter_values"):
            raise ValueError("Choose a valid filter column and at least one value.")
        mask &= (obs[column].notna() & obs[column].astype(str).isin(params["filter_values"])).to_numpy()
    matrix = params.get("matrix", "X")
    var = adata.var
    x = adata.X
    if matrix == "raw":
        if adata.raw is None:
            raise ValueError("This dataset has no raw matrix.")
        x, var = adata.raw.X, adata.raw.var
    elif matrix.startswith("layer:"):
        if matrix[6:] not in adata.layers:
            raise ValueError("Unknown layer.")
        x = adata.layers[matrix[6:]]
    elif matrix != "X":
        raise ValueError("Unknown matrix source.")
    if x is None or x.dtype.kind not in "biuf":
        raise ValueError("The selected matrix must contain real numeric values.")
    features = params.get("features") or var.index.tolist()
    # Correlation/trajectory selectors may refer to features outside the checked subset.
    for key in ("x", "y"):
        name = params.get(key, "")
        if params["operation"] == "correlation" and name.startswith("var:"):
            features = list(dict.fromkeys(features + [name[4:]]))
        elif params["operation"] == "longitudinal" and key == "y":
            features = list(dict.fromkeys(features + [name]))
    if len(features) != len(set(features)) or not set(features).issubset(var.index):
        raise ValueError("Selected feature identifiers must be unique and present in the chosen matrix.")
    if not mask.any() or not features:
        raise ValueError("The selection has no observations or features.")
    estimate = int(mask.sum()) * len(features) * max(x.dtype.itemsize, 8)
    if estimate > max_dense_bytes:
        raise ValueError("Selection exceeds the dense-matrix budget. Select fewer features or observations.")
    # Materialize only the selected matrix; do not copy unused layers, raw, or obsm.
    selected = x[mask][:, var.index.get_indexer(features)].copy()
    work = ad.AnnData(selected, obs=obs.loc[mask].copy(), var=var.loc[features].copy())
    summary = {"n_obs": work.n_obs, "n_vars": work.n_vars, "matrix": matrix, "dense_bytes": estimate}
    if params.get("feature_label_column"):
        labels = feature_labels(var, params["feature_label_column"])
        summary.update(feature_label_column=params["feature_label_column"],
                       feature_labels={name: labels[name] for name in features})
    return work, summary


def group_value(series, value):
    values = [v for v in series.dropna().unique() if str(v) == value]
    if len(values) != 1:
        raise ValueError(f"Group value {value!r} is missing or ambiguous.")
    return values[0]


def run_analysis(adata, params, result_table=None, max_dense_bytes=256 * 1024**2):
    params = validate_request(dict(params))
    op = params["operation"]
    work, summary = prepare_selection(adata, params, max_dense_bytes)
    if CATALOG[op]["category"] in {"Explore", "Results"} and work.n_vars > 24 and op not in {"volcano", "qq", "correlation", "composition", "longitudinal"}:
        raise ValueError("Select at most 24 features for this plot.")
    title = params.get("title") or CATALOG[op]["label"]
    palette = params.get("palette", "colorblind")
    group = params.get("group") or None
    if group and group not in work.obs:
        raise ValueError("Unknown grouping column.")
    for item in CATALOG[op]["fields"]:
        if item["kind"] == "obs" and params.get(item["key"]) and params[item["key"]] not in work.obs:
            raise ValueError(f"Unknown column for {item['label']}.")
    pair = params.get("pair") or None
    if params.get("reference") and params.get("target"):
        ref = group_value(work.obs[group], params["reference"])
        target = group_value(work.obs[group], params["target"])
        if ref == target:
            raise ValueError("Reference and target groups must differ.")
        summary["groups"] = {"reference": str(ref), "reference_n": int((work.obs[group] == ref).sum()),
                             "target": str(target), "target_n": int((work.obs[group] == target).sum())}
        if op == "paired" or (op == "diff_test" and params["test"] in {"ttest_rel", "WilcoxonSigned"}):
            if not pair:
                raise ValueError("A pairing column is required for paired analyses.")
            paired_obs = work.obs.loc[work.obs[group].isin([ref, target])]
            if paired_obs[pair].isna().any() or paired_obs.duplicated([group, pair]).any():
                raise ValueError("Paired analyses require nonmissing, unique subject IDs within each group.")
            a = set(paired_obs.loc[paired_obs[group] == ref, pair])
            b = set(paired_obs.loc[paired_obs[group] == target, pair])
            summary["pairs"] = {"complete": len(a & b), "unmatched": len(a ^ b)}
            if not a & b:
                raise ValueError("There are no complete subject pairs.")
    table = result_table.loc[result_table.index.isin(work.var_names)].copy() if result_table is not None else work.var.copy()
    labels = summary.get("feature_labels", {})
    label_col = "__web_feature_label__"
    while label_col in table or label_col in work.var:
        label_col += "_"
    # Display annotations live only in plotting copies, never in statistical inputs.
    plot_work = work
    if labels and op in {"histogram", "paired", "effects"}:
        plot_work = work.copy()
        plot_work.var[label_col] = pd.Series(labels)
    fig = None
    output = None
    kwargs = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if op == "histogram":
            bins = float(params["bins"])
            if not bins.is_integer() or not 1 <= bins <= 200:
                raise ValueError("Bins must be an integer between 1 and 200.")
            kwargs = dict(var_names=work.var_names.tolist(), subset_obs_key=group, bins=int(bins),
                          palette=palette, title=title, show=False)
            if labels:
                kwargs["subplot_title_var_col"] = label_col
            fig, axes = pl.adata_histograms(plot_work, **kwargs)
            if labels:
                for identifier, axis in axes.items():
                    axis.set_xlabel(labels[identifier])
        elif op == "datapoints":
            kwargs = dict(var_names=work.var_names.tolist(), x_by_obs_key=group, subset_obs_key=group,
                          boxplot=params["distribution"] == "box", violinplot=params["distribution"] == "violin",
                          palette=palette, title=title, yscale=params.get("yscale", "linear"), show=False)
            if labels:
                kwargs["feature_labels"] = labels
            fig, _, output = pl.datapoints(work, **kwargs)
        elif op == "paired":
            kwargs = dict(var_names=work.var_names.tolist(), groupby_key=group,
                          groupby_key_ref_value=ref, groupby_key_target_value=target, pair_by_key=pair,
                          show_paired_difference=params["difference"] != "none",
                          paired_difference_mode="log2fc" if params["difference"] == "log2fc" else "difference",
                          palette=palette, title=title, show=False)
            if labels:
                kwargs["subplot_title_var_col"] = label_col
            fig, _, output = pl.paired_datapoints(plot_work, **kwargs)
        elif op == "correlation":
            frame = pd.DataFrame(index=work.obs_names)
            for axis in ("x", "y"):
                name = params[axis]
                if name.startswith("obs:") and name[4:] in work.obs:
                    values = pd.to_numeric(work.obs[name[4:]], errors="raise").to_numpy()
                elif name.startswith("var:") and name[4:] in work.var_names:
                    values = work[:, [name[4:]]].X
                    values = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
                else:
                    raise ValueError("Choose a feature or numeric observation column for each axis.")
                frame[axis] = values
            if group:
                frame["group"] = work.obs[group]
            kwargs = dict(column_key_x="x", column_key_y="y", hue="group" if group else None,
                          method=params["method"], xlabel=labels.get(params["x"][4:], params["x"][4:]) if params["x"].startswith("var:") else params["x"][4:],
                          ylabel=labels.get(params["y"][4:], params["y"][4:]) if params["y"].startswith("var:") else params["y"][4:],
                          axes_title=title, figsize=(8, 6), palette=palette, show=False)
            rendered = pl.corr_dotplot(frame, **kwargs)
            fig = rendered[0] if isinstance(rendered, tuple) else rendered.figure
            output = frame
        elif op == "longitudinal":
            frame = work.obs.copy()
            values = work[:, [params["y"]]].X
            response_col = "__web_response__"
            while response_col in frame:
                response_col += "_"
            frame[response_col] = values.toarray().ravel() if sparse.issparse(values) else np.asarray(values).ravel()
            order = [group_value(frame[params["x"]], v.strip()) for v in params["order"].split(",")]
            kwargs = dict(x=params["x"], y=response_col, subject=pair, x_order=order,
                          line_color_by=group, point_color_by=group, ylabel=labels.get(params["y"], params["y"]), title=title, show=False)
            rendered = pl.longitudinal_trajectories(frame, **kwargs)
            fig, output = rendered[0], rendered[-1] if isinstance(rendered[-1], pd.DataFrame) else frame
        elif op == "composition":
            kwargs = dict(x=params["x"], category=group, normalize=False if params["normalize"] == "count" else params["normalize"],
                          title=title, show=False)
            rendered = pl.category_composition(work.obs, **kwargs)
            fig = rendered[0]
            output = rendered[-1] if isinstance(rendered[-1], pd.DataFrame) else None
        elif op == "diff_test":
            kwargs = dict(groupby_key=group, groupby_key_ref_values=[ref], groupby_key_target_values=[target],
                          tests=[params["test"]], pair_by_key=pair, save_log=False, log_inputs=False)
            logger = logging.getLogger("adata_web_diff_test")
            logger.handlers = [logging.NullHandler()]
            logger.propagate = False
            output = tl.diff_test(work, logger=logger, **kwargs)
            summary["multiple_testing"] = "FDR is calculated by diff_test over its tested feature set; all-zero features may be excluded."
        elif op in {"ols", "mixedlm"}:
            predictors = params["predictors"]
            if not set(predictors).issubset(work.obs):
                raise ValueError("Unknown predictor column.")
            names = work.var_names.tolist() + predictors + ([group] if group else [])
            if any(not name.isprintable() or '"' in name or "\\" in name for name in names):
                raise ValueError('Regression names cannot contain double quotes, backslashes, or control characters.')
            if set(work.var_names) & set(work.obs):
                raise ValueError("For regression, feature names must not overlap observation column names.")
            kwargs = dict(feature_columns=work.var_names.tolist(), predictors=predictors, model_name="web_model", threads=1)
            if op == "mixedlm":
                kwargs.update(group=group, reml=params["reml"] == "REML")
            output = getattr(tl, CATALOG[op]["function"])(work, **kwargs)
            summary["model"] = {"predictors": predictors, "group": group,
                                "note": "Package complete-case handling and numeric predictor coercion are preserved."}
        elif op == "average":
            work.obs[group] = work.obs[group].astype("category")
            kwargs = dict(groupby_key=group)
            output = tl.average_feature_expression(work, **kwargs)
        elif op in {"volcano", "qq", "effects", "forest"}:
            for f in CATALOG[op]["fields"]:
                if f["kind"] == "table" and params[f["key"]] not in table:
                    raise ValueError(f"Choose a valid result column for {f['label']}.")
                if f["kind"] == "table":
                    table[params[f["key"]]] = pd.to_numeric(table[params[f["key"]]], errors="raise")
            if "pvalue" in params:
                valid_p = table[params["pvalue"]].dropna()
                if ((valid_p < 0) | (valid_p > 1)).any():
                    raise ValueError("P-value / FDR values must be between zero and one (or missing).")
            plot_table = table.copy()
            if labels:
                plot_table[label_col] = pd.Series(labels)
            if op == "volcano":
                cutoff = float(params["cutoff"])
                effect_cutoff = float(params["effect_cutoff"])
                point_size = float(params["point_size"])
                if not 0 < cutoff <= 1 or effect_cutoff < 0:
                    raise ValueError("P-value threshold must be in (0, 1]; effect threshold must be nonnegative.")
                if not 1 <= point_size <= 200 or table.empty:
                    raise ValueError("Choose a point area between 1 and 200 and at least one result feature.")
                kwargs = dict(l2fc_col=params["effect"], pvalue_col=params["pvalue"],
                              set_xlabel=params["effect"], set_ylabel=f"-log10({params['pvalue']})",
                              pvalue_threshold=cutoff, log2FoldChange_threshold=effect_cutoff,
                              dot_size_shrink_factor=len(table) / point_size,
                              legend_bbox_to_anchor=(1.45, 1), comparison_label="",
                              title_text=title, figsize=(8, 6))
                if labels:
                    kwargs.update(feature_label_col=label_col, label_top_features=True,
                                  label_layout="ranked_columns", label_features_char_limit=None)
                fig = pl.volcano_plot_generic(plot_table, **kwargs).figure
            elif op == "qq":
                kwargs = dict(pvalue_column=params["pvalue"], title=title, show=False)
                rendered = pl.qqplot(table, **kwargs)
                fig = rendered["fig"]
            elif op == "effects":
                if not set(work.var_names).issubset(table.index):
                    raise ValueError("Some selected features have no statistical results. Select features present in the result table.")
                plot_work.var = plot_table.loc[work.var_names].copy()
                kwargs = dict(feature_list=work.var_names.tolist(), comparison_col=group,
                              comparison_order=[ref, target], effect_column=params["effect"],
                              pvalue_column=params["pvalue"], distribution_kind=params["distribution"], fig_title=title)
                if labels:
                    kwargs["feature_label_vars_col"] = label_col
                rendered = pl.datapoints_effect_panels_column(plot_work, **kwargs)
                fig = rendered[0]
            else:
                kwargs = dict(feature_list=work.var_names.tolist(), estimate_col=params["effect"],
                              ci_low_col=params["ci_low"], ci_high_col=params["ci_high"], show=False)
                if labels:
                    kwargs.update(feature_label_col=label_col, feature_label_char_limit=None)
                rendered = pl.forest(var_df=plot_table, **kwargs)
                fig = rendered[0]
            output = table
        elif op == "export":
            pass
        warning_messages = list(dict.fromkeys(str(w.message)[:500] for w in caught))[:40]
    return {"figure": fig, "table": output, "selected": work if op == "export" else None,
            "summary": summary, "warnings": warning_messages,
            "function": CATALOG[op]["function"], "kwargs": kwargs}

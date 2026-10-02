"""Prebuilt workflows composed from the existing analysis adapters."""

from .analysis import GROUP, PAIR, REF, TARGET, field, validate_request


PIPELINES = {
    "explore": {
        "label": "Explore groups",
        "description": "Inspect distributions and compare descriptive summaries across groups.",
        "steps": ["Histograms", "Grouped datapoints", "Grouped feature means"],
        "fields": [GROUP],
    },
    "independent": {
        "label": "Compare independent groups",
        "description": "Compare two groups of independent samples using a test you choose.",
        "steps": ["Grouped datapoints", "Grouped feature means", "Differential testing"],
        "fields": [GROUP, REF, TARGET, field("test", "Test", "choice",
                   ["ttest_ind", "mannwhitneyu"], "ttest_ind")],
    },
    "paired": {
        "label": "Compare paired samples",
        "description": "Visualize matched samples and test a within-subject comparison.",
        "steps": ["Paired datapoints", "Grouped feature means", "Differential testing"],
        "fields": [GROUP, REF, TARGET, PAIR, field("test", "Test", "choice",
                   ["ttest_rel", "WilcoxonSigned"], "ttest_rel")],
    },
}


def build_pipeline(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("pipeline"), str) or payload["pipeline"] not in PIPELINES:
        raise ValueError("Choose a supported pipeline.")
    spec = PIPELINES[payload["pipeline"]]
    selection_keys = {"matrix", "features", "filter_column", "filter_values", "numeric_columns", "categorical_columns"}
    field_keys = {f["key"] for f in spec["fields"]}
    if set(payload) - (selection_keys | field_keys | {"pipeline"}):
        raise ValueError("Unrecognized pipeline parameters.")
    if not isinstance(payload.get("features"), list) or not payload["features"]:
        raise ValueError("Select at least one feature in the analysis studio.")
    for item in spec["fields"]:
        value = payload.get(item["key"], item["default"])
        if item["required"] and not value:
            raise ValueError(f"Select {item['label']}.")
        if item["choices"] and value not in item["choices"]:
            raise ValueError(f"Invalid {item['label']} for this pipeline.")
    if "reference" in field_keys and payload["reference"] == payload["target"]:
        raise ValueError("Reference and target groups must differ.")
    selection = {key: payload[key] for key in selection_keys if key in payload}
    group = {"group": payload["group"]}
    plot_selection = dict(selection, features=payload["features"][:24])
    if payload["pipeline"] == "explore":
        steps = [{**plot_selection, **group, "operation": "histogram"},
                 {**plot_selection, **group, "operation": "datapoints"},
                 {**selection, **group, "operation": "average"}]
    else:
        comparison = {key: payload[key] for key in ("reference", "target")}
        paired = payload["pipeline"] == "paired"
        if paired:
            comparison["pair"] = payload["pair"]
        plot = {**plot_selection, **group, "operation": "paired" if paired else "datapoints"}
        if paired:
            plot.update(comparison)
        steps = [plot, {**selection, **group, "operation": "average"},
                 {**selection, **group, **comparison, "operation": "diff_test",
                  "test": payload.get("test", spec["fields"][-1]["default"])}]
    return [validate_request(step) for step in steps]

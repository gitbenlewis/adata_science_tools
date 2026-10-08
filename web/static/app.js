"use strict";
const $ = (id) => document.getElementById(id);
const csrf = document.querySelector('meta[name="csrf-token"]').content;
let state = {catalog: {}, datasets: []};
let activeId = null;
let detail = null;
let selectedFeatures = new Set();
let previewName = "obs";
let pollBusy = false;
let lastJobs = "";
let pipelineName = "explore";
let pipelineFormKey = null;
let selectionVersion = 0;
let datasetLoading = false;
let featureLabels = new Map();
let labelRequestVersion = 0;
let labelsLoading = false;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = `alert${error ? " error" : ""}`;
  $("notice").hidden = !message;
}
async function api(url, options = {}) {
  const response = await fetch(url, {...options, headers: {"X-CSRF-Token": csrf, ...(options.headers || {})}});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Request failed. Reload and try again.");
  return result;
}
function options(select, values, empty = true, previous = undefined) {
  const current = previous === undefined ? select.value : previous;
  select.replaceChildren();
  if (empty) select.add(new Option("Select…", ""));
  for (const value of values) {
    const pair = Array.isArray(value) ? value : [value, value];
    select.add(new Option(pair[1], pair[0]));
  }
  if ([...select.options].some((o) => o.value === current)) select.value = current;
}
function showView(name) {
  for (const view of ["tutorial", "data", "analysis", "results", "pipelines"]) {
    $(`view-${view}`).hidden = view !== name;
    $(`nav-${view}`).classList.toggle("active", view === name);
  }
  if (name === "pipelines") renderPipeline();
  window.scrollTo({top: 0, behavior: "smooth"});
}
for (const view of ["tutorial", "data", "analysis", "results", "pipelines"]) $(`nav-${view}`).onclick = () => showView(view);
$("tutorial-start").onclick = () => showView("data");
$("start-analysis").onclick = () => showView("analysis");
for (const radio of document.querySelectorAll('input[name="format"]')) radio.onchange = () => {
  $("h5ad-input").hidden = radio.value !== "h5ad";
  $("csv-input").hidden = radio.value !== "csv";
};

function table(node, data) {
  node.replaceChildren();
  if (!data || !data.data.length) { node.append(element("p", "No rows to display.", "small muted")); return; }
  const t = document.createElement("table");
  const head = t.createTHead().insertRow();
  for (const name of ["ID", ...data.columns]) head.append(element("th", name));
  const body = t.createTBody();
  data.data.forEach((row, i) => {
    const tr = body.insertRow();
    for (const value of [data.index[i], ...row]) {
      let text = value === null ? "—" : typeof value === "number" && !Number.isInteger(value) ? Number(value.toPrecision(5)).toString() : String(value);
      tr.append(element("td", text));
    }
  });
  node.append(t);
}
function renderOverview() {
  const dataset = detail.dataset;
  renderCovidPresets();
  const ready = dataset.status === "ready";
  $("dataset-overview").hidden = !ready;
  $("analysis-form").hidden = !ready;
  document.querySelector(".needs-data").hidden = ready;
  $("dataset-status").textContent = `${dataset.name} · ${dataset.status}`;
  $("import-progress").hidden = ready;
  if (!ready) {
    const job = detail.jobs.find((j) => j.request.kind === "import");
    $("import-progress").textContent = job?.error || `Dataset ${dataset.status}. Validation runs in the analysis worker.`;
    const del = element("button", "Delete dataset", "quiet danger");
    del.type = "button";
    del.onclick = deleteDataset;
    $("import-progress").append(del);
    return;
  }
  const meta = dataset.metadata;
  $("dataset-title").textContent = dataset.name;
  $("metrics").replaceChildren();
  for (const [value, label] of [[meta.n_obs.toLocaleString(), "Observations"], [meta.n_vars.toLocaleString(), "Features"], [meta.layers.length, "Additional layers"], [meta.sparse ? "Sparse" : "Dense", "Matrix storage"]]) {
    const card = element("div", undefined, "metric");
    card.append(element("strong", value), element("span", label));
    $("metrics").append(card);
  }
  table($("preview-table"), meta.previews[previewName]);
}
for (const button of document.querySelectorAll("[data-preview]")) button.onclick = () => {
  previewName = button.dataset.preview;
  document.querySelectorAll("[data-preview]").forEach((b) => b.classList.toggle("active", b === button));
  if (detail) renderOverview();
};
function features() {
  if (!detail || detail.dataset.status !== "ready") return [];
  return $("matrix").value === "raw" ? detail.dataset.metadata.raw_features : detail.dataset.metadata.features;
}
function featureText(id) {
  const label = featureLabels.get(id) || id;
  return label === id || label.endsWith(`[${id}]`) ? label : `${label} — ${id}`;
}
function matchingFeatures() {
  const query = $("feature-search").value.toLowerCase();
  return features().filter((id) => id.toLowerCase().includes(query) || (featureLabels.get(id) || "").toLowerCase().includes(query));
}
function relabelMethodFeatures() {
  for (const field of state.catalog[$("operation").value]?.fields || []) {
    const input = $(`param-${field.key}`);
    if (field.kind === "feature") options(input, features().map((id) => [id, featureText(id)]));
    if (field.kind === "variable") options(input, [...obsColumns().map((v) => [`obs:${v}`, `obs · ${v}`]), ...features().map((id) => [`var:${id}`, `feature · ${featureText(id)}`])]);
  }
}
async function loadFeatureLabels(column = "", discover = false) {
  const version = ++labelRequestVersion;
  const datasetVersion = selectionVersion;
  const id = activeId;
  const matrix = $("matrix").value;
  const current = () => version === labelRequestVersion && datasetVersion === selectionVersion && id === activeId;
  labelsLoading = true;
  $("feature-label-column").disabled = true;
  $("features-matches").disabled = true;
  featureLabels = new Map();
  $("feature-label-status").textContent = "Loading feature labels…";
  for (const button of ["run-button", "pipeline-run"]) $(button).disabled = true;
  renderFeatures();
  relabelMethodFeatures();
  try {
    const url = `/api/datasets/${id}/feature-labels?matrix=${encodeURIComponent(matrix)}`;
    let result;
    let unavailable = false;
    if (discover) {
      result = await api(url);
      if (!current()) return;
      unavailable = Boolean(column && !result.columns.includes(column));
      if (unavailable) column = "";
      options($("feature-label-column"), [["", "Feature ID (var_names)"], ...result.columns], false, column);
    }
    if (!discover || column) result = await api(`${url}&column=${encodeURIComponent(column)}`);
    if (!current()) return;
    featureLabels = new Map(Object.entries(result.labels));
    $("feature-label-column").value = column;
    $("feature-label-status").textContent = unavailable
      ? "The previous label column is unavailable in this matrix. Showing feature IDs."
      : column ? "Duplicate names include IDs. Missing names use IDs. Each feature remains separate." : "Search by feature ID, or choose an annotation column for recognizable names.";
  } catch (error) {
    if (!current()) return;
    $("feature-label-column").value = "";
    $("feature-label-status").textContent = "Labels could not be loaded. Showing feature IDs.";
    notice(error.message, true);
  } finally {
    if (current()) {
      labelsLoading = false;
      $("feature-label-column").disabled = false;
      $("features-matches").disabled = false;
      renderFeatures();
      relabelMethodFeatures();
      renderPipeline();
      for (const button of ["run-button", "pipeline-run"]) $(button).disabled = datasetLoading || labelsLoading || detail?.dataset.status !== "ready";
    }
  }
}
$("feature-label-column").onchange = () => loadFeatureLabels($("feature-label-column").value);
function renderFeatures() {
  const matches = matchingFeatures();
  $("feature-count").textContent = `${selectedFeatures.size} selected / ${features().length}`;
  $("feature-list").replaceChildren();
  for (const feature of matches.slice(0, 150)) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = selectedFeatures.has(feature);
    input.onchange = () => { input.checked ? selectedFeatures.add(feature) : selectedFeatures.delete(feature); $("feature-count").textContent = `${selectedFeatures.size} selected / ${features().length}`; };
    label.append(input, document.createTextNode(featureText(feature)));
    $("feature-list").append(label);
  }
  if (matches.length > 150) $("feature-list").append(element("p", "Showing 150 matches. Search to narrow the list.", "small muted"));
}
$("feature-search").oninput = renderFeatures;
$("features-matches").onclick = () => { for (const id of matchingFeatures()) selectedFeatures.add(id); renderFeatures(); };
$("features-all").onclick = () => { selectedFeatures = new Set(features()); renderFeatures(); };
$("features-clear").onclick = () => { selectedFeatures.clear(); renderFeatures(); };
$("matrix").onchange = () => {
  const available = new Set(features());
  selectedFeatures = new Set([...selectedFeatures].filter((id) => available.has(id)));
  renderMethod();
  loadFeatureLabels($("feature-label-column").value, true);
};
function obsColumns() { return Object.keys(detail.dataset.metadata.obs_columns); }
function refreshFilter() {
  const col = detail.dataset.metadata.obs_columns[$("filter-column").value];
  options($("filter-values"), col ? col.values : [], false);
}
$("filter-column").onchange = refreshFilter;
function refreshLevels() {
  const group = $("param-group")?.value;
  const col = detail.dataset.metadata.obs_columns[group];
  for (const key of ["reference", "target"]) if ($(`param-${key}`)) options($(`param-${key}`), col ? col.values : []);
}
function tableColumns() {
  const source = detail.jobs.find((j) => j.id === $("source-job").value);
  return source ? source.result.columns || [] : detail.dataset.metadata.var_columns;
}
function renderMethod() {
  if (!detail || detail.dataset.status !== "ready") return;
  const spec = state.catalog[$("operation").value];
  if (!spec) return;
  $("method-description").textContent = spec.function ? `Uses ${spec.function}() from this package.` : "Download the selected matrix and annotations as a new H5AD file. The selected matrix becomes X; unused layers and embeddings are omitted.";
  $("source-label").hidden = spec.category !== "Results";
  $("appearance").hidden = !["Explore", "Results"].includes(spec.category);
  $("yscale-label").hidden = $("operation").value !== "datapoints";
  const sources = detail.jobs.filter((j) => j.status === "complete" && ["diff_test", "ols", "mixedlm"].includes(j.request.parameters?.operation) && j.result.files?.includes("results.csv"));
  options($("source-job"), [["", "Feature metadata (adata.var)"], ...sources.map((j) => [j.id, `${state.catalog[j.request.parameters.operation].label} · ${new Date(j.created * 1000).toLocaleTimeString()}`])], false);
  $("method-fields").replaceChildren();
  for (const field of spec.fields) {
    const label = element("label", field.label);
    const input = document.createElement(["text", "number"].includes(field.kind) ? "input" : "select");
    input.id = `param-${field.key}`;
    input.name = field.key;
    input.required = field.required;
    if (input.tagName === "INPUT") {
      input.type = field.kind;
      if (field.kind === "number") input.step = "any";
      input.value = field.default ?? "";
      input.maxLength = 1000;
    } else {
      let values = [];
      if (["obs", "multi_obs"].includes(field.kind)) values = obsColumns();
      if (field.kind === "feature") values = features().map((id) => [id, featureText(id)]);
      if (field.kind === "variable") values = [...obsColumns().map((v) => [`obs:${v}`, `obs · ${v}`]), ...features().map((v) => [`var:${v}`, `feature · ${featureText(v)}`])];
      if (field.kind === "choice") {
        const testLabels = {ttest_ind: "Welch's independent t-test", mannwhitneyu: "Mann–Whitney U test",
                            ttest_rel: "Paired t-test", WilcoxonSigned: "Wilcoxon signed-rank test"};
        values = field.choices.map((value) => [value, testLabels[value] || value]);
      }
      if (field.kind === "table") values = tableColumns();
      if (field.kind === "multi_obs") { input.multiple = true; input.size = 4; }
      options(input, values, field.kind !== "choice" && field.kind !== "multi_obs", field.default ?? "");
    }
    label.append(input);
    $("method-fields").append(label);
  }
  if ($("param-group")) $("param-group").onchange = refreshLevels;
  refreshLevels();
}
$("operation").onchange = renderMethod;
$("source-job").onchange = () => {
  for (const field of state.catalog[$("operation").value].fields) if (field.kind === "table") options($(`param-${field.key}`), tableColumns());
};
function setupStudio() {
  if (detail.dataset.status !== "ready") return;
  const meta = detail.dataset.metadata;
  options($("matrix"), [["X", "X · primary matrix"], ...meta.layers.map((v) => [`layer:${v}`, `Layer · ${v}`]), ...(meta.raw ? [["raw", "Raw matrix"]] : [])], false, "X");
  selectedFeatures = new Set(features().slice(0, 3));
  $("feature-search").value = "";
  renderFeatures();
  for (const id of ["filter-column", "numeric-columns", "categorical-columns"]) options($(id), obsColumns(), id === "filter-column", "");
  refreshFilter();
  $("operation").replaceChildren();
  for (const category of ["Explore", "Statistics", "Results", "Data"]) {
    const group = document.createElement("optgroup");
    group.label = category;
    for (const [key, spec] of Object.entries(state.catalog)) if (spec.category === category) group.append(new Option(spec.label, key));
    $("operation").append(group);
  }
  $("operation").value = "histogram";
  renderMethod();
  loadFeatureLabels("", true);
}
function renderResults() {
  const jobs = detail.jobs.filter((j) => j.request.kind === "analysis");
  const fingerprint = JSON.stringify(jobs);
  if (fingerprint === lastJobs) return;
  lastJobs = fingerprint;
  $("results-list").replaceChildren();
  if (!jobs.length) $("results-list").append(element("p", "Run an analysis to see results here.", "empty-state"));
  for (const job of jobs) {
    const card = element("article", undefined, "card result-card");
    const header = element("div", undefined, "result-heading");
    const name = element("div");
    name.append(element("h2", state.catalog[job.request.parameters.operation].label), element("p", new Date(job.created * 1000).toLocaleString(), "small muted"));
    if (job.request.pipeline) {
      const pipeline = job.request.pipeline;
      name.append(element("p", `${pipeline.name} · Step ${pipeline.step}/${pipeline.total} · Run ${pipeline.id.slice(0, 8)}`, "small muted"));
    }
    header.append(name, element("span", job.status, `badge ${job.status}`));
    card.append(header);
    if (job.error) card.append(element("p", job.error, "alert error"));
    if (["queued", "running"].includes(job.status)) card.append(element("p", job.status === "queued" ? "Waiting for the analysis worker…" : "Processing your selected data…", "muted"));
    const files = job.result.files || [];
    const url = (name) => `/api/jobs/${job.id}/files/${encodeURIComponent(name)}`;
    if (files.includes("figure.png")) {
      const image = document.createElement("img");
      image.src = url("figure.png");
      image.alt = `${state.catalog[job.request.parameters.operation].label} for ${detail.dataset.name}`;
      image.className = "result-image";
      image.loading = "lazy";
      card.append(image);
    }
    if (job.result.preview) { const wrapper = element("div", undefined, "table-wrap"); table(wrapper, job.result.preview); card.append(wrapper); }
    const links = element("div", undefined, "downloads");
    for (const file of files) {
      const link = element("a", file === "analysis.json" ? "Analysis record" : file === "reproduce.py" ? "Reproduce in Python" : file.toUpperCase());
      link.href = url(file);
      link.download = file;
      links.append(link);
    }
    card.append(links);
    if (job.status === "complete" && ["diff_test", "ols", "mixedlm"].includes(job.request.parameters.operation)) {
      const button = element("button", "Plot these results →", "secondary");
      button.type = "button";
      button.onclick = () => { $("operation").value = "volcano"; renderMethod(); $("source-job").value = job.id; $("source-job").onchange(); showView("analysis"); };
      card.append(button);
    }
    for (const warning of job.result.warnings || []) card.append(element("p", warning, "small muted"));
    const settings = document.createElement("details");
    settings.append(element("summary", "Run settings & selection"), element("pre", JSON.stringify({parameters: job.request.parameters, selection: job.result.summary}, null, 2)));
    card.append(settings);
    $("results-list").append(card);
  }
}
async function selectDataset(id) {
  const version = ++selectionVersion;
  ++labelRequestVersion;
  labelsLoading = false;
  datasetLoading = true;
  for (const button of ["run-button", "pipeline-run", "delete-button"]) $(button).disabled = true;
  $("dataset-status").textContent = "Loading selected dataset…";
  try {
    const nextDetail = id ? await api(`/api/datasets/${id}`) : null;
    if (version !== selectionVersion) return;
    // Commit the target and its data together; earlier requests cannot overwrite them.
    activeId = id;
    detail = nextDetail;
    $("dataset-select").value = id || "";
    lastJobs = "";
    featureLabels = new Map();
    if (detail) { renderOverview(); setupStudio(); renderResults(); }
    else {
      $("covid-presets").hidden = true;
      $("dataset-overview").hidden = true; $("analysis-form").hidden = true; $("import-progress").hidden = true;
      document.querySelector(".needs-data").hidden = false;
      $("dataset-status").textContent = "Load your data or start with the demo.";
      $("results-list").replaceChildren(element("p", "Run an analysis to see results here.", "empty-state"));
    }
    renderPipeline(); renderCovidPresets();
  } catch (error) {
    if (version !== selectionVersion) return;
    if (detail?.dataset.status === "ready") loadFeatureLabels($("feature-label-column").value, true);
    $("dataset-select").value = activeId || "";
    $("dataset-status").textContent = detail ? `${detail.dataset.name} · ${detail.dataset.status}` : "No dataset loaded.";
    throw error;
  } finally {
    if (version === selectionVersion) {
      datasetLoading = false;
      for (const button of ["run-button", "pipeline-run"]) $(button).disabled = labelsLoading || detail?.dataset.status !== "ready";
      $("delete-button").disabled = !activeId;
      renderCovidPresets();
    }
  }
}
$("dataset-select").onchange = () => selectDataset($("dataset-select").value).catch((e) => notice(e.message, true));
async function refresh() {
  if (pollBusy || datasetLoading) return;
  pollBusy = true;
  const version = selectionVersion;
  try {
    const nextState = await api("/api/state");
    if (version !== selectionVersion || datasetLoading) return;
    state = nextState;
    if (!$("pipeline-cards").childElementCount) renderPipeline();
    $("worker-notice").hidden = state.worker_running;
    options($("dataset-select"), state.datasets.map((d) => [d.id, d.name]), !state.datasets.length, activeId || "");
    if (!activeId && state.datasets.length) await selectDataset(state.datasets[0].id);
    else if (activeId && !state.datasets.some((d) => d.id === activeId)) await selectDataset(state.datasets[0]?.id || null);
    else if (activeId) {
      const wasReady = detail?.dataset.status === "ready";
      const nextDetail = await api(`/api/datasets/${activeId}`);
      if (version !== selectionVersion || datasetLoading) return;
      detail = nextDetail;
      renderOverview();
      if (!wasReady && detail.dataset.status === "ready") {
        setupStudio(); renderPipeline();
        for (const button of ["run-button", "pipeline-run"]) $(button).disabled = labelsLoading;
        notice("Dataset loaded. Open the analysis studio to begin.");
      }
      renderResults();
    }
  } catch (error) { if (version === selectionVersion && !datasetLoading) notice(error.message, true); }
  finally { pollBusy = false; }
}
async function loadDataset(form) {
  $("load-button").disabled = true; $("demo-button").disabled = true; $("covid-button").disabled = true;
  try {
    const result = await api("/api/datasets", {method: "POST", body: form});
    notice("Upload received. Validating your dataset…");
    await selectDataset(result.dataset_id);
    await refresh();
  } catch (error) { notice(error.message, true); }
  finally { $("load-button").disabled = false; $("demo-button").disabled = false; $("covid-button").disabled = false; }
}
$("upload-form").onsubmit = (event) => {
  event.preventDefault();
  const format = new FormData(event.target).get("format");
  const data = new FormData();
  data.set("format", format); data.set("name", event.target.elements.name.value);
  for (const key of format === "h5ad" ? ["h5ad"] : ["X", "obs", "var"]) {
    const file = event.target.elements[key].files[0];
    if (!file) { notice("Choose all required files before loading the dataset.", true); return; }
    data.set(key, file);
  }
  loadDataset(data);
};
$("covid-button").onclick = () => { const data = new FormData(); data.set("format", "covid"); loadDataset(data); };
function renderCovidPresets() {
  const ready = !datasetLoading && detail?.dataset.status === "ready" && detail.dataset.metadata.format === "covid";
  $("covid-presets").hidden = !ready;
  if (!ready || $("covid-preset-buttons").childElementCount) return;
  for (const [key, preset] of Object.entries(state.covid_presets || {})) {
    const button = element("button", preset.label, "secondary");
    button.type = "button";
    button.onclick = () => applyCovidPreset(key);
    $("covid-preset-buttons").append(button);
  }
}
async function applyCovidPreset(key) {
  if (datasetLoading || detail?.dataset.status !== "ready" || detail.dataset.metadata.format !== "covid") return;
  const preset = state.covid_presets[key];
  const params = preset.parameters;
  $("matrix").value = params.matrix;
  selectedFeatures = new Set(params.features);
  $("feature-search").value = "";
  renderFeatures();
  $("filter-column").value = params.filter_column;
  refreshFilter();
  for (const [id, values] of [["filter-values", params.filter_values], ["numeric-columns", params.numeric_columns], ["categorical-columns", params.categorical_columns]]) {
    for (const option of $(id).options) option.selected = values.includes(option.value);
  }
  if (preset.view === "pipelines") {
    pipelineName = params.pipeline;
    pipelineFormKey = null;
    renderPipeline();
    $("pipeline-group").value = params.group;
    $("pipeline-group").onchange();
  } else {
    $("operation").value = params.operation;
    renderMethod();
    for (const field of state.catalog[params.operation].fields) $(`param-${field.key}`).value = params[field.key] ?? field.default ?? "";
    refreshLevels();
    for (const key of ["title", "palette", "yscale"]) $("analysis-form").elements[key].value = params[key];
  }
  showView(preset.view);
  const version = selectionVersion;
  await loadFeatureLabels(params.feature_label_column || "", true);
  if (version !== selectionVersion) return;
  notice(`${preset.label}: settings filled. Review the selection, then choose Run. Open Datasets to choose another COVID example.`);
}
$("demo-button").onclick = () => { const data = new FormData(); data.set("format", "demo"); loadDataset(data); };
async function deleteDataset() {
  if (datasetLoading || !activeId || !window.confirm("Delete this dataset and all its results from the app?")) return;
  try { await api(`/api/datasets/${activeId}`, {method: "DELETE"}); await selectDataset(null); await refresh(); notice("Dataset deleted."); }
  catch (error) { notice(error.message, true); }
}
$("delete-button").onclick = deleteDataset;
$("analysis-form").onsubmit = async (event) => {
  event.preventDefault();
  if (datasetLoading || labelsLoading || detail?.dataset.status !== "ready") { notice("Wait for the selected dataset and feature labels to load.", true); return; }
  if (!selectedFeatures.size) { notice("Select at least one feature.", true); return; }
  const form = new FormData(event.target);
  const operation = $("operation").value;
  const payload = {operation, matrix: form.get("matrix"), features: [...selectedFeatures], feature_label_column: $("feature-label-column").value};
  for (const key of ["filter_column", "title", "palette", "yscale"]) payload[key] = form.get(key) || "";
  for (const key of ["filter_values", "numeric_columns", "categorical_columns"]) payload[key] = form.getAll(key);
  if (state.catalog[operation].category === "Results") payload.source_job = form.get("source_job") || "";
  for (const field of state.catalog[operation].fields) payload[field.key] = field.kind === "multi_obs" ? form.getAll(field.key) : form.get(field.key) || "";
  $("run-button").disabled = true;
  try {
    await api(`/api/datasets/${activeId}/jobs`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    notice("Analysis submitted. You can keep exploring while it runs."); showView("results"); await refresh();
  } catch (error) { notice(error.message, true); }
  finally { $("run-button").disabled = datasetLoading || labelsLoading || detail?.dataset.status !== "ready"; }
};
function renderPipeline() {
  const spec = state.pipelines?.[pipelineName];
  if (!spec) return;
  $("pipeline-cards").replaceChildren();
  for (const [key, pipeline] of Object.entries(state.pipelines)) {
    const button = element("button", undefined, `card pipeline-card${key === pipelineName ? " active" : ""}`);
    button.type = "button";
    button.setAttribute("aria-pressed", String(key === pipelineName));
    button.append(element("h3", pipeline.label), element("p", pipeline.description, "small muted"));
    button.onclick = () => { pipelineName = key; renderPipeline(); };
    $("pipeline-cards").append(button);
  }
  $("pipeline-title").textContent = spec.label;
  $("pipeline-description").textContent = spec.description;
  $("pipeline-steps").replaceChildren(...spec.steps.map((step) => element("li", step)));
  const ready = detail?.dataset.status === "ready";
  $("pipeline-form").hidden = !ready;
  $("pipeline-edit-selection").hidden = !ready;
  $("pipeline-selection").textContent = ready
    ? `${detail.dataset.name} · ${$("matrix").selectedOptions[0]?.textContent} · ${selectedFeatures.size} selected features. Feature labels: ${$("feature-label-column").value || "IDs"}. Observation filter: ${$("filter-column").value ? `${$("filter-column").value} = ${[...$("filter-values").selectedOptions].map((o) => o.value).join(", ") || "no values selected"}` : "none"}. Numeric metadata: ${[...$("numeric-columns").selectedOptions].map((o) => o.value).join(", ") || "none"}. Categorical metadata: ${[...$("categorical-columns").selectedOptions].map((o) => o.value).join(", ") || "none"}.`
    : "Load a dataset in Datasets to run a pipeline.";
  if (!ready) { pipelineFormKey = null; return; }
  const formKey = `${activeId}:${pipelineName}`;
  // Navigation updates the selection summary without resetting an existing form.
  if (pipelineFormKey === formKey) return;
  pipelineFormKey = formKey;
  $("pipeline-fields").replaceChildren();
  for (const field of spec.fields) {
    const label = element("label", field.label);
    const select = document.createElement("select");
    select.name = field.key;
    select.id = `pipeline-${field.key}`;
    select.required = field.required;
    const tests = {ttest_ind: "Welch's independent t-test", mannwhitneyu: "Mann–Whitney U test", ttest_rel: "Paired t-test", WilcoxonSigned: "Wilcoxon signed-rank test"};
    const values = field.kind === "choice" ? field.choices.map((v) => [v, tests[v] || v]) : field.kind === "obs" ? obsColumns() : [];
    options(select, values, field.kind !== "choice", field.default ?? "");
    label.append(select);
    $("pipeline-fields").append(label);
  }
  $("pipeline-group").onchange = () => {
    const values = detail.dataset.metadata.obs_columns[$("pipeline-group").value]?.values || [];
    for (const key of ["reference", "target"]) if ($(`pipeline-${key}`)) options($(`pipeline-${key}`), values);
  };
}
$("pipeline-edit-selection").onclick = () => showView("analysis");
$("pipeline-form").onsubmit = async (event) => {
  event.preventDefault();
  if (datasetLoading || labelsLoading || detail?.dataset.status !== "ready") { notice("Wait for the selected dataset and feature labels to load.", true); return; }
  if (!selectedFeatures.size) { notice("Select at least one feature in Analysis studio.", true); return; }
  const form = new FormData(event.target);
  const payload = {pipeline: pipelineName, features: [...selectedFeatures], matrix: $("matrix").value,
                   filter_column: $("filter-column").value, feature_label_column: $("feature-label-column").value};
  for (const [key, id] of [["filter_values", "filter-values"], ["numeric_columns", "numeric-columns"], ["categorical_columns", "categorical-columns"]]) {
    payload[key] = [...$(id).selectedOptions].map((o) => o.value);
  }
  for (const field of state.pipelines[pipelineName].fields) payload[field.key] = form.get(field.key) || "";
  $("pipeline-run").disabled = true;
  try {
    const result = await api(`/api/datasets/${activeId}/pipelines`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    notice(`Pipeline submitted: ${result.job_ids.length} steps. Results retain the shared selection and pipeline run ID.`);
    showView("results"); await refresh();
  } catch (error) { notice(error.message, true); }
  finally { $("pipeline-run").disabled = datasetLoading || labelsLoading || detail?.dataset.status !== "ready"; }
};
refresh();
window.setInterval(refresh, 3000);

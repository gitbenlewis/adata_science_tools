// Browser-state regressions without extra npm dependencies: node --test tests/web_state.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {test} = require('node:test');

function app() {
  const nodes = new Map();
  // Only the DOM operations used by navigation and pipeline controls are needed.
  class Node {
    constructor() {
      this.children = []; this.options = []; this.value = ''; this.content = 'csrf';
      this.classList = {toggle() {}};
    }
    set id(value) { this._id = value; nodes.set(value, this); }
    get id() { return this._id; }
    get childElementCount() { return this.children.length; }
    get selectedOptions() { return this.options.filter(option => option.value === this.value); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; this.options = []; this.value = ''; }
    add(option) { this.options.push(option); if (this.options.length === 1) this.value = option.value; }
    setAttribute() {}
  }
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, new Node());
    return nodes.get(id);
  };
  const context = vm.createContext({
    document: {getElementById: get, querySelector: get, querySelectorAll: () => [], createElement: () => new Node(), createTextNode: text => ({textContent:text})},
    window: {scrollTo() {}, setInterval() {}},
    Option: function(text, value) { this.textContent = text; this.value = value; },
  });
  const source = fs.readFileSync(path.join(__dirname, '../web/static/app.js'), 'utf8');
  // Start requests explicitly so each response order is controlled by the test.
  vm.runInContext(source.replace(/\nrefresh\(\);\nwindow\.setInterval\(refresh, 3000\);\s*$/, ''), context);
  const run = code => vm.runInContext(code, context);
  run('renderOverview = () => {}; setupStudio = () => {}; renderResults = () => {};');
  const pending = [];
  context.api = url => new Promise((resolve, reject) => pending.push({url, resolve, reject}));
  return {run, get, pending};
}
const dataset = (id, status = 'ready') => ({dataset: {id, name: id, status, metadata: {
  obs_columns: {condition: {values: ['A', 'B']}, subject: {values: ['one']}},
}}, jobs: []});
const state = {catalog: {}, datasets: [{id: 'A', name: 'A'}, {id: 'B', name: 'B'}], worker_running: true};

test('out-of-order switches commit only the latest dataset and block submissions while loading', async () => {
  const ui = app();
  const a = ui.run('selectDataset("A")');
  const b = ui.run('selectDataset("B")');
  assert.equal(ui.get('run-button').disabled, true);
  assert.equal(ui.get('pipeline-run').disabled, true);
  assert.equal(ui.get('delete-button').disabled, true);
  for (const form of ['analysis-form', 'pipeline-form']) await ui.get(form).onsubmit({preventDefault() {}});
  assert.equal(ui.pending.length, 2); // No POST was issued during loading.
  ui.pending[1].resolve(dataset('B')); await b;
  ui.pending[0].resolve(dataset('A')); await a;
  assert.equal(ui.run('activeId'), 'B');
  assert.equal(ui.run('detail.dataset.id'), 'B');
  assert.equal(ui.get('dataset-select').value, 'B');
  assert.equal(ui.get('run-button').disabled, false);
});

test('failed latest switch retains the previous dataset and ignores an older late response', async () => {
  const ui = app();
  ui.run('loadFeatureLabels = async () => {};');
  const first = ui.run('selectDataset("A")'); ui.pending[0].resolve(dataset('A')); await first;
  const b = ui.run('selectDataset("B")');
  const c = ui.run('selectDataset("C")');
  ui.pending[2].reject(new Error('Unavailable'));
  await assert.rejects(c, /Unavailable/);
  ui.pending[1].resolve(dataset('B')); await b;
  assert.equal(ui.run('activeId'), 'A');
  assert.equal(ui.run('detail.dataset.id'), 'A');
  assert.equal(ui.get('dataset-select').value, 'A');
  assert.equal(ui.get('run-button').disabled, false);
});

test('poll responses cannot overwrite a newer dataset switch', async () => {
  for (const phase of ['state', 'detail']) {
    const ui = app();
    const initial = ui.run('selectDataset("A")'); ui.pending[0].resolve(dataset('A')); await initial;
    const poll = ui.run('refresh()');
    if (phase === 'detail') {
      ui.pending[1].resolve(state);
      await new Promise(resolve => setImmediate(resolve));
    }
    const stale = ui.pending.at(-1);
    const switchDataset = ui.run('selectDataset("B")');
    ui.pending.at(-1).resolve(dataset('B')); await switchDataset;
    stale.resolve(phase === 'state' ? state : dataset('A')); await poll;
    assert.equal(ui.run('activeId'), 'B', phase);
    assert.equal(ui.run('detail.dataset.id'), 'B', phase);
    assert.equal(ui.get('dataset-select').value, 'B', phase);
  }
});

test('an import becoming ready re-enables the run controls', async () => {
  const ui = app();
  const initial = ui.run('selectDataset("A")'); ui.pending[0].resolve(dataset('A', 'queued')); await initial;
  assert.equal(ui.get('run-button').disabled, true);
  const poll = ui.run('refresh()'); ui.pending[1].resolve(state);
  await new Promise(resolve => setImmediate(resolve));
  ui.pending[2].resolve(dataset('A')); await poll;
  assert.equal(ui.get('run-button').disabled, false);
  assert.equal(ui.get('pipeline-run').disabled, false);
});

test('pipeline settings survive navigation, but do not leak into another dataset or workflow', async () => {
  const ui = app();
  ui.run(`state.pipelines = {
    paired: {label: 'Paired', description: '', steps: [], fields: [
      {key:'group', label:'Group', kind:'obs'}, {key:'reference', label:'Reference', kind:'level'},
      {key:'target', label:'Target', kind:'level'}, {key:'pair', label:'Subject', kind:'obs'},
      {key:'test', label:'Test', kind:'choice', choices:['ttest_rel','WilcoxonSigned'], default:'ttest_rel'}]},
    explore: {label:'Explore', description:'', steps:[], fields:[{key:'group', label:'Group', kind:'obs'}]}
  }; pipelineName = 'paired';`);
  const initial = ui.run('selectDataset("A")'); ui.pending[0].resolve(dataset('A')); await initial;
  ui.get('pipeline-group').value = 'condition'; ui.get('pipeline-group').onchange();
  ui.get('pipeline-reference').value = 'A'; ui.get('pipeline-target').value = 'B';
  ui.get('pipeline-pair').value = 'subject'; ui.get('pipeline-test').value = 'WilcoxonSigned';
  ui.get('pipeline-edit-selection').onclick();
  ui.run('selectedFeatures = new Set(["gene1", "gene2"]); showView("pipelines");');
  for (const [key, value] of Object.entries({group:'condition', reference:'A', target:'B', pair:'subject', test:'WilcoxonSigned'})) {
    assert.equal(ui.get(`pipeline-${key}`).value, value);
  }
  assert.match(ui.get('pipeline-selection').textContent, /2 selected features/);
  const next = ui.run('selectDataset("B")'); ui.pending[1].resolve(dataset('B')); await next;
  assert.equal(ui.get('pipeline-group').value, '');
  assert.equal(ui.get('pipeline-test').value, 'ttest_rel');
  ui.run('pipelineName = "explore"; renderPipeline();');
  assert.equal(ui.get('pipeline-fields').children.length, 1);
});


test('COVID presets fill selections without submitting or leaking to other datasets', async () => {
  const ui = app();
  const presets = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/static/examples/covid_proteomics/presets.json')));
  ui.run(`state.covid_presets = ${JSON.stringify(presets)};
    state.catalog = {histogram: {fields:[{key:'group'}, {key:'bins'}]}, datapoints: {fields:[{key:'group'}, {key:'distribution'}]}};
    state.pipelines = {explore: {label:'Explore', description:'', steps:[], fields:[{key:'group', label:'Group', kind:'obs'}]}};
    detail = {dataset:{status:'ready', metadata:{format:'covid', obs_columns:{Day:{values:['0','3','7','E']}, COVID:{values:['0','1']}}}}, jobs:[]};
    activeId = 'covid'; renderFeatures = () => {}; renderMethod = () => {};
    loadFeatureLabels = async (column) => { $('feature-label-column').value = column; };
    $('analysis-form').elements = {title:{}, palette:{}, yscale:{}};
    options($('numeric-columns'), ['Age cat'], false); $('numeric-columns').options[0].selected = true;
    options($('categorical-columns'), ['COVID'], false);`);
  for (const key of Object.keys(presets)) {
    await ui.run(`applyCovidPreset('${key}')`);
    assert.equal(ui.get('feature-label-column').value, 'gene_name');
    assert.deepEqual(JSON.parse(ui.run('JSON.stringify([...selectedFeatures])')), presets[key].parameters.features);
    assert.equal(ui.get('matrix').value, 'X');
    assert.equal(ui.get('filter-column').value, 'Day');
    assert.deepEqual(ui.get('filter-values').options.filter(o => o.selected).map(o => o.value), presets[key].parameters.filter_values);
    assert.equal(ui.get('numeric-columns').options[0].selected, false);
    const control = presets[key].view === 'analysis' ? 'param-group' : 'pipeline-group';
    assert.equal(ui.get(control).value, presets[key].parameters.group);
  }
  assert.equal(ui.pending.length, 0);
  ui.run('datasetLoading = true;');
  ui.get('matrix').value = 'unchanged'; ui.run("applyCovidPreset('histogram')");
  assert.equal(ui.get('matrix').value, 'unchanged');
  ui.run("datasetLoading = false; detail.dataset.metadata.format = 'demo'; renderCovidPresets(); applyCovidPreset('histogram');");
  assert.equal(ui.get('covid-presets').hidden, true);
  assert.equal(ui.get('matrix').value, 'unchanged');
});

test('feature search matches names and IDs, selects all matches independently, and keeps selection', async () => {
  const ui = app();
  ui.run(`detail = {dataset:{status:'ready', metadata:{features:['a','b','c'], raw_features:['r'], obs_columns:{}}}};
    activeId = 'A'; $('matrix').value = 'X'; selectedFeatures = new Set(['c']);
    options($('feature-label-column'), [['','IDs'], ['symbol','symbol']], false, 'symbol');`);
  const request = ui.run(`loadFeatureLabels('symbol')`);
  assert.equal(ui.get('run-button').disabled, true);
  await ui.get('analysis-form').onsubmit({preventDefault() {}});
  assert.equal(ui.pending.length, 1);
  ui.pending[0].resolve({columns:['symbol'], labels:{a:'IL6 [a]', b:'IL6 [b]', c:'TNF'}});
  await request;
  ui.get('feature-search').value = 'il6';
  assert.deepEqual(JSON.parse(ui.run('JSON.stringify(matchingFeatures())')), ['a', 'b']);
  ui.get('features-matches').onclick();
  assert.deepEqual(JSON.parse(ui.run('JSON.stringify([...selectedFeatures])')), ['c', 'a', 'b']);
  ui.get('features-clear').onclick();
  ui.get('feature-search').value = 'b';
  assert.deepEqual(JSON.parse(ui.run('JSON.stringify(matchingFeatures())')), ['b']);
  ui.get('features-all').onclick();
  assert.equal(ui.run('selectedFeatures.size'), 3);
  const ids = ui.run(`loadFeatureLabels('')`);
  ui.pending[1].resolve({columns:['symbol'], labels:{a:'a', b:'b', c:'c'}}); await ids;
  assert.equal(ui.run('selectedFeatures.size'), 3);
});

test('stale label responses cannot overwrite newer column or matrix choices', async () => {
  const ui = app();
  ui.run(`detail = {dataset:{status:'ready', metadata:{features:['a','b'], raw_features:['r'], obs_columns:{}}}};
    activeId = 'A'; $('matrix').value = 'X'; selectedFeatures = new Set(['a']);`);
  const stale = ui.run(`loadFeatureLabels('old')`);
  const latest = ui.run(`loadFeatureLabels('symbol')`);
  ui.pending[1].resolve({columns:['symbol'], labels:{a:'New A', b:'New B'}}); await latest;
  ui.pending[0].resolve({columns:['old'], labels:{a:'Old A', b:'Old B'}}); await stale;
  assert.equal(ui.run(`featureLabels.get('a')`), 'New A');
  assert.equal(ui.get('feature-label-column').value, 'symbol');
  ui.get('matrix').value = 'raw';
  const raw = ui.run(`loadFeatureLabels('symbol', true)`);
  ui.pending[2].resolve({columns:['raw_symbol'], labels:{r:'r'}}); await raw;
  assert.equal(ui.get('feature-label-column').value, '');
  assert.match(ui.get('feature-label-status').textContent, /unavailable/);
  assert.equal(ui.run(`featureLabels.get('r')`), 'r');
  const pending = ui.run(`loadFeatureLabels('raw_symbol')`);
  ui.run(`selectionVersion++; activeId = 'B'; featureLabels = new Map([['b','Dataset B']]);`);
  ui.pending[3].resolve({columns:['raw_symbol'], labels:{r:'Old dataset'}}); await pending;
  assert.equal(ui.run(`featureLabels.get('b')`), 'Dataset B');
});


test('pipeline summary refreshes after asynchronous label loading without resetting settings', async () => {
  const ui = app();
  ui.run(`state.pipelines = {explore: {label:'Explore', description:'', steps:[], fields:[{key:'group', label:'Group', kind:'obs'}]}};
    detail = {dataset:{name:'Study', status:'ready', metadata:{features:['a'], obs_columns:{condition:{values:['A','B']}}}}};
    activeId = 'A'; $('matrix').value = 'X'; selectedFeatures = new Set(['a']);
    renderPipeline(); $('pipeline-group').value = 'condition';`);
  assert.match(ui.get('pipeline-selection').textContent, /Feature labels: IDs/);
  const request = ui.run(`loadFeatureLabels('symbol', true)`);
  ui.pending[0].resolve({columns:['symbol'], labels:{a:'a'}});
  await new Promise(resolve => setImmediate(resolve));
  ui.pending[1].resolve({columns:['symbol'], labels:{a:'IL6'}});
  await request;
  assert.match(ui.get('pipeline-selection').textContent, /Feature labels: symbol/);
  assert.equal(ui.get('pipeline-group').value, 'condition');
});

# COVID proteomics example

These four data files are byte-for-byte copies of
`example_PMID_33969320/input_files/example_dataset_PMID_33969320/olink_PMID_33969320.*`
in this repository. Only the download filenames differ. See `docs/web.md` for
source attribution, metadata codes, sample counts, matrix scale, and preset scope.

The H5AD contains 784 samples, 383 participants, and 1,429 Olink assay features.
The CSV bundle represents the same X, obs, and var values, including missingness.
No filtering or transformation is applied to these downloads.

`presets.json` contains the editable app settings. `histogram.png` and
`datapoints.png` are illustrative day-0 views rendered by
`scripts/build_web_covid_previews.py`. They use only three explicitly named assays;
this choice does not reduce the downloadable dataset.

"""Bundled COVID proteomics files and editable, descriptive example settings."""

import json
from pathlib import Path

COVID_DIRECTORY = Path(__file__).parent / "static" / "examples" / "covid_proteomics"
COVID_PRESETS = json.loads((COVID_DIRECTORY / "presets.json").read_text())

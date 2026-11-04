# DS-Explorations

Config-driven data science work: exploratory analysis and forecasting models, each run from its own config file. Nothing is hardcoded. To change a dataset, column, date range or model setting, edit the config, not the code.

## Contents

| Config | What it runs | Code |
|---|---|---|
| `configs/eda.yaml` | Yearly exploratory analysis (missing values, cases by district, monthly and weekly trends, heatmaps, age breakdowns) | `src/tasks/eda/` |
| `configs/chronos.yaml` | Chronos-2 zero-shot forecasting of district-level cases, with a sliding window over a train/test period | `src/models/chronos/` |

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```
uv sync
```

This installs the exact versions pinned in `uv.lock`.

## Data

Data is not stored in git. Put the files each config points to under `data/` (see `data.path` or `data.dir` in the config). The data stays local, because it may contain patient-level records.

## Run

Point `main.py` at a config file:

```
uv run python main.py configs/eda.yaml
uv run python main.py configs/chronos.yaml
```

Each run writes to `outputs/<config name>/<timestamp>/`, together with a `run_metadata.yaml` (timestamp, dataset id and version, seed, git commit). Outputs are not tracked in git.

## EDA

Reads every CSV matching `data.pattern` in `data.dir`. With `split_by_year: true`, it runs a separate analysis for each year, using the date column.

Set these in `configs/eda.yaml`:

- `columns`: date, region (district), subregion (taluk), and other categorical columns to break down by
- `labels`: words used in plot titles
- `rename_columns`: per-file column renames, if some files use different names
- `params`: top N, heatmap size, months to compare, plot resolution

Plots whose columns are missing are skipped with a warning.

## Chronos forecasting

**Input:** a weekly sampled CSV with one row per region per sampling date: region id, date, number of cases. Dates can fall on any weekday (it is detected automatically) and must be a whole number of weeks apart. Only case counts are used, with no weather or other features.

**Periods:** you define a train and a test period in `configs/chronos.yaml`.

```yaml
train:
  start: "2021-01-04"   # history the model may use
  end: "2024-09-30"     # first cutoff
test:
  start: "2024-10-07"   # predictions are kept for these weeks
  end: "2024-12-23"
```

The model is zero-shot, so nothing is trained. The train period is the history it can see. The code slides the cutoff forward one week at a time through the test period. At each cutoff the model sees only the data up to that date and forecasts the next `horizon_weeks` weeks for every region. Predictions for weeks inside the test period are saved, with the real values (`y_true`) next to them.

**To forecast the future:** set the train end to your last data date. The test weeks then fall after your data, and `y_true` is blank.

**Output:** one CSV with `model`, region, `originDate`, `targetDate`, `lead_h` (1 to 4 weeks ahead), `y_true`, `prediction`, and the quantile columns.

## Add a new task or model

1. Create `configs/<name>.yaml`. It needs a `seed`, and can include `experiment` and `data` (`id`, `version`) sections.
2. Create `src/tasks/<name>/run.py` or `src/models/<name>/run.py`, with an empty `__init__.py` in each new folder.
3. In `run.py`, define `run(cfg, out_dir)`.
4. Run it: `uv run python main.py configs/<name>.yaml`

To use the same code with a second config, add `module: <name>` at the top of the new yaml.

## Project layout

```
DS-Explorations/
├── main.py              # runs a config file
├── configs/             # one yaml per task or model
├── src/
│   ├── common/          # shared helpers (config loader)
│   ├── tasks/eda/
│   └── models/chronos/
├── docs/                # MkDocs pages
├── data/                # not in git
├── outputs/             # not in git
├── pyproject.toml
└── uv.lock
```

## Reproducibility

- Dependencies are managed with `uv`. `pyproject.toml` and `uv.lock` are committed.
- All settings live in the config files, which are committed.
- Every run records the seed, dataset id and version, and git commit in `run_metadata.yaml`. Commit your config changes before running, so the commit hash matches the settings used.
- Add packages with `uv add <package>`, not `pip install`.

## Development

```
uv run ruff check . --fix    # lint
uv run ruff format .         # format
uv run mkdocs serve          # view docs
```

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).

# LLM TSF Leaderboard

Repository for evaluating and comparing time series forecasting models. It includes datasets, utilities, and a simple interface to run experiments and visualize results.

## Overview

- `data/` — datasets used in the experiments and the dataset card with descriptions.
- `leaderboard/` — code for the benchmark interface and utilities (see `leaderboard/src`).

## Quickstart

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r leaderboard/requirements.txt
```

3. Run the Streamlit interface:

```bash
python leaderboard/src/main.py
```

Note: to run Streamlit directly use `python -m streamlit run leaderboard/src/home.py`.

## Datasets

The dataset card with detailed descriptions is available at `data/DATASET_CARD.md`.

## Repository structure

- `data/` — CSV files and `DATASET_CARD.md`.
- `leaderboard/src/` — Python scripts for the UI and configuration.
- `notebooks/` — Jupyter notebooks used for exploratory analysis, experiments and visualizations.
- `results/` — experiment outputs, logs and reports (model checkpoints, metrics, figures).

## Contributing

Please open an issue or submit a pull request with improvements. I can prepare a PR with these changes if you want.

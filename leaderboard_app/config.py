from __future__ import annotations

from pathlib import Path

APP_TITLE = "LLM4TSF Leaderboard"
APP_SUBTITLE = "Ranking de LLMs para previsão de séries temporais"
DATA_PATH = Path("data/results.csv")

METRICS = [
    "smape",
    "mae",
    "rmse",
    "inference_time_s",
    "co2_g",
    "cost_usd_per_1k_forecasts",
]

METRIC_LABELS = {
    "smape": "SMAPE ↓",
    "mae": "MAE ↓",
    "rmse": "RMSE ↓",
    "inference_time_s": "Tempo inferência (s) ↓",
    "co2_g": "CO₂ (g) ↓",
    "cost_usd_per_1k_forecasts": "Custo / 1k previsões (US$) ↓",
}

DEFAULT_WEIGHTS = {
    "smape": 35,
    "mae": 15,
    "rmse": 15,
    "inference_time_s": 15,
    "co2_g": 10,
    "cost_usd_per_1k_forecasts": 10,
}

REQUIRED_COLUMNS = [
    "model_name",
    "model_id",
    "family",
    "provider",
    "deployment",
    "dataset",
    "horizon",
    "prompt_strategy",
    "smape",
    "mae",
    "rmse",
    "inference_time_s",
    "co2_g",
    "cost_usd_per_1k_forecasts",
    "run_id",
    "evaluation_date",
]

PROMPT_ORDER = ["Geral", "zero-shot", "few-shot", "cot", "cot+few"]

NOTEBOOK_URL = "https://github.com/llm4time/NeurIPS2026/blob/main/Carbon_Qwen3_8b.ipynb"
RAW_NOTEBOOK_URL = "https://raw.githubusercontent.com/llm4time/NeurIPS2026/main/Carbon_Qwen3_8b.ipynb"

OPTIONAL_SUBMISSION_COLUMNS = [
    "parameters_b",
    "context_length_k",
    "representation",
    "hardware",
    "quantization",
    "energy_kwh_per_1k_forecasts",
    "api_pricing_source",
    "execution_source",
    "notes",
]

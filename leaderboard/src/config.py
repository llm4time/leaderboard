from pathlib import Path

import numpy as np
import pandas as pd

APP_TITLE = "LLM4TSF Leaderboard"
APP_SUBTITLE = "Ranking of LLMs for time series forecasting. Compare accuracy, operational robustness, latency, carbon footprint, and financial cost in a single, transparent ranking."

PROMPT_ORDER = ["All", "Zero-shot", "Few-shot", "CoT", "CoT + Few-shot"]
PROMPT_CHART_ORDER = ["Zero-shot", "Few-shot", "CoT", "CoT+Few-shot"]

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "../data" / "experiments.csv"

METRICS = ["smape", "mae", "rmse", "inference_time_s", "financial_cost_usd", "eco2ai_energy_wh"]

METRIC_LABELS = {
  "smape": "SMAPE ↓",
  "mae": "MAE ↓",
  "rmse": "RMSE ↓",
  "inference_time_s": "Inference Time (s) ↓",
  "financial_cost_usd": "Cost / 1k (US$) ↓",
  "eco2ai_energy_wh": "Energy (Wh) ↓",
}

DEFAULT_WEIGHTS = {
  "smape": 35,
  "mae": 15,
  "rmse": 15,
  "inference_time_s": 15,
  "financial_cost_usd": 10,
  "eco2ai_energy_wh": 10,
}

MODEL_PARAMS_B = {
  "glm-5.1": 754,
  "gpt-oss-120b": 120,
  "gpt-oss-20b": 20,
  "gemma-3-1b": 1,
  "gemma-3-4b": 4,
  "gemma-3-12b": 12,
  "gemma-3-27b": 27,
  "kimi-k2": 1_000,
  "llama-3.1-8b": 8,
  "llama-3.3-70b": 70,
  "mistral-3-14b": 14,
  "mistral-3-3b": 3,
  "mistral-3.2-24b": 24,
  "mistral-3.5-128b": 128,
  "qwen3-0.6b": 0.6,
  "qwen3-1.7b": 1.7,
  "qwen3-4b": 4,
  "qwen3-8b": 8,
  "qwen3-14b": 14,
  "qwen3-32b": 32,
  "qwen3.5-397b-a17b": 397,
}

FAMILY_STYLE = {
  "GLM": {"color": "#E63946", "label": "GLM"},
  "GPT": {"color": "#457B9D", "label": "GPT"},
  "Gemma": {"color": "#2A9D8F", "label": "Gemma"},
  "Kimi": {"color": "#F4A261", "label": "Kimi"},
  "Llama": {"color": "#6A4C93", "label": "Llama"},
  "Mistral": {"color": "#E9C46A", "label": "Mistral"},
  "Qwen": {"color": "#264653", "label": "Qwen"},
}

CHARACTERISTICS = {
  "ETTh2": [0.985, 0.859, 0.023, 0.005, 0.287],
  "Electricity": [0.810, 0.992, 0.019, 0.005, 0.438],
  "Traffic": [0.689, 0.955, 0.006, 0.011, 0.308],
  "Covid-19": [0.999, 0.786, 0.288, 0.034, 0.268],
  "Wike2000": [0.843, 0.965, 0.052, 0.011, 0.420],
  "Retail": [0.436, 0.908, 0.202, 0.014, 0.396],
  "Carbon": [0.966, 0.665, 0.050, 0.010, 0.320],
}

ENERGY_MODEL_EXCLUSIONS = [
  "mistralai/mistral-medium-3.5-128b",
  "qwen/qwen3.5-397b-a17b",
  "moonshotai/kimi-k2-instruct-0905",
  "z-ai/glm-5.1",
  "gpt-5-nano-2025-08-07",
]


def _normalize_results(raw: pd.DataFrame) -> pd.DataFrame:
  results = raw.copy()
  results.rename(columns={
    "Model": "model",
    "Temperature": "temperature",
    "Dataset": "dataset",
    "Training Start": "training_start",
    "Training End": "training_end",
    "Forecast Horizon": "forecast_horizon",
    "Prompt Type": "prompt_type",
    "Series Type": "series_type",
    "Format": "format",
    "sMAPE Mean": "smape",
    "MAE Mean": "mae",
    "RMSE Mean": "rmse",
    "Inference Time (s) Mean": "inference_time_s",
    "Financial Cost ($) Mean": "financial_cost_usd",
    "CarbonTracker CO2 (g) Mean": "carbontracker_co2_g",
    "CarbonTracker Energy Consumption (Wh) Mean": "carbontracker_energy_wh",
    "Eco2AI CO2 (g) Mean": "eco2ai_co2_g",
    "Eco2AI Energy Consumption (Wh) Mean": "eco2ai_energy_wh",
    "Input Tokens Mean": "input_tokens",
    "Output Tokens Mean": "output_tokens",
    "Examples": "examples",
    "Sampling Strategy": "sampling_strategy",
    "Actual Values": "actual_values",
    "Predicted Values": "predicted_values",
    "sMAPE SEM": "smape_sem",
    "MAE SEM": "mae_sem",
    "RMSE SEM": "rmse_sem",
    "Deployment": "deployment",
    "Family": "family",
  }, inplace=True)

  for col in ["temperature", "forecast_horizon", "smape", "mae", "rmse", "inference_time_s",
              "financial_cost_usd", "carbontracker_co2_g", "carbontracker_energy_wh",
              "eco2ai_co2_g", "eco2ai_energy_wh", "input_tokens", "output_tokens",
              "examples", "smape_sem", "mae_sem", "rmse_sem"]:
    if col in results.columns:
      results[col] = pd.to_numeric(results[col], errors="coerce")

  results["tokens_total"] = results["input_tokens"].fillna(0) + results["output_tokens"].fillna(0)
  denominator = np.log1p(results["tokens_total"])
  results["fte_score"] = np.where(denominator > 0, (100 - results["smape"]) / denominator, np.nan)
  results["prompt_label"] = results["prompt_type"].astype(str).str.replace(" + ", "+", regex=False)
  results["format_label"] = results["format"].astype(str).str.upper()
  results["series_label"] = results["series_type"].astype(str).str.title()
  results["model_short"] = results["model"].astype(str).str.split("/").str[-1]
  results["model_key"] = results["model_short"].str.lower()
  return results


RESULTS = _normalize_results(pd.read_csv(DATA_PATH))
DATASET_ORDER = list(RESULTS["dataset"].drop_duplicates())
MODEL_ORDER = list(RESULTS["model"].drop_duplicates())


def filter_results(ranking: str) -> pd.DataFrame:
  if ranking == "All":
    return RESULTS.copy()
  return RESULTS.loc[RESULTS["prompt_type"] == ranking].copy()


def compute_leaderboard(df: pd.DataFrame, weights: dict, missing_policy: str) -> pd.DataFrame:
  import math
  if df.empty:
    return pd.DataFrame()

  agg = df.groupby("model", as_index=False).agg(
    smape=("smape", "mean"),
    mae=("mae", "mean"),
    rmse=("rmse", "mean"),
    inference_time_s=("inference_time_s", "mean"),
    financial_cost_usd=("financial_cost_usd", "mean"),
    eco2ai_energy_wh=("eco2ai_energy_wh", "mean"),
    n_runs=("smape", "count"),
    n_datasets=("dataset", pd.Series.nunique),
    family=("family", "first"),
    deployment=("deployment", "first"),
  )

  total_w = sum(max(0.0, v) for v in weights.values())
  norm_w = {m: max(0.0, v) / total_w for m, v in weights.items()} if total_w > 0 else {m: 1/len(weights) for m in weights}

  for m in METRICS:
    col = agg[m].copy() if m in agg.columns else pd.Series(np.nan, index=agg.index)
    if missing_policy == "Penalize as worst":
      col = col.fillna(col.max(skipna=True))
    elif missing_policy == "Fill with median":
      col = col.fillna(col.median(skipna=True))
    mn, mx = col.min(skipna=True), col.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or math.isclose(float(mx), float(mn)):
      agg[f"score_{m}"] = 100.0
    else:
      agg[f"score_{m}"] = 100.0 * (mx - col) / (mx - mn)

  def _overall(row):
    total, tw = 0.0, 0.0
    for m in METRICS:
      s, w = row.get(f"score_{m}", np.nan), norm_w.get(m, 0.0)
      if pd.notna(s):
        total += s * w; tw += w
    return total / tw if tw > 0 else np.nan

  agg["overall_score"] = agg.apply(_overall, axis=1)
  agg = agg.sort_values(["overall_score", "smape", "mae"], ascending=[False, True, True], na_position="last").reset_index(drop=True)
  agg.insert(0, "rank", np.arange(1, len(agg) + 1))
  return agg


def normalize_weights(weights: dict) -> dict:
  total = sum(max(0.0, v) for v in weights.values())
  if total <= 0:
    return {m: 1.0 / len(weights) for m in weights}
  return {m: max(0.0, v) / total for m, v in weights.items()}

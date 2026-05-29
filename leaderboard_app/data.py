from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .config import DATA_PATH, METRICS, REQUIRED_COLUMNS


@st.cache_data(show_spinner=False)
def load_results(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in METRICS + ["parameters_b", "context_length_k", "horizon", "run_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["model_name", "model_id", "family", "provider", "deployment", "dataset", "prompt_strategy"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-8)) * 100.0)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def parse_series(text: str) -> List[float]:
    values: List[float] = []
    clean = text.replace(";", ",").replace("\n", ",").strip()
    for token in clean.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def lower_is_better_score(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)

    min_v = values.min(skipna=True)
    max_v = values.max(skipna=True)
    if pd.isna(min_v) or pd.isna(max_v):
        return pd.Series(np.nan, index=values.index)
    if math.isclose(float(max_v), float(min_v)):
        return pd.Series(100.0, index=values.index)

    return 100.0 * (max_v - values) / (max_v - min_v)


def treat_missing_metric(values: pd.Series, policy: str) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").copy()
    if values.notna().sum() == 0:
        return values

    if policy == "Penalizar ausentes como pior valor":
        return values.fillna(values.max(skipna=True))
    if policy == "Preencher ausentes com mediana":
        return values.fillna(values.median(skipna=True))
    return values


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return {m: 1.0 / len(weights) for m in weights}
    return {m: max(0.0, float(v)) / total for m, v in weights.items()}


def compute_leaderboard(df: pd.DataFrame, weights: Dict[str, float], missing_policy: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    model_fields = [
        "model_name",
        "model_id",
        "family",
        "provider",
        "deployment",
        "hardware",
        "quantization",
    ]
    for field in model_fields:
        if field not in df.columns:
            df[field] = "Unknown"

    agg_spec = {m: "mean" for m in METRICS if m in df.columns}
    agg_spec.update({
        "run_id": "count",
        "dataset": pd.Series.nunique,
        "prompt_strategy": pd.Series.nunique,
    })
    if "parameters_b" in df.columns:
        agg_spec["parameters_b"] = "max"
    if "context_length_k" in df.columns:
        agg_spec["context_length_k"] = "max"

    board = (
        df.groupby(model_fields, dropna=False)
        .agg(agg_spec)
        .reset_index()
        .rename(columns={"run_id": "n_runs", "dataset": "n_datasets", "prompt_strategy": "n_prompt_strategies"})
    )

    normalized_weights = normalize_weights(weights)
    for metric in METRICS:
        if metric not in board.columns:
            board[metric] = np.nan
        treated = treat_missing_metric(board[metric], missing_policy)
        board[f"score_{metric}"] = lower_is_better_score(treated)

    weighted_scores = []
    for _, row in board.iterrows():
        total_score = 0.0
        total_weight = 0.0
        for metric in METRICS:
            s = row[f"score_{metric}"]
            w = normalized_weights.get(metric, 0.0)
            if pd.notna(s):
                total_score += float(s) * w
                total_weight += w
        weighted_scores.append(total_score / total_weight if total_weight > 0 else np.nan)

    board["overall_score"] = weighted_scores
    board = board.sort_values(
        by=["overall_score", "smape", "mae", "rmse"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def normalize_prompt_strategy(value: object) -> str:
    raw = str(value).strip().lower().replace("_", "-")
    aliases = {
        "zero": "zero-shot",
        "zeroshot": "zero-shot",
        "zero shot": "zero-shot",
        "zero-shot": "zero-shot",
        "few": "few-shot",
        "fewshot": "few-shot",
        "few shot": "few-shot",
        "few-shot": "few-shot",
        "chain-of-thought": "cot",
        "chain of thought": "cot",
        "cot": "cot",
        "few-shot-cot": "cot+few",
        "cot-few": "cot+few",
        "few-cot": "cot+few",
        "cot+few": "cot+few",
        "fs-cot": "cot+few",
        "few-shot chain-of-thought": "cot+few",
    }
    return aliases.get(raw, raw)


def parse_prediction_cell(value: object) -> List[float]:
    if pd.isna(value):
        return []
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return [float(x) for x in parsed]
        except Exception:
            pass
    return parse_series(text)


def infer_csv_mode(uploaded_df: pd.DataFrame) -> str:
    cols = set(uploaded_df.columns)
    if {"y_true", "y_pred"}.issubset(cols) or {"actual", "prediction"}.issubset(cols):
        return "Previsões brutas"
    return "Resultados já calculados"


def coerce_uploaded_csv(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    df = uploaded_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "model": "model_name",
        "model_name_or_path": "model_id",
        "id_modelo": "model_id",
        "modelo": "model_name",
        "familia": "family",
        "fornecedor": "provider",
        "execucao": "deployment",
        "estrategia": "prompt_strategy",
        "prompt": "prompt_strategy",
        "prompting": "prompt_strategy",
        "tempo": "inference_time_s",
        "latency": "inference_time_s",
        "latency_s": "inference_time_s",
        "co2": "co2_g",
        "carbon_g": "co2_g",
        "cost": "cost_usd_per_1k_forecasts",
        "cost_usd": "cost_usd_per_1k_forecasts",
        "api_cost": "cost_usd_per_1k_forecasts",
        "run": "run_id",
        "date": "evaluation_date",
        "actual": "y_true",
        "prediction": "y_pred",
        "yhat": "y_pred",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["smape", "mae", "rmse", "inference_time_s", "co2_g", "cost_usd_per_1k_forecasts", "horizon", "run_id", "parameters_b", "context_length_k"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "prompt_strategy" in df.columns:
        df["prompt_strategy"] = df["prompt_strategy"].apply(normalize_prompt_strategy)
    return df


def aggregate_predictions_csv(df: pd.DataFrame, metadata: Dict[str, object]) -> pd.DataFrame:
    work = coerce_uploaded_csv(df)
    if "y_true" not in work.columns or "y_pred" not in work.columns:
        raise ValueError("Para CSV de previsões brutas, inclua colunas `y_true` e `y_pred` ou `actual` e `prediction`.")

    grouping_candidates = [
        "model_name",
        "model_id",
        "family",
        "provider",
        "deployment",
        "dataset",
        "horizon",
        "prompt_strategy",
        "representation",
        "run_id",
    ]
    for col in grouping_candidates:
        if col not in work.columns:
            work[col] = metadata.get(col, np.nan)

    for col, value in metadata.items():
        if col not in ["y_true", "y_pred"]:
            if col not in work.columns or work[col].isna().all() or (work[col].astype(str).str.strip() == "").all():
                work[col] = value

    rows = []
    group_cols = [c for c in grouping_candidates if c in work.columns]
    for keys, group in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))

        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        for _, row in group.iterrows():
            true_values = parse_prediction_cell(row["y_true"])
            pred_values = parse_prediction_cell(row["y_pred"])
            if len(true_values) != len(pred_values):
                raise ValueError(
                    "Uma linha do CSV tem tamanhos diferentes em y_true e y_pred. "
                    f"Linha original: {int(row.name) + 1}."
                )
            y_true_all.extend(true_values)
            y_pred_all.extend(pred_values)

        if not y_true_all:
            continue

        base.update({
            "smape": round(smape(y_true_all, y_pred_all), 6),
            "mae": round(mae(y_true_all, y_pred_all), 6),
            "rmse": round(rmse(y_true_all, y_pred_all), 6),
            "inference_time_s": metadata.get("inference_time_s", np.nan),
            "co2_g": metadata.get("co2_g", np.nan),
            "cost_usd_per_1k_forecasts": metadata.get("cost_usd_per_1k_forecasts", np.nan),
            "evaluation_date": metadata.get("evaluation_date", str(date.today())),
            "hardware": metadata.get("hardware", ""),
            "quantization": metadata.get("quantization", ""),
            "execution_source": metadata.get("execution_source", ""),
            "energy_kwh_per_1k_forecasts": metadata.get("energy_kwh_per_1k_forecasts", np.nan),
            "api_pricing_source": metadata.get("api_pricing_source", ""),
            "notes": metadata.get("notes", "uploaded predictions CSV"),
        })
        rows.append(base)

    return pd.DataFrame(rows)


def prepare_results_csv(df: pd.DataFrame, metadata: Dict[str, object]) -> pd.DataFrame:
    work = coerce_uploaded_csv(df)
    for col, value in metadata.items():
        if col not in work.columns:
            work[col] = value
        else:
            missing = work[col].isna() | (work[col].astype(str).str.strip() == "")
            work.loc[missing, col] = value

    for col in REQUIRED_COLUMNS:
        if col not in work.columns:
            work[col] = np.nan

    essential = [
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
    missing_cells = []
    for col in essential:
        if col not in work.columns or work[col].isna().any() or (work[col].astype(str).str.strip() == "").any():
            missing_cells.append(col)
    if missing_cells:
        raise ValueError(
            "O CSV ainda possui campos obrigatórios ausentes após aplicar os metadados do formulário: "
            + ", ".join(sorted(set(missing_cells)))
        )

    for col in METRICS + ["horizon", "run_id", "parameters_b", "context_length_k", "energy_kwh_per_1k_forecasts"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["prompt_strategy"] = work["prompt_strategy"].apply(normalize_prompt_strategy)
    return work


def safe_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

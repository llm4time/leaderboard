from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .config import APP_SUBTITLE, APP_TITLE, DEFAULT_WEIGHTS, METRIC_LABELS, PROMPT_ORDER
from .data import normalize_weights


def filter_dataframe(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if df.empty:
        return df

    with st.sidebar:
        st.markdown("### Filtros")
        datasets = sorted(df["dataset"].dropna().unique().tolist())
        strategies = sorted(df["prompt_strategy"].dropna().unique().tolist())
        deployments = sorted(df["deployment"].dropna().unique().tolist())
        families = sorted(df["family"].dropna().unique().tolist())
        horizons = sorted([int(x) for x in df["horizon"].dropna().unique().tolist()])

        selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key=f"{prefix}_datasets")
        selected_horizons = st.multiselect("Horizontes", horizons, default=horizons, key=f"{prefix}_horizons")
        selected_deployments = st.multiselect("Deployments", deployments, default=deployments, key=f"{prefix}_deployments")
        selected_families = st.multiselect("Famílias", families, default=families, key=f"{prefix}_families")
        selected_strategies = st.multiselect("Estratégias", strategies, default=strategies, key=f"{prefix}_strategies")

    return df[
        df["dataset"].isin(selected_datasets)
        & df["horizon"].isin(selected_horizons)
        & df["deployment"].isin(selected_deployments)
        & df["family"].isin(selected_families)
        & df["prompt_strategy"].isin(selected_strategies)
    ].copy()


def sidebar_weights(prefix: str = "") -> tuple[dict[str, int], str]:
    with st.sidebar:
        st.markdown("### Fórmula do ranking")
        st.caption("Todas as métricas são normalizadas como ‘menor é melhor’. Ajuste os pesos conforme a prioridade do benchmark.")
        weights = {}
        for metric, default in DEFAULT_WEIGHTS.items():
            weights[metric] = st.slider(
                METRIC_LABELS[metric],
                min_value=0,
                max_value=100,
                value=default,
                step=5,
                key=f"{prefix}_weight_{metric}",
            )
        missing_policy = st.selectbox(
            "Tratamento de métricas ausentes",
            [
                "Penalizar ausentes como pior valor",
                "Preencher ausentes com mediana",
                "Ignorar ausentes no score da linha",
            ],
            index=0,
            key=f"{prefix}_missing_policy",
            help="Útil para modelos API sem telemetria confiável de energia/carbono.",
        )
    return weights, missing_policy


def style_leaderboard_table(board: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "rank",
        "model_name",
        "overall_score",
        "smape",
        "mae",
        "rmse",
        "inference_time_s",
        "co2_g",
        "cost_usd_per_1k_forecasts",
        "deployment",
        "provider",
        "hardware",
        "quantization",
        "n_runs",
    ]
    available = [c for c in cols if c in board.columns]
    out = board[available].copy()
    rename = {
        "rank": "Rank",
        "model_name": "Modelo",
        "overall_score": "Score ↑",
        "smape": "SMAPE ↓",
        "mae": "MAE ↓",
        "rmse": "RMSE ↓",
        "inference_time_s": "Tempo (s) ↓",
        "co2_g": "CO₂ (g) ↓",
        "cost_usd_per_1k_forecasts": "Custo / 1k (US$) ↓",
        "deployment": "Execução",
        "provider": "Provider",
        "hardware": "Hardware",
        "quantization": "Quantização",
        "n_runs": "Runs",
    }
    return out.rename(columns=rename)


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>🏆 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}. Compare acurácia, robustez operacional, latência, carbono e custo financeiro em um único ranking transparente.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_podium(board: pd.DataFrame) -> None:
    if board.empty:
        return
    top = board.head(3).copy()
    cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="podium-card">
                    <div style="font-size:1.8rem;">{medals[i]}</div>
                    <div style="font-weight:780;font-size:1.05rem;margin-top:.2rem;">{row['model_name']}</div>
                    <div style="font-size:.88rem;color:rgba(49,51,63,.68);margin:.2rem 0 .45rem;">Score: {row['overall_score']:.2f}</div>
                    <span class="badge">{row.get('deployment', 'Unknown')}</span>
                    <span class="badge">SMAPE {row.get('smape', np.nan):.2f}</span>
                    <span class="badge">{row.get('n_runs', 0):.0f} runs</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def bar_top_models(board: pd.DataFrame) -> alt.Chart:
    plot_df = board.head(12).copy()
    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7)
        .encode(
            y=alt.Y("model_name:N", sort="-x", title="Modelo"),
            x=alt.X("overall_score:Q", title="Score composto ↑"),
            tooltip=["rank", "model_name", alt.Tooltip("overall_score:Q", format=".2f"), alt.Tooltip("smape:Q", format=".2f")],
            color=alt.Color("deployment:N", title="Execução"),
        )
        .properties(height=360)
    )


def pareto_chart(board: pd.DataFrame) -> alt.Chart:
    plot_df = board.copy()
    return (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.78)
        .encode(
            x=alt.X("inference_time_s:Q", title="Tempo médio de inferência (s) ↓"),
            y=alt.Y("smape:Q", title="SMAPE médio ↓"),
            size=alt.Size("overall_score:Q", title="Score ↑", scale=alt.Scale(range=[80, 900])),
            color=alt.Color("deployment:N", title="Execução"),
            tooltip=[
                "rank",
                "model_name",
                alt.Tooltip("overall_score:Q", format=".2f"),
                alt.Tooltip("smape:Q", format=".2f"),
                alt.Tooltip("inference_time_s:Q", format=".2f"),
                alt.Tooltip("co2_g:Q", format=".3f"),
                alt.Tooltip("cost_usd_per_1k_forecasts:Q", format=".4f"),
            ],
        )
        .interactive()
        .properties(height=380)
    )


def prompt_strategy_chart(df: pd.DataFrame) -> alt.Chart:
    plot_df = df.groupby("prompt_strategy", dropna=False)[["smape", "mae", "rmse"]].mean().reset_index()
    plot_df["prompt_strategy"] = pd.Categorical(plot_df["prompt_strategy"], categories=PROMPT_ORDER[1:], ordered=True)
    plot_df = plot_df.sort_values("prompt_strategy")
    return (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
        .encode(
            x=alt.X("prompt_strategy:N", title="Estratégia de prompt", sort=PROMPT_ORDER[1:]),
            y=alt.Y("smape:Q", title="SMAPE médio ↓"),
            tooltip=["prompt_strategy", alt.Tooltip("smape:Q", format=".2f"), alt.Tooltip("mae:Q", format=".4f"), alt.Tooltip("rmse:Q", format=".4f")],
        )
        .properties(height=320)
    )


def dataset_heatmap(df: pd.DataFrame) -> alt.Chart:
    plot_df = df.groupby(["model_name", "dataset"], dropna=False)["smape"].mean().reset_index()
    if plot_df["model_name"].nunique() > 14:
        top_models = plot_df.groupby("model_name")["smape"].mean().sort_values().head(14).index.tolist()
        plot_df = plot_df[plot_df["model_name"].isin(top_models)]
    return (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=alt.X("dataset:N", title="Dataset"),
            y=alt.Y("model_name:N", title="Modelo"),
            color=alt.Color("smape:Q", title="SMAPE ↓", scale=alt.Scale(scheme="yelloworangered", reverse=True)),
            tooltip=["model_name", "dataset", alt.Tooltip("smape:Q", format=".2f")],
        )
        .properties(height=max(320, 26 * plot_df["model_name"].nunique()))
    )


def cost_score_chart(board: pd.DataFrame) -> alt.Chart:
    plot_df = board.copy()
    return (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.78)
        .encode(
            x=alt.X("cost_usd_per_1k_forecasts:Q", title="Custo / 1k previsões (US$) ↓"),
            y=alt.Y("overall_score:Q", title="Score composto ↑"),
            size=alt.Size("parameters_b:Q", title="Parâmetros (B)", scale=alt.Scale(range=[70, 900])),
            color=alt.Color("family:N", title="Família"),
            tooltip=["rank", "model_name", "family", alt.Tooltip("overall_score:Q", format=".2f"), alt.Tooltip("cost_usd_per_1k_forecasts:Q", format=".4f")],
        )
        .interactive()
        .properties(height=360)
    )


def render_required_csv_schema() -> None:
    st.markdown("**Formato A — resultados já calculados**")
    st.code(
        "model_name,model_id,family,provider,deployment,dataset,horizon,prompt_strategy,"
        "smape,mae,rmse,inference_time_s,co2_g,cost_usd_per_1k_forecasts,run_id,evaluation_date",
        language="text",
    )
    st.markdown("**Formato B — previsões brutas**")
    st.code(
        "model_name,model_id,dataset,horizon,prompt_strategy,run_id,y_true,y_pred\n"
        "Qwen3-8B,Qwen/Qwen3-8B,Carbon,96,zero-shot,1,10.2,10.0\n"
        "Qwen3-8B,Qwen/Qwen3-8B,Carbon,96,zero-shot,1,11.0,10.9",
        language="text",
    )

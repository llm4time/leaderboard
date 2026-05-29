from __future__ import annotations

from datetime import date
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from leaderboard_app.config import (
    APP_SUBTITLE,
    APP_TITLE,
    DATA_PATH,
    DEFAULT_WEIGHTS,
    METRIC_LABELS,
    METRICS,
    NOTEBOOK_URL,
    OPTIONAL_SUBMISSION_COLUMNS,
    PROMPT_ORDER,
    RAW_NOTEBOOK_URL,
    REQUIRED_COLUMNS,
)
from leaderboard_app.data import (
    aggregate_predictions_csv,
    coerce_uploaded_csv,
    compute_leaderboard,
    infer_csv_mode,
    load_results,
    mae,
    normalize_weights,
    parse_prediction_cell,
    parse_series,
    prepare_results_csv,
    rmse,
    safe_download_csv,
    smape,
    validate_schema,
)
from leaderboard_app.ui import (
    bar_top_models,
    cost_score_chart,
    dataset_heatmap,
    filter_dataframe,
    pareto_chart,
    prompt_strategy_chart,
    render_hero,
    render_metric_card,
    render_podium,
    render_required_csv_schema,
    sidebar_weights,
    style_leaderboard_table,
)


# ============================================================
# Estilo visual inspirado em leaderboards modernos
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --leaderboard-card-radius: 18px;
    }

    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 24px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, rgba(255, 244, 230, 0.88), rgba(246, 248, 255, 0.88));
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin-bottom: 0.25rem;
        font-size: 2.25rem;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0.25rem 0 0;
        color: rgba(49, 51, 63, 0.75);
        font-size: 1.02rem;
    }

    .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: var(--leaderboard-card-radius);
        padding: 1rem 1.05rem;
        background: white;
        box-shadow: 0 8px 30px rgba(0,0,0,0.035);
        min-height: 108px;
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: rgba(49, 51, 63, 0.68);
        margin-bottom: 0.3rem;
    }

    .metric-card .value {
        font-size: 1.55rem;
        font-weight: 760;
        letter-spacing: -0.035em;
    }

    .metric-card .caption {
        margin-top: 0.25rem;
        font-size: 0.78rem;
        color: rgba(49, 51, 63, 0.58);
    }

    .podium-card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 20px;
        padding: 1rem;
        background: white;
        box-shadow: 0 8px 28px rgba(0,0,0,0.04);
        min-height: 145px;
    }

    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.18rem 0.58rem;
        font-size: 0.78rem;
        font-weight: 650;
        background: rgba(255, 170, 0, 0.16);
        color: rgb(98, 67, 0);
        margin-right: 0.35rem;
        margin-top: 0.25rem;
    }

    .schema-box {
        border: 1px dashed rgba(49, 51, 63, 0.28);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(250, 250, 250, 0.7);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Página 1 — Leaderboard principal
# ============================================================

def page_main() -> None:
    render_hero()
    df = load_results()
    if df.empty:
        st.warning("Nenhum resultado encontrado. Adicione um arquivo em `data/results.csv` ou use a página de Demo & Submissão.")
        return

    ok, missing = validate_schema(df)
    if not ok:
        st.error(f"CSV com colunas ausentes: {', '.join(missing)}")
        return

    weights, missing_policy = sidebar_weights("main")
    filtered = filter_dataframe(df, "main")

    st.markdown("### Seleção do ranking")
    rank_scope = st.radio(
        "Escolha a visão do ranking",
        PROMPT_ORDER,
        horizontal=True,
        label_visibility="collapsed",
        index=0,
    )

    if rank_scope != "Geral":
        filtered = filtered[filtered["prompt_strategy"] == rank_scope].copy()

    board = compute_leaderboard(filtered, weights, missing_policy)

    if board.empty:
        st.warning("Nenhum resultado para os filtros selecionados.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Modelos", f"{board['model_name'].nunique():,}", "após filtros".replace(",", "."))
    with c2:
        render_metric_card("Resultados", f"{len(filtered):,}".replace(",", "."), "linhas avaliadas")
    with c3:
        render_metric_card("Líder", str(board.iloc[0]["model_name"]), f"score {board.iloc[0]['overall_score']:.2f}")
    with c4:
        render_metric_card("Melhor SMAPE", f"{board['smape'].min():.2f}", "média por modelo")
    with c5:
        render_metric_card("Menor custo", f"US$ {board['cost_usd_per_1k_forecasts'].min():.4f}", "por 1k previsões")

    st.markdown("### Pódio")
    render_podium(board)

    st.markdown("### Ranking")
    st.dataframe(
        style_leaderboard_table(board),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score ↑": st.column_config.ProgressColumn("Score ↑", min_value=0, max_value=100, format="%.2f"),
            "SMAPE ↓": st.column_config.NumberColumn("SMAPE ↓", format="%.3f"),
            "MAE ↓": st.column_config.NumberColumn("MAE ↓", format="%.5f"),
            "RMSE ↓": st.column_config.NumberColumn("RMSE ↓", format="%.5f"),
            "Tempo (s) ↓": st.column_config.NumberColumn("Tempo (s) ↓", format="%.3f"),
            "CO₂ (g) ↓": st.column_config.NumberColumn("CO₂ (g) ↓", format="%.5f"),
            "Custo / 1k (US$) ↓": st.column_config.NumberColumn("Custo / 1k (US$) ↓", format="%.5f"),
        },
    )

    st.download_button(
        "⬇️ Baixar ranking filtrado em CSV",
        data=safe_download_csv(board),
        file_name=f"llm4tsf_leaderboard_{rank_scope}.csv".replace("+", "plus"),
        mime="text/csv",
    )

    st.markdown("### Gráficos")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Top modelos",
        "Pareto acurácia × tempo",
        "Estratégias de prompt",
        "Heatmap por dataset",
        "Custo × score",
    ])
    with tab1:
        st.altair_chart(bar_top_models(board), use_container_width=True)
    with tab2:
        st.caption("Idealmente, os melhores modelos ficam no canto inferior esquerdo: baixo erro e baixa latência.")
        st.altair_chart(pareto_chart(board), use_container_width=True)
    with tab3:
        st.altair_chart(prompt_strategy_chart(filtered), use_container_width=True)
    with tab4:
        st.altair_chart(dataset_heatmap(filtered), use_container_width=True)
    with tab5:
        st.altair_chart(cost_score_chart(board), use_container_width=True)

    with st.expander("Como o score é calculado"):
        normalized = normalize_weights(weights)
        formula_df = pd.DataFrame(
            {
                "Métrica": [METRIC_LABELS[m] for m in METRICS],
                "Peso configurado": [weights[m] for m in METRICS],
                "Peso normalizado": [normalized[m] for m in METRICS],
                "Direção": ["menor é melhor" for _ in METRICS],
            }
        )
        st.dataframe(formula_df, use_container_width=True, hide_index=True)
        st.code(
            "score_metrica = 100 * (max(valor) - valor) / (max(valor) - min(valor))\n"
            "score_final = soma(score_metrica * peso_normalizado)",
            language="text",
        )


# ============================================================
# Página 2 — Resultados completos
# ============================================================

def page_results() -> None:
    st.title("📊 Resultados completos")
    st.caption("Tabela bruta dos experimentos, pivôs por dataset e exportação para auditoria/reprodutibilidade.")

    df = load_results()
    if df.empty:
        st.warning("Nenhum resultado encontrado em `data/results.csv`.")
        return

    filtered = filter_dataframe(df, "results")

    st.markdown("### Tabela bruta")
    search = st.text_input("Buscar por modelo, dataset, família ou provider", placeholder="Ex.: Qwen, ETTh2, API...")
    table_df = filtered.copy()
    if search:
        query = search.strip().lower()
        mask = pd.Series(False, index=table_df.index)
        for col in ["model_name", "model_id", "dataset", "family", "provider", "deployment", "hardware"]:
            if col in table_df.columns:
                mask = mask | table_df[col].astype(str).str.lower().str.contains(query, na=False)
        table_df = table_df[mask]

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Baixar resultados filtrados",
        data=safe_download_csv(table_df),
        file_name="llm4tsf_filtered_results.csv",
        mime="text/csv",
    )

    st.markdown("### Visões agregadas")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Erro médio por modelo e estratégia")
        pivot_prompt = pd.pivot_table(
            filtered,
            index="model_name",
            columns="prompt_strategy",
            values="smape",
            aggfunc="mean",
        ).reset_index()
        st.dataframe(pivot_prompt, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Erro médio por dataset")
        dataset_summary = (
            filtered.groupby("dataset", dropna=False)
            .agg(
                smape=("smape", "mean"),
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                n_results=("model_name", "count"),
                n_models=("model_name", "nunique"),
            )
            .reset_index()
            .sort_values("smape")
        )
        st.dataframe(dataset_summary, use_container_width=True, hide_index=True)

    st.markdown("### Diagnóstico de cobertura")
    coverage = (
        filtered.groupby(["model_name", "prompt_strategy"], dropna=False)
        .agg(n_datasets=("dataset", "nunique"), n_runs=("run_id", "count"), smape=("smape", "mean"))
        .reset_index()
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    with st.expander("Esquema esperado do CSV"):
        st.markdown("Substitua o arquivo `data/results.csv` pelos resultados reais do seu benchmark mantendo, no mínimo, estas colunas:")
        st.code("\n".join(REQUIRED_COLUMNS), language="text")
        st.markdown(
            """
            <div class="schema-box">
            <b>Recomendação:</b> mantenha uma linha por execução experimental. Assim, o leaderboard consegue calcular médias, número de runs, cobertura por dataset e rankings por estratégia de prompt.
            </div>
            """,
            unsafe_allow_html=True,
        )



# ============================================================
# Página 3 — Guia de cálculo das métricas e do ranking
# ============================================================

def error_breakdown_df(y_true: Iterable[float], y_pred: Iterable[float]) -> pd.DataFrame:
    true_arr = np.asarray(list(y_true), dtype=float)
    pred_arr = np.asarray(list(y_pred), dtype=float)
    abs_error = np.abs(pred_arr - true_arr)
    squared_error = (pred_arr - true_arr) ** 2
    denominator = np.abs(true_arr) + np.abs(pred_arr)
    smape_component = 2.0 * abs_error / np.maximum(denominator, 1e-8) * 100.0

    return pd.DataFrame(
        {
            "t": np.arange(1, len(true_arr) + 1),
            "y_true": true_arr,
            "y_pred": pred_arr,
            "erro": pred_arr - true_arr,
            "erro_absoluto": abs_error,
            "erro_quadratico": squared_error,
            "denominador_smape": denominator,
            "termo_smape_%": smape_component,
        }
    )


def page_metric_guide() -> None:
    st.title("📐 Guia: cálculo dos erros e do rank geral")
    st.caption("Demonstração transparente de SMAPE, MAE, RMSE e do score composto usado para ordenar os modelos no leaderboard.")

    st.markdown("### 1) Fórmulas usadas no benchmark")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**SMAPE ↓**")
        st.latex(r"SMAPE = \frac{100}{n}\sum_{t=1}^{n}\frac{2|\hat{y}_t-y_t|}{|y_t|+|\hat{y}_t|+\epsilon}")
        st.caption("É simétrico porque considera a escala do valor real e da previsão no denominador.")
    with c2:
        st.markdown("**MAE ↓**")
        st.latex(r"MAE = \frac{1}{n}\sum_{t=1}^{n}|\hat{y}_t-y_t|")
        st.caption("Mede o erro absoluto médio na unidade original da série.")
    with c3:
        st.markdown("**RMSE ↓**")
        st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(\hat{y}_t-y_t)^2}")
        st.caption("Penaliza erros grandes com mais força por usar erro ao quadrado.")

    st.info("No leaderboard, todas essas métricas são do tipo menor é melhor. O mesmo vale para tempo de inferência, CO₂ e custo financeiro.")

    st.markdown("### 2) Exemplo editável: cálculo de SMAPE, MAE e RMSE")
    c1, c2 = st.columns(2)
    with c1:
        y_true_text = st.text_area(
            "Valores reais, separados por vírgula",
            value="10.2, 11.0, 10.8, 12.1, 12.4",
            key="guide_y_true",
        )
    with c2:
        y_pred_text = st.text_area(
            "Previsões do modelo, separadas por vírgula",
            value="10.0, 10.9, 11.1, 12.0, 12.7",
            key="guide_y_pred",
        )

    try:
        y_true = parse_series(y_true_text)
        y_pred = parse_series(y_pred_text)
        if len(y_true) != len(y_pred):
            st.error(f"As listas precisam ter o mesmo tamanho. Valores reais: {len(y_true)}; previsões: {len(y_pred)}.")
            return
        if len(y_true) == 0:
            st.error("Informe pelo menos um valor real e uma previsão.")
            return

        breakdown = error_breakdown_df(y_true, y_pred)
        smape_value = smape(y_true, y_pred)
        mae_value = mae(y_true, y_pred)
        rmse_value = rmse(y_true, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("SMAPE", f"{smape_value:.4f}%")
        m2.metric("MAE", f"{mae_value:.6f}")
        m3.metric("RMSE", f"{rmse_value:.6f}")

        st.markdown("#### Decomposição ponto a ponto")
        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
            column_config={
                "y_true": st.column_config.NumberColumn("Real", format="%.6f"),
                "y_pred": st.column_config.NumberColumn("Previsto", format="%.6f"),
                "erro": st.column_config.NumberColumn("Erro", format="%.6f"),
                "erro_absoluto": st.column_config.NumberColumn("|erro|", format="%.6f"),
                "erro_quadratico": st.column_config.NumberColumn("Erro²", format="%.6f"),
                "denominador_smape": st.column_config.NumberColumn("|real| + |previsto|", format="%.6f"),
                "termo_smape_%": st.column_config.NumberColumn("Termo SMAPE (%)", format="%.6f"),
            },
        )

        plot_df = breakdown[["t", "y_true", "y_pred"]].melt("t", var_name="Série", value_name="Valor")
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("t:O", title="Passo de previsão"),
                y=alt.Y("Valor:Q", title="Valor"),
                color=alt.Color("Série:N", title="Série"),
                tooltip=["t", "Série", alt.Tooltip("Valor:Q", format=".6f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Como os números acima foram agregados"):
            st.code(
                f"""
SMAPE = média dos termos SMAPE (%) = {breakdown['termo_smape_%'].mean():.6f}
MAE   = média dos erros absolutos = {breakdown['erro_absoluto'].mean():.6f}
RMSE  = raiz da média dos erros quadráticos = sqrt({breakdown['erro_quadratico'].mean():.6f}) = {rmse_value:.6f}
                """.strip(),
                language="text",
            )
    except ValueError as exc:
        st.error(f"Erro ao converter valores numéricos: {exc}")
        return

    st.markdown("### 3) Como o rank geral é calculado")
    st.markdown(
        """
        O rank geral transforma métricas com escalas diferentes em uma escala comum de 0 a 100.
        Como todas as métricas são de minimização, o melhor valor recebe score maior.
        """
    )
    st.latex(r"score_m = 100 \times \frac{\max(x_m)-x_m}{\max(x_m)-\min(x_m)}")
    st.latex(r"score_{geral} = \sum_m score_m \times peso_m")

    weights, missing_policy = sidebar_weights("guide")
    df = load_results()
    if df.empty:
        st.warning("Nenhum `data/results.csv` encontrado para demonstrar o rank geral com modelos reais do app.")
        return

    filtered = filter_dataframe(df, "guide")
    rank_scope = st.radio(
        "Escopo do ranking demonstrado",
        PROMPT_ORDER,
        horizontal=True,
        index=0,
        key="guide_rank_scope",
    )
    if rank_scope != "Geral":
        filtered = filtered[filtered["prompt_strategy"] == rank_scope].copy()

    board = compute_leaderboard(filtered, weights, missing_policy)
    if board.empty:
        st.warning("Nenhum resultado disponível para os filtros selecionados.")
        return

    st.markdown("#### Tabela final do rank geral")
    st.dataframe(
        style_leaderboard_table(board.head(15)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score ↑": st.column_config.ProgressColumn("Score ↑", min_value=0, max_value=100, format="%.2f"),
            "SMAPE ↓": st.column_config.NumberColumn("SMAPE ↓", format="%.3f"),
            "MAE ↓": st.column_config.NumberColumn("MAE ↓", format="%.5f"),
            "RMSE ↓": st.column_config.NumberColumn("RMSE ↓", format="%.5f"),
            "Tempo (s) ↓": st.column_config.NumberColumn("Tempo (s) ↓", format="%.3f"),
            "CO₂ (g) ↓": st.column_config.NumberColumn("CO₂ (g) ↓", format="%.5f"),
            "Custo / 1k (US$) ↓": st.column_config.NumberColumn("Custo / 1k (US$) ↓", format="%.5f"),
        },
    )

    st.markdown("#### Decomposição do score de um modelo")
    selected_model = st.selectbox(
        "Escolha um modelo para auditar o score",
        board["model_name"].tolist(),
        index=0,
        key="guide_selected_model",
    )
    selected_row = board[board["model_name"] == selected_model].iloc[0]
    normalized_weights = normalize_weights(weights)

    explanation_rows = []
    for metric in METRICS:
        raw_value = selected_row.get(metric, np.nan)
        metric_min = board[metric].min(skipna=True) if metric in board.columns else np.nan
        metric_max = board[metric].max(skipna=True) if metric in board.columns else np.nan
        metric_score = selected_row.get(f"score_{metric}", np.nan)
        weight = normalized_weights.get(metric, 0.0)
        contribution = metric_score * weight if pd.notna(metric_score) else np.nan
        explanation_rows.append(
            {
                "Métrica": METRIC_LABELS[metric],
                "Valor do modelo": raw_value,
                "Melhor valor no conjunto": metric_min,
                "Pior valor no conjunto": metric_max,
                "Score normalizado": metric_score,
                "Peso normalizado": weight,
                "Contribuição no score": contribution,
            }
        )

    explanation_df = pd.DataFrame(explanation_rows)
    st.dataframe(
        explanation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Valor do modelo": st.column_config.NumberColumn("Valor do modelo", format="%.6f"),
            "Melhor valor no conjunto": st.column_config.NumberColumn("Melhor valor no conjunto", format="%.6f"),
            "Pior valor no conjunto": st.column_config.NumberColumn("Pior valor no conjunto", format="%.6f"),
            "Score normalizado": st.column_config.NumberColumn("Score normalizado", format="%.4f"),
            "Peso normalizado": st.column_config.NumberColumn("Peso normalizado", format="%.4f"),
            "Contribuição no score": st.column_config.NumberColumn("Contribuição no score", format="%.4f"),
        },
    )

    st.success(
        f"Score geral de {selected_model}: {selected_row['overall_score']:.4f}. "
        f"Rank: #{int(selected_row['rank'])} no escopo {rank_scope}."
    )

    with st.expander("Pseudocódigo do rank geral"):
        st.code(
            """
# 1. Agregar os resultados por modelo, usando a média das métricas.
board = media_por_modelo(results)

# 2. Para cada métrica de minimização, converter valor bruto em score 0-100.
score_metrica = 100 * (max_metrica - valor_modelo) / (max_metrica - min_metrica)

# 3. Normalizar os pesos para somarem 1.
peso_normalizado = peso / soma_dos_pesos

# 4. Calcular score geral.
score_geral = soma(score_metrica * peso_normalizado)

# 5. Ordenar do maior score geral para o menor.
rank = ordenar(score_geral, desc=True)
            """.strip(),
            language="python",
        )


# ============================================================
# Página 4 — Demo, upload CSV e submissão de resultados
# ============================================================


def page_demo() -> None:
    st.title("🧪 Demo & submissão: rodar um LLM e adicionar ao ranking")
    st.caption("Calcule métricas manualmente, envie um CSV de resultados/previsões e gere linhas compatíveis com `data/results.csv`.")

    if "submissions" not in st.session_state:
        st.session_state.submissions = []
    if "uploaded_submissions" not in st.session_state:
        st.session_state.uploaded_submissions = pd.DataFrame()

    st.markdown("### 1) Exemplo de prompt para previsão")
    with st.expander("Prompt base para LLM forecasting"):
        st.code(
            """
Você é um modelo de previsão de séries temporais.
Receberá uma série histórica numérica em formato CSV.
Tarefa: prever os próximos H valores.

Regras:
1. Responda apenas com uma lista CSV de números.
2. Não explique o raciocínio na resposta final.
3. Preserve escala e sazonalidade quando houver.
4. Não use informações externas ao histórico fornecido.

Série histórica:
{context_window}

Horizonte de previsão: {horizon}
Previsão:
            """.strip(),
            language="text",
        )

    st.markdown("### 2) Código mínimo para integrar seu avaliador")
    with st.expander("Exemplo de função de avaliação"):
        st.code(
            """
import time

start = time.perf_counter()

# 1. Monte o prompt com sua janela de contexto.
# 2. Chame o LLM local ou API.
# 3. Parseie a resposta para uma lista de floats.
y_pred = run_llm_forecast(model_id, context_window, horizon)

elapsed = time.perf_counter() - start

# y_true vem do split de teste do dataset.
result = {
    "model_name": model_name,
    "model_id": model_id,
    "dataset": dataset,
    "horizon": horizon,
    "prompt_strategy": prompt_strategy,
    "smape": smape(y_true, y_pred),
    "mae": mae(y_true, y_pred),
    "rmse": rmse(y_true, y_pred),
    "inference_time_s": elapsed,
    "co2_g": measured_or_estimated_co2,
    "cost_usd_per_1k_forecasts": api_or_energy_cost,
}
            """.strip(),
            language="python",
        )

    st.markdown("### 3) Submissão por CSV")
    st.info(
        "Para publicar novos resultados, o usuário precisa informar onde executou o modelo. "
        "Se for API, informe o custo por 1.000 previsões e a fonte do preço. Se for local, marque Local e informe hardware/quantização."
    )
    with st.expander("Esquemas aceitos para upload"):
        render_required_csv_schema()

    uploaded_file = st.file_uploader(
        "Suba um CSV de resultados calculados ou de previsões brutas",
        type=["csv"],
        key="submission_csv_upload",
    )

    st.markdown("#### Metadados obrigatórios da execução")
    m1, m2, m3 = st.columns(3)
    with m1:
        execution_source = st.selectbox(
            "Por onde rodou?",
            ["Local", "OpenAI API", "NVIDIA NIM API", "Hugging Face Inference API", "Together API", "Groq API", "Outro provedor API", "Híbrido"],
            index=0,
            key="upload_execution_source",
        )
        deployment_meta = "API" if "API" in execution_source or "OpenAI" in execution_source or "Groq" in execution_source or "Together" in execution_source else execution_source
        provider_meta = st.text_input("Provider/serviço", value=execution_source, key="upload_provider")
        api_pricing_source = st.text_input("Fonte do preço da API", value="", placeholder="Ex.: URL da tabela de preços ou contrato interno", key="upload_price_source")
    with m2:
        default_cost = 0.0 if execution_source == "Local" else 0.03
        cost_meta = st.number_input(
            "Custo / 1k previsões (US$)",
            min_value=0.0,
            value=default_cost,
            step=0.001,
            format="%.6f",
            key="upload_cost",
        )
        co2_meta = st.number_input("CO₂ por execução ou média (g)", min_value=0.0, value=0.0, step=0.01, format="%.6f", key="upload_co2")
        inference_time_meta = st.number_input("Tempo médio de inferência (s)", min_value=0.0, value=0.0, step=0.05, format="%.6f", key="upload_time")
    with m3:
        hardware_meta = st.text_input("Hardware", value="RTX 4090" if execution_source == "Local" else "API provider", key="upload_hardware")
        quantization_meta = st.text_input("Quantização", value="Q4" if execution_source == "Local" else "N/A", key="upload_quantization")
        energy_meta = st.number_input("Energia / 1k previsões (kWh, se local)", min_value=0.0, value=0.0, step=0.001, format="%.6f", key="upload_energy")

    st.markdown("#### Valores padrão para preencher colunas ausentes no CSV")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        default_model_name = st.text_input("Modelo padrão", value="Meu LLM TSF", key="csv_default_model")
        default_model_id = st.text_input("Model ID padrão", value="org/model-name", key="csv_default_model_id")
    with f2:
        default_family = st.text_input("Família padrão", value="Custom", key="csv_default_family")
        default_dataset = st.text_input("Dataset padrão", value="ETTh2", key="csv_default_dataset")
    with f3:
        default_horizon = st.number_input("Horizonte padrão", min_value=1, value=96, step=1, key="csv_default_horizon")
        default_prompt = st.selectbox("Estratégia padrão", ["zero-shot", "few-shot", "cot", "cot+few"], index=0, key="csv_default_prompt")
    with f4:
        default_run_id = st.number_input("Run ID padrão", min_value=1, value=1, step=1, key="csv_default_run")
        default_representation = st.selectbox("Representação padrão", ["CSV numeric", "plain text numeric", "textual series"], index=0, key="csv_default_repr")

    notes_meta = st.text_area(
        "Notas da submissão",
        value="uploaded through Streamlit submission page",
        key="upload_notes",
    )

    metadata = {
        "model_name": default_model_name,
        "model_id": default_model_id,
        "family": default_family,
        "provider": provider_meta,
        "deployment": deployment_meta,
        "dataset": default_dataset,
        "horizon": int(default_horizon),
        "prompt_strategy": default_prompt,
        "representation": default_representation,
        "run_id": int(default_run_id),
        "evaluation_date": str(date.today()),
        "inference_time_s": float(inference_time_meta),
        "co2_g": float(co2_meta),
        "cost_usd_per_1k_forecasts": float(cost_meta),
        "hardware": hardware_meta,
        "quantization": quantization_meta,
        "execution_source": execution_source,
        "energy_kwh_per_1k_forecasts": float(energy_meta),
        "api_pricing_source": api_pricing_source,
        "notes": notes_meta,
    }

    if execution_source != "Local" and cost_meta <= 0:
        st.warning("Para execução via API, informe um custo financeiro positivo por 1.000 previsões antes de publicar no ranking.")
    if execution_source == "Local" and not hardware_meta.strip():
        st.warning("Para execução local, informe o hardware usado para permitir comparação justa.")

    if uploaded_file is not None:
        try:
            raw_uploaded = pd.read_csv(uploaded_file)
            raw_uploaded = coerce_uploaded_csv(raw_uploaded)
            inferred_mode = infer_csv_mode(raw_uploaded)
            csv_mode = st.radio(
                "Tipo do CSV detectado",
                ["Resultados já calculados", "Previsões brutas"],
                index=0 if inferred_mode == "Resultados já calculados" else 1,
                horizontal=True,
                key="csv_mode",
            )
            st.markdown("#### Prévia do CSV enviado")
            st.dataframe(raw_uploaded.head(20), use_container_width=True, hide_index=True)

            if st.button("Validar e preparar submissão CSV", type="primary"):
                if execution_source != "Local" and cost_meta <= 0:
                    st.error("Submissões via API precisam informar custo financeiro positivo por 1.000 previsões.")
                    return
                if execution_source == "Local" and not hardware_meta.strip():
                    st.error("Submissões locais precisam informar o hardware utilizado.")
                    return

                if csv_mode == "Previsões brutas":
                    prepared = aggregate_predictions_csv(raw_uploaded, metadata)
                else:
                    prepared = prepare_results_csv(raw_uploaded, metadata)

                if prepared.empty:
                    st.error("Nenhuma linha válida foi gerada a partir do CSV.")
                    return

                st.session_state.uploaded_submissions = prepared
                st.success(f"CSV validado. {len(prepared)} linha(s) pronta(s) para submissão.")
        except Exception as exc:
            st.error(f"Não foi possível processar o CSV: {exc}")

    if not st.session_state.uploaded_submissions.empty:
        st.markdown("#### Submissão CSV preparada")
        prepared = st.session_state.uploaded_submissions.copy()
        st.dataframe(prepared, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Baixar CSV já padronizado",
            data=safe_download_csv(prepared),
            file_name="prepared_llm4tsf_submission.csv",
            mime="text/csv",
        )
        if st.button("Adicionar submissão CSV ao `data/results.csv`", type="primary"):
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            current = load_results(DATA_PATH)
            updated = pd.concat([current, prepared], ignore_index=True)
            updated.to_csv(DATA_PATH, index=False)
            load_results.clear()
            st.success("Submissão CSV adicionada a `data/results.csv`. Volte ao ranking principal para ver o rank recalculado.")

    st.markdown("### 4) Calcular métricas manualmente e criar uma linha")
    with st.form("submission_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_name = st.text_input("Nome do modelo", value="Meu LLM TSF")
            model_id = st.text_input("Model ID", value="org/model-name")
            family = st.text_input("Família", value="Custom")
            provider = st.text_input("Provider", value="Local")
            deployment = st.selectbox("Execução", ["Local", "API", "Hybrid"], index=0)
            execution_source_manual = st.selectbox("Por onde rodou?", ["Local", "OpenAI API", "NVIDIA NIM API", "Hugging Face Inference API", "Outro provedor API", "Híbrido"], index=0)
        with c2:
            dataset = st.text_input("Dataset", value="ETTh2")
            horizon = st.number_input("Horizonte", min_value=1, value=96, step=1)
            prompt_strategy = st.selectbox("Estratégia", ["zero-shot", "few-shot", "cot", "cot+few"], index=0)
            representation = st.selectbox("Representação", ["CSV numeric", "plain text numeric", "textual series"], index=0)
            run_id = st.number_input("Run ID", min_value=1, value=1, step=1)
        with c3:
            inference_time = st.number_input("Tempo de inferência (s)", min_value=0.0, value=1.25, step=0.05, format="%.4f")
            co2 = st.number_input("CO₂ estimado/medido (g)", min_value=0.0, value=0.85, step=0.01, format="%.5f")
            cost = st.number_input("Custo / 1k previsões (US$)", min_value=0.0, value=0.03, step=0.001, format="%.5f")
            hardware = st.text_input("Hardware", value="RTX 4090")
            quantization = st.text_input("Quantização", value="Q4")
            api_price_source_manual = st.text_input("Fonte do preço API", value="")

        y_true_text = st.text_area("Valores reais, separados por vírgula", value="10.2, 11.0, 10.8, 12.1, 12.4")
        y_pred_text = st.text_area("Previsões do LLM, separadas por vírgula", value="10.0, 10.9, 11.1, 12.0, 12.7")

        submitted = st.form_submit_button("Calcular métricas e gerar linha")

    if submitted:
        try:
            if execution_source_manual != "Local" and cost <= 0:
                st.error("Execuções via API precisam informar custo financeiro positivo por 1.000 previsões.")
                return
            y_true = parse_series(y_true_text)
            y_pred = parse_series(y_pred_text)
            if len(y_true) != len(y_pred):
                st.error(f"Tamanhos diferentes: y_true tem {len(y_true)} valores e y_pred tem {len(y_pred)} valores.")
                return
            if len(y_true) == 0:
                st.error("Informe pelo menos um valor real e uma previsão.")
                return

            row = {
                "model_name": model_name,
                "model_id": model_id,
                "family": family,
                "provider": provider,
                "deployment": deployment,
                "parameters_b": np.nan,
                "context_length_k": np.nan,
                "dataset": dataset,
                "horizon": int(horizon),
                "prompt_strategy": prompt_strategy,
                "representation": representation,
                "smape": round(smape(y_true, y_pred), 6),
                "mae": round(mae(y_true, y_pred), 6),
                "rmse": round(rmse(y_true, y_pred), 6),
                "inference_time_s": float(inference_time),
                "co2_g": float(co2),
                "cost_usd_per_1k_forecasts": float(cost),
                "run_id": int(run_id),
                "evaluation_date": str(date.today()),
                "hardware": hardware,
                "quantization": quantization,
                "execution_source": execution_source_manual,
                "api_pricing_source": api_price_source_manual,
                "notes": "created from Streamlit demo page",
            }
            st.session_state.submissions.append(row)
            st.success("Linha de resultado criada com sucesso.")
            st.json(row)
        except ValueError as exc:
            st.error(f"Erro ao converter valores numéricos: {exc}")

    if st.session_state.submissions:
        submissions_df = pd.DataFrame(st.session_state.submissions)
        st.markdown("### Submissões manuais nesta sessão")
        st.dataframe(submissions_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Baixar submissões manuais como CSV",
            data=safe_download_csv(submissions_df),
            file_name="new_llm4tsf_submissions.csv",
            mime="text/csv",
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("Adicionar submissões manuais ao CSV local", type="primary"):
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                current = load_results(DATA_PATH)
                updated = pd.concat([current, submissions_df], ignore_index=True)
                updated.to_csv(DATA_PATH, index=False)
                load_results.clear()
                st.success("Resultados adicionados a `data/results.csv`. Atualize a página principal para recalcular o ranking.")
        with c2:
            st.info("Em deploy público, prefira salvar submissões em um dataset/repositório controlado ou exigir revisão humana antes de publicar no ranking.")

    st.markdown("### 5) Boas práticas para submissões")
    st.markdown(
        """
        - Publique o protocolo de avaliação: datasets, splits, horizonte, número de runs e prompts.
        - Mantenha uma linha por execução para permitir média, SEM e auditoria.
        - Não misture resultados com telemetria ausente sem indicar a política de imputação.
        - Para modelos API sem energia/CO₂ confiável, marque a ausência explicitamente e use a política de ranking apropriada.
        - Inclua versão do modelo, data da avaliação, provider, hardware, quantização, custo da API e origem do preço quando aplicável.
        """
    )


# ============================================================
# Página 5 — Notebook de exemplo
# ============================================================

def page_notebook() -> None:
    st.title("📓 Notebook de exemplo")
    st.caption("Referência prática para executar um modelo, coletar previsões, medir tempo/energia/carbono e exportar resultados.")

    st.markdown(
        f"""
        Este leaderboard inclui uma página dedicada ao notebook de exemplo usado como referência de execução.

        **Notebook:** [Carbon_Qwen3_8b.ipynb]({NOTEBOOK_URL})

        Use esse notebook como ponto de partida para:

        - rodar um LLM em uma tarefa de previsão de séries temporais;
        - medir tempo de inferência;
        - registrar energia/carbono quando houver telemetria local;
        - exportar um CSV no formato aceito pela página de submissão.
        """
    )

    st.link_button("Abrir notebook no GitHub", NOTEBOOK_URL)
    st.link_button("Baixar versão raw do notebook", RAW_NOTEBOOK_URL)

    st.markdown("### Checklist para transformar o notebook em submissão")
    st.markdown(
        """
        1. Defina dataset, split, horizonte e estratégia de prompt.
        2. Salve `y_true` e `y_pred` para cada run ou calcule SMAPE/MAE/RMSE no próprio notebook.
        3. Registre `inference_time_s`, `hardware`, `quantization` e `evaluation_date`.
        4. Se usar API, registre `provider`, `cost_usd_per_1k_forecasts` e `api_pricing_source`.
        5. Exporte o CSV e suba na página **Demo & submissão**.
        """
    )

    with st.expander("Template mínimo de CSV gerado pelo notebook"):
        render_required_csv_schema()

# ============================================================
# Navegação
# ============================================================

def run_app() -> None:
    with st.sidebar:
        st.markdown("## 🏆 LLM4TSF")
        st.caption("Leaderboard para LLMs em previsão de séries temporais")
        st.divider()

    # st.navigation é a forma moderna recomendada pelo Streamlit. O fallback mantém compatibilidade com versões antigas.
    if hasattr(st, "navigation") and hasattr(st, "Page"):
        pages = {
            "Leaderboard": [
                st.Page(page_main, title="Ranking principal", icon="🏆"),
                st.Page(page_results, title="Resultados completos", icon="📊"),
                st.Page(page_metric_guide, title="Guia de cálculo", icon="📐"),
                st.Page(page_demo, title="Demo & submissão", icon="🧪"),
                st.Page(page_notebook, title="Notebook exemplo", icon="📓"),
            ]
        }
        pg = st.navigation(pages, position="sidebar", expanded=True)
        pg.run()
    else:
        selected = st.sidebar.radio(
            "Páginas",
            ["Ranking principal", "Resultados completos", "Guia de cálculo", "Demo & submissão", "Notebook exemplo"],
            index=0,
        )
        if selected == "Ranking principal":
            page_main()
        elif selected == "Resultados completos":
            page_results()
        elif selected == "Guia de cálculo":
            page_metric_guide()
        elif selected == "Demo & submissão":
            page_demo()
        else:
            page_notebook()


if __name__ == "__main__":
    run_app()

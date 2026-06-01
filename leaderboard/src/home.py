import streamlit as st
import pandas as pd
from streamlit_theme import st_theme

from config import (
  APP_SUBTITLE, APP_TITLE, METRIC_LABELS, METRICS, PROMPT_ORDER,
  compute_leaderboard, filter_results, normalize_weights, CHARACTERISTICS,
)
import ui as ui


st.set_page_config(page_title=APP_TITLE, layout="wide")

theme = st_theme() or {}
_is_dark = str(theme.get("base", "dark")).lower() == "dark"

_GRAD_DARK = (
  "radial-gradient(ellipse at 75% 15%, rgba(140,80,220,0.07) 0%, transparent 60%),"
  "radial-gradient(ellipse at 20% 80%, rgba(100,60,180,0.05) 0%, transparent 55%),"
  "linear-gradient(135deg, #262730 0%, #27262e 100%)"
)
_GRAD_LIGHT = (
  "radial-gradient(ellipse at 75% 15%, rgba(130,70,210,0.07) 0%, transparent 60%),"
  "radial-gradient(ellipse at 20% 80%, rgba(100,60,180,0.05) 0%, transparent 55%),"
  "linear-gradient(135deg, #f8f7fa 0%, #f5f4f8 100%)"
)
_GRAD_GREEN_DARK = (
  "radial-gradient(ellipse at 75% 15%, rgba(60,200,140,0.08) 0%, transparent 60%),"
  "radial-gradient(ellipse at 20% 80%, rgba(40,160,110,0.06) 0%, transparent 55%),"
  "linear-gradient(135deg, #252830 0%, #262930 100%)"
)
_GRAD_GREEN_LIGHT = (
  "radial-gradient(ellipse at 75% 15%, rgba(60,200,140,0.08) 0%, transparent 60%),"
  "radial-gradient(ellipse at 20% 80%, rgba(40,160,110,0.06) 0%, transparent 55%),"
  "linear-gradient(135deg, #f7f8fa 0%, #f4f7f6 100%)"
)
_CARD_BG = _GRAD_DARK if _is_dark else _GRAD_LIGHT
_CARD_BORDER = "1px solid rgba(140,80,220,0.08)" if _is_dark else "1px solid rgba(120,70,200,0.10)"
_CARD_BG_GREEN = _GRAD_GREEN_DARK if _is_dark else _GRAD_GREEN_LIGHT
_CARD_BORDER_GREEN = "1px solid rgba(60,200,140,0.10)" if _is_dark else "1px solid rgba(40,180,120,0.13)"

st.html(f"""
<style>
[class*="st-key-c1-"] {{
  border-radius: 24px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
  background: {_CARD_BG};
  border: {_CARD_BORDER};
  min-height: 13rem;
}}
[class*="st-key-c2-"] {{
  border-radius: 24px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
  background: {_CARD_BG_GREEN};
  border: {_CARD_BORDER_GREEN};
  min-height: 13rem;
}}
[class*="st-key-c1-col"],
[class*="st-key-c2-col"],
[class*="st-key-c2-guide_smape"],
[class*="st-key-c2-guide_mae"],
[class*="st-key-c2-guide_rmse"] {{
  display: flex;
  flex-direction: column;
}}
[class*="st-key-c1-col"] > div[data-testid="stVerticalBlock"],
[class*="st-key-c2-col"] > div[data-testid="stVerticalBlock"],
[class*="st-key-c2-guide_smape"] > div[data-testid="stVerticalBlock"],
[class*="st-key-c2-guide_mae"] > div[data-testid="stVerticalBlock"],
[class*="st-key-c2-guide_rmse"] > div[data-testid="stVerticalBlock"] {{
  display: flex;
  flex-direction: column;
  flex: 1;
}}
[class*="st-key-c1-col"] [data-testid="stHorizontalBlock"],
[class*="st-key-c2-col"] [data-testid="stHorizontalBlock"],
[class*="st-key-c2-guide_smape"] [data-testid="stHorizontalBlock"],
[class*="st-key-c2-guide_mae"] [data-testid="stHorizontalBlock"],
[class*="st-key-c2-guide_rmse"] [data-testid="stHorizontalBlock"] {{
  margin-top: auto !important;
}}
[class*="st-key-c1-banner"],
[class*="st-key-c2-banner"] {{
  min-height: unset;
}}
</style>
""")


def _format_currency(value: float) -> str:
  return f"US$ {value:.4f}"


def _summary_frame(board: pd.DataFrame) -> pd.DataFrame:
  return board.groupby("model", as_index=False).agg(
    mean_smape=("smape", "mean"),
    mean_cost=("financial_cost_usd", "mean"),
    mean_time=("inference_time_s", "mean"),
  ).sort_values(["mean_smape", "mean_cost", "mean_time"])


def page_main() -> None:
  weights, missing_policy = ui.sidebar_weights("main")

  with ui.CONTAINER(key="banner"):
    st.title(f"🏆 {APP_TITLE}")
    st.caption(APP_SUBTITLE)
    st.markdown("---")
    selected_ranking = ui.SELECT(options=PROMPT_ORDER, key="ranking")

  board_raw = filter_results(selected_ranking)
  board_raw = ui.sidebar_filters(board_raw, "main")

  if board_raw.empty:
    st.warning("No rows matched the current selection.")
    return

  board = compute_leaderboard(board_raw, weights, missing_policy)
  summary = _summary_frame(board)

  # Weight-aware leader: top of the ranked board
  leader_row = board.iloc[0]
  leader_model = leader_row["model"]
  leader_score = leader_row["overall_score"]

  # Best SMAPE: model with lowest mean SMAPE (independent metric)
  best_smape_row = summary.loc[summary["mean_smape"].idxmin()]

  # Lowest Cost: model with lowest mean financial cost (independent metric)
  best_cost_row = summary.loc[summary["mean_cost"].idxmin()]

  # Podium still uses summary sorted by [smape, cost, time] for display
  podium_leader = summary.iloc[0]
  runner_up = summary.iloc[1] if len(summary) > 1 else podium_leader
  third_place = summary.iloc[2] if len(summary) > 2 else runner_up

  metric_cols = st.columns(5)
  with metric_cols[0]:
    with ui.CONTAINER(key="col1"):
      st.markdown("Models")
      st.markdown(f"#### {board['model'].nunique()}")
      st.badge("after filters", color="gray")
  with metric_cols[1]:
    with ui.CONTAINER(key="col2"):
      st.markdown("Results")
      st.markdown(f"#### {len(board_raw):,}")
      st.badge("rows evaluated", color="gray")
  with metric_cols[2]:
    with ui.CONTAINER(key="col3"):
      st.markdown("Leader")
      st.markdown(f"#### {leader_model}")
      st.badge(f"score {leader_score:.2f}", color="orange")
  with metric_cols[3]:
    with ui.CONTAINER(key="col4"):
      st.markdown("Best SMAPE")
      st.markdown(f"#### {best_smape_row['mean_smape']:.2f}")
      st.badge(f"{best_smape_row['model']}", color="green")
  with metric_cols[4]:
    with ui.CONTAINER(key="col5"):
      st.markdown("Lowest Cost")
      st.markdown(f"#### {_format_currency(best_cost_row['mean_cost'])}")
      st.badge(f"{best_cost_row['model']}", color="blue")

  st.markdown("## Podium")
  podium_cols = st.columns(3)
  with podium_cols[0]:
    with ui.CONTAINER(key="col6", color="c2"):
      st.markdown("### 🥇")
      st.markdown(f"##### {podium_leader['model']}")
      st.badge(f"SMAPE {podium_leader['mean_smape']:.2f}", color="orange")
      st.badge(f"cost {_format_currency(podium_leader['mean_cost'])}", color="blue")
  with podium_cols[1]:
    with ui.CONTAINER(key="col7", color="c2"):
      st.markdown("### 🥈")
      st.markdown(f"##### {runner_up['model']}")
      st.badge(f"SMAPE {runner_up['mean_smape']:.2f}", color="orange")
      st.badge(f"cost {_format_currency(runner_up['mean_cost'])}", color="blue")
  with podium_cols[2]:
    with ui.CONTAINER(key="col8", color="c2"):
      st.markdown("### 🥉")
      st.markdown(f"##### {third_place['model']}")
      st.badge(f"SMAPE {third_place['mean_smape']:.2f}", color="orange")
      st.badge(f"cost {_format_currency(third_place['mean_cost'])}", color="blue")

  st.markdown("## Ranking")
  styled = ui.style_leaderboard(board)
  st.dataframe(
    styled, use_container_width=True, hide_index=True,
    column_config={
      "Score ↑": st.column_config.ProgressColumn("Score ↑", min_value=0, max_value=1, format="%.2f"),
      "SMAPE ↓": st.column_config.NumberColumn("SMAPE ↓", format="%.3f"),
      "MAE ↓": st.column_config.NumberColumn("MAE ↓", format="%.5f"),
      "RMSE ↓": st.column_config.NumberColumn("RMSE ↓", format="%.5f"),
      "Time (s) ↓": st.column_config.NumberColumn("Time (s) ↓", format="%.3f"),
      "Cost (US$) ↓": st.column_config.NumberColumn("Cost (US$) ↓", format="%.5f"),
      "Energy (Wh) ↓": st.column_config.NumberColumn("Energy (Wh) ↓", format="%.5f"),
    },
  )

  with st.expander("How the score is calculated"):
    formula_df = pd.DataFrame({
      "Metric": [METRIC_LABELS[m] for m in METRICS],
      "Configured weight": [weights[m] for m in METRICS],
      "Direction": ["lower is better" for _ in METRICS],
    })
    st.dataframe(formula_df, use_container_width=True, hide_index=True)
    st.latex(r"score_m = 100 \times \frac{\max(x_m) - x_m}{\max(x_m) - \min(x_m)}")
    st.latex(r"score_{overall} = \sum_m score_m \times \hat{w}_m")

  st.markdown("## Charts")
  st.badge(f"Filtered results: {selected_ranking}", color="gray")
  tabs = st.tabs(["Prompt Impact", "Model vs Dataset", "SMAPE Heatmap", "Format vs Series", "Efficiency Trade-off", "Inference Time", "Energy Consumption", "FTE Score", "Bubble Chart", "Top Models", "Pareto", "Cost vs Score"])

  with tabs[0]:
    st.altair_chart(ui.CHART_PROMPT_IMPACT(board_raw), use_container_width=True)
  with tabs[1]:
    st.altair_chart(ui.CHART_MEAN_SMAPE_MODEL_DATASET(board_raw), use_container_width=True)
  with tabs[2]:
    st.altair_chart(ui.CHART_SMAPE_HEATMAP(board_raw), use_container_width=True)
  with tabs[3]:
    st.altair_chart(ui.CHART_FORMAT_SERIES(board_raw), use_container_width=True)
  with tabs[4]:
    st.altair_chart(ui.CHART_EFFICIENCY_TRADEOFF(board_raw), use_container_width=True)
  with tabs[5]:
    st.altair_chart(ui.CHART_AVG_INFERENCE_TIME(board_raw), use_container_width=True)
  with tabs[6]:
    st.altair_chart(ui.CHART_AVG_ENERGY_CONSUMPTION(board_raw), use_container_width=True)
  with tabs[7]:
    st.altair_chart(ui.CHART_FTE_SCORE(board_raw), use_container_width=True)
  with tabs[8]:
    st.altair_chart(ui.CHART_BUBBLE_TRADEOFF(board_raw), use_container_width=True)
  with tabs[9]:
    st.altair_chart(ui.CHART_TOP_MODELS(board), use_container_width=True)
  with tabs[10]:
    st.caption("Best models sit in the lower-left: low error and low latency.")
    st.altair_chart(ui.CHART_PARETO(board), use_container_width=True)
  with tabs[11]:
    st.altair_chart(ui.CHART_COST_SCORE(board), use_container_width=True)


def page_results() -> None:
  RESULTS_SECTIONS = ["Raw Table", "Aggregated Views", "Coverage Diagnostics"]

  with ui.CONTAINER(key="banner_results"):
    st.title("📊 Full Results")
    st.caption("Raw experiment table, pivots by dataset, and export for auditing/reproducibility.")
    st.markdown("---")
    selected_section = ui.SELECT(options=RESULTS_SECTIONS, key="results_section")

  from config import RESULTS
  board_raw = ui.sidebar_filters(RESULTS.copy(), "results")

  if selected_section == "Raw Table":
    search = st.text_input("Search by model, dataset, family or deployment", placeholder="e.g.: Qwen, ETTh2, API...")
    table_df = board_raw.copy()
    if search:
      query = search.strip().lower()
      mask = pd.Series(False, index=table_df.index)
      for col in ["model", "dataset", "family", "deployment", "prompt_type", "format"]:
        if col in table_df.columns:
          mask = mask | table_df[col].astype(str).str.lower().str.contains(query, na=False)
      table_df = table_df[mask]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

  elif selected_section == "Aggregated Views":
    c1, c2 = st.columns(2)
    with c1:
      st.subheader("Mean error by model and strategy")
      pivot = pd.pivot_table(board_raw, index="model", columns="prompt_type", values="smape", aggfunc="mean").reset_index()
      st.dataframe(pivot, use_container_width=True, hide_index=True)
    with c2:
      st.subheader("Mean error by dataset")
      ds_summary = board_raw.groupby("dataset", dropna=False).agg(smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"), n_results=("model", "count"), n_models=("model", "nunique")).reset_index().sort_values("smape")
      st.dataframe(ds_summary, use_container_width=True, hide_index=True)

  elif selected_section == "Coverage Diagnostics":
    coverage = board_raw.groupby(["model", "prompt_type"], dropna=False).agg(n_datasets=("dataset", "nunique"), n_runs=("smape", "count"), smape=("smape", "mean")).reset_index()
    st.dataframe(coverage, use_container_width=True, hide_index=True)


def page_metric_guide() -> None:
  import numpy as np
  import math

  GUIDE_SECTIONS = ["Formulas", "Interactive Example", "Score Calculation"]

  with ui.CONTAINER(key="banner_guide"):
    st.title("📐 Metric Calculation Guide")
    st.caption("Transparent demonstration of SMAPE, MAE, RMSE and the composite score used to rank models.")
    st.markdown("---")
    selected_section = ui.SELECT(options=GUIDE_SECTIONS, key="guide_section")

  if selected_section == "Formulas":
    st.markdown("### 1) Formulas used in the benchmark")
    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown("**SMAPE ↓**")
      st.latex(r"SMAPE = \frac{100}{n}\sum_{t=1}^{n}\frac{2|\hat{y}_t-y_t|}{|y_t|+|\hat{y}_t|+\epsilon}")
      st.caption("Symmetric because it considers both actual and predicted scale in the denominator.")
    with c2:
      st.markdown("**MAE ↓**")
      st.latex(r"MAE = \frac{1}{n}\sum_{t=1}^{n}|\hat{y}_t-y_t|")
      st.caption("Measures mean absolute error in the original series unit.")
    with c3:
      st.markdown("**RMSE ↓**")
      st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(\hat{y}_t-y_t)^2}")
      st.caption("Penalizes large errors more heavily due to squared error.")
    st.info("In the leaderboard, all metrics are lower-is-better. The same applies to inference time, energy, and financial cost.")

  elif selected_section == "Interactive Example":
    st.markdown("### 2) Editable example: SMAPE, MAE, RMSE calculation")
    c1, c2 = st.columns(2)
    with c1:
      y_true_text = st.text_area("Actual values, comma-separated", value="10.2, 11.0, 10.8, 12.1, 12.4")
    with c2:
      y_pred_text = st.text_area("Model predictions, comma-separated", value="10.0, 10.9, 11.1, 12.0, 12.7")

    def _parse(text: str):
      clean = text.replace(";", ",").replace("\n", ",").strip()
      return [float(t.strip()) for t in clean.split(",") if t.strip()]

    try:
      y_true = np.array(_parse(y_true_text))
      y_pred = np.array(_parse(y_pred_text))
      if len(y_true) != len(y_pred):
        st.error(f"Lists must have the same length. Actual: {len(y_true)}; predictions: {len(y_pred)}.")
      elif len(y_true) == 0:
        st.error("Provide at least one actual value and one prediction.")
      else:
        denom = np.abs(y_true) + np.abs(y_pred)
        smape_val = float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-8)) * 100.0)
        mae_val = float(np.mean(np.abs(y_pred - y_true)))
        rmse_val = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        m1, m2, m3 = st.columns(3)
        with m1:
          with ui.CONTAINER(key="guide_smape", color="c2"):
            st.markdown("SMAPE")
            st.markdown(f"#### {smape_val:.4f}%")
            st.badge("lower is better", color="green")
        with m2:
          with ui.CONTAINER(key="guide_mae", color="c2"):
            st.markdown("MAE")
            st.markdown(f"#### {mae_val:.6f}")
            st.badge("lower is better", color="green")
        with m3:
          with ui.CONTAINER(key="guide_rmse", color="c2"):
            st.markdown("RMSE")
            st.markdown(f"#### {rmse_val:.6f}")
            st.badge("lower is better", color="green")
    except ValueError as exc:
      st.error(f"Error converting numeric values: {exc}")

  elif selected_section == "Score Calculation":
    weights, missing_policy = ui.sidebar_weights("guide")
    from config import RESULTS
    board_raw = ui.sidebar_filters(RESULTS.copy(), "guide")

    board = compute_leaderboard(board_raw, weights, missing_policy)
    if board.empty:
      st.warning("No results available for the selected filters.")
      return

    with st.expander("How the score is calculated"):
      norm = normalize_weights(weights)
      formula_df = pd.DataFrame({
        "Metric": [METRIC_LABELS[m] for m in METRICS],
        "Configured weight": [weights[m] for m in METRICS],
        "Direction": ["lower is better" for _ in METRICS],
      })
      st.dataframe(formula_df, use_container_width=True, hide_index=True)
      st.latex(r"score_m = 100 \times \frac{\max(x_m) - x_m}{\max(x_m) - \min(x_m)}")
      st.latex(r"score_{overall} = \sum_m score_m \times \hat{w}_m")

    st.markdown("#### Final rank table")
    st.dataframe(
      ui.style_leaderboard(board.head(15)), use_container_width=True, hide_index=True,
      column_config={
        "Score ↑": st.column_config.ProgressColumn("Score ↑", min_value=0, max_value=1, format="%.2f"),
        "SMAPE ↓": st.column_config.NumberColumn("SMAPE ↓", format="%.3f"),
        "MAE ↓": st.column_config.NumberColumn("MAE ↓", format="%.5f"),
      },
    )

    st.markdown("#### Score breakdown for a selected model")
    selected_model = st.selectbox("Choose a model to audit", board["model"].tolist(), index=0)
    row = board[board["model"] == selected_model].iloc[0]
    norm = normalize_weights(weights)
    explanation = []
    for m in METRICS:
      raw = row.get(m, np.nan)
      s = row.get(f"score_{m}", np.nan)
      w = norm.get(m, 0.0)
      contrib = s * w if (not math.isnan(float(s)) and not math.isnan(float(w))) else np.nan
      explanation.append({"Metric": METRIC_LABELS[m], "Model value": raw, "Best in set": board[m].min(skipna=True) if m in board.columns else np.nan, "Worst in set": board[m].max(skipna=True) if m in board.columns else np.nan, "Normalized score": s, "Weight": w, "Contribution": contrib})
    st.dataframe(pd.DataFrame(explanation), use_container_width=True, hide_index=True)
    st.success(f"Overall score of {selected_model}: {row['overall_score']:.4f}. Rank: #{int(row['rank'])}.")


def page_datasets() -> None:
  with ui.CONTAINER(key="banner_datasets"):
    st.title("📡 Dataset Characteristics")
    st.caption("Radar charts showing the structural characteristics of each benchmark dataset.")

  datasets = list(CHARACTERISTICS.keys())
  selected_ds = st.multiselect("Datasets to display", datasets, default=datasets)

  if not selected_ds:
    st.warning("Select at least one dataset.")
    return

  fig = ui.CHART_RADAR_DATASETS(selected_ds)
  st.plotly_chart(fig, use_container_width=True)

  st.markdown("### Characteristics Reference")
  char_df = pd.DataFrame(
    {ds: CHARACTERISTICS[ds] for ds in selected_ds},
    index=["Trend", "Seasonality", "Shifting", "Transition", "Non-Gaussianity"],
  ).T.reset_index().rename(columns={"index": "Dataset"})
  st.dataframe(char_df, use_container_width=True, hide_index=True)


def page_demo_submission() -> None:
  SUBMISSION_SECTIONS = ["Upload Results", "Upload Predictions", "Schema Reference"]

  with ui.CONTAINER(key="banner_submission"):
    st.title("🚀 Demo & Submission")
    st.caption("Submit your model results to the leaderboard — either as pre-computed metrics or raw predictions.")
    st.markdown("---")
    selected_section = ui.SELECT(options=SUBMISSION_SECTIONS, key="submission_section")

  if selected_section == "Upload Results":
    st.markdown("### Upload pre-computed results")
    st.info("Upload a CSV with already-computed SMAPE, MAE, RMSE and operational metrics. Each row represents one experiment run.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="upload_results")
    if uploaded is not None:
      try:
        df_upload = pd.read_csv(uploaded)
        st.success(f"File loaded: **{len(df_upload):,} rows × {len(df_upload.columns)} columns**")
        st.dataframe(df_upload.head(10), use_container_width=True, hide_index=True)

        with st.expander("Column mapping preview"):
          st.caption("Detected columns and their mapping to the leaderboard schema.")
          mapping = {c: c for c in df_upload.columns}
          st.json(mapping)

        st.markdown("#### Metadata (fill in missing fields)")
        c1, c2, c3 = st.columns(3)
        with c1:
          meta_family = st.text_input("Family", placeholder="e.g. Qwen, Llama, Mistral")
          meta_deployment = st.selectbox("Deployment", ["API", "Local", "Cloud", "Edge"])
        with c2:
          meta_provider = st.text_input("Provider", placeholder="e.g. OpenAI, HuggingFace")
          meta_hardware = st.text_input("Hardware", placeholder="e.g. A100 80GB, RTX 4090")
        with c3:
          meta_quantization = st.text_input("Quantization", placeholder="e.g. fp16, int8, none")
          meta_notes = st.text_area("Notes", placeholder="Optional notes about this submission", height=68)

        if st.button("✅ Validate & Preview Leaderboard Impact", type="primary"):
          from config import RESULTS, compute_leaderboard
          required = ["model", "dataset", "smape", "mae", "rmse"]
          missing_cols = [c for c in required if c not in df_upload.columns]
          if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
          else:
            st.success("Schema validated! Preview below shows how this submission would rank.")
            st.dataframe(df_upload.describe(), use_container_width=True)
      except Exception as exc:
        st.error(f"Could not read file: {exc}")
    else:
      st.markdown("""
      **How it works:**
      1. Prepare your CSV following the schema in the *Schema Reference* tab
      2. Upload it here and fill in any missing metadata
      3. Validate and preview how your model would rank
      4. Submit for review — results are added after manual verification
      """)

  elif selected_section == "Upload Predictions":
    st.markdown("### Upload raw predictions")
    st.info("Upload a CSV with `y_true` and `y_pred` columns. SMAPE, MAE and RMSE will be computed automatically.")

    uploaded_pred = st.file_uploader("Choose a predictions CSV", type=["csv"], key="upload_predictions")

    c1, c2 = st.columns(2)
    with c1:
      pred_model = st.text_input("Model name *", placeholder="e.g. Qwen3-8B")
      pred_dataset = st.text_input("Dataset *", placeholder="e.g. ETTh2, Carbon, Traffic")
      pred_horizon = st.number_input("Forecast horizon *", min_value=1, value=96, step=1)
    with c2:
      pred_strategy = st.selectbox("Prompt strategy *", ["Zero-shot", "Few-shot", "CoT", "CoT + Few-shot"])
      pred_family = st.text_input("Model family", placeholder="e.g. Qwen")
      pred_deployment = st.selectbox("Deployment", ["API", "Local", "Cloud", "Edge"])

    if uploaded_pred is not None:
      try:
        import numpy as np
        df_pred = pd.read_csv(uploaded_pred)
        st.success(f"File loaded: **{len(df_pred):,} rows**")

        has_true = "y_true" in df_pred.columns or "actual" in df_pred.columns
        has_pred = "y_pred" in df_pred.columns or "prediction" in df_pred.columns

        if not has_true or not has_pred:
          st.error("File must contain `y_true`/`actual` and `y_pred`/`prediction` columns.")
        else:
          col_true = "y_true" if "y_true" in df_pred.columns else "actual"
          col_pred = "y_pred" if "y_pred" in df_pred.columns else "prediction"
          y_true = pd.to_numeric(df_pred[col_true], errors="coerce").dropna().values
          y_pred = pd.to_numeric(df_pred[col_pred], errors="coerce").dropna().values
          n = min(len(y_true), len(y_pred))
          y_true, y_pred = y_true[:n], y_pred[:n]

          denom = np.abs(y_true) + np.abs(y_pred)
          smape_val = float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-8)) * 100.0)
          mae_val = float(np.mean(np.abs(y_pred - y_true)))
          rmse_val = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

          m1, m2, m3 = st.columns(3)
          with m1:
            with ui.CONTAINER(key="sub_smape", color="c2"):
              st.markdown("SMAPE")
              st.markdown(f"#### {smape_val:.4f}%")
              st.badge("computed", color="green")
          with m2:
            with ui.CONTAINER(key="sub_mae", color="c2"):
              st.markdown("MAE")
              st.markdown(f"#### {mae_val:.6f}")
              st.badge("computed", color="green")
          with m3:
            with ui.CONTAINER(key="sub_rmse", color="c2"):
              st.markdown("RMSE")
              st.markdown(f"#### {rmse_val:.6f}")
              st.badge("computed", color="green")

          st.dataframe(df_pred.head(8), use_container_width=True, hide_index=True)
      except Exception as exc:
        st.error(f"Could not process file: {exc}")
    else:
      st.markdown("""
      **Expected format:**
      ```
      y_true,y_pred
      10.2,10.0
      11.0,10.9
      10.8,11.1
      ```
      Each row is one timestep. The metrics are averaged across all rows.
      """)

  elif selected_section == "Schema Reference":
    st.markdown("### CSV Schema Reference")

    tab1, tab2 = st.tabs(["Format A — Pre-computed results", "Format B — Raw predictions"])
    with tab1:
      st.caption("Use this format when you already have SMAPE, MAE, RMSE computed per experiment.")
      schema_a = pd.DataFrame({
        "Column": ["model", "dataset", "prompt_type", "format", "series_type", "smape", "mae", "rmse",
                   "inference_time_s", "financial_cost_usd", "eco2ai_energy_wh", "family", "deployment"],
        "Type": ["str", "str", "str", "str", "str", "float", "float", "float",
                 "float", "float", "float", "str", "str"],
        "Required": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅",
                     "⬜", "⬜", "⬜", "⬜", "⬜"],
        "Example": ["Qwen/qwen3-8b", "ETTh2", "Zero-shot", "CSV", "Numeric", "12.34", "0.0021", "0.0032",
                    "2.15", "0.00012", "0.00045", "Qwen", "API"],
      })
      st.dataframe(schema_a, use_container_width=True, hide_index=True)
      st.code(
        "model,dataset,prompt_type,format,series_type,smape,mae,rmse,inference_time_s,financial_cost_usd,family,deployment\n"
        "Qwen/qwen3-8b,ETTh2,Zero-shot,CSV,Numeric,12.34,0.0021,0.0032,2.15,0.00012,Qwen,API",
        language="text",
      )

    with tab2:
      st.caption("Use this format when submitting raw forecasts. Metrics are computed automatically.")
      schema_b = pd.DataFrame({
        "Column": ["model", "dataset", "horizon", "prompt_type", "run_id", "y_true", "y_pred"],
        "Type": ["str", "str", "int", "str", "int", "float", "float"],
        "Required": ["✅", "✅", "✅", "✅", "✅", "✅", "✅"],
        "Example": ["Qwen/qwen3-8b", "ETTh2", "96", "Zero-shot", "1", "10.2", "10.0"],
      })
      st.dataframe(schema_b, use_container_width=True, hide_index=True)
      st.code(
        "model,dataset,horizon,prompt_type,run_id,y_true,y_pred\n"
        "Qwen/qwen3-8b,ETTh2,96,Zero-shot,1,10.2,10.0\n"
        "Qwen/qwen3-8b,ETTh2,96,Zero-shot,1,11.0,10.9",
        language="text",
      )

    st.markdown("### Prompt strategy labels")
    strategy_df = pd.DataFrame({
      "Accepted values": ["Zero-shot", "Few-shot", "CoT", "CoT + Few-shot"],
      "Aliases": ["zero, zeroshot, zero-shot", "few, fewshot, few-shot", "chain-of-thought, cot", "cot+few, few-shot-cot, fs-cot"],
    })
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)


def page_notebook() -> None:
  NOTEBOOK_SECTIONS = ["Quick Start", "Full Pipeline", "API Reference"]

  with ui.CONTAINER(key="banner_notebook"):
    st.title("📓 Notebook Example")
    st.caption("Reproducible code snippets to run evaluations and submit results to the leaderboard.")
    st.markdown("---")
    selected_section = ui.SELECT(options=NOTEBOOK_SECTIONS, key="notebook_section")

  if selected_section == "Quick Start":
    st.markdown("### Quick Start — run your first evaluation in 5 minutes")

    with st.expander("1. Install dependencies", expanded=True):
      st.code("""pip install pandas numpy openai transformers datasets""", language="bash")

    with st.expander("2. Load a benchmark dataset", expanded=True):
      st.code("""
import pandas as pd

# ETTh2 — a standard benchmark used in LLM4TSF
df = pd.read_csv("data/ETTh2.csv", parse_dates=["date"], index_col="date")
series = df["OT"].values          # target column
train, test = series[:-96], series[-96:]   # 96-step horizon
print(f"Train: {len(train)} steps | Test: {len(test)} steps")
""", language="python")

    with st.expander("3. Build a zero-shot prompt", expanded=True):
      st.code("""
import numpy as np

CONTEXT_LEN = 512

def build_zero_shot_prompt(history: np.ndarray, horizon: int) -> str:
    values = ", ".join(f"{v:.4f}" for v in history[-CONTEXT_LEN:])
    return (
        f"You are a time series forecasting expert.\\n"
        f"Given the following {len(history[-CONTEXT_LEN:])} historical observations:\\n"
        f"{values}\\n\\n"
        f"Predict the next {horizon} values. "
        f"Return only a comma-separated list of numbers, nothing else."
    )

prompt = build_zero_shot_prompt(train, horizon=96)
print(prompt[:300], "...")
""", language="python")

    with st.expander("4. Call the model and parse predictions", expanded=True):
      st.code("""
import re
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def forecast(prompt: str, model: str = "gpt-4o-mini") -> list[float]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    tokens = re.split(r"[,\\s]+", raw)
    return [float(t) for t in tokens if t]

predictions = forecast(prompt)
print(f"Got {len(predictions)} predictions")
""", language="python")

    with st.expander("5. Compute metrics", expanded=True):
      st.code("""
import numpy as np

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(denom, 1e-8)) * 100.0)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))

n = min(len(test), len(predictions))
print(f"SMAPE : {smape(test[:n], predictions[:n]):.4f}%")
print(f"MAE   : {mae(test[:n], predictions[:n]):.6f}")
print(f"RMSE  : {rmse(test[:n], predictions[:n]):.6f}")
""", language="python")

  elif selected_section == "Full Pipeline":
    st.markdown("### Full evaluation pipeline — all prompt strategies & datasets")

    with st.expander("Experiment loop", expanded=True):
      st.code("""
import time, csv, itertools
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")
DATASETS   = {"ETTh2": "data/ETTh2.csv", "Carbon": "data/carbon.csv"}
STRATEGIES = ["Zero-shot", "Few-shot", "CoT", "CoT + Few-shot"]
HORIZONS   = [96, 192, 336, 720]
MODEL      = "gpt-4o-mini"
RUNS       = 3     # repeated runs for SEM estimation

def build_prompt(history, horizon, strategy, examples=None):
    base = (
        f"You are a time series forecasting expert.\\n"
        f"Historical values: {', '.join(f'{v:.4f}' for v in history[-512:])}\\n"
        f"Predict the next {horizon} values as a comma-separated list."
    )
    if strategy == "CoT":
        base += "\\nThink step by step before giving your answer."
    if strategy in ("Few-shot", "CoT + Few-shot") and examples:
        ex_str = "\\n".join(f"Input: {e['input']} → Output: {e['output']}" for e in examples)
        base = f"Examples:\\n{ex_str}\\n\\n" + base
    return base

results = []
for (ds_name, path), strategy, horizon in itertools.product(DATASETS.items(), STRATEGIES, HORIZONS):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    series = df.iloc[:, 0].values
    train, test = series[:-horizon], series[-horizon:]

    for run in range(RUNS):
        prompt = build_prompt(train, horizon, strategy)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        raw = response.choices[0].message.content.strip()
        preds = [float(x) for x in raw.replace("\\n", ",").split(",") if x.strip()]
        n = min(len(test), len(preds))
        denom = np.abs(test[:n]) + np.abs(preds[:n])
        smape = float(np.mean(2 * np.abs(np.array(preds[:n]) - test[:n]) / np.maximum(denom, 1e-8)) * 100)
        mae   = float(np.mean(np.abs(np.array(preds[:n]) - test[:n])))
        rmse  = float(np.sqrt(np.mean((np.array(preds[:n]) - test[:n]) ** 2)))
        results.append({
            "model": MODEL, "dataset": ds_name, "prompt_type": strategy,
            "forecast_horizon": horizon, "run": run + 1,
            "smape": smape, "mae": mae, "rmse": rmse, "inference_time_s": elapsed,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        })
        print(f"[{ds_name}] {strategy} h={horizon} run={run+1} → SMAPE={smape:.2f}%")

pd.DataFrame(results).to_csv("results.csv", index=False)
print("Saved results.csv")
""", language="python")

    with st.expander("Aggregate and format for submission", expanded=True):
      st.code("""
import pandas as pd

df = pd.read_csv("results.csv")

# Mean and SEM per (model, dataset, prompt_type, horizon)
agg = df.groupby(["model", "dataset", "prompt_type", "forecast_horizon"]).agg(
    smape=("smape", "mean"),
    mae=("mae", "mean"),
    rmse=("rmse", "mean"),
    inference_time_s=("inference_time_s", "mean"),
    smape_sem=("smape", "sem"),
    mae_sem=("mae", "sem"),
    rmse_sem=("rmse", "sem"),
    input_tokens=("input_tokens", "mean"),
    output_tokens=("output_tokens", "mean"),
).reset_index()

# Add required metadata columns
agg["family"]     = "GPT"
agg["deployment"] = "API"
agg["format"]     = "CSV"
agg["series_type"] = "Numeric"

agg.to_csv("submission.csv", index=False)
print(agg.head())
""", language="python")

  elif selected_section == "API Reference":
    st.markdown("### Metric functions — copy-paste ready")

    c1, c2 = st.columns(2)
    with c1:
      st.markdown("**SMAPE**")
      st.code("""
def smape(y_true, y_pred):
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.abs(y_true) + np.abs(y_pred)
    return float(
        np.mean(2.0 * np.abs(y_pred - y_true)
        / np.maximum(denom, 1e-8)) * 100.0
    )
""", language="python")

      st.markdown("**MAE**")
      st.code("""
def mae(y_true, y_pred):
    import numpy as np
    return float(np.mean(
        np.abs(np.asarray(y_pred) - np.asarray(y_true))
    ))
""", language="python")

    with c2:
      st.markdown("**RMSE**")
      st.code("""
def rmse(y_true, y_pred):
    import numpy as np
    return float(np.sqrt(np.mean(
        (np.asarray(y_pred) - np.asarray(y_true)) ** 2
    )))
""", language="python")

      st.markdown("**Composite score (leaderboard formula)**")
      st.code("""
def normalize_score(values, higher_is_better=False):
    import numpy as np
    v = np.asarray(values, dtype=float)
    mn, mx = np.nanmin(v), np.nanmax(v)
    if np.isclose(mn, mx):
        return np.ones_like(v) * 100.0
    raw = 100.0 * (mx - v) / (mx - mn)  # lower-is-better
    return raw if not higher_is_better else 100.0 - raw

def composite_score(metrics_dict, weights):
    # metrics_dict: {metric: array_of_model_values}
    # weights: {metric: float}
    import numpy as np
    total_w = sum(max(0.0, w) for w in weights.values())
    norm_w  = {m: max(0.0, w) / total_w for m, w in weights.items()}
    scores  = {m: normalize_score(v) for m, v in metrics_dict.items()}
    return sum(scores[m] * norm_w[m] for m in weights)
""", language="python")

    st.markdown("### Supported datasets")
    datasets_df = pd.DataFrame({
      "Dataset": ["ETTh2", "Electricity", "Traffic", "Covid-19", "Wike2000", "Retail", "Carbon"],
      "Domain": ["Energy", "Energy", "Transport", "Health", "Web", "Commerce", "Environment"],
      "Frequency": ["Hourly", "Hourly", "Hourly", "Daily", "Daily", "Weekly", "Daily"],
      "Horizons tested": ["96, 192, 336, 720"] * 7,
      "Series count": ["1", "321", "862", "~200", "~1k", "~800", "1"],
    })
    st.dataframe(datasets_df, use_container_width=True, hide_index=True)

    st.markdown("### Prompt strategy reference")
    prompt_df = pd.DataFrame({
      "Strategy": ["Zero-shot", "Few-shot", "CoT", "CoT + Few-shot"],
      "Description": [
        "No examples, no chain-of-thought. Direct instruction only.",
        "2–8 in-context examples sampled from the training set.",
        "Model is instructed to reason step-by-step before predicting.",
        "Combines chain-of-thought reasoning with in-context examples.",
      ],
      "Key label": ["Zero-shot", "Few-shot", "CoT", "CoT + Few-shot"],
    })
    st.dataframe(prompt_df, use_container_width=True, hide_index=True)


def run_app() -> None:
  with st.sidebar:
    st.markdown("## 🏆 LLM4TSF")
    st.caption("Leaderboard for LLMs in time series forecasting")
    st.divider()

  if hasattr(st, "navigation") and hasattr(st, "Page"):
    pages = {
      "Leaderboard": [
        st.Page(page_main, title="Main Ranking", icon="🏆"),
        st.Page(page_results, title="Full Results", icon="📊"),
        st.Page(page_metric_guide, title="Metric Guide", icon="📐"),
        st.Page(page_datasets, title="Dataset Characteristics", icon="📡"),
      ],
      "Community": [
        st.Page(page_demo_submission, title="Demo & Submission", icon="🚀"),
        st.Page(page_notebook, title="Notebook Example", icon="📓"),
      ],
    }
    pg = st.navigation(pages, position="sidebar", expanded=True)
    pg.run()
  else:
    _all_pages = ["Main Ranking", "Full Results", "Metric Guide", "Dataset Characteristics", "Demo & Submission", "Notebook Example"]
    selected = st.sidebar.radio("Pages", _all_pages, index=0)
    if selected == "Main Ranking":
      page_main()
    elif selected == "Full Results":
      page_results()
    elif selected == "Metric Guide":
      page_metric_guide()
    elif selected == "Dataset Characteristics":
      page_datasets()
    elif selected == "Demo & Submission":
      page_demo_submission()
    else:
      page_notebook()


run_app()

from contextlib import contextmanager
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
  DEFAULT_WEIGHTS, ENERGY_MODEL_EXCLUSIONS, FAMILY_STYLE, METRIC_LABELS,
  METRICS, MODEL_PARAMS_B, PROMPT_CHART_ORDER, CHARACTERISTICS,
)


@contextmanager
def CONTAINER(*args, key: str, color: str = "c1", **kwargs):
  with st.container(key=f"{color}-{key}", *args, **kwargs) as container:
    yield container


def SELECT(options: list[str], key: str):
  state_key = f"s1-{key}"
  if state_key not in st.session_state:
    st.session_state[state_key] = options[0]
  cols = st.columns(len(options))
  for col, option in zip(cols, options):
    with col:
      is_selected = st.session_state[state_key] == option
      if st.button(option, key=f"{state_key}-{option}", type="primary" if is_selected else "tertiary", width="stretch"):
        st.session_state[state_key] = option
        st.rerun()
  return st.session_state[state_key]


def sidebar_weights(prefix: str = "") -> tuple[dict, str]:
  with st.sidebar:
    st.markdown("### Ranking Formula")
    st.caption("All metrics are normalized as lower is better. Adjust weights to reflect benchmark priorities.")
    weights = {}
    for m, default in DEFAULT_WEIGHTS.items():
      weights[m] = st.slider(METRIC_LABELS[m], 0.0, 1.0, round(default / 100, 2), 0.05, key=f"{prefix}_w_{m}")
    missing_policy = st.selectbox(
      "Missing metric policy",
      ["Penalize as worst", "Fill with median", "Ignore in score"],
      index=0,
      key=f"{prefix}_policy",
    )
  return weights, missing_policy


def sidebar_filters(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
  with st.sidebar:
    st.markdown("### Filters")
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    families = sorted(df["family"].dropna().unique().tolist())
    horizons = sorted([int(x) for x in df["forecast_horizon"].dropna().unique().tolist()])
    formats = sorted(df["format_label"].dropna().unique().tolist())

    sel_datasets = st.multiselect("Datasets", datasets, default=datasets, key=f"{prefix}_ds")
    sel_horizons = st.multiselect("Horizons", horizons, default=horizons, key=f"{prefix}_hz")
    sel_families = st.multiselect("Families", families, default=families, key=f"{prefix}_fam")
    sel_formats = st.multiselect("Formats", formats, default=formats, key=f"{prefix}_fmt")

  mask = (
    df["dataset"].isin(sel_datasets) &
    df["forecast_horizon"].isin(sel_horizons) &
    df["family"].isin(sel_families) &
    df["format_label"].isin(sel_formats)
  )
  return df[mask].copy()


def style_leaderboard(board: pd.DataFrame) -> pd.DataFrame:
  cols = ["rank", "model", "overall_score", "smape", "mae", "rmse", "inference_time_s", "financial_cost_usd", "eco2ai_energy_wh", "deployment", "family", "n_runs", "n_datasets"]
  available = [c for c in cols if c in board.columns]
  out = board[available].copy()
  return out.rename(columns={
    "rank": "Rank", "model": "Model", "overall_score": "Score ↑",
    "smape": "SMAPE ↓", "mae": "MAE ↓", "rmse": "RMSE ↓",
    "inference_time_s": "Time (s) ↓", "financial_cost_usd": "Cost (US$) ↓",
    "eco2ai_energy_wh": "Energy (Wh) ↓", "deployment": "Deployment",
    "family": "Family", "n_runs": "Runs", "n_datasets": "Datasets",
  })


def _ordered_unique(values: pd.Series) -> list[str]:
  return [v for v in pd.unique(values.dropna())]


def _prompt_order(values: pd.Series) -> list[str]:
  present = set(values.dropna())
  return [label for label in PROMPT_CHART_ORDER if label in present]


def _normalize_model_key(value: str) -> str:
  slug = str(value).lower().split("/")[-1]
  for suffix in ["-it", "-instruct", "-reasoning", "-2506", "-0905", "-2025-08-07"]:
    slug = slug.replace(suffix, "")
  return slug.strip()


def _extract_size_label(value: str) -> str:
  slug = _normalize_model_key(value)
  if "kimi-k2" in slug:
    return "1T"
  if "glm-5.1" in slug:
    return "754B"
  match = re.search(r"(\d+(?:\.\d+)?)b", slug)
  return f"{match.group(1).upper()}B" if match else "?"


def _lookup_params_b(value: str) -> float:
  slug = _normalize_model_key(value)
  for key, params_b in MODEL_PARAMS_B.items():
    if key in slug:
      return params_b
  matches = re.findall(r"(\d+(?:\.\d+)?)b", slug)
  return float(matches[-1]) if matches else np.nan


def _chart_colors() -> dict[str, str]:
  base = str(st.get_option("theme.base") or "dark").lower()
  background = st.get_option("theme.secondaryBackgroundColor") or st.get_option("theme.backgroundColor") or "#0E1117"
  text_color = st.get_option("theme.textColor") or ("#F8FAFC" if base == "dark" else "#111827")
  grid_color = "#334155" if base == "dark" else "#E5E7EB"
  muted_text = "#94A3B8" if base == "dark" else "#64748B"
  return {"background": background, "text": text_color, "grid": grid_color, "muted_text": muted_text}


def _hex_to_rgba(color: str, alpha: float) -> str:
  color = color.lstrip("#")
  if len(color) != 6:
    return color
  r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
  return f"rgba({r}, {g}, {b}, {alpha})"


def _apply_plotly_theme(fig: go.Figure, *, height: int) -> go.Figure:
  colors = _chart_colors()
  fig.update_layout(
    height=height, margin=dict(l=20, r=20, t=30, b=40),
    paper_bgcolor=colors["background"], plot_bgcolor=colors["background"],
    font=dict(color=colors["text"]),
    legend=dict(title_font=dict(color=colors["text"]), font=dict(color=colors["text"])),
  )
  fig.update_xaxes(gridcolor=colors["grid"], linecolor=colors["grid"], tickfont=dict(color=colors["text"]), title_font=dict(color=colors["text"]))
  fig.update_yaxes(gridcolor=colors["grid"], linecolor=colors["grid"], tickfont=dict(color=colors["text"]), title_font=dict(color=colors["text"]))
  return fig


def _chart_base(chart: alt.Chart, *, height: int | None = None, width: int | None = None, hide_axis: bool = False) -> alt.Chart:
  colors = _chart_colors()
  chart = chart.configure_view(stroke=None)
  chart = chart.configure(background=colors["background"])
  chart = chart.configure_axis(labelColor=colors["text"], titleColor=colors["text"], gridColor=colors["grid"], domainColor=colors["grid"], tickColor=colors["grid"])
  chart = chart.configure_legend(labelColor=colors["text"], titleColor=colors["text"])
  chart = chart.configure_title(color=colors["text"])
  if hide_axis:
    chart = chart.configure_axis(labelOpacity=0, titleOpacity=0, gridOpacity=0, domainOpacity=0, tickOpacity=0)
  props = {}
  if height is not None:
    props["height"] = height
  if width is not None:
    props["width"] = width
  if props:
    chart = chart.properties(**props)
  return chart


def _empty_chart(message: str) -> alt.Chart:
  frame = pd.DataFrame({"message": [message]})
  colors = _chart_colors()
  return _chart_base(alt.Chart(frame).mark_text(fontSize=16, color=colors["muted_text"]).encode(text="message:N"), height=220)


def _empty_plotly_chart(message: str) -> go.Figure:
  colors = _chart_colors()
  fig = go.Figure()
  fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=colors["muted_text"], size=16))
  fig.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor=colors["background"], plot_bgcolor=colors["background"], xaxis=dict(visible=False), yaxis=dict(visible=False))
  return fig


def _radar_figure(panels: list[dict], *, columns: int = 4) -> go.Figure:
  colors = _chart_colors()
  labels = ["Trend", "Seasonality", "Shifting", "Transition", "Non-Gaussianity"]
  rows = max(1, (len(panels) + columns - 1) // columns)
  subplot_titles = [str(p["title"]) for p in panels]
  subplot_titles.extend([""] * (rows * columns - len(subplot_titles)))
  specs = [[{"type": "polar"} for _ in range(columns)] for _ in range(rows)]
  fig = make_subplots(rows=rows, cols=columns, specs=specs, subplot_titles=subplot_titles, horizontal_spacing=0.03, vertical_spacing=0.12)
  for idx, panel in enumerate(panels):
    row, col = idx // columns + 1, idx % columns + 1
    values = list(map(float, panel["values"]))
    closed_values = values + values[:1]
    color = str(panel["color"])
    fig.add_trace(go.Scatterpolar(
      r=closed_values, theta=labels + labels[:1], mode="lines+markers",
      line=dict(color=color, width=2), marker=dict(color=color, size=5),
      fill="toself", fillcolor=_hex_to_rgba(color, 0.18),
      hovertemplate="%{theta}: %{r:.3f}<extra></extra>", showlegend=False,
    ), row=row, col=col)
  fig.update_annotations(font=dict(color=colors["text"], size=10))
  fig.update_polars(
    bgcolor=colors["background"],
    radialaxis=dict(range=[0, 1.05], visible=False, gridcolor=colors["grid"], linecolor=colors["grid"]),
    angularaxis=dict(tickmode="array", tickvals=[0, 72, 144, 216, 288], ticktext=labels, direction="clockwise", rotation=90, gridcolor=colors["grid"], linecolor=colors["grid"], tickfont=dict(color=colors["text"], size=9)),
  )
  fig.update_layout(height=max(320, rows * 260), margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor=colors["background"], plot_bgcolor=colors["background"], font=dict(color=colors["text"]), showlegend=False)
  return fig


def CHART_PROMPT_IMPACT(data: pd.DataFrame):
  if data.empty:
    return _empty_chart("No data for the selected ranking")
  df = data.copy()
  prompt_order = _prompt_order(df["prompt_label"])
  palette = ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FDB462"]
  box = alt.Chart(df).mark_boxplot(extent="min-max").encode(
    x=alt.X("prompt_label:N", sort=prompt_order, title="Prompting Strategy", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("smape:Q", title="SMAPE (%)"),
    color=alt.Color("prompt_label:N", sort=prompt_order, scale=alt.Scale(domain=prompt_order, range=palette[:len(prompt_order)]), legend=None),
  )
  mean = alt.Chart(df).transform_aggregate(mean_smape="mean(smape)", groupby=["prompt_label"]).mark_point(filled=True, color="white", stroke="black", size=70).encode(
    x=alt.X("prompt_label:N", sort=prompt_order),
    y=alt.Y("mean_smape:Q"),
  )
  return _chart_base(alt.layer(box, mean).properties(height=330), height=330)


def CHART_MEAN_SMAPE_MODEL_DATASET(data: pd.DataFrame):
  if data.empty:
    return _empty_chart("No data for the selected ranking")
  df = data.groupby(["dataset", "model"], as_index=False)["smape"].mean()
  dataset_order = _ordered_unique(data["dataset"])
  model_order = _ordered_unique(data["model"])
  chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("dataset:N", sort=dataset_order, title="Dataset"),
    xOffset=alt.XOffset("model:N", sort=model_order),
    y=alt.Y("smape:Q", title="Mean SMAPE (%)"),
    color=alt.Color("model:N", title="Model", scale=alt.Scale(scheme="viridis")),
    tooltip=[alt.Tooltip("dataset:N", title="Dataset"), alt.Tooltip("model:N", title="Model"), alt.Tooltip("smape:Q", title="Mean SMAPE", format=".2f")],
  ).properties(height=360)
  return _chart_base(chart, height=360)


def CHART_SMAPE_HEATMAP(data: pd.DataFrame):
  if data.empty:
    return _empty_chart("No data for the selected ranking")
  df = data.groupby(["dataset", "model"], as_index=False)["smape"].mean()
  dataset_order = sorted(df["dataset"].dropna().unique().tolist())
  model_order = sorted(df["model"].dropna().unique().tolist())
  rect = alt.Chart(df).mark_rect().encode(
    x=alt.X("model:N", sort=model_order, title="Model"),
    y=alt.Y("dataset:N", sort=dataset_order, title="Dataset"),
    color=alt.Color("smape:Q", title="SMAPE", scale=alt.Scale(scheme="yellowgreenblue")),
    tooltip=[alt.Tooltip("dataset:N", title="Dataset"), alt.Tooltip("model:N", title="Model"), alt.Tooltip("smape:Q", title="SMAPE", format=".2f")],
  )
  text = alt.Chart(df).mark_text(color="black", fontSize=10).encode(
    x=alt.X("model:N", sort=model_order),
    y=alt.Y("dataset:N", sort=dataset_order),
    text=alt.Text("smape:Q", format=".2f"),
  )
  return _chart_base((rect + text).properties(height=360), height=360)


CHART_MAE_HEATMAP = CHART_SMAPE_HEATMAP


def CHART_FORMAT_SERIES(data: pd.DataFrame):
  if data.empty:
    return _empty_chart("No data for the selected ranking")
  df = data.groupby(["format_label", "series_label"], as_index=False)["smape"].mean()
  format_order = [v for v in ["CSV", "Plain"] if v in df["format_label"].unique()]
  series_order = [v for v in ["Numeric", "Textual"] if v in df["series_label"].unique()]
  chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("format_label:N", sort=format_order, title="Format"),
    xOffset=alt.XOffset("series_label:N", sort=series_order),
    y=alt.Y("smape:Q", title="SMAPE (%)"),
    color=alt.Color("series_label:N", title="Series Type", sort=series_order, scale=alt.Scale(domain=series_order, range=["#457B9D", "#E76F51"])),
    tooltip=[alt.Tooltip("format_label:N", title="Format"), alt.Tooltip("series_label:N", title="Series Type"), alt.Tooltip("smape:Q", title="SMAPE", format=".2f")],
  ).properties(height=330)
  return _chart_base(chart, height=330)


def CHART_EFFICIENCY_TRADEOFF(data: pd.DataFrame):
  df = data.dropna(subset=["smape", "inference_time_s"])
  if df.empty:
    return _empty_chart("No data for the selected ranking")
  prompt_order = _prompt_order(df["prompt_label"])
  model_domain = _ordered_unique(df["model"])
  chart = alt.Chart(df).mark_circle(opacity=0.8, stroke="black", strokeWidth=0.4, size=70).encode(
    x=alt.X("inference_time_s:Q", scale=alt.Scale(type="log"), title="Inference Time (seconds) [log]"),
    y=alt.Y("smape:Q", scale=alt.Scale(type="log"), title="SMAPE (%) [log]"),
    color=alt.Color("model:N", title="Model", scale=alt.Scale(domain=model_domain, scheme="tableau20")),
    shape=alt.Shape("prompt_label:N", title="Prompting Strategy", sort=prompt_order),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("prompt_label:N", title="Prompt"), alt.Tooltip("inference_time_s:Q", title="Inference Time (s)", format=".2f"), alt.Tooltip("smape:Q", title="SMAPE", format=".2f")],
  ).properties(height=380)
  return _chart_base(chart, height=380)


def CHART_AVG_INFERENCE_TIME(data: pd.DataFrame):
  df = data.groupby("model", as_index=False)["inference_time_s"].mean().sort_values("inference_time_s")
  if df.empty:
    return _empty_chart("No data for the selected ranking")
  model_order = df["model"].tolist()
  chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="steelblue").encode(
    x=alt.X("model:N", sort=model_order, title="Model", axis=alt.Axis(labelAngle=-45, labelLimit=140)),
    y=alt.Y("inference_time_s:Q", title="Inference Time (seconds)"),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("inference_time_s:Q", title="Inference Time (s)", format=".2f")],
  ).properties(height=330)
  return _chart_base(chart, height=330)


def CHART_AVG_ENERGY_CONSUMPTION(data: pd.DataFrame):
  df = data.loc[~data["model"].isin(ENERGY_MODEL_EXCLUSIONS)].copy()
  df = df.groupby("model", as_index=False)["eco2ai_energy_wh"].mean().sort_values("eco2ai_energy_wh")
  if df.empty:
    return _empty_chart("No data for the selected ranking")
  model_order = df["model"].tolist()
  chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="steelblue").encode(
    x=alt.X("model:N", sort=model_order, title="Model", axis=alt.Axis(labelAngle=-45, labelLimit=140)),
    y=alt.Y("eco2ai_energy_wh:Q", title="Energy Consumption (Wh)"),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("eco2ai_energy_wh:Q", title="Energy Consumption (Wh)", format=".2f")],
  ).properties(height=330)
  return _chart_base(chart, height=330)


def CHART_FTE_SCORE(data: pd.DataFrame):
  if data.empty:
    return _empty_chart("No data for the selected ranking")
  df = data.groupby(["model", "prompt_label"], as_index=False)["fte_score"].mean()
  model_order = _ordered_unique(data["model"])
  prompt_order = _prompt_order(df["prompt_label"])
  chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("model:N", sort=model_order, title="Model", axis=alt.Axis(labelAngle=-45, labelLimit=140)),
    xOffset=alt.XOffset("prompt_label:N", sort=prompt_order),
    y=alt.Y("fte_score:Q", title="Score (Accuracy / Log Tokens)"),
    color=alt.Color("prompt_label:N", title="Prompting Strategy", sort=prompt_order, scale=alt.Scale(scheme="magma")),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("prompt_label:N", title="Prompt"), alt.Tooltip("fte_score:Q", title="FTE Score", format=".2f")],
  ).properties(height=330)
  return _chart_base(chart, height=330)


def CHART_BUBBLE_TRADEOFF(data: pd.DataFrame):
  df = data.loc[~data["model"].str.contains("gpt-5-nano", case=False, na=False)].copy()
  if df.empty:
    return _empty_chart("No data for the selected ranking")
  df = df.groupby("model", as_index=False).agg(smape=("smape", "mean"), inference_time_s=("inference_time_s", "mean"), family=("family", "first"))
  df["model_short"] = df["model"].apply(_normalize_model_key)
  df["size_label"] = df["model"].apply(_extract_size_label)
  df["params_b"] = df["model"].apply(_lookup_params_b)
  family_domain = [f for f in FAMILY_STYLE if f in df["family"].dropna().unique()]
  family_range = [FAMILY_STYLE[f]["color"] for f in family_domain]
  chart = alt.Chart(df).mark_circle(opacity=0.88).encode(
    x=alt.X("inference_time_s:Q", title="Inference Time (s)", scale=alt.Scale(type="log")),
    y=alt.Y("smape:Q", title="SMAPE (%)"),
    size=alt.Size("params_b:Q", title="Model Parameter", scale=alt.Scale(type="log", range=[80, 900]), legend=alt.Legend(values=[1, 10, 100, 1000], labelExpr="datum.value === 1000 ? '1T' : datum.value === 100 ? '100B' : datum.value === 10 ? '10B' : '1B'")),
    color=alt.Color("family:N", title="Model", scale=alt.Scale(domain=family_domain, range=family_range)),
    tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("family:N", title="Family"), alt.Tooltip("smape:Q", title="SMAPE", format=".2f"), alt.Tooltip("inference_time_s:Q", title="Inference Time (s)", format=".2f"), alt.Tooltip("params_b:Q", title="Parameters (B)", format=".0f")],
  )
  labels = alt.Chart(df).mark_text(fontSize=8, fontWeight="bold", color="white").encode(
    x=alt.X("inference_time_s:Q", scale=alt.Scale(type="log")),
    y=alt.Y("smape:Q"),
    text="size_label:N",
  )
  return _chart_base((chart + labels).properties(height=380), height=380)


def CHART_RADAR_DATASETS(datasets: list[str] | None = None) -> go.Figure:
  target = datasets or list(CHARACTERISTICS.keys())
  panels = []
  for ds in target:
    if ds not in CHARACTERISTICS:
      continue
    color = "#457B9D"
    panels.append({"title": ds, "values": CHARACTERISTICS[ds], "color": color})
  if not panels:
    return _empty_plotly_chart("No dataset characteristics available")
  return _radar_figure(panels, columns=min(4, len(panels)))


def CHART_COST_SCORE(board: pd.DataFrame):
  if board.empty:
    return _empty_chart("No data")
  df = board.copy()
  df["params_b"] = df["model"].apply(_lookup_params_b)
  chart = alt.Chart(df).mark_circle(opacity=0.78).encode(
    x=alt.X("financial_cost_usd:Q", title="Financial Cost (US$) ↓"),
    y=alt.Y("overall_score:Q", title="Overall Score ↑"),
    size=alt.Size("params_b:Q", title="Parameters (B)", scale=alt.Scale(range=[70, 900])),
    color=alt.Color("family:N", title="Family"),
    tooltip=["rank", "model", alt.Tooltip("overall_score:Q", format=".2f"), alt.Tooltip("financial_cost_usd:Q", format=".4f")],
  ).interactive().properties(height=360)
  return _chart_base(chart, height=360)


def CHART_PARETO(board: pd.DataFrame):
  if board.empty:
    return _empty_chart("No data")
  chart = alt.Chart(board).mark_circle(opacity=0.78).encode(
    x=alt.X("inference_time_s:Q", title="Inference Time (s) ↓"),
    y=alt.Y("smape:Q", title="SMAPE ↓"),
    size=alt.Size("overall_score:Q", title="Score ↑", scale=alt.Scale(range=[80, 900])),
    color=alt.Color("deployment:N", title="Deployment"),
    tooltip=["rank", "model", alt.Tooltip("overall_score:Q", format=".2f"), alt.Tooltip("smape:Q", format=".2f"), alt.Tooltip("inference_time_s:Q", format=".2f")],
  ).interactive().properties(height=380)
  return _chart_base(chart, height=380)


def CHART_TOP_MODELS(board: pd.DataFrame):
  if board.empty:
    return _empty_chart("No data")
  df = board.head(12).copy()
  chart = alt.Chart(df).mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7).encode(
    y=alt.Y("model:N", sort="-x", title="Model"),
    x=alt.X("overall_score:Q", title="Overall Score ↑"),
    color=alt.Color("deployment:N", title="Deployment"),
    tooltip=["rank", "model", alt.Tooltip("overall_score:Q", format=".2f"), alt.Tooltip("smape:Q", format=".2f")],
  ).properties(height=360)
  return _chart_base(chart, height=360)

"""FinAgent-Eval Streamlit dashboard."""
import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
ENRICHED_DIR = PROJECT_ROOT / "results/track_a/enriched"
TASKS_PATH = PROJECT_ROOT / "data/track_a/tasks.json"
RELIABILITY_PATH = PROJECT_ROOT / "results/reliability/reliability_probe_results.json"
DOMAIN_ADAPTATION_PATH = PROJECT_ROOT / "reports/domain_adaptation_summary.json"

st.set_page_config(
    page_title="FinAgent-Eval",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

IS_DARK = st.session_state.theme_mode == "Dark"

MODEL_COLORS = {
    "Qwen2.5-VL-7B-Instruct": "#16a34a",
    "LayoutLMv3": "#2563eb",
    "Pix2Struct": "#7c3aed",
    "OCR+RoBERTa": "#0f766e",
    "Donut": "#b45309",
}

CATEGORY_ORDER = [
    "extractive",
    "layout_understanding",
    "numerical_reasoning",
    "chart_interpretation",
]

FAILURE_COLORS = {
    "numeric_mismatch": "#dc2626",
    "manual_diagnosis_required": "#64748b",
    "empty_output": "#d97706",
    "wrong_year": "#7c3aed",
    "abstention": "#0891b2",
    "hallucinated_number": "#be123c",
    "format_mismatch": "#059669",
}

THEME = {
    "bg": "#0b1120" if IS_DARK else "#f6f8fb",
    "surface": "#111827" if IS_DARK else "#ffffff",
    "surface_2": "#0f172a" if IS_DARK else "#f8fafc",
    "border": "#263244" if IS_DARK else "#dbe3ef",
    "text": "#e5e7eb" if IS_DARK else "#0f172a",
    "muted": "#94a3b8" if IS_DARK else "#475569",
    "faint": "#cbd5e1" if IS_DARK else "#64748b",
    "grid": "#253044" if IS_DARK else "#e2e8f0",
    "insight_bg": "#052e2b" if IS_DARK else "#ecfdf5",
    "insight_border": "#047857" if IS_DARK else "#bbf7d0",
    "insight_text": "#d1fae5" if IS_DARK else "#064e3b",
    "note_bg": "#3b2508" if IS_DARK else "#fff7ed",
    "note_border": "#92400e" if IS_DARK else "#fed7aa",
    "note_text": "#fed7aa" if IS_DARK else "#7c2d12",
    "code_bg": "#020617" if IS_DARK else "#111827",
}

HEATMAP_SCALE = (
    [[0.0, "#3f1d2e"], [0.5, "#4a3d1b"], [1.0, "#14532d"]]
    if IS_DARK
    else [[0.0, "#fee2e2"], [0.5, "#fef3c7"], [1.0, "#bbf7d0"]]
)

st.markdown(
    f"""
<style>
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    display: none !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
    height: 0rem;
}}

[data-testid="stAppViewContainer"] {{
    background: {THEME["bg"]};
}}

.block-container {{
    max-width: 1280px;
    padding-top: 1.35rem;
    padding-bottom: 2.5rem;
}}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
label {{
    color: {THEME["text"]} !important;
}}

[data-testid="stMarkdownContainer"] p {{
    line-height: 1.55;
}}

.fe-header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.7rem;
}}

.fe-eyebrow {{
    color: #2563eb;
    font-size: 0.78rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 0.35rem;
}}

.fe-title {{
    color: {THEME["text"]};
    font-size: 2.55rem;
    line-height: 1.02;
    font-weight: 880;
    margin: 0;
}}

.fe-subtitle {{
    color: {THEME["muted"]};
    font-size: 1.05rem;
    margin-top: 0.55rem;
    margin-bottom: 1rem;
    max-width: 900px;
}}

.fe-theme-label {{
    color: {THEME["faint"]};
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}

.fe-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.85rem 0 1.25rem 0;
}}

.fe-chip {{
    background: {THEME["surface"]};
    border: 1px solid {THEME["border"]};
    border-radius: 999px;
    padding: 0.36rem 0.78rem;
    color: {THEME["text"]};
    font-size: 0.86rem;
    font-weight: 750;
}}

.fe-insight {{
    background: {THEME["insight_bg"]};
    border: 1px solid {THEME["insight_border"]};
    border-left: 5px solid #059669;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    color: {THEME["insight_text"]};
    margin: 0.85rem 0 1rem 0;
    font-size: 0.98rem;
}}

.fe-note {{
    background: {THEME["note_bg"]};
    border: 1px solid {THEME["note_border"]};
    border-left: 5px solid #d97706;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    color: {THEME["note_text"]};
    margin: 0.7rem 0 1rem 0;
}}

.kpi-card {{
    background: {THEME["surface"]};
    border: 1px solid {THEME["border"]};
    border-radius: 8px;
    padding: 1rem 1.05rem;
    min-height: 112px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
}}

.kpi-label {{
    color: {THEME["faint"]};
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.045em;
    margin-bottom: 0.45rem;
}}

.kpi-value {{
    color: {THEME["text"]};
    font-size: 1.7rem;
    font-weight: 880;
    line-height: 1.12;
    overflow-wrap: anywhere;
}}

.kpi-caption {{
    color: {THEME["muted"]};
    font-size: 0.86rem;
    margin-top: 0.45rem;
}}

.info-panel {{
    background: {THEME["surface"]};
    border: 1px solid {THEME["border"]};
    border-radius: 8px;
    padding: 1rem 1.05rem;
    color: {THEME["text"]};
}}

.info-panel table {{
    width: 100%;
    border-collapse: collapse;
}}

.info-panel td {{
    color: {THEME["text"]};
    padding: 0.55rem 0;
    border-bottom: 1px solid {THEME["border"]};
    vertical-align: top;
}}

.info-panel td:first-child {{
    color: {THEME["faint"]};
    font-weight: 800;
    width: 32%;
}}

.info-panel ul {{
    margin-bottom: 0;
    color: {THEME["text"]};
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 1.45rem;
    border-bottom: 1px solid {THEME["border"]};
}}

.stTabs [data-baseweb="tab"] {{
    color: {THEME["muted"]};
    font-weight: 760;
    padding: 0.9rem 0.1rem;
}}

.stTabs [aria-selected="true"] {{
    color: {THEME["text"]} !important;
}}

div[data-baseweb="select"] > div,
div[role="radiogroup"] label {{
    background: {THEME["surface"]} !important;
    border-color: {THEME["border"]} !important;
    color: {THEME["text"]} !important;
}}

div[data-baseweb="select"] span {{
    color: {THEME["text"]} !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {THEME["border"]};
    border-radius: 8px;
    overflow: hidden;
}}

pre, code {{
    background: {THEME["code_bg"]} !important;
}}

.fe-footer {{
    text-align: center;
    color: {THEME["faint"]};
    font-size: 0.86rem;
    margin-top: 1.5rem;
}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_scorecards() -> dict:
    scorecards = {}
    for path in sorted(REPORTS_DIR.glob("scorecard_*.json")):
        with path.open(encoding="utf-8") as f:
            scorecard = json.load(f)
        scorecards[scorecard["model"]] = scorecard
    return scorecards


@st.cache_data
def load_comparison() -> dict:
    with (REPORTS_DIR / "comparison.json").open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_enriched() -> pd.DataFrame:
    rows = []
    for path in sorted(ENRICHED_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.load(f))
    return pd.DataFrame(rows)


@st.cache_data
def load_tasks() -> pd.DataFrame:
    with TASKS_PATH.open(encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def load_reliability() -> list[dict]:
    if not RELIABILITY_PATH.exists():
        return []
    with RELIABILITY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_domain_adaptation() -> dict:
    if not DOMAIN_ADAPTATION_PATH.exists():
        return {}
    with DOMAIN_ADAPTATION_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def category_label(category: str) -> str:
    return category.replace("_", " ").title()


def failure_label(label: str) -> str:
    return label.replace("_", " ").title()


def truncate(value: object, limit: int = 90) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "..."


def kpi(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{esc(label)}</div>
  <div class="kpi-value">{esc(value)}</div>
  <div class="kpi-caption">{esc(caption)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def plot_layout(height: int = 360) -> dict:
    return {
        "template": "plotly_dark" if IS_DARK else "plotly_white",
        "height": height,
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            "color": THEME["text"],
        },
        "margin": {"l": 25, "r": 25, "t": 35, "b": 45},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": THEME["surface"],
        "legend": {
            "font": {"color": THEME["text"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "xaxis": {
            "tickfont": {"color": THEME["muted"]},
            "title": {"font": {"color": THEME["muted"]}},
            "gridcolor": THEME["grid"],
            "zerolinecolor": THEME["border"],
        },
        "yaxis": {
            "tickfont": {"color": THEME["muted"]},
            "title": {"font": {"color": THEME["muted"]}},
            "gridcolor": THEME["grid"],
            "zerolinecolor": THEME["border"],
        },
    }


scorecards = load_scorecards()
comparison = load_comparison()
df = load_enriched()
tasks_df = load_tasks()
reliability_rows = load_reliability()
domain_adaptation = load_domain_adaptation()

comparison_rows = sorted(
    comparison["models"],
    key=lambda row: row["overall_anls"],
    reverse=True,
)
model_order = [row["model"] for row in comparison_rows]
best = comparison_rows[0]

all_failures = df[df["failure_label"].notna()].copy()
category_means = {
    category: sum(row.get(f"{category}_anls", 0) for row in comparison_rows) / len(comparison_rows)
    for category in CATEGORY_ORDER
}
hardest_category = min(category_means, key=category_means.get)

header_left, header_right = st.columns([0.82, 0.18], vertical_alignment="top")
with header_left:
    st.markdown(
        """
<div class="fe-eyebrow">Financial Document AI Evaluation</div>
<div class="fe-title">FinAgent-Eval</div>
<div class="fe-subtitle">
  A reproducible benchmark console for visual question answering over SEC filing screenshots.
</div>
<div class="fe-chip-row">
  <span class="fe-chip">397 questions</span>
  <span class="fe-chip">79 document images</span>
  <span class="fe-chip">10 companies</span>
  <span class="fe-chip">4 task categories</span>
  <span class="fe-chip">5 evaluated models</span>
</div>
""",
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("<div class='fe-theme-label'>Theme</div>", unsafe_allow_html=True)
    st.radio(
        "Theme",
        ["Light", "Dark"],
        horizontal=True,
        key="theme_mode",
        label_visibility="collapsed",
    )

tab_overview, tab_categories, tab_failures, tab_reliability, tab_domain, tab_dataset, tab_adapter = st.tabs(
    [
        "Overview",
        "Category Breakdown",
        "Failure Explorer",
        "Reliability",
        "Domain Adaptation",
        "Dataset Card",
        "Run Your Own Model",
    ]
)

with tab_overview:
    st.markdown(
        f"""
<div class="fe-insight">
  <strong>{esc(best["model"])}</strong> leads the full Track A evaluation with
  ANLS <strong>{best["overall_anls"]:.3f}</strong> and exact match
  <strong>{best["overall_exact_match"]:.3f}</strong>. The modern VLM is a clear step-change over
  the legacy document baselines, while numerical reasoning remains the hardest task type.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Best Model", best["model"], "Highest overall ANLS")
    with c2:
        kpi("Best ANLS", f"{best['overall_anls']:.3f}", "Primary VQA metric")
    with c3:
        kpi("Best Exact Match", f"{best['overall_exact_match']:.3f}", "Normalized exact match")
    with c4:
        kpi("Hardest Category", category_label(hardest_category), "Lowest average ANLS")

    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Leaderboard")
        leaderboard = pd.DataFrame(comparison_rows)
        colors = [MODEL_COLORS.get(m, "#64748b") for m in leaderboard["model"]]

        fig = go.Figure(
            go.Bar(
                x=leaderboard["overall_anls"],
                y=leaderboard["model"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in leaderboard["overall_anls"]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>ANLS: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(**plot_layout(390))
        fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(model_order)))
        fig.update_xaxes(title="ANLS", range=[0, max(leaderboard["overall_anls"]) * 1.28])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.subheader("ANLS vs Exact Match")
        fig = go.Figure()
        label_offsets = {
            "Qwen2.5-VL-7B-Instruct": {"ax": -105, "ay": -22},
            "LayoutLMv3": {"ax": 88, "ay": -52},
            "Pix2Struct": {"ax": 92, "ay": 22},
            "OCR+RoBERTa": {"ax": -86, "ay": 36},
            "Donut": {"ax": -72, "ay": 74},
        }
        for row in comparison_rows:
            model = row["model"]
            color = MODEL_COLORS.get(model, "#64748b")
            fig.add_trace(
                go.Scatter(
                    x=[row["overall_anls"]],
                    y=[row["overall_exact_match"]],
                    mode="markers",
                    marker={
                        "size": 20 if model == best["model"] else 15,
                        "color": color,
                        "line": {"width": 2, "color": THEME["surface"]},
                    },
                    name=model,
                    hovertemplate=(
                        f"<b>{model}</b><br>"
                        "ANLS: %{x:.4f}<br>"
                        "Exact Match: %{y:.4f}<extra></extra>"
                    ),
                )
            )
            offset = label_offsets.get(model, {"ax": 55, "ay": -35})
            fig.add_annotation(
                x=row["overall_anls"],
                y=row["overall_exact_match"],
                text=model,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor=color,
                ax=offset["ax"],
                ay=offset["ay"],
                bgcolor=THEME["surface"],
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                font={"size": 11, "color": THEME["text"]},
            )
        fig.update_layout(**plot_layout(390), showlegend=False)
        fig.update_xaxes(title="ANLS", range=[0, max(row["overall_anls"] for row in comparison_rows) * 1.12])
        fig.update_yaxes(title="Exact Match", range=[0, max(row["overall_exact_match"] for row in comparison_rows) * 1.16])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Full Results")
    summary_rows = []
    for row in comparison_rows:
        sc = scorecards[row["model"]]
        summary_rows.append(
            {
                "Model": sc["model"],
                "ANLS": sc["overall_anls"],
                "ANLS 95% CI": f"[{sc['overall_anls_ci'][0]:.3f}, {sc['overall_anls_ci'][1]:.3f}]",
                "Exact Match": sc["overall_exact_match"],
                "EM 95% CI": f"[{sc['overall_exact_match_ci'][0]:.3f}, {sc['overall_exact_match_ci'][1]:.3f}]",
                "Weakest Category": category_label(sc["weakest_category"]),
                "Failures": f"{sc['total_failures']}/397",
            }
        )

    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "ANLS": st.column_config.NumberColumn(format="%.4f"),
            "Exact Match": st.column_config.NumberColumn(format="%.4f"),
        },
    )

with tab_categories:
    st.subheader("Category Heatmap")

    heatmap = [
        [row.get(f"{category}_anls", 0) for category in CATEGORY_ORDER]
        for row in comparison_rows
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap,
            zmin=0,
            zmax=max(0.7, max(max(row) for row in heatmap)),
            x=[category_label(category) for category in CATEGORY_ORDER],
            y=model_order,
            colorscale=HEATMAP_SCALE,
            text=[[f"{value:.3f}" for value in row] for row in heatmap],
            texttemplate="%{text}",
            textfont={"size": 14, "color": THEME["text"]},
            hovertemplate="Model: %{y}<br>Category: %{x}<br>ANLS: %{z:.4f}<extra></extra>",
            colorbar={"title": "ANLS"},
        )
    )
    fig.update_layout(**plot_layout(370))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
<div class="fe-note">
  Chart interpretation has only 17 questions, so confidence intervals are wide.
  Treat chart-category comparisons as directional rather than definitive.
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Model Detail")
    selected_model = st.selectbox("Model", model_order, key="category_model")
    selected_scorecard = scorecards[selected_model]

    category_rows = []
    for category in CATEGORY_ORDER:
        vals = selected_scorecard["category_breakdown"].get(category)
        if vals:
            category_rows.append(
                {
                    "Category": category_label(category),
                    "N": vals["n"],
                    "ANLS": vals["anls"],
                    "ANLS 95% CI": f"[{vals['anls_ci'][0]:.3f}, {vals['anls_ci'][1]:.3f}]",
                    "Exact Match": vals["exact_match"],
                    "EM 95% CI": f"[{vals['exact_match_ci'][0]:.3f}, {vals['exact_match_ci'][1]:.3f}]",
                }
            )

    st.dataframe(
        pd.DataFrame(category_rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "ANLS": st.column_config.NumberColumn(format="%.4f"),
            "Exact Match": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.subheader("Task-Type Profile")
    fig = go.Figure()
    for category in CATEGORY_ORDER:
        fig.add_trace(
            go.Bar(
                name=category_label(category),
                x=model_order,
                y=[
                    scorecards[model]["category_breakdown"].get(category, {}).get("anls", 0)
                    for model in model_order
                ],
                hovertemplate="%{x}<br>ANLS: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(**plot_layout(400), barmode="group")
    fig.update_yaxes(title="ANLS")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with tab_failures:
    st.subheader("Failure Explorer")

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_failure_model = st.selectbox("Model", ["All"] + model_order, key="failure_model")
    with f2:
        selected_category_label = st.selectbox(
            "Category",
            ["All"] + [category_label(category) for category in CATEGORY_ORDER],
            key="failure_category",
        )
    with f3:
        failure_options = sorted(all_failures["failure_label"].dropna().unique().tolist())
        selected_failure_label = st.selectbox(
            "Failure Label",
            ["All"] + failure_options,
            key="failure_label",
        )

    filtered = all_failures.copy()
    if selected_failure_model != "All":
        filtered = filtered[filtered["model"] == selected_failure_model]
    if selected_category_label != "All":
        selected_raw_category = selected_category_label.lower().replace(" ", "_")
        filtered = filtered[filtered["category"] == selected_raw_category]
    if selected_failure_label != "All":
        filtered = filtered[filtered["failure_label"] == selected_failure_label]

    current_top_failure = (
        failure_label(filtered["failure_label"].value_counts().index[0])
        if not filtered.empty
        else "None"
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        kpi("Visible Failures", f"{len(filtered):,}", "After active filters")
    with m2:
        kpi("Dominant Failure", current_top_failure, "Most common visible label")
    with m3:
        kpi(
            "Average ANLS",
            "n/a" if filtered.empty else f"{filtered['recomputed_anls'].mean():.3f}",
            "Among visible failures",
        )

    if filtered.empty:
        st.info("No failures match the selected filters.")
    else:
        counts = filtered["failure_label"].value_counts()
        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=[failure_label(label) for label in counts.index],
                orientation="h",
                marker_color=[FAILURE_COLORS.get(label, "#64748b") for label in counts.index],
                text=counts.values,
                textposition="outside",
                hovertemplate="%{y}<br>Count: %{x}<extra></extra>",
            )
        )
        fig.update_layout(**plot_layout(300))
        fig.update_xaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        case_df = (
            filtered.sort_values(["recomputed_anls", "model", "task_id"], ascending=[True, True, True])
            .reset_index(drop=True)
        )
        options = list(range(min(len(case_df), 250)))

        def option_label(index: int) -> str:
            row = case_df.iloc[index]
            return (
                f"{index:03d} | {row['model']} | {row['task_id']} | "
                f"{failure_label(row['failure_label'])} | {truncate(row.get('question'), 72)}"
            )

        selected_index = st.selectbox(
            "Inspect failure",
            options,
            format_func=option_label,
            key="failure_case",
        )
        row = case_df.iloc[selected_index]

        left, right = st.columns([0.95, 1.05])
        with left:
            st.markdown(f"**Task:** `{row['task_id']}`")
            st.markdown(f"**Model:** {row['model']}")
            st.markdown(f"**Category:** {category_label(row['category'])}")
            st.markdown(f"**Failure Label:** `{failure_label(row['failure_label'])}`")
            st.markdown(f"**ANLS:** `{row['recomputed_anls']:.4f}`")
            st.markdown("**Question**")
            st.write(str(row.get("question", "")))
            st.markdown("**Gold Answer**")
            st.code(str(row.get("gold_answer", "")), language="text")
            st.markdown("**Prediction**")
            st.code(str(row.get("prediction", ""))[:800], language="text")

        with right:
            image_path = row.get("image_path")
            full_path = PROJECT_ROOT / str(image_path) if pd.notna(image_path) else None
            if full_path and full_path.exists():
                st.image(str(full_path), caption=str(image_path), use_container_width=True)
            else:
                st.info("Source image is unavailable for this row.")

        st.subheader("Filtered Rows")
        visible_columns = [
            "task_id",
            "model",
            "category",
            "question",
            "gold_answer",
            "prediction",
            "failure_label",
            "recomputed_anls",
        ]
        st.dataframe(
            case_df[visible_columns].head(100),
            hide_index=True,
            use_container_width=True,
            column_config={
                "question": st.column_config.TextColumn(width="large"),
                "prediction": st.column_config.TextColumn(width="medium"),
                "recomputed_anls": st.column_config.NumberColumn("ANLS", format="%.4f"),
            },
        )

with tab_reliability:
    st.subheader("Reliability Mini-Probe")

    if not reliability_rows:
        st.info("Reliability probe results are not available yet. Expected file: results/reliability/reliability_probe_results.json")
    else:
        total_probe = len(reliability_rows)
        variants_per_task = len(reliability_rows[0].get("variants", [])) if reliability_rows else 0
        consistent_count = sum(bool(row.get("consistent_answer")) for row in reliability_rows)
        stable_count = sum(bool(row.get("correctness_stable")) for row in reliability_rows)
        original_correct = sum(bool(row.get("original_correct")) for row in reliability_rows)
        all_correct = sum(bool(row.get("all_correct")) for row in reliability_rows)
        any_correct = sum(bool(row.get("any_correct")) for row in reliability_rows)

        st.markdown(
            f"""
<div class="fe-insight">
  Qwen2.5-VL was tested on <strong>{total_probe}</strong> stratified questions with
  <strong>{variants_per_task}</strong> phrasings each. It reached
  <strong>{consistent_count / total_probe:.1%}</strong> exact answer consistency and
  <strong>{stable_count / total_probe:.1%}</strong> correctness stability, showing that
  paraphrase sensitivity remains a real reliability issue.
</div>
""",
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            kpi("Answer Consistency", f"{consistent_count}/{total_probe}", f"{consistent_count / total_probe:.1%} same answer")
        with r2:
            kpi("Correctness Stable", f"{stable_count}/{total_probe}", f"{stable_count / total_probe:.1%} stable right/wrong")
        with r3:
            kpi("Original Accuracy", f"{original_correct}/{total_probe}", f"{original_correct / total_probe:.1%}")
        with r4:
            kpi("All Variants Correct", f"{all_correct}/{total_probe}", f"{all_correct / total_probe:.1%}")
        with r5:
            kpi("Any Variant Correct", f"{any_correct}/{total_probe}", f"{any_correct / total_probe:.1%}")

        st.markdown(
            """
<div class="fe-note">
  This is a diagnostic reliability probe, not a second leaderboard. Paraphrases were model-generated,
  and a few rewrites may slightly change the original meaning. Treat the results as robustness evidence,
  not as a definitive benchmark.
</div>
""",
            unsafe_allow_html=True,
        )

        reliability_df = pd.DataFrame(
            [
                {
                    "Task ID": row.get("task_id"),
                    "Category": category_label(row.get("category", "")),
                    "Gold Answer": row.get("gold_answer"),
                    "Answer Consistent": bool(row.get("consistent_answer")),
                    "Correctness Stable": bool(row.get("correctness_stable")),
                    "Original Correct": bool(row.get("original_correct")),
                    "All Correct": bool(row.get("all_correct")),
                    "Any Correct": bool(row.get("any_correct")),
                    "Original Question": row.get("original_question"),
                }
                for row in reliability_rows
            ]
        )

        category_summary_rows = []
        for category in CATEGORY_ORDER:
            cat_rows = [row for row in reliability_rows if row.get("category") == category]
            if not cat_rows:
                continue
            n = len(cat_rows)
            answer_consistency = sum(bool(row.get("consistent_answer")) for row in cat_rows) / n
            all_correct_rate = sum(bool(row.get("all_correct")) for row in cat_rows) / n
            original_accuracy = sum(bool(row.get("original_correct")) for row in cat_rows) / n
            category_summary_rows.append(
                {
                    "Category": category_label(category),
                    "N": n,
                    "Answer Consistency": answer_consistency,
                    "All Variants Correct": all_correct_rate,
                    "Original Accuracy": original_accuracy,
                    "Answer Consistency %": f"{answer_consistency:.1%}",
                    "All Variants Correct %": f"{all_correct_rate:.1%}",
                    "Original Accuracy %": f"{original_accuracy:.1%}",
                }
            )

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Robustness by Category")
            category_summary = pd.DataFrame(category_summary_rows)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Answer Consistency",
                    y=category_summary["Category"],
                    x=category_summary["Answer Consistency"],
                    orientation="h",
                    marker_color="#2563eb",
                    text=[f"{value:.0%}" for value in category_summary["Answer Consistency"]],
                    textposition="outside",
                    hovertemplate="%{y}<br>Consistency: %{x:.1%}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="All Variants Correct",
                    y=category_summary["Category"],
                    x=category_summary["All Variants Correct"],
                    orientation="h",
                    marker_color="#16a34a",
                    text=[f"{value:.0%}" for value in category_summary["All Variants Correct"]],
                    textposition="outside",
                    hovertemplate="%{y}<br>All-correct: %{x:.1%}<extra></extra>",
                )
            )
            reliability_chart_layout = plot_layout(360)
            reliability_chart_layout["legend"].update(
                {
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.03,
                    "xanchor": "left",
                    "x": 0,
                }
            )
            fig.update_layout(**reliability_chart_layout, barmode="group")
            fig.update_xaxes(title="Rate", range=[0, 1.08], tickformat=".0%")
            fig.update_yaxes(title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with right:
            st.subheader("Probe Summary")
            category_display = category_summary[
                [
                    "Category",
                    "N",
                    "Answer Consistency %",
                    "All Variants Correct %",
                    "Original Accuracy %",
                ]
            ].rename(
                columns={
                    "Answer Consistency %": "Answer Consistency",
                    "All Variants Correct %": "All Variants Correct",
                    "Original Accuracy %": "Original Accuracy",
                }
            )
            st.dataframe(
                category_display,
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Inspect Paraphrase Sensitivity")
        inspect_mode = st.radio(
            "Cases",
            ["Inconsistent only", "All probe tasks"],
            horizontal=True,
            key="reliability_case_mode",
        )
        case_pool = [row for row in reliability_rows if not row.get("consistent_answer")] if inspect_mode == "Inconsistent only" else reliability_rows

        if not case_pool:
            st.success("No cases match this filter.")
        else:
            def reliability_option(index: int) -> str:
                row = case_pool[index]
                return f"{index:02d} | {row.get('task_id')} | {category_label(row.get('category', ''))} | {truncate(row.get('original_question'), 82)}"

            selected_case = st.selectbox(
                "Probe case",
                list(range(len(case_pool))),
                format_func=reliability_option,
                key="reliability_case",
            )
            selected = case_pool[selected_case]

            detail_left, detail_right = st.columns([0.75, 1.25])
            with detail_left:
                st.markdown(f"**Task:** `{selected.get('task_id')}`")
                st.markdown(f"**Category:** {category_label(selected.get('category', ''))}")
                st.markdown(f"**Gold Answer:** `{selected.get('gold_answer')}`")
                st.markdown(f"**Answer Consistent:** `{bool(selected.get('consistent_answer'))}`")
                st.markdown(f"**Correctness Stable:** `{bool(selected.get('correctness_stable'))}`")
                st.markdown(f"**All Variants Correct:** `{bool(selected.get('all_correct'))}`")
                st.markdown("**Original Question**")
                st.write(str(selected.get("original_question", "")))

            with detail_right:
                variant_rows = []
                for variant in selected.get("variants", []):
                    variant_rows.append(
                        {
                            "Variant": variant.get("variant"),
                            "Question": variant.get("question"),
                            "Prediction": variant.get("prediction"),
                            "Correct": bool(variant.get("correct")),
                            "Latency ms": variant.get("latency_ms"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(variant_rows),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Question": st.column_config.TextColumn(width="large"),
                        "Prediction": st.column_config.TextColumn(width="medium"),
                    },
                )

        st.subheader("All Probe Tasks")
        st.dataframe(
            reliability_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Original Question": st.column_config.TextColumn(width="large"),
            },
        )



with tab_domain:
    st.subheader("Domain Adaptation")

    if not domain_adaptation:
        st.info("Domain adaptation summary is not available yet. Expected file: reports/domain_adaptation_summary.json")
    else:
        stored = domain_adaptation["stored_original_evaluator"]
        recomputed = domain_adaptation["recomputed_project_grader"]
        zero = stored["zero_shot"]
        lora = stored["lora"]
        delta = stored["delta"]
        outcomes = domain_adaptation["paired_outcomes_stored_anls"]

        st.markdown(
            f"""
<div class="fe-insight">
  Pix2Struct LoRA was trained on <strong>{domain_adaptation["train_pairs"]}</strong> QA pairs from
  <strong>{len(domain_adaptation["train_companies"])}</strong> companies and tested on
  <strong>{domain_adaptation["test_pairs"]}</strong> held-out questions from
  <strong>{", ".join(domain_adaptation["test_companies"])}</strong>. This is a separate domain-adaptation
  analysis, not part of the full 397-question main leaderboard.
</div>
""",
            unsafe_allow_html=True,
        )

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            kpi("Zero-Shot ANLS", f"{zero['overall_anls']:.3f}", "Pix2Struct on held-out subset")
        with d2:
            kpi("LoRA ANLS", f"{lora['overall_anls']:.3f}", "Fine-tuned Pix2Struct")
        with d3:
            kpi("ANLS Delta", f"{delta['anls']:+.3f}", f"{delta['relative_anls_change_pct']:+.1f}% relative")
        with d4:
            kpi("Exact Match Delta", f"{delta['exact_match']:+.3f}", "LoRA did not improve EM")

        st.markdown(
            """
<div class="fe-note">
  Interpretation: LoRA improves partial-answer similarity on held-out companies, but exact-match accuracy slightly drops.
  This suggests domain adaptation helps the model move closer to correct answers without reliably producing fully exact answers.
</div>
""",
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Held-Out Performance")
            metric_rows = [
                {
                    "Model": "Pix2Struct zero-shot",
                    "ANLS": zero["overall_anls"],
                    "Exact Match": zero["overall_exact_match"],
                    "Questions": zero["n"],
                },
                {
                    "Model": "Pix2Struct LoRA",
                    "ANLS": lora["overall_anls"],
                    "Exact Match": lora["overall_exact_match"],
                    "Questions": lora["n"],
                },
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="ANLS",
                    x=[row["Model"] for row in metric_rows],
                    y=[row["ANLS"] for row in metric_rows],
                    marker_color="#2563eb",
                    text=[f"{row['ANLS']:.3f}" for row in metric_rows],
                    textposition="outside",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Exact Match",
                    x=[row["Model"] for row in metric_rows],
                    y=[row["Exact Match"] for row in metric_rows],
                    marker_color="#16a34a",
                    text=[f"{row['Exact Match']:.3f}" for row in metric_rows],
                    textposition="outside",
                )
            )
            fig.update_layout(**plot_layout(360), barmode="group")
            fig.update_yaxes(title="Score", range=[0, max(lora["overall_anls"], zero["overall_anls"]) * 1.45])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with right:
            st.subheader("Training / Test Split")
            split_df = pd.DataFrame(
                [
                    {"Split": "Train", "Companies": ", ".join(domain_adaptation["train_companies"]), "QA Pairs": domain_adaptation["train_pairs"]},
                    {"Split": "Held-out Test", "Companies": ", ".join(domain_adaptation["test_companies"]), "QA Pairs": domain_adaptation["test_pairs"]},
                ]
            )
            st.dataframe(split_df, hide_index=True, use_container_width=True)

            outcome_df = pd.DataFrame(
                [
                    {"Outcome": "Improved ANLS", "Tasks": outcomes["improved"]},
                    {"Outcome": "Worse ANLS", "Tasks": outcomes["worse"]},
                    {"Outcome": "Unchanged ANLS", "Tasks": outcomes["unchanged"]},
                ]
            )
            st.dataframe(outcome_df, hide_index=True, use_container_width=True)

        st.subheader("Category Impact")
        category_rows = []
        for category in CATEGORY_ORDER:
            if category not in zero["by_category"]:
                continue
            z = zero["by_category"][category]
            l = lora["by_category"][category]
            category_rows.append(
                {
                    "Category": category_label(category),
                    "N": z["n"],
                    "Zero-Shot ANLS": z["anls"],
                    "LoRA ANLS": l["anls"],
                    "Delta ANLS": round(l["anls"] - z["anls"], 4),
                    "Zero-Shot EM": z["exact_match"],
                    "LoRA EM": l["exact_match"],
                    "Delta EM": round(l["exact_match"] - z["exact_match"], 4),
                }
            )

        category_df = pd.DataFrame(category_rows)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Zero-Shot ANLS",
                x=category_df["Category"],
                y=category_df["Zero-Shot ANLS"],
                marker_color="#7c3aed",
                text=[f"{v:.3f}" for v in category_df["Zero-Shot ANLS"]],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                name="LoRA ANLS",
                x=category_df["Category"],
                y=category_df["LoRA ANLS"],
                marker_color="#2563eb",
                text=[f"{v:.3f}" for v in category_df["LoRA ANLS"]],
                textposition="outside",
            )
        )
        fig.update_layout(**plot_layout(390), barmode="group")
        fig.update_yaxes(title="ANLS")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.dataframe(
            category_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Zero-Shot ANLS": st.column_config.NumberColumn(format="%.4f"),
                "LoRA ANLS": st.column_config.NumberColumn(format="%.4f"),
                "Delta ANLS": st.column_config.NumberColumn(format="%+.4f"),
                "Zero-Shot EM": st.column_config.NumberColumn(format="%.4f"),
                "LoRA EM": st.column_config.NumberColumn(format="%.4f"),
                "Delta EM": st.column_config.NumberColumn(format="%+.4f"),
            },
        )

        st.subheader("Largest Paired Changes")
        view = st.radio(
            "Examples",
            ["Top improvements", "Top regressions"],
            horizontal=True,
            key="domain_examples",
        )
        examples = domain_adaptation["top_improvements"] if view == "Top improvements" else domain_adaptation["top_regressions"]
        example_df = pd.DataFrame(
            [
                {
                    "Task ID": row["task_id"],
                    "Ticker": row["ticker"],
                    "Category": category_label(row["category"]),
                    "Gold": row["gold_answer"],
                    "Zero-Shot Prediction": row["zero_shot_prediction"],
                    "LoRA Prediction": row["lora_prediction"],
                    "Delta ANLS": row["stored_delta_anls"],
                    "Question": row["question"],
                }
                for row in examples
            ]
        )
        st.dataframe(
            example_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
                "Zero-Shot Prediction": st.column_config.TextColumn(width="medium"),
                "LoRA Prediction": st.column_config.TextColumn(width="medium"),
                "Delta ANLS": st.column_config.NumberColumn(format="%+.4f"),
            },
        )


with tab_dataset:
    st.subheader("Dataset Card")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        kpi("Questions", f"{len(tasks_df):,}", "Hand-curated QA pairs")
    with d2:
        kpi("Images", f"{tasks_df['image_path'].nunique():,}", "SEC screenshots")
    with d3:
        kpi("Companies", f"{tasks_df['ticker'].nunique():,}", "US public companies")
    with d4:
        kpi("Categories", f"{tasks_df['category'].nunique():,}", "Task types")

    left, right = st.columns([0.92, 1.08])
    with left:
        st.markdown(
            """
<div class="info-panel">
  <table>
    <tr><td>Name</td><td>Financial Document VQA Track A</td></tr>
    <tr><td>Source</td><td>Public SEC/company filing screenshots</td></tr>
    <tr><td>Construction</td><td>Hand-curated question-answer pairs</td></tr>
    <tr><td>Scope</td><td>English, US public-company filings</td></tr>
    <tr><td>Use</td><td>Non-commercial educational and portfolio evaluation</td></tr>
  </table>
  <br>
  <strong>Limitations</strong>
  <ul>
    <li>Single annotator, with no inter-annotator agreement study.</li>
    <li>Chart interpretation has only 17 examples.</li>
    <li>Visual quality varies because the data uses filing screenshots.</li>
    <li>Existing 2024 baselines do not provide evidence regions.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        counts = tasks_df["category"].value_counts().reindex(CATEGORY_ORDER)
        fig = px.bar(
            x=[category_label(category) for category in counts.index],
            y=counts.values,
            color=[category_label(category) for category in counts.index],
            color_discrete_sequence=["#2563eb", "#0f766e", "#dc2626", "#7c3aed"],
            labels={"x": "Category", "y": "Questions"},
        )
        fig.update_layout(**plot_layout(360), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Company Coverage")
    company_rows = (
        tasks_df.groupby(["ticker", "company"])
        .agg(questions=("task_id", "count"), images=("image_path", "nunique"))
        .reset_index()
        .sort_values(["questions", "ticker"], ascending=[False, True])
    )
    st.dataframe(company_rows, hide_index=True, use_container_width=True)

with tab_adapter:
    st.subheader("Run Your Own Model")
    st.write(
        "Create one prediction row per task, place the JSON file in results/track_a, "
        "then run the enrichment and report scripts."
    )

    st.code(
        """
def predict(image_path: str, question: str) -> str:
    image = load_image(image_path)
    answer = model.answer(image=image, question=question)
    return str(answer)
""".strip(),
        language="python",
    )

    st.subheader("Expected Result Row")
    st.code(
        """
{
  "task_id": "AAPL_001",
  "model": "YourModel",
  "prediction": "394,328",
  "gold_answer": "394,328",
  "category": "extractive",
  "latency_ms": 1250,
  "evidence_available": false,
  "evidence": null,
  "raw_output": "394,328"
}
""".strip(),
        language="json",
    )

    st.subheader("Workflow")
    st.code(
        """
python scripts/enrich_results.py
python scripts/generate_reports.py
streamlit run dashboard/app.py
""".strip(),
        language="bash",
    )

    st.info("The dashboard uses cached JSON artifacts, so viewing it does not require a GPU or API calls.")

st.divider()
st.markdown(
    "<div class='fe-footer'>FinAgent-Eval | Evaluation harness for financial document VQA</div>",
    unsafe_allow_html=True,
)

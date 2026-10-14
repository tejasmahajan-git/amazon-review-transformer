from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# AMAZON CUSTOMER INTELLIGENCE — POLISHED STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Amazon Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, consistent dashboard styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #17243A;
        --muted: #6B7890;
        --line: #E5EAF2;
        --panel: #FFFFFF;
        --canvas: #F4F7FB;
        --teal: #0F9D91;
        --blue: #4169E1;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp { background: var(--canvas); color: var(--navy); }
    .block-container {
        max-width: 1600px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #17243A;
        border-right: 1px solid #26364F;
    }
    [data-testid="stSidebar"] * { color: #F2F6FC; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #17243A; }
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] span {
        color: #17243A;
    }
    [data-testid="stSidebar"] hr { border-color: #35445A; }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--navy);
        letter-spacing: -0.035em;
    }
    h1 { font-size: 2.15rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.02rem !important; font-weight: 700 !important; }

    .hero {
        background: linear-gradient(115deg, #17243A 0%, #233B5E 62%, #167D83 100%);
        color: #FFFFFF;
        padding: 1.55rem 1.7rem;
        border-radius: 18px;
        margin-bottom: 1.15rem;
        box-shadow: 0 10px 28px rgba(23,36,58,.12);
    }
    .hero h1 { color: #FFFFFF !important; margin: 0 0 .3rem 0; }
    .hero p { color: #DDE8F6; margin: 0; font-size: .98rem; }
    .eyebrow {
        text-transform: uppercase; letter-spacing: .14em;
        font-size: .70rem; font-weight: 700; color: #8DE0D7;
        margin-bottom: .45rem;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.05rem;
        box-shadow: 0 3px 12px rgba(23,36,58,.035);
        min-height: 118px;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted); font-size: .83rem; font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--navy); font-size: 1.8rem; font-weight: 800;
    }
    [data-testid="stMetricDelta"] { font-size: .78rem; }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: .45rem .45rem .1rem .45rem;
        box-shadow: 0 3px 12px rgba(23,36,58,.025);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stTabs"] button { font-weight: 700; }
    .section-note { color: var(--muted); font-size: .88rem; margin-top: -.45rem; }
    .small-note { color: var(--muted); font-size: .78rem; }
    .status-pill {
        display: inline-block; background: #E2F5F1; color: #087F73;
        border-radius: 999px; padding: .28rem .65rem;
        font-size: .76rem; font-weight: 700;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 9px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Dataset discovery and loading
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "amazon_electronics_combined.csv",
    PROJECT_ROOT / "data" / "processed" / "merged_reviews.csv",
    PROJECT_ROOT / "data" / "processed" / "reviews_with_metadata.csv",
    PROJECT_ROOT / "data" / "processed" / "combined_reviews.csv",
    PROJECT_ROOT / "data" / "merged_reviews.csv",
    PROJECT_ROOT / "data" / "combined_reviews.csv",
    PROJECT_ROOT / "amazon_electronics_combined.csv",
    PROJECT_ROOT / "merged_reviews.csv",
    PROJECT_ROOT / "combined_reviews.csv",
    HERE / "amazon_electronics_combined.csv",
    HERE / "merged_reviews.csv",
    HERE / "combined_reviews.csv",
]


def discover_dataset() -> Path | None:
    for candidate in CANDIDATE_PATHS:
        if candidate.is_file():
            return candidate

    # As a fallback, search common project data folders, but avoid
    # selecting unrelated environment/package CSVs.
    search_roots = [PROJECT_ROOT / "data", PROJECT_ROOT]
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for candidate in root.glob("**/*.csv"):
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if any(part.lower() in {".venv", "venv", "node_modules", ".git"} for part in resolved.parts):
                    continue
                name = candidate.name.lower()
                if any(token in name for token in ("review", "amazon", "combined", "merged", "metadata")):
                    return candidate
        except (OSError, PermissionError):
            continue
    return None


@st.cache_data(show_spinner="Loading your review dataset…")
def read_dataset(path_string: str, modified_time: float) -> pd.DataFrame:
    # modified_time is included in the cache key so edits to the CSV reload it.
    return pd.read_csv(path_string, low_memory=False)


data_path = discover_dataset()
if data_path is None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Customer intelligence</div>
          <h1>Amazon Customer Intelligence</h1>
          <p>Your review dashboard is ready to run — it just needs to find the dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("I couldn't automatically find your merged reviews CSV.")
    st.markdown(
        f"""
        **Expected project structure**
        ```text
        {PROJECT_ROOT.name}/
        ├── data/
        │   └── amazon_electronics_combined.csv
        └── src/
            └── app.py
        ```
        """
    )
    st.write("Move your existing merged CSV into the `data` folder and name it `amazon_electronics_combined.csv`, or update `CANDIDATE_PATHS` near the top of `app.py` to match its actual path.")
    st.stop()

try:
    raw = read_dataset(str(data_path), data_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Couldn't read the dataset at `{data_path}`.")
    st.exception(exc)
    st.stop()

if raw.empty:
    st.error(f"The dataset at `{data_path}` contains no rows.")
    st.stop()

raw.columns = [str(column).strip() for column in raw.columns]


def find_column(candidates: list[str]) -> str | None:
    columns = {str(column).strip().lower(): column for column in raw.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


rating_col = find_column(["rating", "stars", "star_rating", "review_rating", "overall"])
text_col = find_column(["review_text", "review_body", "body", "review", "text"])
title_col = find_column(["review_title", "summary", "review_summary", "title"])
date_col = find_column(["timestamp", "review_date", "date", "created_at", "unix_review_time"])
product_col = find_column(["product_title", "product_name", "product_id", "asin", "parent_asin"])
brand_col = find_column(["brand", "brand_name"])
category_col = find_column(["main_category", "category", "product_category"])
verified_col = find_column(["verified_purchase", "verified", "is_verified_purchase"])
helpful_col = find_column(["helpful_votes", "helpful_vote", "helpful", "vote_count"])
sentiment_col = find_column([
    "sentiment_label", "predicted_sentiment", "sentiment",
    "distilbert_sentiment", "sentiment_class"
])
theme_col = find_column(["theme", "review_theme", "issue_theme", "complaint_theme", "praise_theme"])


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


df = raw.copy()

if rating_col:
    df["_rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    df.loc[~df["_rating"].between(1, 5), "_rating"] = np.nan
else:
    df["_rating"] = np.nan

df["_review_text"] = df[text_col].map(clean_text) if text_col else ""
df["_review_title"] = df[title_col].map(clean_text) if title_col else ""
df["_evidence"] = (df["_review_title"] + " — " + df["_review_text"]).str.strip(" —")

if date_col:
    dates = df[date_col]
    if pd.api.types.is_numeric_dtype(dates):
        numeric_dates = pd.to_numeric(dates, errors="coerce")
        non_null = numeric_dates.dropna()
        if len(non_null):
            unit = "ms" if non_null.abs().median() > 1e11 else "s"
            df["_date"] = pd.to_datetime(numeric_dates, errors="coerce", unit=unit, utc=True).dt.tz_convert(None)
        else:
            df["_date"] = pd.NaT
    else:
        df["_date"] = pd.to_datetime(dates, errors="coerce", utc=True).dt.tz_convert(None)
else:
    df["_date"] = pd.NaT

df["_helpful"] = pd.to_numeric(df[helpful_col], errors="coerce") if helpful_col else np.nan


def classify_sentiment(row) -> str:
    if sentiment_col:
        value = clean_text(row[sentiment_col]).lower()
        if any(token in value for token in ("positive", "pos", "label_2")):
            return "Positive"
        if any(token in value for token in ("negative", "neg", "label_0")):
            return "Negative"
        if any(token in value for token in ("neutral", "mixed", "label_1")):
            return "Neutral"

    rating = row["_rating"]
    if pd.isna(rating):
        return "Unknown"
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


df["_sentiment"] = df.apply(classify_sentiment, axis=1)
sentiment_method = (
    f"Existing label column: {sentiment_col}"
    if sentiment_col
    else "Rating proxy: 4–5 positive, 3 neutral, 1–2 negative; this is not NLP sentiment."
)

THEME_RULES = {
    "Battery & power": ["battery", "charging", "charge", "power", "drain", "battery life"],
    "Reliability & durability": ["broken", "defective", "faulty", "stopped working", "durability", "overheating", "failed"],
    "Performance & speed": ["slow", "lag", "freezes", "crash", "performance", "speed", "responsive"],
    "Setup & ease of use": ["setup", "installation", "instructions", "manual", "easy to use", "confusing"],
    "Compatibility & connectivity": ["bluetooth", "wifi", "wi-fi", "pairing", "connectivity", "disconnect", "compatible"],
    "Sound & display": ["sound", "audio", "speaker", "display", "screen", "picture quality", "resolution"],
    "Value & price": ["price", "value", "worth it", "overpriced", "expensive", "cheap"],
    "Build & design": ["build quality", "design", "material", "plastic", "lightweight", "finish"],
    "Delivery & packaging": ["delivery", "shipping", "packaging", "arrived", "damaged box"],
    "Customer support & warranty": ["customer service", "support", "warranty", "refund", "replacement", "return"],
    "Overall satisfaction": ["excellent", "love it", "amazing", "works well", "good quality", "satisfied", "recommend", "disappointed"],
}


def detect_themes(text: str) -> list[str]:
    text = clean_text(text).lower()
    matches = [
        theme for theme, keywords in THEME_RULES.items()
        if any(keyword in text for keyword in keywords)
    ]
    return matches or ["Other / not classified"]


if theme_col:
    df["_themes"] = df[theme_col].apply(
        lambda value: [clean_text(value)] if clean_text(value) else ["Other / not classified"]
    )
    theme_method = f"Existing theme column: {theme_col}"
else:
    df["_themes"] = df["_evidence"].apply(detect_themes)
    theme_method = "Transparent keyword rules applied to review title/text; not a trained topic model."


# ----------------------------
# Helpers for charts and filters
# ----------------------------
NAVY = "#17243A"
TEAL = "#0F9D91"
BLUE = "#4169E1"
CORAL = "#E66B5B"
GOLD = "#E5A63B"
MUTED = "#6B7890"
PALETTE = ["#0F9D91", "#4169E1", "#E5A63B", "#E66B5B", "#8B6AD9", "#4C9F70", "#8795AA"]


def style_fig(fig, height: int = 330):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=18, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=NAVY, size=12),
        title=dict(font=dict(family="Manrope, sans-serif", size=15, color=NAVY), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5EAF2", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False, linecolor="#E5EAF2")
    return fig


def multiselect_filter(frame: pd.DataFrame, label: str, column: str | None) -> pd.DataFrame:
    if not column:
        st.sidebar.caption(f"{label}: not available in this dataset")
        return frame
    values = frame[column].dropna().astype(str).unique().tolist()
    values = sorted(values, key=str.casefold)
    selected = st.sidebar.multiselect(label, options=values, default=[], key=f"filter_{label}")
    if selected:
        frame = frame[frame[column].astype(str).isin(selected)]
    return frame


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.markdown("## Explore reviews")
st.sidebar.caption("Choose filters to update every chart and table.")

filtered = df.copy()
filtered = multiselect_filter(filtered, "Brand", brand_col)
filtered = multiselect_filter(filtered, "Category", category_col)
filtered = multiselect_filter(filtered, "Product", product_col)

sentiment_options = sorted(filtered["_sentiment"].dropna().unique().tolist())
selected_sentiments = st.sidebar.multiselect(
    "Sentiment", options=sentiment_options, default=[], key="filter_sentiment"
)
if selected_sentiments:
    filtered = filtered[filtered["_sentiment"].isin(selected_sentiments)]

valid_dates = filtered["_date"].dropna()
if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    chosen_dates = st.sidebar.date_input(
        "Review date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="filter_date_range",
    )
    if isinstance(chosen_dates, (tuple, list)) and len(chosen_dates) == 2:
        start_date, end_date = chosen_dates
        filtered = filtered[
            filtered["_date"].notna()
            & filtered["_date"].dt.date.between(start_date, end_date)
        ]
else:
    st.sidebar.caption("Date range: no valid dates available")

min_mentions = st.sidebar.slider(
    "Minimum theme mentions", min_value=1, max_value=50, value=3, step=1,
    help="Used in the issue-priority and theme charts."
)
st.sidebar.divider()
st.sidebar.markdown("**Dataset status**")
st.sidebar.markdown(f"`{data_path.name}`")
st.sidebar.caption(f"{len(raw):,} rows · {len(raw.columns)} columns")
if st.sidebar.button("Reset all filters", use_container_width=True):
    for key in ["filter_Brand", "filter_Category", "filter_Product", "filter_sentiment", "filter_date_range"]:
        st.session_state.pop(key, None)
    st.rerun()

if filtered.empty:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Customer intelligence</div><h1>No matching reviews</h1><p>Adjust or reset the filters to bring reviews back into view.</p></div>',
        unsafe_allow_html=True,
    )
    st.warning("No reviews match the current filters.")
    st.stop()


# ----------------------------
# Page header and KPI row
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Amazon · Customer intelligence</div>
      <h1>Understand what customers really think.</h1>
      <p>Explore review trends, uncover recurring feedback themes, and identify issues worth investigating.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

avg_rating = filtered["_rating"].mean()
positive_pct = filtered["_sentiment"].eq("Positive").mean() * 100
negative_pct = filtered["_sentiment"].eq("Negative").mean() * 100
product_count = filtered[product_col].nunique() if product_col else None
dated = filtered.dropna(subset=["_date"])
date_coverage = (
    f"{dated['_date'].min():%d %b %Y} – {dated['_date'].max():%d %b %Y}"
    if not dated.empty else "Unavailable"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reviews in view", f"{len(filtered):,}", help="Number of reviews after applying all filters.")
k2.metric("Average rating", f"{avg_rating:.2f} / 5" if pd.notna(avg_rating) else "N/A")
k3.metric("Positive", f"{positive_pct:.1f}%")
k4.metric("Negative", f"{negative_pct:.1f}%")
k5.metric("Products", f"{product_count:,}" if product_count is not None else "N/A")

st.caption(
    f"Date coverage: {date_coverage}  ·  Valid ratings: {filtered['_rating'].notna().sum():,}/{len(filtered):,}  ·  "
    f"Sentiment: {sentiment_method}"
)

overview_tab, voice_tab, priorities_tab, evidence_tab = st.tabs(
    ["Overview", "Customer voice", "Issue priorities", "Review evidence"]
)


# ----------------------------
# Shared theme-level data
# ----------------------------
theme_rows = []
for _, row in filtered.iterrows():
    for theme in row["_themes"]:
        theme_rows.append({
            "Theme": theme,
            "Rating": row["_rating"],
            "Sentiment": row["_sentiment"],
            "Evidence": row["_evidence"],
            "Product": clean_text(row[product_col]) if product_col else "Unavailable",
            "Brand": clean_text(row[brand_col]) if brand_col else "Unavailable",
            "Category": clean_text(row[category_col]) if category_col else "Unavailable",
            "Date": row["_date"],
            "Helpful votes": row["_helpful"],
        })

theme_df = pd.DataFrame(theme_rows)
if theme_df.empty:
    theme_df = pd.DataFrame(columns=["Theme", "Rating", "Sentiment", "Evidence", "Product", "Brand", "Category", "Date", "Helpful votes"])

theme_summary = (
    theme_df.groupby("Theme", dropna=False)
    .agg(
        Mentions=("Theme", "size"),
        Average_rating=("Rating", "mean"),
        Negative_pct=("Sentiment", lambda s: (s == "Negative").mean() * 100),
        Positive_pct=("Sentiment", lambda s: (s == "Positive").mean() * 100),
        Low_rating_pct=("Rating", lambda s: (s <= 2).mean() * 100),
        Products=("Product", "nunique"),
    )
    .reset_index()
) if not theme_df.empty else pd.DataFrame()


with overview_tab:
    st.subheader("Review activity")
    st.markdown('<p class="section-note">Volume and rating signals across the currently selected reviews.</p>', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])

    with left:
        if not dated.empty:
            monthly = dated.assign(Month=dated["_date"].dt.to_period("M").dt.to_timestamp())
            monthly = monthly.groupby("Month").size().reset_index(name="Reviews")
            fig = px.area(
                monthly, x="Month", y="Reviews",
                title="Review volume over time",
                color_discrete_sequence=[TEAL],
            )
            fig.update_traces(line=dict(width=2.5), fillcolor="rgba(15,157,145,0.13)")
            st.plotly_chart(style_fig(fig, 350), use_container_width=True)
        else:
            st.info("No valid dates are available for a review-volume trend.")

    with right:
        rating_counts = (
            filtered.dropna(subset=["_rating"])["_rating"]
            .round().astype(int).value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename_axis("Stars").reset_index(name="Reviews")
        )
        rating_counts["Rating"] = rating_counts["Stars"].astype(str) + " star"
        fig = px.bar(
            rating_counts, x="Rating", y="Reviews", text="Reviews",
            title="Star rating distribution",
            color="Stars", color_continuous_scale=["#E66B5B", "#E5A63B", "#4169E1", "#0F9D91"],
        )
        fig.update_layout(coloraxis_showscale=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        sentiment_counts = filtered["_sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        fig = px.pie(
            sentiment_counts, names="Sentiment", values="Reviews", hole=0.68,
            title="Sentiment mix",
            color="Sentiment",
            color_discrete_map={"Positive": TEAL, "Neutral": GOLD, "Negative": CORAL, "Unknown": "#A8B2C2"},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    with c_right:
        if not theme_summary.empty:
            top_themes = theme_summary.sort_values("Mentions", ascending=True).tail(8)
            fig = px.bar(
                top_themes, x="Mentions", y="Theme", orientation="h",
                title="Most-mentioned feedback themes",
                text="Mentions", color_discrete_sequence=[BLUE],
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        else:
            st.info("No feedback themes are available.")

    st.caption("Review counts reflect the selected filters. Theme labels are rule-based unless the dataset already contains a theme field.")


with voice_tab:
    st.subheader("What customers mention")
    st.markdown('<p class="section-note">Compare how often themes appear with the sentiment associated with those mentions.</p>', unsafe_allow_html=True)
    if theme_summary.empty:
        st.info("No themes are available for these reviews.")
    else:
        filtered_themes = theme_summary[theme_summary["Mentions"] >= min_mentions].copy()
        if filtered_themes.empty:
            st.info("No themes meet the minimum mention threshold. Lower the threshold in the sidebar.")
        else:
            left, right = st.columns(2)
            with left:
                complaint = filtered_themes.sort_values(["Negative_pct", "Mentions"]).tail(10).sort_values("Negative_pct")
                fig = px.bar(
                    complaint, x="Negative_pct", y="Theme", orientation="h",
                    title="Themes with the highest negative share",
                    labels={"Negative_pct": "Negative mentions (%)"},
                    color_discrete_sequence=[CORAL],
                    hover_data={"Mentions": True, "Average_rating": ":.2f"},
                )
                fig.update_xaxes(range=[0, 100])
                st.plotly_chart(style_fig(fig, 390), use_container_width=True)
            with right:
                praise = filtered_themes.sort_values(["Positive_pct", "Mentions"]).tail(10).sort_values("Positive_pct")
                fig = px.bar(
                    praise, x="Positive_pct", y="Theme", orientation="h",
                    title="Themes with the highest positive share",
                    labels={"Positive_pct": "Positive mentions (%)"},
                    color_discrete_sequence=[TEAL],
                    hover_data={"Mentions": True, "Average_rating": ":.2f"},
                )
                fig.update_xaxes(range=[0, 100])
                st.plotly_chart(style_fig(fig, 390), use_container_width=True)

            st.markdown("#### Theme summary")
            display_summary = filtered_themes.rename(columns={
                "Average_rating": "Average rating",
                "Negative_pct": "Negative share (%)",
                "Positive_pct": "Positive share (%)",
                "Low_rating_pct": "1–2 star share (%)",
                "Products": "Products mentioned",
            }).copy()
            for column in ["Negative share (%)", "Positive share (%)", "1–2 star share (%)"]:
                display_summary[column] = display_summary[column].round(1)
            display_summary["Average rating"] = display_summary["Average rating"].round(2)
            st.dataframe(
                display_summary[["Theme", "Mentions", "Average rating", "Negative share (%)", "Positive share (%)", "1–2 star share (%)", "Products mentioned"]]
                .sort_values("Mentions", ascending=False),
                use_container_width=True, hide_index=True,
            )
    st.caption(f"Theme method: {theme_method} These are associations, not proof that a theme caused a rating.")


with priorities_tab:
    st.subheader("Issues worth investigating")
    st.markdown('<p class="section-note">A transparent triage score combines theme frequency, negative share, and low-rating share. It is a screening aid—not a validated business-impact model.</p>', unsafe_allow_html=True)
    if theme_summary.empty:
        st.info("No themes are available.")
    else:
        priority = theme_summary[theme_summary["Mentions"] >= min_mentions].copy()
        if priority.empty:
            st.info("No themes meet the minimum mention threshold. Lower the threshold in the sidebar.")
        else:
            priority["Sample size"] = np.select(
                [priority["Mentions"] >= 30, priority["Mentions"] >= 10],
                ["Higher (30+)", "Moderate (10–29)"],
                default="Small (<10)",
            )
            frequency_rank = priority["Mentions"].rank(pct=True)
            negative_rank = priority["Negative_pct"].rank(pct=True)
            low_rank = priority["Low_rating_pct"].rank(pct=True)
            priority["Triage score"] = 100 * (
                0.50 * frequency_rank + 0.30 * negative_rank + 0.20 * low_rank
            )
            priority = priority.sort_values("Triage score", ascending=False)

            fig = px.bar(
                priority.head(10).sort_values("Triage score"),
                x="Triage score", y="Theme", orientation="h",
                title="Top 10 themes by triage score",
                labels={"Triage score": "Relative triage score (0–100)"},
                color="Triage score", color_continuous_scale=["#B9E7DF", "#0F9D91", "#17243A"],
                hover_data={"Mentions": True, "Negative_pct": ":.1f", "Low_rating_pct": ":.1f"},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 390), use_container_width=True)

            priority_display = priority.rename(columns={
                "Average_rating": "Average rating",
                "Negative_pct": "Negative share (%)",
                "Positive_pct": "Positive share (%)",
                "Low_rating_pct": "1–2 star share (%)",
                "Products": "Products mentioned",
            }).copy()
            for column in ["Negative share (%)", "Positive share (%)", "1–2 star share (%)", "Triage score"]:
                priority_display[column] = priority_display[column].round(1)
            priority_display["Average rating"] = priority_display["Average rating"].round(2)
            st.dataframe(
                priority_display[[
                    "Theme", "Triage score", "Mentions", "Sample size",
                    "Negative share (%)", "1–2 star share (%)", "Average rating",
                    "Products mentioned",
                ]],
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Download issue-priority table",
                data=priority_display.to_csv(index=False).encode("utf-8"),
                file_name="amazon_issue_priorities.csv",
                mime="text/csv",
            )


with evidence_tab:
    st.subheader("Supporting review evidence")
    st.markdown('<p class="section-note">Use real review text to validate the patterns before making product or customer-experience decisions.</p>', unsafe_allow_html=True)

    evidence = filtered.copy()
    search_text = st.text_input("Search review text", placeholder="Try: battery, broken, easy to use…")
    if search_text.strip():
        search_lower = search_text.strip().lower()
        evidence = evidence[evidence["_evidence"].str.lower().str.contains(re.escape(search_lower), na=False)]

    theme_options = sorted({theme for themes in evidence["_themes"] for theme in themes})
    chosen_theme = st.selectbox("Theme", ["All themes"] + theme_options)
    if chosen_theme != "All themes":
        evidence = evidence[evidence["_themes"].apply(lambda themes: chosen_theme in themes)]

    sort_options = ["Most helpful", "Lowest rating", "Highest rating", "Newest first", "Oldest first"]
    sort_by = st.selectbox("Sort evidence", sort_options)
    if sort_by == "Most helpful":
        evidence = evidence.sort_values("_helpful", ascending=False, na_position="last")
    elif sort_by == "Lowest rating":
        evidence = evidence.sort_values("_rating", ascending=True, na_position="last")
    elif sort_by == "Highest rating":
        evidence = evidence.sort_values("_rating", ascending=False, na_position="last")
    elif sort_by == "Newest first":
        evidence = evidence.sort_values("_date", ascending=False, na_position="last")
    else:
        evidence = evidence.sort_values("_date", ascending=True, na_position="last")

    st.caption(f"{len(evidence):,} matching reviews")
    show = evidence.head(500).copy()
    show["Rating"] = show["_rating"]
    show["Sentiment"] = show["_sentiment"]
    show["Review date"] = show["_date"].dt.strftime("%Y-%m-%d")
    show["Review evidence"] = show["_evidence"].replace("", "(No review text available)")
    show["Theme(s)"] = show["_themes"].apply(lambda themes: ", ".join(themes))
    if product_col:
        show["Product"] = show[product_col].astype(str)
    if brand_col:
        show["Brand"] = show[brand_col].astype(str)
    if helpful_col:
        show["Helpful votes"] = show["_helpful"]

    columns_to_show = ["Rating", "Sentiment", "Theme(s)", "Review evidence", "Review date"]
    if "Product" in show.columns:
        columns_to_show.insert(0, "Product")
    if "Brand" in show.columns:
        columns_to_show.insert(1 if "Product" in show.columns else 0, "Brand")
    if "Helpful votes" in show.columns:
        columns_to_show.append("Helpful votes")

    st.dataframe(
        show[columns_to_show],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Review evidence": st.column_config.TextColumn("Review evidence", width="large"),
            "Theme(s)": st.column_config.TextColumn("Theme(s)", width="medium"),
        },
    )

    st.download_button(
        "Download filtered reviews",
        data=evidence.drop(columns=["_themes"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="amazon_filtered_reviews.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    f"""
    <div class="small-note">
    <b>Data notes:</b> Source file: {data_path.name} · {len(raw):,} source rows · {len(raw.columns)} source columns.
    Sentiment method: {sentiment_method} Theme method: {theme_method}
    </div>
    """,
    unsafe_allow_html=True,
)

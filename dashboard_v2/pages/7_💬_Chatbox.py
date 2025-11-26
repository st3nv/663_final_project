import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    load_geodata,
    load_population_data,
    load_crime_data,
    load_housing_data,
    get_exclude_settings,
    apply_zip_exclusion,
    prepare_map_data,
)


st.set_page_config(page_title="Insight Chatbox", page_icon="💬", layout="wide")

st.title("💬 Chatbox")
st.markdown(
    "Chat with a simple assistant to explore how home values and crime relate across Chicago ZIP codes. "
    "The answers are grounded in the same data used in the other dashboard pages."
)

# Sidebar settings (reuse shared controls so filters stay consistent)
st.sidebar.header("📌 Dashboard")
exclude_zips, exclude_2025 = get_exclude_settings()

# Load and filter data
gdf = load_geodata()
gdf = apply_zip_exclusion(gdf, exclude_zips)
pop_df = load_population_data()
crime_df = load_crime_data()
zhvi_df = load_housing_data()

if exclude_2025:
    if "year" in crime_df.columns:
        crime_df = crime_df[crime_df["year"] != 2025]
    if "Year" in zhvi_df.columns:
        zhvi_df = zhvi_df[zhvi_df["Year"] != 2025]

map_data, gdf_merged, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year = prepare_map_data(
    gdf, crime_df, zhvi_df, pop_df
)


def build_zip_df(map_data_list, gdf_frame, crime_latest_year, zhvi_latest_year):
    """Construct a ZIP-level dataframe with latest crime rate and home value."""
    zip_name_map = {}
    if "ZipName" in gdf_frame.columns:
        zip_name_map = (
            gdf_frame[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )

    metrics = []
    for z in map_data_list:
        zip_code = str(z["zip"])
        pop_val = float(z["population"])
        crimes_latest = float(z["crimes"].get(str(crime_latest_year), 0.0))
        zhvi_latest = float(z["zhvi"].get(str(zhvi_latest_year), 0.0))
        crime_rate_latest = (crimes_latest / pop_val * 1000.0) if pop_val > 0 else 0.0
        metrics.append(
            {
                "ZIP": zip_code,
                "ZipName": zip_name_map.get(zip_code, None),
                "population": pop_val,
                "crimes_latest": crimes_latest,
                "crime_rate_latest": crime_rate_latest,
                "zhvi_latest": zhvi_latest,
            }
        )

    df = pd.DataFrame(metrics)
    if df.empty:
        return df

    # Build a readable label
    df["Label"] = df.apply(
        lambda r: f"{r['ZIP']} – {r['ZipName']}"
        if isinstance(r["ZipName"], str) and r["ZipName"]
        else r["ZIP"],
        axis=1,
    )

    return df


zip_df = build_zip_df(map_data, gdf_merged, crime_max_year, zhvi_max_year)
zip_df = zip_df[(zip_df["zhvi_latest"] > 0) & (zip_df["crime_rate_latest"] > 0)].copy()

if zip_df.empty:
    st.warning(
        "No ZIP-level data with both crime rate and home values is available with the current filters. "
        "Try relaxing the ZIP exclusions or including 2025 data."
    )
    st.stop()

# Pre-compute core statistics that the chatbot will reference
corr = zip_df["crime_rate_latest"].corr(zip_df["zhvi_latest"])
crime_med = zip_df["crime_rate_latest"].median()
zhvi_med = zip_df["zhvi_latest"].median()


def classify_archetype(row):
    if row["crime_rate_latest"] <= crime_med and row["zhvi_latest"] >= zhvi_med:
        return "High value / safer"
    if row["crime_rate_latest"] > crime_med and row["zhvi_latest"] >= zhvi_med:
        return "High value / riskier"
    if row["crime_rate_latest"] <= crime_med and row["zhvi_latest"] < zhvi_med:
        return "Lower value / safer"
    return "Lower value / riskier"


zip_df["archetype"] = zip_df.apply(classify_archetype, axis=1)


def describe_correlation(value):
    if pd.isna(value):
        return "I couldn't compute a correlation between crime and prices."
    strength = "very weak"
    if abs(value) >= 0.2:
        strength = "weak"
    if abs(value) >= 0.4:
        strength = "moderate"
    if abs(value) >= 0.6:
        strength = "strong"

    if value > 0:
        direction = "higher crime is associated with higher home values"
    elif value < 0:
        direction = "higher crime is associated with lower home values"
    else:
        direction = "crime and home values are basically unrelated"

    return (
        f"The correlation between latest crime rate and home values across ZIPs is **{value:.2f}** "
        f"({strength}). Overall, {direction} in Chicago's ZIP-level data."
    )


def format_zip_list(df, n=5):
    rows = df.head(n)
    if rows.empty:
        return "I couldn't find any ZIP codes that match that pattern."

    lines = []
    for _, r in rows.iterrows():
        label = r["Label"]
        crime = r["crime_rate_latest"]
        price = r["zhvi_latest"]
        lines.append(
            f"- **{label}** · crime rate ~{crime:.1f} per 1,000, "
            f"home value ~${price:,.0f}"
        )
    return "\n".join(lines)


def build_insights_summary(df: pd.DataFrame) -> str:
    """Create a compact text summary of key price–crime insights to feed into the LLM."""
    lines = []

    lines.append(
        f"Number of ZIP codes with valid data: {len(df)} "
        f"(latest crime_rate_latest and zhvi_latest)."
    )
    if not pd.isna(corr):
        lines.append(
            f"Correlation between crime_rate_latest and zhvi_latest across ZIPs: {corr:.3f}."
        )
    lines.append(
        f"Median crime_rate_latest: {crime_med:.2f} crimes per 1,000 residents; "
        f"median zhvi_latest: ${zhvi_med:,.0f}."
    )

    archetype_counts = df["archetype"].value_counts()
    total = int(archetype_counts.sum())
    if total > 0:
        lines.append("ZIPs grouped into four archetypes (above/below medians):")
        for label in [
            "High value / safer",
            "High value / riskier",
            "Lower value / safer",
            "Lower value / riskier",
        ]:
            if label in archetype_counts:
                count = int(archetype_counts[label])
                pct = 100.0 * count / total
                lines.append(f"- {label}: {count} ZIPs ({pct:.0f}%)")

    # Top examples for each archetype
    def add_group(title, subset, ascending):
        subset_sorted = subset.sort_values("zhvi_latest", ascending=ascending)
        if subset_sorted.empty:
            return
        lines.append(title)
        lines.append(format_zip_list(subset_sorted, n=5))

    add_group(
        "High value / safer examples (expensive but below-median crime):",
        df[df["archetype"] == "High value / safer"],
        ascending=False,
    )
    add_group(
        "Lower value / safer examples (more affordable with below-median crime):",
        df[df["archetype"] == "Lower value / safer"],
        ascending=True,
    )
    add_group(
        "High value / riskier examples (expensive with above-median crime):",
        df[df["archetype"] == "High value / riskier"],
        ascending=False,
    )
    add_group(
        "Lower value / riskier examples (more affordable but above-median crime):",
        df[df["archetype"] == "Lower value / riskier"],
        ascending=True,
    )

    return "\n".join(lines)


INSIGHTS_SUMMARY = build_insights_summary(zip_df)


def call_llm(chat_history, user_query: str) -> str:
    """Call a Groq-hosted LLM (GPT-style) with the user's question and the pre-computed insights summary."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return (
            "The `openai` Python package is not installed. "
            "Install it with `pip install openai` to enable the Insight Chatbox."
        )

    # Prefer GROQ_API_KEY env var; fall back to dashboard_v2/key.txt
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        key_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "key.txt",
        )
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip() or None
        except OSError:
            api_key = None

    if not api_key:
        return (
            "Groq API key is not configured. "
            "Set `GROQ_API_KEY` as an environment variable or put it in `dashboard_v2/key.txt`."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    system_prompt = (
        "You are a helpful data analyst focused on the relationship between home values (ZHVI) and "
        "crime rates across Chicago ZIP codes. "
        "You are given a structured summary of the latest-year data from a dashboard. "
        "Use only that summary and the conversation when answering.\n\n"
        "Be concise, quantitative when possible, and explain patterns clearly for non-technical users.\n\n"
        "DATA SUMMARY:\n"
        f"{INSIGHTS_SUMMARY}"
    )

    history_text_lines = []
    for msg in chat_history:
        role = msg.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        history_text_lines.append(f"{prefix}: {msg.get('content', '')}")

    full_prompt = (
        system_prompt
        + "\n\nCONVERSATION SO FAR:\n"
        + "\n".join(history_text_lines)
        + "\n\nUser question:\n"
        + user_query
    )

    try:
        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=full_prompt,
            temperature=0.3,
        )
        # `response.output_text` is a convenience property in Groq's examples
        return getattr(response, "output_text", str(response))
    except Exception as e:
        return f"Error while calling the Groq language model: {e}"


with st.expander("💡 What this chatbot can do", expanded=True):
    st.markdown(
        "- Uses a **Groq-hosted GPT-style model** via the OpenAI-compatible API, grounded in the ZIP-level crime and ZHVI data used elsewhere in the dashboard.\n"
        "- Focuses on the latest available year, summarizing how prices and crime relate across ZIP codes.\n"
    )

# Simple chat-style interface using Streamlit's chat components
if "price_crime_chat_history" not in st.session_state:
    st.session_state["price_crime_chat_history"] = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a data-driven assistant focused on the relationship between **home values** and "
                "**crime rates** in Chicago. Ask me about correlations, high value but safer ZIPs, "
                "more affordable but safer areas, or where crime and prices are both high."
            ),
        }
    ]

for msg in st.session_state["price_crime_chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about prices and crime across Chicago ZIP codes...")

if prompt:
    st.session_state["price_crime_chat_history"].append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = call_llm(st.session_state["price_crime_chat_history"], prompt)
    st.session_state["price_crime_chat_history"].append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)

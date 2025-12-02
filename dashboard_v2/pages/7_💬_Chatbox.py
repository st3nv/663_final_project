import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import math

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


def build_panel_model_dataframe_for_chat(
    _gdf_in: pd.DataFrame,
    pop_2021: pd.DataFrame,
    crime_data: pd.DataFrame,
    zhvi_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a per-(ZIP, year) panel with crime rate and ZHVI, mirroring the Statistical Analysis page.
    This is used only to generate a concise text summary for the chatbot.
    """
    gdf_local = _gdf_in.copy()
    gdf_local["ZIP"] = gdf_local["ZIP"].astype(str)

    # Attach 2021 population to ZIP shapes
    pop_local = pop_2021.copy()
    pop_local["ZIP"] = pop_local["ZIP"].astype(str)
    gdf_local = gdf_local.merge(
        pop_local[["ZIP", "Population - Total"]],
        on="ZIP",
        how="left",
    )
    gdf_local["Population - Total"] = gdf_local["Population - Total"].fillna(0.0)

    # Approx distance to Loop (degree space, used as static location control)
    loop_lat, loop_lon = 41.8781, -87.6298
    gdf_local["centroid"] = gdf_local.geometry.centroid
    gdf_local["centroid_lon"] = gdf_local["centroid"].x
    gdf_local["centroid_lat"] = gdf_local["centroid"].y
    gdf_local["distance_from_loop"] = np.sqrt(
        (gdf_local["centroid_lat"] - loop_lat) ** 2
        + (gdf_local["centroid_lon"] - loop_lon) ** 2
    )

    # Crime by ZIP-year
    crime_local = crime_data.copy()
    crime_local["ZIP"] = crime_local["ZIP"].astype(str)
    crime_yearly = (
        crime_local.groupby(["ZIP", "year"])["crime_count"]
        .sum()
        .reset_index()
    )

    # ZHVI by ZIP-year
    zhvi_local = zhvi_data.copy()
    zhvi_local["RegionName"] = zhvi_local["RegionName"].astype(str)
    zhvi_yearly = (
        zhvi_local.groupby(["RegionName", "Year"])["Zhvi"]
        .mean()
        .reset_index()
    )
    zhvi_yearly = zhvi_yearly.rename(
        columns={"RegionName": "ZIP", "Year": "year"},
    )

    # Inner join on overlapping (ZIP, year)
    panel = crime_yearly.merge(zhvi_yearly, on=["ZIP", "year"], how="inner")

    # Attach static ZIP attributes
    meta_cols = ["ZIP", "ZipName", "Population - Total", "distance_from_loop"]
    meta_df = gdf_local[meta_cols].drop_duplicates()
    panel = panel.merge(meta_df, on="ZIP", how="left")

    # Compute crime rate per 1,000 residents
    panel["crime_rate"] = np.where(
        panel["Population - Total"] > 0,
        panel["crime_count"] / panel["Population - Total"] * 1000.0,
        np.nan,
    )
    panel["log_zhvi"] = np.log(panel["Zhvi"])

    # Drop rows without key variables
    panel = panel.dropna(subset=["crime_rate", "log_zhvi", "distance_from_loop"])

    return panel


def build_statistical_analysis_summary(panel_df: pd.DataFrame) -> str:
    """
    Run a pooled regression with year fixed effects on the ZIP-year panel
    and return a concise text summary (mirrors the Statistical Analysis tab).
    """
    if panel_df.empty or "year" not in panel_df.columns:
        return (
            "I could not build a ZIP-year panel that overlaps crime and ZHVI data, "
            "so detailed pooled regression results are not available for this filter combination."
        )

    reg_df = panel_df.copy()
    years = sorted(reg_df["year"].unique())
    year_min = int(min(years))
    year_max = int(max(years))

    # Core regressors
    X_core = reg_df[["crime_rate", "distance_from_loop"]].to_numpy(dtype=float)
    y_vals = reg_df["log_zhvi"].to_numpy(dtype=float).reshape(-1, 1)

    # Add intercept and year fixed effects
    year_dummies = pd.get_dummies(
        reg_df["year"].astype(int),
        prefix="year",
        drop_first=True,
    )
    X_parts = [np.ones((X_core.shape[0], 1)), X_core]
    if not year_dummies.empty:
        X_parts.append(year_dummies.to_numpy(dtype=float))
    X_design = np.column_stack(X_parts)

    try:
        beta = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y_vals)
    except np.linalg.LinAlgError:
        return (
            "The pooled regression with year fixed effects was numerically unstable, "
            "so I cannot summarize its coefficients here."
        )

    # Overall fit
    fitted = (X_design @ beta).ravel()
    resid = reg_df["log_zhvi"].to_numpy(dtype=float) - fitted
    ss_tot = np.sum((reg_df["log_zhvi"] - reg_df["log_zhvi"].mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    beta_crime = float(beta[1, 0])

    lines = []
    lines.append(
        f"Pooled panel regression uses {len(reg_df):,} ZIP-year observations from {year_min}–{year_max}."
    )
    lines.append(
        "Model: log(ZHVI_it) = β0 + β1·crime_rate_it + β2·distance_from_Loop_i + year fixed effects + ε_it."
    )
    if math.isfinite(beta_crime) and math.isfinite(r2):
        lines.append(
            f"Estimated β1 (crime_rate) ≈ {beta_crime:.4f} in log-ZHVI units; pooled R² ≈ {r2:.3f}."
        )
    elif math.isfinite(beta_crime):
        lines.append(
            f"Estimated β1 (crime_rate) ≈ {beta_crime:.4f} in log-ZHVI units."
        )

    if beta_crime < 0:
        lines.append(
            "Negative β1 means that, holding distance and year constant, higher crime is associated with lower home values."
        )
    elif beta_crime > 0:
        lines.append(
            "Positive β1 means that, holding distance and year constant, higher crime is associated with higher home values."
        )
    else:
        lines.append(
            "β1 is approximately zero, suggesting little linear relationship between crime and home values after controlling for distance and year."
        )

    # Year-by-year slopes (partial correlation controlling for distance)
    slope_records = []
    for y_year in years:
        df_y = reg_df[reg_df["year"] == y_year]
        if (
            df_y["crime_rate"].var() > 0
            and df_y["log_zhvi"].var() > 0
            and len(df_y) > 5
        ):
            X_y = df_y[["crime_rate", "distance_from_loop"]].to_numpy(dtype=float)
            y_y = df_y["log_zhvi"].to_numpy(dtype=float).reshape(-1, 1)
            Xy_design = np.column_stack([np.ones(X_y.shape[0]), X_y])
            try:
                beta_y = np.linalg.inv(Xy_design.T @ Xy_design) @ (Xy_design.T @ y_y)
                slope_records.append(float(beta_y[1, 0]))
            except np.linalg.LinAlgError:
                continue

    if slope_records:
        slopes = np.array(slope_records, dtype=float)
        neg_share = float(np.mean(slopes < 0.0))
        pos_share = float(np.mean(slopes > 0.0))
        lines.append(
            "Year-by-year regressions (controlling for distance) mostly show a negative crime effect: "
            f"crime coefficients are negative in about {neg_share * 100:.0f}% of years "
            f"and positive in about {pos_share * 100:.0f}% of years."
        )

    return "\n".join(lines)


PANEL_MODEL_DF = build_panel_model_dataframe_for_chat(gdf, pop_df, crime_df, zhvi_df)
STATISTICAL_ANALYSIS_SUMMARY = build_statistical_analysis_summary(PANEL_MODEL_DF)
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
        "You are given structured summaries of (1) latest-year ZIP-level patterns and "
        "(2) a pooled regression and ZIP-year analysis from the Statistical Analysis tab. "
        "Use only those summaries and the conversation when answering.\n\n"
        "Be concise, quantitative when possible, and explain patterns clearly for non-technical users.\n\n"
        "DATA SUMMARY – latest-year ZIP-level patterns:\n"
        f"{INSIGHTS_SUMMARY}\n\n"
        "ADDITIONAL STATISTICAL ANALYSIS – pooled regression and ZIP-year patterns:\n"
        f"{STATISTICAL_ANALYSIS_SUMMARY}"
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
        "- Also incorporates results from the **Statistical Analysis** page, including a pooled regression with year fixed effects and ZIP-year patterns, so it can answer questions about trends and controlled relationships over time.\n"
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

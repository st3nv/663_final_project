# pages/3_🏠_Housing_Prices.py
import streamlit as st
import streamlit.components.v1 as components
import json
import sys
import os
import pandas as pd
import plotly.express as px

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    load_geodata, load_population_data, load_crime_data, load_housing_data,
    get_exclude_settings, apply_zip_exclusion, prepare_map_data
)
from utils.map_generator import generate_map_html

st.set_page_config(page_title="Housing Prices Map", page_icon="🏠", layout="wide")

# Get selected ZIP from query params (set by map clicks)
query_params = st.query_params
selected_zip_from_url = query_params.get("selected_zip", None)
if isinstance(selected_zip_from_url, list):
    selected_zip_from_url = selected_zip_from_url[0] if selected_zip_from_url else None

st.title("🏠 Housing Prices")
st.markdown("Track home value changes across Chicago neighborhoods over time")

# Sidebar settings
st.sidebar.header("📌 Dashboard")
exclude_zips, exclude_2025 = get_exclude_settings()

# Load data
gdf = load_geodata()
gdf = apply_zip_exclusion(gdf, exclude_zips)
pop_2021 = load_population_data()
crime_df = load_crime_data()
zhvi_df = load_housing_data()

if exclude_2025:
    if 'year' in crime_df.columns:
        crime_df = crime_df[crime_df['year'] != 2025]
    if 'Year' in zhvi_df.columns:
        zhvi_df = zhvi_df[zhvi_df['Year'] != 2025]

# Prepare map data
map_data, gdf_merged, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year = prepare_map_data(
    gdf, crime_df, zhvi_df, pop_2021
)
map_data_json = json.dumps(map_data)

# Prepare bar chart race data (reuses map_data like v1 dashboard)
race_zhvi_records = []
race_crime_records = []
for z in map_data:
    # Home value race (ZHVI)
    for y_str, val in z["zhvi"].items():
        year = int(y_str)
        if val <= 0:
            continue
        race_zhvi_records.append({"ZIP": z["zip"], "Year": year, "Zhvi": val})

    # Crime rate race (per 1,000 residents)
    pop_val = float(z.get("population", 0.0) or 0.0)
    for y_str, crimes in z["crimes"].items():
        year = int(y_str)
        rate = crimes / pop_val * 1000.0 if pop_val > 0 else 0.0
        race_crime_records.append(
            {"ZIP": z["zip"], "Year": year, "CrimeRate": rate}
        )

race_zhvi_df = pd.DataFrame(race_zhvi_records)
race_crime_df = pd.DataFrame(race_crime_records)

# Get available ZIP codes (only those with housing data)
zips_with_data = set(zhvi_df['RegionName'].astype(str).unique())
available_zips = sorted([str(z) for z in gdf_merged['ZIP'].unique() if str(z) in zips_with_data])

# Validate selected ZIP from URL
if selected_zip_from_url and selected_zip_from_url not in available_zips:
    selected_zip_from_url = None

# Use map selection directly for the trend view
selected_zip = selected_zip_from_url

# Generate and display map (pass selected_zip to highlight it)
html_code = generate_map_html(
    map_data_json, "Housing Prices", 
    crime_min_year, crime_max_year, 
    zhvi_min_year, zhvi_max_year,
    selected_zip=selected_zip_from_url
)


# Combined statistics row (market stats + YoY)
st.subheader("📊 Housing Market Overview")

if selected_zip:
    zip_zhvi = zhvi_df[zhvi_df['RegionName'] == selected_zip]
    yearly_avg = zip_zhvi.groupby('Year')['Zhvi'].mean()
else:
    yearly_avg = zhvi_df.groupby('Year')['Zhvi'].mean()

if len(yearly_avg) > 0:
    yoy_changes = yearly_avg.pct_change() * 100

    col_list = st.columns(7)

    with col_list[0]:
        st.metric(
            "Average Value",
            f"${yearly_avg.mean():,.0f}",
            help="Average home value across all years"
        )
    with col_list[1]:
        max_year = int(yearly_avg.idxmax())
        st.metric(
            "Peak Year",
            f"{max_year}",
            f"${yearly_avg.max():,.0f}"
        )
    with col_list[2]:
        min_year_val = int(yearly_avg.idxmin())
        st.metric(
            "Lowest Year",
            f"{min_year_val}",
            f"${yearly_avg.min():,.0f}"
        )
    with col_list[3]:
        first_value = yearly_avg.iloc[0]
        last_value = yearly_avg.iloc[-1]
        total_growth = (
            ((last_value - first_value) / first_value) * 100
            if first_value > 0
            else 0
        )
        st.metric(
            "Total Growth",
            f"{total_growth:+.1f}%",
            f"${last_value - first_value:+,.0f}"
        )

    if len(yoy_changes.dropna()) > 0:
        with col_list[4]:
            best_year = int(yoy_changes.idxmax())
            best_growth = yoy_changes.max()
            st.metric(
                "Best Year",
                f"{best_year}",
                f"{best_growth:+.1f}%"
            )
        with col_list[5]:
            worst_year = int(yoy_changes.idxmin())
            worst_growth = yoy_changes.min()
            st.metric(
                "Worst Year",
                f"{worst_year}",
                f"{worst_growth:+.1f}%"
            )
        with col_list[6]:
            avg_annual = yoy_changes.mean()
            st.metric(
                "Avg Annual Growth",
                f"{avg_annual:+.1f}%"
            )
st.markdown("---")


components.html(html_code, height=900, scrolling=False)

st.markdown("---")
st.subheader("🏃‍♀️ Price Race")

df_race = race_zhvi_df.copy()
df_race = df_race[df_race["Zhvi"] > 0]
if df_race.empty:
    st.info("No ZHVI data available for the bar chart race.")
else:
    # Use ZIP + neighborhood label when available
    if "ZipName" in gdf_merged.columns:
        name_map = (
            gdf_merged[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )
    else:
        name_map = {}

    df_race["ZipLabel"] = df_race["ZIP"].apply(
        lambda z: f"{z} - {name_map.get(z, '')}" if name_map.get(z, "") else z
    )
    # Sort within each year so rankings move over time
    df_race = df_race.sort_values(["Year", "Zhvi"])

    # Ensure all ZIP labels appear on the y-axis (no automatic skipping)
    zip_labels_order = sorted(df_race["ZipLabel"].unique())

    fig_race = px.bar(
        df_race,
        x="Zhvi",
        y="ZipLabel",
        color="ZIP",
        orientation="h",
        animation_frame="Year",
        range_x=[0, df_race["Zhvi"].max() * 1.1],
        labels={
            "Zhvi": "Average Home Value (USD)",
            "ZipLabel": "ZIP & Neighborhood",
        },
        title="Home Value Race",
    )

if "fig_race" in locals() and fig_race is not None:
    fig_race.update_layout(
        template="simple_white",
        height=800,
        margin=dict(l=80, r=40, t=60, b=40),
        legend_title_text="ZIP Code",
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.35)",
            gridwidth=1,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=zip_labels_order,
            tickmode="array",
            tickvals=zip_labels_order,
            ticktext=zip_labels_order,
            automargin=True,
        ),
    )
    st.plotly_chart(fig_race, use_container_width=True)

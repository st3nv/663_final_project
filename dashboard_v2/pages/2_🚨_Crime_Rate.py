# pages/2_🚨_Crime_Rate.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    load_geodata, load_population_data, load_crime_data, load_housing_data,
    get_exclude_settings, apply_zip_exclusion, prepare_map_data
)
from utils.map_generator import generate_map_html

st.set_page_config(page_title="Crime Rate Map", page_icon="🚨", layout="wide")
# Get selected ZIP from query params (set by map clicks)
query_params = st.query_params
selected_zip_from_url = query_params.get("selected_zip", None)
if isinstance(selected_zip_from_url, list):
    selected_zip_from_url = selected_zip_from_url[0] if selected_zip_from_url else None
st.title("🚨 Crime Rate Analysis")
st.markdown("Explore crime rates per 1,000 residents with animated timeline")

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

# Get available ZIP codes
available_zips = sorted(gdf_merged['ZIP'].astype(str).unique().tolist())

# Validate selected ZIP from URL
if selected_zip_from_url and selected_zip_from_url not in available_zips:
    selected_zip_from_url = None

# Use map selection directly for the trend view
selected_zip = selected_zip_from_url

# Generate and display map (pass selected_zip to highlight it)
html_code = generate_map_html(
    map_data_json, "Crime Rate", 
    crime_min_year, crime_max_year, 
    zhvi_min_year, zhvi_max_year,
    selected_zip=selected_zip_from_url
)


# Statistics Section
st.subheader("📊 Crime Statistics")

if selected_zip:
    zip_crimes = crime_df[crime_df['ZIP'] == selected_zip]
    zip_pop = pop_2021[pop_2021['ZIP'] == selected_zip]['Population - Total'].values
    population = zip_pop[0] if len(zip_pop) > 0 else 1
    yearly_rates = zip_crimes.groupby('year')['crime_count'].sum() / population * 1000
else:
    total_pop = pop_2021['Population - Total'].sum()
    yearly_crimes = crime_df.groupby('year')['crime_count'].sum()
    yearly_rates = yearly_crimes / total_pop * 1000

if len(yearly_rates) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Rate",
            f"{yearly_rates.mean():.2f}",
            help="Average crime rate per 1,000 residents across all years"
        )
    with col2:
        max_year = int(yearly_rates.idxmax())
        st.metric(
            "Peak Year",
            f"{max_year}"        )
    with col3:
        min_year_val = int(yearly_rates.idxmin())
        st.metric(
            "Lowest Year", 
            f"{min_year_val}"        )
    with col4:
        # Calculate trend
        first_years = yearly_rates.head(3).mean()
        last_years = yearly_rates.tail(3).mean()
        trend_pct = ((last_years - first_years) / first_years) * 100 if first_years > 0 else 0
        trend_label = "📈 Up" if trend_pct > 5 else "📉 Down" if trend_pct < -5 else "➡️ Stable"
        st.metric(
            "Trend",
            trend_label,
            f"{trend_pct:+.1f}%"
        )

    # Only compute city-wide safest/most dangerous when no ZIP is selected
    if not selected_zip:
        latest_year = int(crime_df["year"].max())

        # Aggregate crimes by ZIP for the latest year, only for ZIPs on the map
        valid_zips = set(gdf_merged["ZIP"].astype(str))
        crime_latest = (
            crime_df[crime_df["year"] == latest_year]
            .copy()
        )
        crime_latest["ZIP"] = crime_latest["ZIP"].astype(str)
        crime_latest = crime_latest[crime_latest["ZIP"].isin(valid_zips)]

        crime_by_zip = (
            crime_latest.groupby("ZIP")["crime_count"]
            .sum()
            .reset_index()
        )

        pop_by_zip = (
            pop_2021[["ZIP", "Population - Total"]]
            .copy()
        )
        pop_by_zip["ZIP"] = pop_by_zip["ZIP"].astype(str)

        rate_df = crime_by_zip.merge(pop_by_zip, on="ZIP", how="left")
        rate_df = rate_df[rate_df["Population - Total"] > 0]
        if len(rate_df) > 0:
            rate_df["rate"] = (
                rate_df["crime_count"] / rate_df["Population - Total"] * 1000
            )

            # Attach neighborhood names (ZipName) when available
            if "ZipName" in gdf_merged.columns:
                name_lookup = (
                    gdf_merged[["ZIP", "ZipName"]]
                    .drop_duplicates()
                    .copy()
                )
                name_lookup["ZIP"] = name_lookup["ZIP"].astype(str)
                rate_df = rate_df.merge(name_lookup, on="ZIP", how="left")

            most_dangerous = rate_df.loc[rate_df["rate"].idxmax()]
            safest = rate_df.loc[rate_df["rate"].idxmin()]

            dangerous_label = (
                f'{most_dangerous["ZipName"]}'
                if "ZipName" in most_dangerous.index and pd.notna(most_dangerous["ZipName"])
                else str(most_dangerous["ZIP"])
            )
            safest_label = (
                f'{safest["ZipName"]}'
                if "ZipName" in safest.index and pd.notna(safest["ZipName"])
                else str(safest["ZIP"])
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Most Dangerous Neighborhood",
                    dangerous_label,
                    f'{most_dangerous["rate"]:.2f} crimes per 1,000 residents ({latest_year})',
                    delta_arrow="off"
                )
            with col_b:
                st.metric(
                    "Safest Neighborhood",
                    safest_label,
                    f'{safest["rate"]:.2f} crimes per 1,000 residents ({latest_year})',
                    delta_arrow="off"
                )
        
st.markdown("---")

components.html(html_code, height=900, scrolling=False)


# Additional info
with st.expander("ℹ️ How to use"):
    st.markdown(f"""
    **Interactive Map:**
    - **Click on a ZIP code** on the map to select it - the chart below will update automatically
    - **Click outside Chicago** (on the gray area) to reset to overall view
    - Use the **Play** button to animate through years ({crime_min_year}–{crime_max_year})
    - Drag the **slider** to jump to a specific year
    
    **Trend Chart:**
    - Shows crime rate changes over time as a filled step chart
    - When a ZIP is selected, it is automatically compared against the Chicago average
    - Hover over points for exact values
    
    **Data:** {crime_min_year}–{crime_max_year} | {len(gdf_merged)} ZIP codes
    """)

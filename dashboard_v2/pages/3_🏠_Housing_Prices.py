# pages/3_🏠_Housing_Prices.py
import streamlit as st
import streamlit.components.v1 as components
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

st.set_page_config(page_title="Housing Prices Map", page_icon="🏠", layout="wide")

# Get selected ZIP from query params (set by map clicks)
query_params = st.query_params
selected_zip_from_url = query_params.get("selected_zip", None)
if isinstance(selected_zip_from_url, list):
    selected_zip_from_url = selected_zip_from_url[0] if selected_zip_from_url else None

st.title("🏠 Housing Prices")
st.markdown("Track home value changes across Chicago neighborhoods over time")

# Sidebar settings
st.sidebar.header("📌 Chicago Dashboard")
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
                f"{best_growth:+.1f}% growth"
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


components.html(html_code, height=840, scrolling=False)

# Additional info
with st.expander("ℹ️ How to use"):
    st.markdown(f"""
    **Interactive Map:**
    - **Click on a ZIP code** to select it – the trend panel below the map and the summary metrics will update automatically
    - **Click outside Chicago** (on the gray area) to reset to the overall view
    - Use the **Play** button to animate through years ({zhvi_min_year}–{zhvi_max_year})
    - Drag the **slider** to jump to a specific year
    
    **Trend Panel:**
    - Shows home value changes over time using a stepped line/area chart
    - When a ZIP is selected, it is automatically compared against the Chicago average
    - Hover over points for exact values
    
    **About ZHVI:** Zillow Home Value Index - a smoothed, seasonally adjusted measure of typical home values
    
    **Data:** {zhvi_min_year}–{zhvi_max_year} | {len(available_zips)} ZIP codes with data
    """)

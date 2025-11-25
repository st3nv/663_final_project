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

st.title("🏠 Housing Prices")
st.markdown("Track home value changes across Chicago neighborhoods over time")

# Sidebar settings
st.sidebar.header("📌 Chicago Dashboard")
exclude_zips = get_exclude_settings()

# Load data
gdf = load_geodata()
gdf = apply_zip_exclusion(gdf, exclude_zips)
pop_2021 = load_population_data()
crime_df = load_crime_data()
zhvi_df = load_housing_data()

# Prepare map data
map_data, gdf_merged, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year = prepare_map_data(
    gdf, crime_df, zhvi_df, pop_2021
)
map_data_json = json.dumps(map_data)

# Generate and display map
html_code = generate_map_html(
    map_data_json, "Housing Prices", 
    crime_min_year, crime_max_year, 
    zhvi_min_year, zhvi_max_year
)
components.html(html_code, height=750, scrolling=False)

# Housing Market Statistics
st.subheader("Housing Market Statistics")
st.info("💡 Use the controls in the top-right corner of the map to see how home values have changed over time.")

# Calculate statistics
latest_year = zhvi_max_year
earliest_year = zhvi_min_year

latest_zhvi = zhvi_df[zhvi_df['Year'] == latest_year].groupby('RegionName')['Zhvi'].mean()
earliest_zhvi = zhvi_df[zhvi_df['Year'] == earliest_year].groupby('RegionName')['Zhvi'].mean()

col1, col2, col3 = st.columns(3)
with col1:
    avg_latest = latest_zhvi.mean()
    st.metric(f"Avg Home Value ({latest_year})", f"${avg_latest:,.0f}")
with col2:
    avg_earliest = earliest_zhvi.mean()
    st.metric(f"Avg Home Value ({earliest_year})", f"${avg_earliest:,.0f}")
with col3:
    pct_change = ((avg_latest - avg_earliest) / avg_earliest) * 100
    st.metric("Overall Growth", f"{pct_change:.1f}%")

# Additional info
with st.expander("ℹ️ About this map"):
    st.markdown("""
    This map shows the Zillow Home Value Index (ZHVI) across Chicago ZIP codes over time.
    
    **How to use:**
    - Use the **Play** button to animate through years
    - Drag the **slider** to jump to a specific year
    - Click **Reset** to return to the earliest year
    - Click on any ZIP code to see detailed housing statistics
    
    **About ZHVI:** The Zillow Home Value Index is a smoothed, seasonally adjusted measure of the typical home value.
    """)

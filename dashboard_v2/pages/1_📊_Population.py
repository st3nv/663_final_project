# pages/1_📊_Population.py
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

st.set_page_config(page_title="Population Map", page_icon="📊", layout="wide")

st.title("📊 Population Distribution")
st.markdown("View population distribution across Chicago ZIP codes (2021 Census data)")

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

# Generate and display map
html_code = generate_map_html(
    map_data_json, "Population", 
    crime_min_year, crime_max_year, 
    zhvi_min_year, zhvi_max_year
)
components.html(html_code, height=620, scrolling=False)

# Statistics below the map
st.subheader("Population Statistics (2021)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Population", f"{int(gdf_merged['Population - Total'].sum()):,}")
with col2:
    st.metric("Average ZIP Population", f"{int(gdf_merged['Population - Total'].mean()):,}")
with col3:
    st.metric("Number of ZIP Codes", len(gdf_merged))

# Additional info
with st.expander("ℹ️ About this map"):
    st.markdown("""
    This map shows the population distribution across Chicago ZIP codes based on 2021 Census data.
    
    **How to use:**
    - Click on any ZIP code to see detailed population information
    - Use the settings in the sidebar to exclude specific ZIP codes
    - Darker colors indicate higher population density
    """)

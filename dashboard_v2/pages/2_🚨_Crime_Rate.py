# pages/2_🚨_Crime_Rate.py
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

st.set_page_config(page_title="Crime Rate Map", page_icon="🚨", layout="wide")

st.title("🚨 Crime Rate Analysis")
st.markdown("Explore crime rates per 1,000 residents with animated timeline")

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
    map_data_json, "Crime Rate", 
    crime_min_year, crime_max_year, 
    zhvi_min_year, zhvi_max_year
)
components.html(html_code, height=750, scrolling=False)

# Additional analysis
st.subheader("Additional Analysis")
st.info("💡 Use the controls in the top-right corner of the map to play the animation, adjust the year, or reset.")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Data Range:** {crime_min_year} - {crime_max_year}")
with col2:
    st.markdown(f"**ZIP Codes Displayed:** {len(gdf_merged)}")

# Additional info
with st.expander("ℹ️ About this map"):
    st.markdown("""
    This map shows crime rates (crimes per 1,000 residents) across Chicago ZIP codes over time.
    
    **How to use:**
    - Use the **Play** button to animate through years
    - Drag the **slider** to jump to a specific year
    - Click **Reset** to return to the earliest year
    - Click on any ZIP code to see detailed crime statistics
    
    **Note:** Crime rate is calculated as (Total Crimes / Population) × 1000
    """)

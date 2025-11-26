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

# Get selected ZIP from query params (set by map clicks)
query_params = st.query_params
selected_zip_from_url = query_params.get("selected_zip", None)
if isinstance(selected_zip_from_url, list):
    selected_zip_from_url = selected_zip_from_url[0] if selected_zip_from_url else None

st.title("🚨 Crime Rate Analysis")
st.markdown("Explore crime rates per 1,000 residents with animated timeline")

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
            f"{max_year}",
            f"Rate: {yearly_rates.max():.2f}"
        )
    with col3:
        min_year_val = int(yearly_rates.idxmin())
        st.metric(
            "Lowest Year", 
            f"{min_year_val}",
            f"Rate: {yearly_rates.min():.2f}"
        )
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
        
st.markdown("---")

components.html(html_code, height=840, scrolling=False)


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

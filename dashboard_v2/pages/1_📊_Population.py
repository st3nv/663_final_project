# pages/1_📊_Population.py
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
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

# Generate and display map
html_code = generate_map_html(
    map_data_json, "Population",
    crime_min_year, crime_max_year,
    zhvi_min_year, zhvi_max_year
)

# Population statistics
st.subheader("Population Statistics (2021)")
total_population = gdf_merged["Population - Total"].sum()
avg_population = gdf_merged["Population - Total"].mean()
zip_count = len(gdf_merged)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Population", f"{int(total_population):,}")
with col2:
    st.metric("Average ZIP Population", f"{int(avg_population):,}")
with col3:
    st.metric("Number of ZIP Codes", zip_count)

# Additional statistics using ZIP names
if zip_count > 0:
    pop_series = gdf_merged["Population - Total"]
    max_idx = pop_series.idxmax()
    min_idx = pop_series.idxmin()
    max_row = gdf_merged.loc[max_idx]
    min_row = gdf_merged.loc[min_idx]

    has_zip_name = "ZipName" in gdf_merged.columns
    max_label = f"ZIP {max_row['ZIP']}" if not has_zip_name else f"ZIP {max_row['ZIP']} – {max_row['ZipName']}"
    min_label = f"ZIP {min_row['ZIP']}" if not has_zip_name else f"ZIP {min_row['ZIP']} – {min_row['ZipName']}"

    median_population = int(pop_series.median())

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(
            "Median ZIP Population",
            f"{median_population:,}"
        )
    with col5:
        st.metric(
            "Most Populated ZIP",
            f"{int(max_row['Population - Total']):,}",
            help=max_label
        )
    with col6:
        st.metric(
            "Least Populated ZIP",
            f"{int(min_row['Population - Total']):,}",
            help=min_label
        )

st.markdown("---")
components.html(html_code, height=620, scrolling=False)

st.markdown("---")
st.subheader("Top 10 Most Populated ZIP Codes")

# Top 10 ZIP codes by population
top_cols = ["ZIP", "Population - Total"]
if "ZipName" in gdf_merged.columns:
    top_cols.append("ZipName")

top10 = (
    gdf_merged[top_cols]
    .sort_values("Population - Total", ascending=False)
    .head(10)
    .copy()
)

top10["ZIP"] = top10["ZIP"].astype(str)

# Build display label: \"00000 - Name\" when ZipName is available
if "ZipName" in top10.columns:
    top10["zip_display"] = top10["ZIP"] + " - " + top10["ZipName"].astype(str)
else:
    top10["zip_display"] = top10["ZIP"]

# Horizontal bar chart, sorted by population, using a single color
if "ZipName" in top10.columns:
    y_field = alt.Y(
        "zip_display:N",
        sort="-x",
        title="",
        axis=alt.Axis(labelLimit=0)
    )
    tooltip_fields = ["ZIP", "ZipName", "Population - Total"]
else:
    y_field = alt.Y(
        "zip_display:N",
        sort="-x",
        title="ZIP",
        axis=alt.Axis(labelLimit=0)
    )
    tooltip_fields = ["ZIP", "Population - Total"]

chart_height = max(320, 32 * len(top10))

top10_chart = (
    alt.Chart(top10)
    .mark_bar(color="#3778bf")
    .encode(
        x=alt.X("Population - Total:Q", title="Population (2021)"),
        y=y_field,
        tooltip=tooltip_fields
    )
    .properties(height=chart_height)
)
st.altair_chart(top10_chart, use_container_width=True)

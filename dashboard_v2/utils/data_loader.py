# utils/data_loader.py - Shared data loading utilities
import streamlit as st
import geopandas as gpd
import pandas as pd
import json

@st.cache_data
def load_geodata(geojson_path="data/chicago_zipcodes.geojson"):
    """Load and cache the GeoJSON data"""
    gdf = gpd.read_file(geojson_path)
    gdf['ZIP'] = gdf['ZIP'].astype(str)
    return gdf

@st.cache_data
def load_population_data(pop_path="data/Chicago_Population_Counts.csv"):
    """Load and cache population data"""
    pop_df = pd.read_csv(pop_path)
    pop_2021 = pop_df[pop_df['Year'] == 2021].copy()
    pop_2021['ZIP'] = pop_2021['Geography'].astype(str)
    return pop_2021

@st.cache_data
def load_crime_data(crime_path="data/chicago_crime_preprocessed.csv"):
    """Load and cache crime data"""
    crime_df = pd.read_csv(crime_path)
    crime_df['ZIP'] = crime_df['ZIP'].astype(str)
    return crime_df

@st.cache_data
def load_housing_data(zhvi_path="data/chicago_zhvi_preprocessed.csv"):
    """Load and cache housing data"""
    zhvi_df = pd.read_csv(zhvi_path)
    zhvi_df['RegionName'] = zhvi_df['RegionName'].astype(str)
    return zhvi_df

def apply_zip_exclusion(gdf, exclude_zips):
    """Filter out excluded ZIP codes"""
    if exclude_zips:
        return gdf[~gdf['ZIP'].isin(exclude_zips)]
    return gdf

def get_exclude_settings():
    """Shared ZIP code exclusion and year settings for sidebar"""
    with st.sidebar.expander("⚙️ Settings", expanded=True):
        exclude_enabled = st.checkbox("Exclude specific ZIP codes", value=True)
        
        if exclude_enabled:
            default_exclude = "60602, 60603, 60604, 60666"
            exclude_input = st.text_input(
                "ZIP codes to exclude",
                value=default_exclude,
                help="Enter ZIP codes separated by commas"
            )
            
            if exclude_input.strip():
                exclude_zips = [zip_code.strip() for zip_code in exclude_input.split(',') if zip_code.strip()]
                st.caption(f"📍 Excluding {len(exclude_zips)} ZIP code(s)")
            else:
                exclude_zips = []
        else:
            exclude_zips = []
            st.caption("📍 Showing all ZIP codes")

        exclude_2025 = st.checkbox(
            "Exclude year 2025",
            value=True,
            help="When checked, any 2025 values are removed from maps and trend charts."
        )
    
    return exclude_zips, exclude_2025

def prepare_map_data(gdf, crime_df, zhvi_df, pop_2021):
    """Prepare data for JavaScript map visualization"""
    # Merge population data
    gdf = gdf.merge(pop_2021[['ZIP', 'Population - Total']], on='ZIP', how='left')
    gdf['Population - Total'] = gdf['Population - Total'].fillna(0)
    
    # Get year ranges
    crime_min_year = int(crime_df['year'].min())
    crime_max_year = int(crime_df['year'].max())
    zhvi_min_year = int(zhvi_df['Year'].min())
    zhvi_max_year = int(zhvi_df['Year'].max())
    
    # Prepare crime data for all years
    crime_by_year = {}
    for year in range(crime_min_year, crime_max_year + 1):
        crime_year = crime_df[crime_df['year'] == year].groupby('ZIP')['crime_count'].sum().reset_index()
        crime_year.columns = ['ZIP', 'total_crimes']
        crime_by_year[year] = crime_year.set_index('ZIP')['total_crimes'].to_dict()
    
    # Prepare housing data for all years
    zhvi_by_year = {}
    for year in range(zhvi_min_year, zhvi_max_year + 1):
        zhvi_year = zhvi_df[zhvi_df['Year'] == year].groupby('RegionName')['Zhvi'].mean().reset_index()
        zhvi_year.columns = ['ZIP', 'avg_zhvi']
        zhvi_by_year[year] = zhvi_year.set_index('ZIP')['avg_zhvi'].to_dict()
    
    # Prepare data for JavaScript
    map_data = []
    for _, row in gdf.iterrows():
        zip_code = row['ZIP']
        population = float(row['Population - Total'])
        
        # Get crimes for all years
        crimes_all_years = {}
        for year in range(crime_min_year, crime_max_year + 1):
            crimes = crime_by_year[year].get(zip_code, 0)
            crimes_all_years[str(year)] = float(crimes)
        
        # Get housing prices for all years
        zhvi_all_years = {}
        for year in range(zhvi_min_year, zhvi_max_year + 1):
            zhvi = zhvi_by_year[year].get(zip_code, 0)
            zhvi_all_years[str(year)] = float(zhvi)
        
        # Get centroid for positioning
        centroid = row['geometry'].centroid
        
        # Get geometry coordinates for polygon drawing
        if row['geometry'].geom_type == 'Polygon':
            coords = list(row['geometry'].exterior.coords)
        elif row['geometry'].geom_type == 'MultiPolygon':
            largest = max(row['geometry'].geoms, key=lambda p: p.area)
            coords = list(largest.exterior.coords)
        else:
            coords = []
        
        map_data.append({
            'zip': zip_code,
            'population': population,
            'crimes': crimes_all_years,
            'zhvi': zhvi_all_years,
            'centroid_lat': centroid.y,
            'centroid_lng': centroid.x,
            'coordinates': [[coord[1], coord[0]] for coord in coords]
        })
    
    return map_data, gdf, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year

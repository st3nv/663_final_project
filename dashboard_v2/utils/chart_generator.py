# utils/chart_generator.py - Altair chart generation utilities
import altair as alt
import pandas as pd


def create_crime_rate_chart(crime_df, pop_df, selected_zip=None, min_year=None, max_year=None):
    """
    Create a filled step area chart showing crime rate over time.
    
    Args:
        crime_df: DataFrame with crime data (columns: ZIP, year, crime_count)
        pop_df: DataFrame with population data (columns: ZIP, Population - Total)
        selected_zip: ZIP code to display, or None for overall Chicago
        min_year: Minimum year to display
        max_year: Maximum year to display
    
    Returns:
        Altair chart object
    """
    # Ensure ZIP columns are strings
    crime_df = crime_df.copy()
    crime_df['ZIP'] = crime_df['ZIP'].astype(str)
    pop_df = pop_df.copy()
    pop_df['ZIP'] = pop_df['ZIP'].astype(str)
    
    # Get year range
    if min_year is None:
        min_year = int(crime_df['year'].min())
    if max_year is None:
        max_year = int(crime_df['year'].max())
    
    if selected_zip is None:
        # Calculate overall Chicago crime rate by year
        yearly_crimes = crime_df.groupby('year')['crime_count'].sum().reset_index()
        total_population = pop_df['Population - Total'].sum()
        yearly_crimes['crime_rate'] = (yearly_crimes['crime_count'] / total_population) * 1000
        yearly_crimes['area'] = 'All Chicago'
        chart_data = yearly_crimes[['year', 'crime_rate', 'area']].copy()
        title = 'Chicago Overall Crime Rate Over Time'
        color = '#1f77b4'
    else:
        # Calculate crime rate for specific ZIP
        zip_crimes = crime_df[crime_df['ZIP'] == str(selected_zip)].groupby('year')['crime_count'].sum().reset_index()
        zip_pop = pop_df[pop_df['ZIP'] == str(selected_zip)]['Population - Total'].values
        
        if len(zip_pop) > 0 and zip_pop[0] > 0:
            population = zip_pop[0]
            zip_crimes['crime_rate'] = (zip_crimes['crime_count'] / population) * 1000
        else:
            zip_crimes['crime_rate'] = 0
        
        zip_crimes['area'] = f'ZIP {selected_zip}'
        chart_data = zip_crimes[['year', 'crime_rate', 'area']].copy()
        title = f'Crime Rate Over Time - ZIP {selected_zip}'
        color = '#ff6600'
    
    # Filter to year range and ensure year is integer
    chart_data = chart_data[(chart_data['year'] >= min_year) & (chart_data['year'] <= max_year)].copy()
    chart_data['year'] = chart_data['year'].astype(int)
    
    # Sort by year
    chart_data = chart_data.sort_values('year').reset_index(drop=True)
    
    # Create the filled step area chart using mark_area with step interpolation
    area = alt.Chart(chart_data).mark_area(
        interpolate='step-after',
        opacity=0.5,
        color=color
    ).encode(
        x=alt.X('year:Q',
                title='Year',
                scale=alt.Scale(domain=[min_year, max_year]),
                axis=alt.Axis(format='d', tickCount=10, labelAngle=-45)),
        y=alt.Y('crime_rate:Q',
                title='Crime Rate (per 1,000 residents)',
                scale=alt.Scale(zero=True)),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('crime_rate:Q', title='Crime Rate', format='.2f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    # Add line on top for better visibility
    line = alt.Chart(chart_data).mark_line(
        interpolate='step-after',
        strokeWidth=2,
        color=color
    ).encode(
        x=alt.X('year:Q'),
        y=alt.Y('crime_rate:Q')
    )
    
    # Add points for each data point
    points = alt.Chart(chart_data).mark_circle(
        size=50,
        color=color
    ).encode(
        x=alt.X('year:Q'),
        y=alt.Y('crime_rate:Q'),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('crime_rate:Q', title='Crime Rate', format='.2f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    chart = (area + line + points).properties(
        title=alt.Title(text=title, fontSize=16, anchor='start'),
        width='container',
        height=350
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=True,
        gridOpacity=0.3,
        labelFontSize=11,
        titleFontSize=13
    )
    
    return chart


def create_housing_price_chart(zhvi_df, selected_zip=None, min_year=None, max_year=None):
    """
    Create a filled step area chart showing housing prices over time.
    
    Args:
        zhvi_df: DataFrame with ZHVI data (columns: RegionName, Year, Zhvi)
        selected_zip: ZIP code to display, or None for overall Chicago
        min_year: Minimum year to display
        max_year: Maximum year to display
    
    Returns:
        Altair chart object
    """
    # Ensure ZIP column is string
    zhvi_df = zhvi_df.copy()
    zhvi_df['RegionName'] = zhvi_df['RegionName'].astype(str)
    
    # Get year range
    if min_year is None:
        min_year = int(zhvi_df['Year'].min())
    if max_year is None:
        max_year = int(zhvi_df['Year'].max())
    
    if selected_zip is None:
        # Calculate overall Chicago average ZHVI by year
        yearly_zhvi = zhvi_df.groupby('Year')['Zhvi'].mean().reset_index()
        yearly_zhvi['area'] = 'All Chicago'
        chart_data = yearly_zhvi.rename(columns={'Year': 'year', 'Zhvi': 'zhvi'})
        title = 'Chicago Overall Home Value Index Over Time'
        color = '#2171b5'
    else:
        # Get ZHVI for specific ZIP
        zip_zhvi = zhvi_df[zhvi_df['RegionName'] == str(selected_zip)].groupby('Year')['Zhvi'].mean().reset_index()
        zip_zhvi['area'] = f'ZIP {selected_zip}'
        chart_data = zip_zhvi.rename(columns={'Year': 'year', 'Zhvi': 'zhvi'})
        title = f'Home Value Index Over Time - ZIP {selected_zip}'
        color = '#ff6600'
    
    # Filter to year range and ensure year is integer
    chart_data = chart_data[(chart_data['year'] >= min_year) & (chart_data['year'] <= max_year)].copy()
    chart_data['year'] = chart_data['year'].astype(int)
    
    # Sort by year
    chart_data = chart_data.sort_values('year').reset_index(drop=True)
    
    # Handle empty data
    if len(chart_data) == 0:
        chart_data = pd.DataFrame({'year': [min_year], 'zhvi': [0], 'area': ['No Data']})
    
    # Create the filled step area chart
    area = alt.Chart(chart_data).mark_area(
        interpolate='step-after',
        opacity=0.5,
        color=color
    ).encode(
        x=alt.X('year:Q',
                title='Year',
                scale=alt.Scale(domain=[min_year, max_year]),
                axis=alt.Axis(format='d', tickCount=10, labelAngle=-45)),
        y=alt.Y('zhvi:Q',
                title='Home Value Index ($)',
                scale=alt.Scale(zero=True),
                axis=alt.Axis(format='$,.0f')),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('zhvi:Q', title='Home Value', format='$,.0f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    # Add line on top
    line = alt.Chart(chart_data).mark_line(
        interpolate='step-after',
        strokeWidth=2,
        color=color
    ).encode(
        x=alt.X('year:Q'),
        y=alt.Y('zhvi:Q')
    )
    
    # Add points
    points = alt.Chart(chart_data).mark_circle(
        size=50,
        color=color
    ).encode(
        x=alt.X('year:Q'),
        y=alt.Y('zhvi:Q'),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('zhvi:Q', title='Home Value', format='$,.0f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    chart = (area + line + points).properties(
        title=alt.Title(text=title, fontSize=16, anchor='start'),
        width='container',
        height=350
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=True,
        gridOpacity=0.3,
        labelFontSize=11,
        titleFontSize=13
    )
    
    return chart


def create_comparison_crime_chart(crime_df, pop_df, selected_zip, min_year=None, max_year=None):
    """
    Create a comparison step chart showing both Chicago average and selected ZIP crime rates.
    """
    crime_df = crime_df.copy()
    crime_df['ZIP'] = crime_df['ZIP'].astype(str)
    pop_df = pop_df.copy()
    pop_df['ZIP'] = pop_df['ZIP'].astype(str)
    
    if min_year is None:
        min_year = int(crime_df['year'].min())
    if max_year is None:
        max_year = int(crime_df['year'].max())
    
    # Chicago overall
    yearly_crimes_all = crime_df.groupby('year')['crime_count'].sum().reset_index()
    total_population = pop_df['Population - Total'].sum()
    yearly_crimes_all['crime_rate'] = (yearly_crimes_all['crime_count'] / total_population) * 1000
    yearly_crimes_all['area'] = 'All Chicago'
    
    # Selected ZIP
    zip_crimes = crime_df[crime_df['ZIP'] == str(selected_zip)].groupby('year')['crime_count'].sum().reset_index()
    zip_pop = pop_df[pop_df['ZIP'] == str(selected_zip)]['Population - Total'].values
    if len(zip_pop) > 0 and zip_pop[0] > 0:
        zip_crimes['crime_rate'] = (zip_crimes['crime_count'] / zip_pop[0]) * 1000
    else:
        zip_crimes['crime_rate'] = 0
    zip_crimes['area'] = f'ZIP {selected_zip}'
    
    # Combine data
    chart_data = pd.concat([
        yearly_crimes_all[['year', 'crime_rate', 'area']],
        zip_crimes[['year', 'crime_rate', 'area']]
    ]).reset_index(drop=True)
    
    # Filter and clean
    chart_data = chart_data[(chart_data['year'] >= min_year) & (chart_data['year'] <= max_year)].copy()
    chart_data['year'] = chart_data['year'].astype(int)
    chart_data = chart_data.sort_values(['area', 'year']).reset_index(drop=True)
    
    # Create layered chart with different colors
    color_scale = alt.Scale(
        domain=['All Chicago', f'ZIP {selected_zip}'],
        range=['#1f77b4', '#ff6600']
    )
    
    area = alt.Chart(chart_data).mark_area(
        interpolate='step-after',
        opacity=0.3
    ).encode(
        x=alt.X('year:Q',
                title='Year',
                scale=alt.Scale(domain=[min_year, max_year]),
                axis=alt.Axis(format='d', tickCount=10, labelAngle=-45)),
        y=alt.Y('crime_rate:Q',
                title='Crime Rate (per 1,000 residents)',
                scale=alt.Scale(zero=True),
                stack=None),
        color=alt.Color('area:N', scale=color_scale, legend=alt.Legend(title='Area', orient='top'))
    )
    
    line = alt.Chart(chart_data).mark_line(
        interpolate='step-after',
        strokeWidth=2.5
    ).encode(
        x=alt.X('year:Q'),
        y=alt.Y('crime_rate:Q'),
        color=alt.Color('area:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('crime_rate:Q', title='Crime Rate', format='.2f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    points = alt.Chart(chart_data).mark_circle(size=40).encode(
        x=alt.X('year:Q'),
        y=alt.Y('crime_rate:Q'),
        color=alt.Color('area:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('crime_rate:Q', title='Crime Rate', format='.2f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    chart = (area + line + points).properties(
        title=alt.Title(text=f'Crime Rate Comparison: Chicago vs ZIP {selected_zip}', fontSize=16, anchor='start'),
        width='container',
        height=350
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=True,
        gridOpacity=0.3,
        labelFontSize=11,
        titleFontSize=13
    )
    
    return chart


def create_comparison_housing_chart(zhvi_df, selected_zip, min_year=None, max_year=None):
    """
    Create a comparison step chart showing both Chicago average and selected ZIP housing prices.
    """
    zhvi_df = zhvi_df.copy()
    zhvi_df['RegionName'] = zhvi_df['RegionName'].astype(str)
    
    if min_year is None:
        min_year = int(zhvi_df['Year'].min())
    if max_year is None:
        max_year = int(zhvi_df['Year'].max())
    
    # Chicago overall
    yearly_zhvi_all = zhvi_df.groupby('Year')['Zhvi'].mean().reset_index()
    yearly_zhvi_all['area'] = 'All Chicago'
    yearly_zhvi_all = yearly_zhvi_all.rename(columns={'Year': 'year', 'Zhvi': 'zhvi'})
    
    # Selected ZIP
    zip_zhvi = zhvi_df[zhvi_df['RegionName'] == str(selected_zip)].groupby('Year')['Zhvi'].mean().reset_index()
    zip_zhvi['area'] = f'ZIP {selected_zip}'
    zip_zhvi = zip_zhvi.rename(columns={'Year': 'year', 'Zhvi': 'zhvi'})
    
    # Combine data
    chart_data = pd.concat([yearly_zhvi_all, zip_zhvi]).reset_index(drop=True)
    
    # Filter and clean
    chart_data = chart_data[(chart_data['year'] >= min_year) & (chart_data['year'] <= max_year)].copy()
    chart_data['year'] = chart_data['year'].astype(int)
    chart_data = chart_data.sort_values(['area', 'year']).reset_index(drop=True)
    
    # Color scale
    color_scale = alt.Scale(
        domain=['All Chicago', f'ZIP {selected_zip}'],
        range=['#2171b5', '#ff6600']
    )
    
    area = alt.Chart(chart_data).mark_area(
        interpolate='step-after',
        opacity=0.3
    ).encode(
        x=alt.X('year:Q',
                title='Year',
                scale=alt.Scale(domain=[min_year, max_year]),
                axis=alt.Axis(format='d', tickCount=10, labelAngle=-45)),
        y=alt.Y('zhvi:Q',
                title='Home Value Index ($)',
                scale=alt.Scale(zero=True),
                axis=alt.Axis(format='$,.0f'),
                stack=None),
        color=alt.Color('area:N', scale=color_scale, legend=alt.Legend(title='Area', orient='top'))
    )
    
    line = alt.Chart(chart_data).mark_line(
        interpolate='step-after',
        strokeWidth=2.5
    ).encode(
        x=alt.X('year:Q'),
        y=alt.Y('zhvi:Q'),
        color=alt.Color('area:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('zhvi:Q', title='Home Value', format='$,.0f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    points = alt.Chart(chart_data).mark_circle(size=40).encode(
        x=alt.X('year:Q'),
        y=alt.Y('zhvi:Q'),
        color=alt.Color('area:N', scale=color_scale, legend=None),
        tooltip=[
            alt.Tooltip('year:Q', title='Year', format='d'),
            alt.Tooltip('zhvi:Q', title='Home Value', format='$,.0f'),
            alt.Tooltip('area:N', title='Area')
        ]
    )
    
    chart = (area + line + points).properties(
        title=alt.Title(text=f'Home Value Comparison: Chicago vs ZIP {selected_zip}', fontSize=16, anchor='start'),
        width='container',
        height=350
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=True,
        gridOpacity=0.3,
        labelFontSize=11,
        titleFontSize=13
    )
    
    return chart

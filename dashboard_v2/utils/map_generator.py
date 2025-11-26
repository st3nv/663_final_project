# utils/map_generator.py - Shared map HTML generation
import json

def generate_map_html(map_data_json, map_type, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year, selected_zip=None):
    """Generate the HTML/JavaScript code for the interactive map with ZIP selection"""
    
    # Pass selected_zip to JavaScript
    selected_zip_js = f'"{selected_zip}"' if selected_zip else 'null'
    
    html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            width: 100%;
        }}
        #mapWrapper {{
            position: relative;
            width: 100%;
            height: 560px;  /* slightly taller map area */
        }}
        #map {{
            width: 100%;
            height: 100%;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            min-width: 250px;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        .control-label {{
            font-weight: bold;
            margin-bottom: 5px;
            display: block;
            font-size: 14px;
        }}
        .year-display {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }}
        .slider {{
            width: 100%;
            margin: 10px 0;
        }}
        .buttons {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        .btn {{
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .btn-play {{
            background: #28a745;
            color: white;
        }}
        .btn-play:hover {{
            background: #218838;
        }}
        .btn-pause {{
            background: #ffc107;
            color: #333;
        }}
        .btn-pause:hover {{
            background: #e0a800;
        }}
        .btn-reset {{
            background: #6c757d;
            color: white;
        }}
        .btn-reset:hover {{
            background: #5a6268;
        }}
        .stats {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 2px solid #eee;
        }}
        .stat-item {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 13px;
        }}
        .stat-label {{
            color: #666;
        }}
        .stat-value {{
            font-weight: bold;
            color: #333;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            right: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .legend-scale {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-gradient {{
            width: 200px;
            height: 20px;
            border-radius: 3px;
        }}
        .zip-label {{
            pointer-events: none !important;
        }}
        .leaflet-interactive {{
            outline: none !important;
        }}
        .leaflet-interactive:focus {{
            outline: none !important;
        }}
        .leaflet-container {{
            outline: none !important;
        }}
        .zip-tooltip {{
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        /* Selected ZIP indicator */
        .selected-zip-indicator {{
            position: absolute;
            top: 10px;
            left: 60px;
            background: #ff6600;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            font-size: 14px;
            font-weight: bold;
        }}
        .selected-zip-indicator .clear-btn {{
            margin-left: 10px;
            background: white;
            color: #ff6600;
            border: none;
            border-radius: 3px;
            padding: 3px 8px;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
        }}
        .selected-zip-indicator .clear-btn:hover {{
            background: #ffe0cc;
        }}
        /* Click instruction */
        .click-instruction {{
            position: absolute;
            bottom: 20px;
            left: 10px;
            background: rgba(255,255,255,0.95);
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 12px;
            color: #666;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            z-index: 1000;
        }}
        /* Trend chart panel for Crime Rate map (placed below the map) */
        .trend-container {{
            margin: 15px 10px 10px;  /* extra vertical gap below the map */
            background: rgba(255,255,255,0.98);
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .trend-title {{
            margin: 0 0 4px 0;
            font-size: 14px;
            font-weight: bold;
            color: #333;
        }}
        .trend-subtitle {{
            margin: 0 0 6px 0;
            font-size: 11px;
            color: #666;
        }}
        .trend-chart-wrapper {{
            position: relative;
            width: 100%;
            height: 220px;  /* extra height to avoid clipping axes */
        }}
    </style>
</head>
<body>
    <div id="mapWrapper">
        <div id="map"></div>
        
        <div class="selected-zip-indicator" id="selectedIndicator" style="display: none;">
            📍 ZIP: <span id="selectedZipDisplay">-</span>
            <button class="clear-btn" id="clearSelectionBtn">✕</button>
        </div>
        
        <div class="controls" id="controls" style="display: {'block' if map_type != 'Population' else 'none'};">
            <div class="year-display" id="yearDisplay">2021</div>
            <div class="control-group">
                <input type="range" class="slider" id="yearSlider" 
                       min="2000" max="2024" value="2021" step="1">
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666;">
                    <span id="minYearLabel">2000</span>
                    <span id="maxYearLabel">2024</span>
                </div>
            </div>
            <div class="buttons">
                <button class="btn btn-play" id="playBtn">▶ Play</button>
                <button class="btn btn-reset" id="resetBtn">↻ Reset</button>
            </div>
            <div class="stats" id="stats"></div>
        </div>
        
        <div class="legend">
            <div class="legend-title" id="legendTitle">Crime Rate (per 1000 residents)</div>
            <div class="legend-scale">
                <span style="font-size: 12px;">Low</span>
                <div class="legend-gradient" id="legendGradient"></div>
                <span style="font-size: 12px;">High</span>
            </div>
        </div>
        
        <div class="click-instruction" id="clickInstruction">
            👆 Click a ZIP code to view its trend below. Click outside to reset.
        </div>
    </div>

    <div class="trend-container" id="trendContainer" style="display: none;">
        <div class="trend-title" id="trendTitle">Trend</div>
        <div class="trend-subtitle" id="trendSubtitle">
            Showing city-wide averages over time.
        </div>
        <div class="trend-chart-wrapper">
            <canvas id="trendChart"></canvas>
        </div>
    </div>

    <script>
        const mapData = {map_data_json};
        const mapType = "{map_type}";
        const crimeMinYear = {crime_min_year};
        const crimeMaxYear = {crime_max_year};
        const zhviMinYear = {zhvi_min_year};
        const zhviMaxYear = {zhvi_max_year};
        
        let currentYear = 2021;
        let minYear, maxYear;
        let isPlaying = false;
        let animationInterval = null;
        let map, geoJsonLayer;
        let labelLayerGroup;
        let selectedZip = {selected_zip_js};
        let layersByZip = {{}};  // Store layer references by ZIP code
        let trendYears = [];
        let trendChart = null;
        
        // Set year range based on map type
        if (mapType === 'Crime Rate') {{
            minYear = crimeMinYear;
            maxYear = crimeMaxYear;
        }} else if (mapType === 'Housing Prices') {{
            minYear = zhviMinYear;
            maxYear = zhviMaxYear;
        }} else {{
            minYear = 2021;
            maxYear = 2021;
        }}
        
        // Update slider range
        document.getElementById('yearSlider').min = minYear;
        document.getElementById('yearSlider').max = maxYear;
        document.getElementById('minYearLabel').textContent = minYear;
        document.getElementById('maxYearLabel').textContent = maxYear;
        
        if (mapType !== 'Population') {{
            currentYear = minYear;
        }}
        
        // Initialize map
        map = L.map('map', {{
            scrollWheelZoom: false,
            doubleClickZoom: true,
            touchZoom: true,
            dragging: true,
            zoomControl: true,
            minZoom: 10,   // prevent zooming out too far
            maxZoom: 18    // reasonable upper zoom for city detail
        }}).setView([41.85, -87.65], 11);
        
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);
        
        labelLayerGroup = L.layerGroup().addTo(map);

        // Fit initial view to cover all ZIP centroids (full Chicago extent)
        if (Array.isArray(mapData) && mapData.length > 0) {{
            const bounds = L.latLngBounds(
                mapData
                    .filter(z => typeof z.centroid_lat === 'number' && typeof z.centroid_lng === 'number')
                    .map(z => [z.centroid_lat, z.centroid_lng])
            );
            if (bounds.isValid()) {{
                map.fitBounds(bounds.pad(0.05));
            }}
        }}
        
        // Color scales
        const crimeColors = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'];
        const popColors = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'];
        const zhviColors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
        
        function getColor(value, maxValue, colors) {{
            if (value === 0 || maxValue === 0) return '#cccccc';
            const ratio = Math.min(value / maxValue, 1);
            const index = Math.min(Math.floor(ratio * colors.length), colors.length - 1);
            return colors[index];
        }}
        
        function getCrimeRate(zip, year) {{
            const crimes = zip.crimes[year.toString()] || 0;
            return zip.population > 0 ? (crimes / zip.population * 1000) : 0;
        }}
        
        function getZhvi(zip, year) {{
            return zip.zhvi[year.toString()] || 0;
        }}
        
        function formatCurrency(value) {{
            return '$' + value.toLocaleString('en-US', {{maximumFractionDigits: 0}});
        }}

        // Initialize the trend chart (Crime Rate and Housing Prices maps)
        function initTrendChart() {{
            if (mapType !== 'Crime Rate' && mapType !== 'Housing Prices') {{
                return;
            }}
            const container = document.getElementById('trendContainer');
            const canvas = document.getElementById('trendChart');
            if (!container || !canvas || typeof Chart === 'undefined') {{
                return;
            }}

            const titleEl = document.getElementById('trendTitle');
            const subtitleEl = document.getElementById('trendSubtitle');

            // Precompute Chicago-wide baseline series (crime rate or home value)
            const years = [];
            const baseline = [];

            if (mapType === 'Crime Rate') {{
                const totalPopulation = mapData.reduce((sum, z) => sum + (z.population || 0), 0);
                for (let year = crimeMinYear; year <= crimeMaxYear; year++) {{
                    years.push(year);
                    const totalCrimesYear = mapData.reduce(
                        (sum, z) => sum + (z.crimes[String(year)] || 0),
                        0
                    );
                    const rate = totalPopulation > 0 ? (totalCrimesYear / totalPopulation) * 1000 : 0;
                    baseline.push(rate);
                }}
                if (titleEl) {{
                    titleEl.textContent = 'Crime Rate Trend';
                }}
                if (subtitleEl) {{
                    subtitleEl.textContent = 'Showing Chicago average crime rate over time (per 1,000 residents).';
                }}
            }} else if (mapType === 'Housing Prices') {{
                for (let year = zhviMinYear; year <= zhviMaxYear; year++) {{
                    years.push(year);
                    const values = mapData
                        .map(z => z.zhvi[String(year)] || 0)
                        .filter(v => v > 0);
                    const avgZhvi = values.length > 0
                        ? values.reduce((sum, v) => sum + v, 0) / values.length
                        : 0;
                    baseline.push(avgZhvi);
                }}
                if (titleEl) {{
                    titleEl.textContent = 'Home Value Trend';
                }}
                if (subtitleEl) {{
                    subtitleEl.textContent = 'Showing Chicago average home values over time.';
                }}
            }}

            trendYears = years;

            const ctx = canvas.getContext('2d');
            const isCrime = (mapType === 'Crime Rate');
            const yTitle = isCrime
                ? 'Crime rate (per 1,000 residents)'
                : 'Home value index ($)';
            const baseColor = isCrime ? '#1f77b4' : '#2171b5';
            const baseBgColor = isCrime
                ? 'rgba(31,119,180,0.3)'
                : 'rgba(33,113,181,0.3)';

            trendChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: years,
                    datasets: [
                        {{
                            label: 'All Chicago',
                            data: baseline,
                            borderColor: baseColor,
                            backgroundColor: baseBgColor,
                            borderWidth: 2,
                            fill: true,
                            stepped: true,
                            pointRadius: 3
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }},
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Year'
                            }},
                            ticks: {{
                                callback: function(value, index) {{
                                    return years[index];
                                }}
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: yTitle
                            }},
                            beginAtZero: true
                        }}
                    }}
                }}
            }});

            container.style.display = 'block';
        }}

        // Update the trend chart when a ZIP is selected/cleared
        function updateTrendChartForZip(zipCode) {{
            if (!trendChart || (mapType !== 'Crime Rate' && mapType !== 'Housing Prices')) {{
                return;
            }}

            const subtitle = document.getElementById('trendSubtitle');

            // Keep the first dataset (Chicago average), replace any ZIP comparison dataset
            const baseDataset = trendChart.data.datasets[0];
            const datasets = [baseDataset];

            if (zipCode) {{
                const zipData = mapData.find(z => String(z.zip) === String(zipCode));
                if (zipData) {{
                    let zipSeries = [];
                    if (mapType === 'Crime Rate') {{
                        const zipPopulation = zipData.population || 0;
                        zipSeries = trendYears.map(year => {{
                            const crimes = zipData.crimes[String(year)] || 0;
                            return zipPopulation > 0 ? (crimes / zipPopulation) * 1000 : 0;
                        }});
                    }} else if (mapType === 'Housing Prices') {{
                        zipSeries = trendYears.map(year => zipData.zhvi[String(year)] || 0);
                    }}

                    datasets.push({{
                        label: `ZIP ${{zipCode}}`,
                        data: zipSeries,
                        borderColor: '#ff6600',
                        backgroundColor: 'rgba(255,102,0,0.25)',
                        borderWidth: 2,
                        fill: true,
                        stepped: true,
                        pointRadius: 3
                    }});
                    if (subtitle) {{
                        if (mapType === 'Crime Rate') {{
                            subtitle.textContent = `Comparing ZIP ${{zipCode}} vs Chicago average (per 1,000 residents).`;
                        }} else if (mapType === 'Housing Prices') {{
                            subtitle.textContent = `Comparing ZIP ${{zipCode}} vs Chicago average home values.`;
                        }}
                    }}
                }}
            }} else if (subtitle) {{
                if (mapType === 'Crime Rate') {{
                    subtitle.textContent = 'Showing Chicago average crime rate over time (per 1,000 residents).';
                }} else if (mapType === 'Housing Prices') {{
                    subtitle.textContent = 'Showing Chicago average home values over time.';
                }}
            }}

            trendChart.data.datasets = datasets;
            trendChart.update();
        }}
        
        // Notify Streamlit of ZIP selection using query param message
        function notifyStreamlit(zipCode) {{
            const params = zipCode ? {{ selected_zip: zipCode }} : {{}};
            // Prefer Streamlit postMessage API (works within Streamlit's sandboxed iframe)
            try {{
                window.parent.postMessage({{
                    type: 'streamlit:setQueryParams',
                    queryParams: params
                }}, '*');
                window.parent.postMessage({{
                    type: 'streamlit:rerun'
                }}, '*');
            }} catch (e) {{
                console.warn('postMessage failed; unable to notify Streamlit of ZIP change', e);
            }}
            // Optional fallback: only when this map is NOT inside an iframe.
            // In Streamlit, the component runs in a sandboxed iframe, so we skip this
            // and rely purely on postMessage (avoids SecurityError noise in console).
            setTimeout(() => {{
                if (window.parent === window) {{
                    const base = window.location.href.split('?')[0].split('#')[0];
                    const newUrl = zipCode ? `${{base}}?selected_zip=${{zipCode}}` : base;
                    if (window.location.href !== newUrl) {{
                        window.location.href = newUrl;
                    }}
                }}
            }}, 150);
        }}
        
        // Clear all highlights from all layers
        function clearAllHighlights() {{
            Object.values(layersByZip).forEach(layer => {{
                if (layer) {{
                    layer.setStyle({{
                        weight: 1,
                        color: 'black',
                        fillOpacity: 0.7
                    }});
                }}
            }});
        }}
        
        // Apply highlight to a specific ZIP
        function highlightZip(zipCode) {{
            const layer = layersByZip[zipCode];
            if (layer) {{
                layer.setStyle({{
                    weight: 4,
                    color: '#ff6600',
                    fillOpacity: 0.9
                }});
                layer.bringToFront();
            }}
        }}
        
        // Handle clearing selection
        function clearSelection() {{
            clearAllHighlights();
            selectedZip = null;
            document.getElementById('selectedIndicator').style.display = 'none';
            document.getElementById('clickInstruction').style.display = 'block';
            notifyStreamlit(null);
            updateTrendChartForZip(null);
        }}
        
        // Clear button click
        document.getElementById('clearSelectionBtn').addEventListener('click', function(e) {{
            e.stopPropagation();
            clearSelection();
        }});
        
        // Map click handler - detect clicks on the tile layer (outside polygons)
        map.on('click', function(e) {{
            // This fires for all clicks, but we only want to clear when clicking outside polygons
            // We use a flag that gets set by polygon clicks
            setTimeout(function() {{
                if (!window._polygonClicked) {{
                    clearSelection();
                }}
                window._polygonClicked = false;
            }}, 50);
        }});
        
        function updateMap() {{
            if (geoJsonLayer) {{
                map.removeLayer(geoJsonLayer);
            }}
            labelLayerGroup.clearLayers();
            layersByZip = {{}};  // Reset layer references
            
            let maxValue, getValue, colors, legendTitle;
            
            if (mapType === 'Crime Rate') {{
                maxValue = Math.max(...mapData.map(z => getCrimeRate(z, currentYear)));
                getValue = (z) => getCrimeRate(z, currentYear);
                colors = crimeColors;
                legendTitle = `Crime Rate (per 1000 residents, ${{currentYear}})`;
                document.getElementById('legendGradient').style.background = 
                    'linear-gradient(to right, ' + crimeColors.join(', ') + ')';
            }} else if (mapType === 'Housing Prices') {{
                maxValue = Math.max(...mapData.map(z => getZhvi(z, currentYear)).filter(v => v > 0));
                getValue = (z) => getZhvi(z, currentYear);
                colors = zhviColors;
                legendTitle = `Avg Home Value (${{currentYear}})`;
                document.getElementById('legendGradient').style.background = 
                    'linear-gradient(to right, ' + zhviColors.join(', ') + ')';
            }} else {{
                maxValue = Math.max(...mapData.map(z => z.population));
                getValue = (z) => z.population;
                colors = popColors;
                legendTitle = 'Population (2021)';
                document.getElementById('legendGradient').style.background = 
                    'linear-gradient(to right, ' + popColors.join(', ') + ')';
            }}
            
            document.getElementById('legendTitle').textContent = legendTitle;
            
            const features = mapData.map(zip => ({{
                type: 'Feature',
                properties: {{
                    zip: zip.zip,
                    value: getValue(zip),
                    population: zip.population,
                    crimes: zip.crimes[currentYear.toString()] || 0,
                    zhvi: zip.zhvi[currentYear.toString()] || 0
                }},
                geometry: {{
                    type: 'Polygon',
                    coordinates: [zip.coordinates.map(c => [c[1], c[0]])]
                }}
            }}));
            
            geoJsonLayer = L.geoJSON({{
                type: 'FeatureCollection',
                features: features
            }}, {{
                style: function(feature) {{
                    const isSelected = feature.properties.zip === selectedZip;
                    return {{
                        fillColor: getColor(feature.properties.value, maxValue, colors),
                        weight: isSelected ? 4 : 1,
                        opacity: 1,
                        color: isSelected ? '#ff6600' : 'black',
                        fillOpacity: isSelected ? 0.9 : 0.7
                    }};
                }},
                onEachFeature: function(feature, layer) {{
                    const props = feature.properties;
                    const zipCode = String(props.zip);
                    
                    // Store layer reference
                    layersByZip[zipCode] = layer;
                    
                    const crimeRate = getCrimeRate({{
                        population: props.population,
                        crimes: {{ [currentYear]: props.crimes }}
                    }}, currentYear);
                    
                    // Hover effects
                    layer.on('mouseover', function(e) {{
                        if (zipCode !== selectedZip) {{
                            this.setStyle({{
                                weight: 3,
                                color: '#333',
                                fillOpacity: 0.85
                            }});
                        }}
                        this.bringToFront();
                        
                        // Re-highlight selected if exists
                        if (selectedZip && selectedZip !== zipCode) {{
                            highlightZip(selectedZip);
                        }}
                    }});
                    
                    layer.on('mouseout', function(e) {{
                        if (zipCode !== selectedZip) {{
                            this.setStyle({{
                                weight: 1,
                                color: 'black',
                                fillOpacity: 0.7
                            }});
                        }}
                    }});
                    
                    // Click to select
                    layer.on('click', function(e) {{
                        // Set flag to prevent map click from clearing
                        window._polygonClicked = true;
                        
                        // Clear ALL previous highlights first
                        clearAllHighlights();
                        
                        // Set new selection
                        selectedZip = zipCode;
                        
                        // Highlight the clicked ZIP
                        highlightZip(zipCode);
                        
                        // Update UI
                        document.getElementById('selectedZipDisplay').textContent = zipCode;
                        document.getElementById('selectedIndicator').style.display = 'block';
                        document.getElementById('clickInstruction').style.display = 'none';
                        
                        // Notify Streamlit and update the embedded trend chart
                        notifyStreamlit(zipCode);
                        updateTrendChartForZip(zipCode);
                        
                        L.DomEvent.stopPropagation(e);
                    }});
                    
                    // Tooltip
                    let tooltipContent = `<b>ZIP: ${{props.zip}}</b><br>`;
                    if (mapType === 'Crime Rate') {{
                        tooltipContent += `Crime Rate: ${{crimeRate.toFixed(2)}} per 1000<br>`;
                        tooltipContent += `Total Crimes (${{currentYear}}): ${{props.crimes.toLocaleString()}}`;
                    }} else if (mapType === 'Housing Prices') {{
                        tooltipContent += `Avg Home Value (${{currentYear}}): ${{formatCurrency(props.zhvi)}}`;
                    }} else {{
                        tooltipContent += `Population: ${{props.population.toLocaleString()}}`;
                    }}
                    
                    layer.bindTooltip(tooltipContent, {{
                        permanent: false,
                        direction: 'top',
                        sticky: true,
                        opacity: 0.95,
                        className: 'zip-tooltip'
                    }});
                    
                    // ZIP label
                    const center = layer.getBounds().getCenter();
                    const labelMarker = L.marker(center, {{
                        icon: L.divIcon({{
                            className: 'zip-label',
                            html: `<div style="font-size: 9px; font-weight: bold; color: black; text-align: center; text-shadow: 1px 1px 2px white, -1px -1px 2px white;">${{props.zip}}</div>`,
                            iconSize: [40, 20],
                            iconAnchor: [20, 10]
                        }}),
                        interactive: false,
                        keyboard: false
                    }});
                    labelLayerGroup.addLayer(labelMarker);
                }}
            }}).addTo(map);
            
            // If there's a selected ZIP, make sure it's highlighted
            if (selectedZip) {{
                highlightZip(selectedZip);
                document.getElementById('selectedZipDisplay').textContent = selectedZip;
                document.getElementById('selectedIndicator').style.display = 'block';
                document.getElementById('clickInstruction').style.display = 'none';
            }}
            
            updateStats();
        }}
        
        function updateStats() {{
            let statsHTML = '';
            
            if (mapType === 'Crime Rate') {{
                const totalCrimes = mapData.reduce((sum, z) => sum + (z.crimes[currentYear.toString()] || 0), 0);
                const avgCrimeRate = mapData.reduce((sum, z) => sum + getCrimeRate(z, currentYear), 0) / mapData.length;
                const maxCrimeRate = Math.max(...mapData.map(z => getCrimeRate(z, currentYear)));
                
                statsHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Total Crimes:</span>
                        <span class="stat-value">${{totalCrimes.toLocaleString()}}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Crime Rate:</span>
                        <span class="stat-value">${{avgCrimeRate.toFixed(2)}}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Max Crime Rate:</span>
                        <span class="stat-value">${{maxCrimeRate.toFixed(2)}}</span>
                    </div>
                `;
            }} else if (mapType === 'Housing Prices') {{
                const validZhvi = mapData.map(z => getZhvi(z, currentYear)).filter(v => v > 0);
                const avgZhvi = validZhvi.reduce((sum, v) => sum + v, 0) / validZhvi.length;
                const maxZhvi = Math.max(...validZhvi);
                const minZhvi = Math.min(...validZhvi);
                
                statsHTML = `
                    <div class="stat-item">
                        <span class="stat-label">Avg Home Value:</span>
                        <span class="stat-value">${{formatCurrency(avgZhvi)}}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Max Home Value:</span>
                        <span class="stat-value">${{formatCurrency(maxZhvi)}}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Min Home Value:</span>
                        <span class="stat-value">${{formatCurrency(minZhvi)}}</span>
                    </div>
                `;
            }}
            
            document.getElementById('stats').innerHTML = statsHTML;
        }}
        
        function updateYear(year) {{
            currentYear = year;
            document.getElementById('yearDisplay').textContent = year;
            document.getElementById('yearSlider').value = year;
            updateMap();
        }}
        
        function play() {{
            if (isPlaying) return;
            isPlaying = true;
            document.getElementById('playBtn').textContent = '⏸ Pause';
            document.getElementById('playBtn').className = 'btn btn-pause';
            
            animationInterval = setInterval(() => {{
                if (currentYear >= maxYear) {{
                    stop();
                    return;
                }}
                updateYear(currentYear + 1);
            }}, 500);
        }}
        
        function stop() {{
            isPlaying = false;
            document.getElementById('playBtn').textContent = '▶ Play';
            document.getElementById('playBtn').className = 'btn btn-play';
            if (animationInterval) {{
                clearInterval(animationInterval);
                animationInterval = null;
            }}
        }}
        
        function reset() {{
            stop();
            updateYear(minYear);
        }}
        
        document.getElementById('yearSlider').addEventListener('input', (e) => {{
            stop();
            updateYear(parseInt(e.target.value));
        }});
        
        document.getElementById('playBtn').addEventListener('click', () => {{
            if (isPlaying) {{
                stop();
            }} else {{
                play();
            }}
        }});
        
        document.getElementById('resetBtn').addEventListener('click', reset);
        
        setTimeout(function() {{
            map.invalidateSize();
        }}, 100);
        
        // Initialize the trend chart for Crime Rate map (shows Chicago average by default)
        initTrendChart();
        if (selectedZip) {{
            updateTrendChartForZip(selectedZip);
        }}

        updateYear(currentYear);
    </script>
</body>
</html>
"""
    return html_code

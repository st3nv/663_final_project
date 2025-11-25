# utils/map_generator.py - Shared map HTML generation
import json

def generate_map_html(map_data_json, map_type, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year):
    """Generate the HTML/JavaScript code for the interactive map"""
    
    html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
        #map {{ width: 100%; height: 700px; }}
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
            bottom: 120px;
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
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            width: 200px;
            font-size: 12px;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    
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
        
        // Set initial year to minYear for time-based visualizations
        if (mapType !== 'Population') {{
            currentYear = minYear;
        }}
        
        // Initialize map with scroll zoom disabled
        map = L.map('map', {{
            scrollWheelZoom: false,
            doubleClickZoom: true,
            touchZoom: true,
            dragging: true
        }}).setView([41.85, -87.65], 11);
        
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);
        
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
        
        function updateMap() {{
            if (geoJsonLayer) {{
                map.removeLayer(geoJsonLayer);
            }}
            
            // Remove existing labels
            map.eachLayer(function(layer) {{
                if (layer instanceof L.Marker) {{
                    map.removeLayer(layer);
                }}
            }});
            
            let maxValue, getValue, colors, metric, legendTitle;
            
            if (mapType === 'Crime Rate') {{
                maxValue = Math.max(...mapData.map(z => getCrimeRate(z, currentYear)));
                getValue = (z) => getCrimeRate(z, currentYear);
                colors = crimeColors;
                metric = 'crime rate';
                legendTitle = `Crime Rate (per 1000 residents, ${{currentYear}})`;
                document.getElementById('legendGradient').style.background = 
                    'linear-gradient(to right, ' + crimeColors.join(', ') + ')';
            }} else if (mapType === 'Housing Prices') {{
                maxValue = Math.max(...mapData.map(z => getZhvi(z, currentYear)).filter(v => v > 0));
                getValue = (z) => getZhvi(z, currentYear);
                colors = zhviColors;
                metric = 'housing price';
                legendTitle = `Avg Home Value (${{currentYear}})`;
                document.getElementById('legendGradient').style.background = 
                    'linear-gradient(to right, ' + zhviColors.join(', ') + ')';
            }} else {{
                maxValue = Math.max(...mapData.map(z => z.population));
                getValue = (z) => z.population;
                colors = popColors;
                metric = 'population';
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
                    return {{
                        fillColor: getColor(feature.properties.value, maxValue, colors),
                        weight: 1,
                        opacity: 1,
                        color: 'black',
                        fillOpacity: 0.7
                    }};
                }},
                onEachFeature: function(feature, layer) {{
                    const props = feature.properties;
                    const crimeRate = getCrimeRate({{
                        population: props.population,
                        crimes: {{ [currentYear]: props.crimes }}
                    }}, currentYear);
                    
                    let popupContent = `<b>ZIP: ${{props.zip}}</b><br>`;
                    if (mapType === 'Crime Rate') {{
                        popupContent += `Crime Rate: ${{crimeRate.toFixed(2)}} per 1000<br>`;
                        popupContent += `Total Crimes (${{currentYear}}): ${{props.crimes.toLocaleString()}}<br>`;
                        popupContent += `Population (2021): ${{props.population.toLocaleString()}}`;
                    }} else if (mapType === 'Housing Prices') {{
                        popupContent += `Avg Home Value (${{currentYear}}): ${{formatCurrency(props.zhvi)}}<br>`;
                        popupContent += `Population (2021): ${{props.population.toLocaleString()}}<br>`;
                        popupContent += `Total Crimes (${{currentYear}}): ${{props.crimes.toLocaleString()}}`;
                    }} else {{
                        popupContent += `Population: ${{props.population.toLocaleString()}}<br>`;
                        popupContent += `Total Crimes (${{currentYear}}): ${{props.crimes.toLocaleString()}}`;
                    }}
                    
                    layer.bindPopup(popupContent);
                    
                    // Add label
                    const center = layer.getBounds().getCenter();
                    L.marker(center, {{
                        icon: L.divIcon({{
                            className: 'zip-label',
                            html: `<div style="font-size: 9px; font-weight: bold; color: black; text-align: center; text-shadow: 1px 1px 2px white, -1px -1px 2px white;">${{props.zip}}</div>`,
                            iconSize: [40, 20]
                        }})
                    }}).addTo(map);
                }}
            }}).addTo(map);
            
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
        
        // Event listeners
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
        
        // Initialize
        updateYear(currentYear);
    </script>
</body>
</html>
"""
    return html_code

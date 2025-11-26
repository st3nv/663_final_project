# pages/6_💡_Other_Cool_Visualizations.py
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import Point


st.set_page_config(
    page_title="Other Cool Visualizations",
    page_icon="🌆",
    layout="wide",
)


def normalize_series(s: pd.Series, reverse: bool = False) -> pd.Series:
    """Normalize a series to 0–1 for scoring."""
    s = s.astype(float)
    if s.empty:
        return s
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    scaled = (s - mn) / (mx - mn)
    return 1.0 - scaled if reverse else scaled


@st.cache_data
def load_data():
    """Load and prepare core ZIP-level data used by all three visualizations."""
    # Geo with ZipName when available
    geojson_path = os.path.join("data", "chicago_zipcodes.geojson")
    gdf = gpd.read_file(geojson_path)
    gdf["ZIP"] = gdf["ZIP"].astype(str)

    # Population (2021)
    pop_df = pd.read_csv(os.path.join("data", "Chicago_Population_Counts.csv"))
    pop_2021 = pop_df[pop_df["Year"] == 2021].copy()
    pop_2021["ZIP"] = pop_2021["Geography"].astype(str)
    gdf = gdf.merge(
        pop_2021[["ZIP", "Population - Total"]],
        on="ZIP",
        how="left",
    )
    gdf["Population - Total"] = gdf["Population - Total"].fillna(0)

    # Crime
    crime_df = pd.read_csv(os.path.join("data", "chicago_crime_preprocessed.csv"))
    crime_df["ZIP"] = crime_df["ZIP"].astype(str)
    crime_min_year = int(crime_df["year"].min())
    crime_max_year = int(crime_df["year"].max())

    crime_by_year = {}
    for year in range(crime_min_year, crime_max_year + 1):
        crime_year = (
            crime_df[crime_df["year"] == year]
            .groupby("ZIP")["crime_count"]
            .sum()
            .reset_index()
        )
        crime_year.columns = ["ZIP", "total_crimes"]
        crime_by_year[year] = crime_year.set_index("ZIP")["total_crimes"].to_dict()

    # Housing (ZHVI)
    zhvi_df = pd.read_csv(os.path.join("data", "chicago_zhvi_preprocessed.csv"))
    zhvi_df["RegionName"] = zhvi_df["RegionName"].astype(str)
    zhvi_min_year = int(zhvi_df["Year"].min())
    zhvi_max_year = int(zhvi_df["Year"].max())

    zhvi_by_year = {}
    for year in range(zhvi_min_year, zhvi_max_year + 1):
        zhvi_year = (
            zhvi_df[zhvi_df["Year"] == year]
            .groupby("RegionName")["Zhvi"]
            .mean()
            .reset_index()
        )
        zhvi_year.columns = ["ZIP", "avg_zhvi"]
        zhvi_by_year[year] = zhvi_year.set_index("ZIP")["avg_zhvi"].to_dict()

    # Centroids for distance and simple spatial reasoning
    gdf["centroid"] = gdf.geometry.centroid

    # ZipName lookup
    zip_name_map = {}
    if "ZipName" in gdf.columns:
        zip_name_map = (
            gdf[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )

    # map_data: yearly crimes/ZHVI plus centroids (reused by multiple views)
    map_data = []
    for _, row in gdf.iterrows():
        zip_code = row["ZIP"]
        population = float(row["Population - Total"])

        crimes_all_years = {}
        for year in range(crime_min_year, crime_max_year + 1):
            crimes = crime_by_year[year].get(zip_code, 0)
            crimes_all_years[str(year)] = float(crimes)

        zhvi_all_years = {}
        for year in range(zhvi_min_year, zhvi_max_year + 1):
            zhvi = zhvi_by_year[year].get(zip_code, 0)
            zhvi_all_years[str(year)] = float(zhvi)

        centroid = row["centroid"]

        map_data.append(
            {
                "zip": zip_code,
                "population": population,
                "crimes": crimes_all_years,
                "zhvi": zhvi_all_years,
                "centroid_lat": centroid.y,
                "centroid_lng": centroid.x,
            }
        )

    # Latest-year metrics for scoring
    zip_metrics = []
    for z in map_data:
        zip_code = z["zip"]
        pop_val = z["population"]
        crimes_latest = z["crimes"].get(str(crime_max_year), 0.0)
        zhvi_latest = z["zhvi"].get(str(zhvi_max_year), 0.0)
        crime_rate_latest = (crimes_latest / pop_val * 1000) if pop_val > 0 else 0.0
        zip_metrics.append(
            {
                "ZIP": zip_code,
                "ZipName": zip_name_map.get(zip_code, None),
                "population": pop_val,
                "crimes_latest": crimes_latest,
                "crime_rate_latest": crime_rate_latest,
                "zhvi_latest": zhvi_latest,
            }
        )
    zip_df = pd.DataFrame(zip_metrics)

    # Distance from Chicago's Loop (used as the "sun" in Orbit View)
    loop_center = Point(-87.6298, 41.8781)
    gdf["distance_from_loop"] = gdf["centroid"].distance(loop_center)

    return (
        gdf,
        map_data,
        zip_df,
        crime_min_year,
        crime_max_year,
        zhvi_min_year,
        zhvi_max_year,
    )


# ============ Load data & derive scores ============ #
(
    gdf,
    map_data,
    zip_df,
    crime_min_year,
    crime_max_year,
    zhvi_min_year,
    zhvi_max_year,
) = load_data()


# Sidebar settings for overall score weights
st.sidebar.header("📌 Dashboard")

# Initialize default weights in session state
if "w_pop" not in st.session_state:
    st.session_state["w_pop"] = 1.0
if "w_safety" not in st.session_state:
    st.session_state["w_safety"] = 1.0
if "w_value" not in st.session_state:
    st.session_state["w_value"] = 1.0

def _reset_score_weights():
    st.session_state["w_pop"] = 1.0
    st.session_state["w_safety"] = 1.0
    st.session_state["w_value"] = 1.0

with st.sidebar.expander("⚙️ Score Settings", expanded=True):
    st.markdown(
        "Adjust how each component contributes to the **overall score** used in these visualizations."
    )
    w_pop = st.slider(
        "Population weight",
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        key="w_pop",
    )
    w_safety = st.slider(
        "Safety weight",
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        key="w_safety",
    )
    w_value = st.slider(
        "Home value weight",
        min_value=0.0,
        max_value=5.0,
        step=0.1,
        key="w_value",
    )
    st.button("Reset weights", on_click=_reset_score_weights)

    total_w = w_pop + w_safety + w_value
    if total_w <= 0:
        pop_w_norm = saf_w_norm = val_w_norm = 1.0 / 3.0
        st.caption("All weights are zero — falling back to equal weights (⅓, ⅓, ⅓).")
    else:
        pop_w_norm = w_pop / total_w
        saf_w_norm = w_safety / total_w
        val_w_norm = w_value / total_w
        st.caption(
            f"Effective mix → Population **{pop_w_norm:.0%}**, "
            f"Safety **{saf_w_norm:.0%}**, Home value **{val_w_norm:.0%}**."
        )



if not zip_df.empty:
    zip_df["pop_score"] = normalize_series(zip_df["population"], reverse=False)
    # Spread safety more evenly: use percentile rank of crime rate (lower crime → higher safety_score)
    safety_rank = zip_df["crime_rate_latest"].rank(pct=True, method="average")
    zip_df["safety_score"] = 1.0 - safety_rank
    zip_df["value_score"] = normalize_series(zip_df["zhvi_latest"], reverse=True)
    zip_df["overall_score"] = (
        pop_w_norm * zip_df["pop_score"]
        + saf_w_norm * zip_df["safety_score"]
        + val_w_norm * zip_df["value_score"]
    )

    if "ZipName" in zip_df.columns:
        zip_df["ZIP_label"] = zip_df.apply(
            lambda r: f"{r['ZIP']} – {r['ZipName']}"
            if isinstance(r["ZipName"], str) and r["ZipName"]
            else r["ZIP"],
            axis=1,
        )
    else:
        zip_df["ZIP_label"] = zip_df["ZIP"]
else:
    zip_df["ZIP_label"] = zip_df["ZIP"]

# Adjacency list for network view (who touches whom)
neighbors = []
gdf_buf = gdf[["ZIP", "geometry"]].copy()
gdf_buf["geometry"] = gdf_buf["geometry"].buffer(0)
for i in range(len(gdf_buf)):
    zi = gdf_buf.iloc[i]
    for j in range(i + 1, len(gdf_buf)):
        zj = gdf_buf.iloc[j]
        if zi["geometry"].touches(zj["geometry"]):
            neighbors.append({"source": zi["ZIP"], "target": zj["ZIP"]})

nodes_df = zip_df[["ZIP", "overall_score", "safety_score"]].copy()
nodes_json = json.dumps(nodes_df.to_dict(orient="records"))
links_json = json.dumps(neighbors)


# ============ Views ============ #

def render_bubble_playground():
    st.markdown("### 🔵 Bubble Playground")
    st.caption(
        "Each ZIP is a floating circle: radius encodes population, home value, or crime rate, "
        "and color encodes safety/overall/population/home-value score. Hover to inspect details."
    )

    if zip_df.empty:
        st.info("No ZIP-level data available for the bubble playground.")
        return

    inner_cols = st.columns(2)
    color_metric = inner_cols[0].pills(
        "Color metric",
        options=["Safety Score", "Overall Score", "Population Score", "Home Value Score"],
        default="Home Value Score",
        key="bubble_color_metric",
    )
    size_metric = inner_cols[1].pills(
        "Bubble size metric",
        options=["Population", "Home Value (latest)", "Crime Rate"],
        default="Home Value (latest)",
        key="bubble_size_metric",
    )

    bubbles = []
    for _, row in zip_df.iterrows():
        bubbles.append(
            {
                "zip": row["ZIP"],
                "label": row["ZIP_label"],
                "population": float(row["population"]),
                "zhvi_latest": float(row["zhvi_latest"]),
                "crime_rate_latest": float(row["crime_rate_latest"]),
                "safety_score": float(row["safety_score"]),
                "overall_score": float(row["overall_score"]),
                "pop_score": float(row["pop_score"]),
                "value_score": float(row["value_score"]),
            }
        )
    bubbles_json = json.dumps(bubbles)

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Bubble Playground</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/matter-js/0.19.0/matter.min.js"></script>
    <style>
        :root {
            color-scheme: light;
        }
        body {
            margin: 0;
            padding: 0;
            background: #ffffff;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        #worldContainer {
            width: 100%;
            height: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            position: relative;
        }
        #worldCanvas {
            border-radius: 0px;
            overflow: hidden;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.15);
            background: #e5f0ff;
        }
        #tooltip {
            position: absolute;
            pointer-events: none;
            background: rgba(15,23,42,0.96);
            color: #e5e7eb;
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 11px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            opacity: 0;
            white-space: nowrap;
        }
        #legend {
            margin-top: 8px;
            background: rgba(255,255,255,0.95);
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 8px 12px;
            font-size: 11px;
            color: #0f172a;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
        }
        #legend .bar {
            width: 160px;
            height: 10px;
            border-radius: 6px;
            background: linear-gradient(90deg, #e7f1ff 0%, #0f316a 100%);
            margin-top: 6px;
        }
        #legend .labels {
            display: flex;
            justify-content: space-between;
            margin-top: 4px;
            color: #475569;
        }
    </style>
</head>
<body>
<div id="worldContainer">
    <canvas id="worldCanvas" width="960" height="480"></canvas>
    <div id="tooltip"></div>
    <div id="legend">
        <div id="legendTitle"></div>
        <div class="bar"></div>
        <div class="labels"><span>Lower</span><span>Higher</span></div>
    </div>
</div>
<script>
    const bubbles = __BUBBLES__;
    const sizeMetric = "__SIZE_METRIC__";
    const colorMetric = "__COLOR_METRIC__";

    const Engine = Matter.Engine,
          Render = Matter.Render,
          Runner = Matter.Runner,
          World = Matter.World,
          Bodies = Matter.Bodies,
          Body = Matter.Body,
          Mouse = Matter.Mouse,
          MouseConstraint = Matter.MouseConstraint;

    const engine = Engine.create();
    const world = engine.world;
    world.gravity.y = 0;

    const canvas = document.getElementById('worldCanvas');
    const container = document.getElementById('worldContainer');
    const render = Render.create({
        canvas: canvas,
        engine: engine,
        options: {
            width: canvas.width,
            height: canvas.height,
            wireframes: false,
            background: 'transparent'
        }
    });

    const width = canvas.width;
    const height = canvas.height;

    const walls = [
        Bodies.rectangle(width/2, -40, width, 80, { isStatic: true }),
        Bodies.rectangle(width/2, height+40, width, 80, { isStatic: true }),
        Bodies.rectangle(-40, height/2, 80, height, { isStatic: true }),
        Bodies.rectangle(width+40, height/2, 80, height, { isStatic: true })
    ];
    World.add(world, walls);

    function scaleValue(value, min, max, outMin, outMax) {
        if (max === min) return (outMin + outMax) / 2;
        const r = (value - min) / (max - min);
        return outMin + r * (outMax - outMin);
    }

    function pickSizeValue(b) {
        if (sizeMetric === "Population") return b.population;
        if (sizeMetric === "Home Value (latest)") return b.zhvi_latest;
        if (sizeMetric === "Crime Rate") return b.crime_rate_latest;
        return b.population;
    }

    function pickColorValue(b) {
        if (colorMetric === "Safety Score") return b.safety_score;
        if (colorMetric === "Overall Score") return b.overall_score;
        if (colorMetric === "Population Score") return b.pop_score;
        if (colorMetric === "Home Value Score") return b.value_score;
        return b.safety_score;
    }

    const sizeVals = bubbles.map(b => pickSizeValue(b));
    const sizeMin = Math.min(...sizeVals);
    const sizeMax = Math.max(...sizeVals);

    const colorVals = bubbles.map(b => pickColorValue(b));
    const colorMin = Math.min(...colorVals);
    const colorMax = Math.max(...colorVals);

    function colorForScore(s) {
        const t = (s - colorMin) / (colorMax - colorMin || 1);
        // Interpolate light-to-dark blue
        const c1 = [231, 241, 255]; // light
        const c2 = [15, 49, 106];   // dark
        const r = Math.round(c1[0] + t * (c2[0] - c1[0]));
        const g = Math.round(c1[1] + t * (c2[1] - c1[1]));
        const b = Math.round(c1[2] + t * (c2[2] - c1[2]));
        return `rgb(${r},${g},${b})`;
    }

    const bodies = [];

    bubbles.forEach((b, i) => {
        const metricValue = pickSizeValue(b);
        const radius = scaleValue(metricValue, sizeMin, sizeMax, 10, 32);

        const angle = (2 * Math.PI / bubbles.length) * i;
        const orbitRadius = Math.min(width, height) / 3;
        const x = width/2 + orbitRadius * Math.cos(angle);
        const y = height/2 + orbitRadius * Math.sin(angle);

        const body = Bodies.circle(x, y, radius, {
            restitution: 0.9,
            frictionAir: 0.03,
            friction: 0.01,
            render: {
                fillStyle: colorForScore(
                    pickColorValue(b)
                ),
                strokeStyle: "rgba(15,23,42,0.35)",
                lineWidth: 1.0
            }
        });
        body.customData = b;
        bodies.push(body);
    });

    World.add(world, bodies);

    bodies.forEach((body, i) => {
        const angle = (2 * Math.PI / bodies.length) * i;
        const speed = 0.6;
        Body.setVelocity(body, {
            x: speed * Math.cos(angle + Math.PI/4),
            y: speed * Math.sin(angle + Math.PI/4)
        });
    });

    const mouse = Mouse.create(canvas);
    const mouseConstraint = MouseConstraint.create(engine, {
        mouse: mouse,
        constraint: {
            stiffness: 0.16,
            render: { visible: false }
        }
    });
    World.add(world, mouseConstraint);

    const tooltip = document.getElementById('tooltip');

    Matter.Events.on(mouseConstraint, 'mousemove', function(event) {
        const mousePos = event.mouse.position;
        let found = null;
        for (const b of bodies) {
            const dx = b.position.x - mousePos.x;
            const dy = b.position.y - mousePos.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist <= b.circleRadius) {
                found = b;
                break;
            }
        }
        if (found && found.customData) {
            const d = found.customData;
            tooltip.innerHTML =
                d.label +
                '<br/>Population ' + d.population.toLocaleString() +
                '<br/>Home value ' + (d.zhvi_latest > 0 ? '$' + d.zhvi_latest.toLocaleString() : 'N/A') +
                '<br/>Safety ' + d.safety_score.toFixed(2) +
                ' · Overall ' + d.overall_score.toFixed(2) +
                '<br/>Pop score ' + d.pop_score.toFixed(2) +
                ' · Home value score ' + d.value_score.toFixed(2);

            const containerRect = container.getBoundingClientRect();
            const canvasRect = canvas.getBoundingClientRect();
            const offsetX = canvasRect.left - containerRect.left;
            const offsetY = canvasRect.top - containerRect.top;

            let x = offsetX + mousePos.x;
            let y = offsetY + mousePos.y;

            tooltip.style.opacity = 1;
            tooltip.style.left = x + 'px';
            tooltip.style.top = y + 'px';

            const tooltipRect = tooltip.getBoundingClientRect();
            const padding = 8;
            let left = tooltipRect.left - containerRect.left;
            let top = tooltipRect.top - containerRect.top;
            let right = left + tooltipRect.width;
            let bottom = top + tooltipRect.height;

            if (left < padding) {
                x += padding - left;
            } else if (right > containerRect.width - padding) {
                x -= right - (containerRect.width - padding);
            }
            if (top < padding) {
                y += padding - top;
            } else if (bottom > containerRect.height - padding) {
                y -= bottom - (containerRect.height - padding);
            }

            tooltip.style.left = x + 'px';
            tooltip.style.top = y + 'px';
        } else {
            tooltip.style.opacity = 0;
        }
    });

    Render.run(render);
    const runner = Runner.create();
    Runner.run(runner, engine);

    // Legend title based on color metric
    document.getElementById('legendTitle').textContent = 'Color = ' + colorMetric;
</script>
</body>
</html>
"""
    html = (
        html_template.replace("__BUBBLES__", bubbles_json)
        .replace("__SIZE_METRIC__", size_metric)
        .replace("__COLOR_METRIC__", color_metric)
    )
    components.html(html, height=620, scrolling=False)


def render_orbit_view():
    st.markdown("### 🪐 Orbit View")
    st.caption(
        "Chicago's Loop is the sun. Each ZIP is a planet: distance from the Loop sets the orbit radius, "
        "point size reflects a chosen metric, color indicates a chosen score, and hover shows ZIP + name."
    )

    inner_cols = st.columns(2)
    color_choice = inner_cols[0].pills(
        "Color metric",
        options=["Safety Score", "Overall Score", "Population Score", "Home Value Score"],
        default="Safety Score",
        key="orbit_color_metric",
    )
    size_choice = inner_cols[1].pills(
        "Bubble size metric",
        options=["Population", "Home Value (latest)", "Crime Rate"],
        default="Home Value (latest)",
        key="orbit_size_metric",
    )

    color_field = {
        "Safety Score": "safety_score",
        "Overall Score": "overall_score",
        "Population Score": "pop_score",
        "Home Value Score": "value_score",
    }[color_choice]

    orbit_df = gdf.merge(
        zip_df[
            [
                "ZIP",
                "ZipName",
                "population",
                "crime_rate_latest",
                "safety_score",
                "overall_score",
                "pop_score",
                "value_score",
                "zhvi_latest",
            ]
        ],
        on="ZIP",
        how="left",
    )
    # Resolve ZipName collisions from merge (gdf may also have ZipName)
    if "ZipName" not in orbit_df.columns and "ZipName_y" in orbit_df.columns:
        orbit_df["ZipName"] = orbit_df["ZipName_y"].combine_first(
            orbit_df.get("ZipName_x")
        )
    elif "ZipName" in orbit_df.columns and "ZipName_y" in orbit_df.columns:
        orbit_df["ZipName"] = orbit_df["ZipName"].combine_first(orbit_df["ZipName_y"])
    elif "ZipName_y" in orbit_df.columns:
        orbit_df["ZipName"] = orbit_df["ZipName_y"]
    elif "ZipName_x" in orbit_df.columns:
        orbit_df["ZipName"] = orbit_df["ZipName_x"]

    # Clean up duplicate ZipName columns if present
    for col in ["ZipName_x", "ZipName_y"]:
        if col in orbit_df.columns:
            orbit_df.drop(columns=[col], inplace=True, errors="ignore")

    orbit_df = orbit_df.dropna(subset=["safety_score", "zhvi_latest"])

    if orbit_df.empty:
        st.info("No data available for the orbit view.")
        return

    orbit_df = orbit_df.sort_values("ZIP").reset_index(drop=True)
    orbit_df["angle_index"] = np.arange(len(orbit_df))
    orbit_df["theta"] = orbit_df["angle_index"] / len(orbit_df) * 360.0
    orbit_df["label"] = orbit_df.apply(
        lambda r: f"{r['ZIP']} – {r['ZipName']}" if pd.notna(r.get("ZipName")) and r.get("ZipName") else r["ZIP"],
        axis=1,
    )

    size_field = {
        "Population": "population",
        "Home Value (latest)": "zhvi_latest",
        "Crime Rate": "crime_rate_latest",
    }[size_choice]

    fig = px.scatter_polar(
        orbit_df,
        r="distance_from_loop",
        theta="theta",
        size=size_field,
        color=color_field,
        hover_name="label",
        hover_data={
            "ZIP": True,
            "ZipName": True,
            "population": ":,.0f",
            "zhvi_latest": ":,.0f",
            "crime_rate_latest": ":.2f",
            "safety_score": ":.2f",
            "overall_score": ":.2f",
            "pop_score": ":.2f",
            "value_score": ":.2f",
            "distance_from_loop": ":.3f",
        },
        color_continuous_scale="Blues",
        size_max=26,
        title="ZIPs Orbiting the Loop",
    )
    fig.update_layout(
        template="simple_white",
        height=640,
        margin=dict(l=40, r=40, t=80, b=40),
        polar=dict(
            radialaxis=dict(
                title="Distance from Loop",
                gridcolor="rgba(148,163,184,0.25)",
                linecolor="rgba(148,163,184,0.5)",
            ),
            angularaxis=dict(
                showticklabels=False,
                gridcolor="rgba(148,163,184,0.12)",
                tickcolor="rgba(0,0,0,0)",
            ),
        ),
        coloraxis_colorbar=dict(
            title=color_choice,
            ticks="outside",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_adjacency_network():
    st.markdown("### 🕸️ Adjacency Network")
    st.caption(
        "Each node is a ZIP; neighboring ZIPs are linked. Node size shows the overall score, "
        "and color reflects safety."
    )

    if nodes_df.empty or not neighbors:
        st.info("No data available for the adjacency network.")
        return

    # Ensure lists (not numpy types) for JSON serialization
    nodes_payload = json.dumps(nodes_df.to_dict(orient="records"))
    links_payload = json.dumps(neighbors)

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>ZIP Adjacency Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            color-scheme: light;
        }
        body {
            margin: 0;
            padding: 0;
            background: #f5f5f7;
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        #network {
            width: 100%;
            height: 620px;
        }
        .node text {
            pointer-events: none;
            font-size: 10px;
            fill: #0f172a;
        }
        .tooltip {
            position: absolute;
            pointer-events: none;
            background: rgba(15,23,42,0.96);
            color: #e5e7eb;
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 11px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            opacity: 0;
            transform: translate(-50%, -140%);
        }
    </style>
</head>
<body>
<div id="network"></div>
<div id="tooltip" class="tooltip"></div>
<script>
    const nodesData = __NODES_DATA__;
    const linksData = __LINKS_DATA__;

    const width = document.getElementById('network').clientWidth;
    const height = 620;

    const svg = d3.select('#network')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const tooltip = d3.select('#tooltip');

    const sizeVals = nodesData.map(d => d.overall_score);
    const sizeMin = d3.min(sizeVals);
    const sizeMax = d3.max(sizeVals);

    const sizeScale = d3.scaleLinear()
        .domain([sizeMin, sizeMax])
        .range([8, 26]);

    const colorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain([0, 1]);

    const nodes = nodesData.map(d => Object.assign({}, d));
    const links = linksData.map(d => Object.assign({}, d));

    const nodeById = new Map(nodes.map(d => [d.ZIP, d]));
    links.forEach(l => {
        l.source = nodeById.get(l.source);
        l.target = nodeById.get(l.target);
    });

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).distance(60).strength(0.7))
        .force('charge', d3.forceManyBody().strength(-80))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => sizeScale(d.overall_score) + 4));

    const link = svg.append('g')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-opacity', 0.7)
        .attr('stroke-width', 1)
        .selectAll('line')
        .data(links)
        .enter()
        .append('line');

    const node = svg.append('g')
        .selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .call(
            d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded)
        );

    node.append('circle')
        .attr('r', d => sizeScale(d.overall_score))
        .attr('fill', d => colorScale(d.safety_score))
        .attr('stroke', '#0f172a')
        .attr('stroke-width', 1.1);

    node.append('text')
        .text(d => d.ZIP)
        .attr('dy', 3)
        .attr('text-anchor', 'middle');

    node.on('mousemove', (event, d) => {
        tooltip.style('opacity', 1)
            .style('left', event.pageX + 'px')
            .style('top', event.pageY + 'px')
            .html(
                'ZIP ' + d.ZIP +
                '<br/>Overall score: ' + d.overall_score.toFixed(3) +
                '<br/>Safety score: ' + d.safety_score.toFixed(3)
            );
    }).on('mouseout', () => {
        tooltip.style('opacity', 0);
    });

    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node.attr('transform', d => `translate(${d.x}, ${d.y})`);
    });

    function dragStarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragEnded(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
</script>
</body>
</html>
"""
    html = (
        html_template.replace("__NODES_DATA__", nodes_payload)
        .replace("__LINKS_DATA__", links_payload)
    )
    components.html(html, height=640, scrolling=False)


# ============ Page layout ============ #

st.title("🌆 Other Cool Visualizations")
st.write(
    "Play with three experimental views of Chicago ZIP codes. "
    "They combine population, home values, crime rates, and neighborhood adjacency into "
    "simple, minimalist visual stories.\n"
    "- **Safety score** = `1 - percentile_rank(crime_rate_latest)` (lower crime → higher safety, better color spread). \n"
    "- **Population score** = 0-1 normalized population (higher population → higher score). \n"
    "- **Home value score** = reversed 0-1 normalized latest average home value (higher value → lower score). \n"
    "- **Overall score** is a weighted mix of population, safety, and home value scores (weights set in the sidebar)."
)

tab_bubble, tab_orbit, tab_network = st.tabs(
    ["Bubble Playground", "Orbit View", "Adjacency Network"]
)

with tab_bubble:
    render_bubble_playground()

with tab_orbit:
    render_orbit_view()

with tab_network:
    render_adjacency_network()

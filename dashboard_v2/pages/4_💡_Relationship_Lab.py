# pages/4_💡_Insight_1.py
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import pandas as pd
import numpy as np
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    load_geodata,
    load_population_data,
    load_crime_data,
    load_housing_data,
    get_exclude_settings,
    apply_zip_exclusion,
    prepare_map_data,
)

st.set_page_config(page_title="Insight 1 · Relationship Lab", page_icon="💡", layout="wide")

st.title("💡 Relationship Lab")
st.markdown(
    "Explore how crime, population, and home values interact across Chicago ZIP codes."
)

# Sidebar settings (same pattern as other v2 pages)
st.sidebar.header("📌 Dashboard")
exclude_zips, exclude_2025 = get_exclude_settings()

# Load and filter data
gdf = load_geodata()
gdf = apply_zip_exclusion(gdf, exclude_zips)
pop_df = load_population_data()
crime_df = load_crime_data()
zhvi_df = load_housing_data()

if exclude_2025:
    if "year" in crime_df.columns:
        crime_df = crime_df[crime_df["year"] != 2025]
    if "Year" in zhvi_df.columns:
        zhvi_df = zhvi_df[zhvi_df["Year"] != 2025]

# Prepare shared map-style data (reuses v2 utility)
# Note: load_population_data() already returns 2021 rows with a 'ZIP' column,
# so we can pass it directly without renaming to avoid duplicate 'ZIP' labels.
map_data, gdf_merged, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year = prepare_map_data(
    gdf, crime_df, zhvi_df, pop_df
)


def build_zip_df(map_data_list, gdf_frame, crime_latest_year, zhvi_latest_year):
    """Construct a ZIP-level dataframe with latest crime rate and home value."""
    zip_name_map = {}
    if "ZipName" in gdf_frame.columns:
        zip_name_map = (
            gdf_frame[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )

    metrics = []
    for z in map_data_list:
        zip_code = str(z["zip"])
        pop_val = float(z["population"])
        crimes_latest = float(z["crimes"].get(str(crime_latest_year), 0.0))
        zhvi_latest = float(z["zhvi"].get(str(zhvi_latest_year), 0.0))
        crime_rate_latest = (crimes_latest / pop_val * 1000.0) if pop_val > 0 else 0.0
        metrics.append(
            {
                "ZIP": zip_code,
                "ZipName": zip_name_map.get(zip_code, None),
                "population": pop_val,
                "crimes_latest": crimes_latest,
                "crime_rate_latest": crime_rate_latest,
                "zhvi_latest": zhvi_latest,
            }
        )

    df = pd.DataFrame(metrics)
    if df.empty:
        return df

    # Build a readable label
    if "ZipName" in df.columns:
        df["Label"] = df.apply(
            lambda r: f"{r['ZIP']} – {r['ZipName']}"
            if isinstance(r["ZipName"], str) and r["ZipName"]
            else r["ZIP"],
            axis=1,
        )
    else:
        df["Label"] = df["ZIP"]

    return df


def build_traj_df(map_data_list, crime_start, crime_end, zhvi_start, zhvi_end):
    """Trajectory dataframe, using percentage change and carrying population for sizing."""
    records = []
    for z in map_data_list:
        zh_earliest = z["zhvi"].get(str(zhvi_start), 0.0)
        zh_latest = z["zhvi"].get(str(zhvi_end), 0.0)
        if zh_earliest <= 0 or zh_latest <= 0:
            continue

        c_earliest = z["crimes"].get(str(crime_start), 0.0)
        c_latest = z["crimes"].get(str(crime_end), 0.0)
        pop = z["population"]
        if pop <= 0:
            continue

        cr_earliest = c_earliest / pop * 1000.0
        cr_latest = c_latest / pop * 1000.0
        # Require a positive baseline to compute percentage change
        if cr_earliest <= 0:
            continue

        records.append(
            {
                "ZIP": str(z["zip"]),
                "population": float(pop),
                "zhvi_earliest": zh_earliest,
                "zhvi_latest": zh_latest,
                "crime_rate_earliest": cr_earliest,
                "crime_rate_latest": cr_latest,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["zhvi_change_pct"] = (
        (df["zhvi_latest"] - df["zhvi_earliest"]) / df["zhvi_earliest"] * 100.0
    )
    df["crime_change_pct"] = (
        (df["crime_rate_latest"] - df["crime_rate_earliest"])
        / df["crime_rate_earliest"]
        * 100.0
    )
    return df


zip_df = build_zip_df(map_data, gdf_merged, crime_max_year, zhvi_max_year)

tabs = st.tabs(
    [
        "Value vs Safety",
        "Value vs Safety Animated",
        "Trajectories",
        "Neighborhood Archetypes",
    ]
)


with tabs[0]:
    st.subheader(f"💙 Most recent Value vs Safety - {crime_max_year}")
    st.markdown(
        "This view uses the most recent year of data to compare each ZIP's home values to its crime rate using a simple regression line. "
        "ZIPs above or below the line appear over- or under-valued given their crime levels."
    )

    df = zip_df.copy()
    df = df[(df["zhvi_latest"] > 0) & (df["crime_rate_latest"] > 0)].copy()
    if df.empty:
        st.warning("No ZIPs with both crime rate and home value.")
    else:
        x = df["crime_rate_latest"].values
        y = df["zhvi_latest"].values

        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.mean((x - x_mean) ** 2)
        b = cov_xy / (var_x + 1e-9)
        a = y_mean - b * x_mean

        df["zhvi_pred"] = a + b * df["crime_rate_latest"]
        df["zhvi_resid"] = df["zhvi_latest"] - df["zhvi_pred"]

        resid_std = df["zhvi_resid"].std() if df["zhvi_resid"].std() > 0 else 1.0
        df["resid_z"] = (df["zhvi_resid"] - df["zhvi_resid"].mean()) / resid_std

        def classify_resid(z):
            if z <= -0.5:
                return "Undervalued (cheap vs crime)"
            if z >= 0.5:
                return "Overvalued (expensive vs crime)"
            return "On-model"

        df["value_class"] = df["resid_z"].apply(classify_resid)

        x_line = np.linspace(df["crime_rate_latest"].min(), df["crime_rate_latest"].max(), 100)
        y_line = a + b * x_line
        line_df = pd.DataFrame(
            {"crime_rate_latest": x_line, "zhvi_pred": y_line}
        )

        scatter = (
            alt.Chart(df)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X(
                    "crime_rate_latest:Q",
                    title="Crime rate (per 1,000 residents)",
                ),
                y=alt.Y(
                    "zhvi_latest:Q",
                    title="Home value (latest ZHVI, USD)",
                ),
                color=alt.Color(
                    "value_class:N",
                    scale=alt.Scale(
                        domain=[
                            "Undervalued (cheap vs crime)",
                            "On-model",
                            "Overvalued (expensive vs crime)",
                        ],
                        range=["#e45756", "#1f4b99", "#8fb9ff"],
                    ),
                    legend=alt.Legend(
                        title="Value category",
                        labelLimit=200,
                    ),
                ),
                size=alt.Size(
                    "population:Q",
                    title="Population",
                    legend=None,
                    scale=alt.Scale(range=[40, 900]),
                ),
                tooltip=[
                    "ZIP",
                    "ZipName",
                    alt.Tooltip("population:Q", title="Population", format=","),
                    alt.Tooltip(
                        "crime_rate_latest:Q",
                        title="Crime rate (per 1,000)",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "zhvi_latest:Q",
                        title="Home value (ZHVI)",
                        format=",.0f",
                    ),
                    "value_class",
                ],
            )
        )

        reg_line = (
            alt.Chart(line_df)
            .mark_line(color="black", strokeDash=[4, 4])
            .encode(
                x="crime_rate_latest:Q",
                y="zhvi_pred:Q",
            )
        )

        chart = (scatter + reg_line).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Points show ZIPs; size encodes population. "
            "Color shows whether home values are under- or over-valued relative to crime levels, "
            "based on the simple regression line."
        )

with tabs[1]:
    st.subheader("💙 Value vs Safety Animated")
    st.markdown(
        "This animated view shows how each ZIP's home values and crime rates move together over time. "
        "Use play, pause, or drag the year slider to see the relationship evolve."
    )

    # Build year-by-year records for animation (using overlapping crime/ZHVI years)
    year_start = max(crime_min_year, zhvi_min_year)
    year_end = min(crime_max_year, zhvi_max_year)

    if year_start > year_end:
        st.warning("No overlapping years available to animate value vs safety.")
    else:
        animated_records = []
        for z in map_data:
            zip_code = str(z["zip"])
            zip_name = z.get("zipName") or ""
            label = f"{zip_code} - {zip_name}" if zip_name else zip_code
            pop_val = float(z.get("population", 0.0) or 0.0)
            if pop_val <= 0:
                continue

            for year in range(year_start, year_end + 1):
                crimes_y = float(z["crimes"].get(str(year), 0.0))
                zhvi_y = float(z["zhvi"].get(str(year), 0.0))
                if crimes_y <= 0 or zhvi_y <= 0:
                    continue

                crime_rate_y = crimes_y / pop_val * 1000.0
                animated_records.append(
                    {
                        "ZIP": zip_code,
                        "zipName": zip_name,
                        "label": label,
                        "year": year,
                        "crime_rate": crime_rate_y,
                        "zhvi": zhvi_y,
                        "population": pop_val,
                    }
                )

        if not animated_records:
            st.warning(
                "No ZIPs with both crime and home value data across years for the animated view."
            )
        else:
            animated_json = json.dumps(animated_records)

            html_template = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jstat/1.9.6/jstat.min.js"></script>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      font-family: Arial, sans-serif;
      width: 100%;
    }
    #valueSafetyAnimatedWrapper {
      position: relative;
      width: 100%;
    }
    #valueSafetyAnimatedChart {
      width: 100%;
    }
    #valueSafetyAnimatedChart svg {
      border-radius: 8px;
      background-color: #ffffff;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }
    .vs-tooltip {
      position: absolute;
      background-color: white;
      border: 1px solid #ccc;
      border-radius: 4px;
      padding: 8px;
      font-size: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
      pointer-events: none;
      display: none;
      z-index: 1000;
    }
    .vs-controls-container {
      width: 100%;
      display: flex;
      justify-content: center;
      margin-bottom: 10px;
      margin-top: 2px;
      pointer-events: none;
    }
    .vs-controls {
      pointer-events: auto;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 6px 14px;
      font-size: 12px;
      border-radius: 14px;
      border: 1px solid #e5e7eb;
      background: #f8fafc;
      box-shadow: none;
    }
    .vs-btn {
      padding: 4px 12px;
      border-radius: 6px;
      border: 1px solid transparent;
      background-color: #e5e7eb;
      color: #111827;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      min-width: 90px;
      justify-content: center;
      height: 28px;
      line-height: 1;
    }
    .vs-btn-primary {
      background-color: #2563eb;
      color: #ffffff;
      border-color: #1d4ed8;
    }
    .vs-btn:hover {
      filter: brightness(0.97);
    }
    .vs-btn-primary:hover {
      background-color: #1d4ed8;
    }
    .vs-btn-icon {
      font-size: 12px;
      display: inline-flex;
      align-items: center;
    }
    .vs-btn-text {
      display: inline-flex;
      align-items: center;
    }
    .vs-slider {
      -webkit-appearance: none;
      appearance: none;
      width: 220px;
      height: 4px;
      border-radius: 999px;
      background: #e5e7eb;
      outline: none;
      cursor: pointer;
    }
    .vs-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #2563eb;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 1px #2563eb;
    }
    .vs-slider::-moz-range-thumb {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #2563eb;
      border: 2px solid #ffffff;
      box-shadow: 0 0 0 1px #2563eb;
    }
    .vs-year-label {
      margin-left: 4px;
      font-weight: bold;
    }
    .vs-grid line {
      stroke: #e2e8f0;
      stroke-opacity: 0.9;
      shape-rendering: crispEdges;
    }
    .vs-grid path {
      display: none;
    }
    .vs-axis path,
    .vs-axis line {
      stroke: #94a3b8;
      stroke-width: 1;
      shape-rendering: crispEdges;
    }
    .vs-axis text {
      fill: #475569;
      font-size: 11px;
    }
    .vs-line-r {
      stroke: #2563eb;
      stroke-width: 2;
      fill: none;
    }
    .vs-line-p {
      stroke: #f97316;
      stroke-width: 2;
      fill: none;
      stroke-dasharray: 4 4;
    }
    .vs-year-marker {
      stroke: #111827;
      stroke-width: 1.5;
      stroke-dasharray: 4 4;
      pointer-events: none;
    }
    .vs-corr-label {
      font-size: 11px;
      fill: #334155;
    }
  </style>
</head>
<body>
  <div id="valueSafetyAnimatedWrapper">
    <div class="vs-controls-container">
      <div class="vs-controls">
        <button id="vsPlayPause" class="vs-btn vs-btn-primary">
          <span id="vsPlayPauseIcon" class="vs-btn-icon">▶</span>
          <span id="vsPlayPauseText" class="vs-btn-text">Play</span>
        </button>
        <input type="range" id="vsYearSlider" class="vs-slider" min="__MIN_YEAR__" max="__MAX_YEAR__" step="1" value="__MAX_YEAR__" />
        <span class="vs-year-label" id="vsYearLabel"></span>
      </div>
    </div>
    <div id="valueSafetyAnimatedChart"></div>
    <div class="vs-tooltip" id="vsTooltip"></div>
  </div>
  <script>
    const data = __DATA__;
    if (!Array.isArray(data) || data.length === 0) {
      document.getElementById("valueSafetyAnimatedChart").innerHTML = "<p>No data available.</p>";
    } else {
      const container = d3.select("#valueSafetyAnimatedChart");
      const tooltip = d3.select("#vsTooltip");
      const allYears = Array.from(new Set(data.map(d => d.year))).sort((a, b) => a - b);
      const minYear = allYears[0];
      const maxYear = allYears[allYears.length - 1];

      const width = 720;
      const height = 320;
      const margin = { top: 10, right: 20, bottom: 50, left: 80 };

      const svg = container
        .append("svg")
        .attr("width", "100%")
        .attr("height", height + margin.top + margin.bottom)
        .attr(
          "viewBox",
          `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`
        );

      const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear()
        .domain(d3.extent(data, d => d.crime_rate))
        .nice()
        .range([0, width]);

      const y = d3.scaleLinear()
        .domain(d3.extent(data, d => d.zhvi))
        .nice()
        .range([height, 0]);

      const sizeScale = d3.scaleSqrt()
        .domain(d3.extent(data, d => d.population))
        .range([3, 16]);

      const colorScale = d3.scaleOrdinal()
        .domain([
          "Undervalued (cheap vs crime)",
          "On-model",
          "Overvalued (expensive vs crime)"
        ])
        .range([
          "#e45756",  // Undervalued: warm red
          "#1f4b99",  // On-model: dark blue
          "#8fb9ff"   // Overvalued: light blue
        ]);

      const xAxis = d3.axisBottom(x).ticks(10);
      const yAxis = d3.axisLeft(y)
        .ticks(10)
        .tickFormat(d3.format(","));

      const xGrid = d3.axisBottom(x)
        .ticks(12)
        .tickSize(-height)
        .tickFormat("");

      const yGrid = d3.axisLeft(y)
        .ticks(12)
        .tickSize(-width)
        .tickFormat("");

      g.append("g")
        .attr("class", "vs-grid vs-grid-x")
        .attr("transform", `translate(0,${height})`)
        .call(xGrid);

      g.append("g")
        .attr("class", "vs-grid vs-grid-y")
        .call(yGrid);

      g.append("g")
        .attr("class", "vs-axis")
        .attr("transform", `translate(0,${height})`)
        .call(xAxis);

      g.append("g")
        .attr("class", "vs-axis")
        .call(yAxis);

      g.append("text")
        .attr("x", width / 2)
        .attr("y", height + 40)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .text("Crime rate (per 1,000 residents)");

      g.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -height / 2)
        .attr("y", -60)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .text("Home value (ZHVI, USD)");

      const pointsLayer = g.append("g");
      const regLine = g.append("line")
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "4,4");

      // Legend for value categories
      const legendData = [
        { label: "On-model", key: "On-model" },
        { label: "Overvalued (expensive vs crime)", key: "Overvalued (expensive vs crime)" },
        { label: "Undervalued (cheap vs crime)", key: "Undervalued (cheap vs crime)" },
      ];

      const legend = g.append("g")
        .attr("class", "vs-legend")
        .attr("transform", `translate(${width - 220}, 10)`);

      const legendItem = legend.selectAll("g")
        .data(legendData)
        .enter()
        .append("g")
        .attr("transform", (d, i) => `translate(0, ${i * 18})`);

      legendItem.append("circle")
        .attr("cx", 0)
        .attr("cy", 6)
        .attr("r", 5)
        .attr("fill", d => colorScale(d.key))
        .attr("stroke", "#ffffff")
        .attr("stroke-width", 1);

      legendItem.append("text")
        .attr("x", 12)
        .attr("y", 9)
        .text(d => d.label)
        .attr("fill", "#334155")
        .attr("font-size", 11);

      function classifyAndFit(yearData) {
        const n = yearData.length;
        if (!n) {
          return { points: yearData, line: null };
        }

        const xs = yearData.map(d => d.crime_rate);
        const ys = yearData.map(d => d.zhvi);
        const xMean = d3.mean(xs);
        const yMean = d3.mean(ys);

        let covXY = 0;
        let varX = 0;
        for (let i = 0; i < n; i++) {
          const dx = xs[i] - xMean;
          const dy = ys[i] - yMean;
          covXY += dx * dy;
          varX += dx * dx;
        }
        covXY /= n;
        varX /= n;

        if (!isFinite(varX) || varX === 0) {
          yearData.forEach(d => {
            d.value_class = "On-model";
          });
          return { points: yearData, line: null };
        }

        const b = covXY / (varX + 1e-9);
        const a = yMean - b * xMean;

        const residuals = yearData.map(d => d.zhvi - (a + b * d.crime_rate));
        const residMean = d3.mean(residuals);
        const residStd = d3.deviation(residuals) || 1.0;

        for (let i = 0; i < n; i++) {
          const z = (residuals[i] - residMean) / residStd;
          let cls = "On-model";
          if (z <= -0.5) {
            cls = "Undervalued (cheap vs crime)";
          } else if (z >= 0.5) {
            cls = "Overvalued (expensive vs crime)";
          }
          yearData[i].value_class = cls;
        }

        return { points: yearData, line: { a, b } };
      }

      // --- Correlation & p-value over time (line chart) ---
      function computeYearStats(year) {
        const yearData = data.filter(d => d.year === year);
        const n = yearData.length;
        if (n < 3) {
          return { year, r: NaN, p: NaN };
        }
        const xs = yearData.map(d => d.crime_rate);
        const ys = yearData.map(d => d.zhvi);
        const xMean = d3.mean(xs);
        const yMean = d3.mean(ys);
        let num = 0;
        let sumDx2 = 0;
        let sumDy2 = 0;
        for (let i = 0; i < n; i++) {
          const dx = xs[i] - xMean;
          const dy = ys[i] - yMean;
          num += dx * dy;
          sumDx2 += dx * dx;
          sumDy2 += dy * dy;
        }
        const denom = Math.sqrt(sumDx2 * sumDy2);
        let r = denom > 0 ? num / denom : NaN;
        let p = NaN;
        if (isFinite(r) && n > 2) {
          const df = n - 2;
          const t = r * Math.sqrt(df / Math.max(1e-9, 1 - r * r));
          if (window.jStat && jStat.studentt) {
            const cdf = jStat.studentt.cdf(Math.abs(t), df);
            p = 2 * (1 - cdf);
          }
        }
        return { year, r, p };
      }

      const statsByYear = allYears.map(computeYearStats);
      const statsIndex = new Map(statsByYear.map(s => [s.year, s]));

      const corrMargin = { top: 24, right: 60, bottom: 40, left: 80 };
      const corrHeight = 120;

      const svgCorr = container
        .append("svg")
        .attr("width", "100%")
        .attr("height", corrHeight + corrMargin.top + corrMargin.bottom)
        .attr(
          "viewBox",
          `0 0 ${width + corrMargin.left + corrMargin.right} ${corrHeight + corrMargin.top + corrMargin.bottom}`
        )
        .style("margin-top", "24px");

      const gCorr = svgCorr.append("g")
        .attr("transform", `translate(${corrMargin.left},${corrMargin.top})`);

      const xYear = d3.scaleLinear()
        .domain(d3.extent(allYears))
        .range([0, width]);

      const yR = d3.scaleLinear()
        .domain([-1, 1])
        .range([corrHeight, 0]);

      const yP = d3.scaleLinear()
        .domain([0, 1])
        .range([corrHeight, 0]);

      const xYearAxis = d3.axisBottom(xYear)
        .ticks(allYears.length)
        .tickFormat(d3.format("d"));

      const xYearGrid = d3.axisBottom(xYear)
        .ticks(allYears.length)
        .tickSize(-corrHeight)
        .tickFormat("");

      const yRAxis = d3.axisLeft(yR).ticks(5);
      const yPAxis = d3.axisRight(yP).ticks(5);

      gCorr.append("g")
        .attr("class", "vs-grid vs-grid-x")
        .attr("transform", `translate(0,${corrHeight})`)
        .call(xYearGrid);

      gCorr.append("g")
        .attr("class", "vs-axis")
        .attr("transform", `translate(0,${corrHeight})`)
        .call(xYearAxis);

      gCorr.append("g")
        .attr("class", "vs-axis")
        .call(yRAxis);

      gCorr.append("g")
        .attr("class", "vs-axis")
        .attr("transform", `translate(${width},0)`)
        .call(yPAxis);

      gCorr.append("text")
        .attr("x", width / 2)
        .attr("y", corrHeight + 32)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .text("Year");

      gCorr.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -corrHeight / 2)
        .attr("y", -55)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .text("Pearson r");

      gCorr.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -corrHeight / 2)
        .attr("y", width + 45)
        .attr("text-anchor", "middle")
        .attr("font-size", 11)
        .text("p-value");

      const rLine = d3.line()
        .defined(d => isFinite(d.r))
        .x(d => xYear(d.year))
        .y(d => yR(d.r));

      const pLine = d3.line()
        .defined(d => isFinite(d.p))
        .x(d => xYear(d.year))
        .y(d => yP(d.p));

      gCorr.append("path")
        .datum(statsByYear)
        .attr("class", "vs-line-r")
        .attr("d", rLine);

      gCorr.append("path")
        .datum(statsByYear)
        .attr("class", "vs-line-p")
        .attr("d", pLine);

      const yearMarker = gCorr.append("line")
        .attr("class", "vs-year-marker")
        .attr("y1", 0)
        .attr("y2", corrHeight);

      const statsLabel = gCorr.append("text")
        .attr("class", "vs-corr-label")
        .attr("x", 0)
        .attr("y", -8);

      // Interactive overlay on correlation chart: hover/click to scrub year
      const corrOverlay = gCorr.append("rect")
        .attr("width", width)
        .attr("height", corrHeight)
        .attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("mousemove", (event) => {
          if (playing) {
            setPlaying(false);
          }
          const [mx] = d3.pointer(event, gCorr.node());
          const year = Math.round(xYear.invert(mx));
          const clamped = Math.max(minYear, Math.min(maxYear, year));
          update(clamped);
        })
        .on("click", (event) => {
          if (playing) {
            setPlaying(false);
          }
          const [mx] = d3.pointer(event, gCorr.node());
          const year = Math.round(xYear.invert(mx));
          const clamped = Math.max(minYear, Math.min(maxYear, year));
          update(clamped);
        });

      function updateStatsDisplay(year) {
        const s = statsIndex.get(year);
        if (!s || !isFinite(s.r)) {
          statsLabel.text("");
          yearMarker.style("display", "none");
          return;
        }
        yearMarker
          .style("display", null)
          .attr("x1", xYear(year))
          .attr("x2", xYear(year));

        let pText = "N/A";
        if (isFinite(s.p)) {
          if (s.p < 0.001) {
            pText = "< 0.001";
          } else {
            pText = s.p.toFixed(3);
          }
        }
        statsLabel.text(`Year ${year}: r = ${s.r.toFixed(2)}, p-value = ${pText}`);
      }

      const slider = document.getElementById("vsYearSlider");
      const yearLabel = document.getElementById("vsYearLabel");
      let currentYear = parseInt(slider.value, 10);
      yearLabel.textContent = `Year: ${currentYear}`;

      function formatNumber(value) {
        return d3.format(",")(Math.round(value));
      }

      function update(year) {
        currentYear = year;
        slider.value = String(year);
        yearLabel.textContent = `Year: ${year}`;

        const yearData = data.filter(d => d.year === year);
        const { points, line } = classifyAndFit(yearData);

        const circles = pointsLayer.selectAll("circle")
          .data(points, d => d.ZIP);

        circles.exit()
          .transition()
          .duration(200)
          .attr("r", 0)
          .remove();

        const entering = circles.enter()
          .append("circle")
          .attr("r", 0)
          .attr("cx", d => x(d.crime_rate))
          .attr("cy", d => y(d.zhvi))
          .attr("fill", d => colorScale(d.value_class || "On-model"))
          .attr("fill-opacity", 0.85)
          .attr("stroke", "none")
          .on("mouseover", (event, d) => {
            const wrapper = document.getElementById("valueSafetyAnimatedWrapper");
            const [mx, my] = d3.pointer(event, wrapper);
            tooltip
              .style("display", "block")
              .style("left", `${mx + 12}px`)
              .style("top", `${my}px`)
              .html(
                `<strong>${d.label}</strong><br/>
                 Crime rate: ${d.crime_rate.toFixed(2)} per 1,000<br/>
                 Home value: $${formatNumber(d.zhvi)}<br/>
                 Population: ${formatNumber(d.population)}<br/>
                 Status: ${d.value_class}`
              );
          })
          .on("mousemove", (event) => {
            const wrapper = document.getElementById("valueSafetyAnimatedWrapper");
            const [mx, my] = d3.pointer(event, wrapper);
            tooltip
              .style("left", `${mx + 12}px`)
              .style("top", `${my}px`);
          })
          .on("mouseout", () => {
            tooltip.style("display", "none");
          });

        entering.merge(circles)
          .transition()
          .duration(300)
          .attr("r", d => sizeScale(d.population))
          .attr("cx", d => x(d.crime_rate))
          .attr("cy", d => y(d.zhvi))
          .attr("fill", d => colorScale(d.value_class || "On-model"))
          .attr("stroke", "none");

        if (line && isFinite(line.b)) {
          const xDomain = x.domain();
          const x1 = xDomain[0];
          const x2 = xDomain[1];
          const y1 = line.a + line.b * x1;
          const y2 = line.a + line.b * x2;

          regLine
            .style("display", null)
            .transition()
            .duration(300)
            .attr("x1", x(x1))
            .attr("y1", y(y1))
            .attr("x2", x(x2))
            .attr("y2", y(y2));
        } else {
          regLine.style("display", "none");
        }

        updateStatsDisplay(year);
      }

      slider.addEventListener("input", event => {
        const year = parseInt(event.target.value, 10);
        // When user scrubs the slider, pause playback and update to that year
        if (playing) {
          setPlaying(false);
        }
        update(year);
      });

      let playing = false;
      let timer = null;

      const playPauseButton = document.getElementById("vsPlayPause");
      const playPauseIcon = document.getElementById("vsPlayPauseIcon");
      const playPauseText = document.getElementById("vsPlayPauseText");

      function setPlaying(shouldPlay) {
        if (shouldPlay === playing) return;
        playing = shouldPlay;
        if (playing) {
          playPauseIcon.textContent = "⏸";
          playPauseText.textContent = "Pause";
          if (timer) {
            clearInterval(timer);
          }
          timer = setInterval(() => {
            let nextYear = currentYear + 1;
            const maxYear = parseInt(slider.max, 10);
            const minYear = parseInt(slider.min, 10);
            if (nextYear > maxYear) {
              nextYear = minYear;
            }
            update(nextYear);
          }, 800);
        } else {
          playPauseIcon.textContent = "▶";
          playPauseText.textContent = "Play";
          if (timer) {
            clearInterval(timer);
            timer = null;
          }
        }
      }

      playPauseButton.addEventListener("click", () => {
        setPlaying(!playing);
      });

      update(currentYear);
      // Animation is paused by default; use the controls to start.
    }
  </script>
</body>
</html>
"""

            html_code = (
                html_template.replace("__DATA__", animated_json)
                .replace("__MIN_YEAR__", str(year_start))
                .replace("__MAX_YEAR__", str(year_end))
            )
            components.html(html_code, height=640, scrolling=False)


with tabs[2]:
    st.subheader("📈 Trajectories")
    st.markdown(
        "This view tracks how each ZIP's crime rate and home values have changed over time. "
        "Points in each quadrant show combinations of rising or falling crime and rising or falling home values."
    )

    traj_df = build_traj_df(
        map_data, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year
    )
    if traj_df.empty:
        st.warning("No trajectory data available.")
    else:
        if "ZipName" in gdf_merged.columns:
            name_map = (
                gdf_merged[["ZIP", "ZipName"]]
                .drop_duplicates()
                .set_index("ZIP")["ZipName"]
                .to_dict()
            )
            traj_df["Label"] = traj_df["ZIP"].apply(
                lambda z: f"{z} – {name_map.get(z, '')}" if name_map.get(z, "") else z
            )
        else:
            traj_df["Label"] = traj_df["ZIP"]

        base = (
            alt.Chart(traj_df)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X(
                    "crime_change_pct:Q",
                    title="Crime rate change (%)",
                ),
                y=alt.Y(
                    "zhvi_change_pct:Q",
                    title="Home value change (%)",
                ),
                size=alt.Size(
                    "population:Q",
                    title="Population",
                    legend=None,
                    scale=alt.Scale(range=[40, 900]),
                ),
                tooltip=[
                    "Label",
                    alt.Tooltip(
                        "crime_change_pct:Q",
                        title="Crime rate change (%)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "zhvi_change_pct:Q",
                        title="Home value change (%)",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "population:Q",
                        title="Population",
                        format=",",
                    ),
                ],
            )
        )

        vline = alt.Chart(pd.DataFrame({"crime_change_pct": [0]})).mark_rule(
            strokeDash=[4, 4], color="gray"
        ).encode(x="crime_change_pct:Q")

        hline = alt.Chart(pd.DataFrame({"zhvi_change_pct": [0]})).mark_rule(
            strokeDash=[4, 4], color="gray"
        ).encode(y="zhvi_change_pct:Q")

        chart = (base + vline + hline).properties(
            height=420,
            title=f"Change from {crime_min_year} to {crime_max_year}",
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Upper-right: home values and crime both increased. "
            "Upper-left: home values up, crime down. "
            "Lower-right: home values down, crime up. "
            "Lower-left: both decreased."
        )


with tabs[3]:
    st.subheader("🧩 Neighborhood Archetypes")
    st.markdown(
        "This tab groups ZIPs conceptually by crime and home value levels. "
        "A full clustering view (with ML-based clusters and maps) can be added later if needed."
    )

    if zip_df.empty:
        st.warning("No ZIP-level statistics available.")
    else:
        df = zip_df.copy()

        # Simple archetype classification by median splits
        crime_med = df["crime_rate_latest"].median()
        zhvi_med = df["zhvi_latest"].median()

        def classify_row(r):
            if r["crime_rate_latest"] <= crime_med and r["zhvi_latest"] >= zhvi_med:
                return "High value / safer"
            if r["crime_rate_latest"] > crime_med and r["zhvi_latest"] >= zhvi_med:
                return "High value / riskier"
            if r["crime_rate_latest"] <= crime_med and r["zhvi_latest"] < zhvi_med:
                return "Lower value / safer"
            return "Lower value / riskier"

        df["archetype"] = df.apply(classify_row, axis=1)

        chart = (
            alt.Chart(df)
            .mark_circle(opacity=0.85)
            .encode(
                x=alt.X(
                    "crime_rate_latest:Q",
                    title="Crime rate (per 1,000 residents)",
                ),
                y=alt.Y(
                    "zhvi_latest:Q",
                    title="Home value (latest ZHVI, USD)",
                ),
                color=alt.Color("archetype:N", title="Archetype"),
                size=alt.Size(
                    "population:Q",
                    title="Population",
                    legend=None,
                    scale=alt.Scale(range=[40, 900]),
                ),
                tooltip=[
                    "ZIP",
                    "ZipName",
                    alt.Tooltip("population:Q", title="Population", format=","),
                    alt.Tooltip(
                        "crime_rate_latest:Q",
                        title="Crime rate (per 1,000)",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "zhvi_latest:Q",
                        title="Home value (ZHVI)",
                        format=",.0f",
                    ),
                    "archetype",
                ],
            )
        ).properties(height=420)

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Archetypes are defined by whether a ZIP is above/below the city median "
            "for crime rate and home value."
        )

        # --- JS map: Neighborhood Archetypes by ZIP (Leaflet) ---
        archetype_by_zip = {
            str(row["ZIP"]): row["archetype"] for _, row in df.iterrows()
        }

        zipname_by_zip = {}
        if "ZipName" in gdf_merged.columns:
            name_lookup = (
                gdf_merged[["ZIP", "ZipName"]]
                .drop_duplicates()
            )
            name_lookup["ZIP"] = name_lookup["ZIP"].astype(str)
            zipname_by_zip = dict(zip(name_lookup["ZIP"], name_lookup["ZipName"]))

        map_data_archetype = []
        for z in map_data:
            zip_code = str(z["zip"])
            archetype = archetype_by_zip.get(zip_code)
            if archetype is None:
                continue
            map_data_archetype.append(
                {
                    "zip": zip_code,
                    "zipName": zipname_by_zip.get(zip_code, ""),
                    "archetype": archetype,
                    "centroid_lat": z["centroid_lat"],
                    "centroid_lng": z["centroid_lng"],
                    "coordinates": z["coordinates"],
                }
            )

        st.markdown("---")
        st.subheader("Neighborhood Archetypes Map")

        if not map_data_archetype:
            st.info("No ZIPs available for the archetype map with current filters.")
        else:
            map_data_json = json.dumps(map_data_archetype)

            html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <link
      rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            width: 100%;
            height: 100%;
        }}
        #mapWrapper {{
            position: relative;
            width: 100%;
            height: 520px;
        }}
        #archetypeMap {{
            width: 100%;
            height: 100%;
        }}
        .zip-tooltip {{
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            right: 10px;
            background: white;
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-size: 12px;
        }}
        .legend-title {{
            font-weight: bold;
            margin-bottom: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 4px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 6px;
        }}
        /* Remove default blue focus outline on click */
        .leaflet-container .leaflet-interactive:focus {{
            outline: none;
        }}
        /* Selected ZIP indicator (match main map style) */
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
    </style>
</head>
<body>
    <div id="mapWrapper">
        <div id="archetypeMap"></div>

        <div class="selected-zip-indicator" id="selectedIndicator" style="display: none;">
            📍 ZIP: <span id="selectedZipDisplay">-</span>
            <button class="clear-btn" id="clearSelectionBtn">✕</button>
            <div id="selectedZipName" style="font-size: 12px; font-weight: normal; margin-top: 4px;"></div>
            <div id="selectedArchetype" style="font-size: 12px; font-weight: normal;"></div>
        </div>

        <div class="click-instruction" id="clickInstruction">
            👆 Click a ZIP to see its archetype. Click outside to clear.
        </div>

        <div class="legend" id="legend"></div>
    </div>
    <script>
        const mapData = {map_data_json};

        const archetypeColors = {{
            "High value / safer": "#1a9850",
            "High value / riskier": "#fee08b",
            "Lower value / safer": "#91bfdb",
            "Lower value / riskier": "#d73027"
        }};

        function colorForArchetype(a) {{
            return archetypeColors[a] || "#999999";
        }}

        let selectedZip = null;
        let layersByZip = {{}};

        // Initialize map
        const map = L.map('archetypeMap', {{
            scrollWheelZoom: false,
            doubleClickZoom: false,
            dragging: true,
            zoomControl: true,
            minZoom: 9,
            maxZoom: 18
        }}).setView([41.85, -87.65], 11);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);

        const labelLayerGroup = L.layerGroup().addTo(map);

        if (Array.isArray(mapData) && mapData.length > 0) {{
            const bounds = L.latLngBounds(
                mapData
                    .filter(z => typeof z.centroid_lat === 'number' && typeof z.centroid_lng === 'number')
                    .map(z => [z.centroid_lat, z.centroid_lng])
            );
            if (bounds.isValid()) {{
                map.fitBounds(bounds.pad(0.01));
                // Zoom in a bit more than the default fit
                map.setZoom(map.getZoom() + 1);
            }}
        }}

        const features = mapData.map(z => ({{
                type: 'Feature',
                properties: {{
                    zip: z.zip,
                    zipName: z.zipName || '',
                    archetype: z.archetype
            }},
            geometry: {{
                type: 'Polygon',
                coordinates: [z.coordinates.map(c => [c[1], c[0]])]
            }}
        }}));

        function clearAllHighlights() {{
            Object.values(layersByZip).forEach(layer => {{
                if (layer) {{
                    layer.setStyle({{
                        weight: 1,
                        color: '#333',
                        fillOpacity: 0.8
                    }});
                }}
            }});
        }}

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

        function clearSelection() {{
            clearAllHighlights();
            selectedZip = null;
            document.getElementById('selectedIndicator').style.display = 'none';
            document.getElementById('clickInstruction').style.display = 'block';
        }}

        const geoLayer = L.geoJSON({{
            type: 'FeatureCollection',
            features: features
        }}, {{
            style: function(feature) {{
                return {{
                    fillColor: colorForArchetype(feature.properties.archetype),
                    weight: 1.2,
                    color: '#555',
                    fillOpacity: 0.8
                }};
            }},
            onEachFeature: function(feature, layer) {{
                const p = feature.properties;
                const zipCode = String(p.zip);

                layersByZip[zipCode] = layer;

                let tooltip = `<b>ZIP: ${{p.zip}}</b>`;
                if (p.zipName) {{
                    tooltip += `<br>${{p.zipName}}`;
                }}
                tooltip += `<br>${{p.archetype}}`;

                layer.bindTooltip(tooltip, {{
                    className: 'zip-tooltip',
                    permanent: false,
                    direction: 'top',
                    sticky: true,
                    opacity: 0.95
                }});

                layer.on('mouseover', function() {{
                    if (zipCode !== selectedZip) {{
                        this.setStyle({{
                            weight: 3,
                            color: '#000',
                            fillOpacity: 0.9
                        }});
                    }}
                    this.bringToFront();
                }});
                layer.on('mouseout', function() {{
                    if (zipCode !== selectedZip) {{
                        this.setStyle({{
                            weight: 1,
                            color: '#333',
                            fillOpacity: 0.8
                        }});
                    }}
                }});

                layer.on('click', function(e) {{
                    window._polygonClicked = true;
                    clearAllHighlights();
                    selectedZip = zipCode;
                    highlightZip(zipCode);

                    document.getElementById('selectedZipDisplay').textContent = zipCode;
                    document.getElementById('selectedZipName').textContent = p.zipName || '';
                    document.getElementById('selectedArchetype').textContent = p.archetype;
                    document.getElementById('selectedIndicator').style.display = 'block';
                    document.getElementById('clickInstruction').style.display = 'none';

                    L.DomEvent.stopPropagation(e);
                }});

                // ZIP label in the centroid (like main map)
                const center = layer.getBounds().getCenter();
                const labelMarker = L.marker(center, {{
                    icon: L.divIcon({{
                        className: 'zip-label',
                        html: `<div style="font-size: 9px; font-weight: bold; color: black; text-align: center; text-shadow: 1px 1px 2px white, -1px -1px 2px white;">${{p.zip}}</div>`,
                        iconSize: [40, 20],
                        iconAnchor: [20, 10]
                    }}),
                    interactive: false,
                    keyboard: false
                }});
                labelLayerGroup.addLayer(labelMarker);
            }}
        }}).addTo(map);

        // Map click to clear selection when clicking outside polygons
        map.on('click', function() {{
            setTimeout(function() {{
                if (!window._polygonClicked) {{
                    clearSelection();
                }}
                window._polygonClicked = false;
            }}, 50);
        }});

        document.getElementById('clearSelectionBtn').addEventListener('click', function(e) {{
            e.stopPropagation();
            clearSelection();
        }});

        // Legend
        const legendEl = document.getElementById('legend');
        const entries = Object.keys(archetypeColors);
        legendEl.innerHTML = `
            <div class="legend-title">Archetypes</div>
            ${{entries.map(a => `
                <div class="legend-item">
                    <span class="legend-color" style="background:${{archetypeColors[a]}}"></span>
                    <span>${{a}}</span>
                </div>
            `).join('')}}
        `;
    </script>
</body>
</html>
"""
            components.html(html_code, height=540, scrolling=False)

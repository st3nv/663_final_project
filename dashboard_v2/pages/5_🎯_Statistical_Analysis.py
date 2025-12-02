import os
import sys
import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import json
import streamlit.components.v1 as components


# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    load_crime_data,
    load_geodata,
    load_housing_data,
    load_population_data,
    get_exclude_settings,
    apply_zip_exclusion,
    prepare_map_data,
)

st.set_page_config(page_title="Statistical Analysis", page_icon="💡", layout="wide")

st.title("🎯 Statistical Analysis of Relationship Between Home Values and Crime")
st.markdown(
    "This lab looks at how **home values (ZHVI)** and **crime rates** are linked across Chicago ZIP codes "
    "and over time, using pooled regression models, clustering, and spatial/temporal views."
)

# Sidebar settings (consistent with other v2 pages)
st.sidebar.header("📌 Dashboard")
exclude_zips, exclude_2025 = get_exclude_settings()

# Load core data
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


@st.cache_data(show_spinner=False)
def build_model_dataframe(
    _gdf_in: pd.DataFrame,
    pop_2021: pd.DataFrame,
    crime_data: pd.DataFrame,
    zhvi_data: pd.DataFrame,
):
    """Construct per-(ZIP, year) panel with crime rate and ZHVI, plus static ZIP attributes."""
    gdf_local = _gdf_in.copy()
    gdf_local["ZIP"] = gdf_local["ZIP"].astype(str)

    # Attach 2021 population to ZIP shapes
    pop_local = pop_2021.copy()
    pop_local["ZIP"] = pop_local["ZIP"].astype(str)
    gdf_local = gdf_local.merge(
        pop_local[["ZIP", "Population - Total"]],
        on="ZIP",
        how="left",
    )
    gdf_local["Population - Total"] = gdf_local["Population - Total"].fillna(0.0)

    # Approx distance to Loop (degree space, used as static location control)
    loop_lat, loop_lon = 41.8781, -87.6298
    gdf_local["centroid"] = gdf_local.geometry.centroid
    gdf_local["centroid_lon"] = gdf_local["centroid"].x
    gdf_local["centroid_lat"] = gdf_local["centroid"].y
    gdf_local["distance_from_loop"] = np.sqrt(
        (gdf_local["centroid_lat"] - loop_lat) ** 2
        + (gdf_local["centroid_lon"] - loop_lon) ** 2
    )

    # Crime by ZIP-year
    crime_local = crime_data.copy()
    crime_local["ZIP"] = crime_local["ZIP"].astype(str)
    crime_yearly = (
        crime_local.groupby(["ZIP", "year"])["crime_count"]
        .sum()
        .reset_index()
    )

    # ZHVI by ZIP-year
    zhvi_local = zhvi_data.copy()
    zhvi_local["RegionName"] = zhvi_local["RegionName"].astype(str)
    zhvi_yearly = (
        zhvi_local.groupby(["RegionName", "Year"])["Zhvi"]
        .mean()
        .reset_index()
    )
    zhvi_yearly = zhvi_yearly.rename(
        columns={"RegionName": "ZIP", "Year": "year"},
    )

    # Inner join on overlapping (ZIP, year)
    panel = crime_yearly.merge(zhvi_yearly, on=["ZIP", "year"], how="inner")

    # Attach static ZIP attributes
    meta_cols = ["ZIP", "ZipName", "Population - Total", "distance_from_loop"]
    meta_df = gdf_local[meta_cols].drop_duplicates()
    panel = panel.merge(meta_df, on="ZIP", how="left")

    # Compute crime rate per 1,000 residents
    panel["crime_rate"] = np.where(
        panel["Population - Total"] > 0,
        panel["crime_count"] / panel["Population - Total"] * 1000.0,
        np.nan,
    )
    panel["log_zhvi"] = np.log(panel["Zhvi"])

    # Drop rows without key variables
    panel = panel.dropna(subset=["crime_rate", "log_zhvi", "distance_from_loop"])

    return panel


model_df = build_model_dataframe(gdf, pop_df, crime_df, zhvi_df)

if model_df.empty:
    st.warning("Not enough overlapping crime and ZHVI data to run the analysis.")
    st.stop()

# Derive helper data
years_available = sorted(model_df["year"].unique())
zip_labels = (
    model_df[["ZIP", "ZipName"]]
    .drop_duplicates()
    .assign(
        Label=lambda d: d.apply(
            lambda r: f"{r['ZIP']} – {r['ZipName']}"
            if isinstance(r["ZipName"], str) and r["ZipName"]
            else r["ZIP"],
            axis=1,
        )
    )
    .sort_values("Label")
)

tabs = st.tabs(
    [
        "Pooled Regression Model",
        "ZIP-level Clustering",
        "Crime Types vs Home Values",
    ]
)


with tabs[0]:
    st.subheader("📊 Pooled Regression: Does Crime Rate Predict Home Values?")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        year_filter = st.select_slider(
            "Limit analysis to years (inclusive)",
            options=years_available,
            value=(years_available[0], years_available[-1]),
        )
    with col_right:
        st.markdown(
            "We model log(ZHVI) as a function of crime rate and distance from the Loop, "
            "and include **year fixed effects** so that we only compare ZIP codes **within the same year**. "
            "This avoids misleading results that would come from pooling across years while home values "
            "trend up and crime trends down over time."
        )

    year_min, year_max = year_filter
    reg_df = model_df[
        (model_df["year"] >= year_min) & (model_df["year"] <= year_max)
    ].copy()

    if reg_df.empty:
        st.warning("No observations in the selected year range.")
    else:
        # Manual OLS using closed-form solution for transparency (no statsmodels dependency in v2)
        # Core regressors
        X_core = reg_df[["crime_rate", "distance_from_loop"]].to_numpy(dtype=float)
        y = reg_df["log_zhvi"].to_numpy(dtype=float).reshape(-1, 1)

        # Add intercept and year fixed effects so that time trends in prices/crime
        # are absorbed by year dummies rather than driving the crime coefficient.
        year_dummies = pd.get_dummies(
            reg_df["year"].astype(int),
            prefix="year",
            drop_first=True,
        )
        X_parts = [np.ones((X_core.shape[0], 1)), X_core]
        fe_term_names = []
        if not year_dummies.empty:
            X_parts.append(year_dummies.to_numpy(dtype=float))
            fe_term_names = list(year_dummies.columns)

        X_design = np.column_stack(X_parts)

        # Prepare containers for standard errors and p-values
        core_coef_names = ["Intercept", "crime_rate", "distance_from_loop"]
        all_coef_names = core_coef_names + fe_term_names
        std_err_vals = np.full(len(all_coef_names), np.nan, dtype=float)
        p_vals = np.full(len(all_coef_names), np.nan, dtype=float)

        try:
            beta = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y)
            reg_df["fitted_log_zhvi"] = (X_design @ beta).ravel()
            reg_df["resid"] = reg_df["log_zhvi"] - reg_df["fitted_log_zhvi"]

            # Simple R²
            ss_tot = np.sum((reg_df["log_zhvi"] - reg_df["log_zhvi"].mean()) ** 2)
            ss_res = np.sum(reg_df["resid"] ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # Approximate standard errors and p-values using normal approximation
            n_obs, n_params = X_design.shape
            if n_obs > n_params:
                sigma2 = ss_res / (n_obs - n_params)
                xtx_inv = np.linalg.inv(X_design.T @ X_design)
                var_beta = sigma2 * xtx_inv
                std_err_vals = np.sqrt(np.diag(var_beta))

                t_vals = beta.ravel() / std_err_vals
                # Normal CDF via error function (approximate t distribution)
                p_vals = np.array(
                    [
                        2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))
                        if np.isfinite(t)
                        else np.nan
                        for t in t_vals
                    ]
                )
        except np.linalg.LinAlgError:
            beta = None
            r2 = np.nan

        coef_df = (
            pd.DataFrame(
                {
                    "term": all_coef_names,
                    "coef": beta.ravel() if beta is not None else [np.nan] * len(all_coef_names),
                    "std_err": std_err_vals if beta is not None else [np.nan] * len(all_coef_names),
                    "p_value": p_vals if beta is not None else [np.nan] * len(all_coef_names),
                }
            )
            if beta is not None
            else pd.DataFrame(
                {
                    "term": all_coef_names,
                    "coef": [np.nan] * len(all_coef_names),
                    "std_err": [np.nan] * len(all_coef_names),
                    "p_value": [np.nan] * len(all_coef_names),
                }
            )
        )

        # Show the main coefficients of interest (intercept, crime_rate, distance_from_loop)
        coef_display = coef_df[coef_df["term"].isin(core_coef_names)].copy()
        coef_display["term"] = pd.Categorical(
            coef_display["term"],
            categories=core_coef_names,
            ordered=True,
        )
        coef_display = coef_display.sort_values("term")

        st.markdown("**Model specification**")
        st.latex(
            r"\log(\text{ZHVI}_{it}) = \beta_0 + \beta_1 \cdot \text{CrimeRate}_{it} + "
            r"\beta_2 \cdot \text{DistanceFromLoop}_i + \sum_{t} \gamma_t \mathbf{1}\{\text{Year}=t\} + \varepsilon_{it}"
        )

        st.markdown("**Estimated coefficients (pooled OLS)**")
        st.dataframe(
            coef_display.style.format(
                {
                    "coef": "{:.4f}",
                    "std_err": "{:.4f}",
                    "p_value": "{:.3g}",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            f"Using {len(reg_df):,} ZIP-year observations from {int(year_min)}–{int(year_max)}. "
            f"Pooled R² with year fixed effects (approximate) ≈ **{r2:.3f}**."
        )

        st.markdown("**Predicted vs actual log(ZHVI)**")
        diag_df = reg_df[["log_zhvi", "fitted_log_zhvi"]].rename(
            columns={
                "log_zhvi": "actual_log_zhvi",
                "fitted_log_zhvi": "fitted_log_zhvi",
            }
        )
        scatter = (
            alt.Chart(diag_df)
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X(
                    "fitted_log_zhvi:Q",
                    title="Predicted log(ZHVI)",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "actual_log_zhvi:Q",
                    title="Actual log(ZHVI)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("fitted_log_zhvi:Q", title="Predicted", format=".3f"),
                    alt.Tooltip("actual_log_zhvi:Q", title="Actual", format=".3f"),
                ],
            )
        )
        ref_line = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x": [diag_df["actual_log_zhvi"].min(), diag_df["actual_log_zhvi"].max()]
                    }
                )
            )
            .transform_calculate(y="datum.x")
            .mark_line(color="black", strokeDash=[4, 4])
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart((scatter + ref_line).properties(height=360), use_container_width=True)

        st.markdown("**Year-by-year crime effect (partial correlation)**")
        slope_records = []
        for y_year in sorted(reg_df["year"].unique()):
            df_y = reg_df[reg_df["year"] == y_year]
            if df_y["crime_rate"].var() > 0 and df_y["log_zhvi"].var() > 0 and len(df_y) > 5:
                X_y = df_y[["crime_rate", "distance_from_loop"]].to_numpy(dtype=float)
                y_y = df_y["log_zhvi"].to_numpy(dtype=float).reshape(-1, 1)
                Xy_design = np.column_stack([np.ones(X_y.shape[0]), X_y])
                try:
                    beta_y = np.linalg.inv(Xy_design.T @ Xy_design) @ (Xy_design.T @ y_y)
                    slope_records.append(
                        {"year": int(y_year), "beta_crime": float(beta_y[1, 0])}
                    )
                except np.linalg.LinAlgError:
                    continue

        if slope_records:
            slope_df = pd.DataFrame(slope_records).sort_values("year")
            slope_chart = (
                alt.Chart(slope_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")),
                    y=alt.Y(
                        "beta_crime:Q",
                        title="Coefficient on crime_rate (log ZHVI model)",
                    ),
                    tooltip=[
                        alt.Tooltip("year:Q", title="Year", format="d"),
                        alt.Tooltip("beta_crime:Q", title="β₁ (crime rate)", format=".4f"),
                    ],
                )
            )
            zero_rule = alt.Chart(slope_df).mark_rule(
                strokeDash=[4, 4], color="gray"
            ).encode(y=alt.datum(0))
            st.altair_chart(
                (slope_chart + zero_rule).properties(height=280),
                use_container_width=True,
            )
            st.caption(
                "Negative coefficients indicate that, controlling for distance to the Loop, higher crime "
                "is associated with lower home values in that year."
            )


with tabs[1]:
    st.subheader("🧩 ZIP-level Clusters: Grouping Neighborhoods by Risk & Value")
    st.markdown(
        "Here we summarize **average crime rate** and **average ZHVI** for each ZIP over the selected period, "
        "group ZIPs into simple archetypes, and show where those archetypes live on the map."
    )

    year_min, year_max = st.select_slider(
        "Aggregate over years (inclusive)",
        options=years_available,
        value=(years_available[0], years_available[-1]),
    )

    agg_df = model_df[
        (model_df["year"] >= year_min) & (model_df["year"] <= year_max)
    ].copy()

    if agg_df.empty:
        st.warning("No observations in the selected year range.")
    else:
        # ZIP-level summary over the selected window (Statistical Analysis view)
        zip_summary = (
            agg_df.groupby("ZIP")
            .agg(
                mean_zhvi=("Zhvi", "mean"),
                mean_crime_rate=("crime_rate", "mean"),
                population=("Population - Total", "first"),
                ZipName=("ZipName", "first"),
            )
            .reset_index()
        )
        zip_summary = zip_summary[zip_summary["population"] > 0].copy()

        # Archetypes based on period averages (this view's own definition)
        crime_med = zip_summary["mean_crime_rate"].median()
        zhvi_med = zip_summary["mean_zhvi"].median()

        def archetype_row(r):
            if r["mean_crime_rate"] <= crime_med and r["mean_zhvi"] >= zhvi_med:
                return "High value / safer"
            if r["mean_crime_rate"] > crime_med and r["mean_zhvi"] >= zhvi_med:
                return "High value / riskier"
            if r["mean_crime_rate"] <= crime_med and r["mean_zhvi"] < zhvi_med:
                return "Lower value / safer"
            return "Lower value / riskier"

        zip_summary["archetype"] = zip_summary.apply(archetype_row, axis=1)

        # --- Scatter view (Statistical Analysis clustering) ---
        cluster_chart = (
            alt.Chart(zip_summary)
            .mark_circle(opacity=0.8)
            .encode(
                x=alt.X(
                    "mean_crime_rate:Q",
                    title="Average crime rate (per 1,000 residents)",
                ),
                y=alt.Y(
                    "mean_zhvi:Q",
                    title="Average home value (ZHVI, USD)",
                    axis=alt.Axis(format="$,.0f"),
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
                    alt.Tooltip(
                        "mean_crime_rate:Q",
                        title="Avg crime rate",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        "mean_zhvi:Q",
                        title="Avg ZHVI",
                        format=",.0f",
                    ),
                    alt.Tooltip(
                        "population:Q",
                        title="Population",
                        format=",",
                    ),
                    "archetype",
                ],
            )
        )

        vline = alt.Chart(
            pd.DataFrame({"mean_crime_rate": [crime_med]})
        ).mark_rule(strokeDash=[4, 4], color="gray").encode(x="mean_crime_rate:Q")
        hline = alt.Chart(
            pd.DataFrame({"mean_zhvi": [zhvi_med]})
        ).mark_rule(strokeDash=[4, 4], color="gray").encode(y="mean_zhvi:Q")

        st.altair_chart(
            (cluster_chart + vline + hline).properties(
                height=420,
                title=f"ZIP-level clusters ({int(year_min)}–{int(year_max)})",
            ),
            use_container_width=True,
        )

        st.caption(
            "Archetypes in this scatter are defined by whether a ZIP is above/below the city medians for "
            "average crime rate and average ZHVI over the selected period."
        )

        st.markdown("**Cluster membership table**")
        table_df = zip_summary[
            ["ZIP", "ZipName", "mean_crime_rate", "mean_zhvi", "population", "archetype"]
        ].rename(
            columns={
                "mean_crime_rate": "Avg crime rate",
                "mean_zhvi": "Avg ZHVI",
            }
        )
        st.dataframe(
            table_df.style.format(
                {
                    "Avg crime rate": "{:.1f}",
                    "Avg ZHVI": "{:,.0f}",
                    "population": "{:,}",
                }
            ),
            use_container_width=True,
        )

        # --- Map view (Neighborhood Archetypes-style Leaflet map, using this tab's archetypes) ---
        st.markdown("---")
        st.subheader("Neighborhood Archetypes Map")

        # Reuse the load_geodata/prepare_map_data-style structure to mirror Relationship Lab
        # Build a ZIP→archetype lookup from this tab's period-averaged clusters
        archetype_by_zip = {
            str(row["ZIP"]): row["archetype"] for _, row in zip_summary.iterrows()
        }

        # Build a ZipName lookup from the merged geo dataframe
        zipname_by_zip = {}
        if "ZipName" in gdf.columns:
            name_lookup = gdf[["ZIP", "ZipName"]].drop_duplicates().copy()
            name_lookup["ZIP"] = name_lookup["ZIP"].astype(str)
            zipname_by_zip = dict(zip(name_lookup["ZIP"], name_lookup["ZipName"]))

        # Prepare map-style data (zip, centroid, coordinates) using prepare_map_data output
        # We recreate a minimal map_data-like structure here
        map_data_like, gdf_merged_like, *_ = prepare_map_data(gdf.copy(), crime_df, zhvi_df, pop_df)

        map_data_archetype = []
        for z in map_data_like:
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

        if not map_data_archetype:
            st.info("No ZIPs available for the archetype map with current filters and year range.")
        else:
            map_data_json = json.dumps(map_data_archetype)
            all_map_data_json = json.dumps(map_data_like)

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
            background: rgba(255, 255, 255, 0.96);
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-size: 12px;
            z-index: 1000;
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
        .leaflet-container .leaflet-interactive:focus {{
            outline: none;
        }}
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
        const allMapData = {all_map_data_json};
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

        const map = L.map('archetypeMap', {{
            scrollWheelZoom: false,
            doubleClickZoom: false,
            dragging: true,
            zoomControl: true,
            minZoom: 9,
            maxZoom: 18
        }});

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);

        const labelLayerGroup = L.layerGroup().addTo(map);

        if (Array.isArray(allMapData) && allMapData.length > 0) {{
            const bounds = L.latLngBounds(
                allMapData
                    .filter(z => typeof z.centroid_lat === 'number' && typeof z.centroid_lng === 'number')
                    .map(z => [z.centroid_lat, z.centroid_lng])
            );
            if (bounds.isValid()) {{
                map.fitBounds(bounds.pad(0.01));
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

with tabs[2]:
    st.subheader("🧪 Which Crime Types Are Most Linked to Home Values?")
    st.markdown(
        "Here we look at how **crime rates by type** line up with **home values (ZHVI)** across ZIP codes. "
        "We summarize the relationship using simple correlations across ZIP-year observations "
        "for the most common crime types. This shows association, not causation."
    )

    if "Primary Type" not in crime_df.columns:
        st.warning("Crime type information is not available in this dataset.")
    else:
        year_min, year_max = st.select_slider(
            "Year range for crime-type analysis",
            options=years_available,
            value=(2020, 2024)
        )

        crime_types = crime_df.copy()
        crime_types["ZIP"] = crime_types["ZIP"].astype(str)
        crime_types = crime_types[
            (crime_types["year"] >= year_min) & (crime_types["year"] <= year_max)
        ]

        if crime_types.empty:
            st.warning("No crime data in the selected year range.")
        else:
            pop_local = pop_df.copy()
            pop_local["ZIP"] = pop_local["ZIP"].astype(str)

            zhvi_local = zhvi_df.copy()
            zhvi_local["RegionName"] = zhvi_local["RegionName"].astype(str)
            zhvi_local = zhvi_local[
                (zhvi_local["Year"] >= year_min) & (zhvi_local["Year"] <= year_max)
            ]

            zhvi_panel = (
                zhvi_local.groupby(["RegionName", "Year"])["Zhvi"]
                .mean()
                .reset_index()
                .rename(columns={"RegionName": "ZIP", "Year": "year"})
            )

            type_panel = (
                crime_types.groupby(["ZIP", "year", "Primary Type"])["crime_count"]
                .sum()
                .reset_index()
            )

            type_panel = type_panel.merge(
                pop_local[["ZIP", "Population - Total"]],
                on="ZIP",
                how="left",
            )

            type_panel = type_panel.merge(
                zhvi_panel,
                on=["ZIP", "year"],
                how="inner",
            )

            type_panel["Population - Total"] = type_panel["Population - Total"].fillna(0.0)
            type_panel["crime_rate_type"] = np.where(
                type_panel["Population - Total"] > 0,
                type_panel["crime_count"] / type_panel["Population - Total"] * 1000.0,
                np.nan,
            )

            type_panel = type_panel.dropna(subset=["crime_rate_type", "Zhvi"])

            if type_panel.empty:
                st.warning(
                    "Not enough overlapping crime-type and ZHVI data to compute relationships "
                    f"for {int(year_min)}–{int(year_max)}."
                )
            else:
                type_totals = (
                    type_panel.groupby("Primary Type")["crime_count"]
                    .sum()
                    .sort_values(ascending=False)
                )

                top_n = 10
                top_types = type_totals.head(top_n).index.tolist()

                type_panel_top = type_panel[
                    type_panel["Primary Type"].isin(top_types)
                ].copy()

                corr_records = []
                for t in top_types:
                    sub = type_panel_top[type_panel_top["Primary Type"] == t]
                    if (
                        sub["crime_rate_type"].var() <= 0
                        or sub["Zhvi"].var() <= 0
                        or len(sub) < 5
                    ):
                        continue
                    corr_val = sub["crime_rate_type"].corr(sub["Zhvi"])
                    if pd.isna(corr_val):
                        continue
                    corr_records.append(
                        {
                            "Primary Type": t,
                            "correlation": float(corr_val),
                            "total_crimes": int(sub["crime_count"].sum()),
                        }
                    )

                if not corr_records:
                    st.warning(
                        "Could not compute stable correlations for the selected period. "
                        "Try widening the year range."
                    )
                else:
                    corr_df = (
                        pd.DataFrame(corr_records)
                        .sort_values("correlation")
                        .reset_index(drop=True)
                    )

                    top_row = corr_df.iloc[0]
                    st.metric(
                        "Crime type most linked to lower values",
                        top_row["Primary Type"],
                        f"corr = {top_row['correlation']:.2f}",
                        help=(
                            "Correlation between this crime type's rate (per 1,000 residents) "
                            "and ZHVI across ZIP-year observations in the selected period. "
                            "More negative means areas with more of this crime tend to have lower home values."
                        ),
                    )

                    st.markdown(
                        "Below, bars show how strongly each of the most common crime types is associated with "
                        "home values in the selected years. Negative correlations indicate that ZIPs with more of "
                        "that crime type tend to have lower ZHVI."
                    )

                    sorted_types = (
                        corr_df.sort_values("correlation")["Primary Type"].tolist()
                    )

                    bar_chart = (
                        alt.Chart(corr_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "correlation:Q",
                                title="Correlation with home value (ZHVI)",
                                scale=alt.Scale(domain=(-1, 1)),
                            ),
                            y=alt.Y(
                                "Primary Type:N",
                                title="Crime type",
                                sort=sorted_types,
                            ),
                            color=alt.Color(
                                "correlation:Q",
                                title="Correlation",
                                scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Primary Type:N",
                                    title="Crime type",
                                ),
                                alt.Tooltip(
                                    "correlation:Q",
                                    title="Correlation",
                                    format=".2f",
                                ),
                                alt.Tooltip(
                                    "total_crimes:Q",
                                    title="Total crimes in period",
                                    format=",",
                                ),
                            ],
                        )
                        .properties(height=320)
                    )

                    st.altair_chart(bar_chart, use_container_width=True)

                    st.caption(
                        "Bars are based on correlations across ZIP-year observations in the selected window. "
                        "These are associations, not causal effects: other neighborhood factors may also drive "
                        "home values."
                    )

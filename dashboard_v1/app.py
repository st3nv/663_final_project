import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Chicago Housing & Crime Lab", layout="wide")

# -----------------------------
# ZIP name dictionary
# -----------------------------
ZIP_NAME_MAP = {
    60601: "Lakeshore East / New Eastside",
    60602: "Central Loop",
    60603: "Financial District",
    60604: "South Loop (Historic Core)",
    60605: "South Loop / Museum Campus",
    60606: "West Loop (Financial Annex)",
    60607: "West Loop / UIC",
    60608: "Pilsen / Lower West Side",
    60609: "Back of the Yards / Fuller Park",
    60610: "Old Town / Near North Side",
    60611: "Streeterville / Magnificent Mile",
    60612: "Near West Side / Illinois Medical District",
    60613: "Lakeview / Wrigleyville",
    60614: "Lincoln Park",
    60615: "Hyde Park (North)",
    60616: "Chinatown / Near South Side",
    60617: "South Chicago / East Side",
    60618: "Avondale / Irving Park",
    60619: "Chatham / Avalon Park",
    60620: "Auburn Gresham",
    60621: "Englewood",
    60622: "Wicker Park / Ukrainian Village",
    60623: "Little Village",
    60624: "East Garfield Park",
    60625: "Lincoln Square / Ravenswood",
    60626: "Rogers Park",
    60628: "Roseland / Pullman",
    60629: "West Lawn / Chicago Lawn",
    60630: "Jefferson Park",
    60631: "Edison Park / Norwood Park",
    60632: "Brighton Park",
    60633: "Hegewisch",
    60634: "Belmont Cragin",
    60636: "West Englewood",
    60637: "Hyde Park (South) / Woodlawn",
    60638: "Garfield Ridge",
    60639: "Hermosa / Belmont Cragin (West)",
    60640: "Uptown",
    60641: "Irving Park / Portage Park",
    60642: "River West / Noble Square",
    60643: "Morgan Park / Beverly (North)",
    60644: "Austin (West)",
    60645: "West Ridge",
    60646: "Forest Glen / Edgebrook",
    60647: "Logan Square",
    60649: "South Shore",
    60651: "Humboldt Park / West Humboldt Park",
    60652: "Ashburn",
    60653: "Bronzeville",
    60654: "River North",
    60655: "Mount Greenwood",
    60656: "O'Hare / Norwood Park East",
    60657: "Lakeview East / Boystown",
    60659: "North Park / West Ridge (North)",
    60660: "Edgewater",
    60661: "Fulton River District",
}

# -----------------------------
# Data loading & preprocessing
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    # Geo
    geojson_path = "data/chicago_zipcodes.geojson"
    gdf = gpd.read_file(geojson_path)
    gdf["ZIP"] = gdf["ZIP"].astype(str)

    # Map ZipName: use GeoJSON column first, then dict, then fallback
    zip_int = pd.to_numeric(gdf["ZIP"], errors="coerce")
    mapped_names = zip_int.map(ZIP_NAME_MAP)
    if "ZipName" in gdf.columns:
        gdf["ZipName"] = gdf["ZipName"].fillna(mapped_names)
    else:
        gdf["ZipName"] = mapped_names
    gdf["ZipName"] = gdf["ZipName"].fillna("ZIP " + gdf["ZIP"])

    # Population (2021)
    pop_df = pd.read_csv("data/Chicago_Population_Counts.csv")
    pop_2021 = pop_df[pop_df["Year"] == 2021].copy()
    pop_2021["ZIP"] = pop_2021["Geography"].astype(str)
    gdf = gdf.merge(
        pop_2021[["ZIP", "Population - Total"]],
        on="ZIP",
        how="left",
    )
    gdf["Population - Total"] = gdf["Population - Total"].fillna(0.0)

    # Geometry helpers
    gdf["centroid"] = gdf.geometry.centroid
    gdf["centroid_lon"] = gdf["centroid"].x
    gdf["centroid_lat"] = gdf["centroid"].y

    # Approx distance to Loop center (Euclidean in degrees, just for ranking)
    loop_lat, loop_lon = 41.881832, -87.623177
    gdf["distance_from_loop"] = np.sqrt(
        (gdf["centroid_lat"] - loop_lat) ** 2
        + (gdf["centroid_lon"] - loop_lon) ** 2
    )

    # Crime yearly
    crime_df = pd.read_csv("data/chicago_crime_preprocessed.csv")
    crime_df["ZIP"] = crime_df["ZIP"].astype(str)
    crime_yearly = (
        crime_df.groupby(["ZIP", "year"])["crime_count"]
        .sum()
        .reset_index()
    )

    # Housing yearly (ZHVI)
    zhvi_df = pd.read_csv("data/chicago_zhvi_preprocessed.csv")
    zhvi_df["RegionName"] = zhvi_df["RegionName"].astype(str)
    zhvi_yearly = (
        zhvi_df.groupby(["RegionName", "Year"])["Zhvi"]
        .mean()
        .reset_index()
    )
    zhvi_yearly = zhvi_yearly.rename(
        columns={
            "RegionName": "ZIP",
            "Year": "year",
        }
    )

    # Model dataset: one row per (ZIP, year) where we have both crime & ZHVI
    model_df = crime_yearly.merge(
        zhvi_yearly,
        on=["ZIP", "year"],
        how="inner",
    )

    # Attach static ZIP attributes
    model_df = model_df.merge(
        gdf[["ZIP", "ZipName", "Population - Total", "distance_from_loop"]],
        on="ZIP",
        how="left",
    )
    model_df = model_df[model_df["Population - Total"] > 0].copy()

    model_df["crime_rate"] = (
        model_df["crime_count"] / model_df["Population - Total"] * 1000.0
    )
    model_df["log_zhvi"] = np.log(model_df["Zhvi"])

    # ZIP-level summary for radar & rankings
    summary = (
        model_df.groupby("ZIP")
        .agg(
            mean_zhvi=("Zhvi", "mean"),
            mean_crime_rate=("crime_rate", "mean"),
            population=("Population - Total", "first"),
            distance_from_loop=("distance_from_loop", "first"),
            ZipName=("ZipName", "first"),
        )
        .reset_index()
    )

    # Standardize for composite score
    feats = summary[["mean_zhvi", "mean_crime_rate", "population", "distance_from_loop"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats.fillna(0.0))
    summary["zhvi_z"] = scaled[:, 0]
    summary["crime_z"] = scaled[:, 1]
    summary["pop_z"] = scaled[:, 2]
    summary["dist_z"] = scaled[:, 3]

    # Higher ZHVI, higher population, closer to Loop, lower crime = better
    summary["composite_score"] = (
        summary["zhvi_z"]
        - summary["crime_z"]
        + 0.3 * summary["pop_z"]
        - 0.3 * summary["dist_z"]
    )

    return gdf, crime_yearly, zhvi_yearly, model_df, summary


# -----------------------------
# Helper: build geojson for map (drop centroid Point column)
# -----------------------------
@st.cache_data(show_spinner=False)
def build_geojson():
    gdf, _, _, _, _ = load_data()
    if "centroid" in gdf.columns:
        gdf_no_centroid = gdf.drop(columns=["centroid"])
    else:
        gdf_no_centroid = gdf
    geojson = json.loads(gdf_no_centroid.to_json())
    return geojson


# -----------------------------
# Overview Map
# -----------------------------
def render_overview_map():
    gdf, crime_yearly, zhvi_yearly, _, _ = load_data()
    geojson = build_geojson()

    st.markdown("## 🗺️ Overview Map of Chicago ZIP Codes")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        map_metric = st.selectbox(
            "Select variable to visualize on the map",
            ["Population (2021)", "Crime Rate", "Housing Price (ZHVI)"],
        )

    with col_right:
        st.markdown(
            "Fine-tune the year and see how the city evolves over time. "
            "Hover any area for details."
        )

    # ---- Population map (2021) ----
    if map_metric == "Population (2021)":
        year = 2021
        map_df = gdf.copy()
        color_col = "Population - Total"
        color_label = "Population (2021)"
        title = "Population Distribution (2021 Census)"

    # ---- Crime rate map ----
    elif map_metric == "Crime Rate":
        years = sorted(crime_yearly["year"].unique())
        year = st.slider(
            "Select year for crime data",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=int(years[-1]),
            step=1,
        )
        tmp = crime_yearly[crime_yearly["year"] == year].copy()
        tmp = tmp.merge(
            gdf[["ZIP", "ZipName", "Population - Total"]],
            on="ZIP",
            how="right",
        )
        tmp["crime_count"] = tmp["crime_count"].fillna(0)
        tmp["crime_rate"] = np.where(
            tmp["Population - Total"] > 0,
            tmp["crime_count"] / tmp["Population - Total"] * 1000.0,
            np.nan,
        )
        map_df = tmp
        color_col = "crime_rate"
        color_label = "Crime Rate (per 1,000 residents)"
        title = f"Crime Rate by ZIP – {year}"

    # ---- Housing (ZHVI) map ----
    else:  # Housing Price (ZHVI)
        years = sorted(zhvi_yearly["year"].unique())
        year = st.slider(
            "Select year for housing prices (ZHVI)",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=int(years[-1]),
            step=1,
        )
        tmp = zhvi_yearly[zhvi_yearly["year"] == year].copy()
        tmp = tmp.merge(
            gdf[["ZIP", "ZipName", "Population - Total"]],
            on="ZIP",
            how="right",
        )
        map_df = tmp
        color_col = "Zhvi"
        color_label = "ZHVI (Home Value Index)"
        title = f"Home Values by ZIP (ZHVI) – {year}"

    # ---- Hover info ----
    if "ZipName" not in map_df.columns:
        map_df = map_df.merge(
            gdf[["ZIP", "ZipName"]],
            on="ZIP",
            how="left",
        )

    hover_data = {"ZIP": True}
    if "ZipName" in map_df.columns:
        hover_data["ZipName"] = True
    if "Population - Total" in map_df.columns:
        hover_data["Population - Total"] = ":,"
    if color_col in map_df.columns:
        if color_col == "Zhvi":
            hover_data[color_col] = ":,.0f"
        else:
            hover_data[color_col] = ":.1f"

    fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="ZIP",
        featureidkey="properties.ZIP",
        color=color_col,
        hover_name="ZipName",
        hover_data=hover_data,
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        center={"lat": 41.85, "lon": -87.65},
        zoom=9.2,
        opacity=0.85,
    )
    fig.update_layout(
        margin=dict(r=0, t=40, l=0, b=0),
        height=620,
        title=title,
        coloraxis_colorbar=dict(title=color_label),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Quick stats ----
    st.markdown("### 🔍 Quick Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of ZIP codes", len(gdf))
    with col2:
        st.metric("Total population (2021)", f"{int(gdf['Population - Total'].sum()):,}")
    with col3:
        if map_metric == "Crime Rate":
            total_crimes = map_df["crime_count"].fillna(0).sum()
            st.metric(f"Total crimes in {year}", f"{int(total_crimes):,}")
        elif map_metric == "Housing Price (ZHVI)":
            mean_val = map_df["Zhvi"].mean()
            st.metric(f"Average ZHVI in {year}", f"${mean_val:,.0f}")
        else:
            mean_pop = gdf["Population - Total"].mean()
            st.metric("Average population per ZIP", f"{int(mean_pop):,}")


# -----------------------------
# ZIP Explorer – Bubble Time Series
# -----------------------------
def render_zip_explorer():
    gdf, crime_yearly, zhvi_yearly, _, _ = load_data()
    st.markdown("## 📍 ZIP Explorer – Bubble Time Series")

    zip_options = (
        gdf[["ZIP", "ZipName"]]
        .drop_duplicates()
        .sort_values("ZIP")
        .assign(label=lambda df: df["ZIP"] + " – " + df["ZipName"])
    )

    selected_label = st.selectbox(
        "Choose a ZIP code to explore",
        zip_options["label"],
        index=0,
    )
    selected_zip = selected_label.split(" – ")[0]

    pop = float(
        gdf.loc[gdf["ZIP"] == selected_zip, "Population - Total"].iloc[0]
        if (gdf["ZIP"] == selected_zip).any()
        else np.nan
    )

    crime_ts = crime_yearly[crime_yearly["ZIP"] == selected_zip].copy()
    housing_ts = zhvi_yearly[zhvi_yearly["ZIP"] == selected_zip].copy()

    merged_ts = pd.merge(
        crime_ts,
        housing_ts,
        on=["ZIP", "year"],
        how="outer",
    ).sort_values("year")

    if "crime_count" not in merged_ts.columns:
        merged_ts["crime_count"] = 0
    if "Zhvi" not in merged_ts.columns:
        merged_ts["Zhvi"] = np.nan

    merged_ts["crime_count"] = merged_ts["crime_count"].fillna(0)
    merged_ts["Zhvi"] = merged_ts["Zhvi"].astype(float)

    if pop > 0:
        merged_ts["crime_rate"] = merged_ts["crime_count"] / pop * 1000.0
    else:
        merged_ts["crime_rate"] = np.nan

    tabs = st.tabs(["Housing over time", "Crime over time", "Bubble comparison"])

    with tabs[0]:
        fig_h = px.scatter(
            merged_ts,
            x="year",
            y="Zhvi",
            size=np.maximum(merged_ts["Zhvi"], 1),
            size_max=40,
            trendline="ols",
            labels={"year": "Year", "Zhvi": "ZHVI (Home Value Index)"},
            title=f"Housing Values over Time – {selected_label}",
        )
        fig_h.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_h, use_container_width=True)

    with tabs[1]:
        fig_c = px.scatter(
            merged_ts,
            x="year",
            y="crime_rate",
            size=np.maximum(merged_ts["crime_count"], 1),
            size_max=40,
            trendline="ols",
            labels={"year": "Year", "crime_rate": "Crime Rate (per 1,000 residents)"},
            title=f"Crime Rate over Time – {selected_label}",
        )
        fig_c.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_c, use_container_width=True)

    with tabs[2]:
        comp_df = merged_ts.dropna(subset=["Zhvi", "crime_rate"]).copy()
        if not comp_df.empty:
            fig_b = px.scatter(
                comp_df,
                x="crime_rate",
                y="Zhvi",
                size=np.maximum(comp_df["Zhvi"], 1),
                size_max=45,
                color="year",
                color_continuous_scale="Plasma",
                labels={
                    "crime_rate": "Crime Rate (per 1,000 residents)",
                    "Zhvi": "ZHVI (Home Value Index)",
                    "year": "Year",
                },
                title=f"ZHVI vs Crime Rate over Time – {selected_label}",
            )
            fig_b.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("No overlapping crime & ZHVI data for this ZIP.")

    st.markdown("### Snapshot")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Population (2021)", f"{int(pop) if not np.isnan(pop) else 0:,}")
    with col2:
        if merged_ts["Zhvi"].notna().any():
            latest_row = merged_ts[merged_ts["Zhvi"].notna()].iloc[-1]
            st.metric(
                f"Latest ZHVI ({int(latest_row['year'])})",
                f"${latest_row['Zhvi']:,.0f}",
            )
        else:
            st.metric("Latest ZHVI", "N/A")
    with col3:
        if merged_ts["crime_rate"].notna().any():
            latest_row = merged_ts[merged_ts["crime_rate"].notna()].iloc[-1]
            st.metric(
                f"Latest Crime Rate ({int(latest_row['year'])})",
                f"{latest_row['crime_rate']:.1f} / 1k",
            )
        else:
            st.metric("Latest Crime Rate", "N/A")


# -----------------------------
# Radar & Rankings
# -----------------------------
def _z_to_score(z: float) -> float:
    return float(np.clip(5 + 2 * z, 0, 10))


def render_radar_rankings():
    _, _, _, _, summary = load_data()

    st.markdown("## 🧭 Multi-metric Radar & Rankings")

    min_pop = int(summary["population"].min())
    max_pop = int(summary["population"].max())
    pop_cut = st.slider(
        "Filter ZIPs by minimum population",
        min_value=min_pop,
        max_value=max_pop,
        value=min(min_pop + 5000, max_pop),
        step=1000,
    )
    filtered = summary[summary["population"] >= pop_cut].copy()
    if filtered.empty:
        st.warning("No ZIP codes above this population threshold.")
        return

    top_n = st.slider(
        "Show top N ZIP codes by composite score",
        min_value=5,
        max_value=min(30, len(filtered)),
        value=min(10, len(filtered)),
        step=1,
    )

    ranked = filtered.sort_values("composite_score", ascending=False).head(top_n)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_bar = px.bar(
            ranked,
            x="composite_score",
            y="ZipName",
            orientation="h",
            color="composite_score",
            color_continuous_scale="Tealgrn",
            labels={"composite_score": "Composite Score", "ZipName": ""},
            title="Top ZIPs by Composite Score (Higher = Better)",
        )
        fig_bar.update_layout(
            template="plotly_white",
            height=520,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**How is the score constructed?**")
        st.markdown(
            "- Higher home values (ZHVI) ↑\n"
            "- Lower crime rate ↑\n"
            "- Higher population (market size) ↑\n"
            "- Closer to the Loop (centrality) ↑\n"
            "All features are standardized, then combined into one composite index."
        )

    st.markdown("### 🕸 Radar View for a Single ZIP")
    options = (
        ranked[["ZIP", "ZipName"]]
        .assign(label=lambda df: df["ZIP"] + " – " + df["ZipName"])
        .sort_values("label")
    )
    selected = st.selectbox(
        "Select a ZIP from the ranked list",
        options["label"],
    )
    sel_zip = selected.split(" – ")[0]
    row = ranked[ranked["ZIP"] == sel_zip].iloc[0]

    categories = [
        "Housing Value",
        "Crime Safety",
        "Population Size",
        "Proximity to Loop",
    ]
    values = [
        _z_to_score(row["zhvi_z"]),
        _z_to_score(-row["crime_z"]),
        _z_to_score(row["pop_z"]),
        _z_to_score(-row["dist_z"]),
    ]
    values.append(values[0])
    categories_closed = categories + [categories[0]]

    fig_rad = go.Figure()
    fig_rad.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill="toself",
            name=selected,
            line=dict(color="#2563eb"),
        )
    )
    fig_rad.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10]),
        ),
        showlegend=False,
        height=420,
        title="Multi-dimensional Profile",
    )
    st.plotly_chart(fig_rad, use_container_width=True)


# -----------------------------
# Creative Bubble Lab
# -----------------------------
def render_creative_bubbles():
    _, _, _, _, summary = load_data()
    st.markdown("## 🫧 Creative Bubble Lab – City as a Bubble Galaxy")

    bubble_df = summary.copy()
    bubble_df["bubble_size"] = np.maximum(bubble_df["population"], 1)
    bubble_df["safety_score"] = -bubble_df["crime_z"]

    fig = px.scatter(
        bubble_df,
        x="distance_from_loop",
        y="mean_zhvi",
        size="bubble_size",
        color="safety_score",
        hover_name="ZipName",
        hover_data={
            "ZIP": True,
            "mean_zhvi": ":,.0f",
            "mean_crime_rate": ":.1f",
            "population": ":,",
            "distance_from_loop": ":.3f",
        },
        size_max=50,
        color_continuous_scale="RdYlGn",
        labels={
            "distance_from_loop": "Distance from Loop (relative)",
            "mean_zhvi": "Average ZHVI",
            "safety_score": "Safety (higher = better)",
        },
        title="Each ZIP as a Bubble – Value, Safety & Location Combined",
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Interpretation:\n"
        "- **Bigger bubbles** → more population.\n"
        "- **Greener bubbles** → safer (lower standardized crime).\n"
        "- **Higher bubbles** → higher average home values.\n"
        "- **Left side** → closer to downtown."
    )


# -----------------------------
# Statistical Analysis – Linear Models (enriched)
# -----------------------------
def render_statistical_analysis():
    _, _, _, model_df, summary = load_data()

    st.markdown("## 📊 Statistical Analysis – Linking Crime, Housing & Location")

    # ---------- 1. Pooled linear model ----------
    st.markdown("### 1. Baseline Model: Does Crime Rate Predict Home Values?")

    model_use = model_df.dropna(subset=["crime_rate", "log_zhvi", "distance_from_loop"]).copy()
    if model_use.empty:
        st.warning("Not enough overlapping crime & ZHVI data to fit a model.")
        return

    # 基准模型：log(ZHVI) ~ crime_rate + distance_from_loop
    X = model_use[["crime_rate", "distance_from_loop"]]
    X = sm.add_constant(X)
    y = model_use["log_zhvi"]

    ols_model = sm.OLS(y, X).fit()

    params = ols_model.params.rename("coef")
    bse = ols_model.bse.rename("std_err")
    pvals = ols_model.pvalues.rename("p_value")
    coef_df = pd.concat([params, bse, pvals], axis=1)
    coef_df.index.name = "term"

    st.write("**Model specification:**")
    st.latex(
        r"\log(\text{ZHVI}_{it}) = \beta_0 + \beta_1 \cdot \text{CrimeRate}_{it} + "
        r"\beta_2 \cdot \text{DistanceFromLoop}_i + \varepsilon_{it}"
    )

    st.write("**Estimated coefficients:**")
    st.dataframe(
        coef_df.style.format({"coef": "{:.3f}", "std_err": "{:.3f}", "p_value": "{:.3g}"}),
        use_container_width=True,
    )

    st.markdown(
        f"- R² = **{ols_model.rsquared:.3f}**,  Adjusted R² = **{ols_model.rsquared_adj:.3f}**  \n"
        "- Negative coefficient on crime rate ⇒ higher crime associated with lower home values "
        "(conditional on distance)."
    )

    # Predicted vs Actual
    model_use["resid"] = ols_model.resid
    pred = ols_model.fittedvalues
    res_df = pd.DataFrame(
        {
            "fitted_log_zhvi": pred,
            "actual_log_zhvi": y,
        }
    )

    fig_scatter = px.scatter(
        res_df,
        x="fitted_log_zhvi",
        y="actual_log_zhvi",
        trendline="ols",
        labels={
            "fitted_log_zhvi": "Predicted log(ZHVI)",
            "actual_log_zhvi": "Actual log(ZHVI)",
        },
        title="Predicted vs Actual log(ZHVI)",
    )
    fig_scatter.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---------- 2. Extended model with time trend ----------
    st.markdown("### 2. Extended Model: Adding Time Trend")

    X2 = model_use[["crime_rate", "distance_from_loop", "year"]].copy()
    X2 = sm.add_constant(X2)
    y2 = model_use["log_zhvi"]
    extended_model = sm.OLS(y2, X2).fit()

    params2 = extended_model.params.rename("coef")
    bse2 = extended_model.bse.rename("std_err")
    pvals2 = extended_model.pvalues.rename("p_value")
    coef_df2 = pd.concat([params2, bse2, pvals2], axis=1)
    coef_df2.index.name = "term"

    st.write("**Extended model specification:**")
    st.latex(
        r"\log(\text{ZHVI}_{it}) = \beta_0 + \beta_1 \cdot \text{CrimeRate}_{it} + "
        r"\beta_2 \cdot \text{DistanceFromLoop}_i + \beta_3 \cdot \text{Year}_t + \varepsilon_{it}"
    )

    with st.expander("See extended model coefficients"):
        st.dataframe(
            coef_df2.style.format({"coef": "{:.3f}", "std_err": "{:.3f}", "p_value": "{:.3g}"}),
            use_container_width=True,
        )
        st.markdown(
            f"- Extended model R² = **{extended_model.rsquared:.3f}**, "
            f"Adjusted R² = **{extended_model.rsquared_adj:.3f}**"
        )

    # ---------- 3. Yearly correlation ----------
    st.markdown("### 3. Yearly Correlation between Crime & Home Values")

    corr_list = []
    for year in sorted(model_use["year"].unique()):
        df_y = model_use[model_use["year"] == year]
        if df_y["crime_rate"].var() > 0 and df_y["Zhvi"].var() > 0:
            corr = df_y["crime_rate"].corr(df_y["Zhvi"])
            corr_list.append({"year": year, "corr": corr})

    corr_df = pd.DataFrame(corr_list)
    if not corr_df.empty:
        fig_corr = px.line(
            corr_df,
            x="year",
            y="corr",
            markers=True,
            labels={"year": "Year", "corr": "Correlation (crime rate vs ZHVI)"},
            title="Correlation between Crime Rate and Home Values Over Time",
        )
        fig_corr.add_hline(y=0, line_dash="dash")
        fig_corr.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough data per year to compute meaningful correlations.")

    # ---------- 4. Cross-sectional extremes ----------
    st.markdown("### 4. Cross-sectional View: Lowest vs Highest Crime ZIPs")

    extremes = summary.sort_values("mean_crime_rate")
    low_crime = extremes.head(5)
    high_crime = extremes.tail(5)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Lowest average crime rate ZIPs**")
        st.dataframe(
            low_crime[["ZIP", "ZipName", "mean_crime_rate", "mean_zhvi", "population"]]
            .rename(
                columns={
                    "mean_crime_rate": "Crime rate",
                    "mean_zhvi": "ZHVI",
                    "population": "Population",
                }
            )
            .style.format({"Crime rate": "{:.1f}", "ZHVI": "{:,.0f}", "Population": "{:,}"}),
            use_container_width=True,
        )
    with col2:
        st.write("**Highest average crime rate ZIPs**")
        st.dataframe(
            high_crime[["ZIP", "ZipName", "mean_crime_rate", "mean_zhvi", "population"]]
            .rename(
                columns={
                    "mean_crime_rate": "Crime rate",
                    "mean_zhvi": "ZHVI",
                    "population": "Population",
                }
            )
            .style.format({"Crime rate": "{:.1f}", "ZHVI": "{:,.0f}", "Population": "{:,}"}),
            use_container_width=True,
        )

    # ---------- 5. Residual analysis ----------
    st.markdown("### 5. Residual Analysis: Over- and Under-valued ZIPs")

    zip_resid = (
        model_use.groupby("ZIP")
        .agg(
            mean_resid=("resid", "mean"),
            mean_zhvi=("Zhvi", "mean"),
            mean_crime_rate=("crime_rate", "mean"),
            population=("Population - Total", "first"),
            ZipName=("ZipName", "first"),
        )
        .reset_index()
    )

    top_over = zip_resid.sort_values("mean_resid", ascending=False).head(10)
    top_under = zip_resid.sort_values("mean_resid", ascending=True).head(10)

    col3, col4 = st.columns(2)
    with col3:
        st.write("**ZIPs with positive residuals (actual prices > model prediction)**")
        fig_over = px.bar(
            top_over.sort_values("mean_resid"),
            x="mean_resid",
            y="ZipName",
            orientation="h",
            labels={"mean_resid": "Average residual (log ZHVI)", "ZipName": ""},
            title="Top 10 Over-performing ZIPs",
        )
        fig_over.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_over, use_container_width=True)

    with col4:
        st.write("**ZIPs with negative residuals (actual prices < model prediction)**")
        fig_under = px.bar(
            top_under.sort_values("mean_resid"),
            x="mean_resid",
            y="ZipName",
            orientation="h",
            labels={"mean_resid": "Average residual (log ZHVI)", "ZipName": ""},
            title="Top 10 Under-performing ZIPs",
        )
        fig_under.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_under, use_container_width=True)

    st.info(
        "Residuals measure how much actual log(ZHVI) deviates from the level predicted "
        "by crime and distance from the Loop. Positive residuals ⇒ relatively stronger "
        "housing markets than fundamentals alone would suggest."
    )

    # ---------- 6. Year-by-year crime slope ----------
    st.markdown("### 6. Year-by-year Effect of Crime on Home Values")

    slope_list = []
    for year in sorted(model_use["year"].unique()):
        df_y = model_use[model_use["year"] == year]
        if df_y["crime_rate"].var() > 0 and df_y["log_zhvi"].var() > 0 and len(df_y) > 5:
            X_y = df_y[["crime_rate", "distance_from_loop"]]
            X_y = sm.add_constant(X_y)
            y_y = df_y["log_zhvi"]
            try:
                m_y = sm.OLS(y_y, X_y).fit()
                slope_list.append(
                    {
                        "year": year,
                        "beta_crime": m_y.params.get("crime_rate", np.nan),
                        "se_beta": m_y.bse.get("crime_rate", np.nan),
                        "p_value": m_y.pvalues.get("crime_rate", np.nan),
                        "r2": m_y.rsquared,
                    }
                )
            except Exception:
                continue

    slope_df = pd.DataFrame(slope_list)
    if not slope_df.empty:
        fig_slope = px.line(
            slope_df,
            x="year",
            y="beta_crime",
            markers=True,
            labels={
                "year": "Year",
                "beta_crime": "Coefficient on crime_rate (log ZHVI model)",
            },
            title="Year-by-year Crime Effect on Home Values (controlling for distance)",
        )
        fig_slope.add_hline(y=0, line_dash="dash")
        fig_slope.update_layout(template="plotly_white", height=420)
        st.plotly_chart(fig_slope, use_container_width=True)

        st.dataframe(
            slope_df[["year", "beta_crime", "se_beta", "p_value", "r2"]]
            .sort_values("year")
            .style.format(
                {
                    "beta_crime": "{:.3f}",
                    "se_beta": "{:.3f}",
                    "p_value": "{:.3g}",
                    "r2": "{:.3f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("Not enough yearly data to estimate year-by-year slopes robustly.")

    # ---------- 7. Correlation matrix ----------
    st.markdown("### 7. ZIP-level Correlation Matrix of Key Features")

    corr_base = summary.copy()
    corr_base["log_mean_zhvi"] = np.log(corr_base["mean_zhvi"])

    corr_vars = ["log_mean_zhvi", "mean_crime_rate", "population", "distance_from_loop"]
    corr_matrix = corr_base[corr_vars].corr()

    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
        title="Correlation Matrix (ZIP-level averages)",
    )
    fig_heat.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------
# Model Residual Map – residual + composite score toggle
# -----------------------------
def render_model_residual_map():
    gdf, _, _, model_df, summary = load_data()
    geojson = build_geojson()

    st.markdown("## 🗺️ Model Residual & Composite Score Map")

    # --- 1. 拟合 pooled 线性模型，拿到 residual ---
    model_use = model_df.dropna(subset=["crime_rate", "log_zhvi", "distance_from_loop"]).copy()
    if model_use.empty:
        st.warning("Not enough data to estimate the residual model.")
        return

    X = model_use[["crime_rate", "distance_from_loop"]]
    X = sm.add_constant(X)
    y = model_use["log_zhvi"]
    ols_model = sm.OLS(y, X).fit()
    model_use["resid"] = ols_model.resid

    # 每个 ZIP 的平均残差
    zip_resid = (
        model_use.groupby("ZIP")
        .agg(
            mean_resid=("resid", "mean"),
            mean_zhvi=("Zhvi", "mean"),
            mean_crime_rate=("crime_rate", "mean"),
        )
        .reset_index()
    )

    # 和 summary（composite_score 等）合并
    # 注意：summary 里列名是 population，不是 "Population - Total"
    zip_metrics = summary.merge(zip_resid[["ZIP", "mean_resid"]], on="ZIP", how="left")

    # 把这些指标挂到几何数据上
    map_df = gdf.merge(
        zip_metrics[["ZIP", "ZipName", "composite_score", "mean_resid"]],
        on="ZIP",
        how="left",
    )

    # 处理 ZipName_x / ZipName_y 的情况，统一成 ZipName
    if "ZipName_x" in map_df.columns and "ZipName_y" in map_df.columns:
        map_df["ZipName"] = map_df["ZipName_x"].fillna(map_df["ZipName_y"])
    elif "ZipName_x" in map_df.columns:
        map_df["ZipName"] = map_df["ZipName_x"]
    elif "ZipName_y" in map_df.columns:
        map_df["ZipName"] = map_df["ZipName_y"]
    # 如果只存在 gdf 原来的 ZipName，前面 merge 不会产生 _x/_y，直接用已有的即可

    # --- 2. UI toggle：切换主地图指标 ---
    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        primary_metric = st.radio(
            "Primary map metric",
            ["Model residual (log ZHVI)", "Composite score"],
            index=0,
            horizontal=True,
        )
    with col_top2:
        st.markdown(
            "Use the toggle to **switch the main map** between residuals and composite "
            "scores. The secondary map shows the other metric for visual comparison."
        )

    if primary_metric == "Model residual (log ZHVI)":
        main_col = "mean_resid"
        main_label = "Avg residual (log ZHVI)"
        main_scale = "RdBu_r"
        secondary_col = "composite_score"
        secondary_label = "Composite score"
        secondary_scale = "Viridis"
    else:
        main_col = "composite_score"
        main_label = "Composite score"
        main_scale = "Viridis"
        secondary_col = "mean_resid"
        secondary_label = "Avg residual (log ZHVI)"
        secondary_scale = "RdBu_r"

    # 残差地图用对称范围
    max_abs_resid = np.nanmax(np.abs(map_df["mean_resid"]))
    if not np.isfinite(max_abs_resid) or max_abs_resid == 0:
        max_abs_resid = 0.1

    hover_data = {
        "ZIP": True,
        "ZipName": True,
        "Population - Total": ":,",     # 来自 gdf
        "composite_score": ":.2f",
        "mean_resid": ":.3f",
    }

    col_map1, col_map2 = st.columns(2)

    # --- 主地图 ---
    with col_map1:
        if main_col == "mean_resid":
            range_color = (-max_abs_resid, max_abs_resid)
        else:
            range_color = None

        fig_main = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="ZIP",
            featureidkey="properties.ZIP",
            color=main_col,
            hover_name="ZipName",
            hover_data=hover_data,
            color_continuous_scale=main_scale,
            range_color=range_color,
            mapbox_style="carto-positron",
            center={"lat": 41.85, "lon": -87.65},
            zoom=9.2,
            opacity=0.85,
        )
        fig_main.update_layout(
            margin=dict(r=0, t=40, l=0, b=0),
            height=620,
            title=f"Primary Map – {main_label}",
            coloraxis_colorbar=dict(title=main_label),
            template="plotly_white",
        )
        st.plotly_chart(fig_main, use_container_width=True)

    # --- 副地图 ---
    with col_map2:
        if secondary_col == "mean_resid":
            range_color_sec = (-max_abs_resid, max_abs_resid)
        else:
            range_color_sec = None

        fig_sec = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="ZIP",
            featureidkey="properties.ZIP",
            color=secondary_col,
            hover_name="ZipName",
            hover_data=hover_data,
            color_continuous_scale=secondary_scale,
            range_color=range_color_sec,
            mapbox_style="carto-positron",
            center={"lat": 41.85, "lon": -87.65},
            zoom=9.2,
            opacity=0.85,
        )
        fig_sec.update_layout(
            margin=dict(r=0, t=40, l=0, b=0),
            height=620,
            title=f"Secondary Map – {secondary_label}",
            coloraxis_colorbar=dict(title=secondary_label),
            template="plotly_white",
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    # --- residual vs composite score 散点图 ---
    st.markdown("### Residual vs Composite Score")

    scatter_df = zip_metrics.copy()
    scatter_df = scatter_df.dropna(subset=["composite_score", "mean_resid"])

    if not scatter_df.empty:
        fig_sc = px.scatter(
            scatter_df,
            x="composite_score",
            y="mean_resid",
            hover_name="ZipName",
            hover_data={
                "ZIP": True,
                "composite_score": ":.2f",
                "mean_resid": ":.3f",
                "mean_zhvi": ":,.0f",
                "mean_crime_rate": ":.1f",
                "population": ":,",
            },
            trendline="ols",
            labels={
                "composite_score": "Composite score (higher = better)",
                "mean_resid": "Avg residual (log ZHVI)",
            },
            title="Model Residual vs Composite Score (ZIP-level)",
        )
        fig_sc.update_layout(template="plotly_white", height=460)
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Not enough data to compute residual vs composite scatter.")

    st.info(
        "- **Composite score** combines value, crime, population and location.\n"
        "- **Residual** tells you whether prices are higher or lower than the model "
        "predicts, *after* controlling for crime and distance.\n"
        "You can look for ZIPs with **high composite score + positive residual** as "
        "“strong fundamentals + market enthusiasm”."
    )


# -----------------------------
# Main
# -----------------------------
def main():
    st.sidebar.title("Chicago Housing & Crime Dashboard")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview Map",
            "ZIP Explorer",
            "Radar & Rankings",
            "Model Residual Map",
            "Creative Bubble Lab",
            "Statistical Analysis",
        ],
    )

    if page == "Overview Map":
        render_overview_map()
    elif page == "ZIP Explorer":
        render_zip_explorer()
    elif page == "Radar & Rankings":
        render_radar_rankings()
    elif page == "Model Residual Map":
        render_model_residual_map()
    elif page == "Creative Bubble Lab":
        render_creative_bubbles()
    elif page == "Statistical Analysis":
        render_statistical_analysis()


if __name__ == "__main__":
    main()



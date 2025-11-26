import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Chicago ZIP Visual Playground", layout="wide")


# ============ 工具函数 ============

def normalize_series(s: pd.Series, reverse: bool = False) -> pd.Series:
    s = s.astype(float)
    if s.empty:
        return s
    mn = s.min()
    mx = s.max()
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    scaled = (s - mn) / (mx - mn)
    return 1.0 - scaled if reverse else scaled


# ============ 读数据 & 预处理 ============

def load_data():
    # --- Geo（含 ZipName） ---
    geojson_path = "data/chicago_zipcodes.geojson"
    gdf = gpd.read_file(geojson_path)
    gdf["ZIP"] = gdf["ZIP"].astype(str)

    # --- Population ---
    pop_df = pd.read_csv("data/Chicago_Population_Counts.csv")
    pop_2021 = pop_df[pop_df["Year"] == 2021].copy()
    pop_2021["ZIP"] = pop_2021["Geography"].astype(str)
    gdf = gdf.merge(
        pop_2021[["ZIP", "Population - Total"]],
        on="ZIP",
        how="left",
    )
    gdf["Population - Total"] = gdf["Population - Total"].fillna(0)

    # --- Crime ---
    crime_df = pd.read_csv("data/chicago_crime_preprocessed.csv")
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

    # --- Housing (ZHVI) ---
    zhvi_df = pd.read_csv("data/chicago_zhvi_preprocessed.csv")
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

    # --- Centroid 列（注意：后面 to_json 不会把它带进去） ---
    gdf["centroid"] = gdf.geometry.centroid

    # --- ZipName 映射 ---
    zip_name_map = {}
    if "ZipName" in gdf.columns:
        zip_name_map = (
            gdf[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )

    # --- map_data: 用于各种时间序列玩具 ---
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

    # --- zip_df: 最新年份的综合指标 ---
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
                "centroid_lat": z["centroid_lat"],
                "centroid_lng": z["centroid_lng"],
            }
        )
    zip_df = pd.DataFrame(zip_metrics)

    # --- 距离 Loop，用于 Orbit & 聚类 ---
    loop_center = Point(-87.6298, 41.8781)
    gdf["distance_from_loop"] = gdf["centroid"].distance(loop_center)

    # --- Bar chart race 数据 ---
    race_zhvi_records = []
    race_crime_records = []
    for z in map_data:
        for y_str, val in z["zhvi"].items():
            year = int(y_str)
            if val <= 0:
                continue
            race_zhvi_records.append({"ZIP": z["zip"], "Year": year, "Zhvi": val})

        for y_str, crimes in z["crimes"].items():
            year = int(y_str)
            pop = z["population"]
            rate = crimes / pop * 1000 if pop > 0 else 0.0
            race_crime_records.append(
                {"ZIP": z["zip"], "Year": year, "CrimeRate": rate}
            )

    race_zhvi_df = pd.DataFrame(race_zhvi_records)
    race_crime_df = pd.DataFrame(race_crime_records)

    centroids = gdf["centroid"]
    center_lat = centroids.y.mean()
    center_lon = centroids.x.mean()

    return (
        gdf,
        pop_df,
        crime_df,
        zhvi_df,
        map_data,
        zip_df,
        crime_min_year,
        crime_max_year,
        zhvi_min_year,
        zhvi_max_year,
        race_zhvi_df,
        race_crime_df,
        center_lat,
        center_lon,
    )


# ===== 读数据 =====
(
    gdf,
    pop_df,
    crime_df,
    zhvi_df,
    map_data,
    zip_df,
    crime_min_year,
    crime_max_year,
    zhvi_min_year,
    zhvi_max_year,
    race_zhvi_df,
    race_crime_df,
    center_lat,
    center_lon,
) = load_data()

# ===== 打分（Population / Safety / Value / Overall） =====
if not zip_df.empty:
    zip_df["pop_score"] = normalize_series(zip_df["population"], reverse=False)
    zip_df["safety_score"] = normalize_series(zip_df["crime_rate_latest"], reverse=True)
    zip_df["value_score"] = normalize_series(zip_df["zhvi_latest"], reverse=False)
    zip_df["overall_score"] = (
        zip_df["pop_score"] + zip_df["safety_score"] + zip_df["value_score"]
    ) / 3.0

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

# ===== 邻接表（Network 用） =====
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


# ============ Relationship Lab 辅助函数 ============

def build_traj_df(
    map_data, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year
):
    records = []
    for z in map_data:
        zh_earliest = z["zhvi"].get(str(zhvi_min_year), 0.0)
        zh_latest = z["zhvi"].get(str(zhvi_max_year), 0.0)
        if zh_earliest <= 0 or zh_latest <= 0:
            continue
        c_earliest = z["crimes"].get(str(crime_min_year), 0.0)
        c_latest = z["crimes"].get(str(crime_max_year), 0.0)
        pop = z["population"]
        if pop <= 0:
            continue
        cr_earliest = c_earliest / pop * 1000
        cr_latest = c_latest / pop * 1000
        records.append(
            {
                "ZIP": z["zip"],
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
    df["crime_change"] = df["crime_rate_latest"] - df["crime_rate_earliest"]
    return df


def build_cluster_df(zip_df, gdf, n_clusters: int = 4):
    tmp = gdf[["ZIP", "distance_from_loop"]].merge(
        zip_df[["ZIP", "population", "crime_rate_latest", "zhvi_latest"]],
        on="ZIP",
        how="inner",
    )
    tmp = tmp.dropna(subset=["population", "crime_rate_latest", "zhvi_latest"])
    if tmp.empty:
        return tmp

    features = tmp[
        ["population", "crime_rate_latest", "zhvi_latest", "distance_from_loop"]
    ].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    if n_clusters < 2:
        n_clusters = 2
    if n_clusters > len(tmp):
        n_clusters = max(2, len(tmp) // 2)

    # 修复：使用数值 n_init 而不是 "auto"，提高兼容性
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tmp["cluster"] = km.fit_predict(X)
    return tmp


# ============ 各个视图 ============

def render_overview_map():
    st.markdown("### 🗺️ Overview Map · 总览地图")
    variable = st.selectbox(
        "Map metric / 地图指标",
        ["Population (2021)", "Crime Count", "Crime Rate", "Home Value (ZHVI)"],
    )

    year = None
    if variable in ["Crime Count", "Crime Rate"]:
        year = st.slider(
            "Crime year / 犯罪年份",
            min_value=crime_min_year,
            max_value=crime_max_year,
            value=crime_max_year,
        )
    elif variable == "Home Value (ZHVI)":
        year = st.slider(
            "ZHVI year / 房价年份",
            min_value=zhvi_min_year,
            max_value=zhvi_max_year,
            value=zhvi_max_year,
        )

    value_records = []
    for z in map_data:
        val = 0.0
        if variable == "Population (2021)":
            val = z["population"]
        elif variable == "Crime Count":
            val = z["crimes"].get(str(year), 0.0)
        elif variable == "Crime Rate":
            crimes = z["crimes"].get(str(year), 0.0)
            pop = z["population"]
            val = crimes / pop * 1000 if pop > 0 else 0.0
        elif variable == "Home Value (ZHVI)":
            val = z["zhvi"].get(str(year), 0.0)
        value_records.append({"ZIP": z["zip"], "value": val})

    value_df = pd.DataFrame(value_records)
    map_df = gdf.merge(value_df, on="ZIP", how="left").copy()
    map_df = map_df[map_df["value"].notna()].copy()

    # Label：ZIP + 名称
    if "ZipName" in map_df.columns:
        map_df["Label"] = map_df.apply(
            lambda r: f"{r['ZIP']} – {r['ZipName']}"
            if isinstance(r["ZipName"], str) and r["ZipName"]
            else r["ZIP"],
            axis=1,
        )
    else:
        map_df["Label"] = map_df["ZIP"]

    # *** 关键修复：只用 ["ZIP", "geometry"] 生成 geojson，避免 Point 序列化错误 ***
    geojson = json.loads(map_df[["ZIP", "geometry"]].to_json())

    color_label = {
        "Population (2021)": "Population",
        "Crime Count": f"Crime Count ({year})" if year is not None else "Crime Count",
        "Crime Rate": f"Crime Rate ({year}) per 1000" if year is not None else "Crime Rate per 1000",
        "Home Value (ZHVI)": f"ZHVI ({year})" if year is not None else "ZHVI",
    }[variable]

    color_scale = "YlGnBu" if variable in [
        "Population (2021)",
        "Home Value (ZHVI)",
    ] else "YlOrRd"

    # 修复：hover_data 动态构造，防止 ZipName 不存在时报错
    hover_data = {"ZIP": True, "value": ":,.0f"}
    if "ZipName" in map_df.columns:
        hover_data["ZipName"] = True

    fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="ZIP",
        featureidkey="properties.ZIP",
        color="value",
        color_continuous_scale=color_scale,
        mapbox_style="carto-positron",
        center={"lat": center_lat, "lon": center_lon},
        zoom=9.5,
        opacity=0.8,
        hover_name="Label",
        hover_data=hover_data,
        labels={"value": color_label},
    )
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title=color_label,
            thickness=14,
            len=0.6,
        ),
        paper_bgcolor="#f9fafb",
        mapbox_accesstoken=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_value_safety_lab():
    st.subheader("💙 Value vs Safety · 房价-安全关系")

    df = zip_df.copy()
    df = df[(df["zhvi_latest"] > 0) & (df["crime_rate_latest"] > 0)].copy()
    if df.empty:
        st.warning("No valid ZIPs with both crime rate and home value.")
        return

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
        elif z >= 0.5:
            return "Overvalued (expensive vs crime)"
        else:
            return "On-model"

    df["value_class"] = df["resid_z"].apply(classify_resid)

    # Label
    if "ZipName" in df.columns:
        df["Label"] = df.apply(
            lambda r: f"{r['ZIP']} – {r['ZipName']}"
            if isinstance(r["ZipName"], str) and r["ZipName"]
            else r["ZIP"],
            axis=1,
        )
    else:
        df["Label"] = df["ZIP"]

    col1, col2 = st.columns([1.3, 1])

    # 散点 + 回归线
    with col1:
        fig_scatter = px.scatter(
            df,
            x="crime_rate_latest",
            y="zhvi_latest",
            color="value_class",
            hover_name="Label",
            size="population",
            labels={
                "crime_rate_latest": "Crime Rate (per 1000 residents)",
                "zhvi_latest": "Home Value (latest ZHVI, USD)",
                "value_class": "Value Category",
            },
            title="Home Value vs Crime Rate",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        x_line = np.linspace(df["crime_rate_latest"].min(), df["crime_rate_latest"].max(), 100)
        y_line = a + b * x_line
        fig_scatter.add_traces(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regression line",
                line=dict(dash="dash", color="black"),
            )
        )
        fig_scatter.update_layout(
            template="simple_white",
            height=520,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption(
            "回归线刻画在城市整体水平下，给定犯罪率时房价的“期望”；"
            "散点颜色表示实际房价相对这条线是偏便宜还是偏贵。"
        )

    # 残差地图
    with col2:
        map_df = gdf.merge(
            df[["ZIP", "zhvi_resid"]],
            on="ZIP",
            how="left",
        )
        map_df = map_df[map_df["zhvi_resid"].notna()].copy()
        if map_df.empty:
            st.info("No residuals to map.")
            return

        geojson = json.loads(map_df[["ZIP", "geometry"]].to_json())

        fig_map = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="ZIP",
            featureidkey="properties.ZIP",
            color="zhvi_resid",
            color_continuous_scale="RdBu_r",
            mapbox_style="carto-positron",
            center={"lat": center_lat, "lon": center_lon},
            zoom=9.5,
            opacity=0.7,
            hover_name="ZIP",
            hover_data={"zhvi_resid": ":,.0f"},
            labels={"zhvi_resid": "Price residual (actual - predicted)"},
        )
        fig_map.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Residual"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            "蓝色区域：相对给定的犯罪率，房价偏便宜；红色区域：房价偏贵。"
        )


def render_trajectory_lab():
    st.subheader("📈 Trajectories · 安全 & 房价变化轨迹")

    traj_df = build_traj_df(
        map_data, crime_min_year, crime_max_year, zhvi_min_year, zhvi_max_year
    )
    if traj_df.empty:
        st.warning("No trajectory data available.")
        return

    # 加名字
    if "ZipName" in gdf.columns:
        name_map = (
            gdf[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )
        traj_df["Label"] = traj_df["ZIP"].apply(
            lambda z: f"{z} – {name_map.get(z, '')}" if name_map.get(z, "") else z
        )
    else:
        traj_df["Label"] = traj_df["ZIP"]

    fig_traj = px.scatter(
        traj_df,
        x="crime_change",
        y="zhvi_change_pct",
        hover_name="Label",
        labels={
            "crime_change": "Δ Crime Rate (per 1000 residents)",
            "zhvi_change_pct": "Home Value Change (%)",
        },
        title=f"Change from {crime_min_year}/{zhvi_min_year} to {crime_max_year}/{zhvi_max_year}",
        color_discrete_sequence=["#0ea5e9"],
    )
    fig_traj.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_traj.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_traj.update_layout(
        template="simple_white",
        height=520,
    )
    st.plotly_chart(fig_traj, use_container_width=True)
    st.caption(
        "右上角象限：房价上涨且犯罪率上升；左上角：房价上涨但治安改善；"
        "右下角：房价下跌且治安恶化；左下角：房价下跌但治安改善。"
    )


def render_cluster_lab():
    st.subheader("🧩 Neighborhood Archetypes · 社区原型聚类")
    if zip_df.empty:
        st.warning("No ZIP data for clustering.")
        return

    k = st.slider("Number of clusters / 聚类数", 3, 7, 4)
    cluster_df = build_cluster_df(zip_df, gdf, n_clusters=k)
    if cluster_df.empty:
        st.warning("No valid ZIPs for clustering.")
        return

    # Label
    if "ZipName" in gdf.columns:
        name_map = (
            gdf[["ZIP", "ZipName"]]
            .drop_duplicates()
            .set_index("ZIP")["ZipName"]
            .to_dict()
        )
        cluster_df["Label"] = cluster_df["ZIP"].apply(
            lambda z: f"{z} – {name_map.get(z, '')}" if name_map.get(z, "") else z
        )
    else:
        cluster_df["Label"] = cluster_df["ZIP"]

    fig_scatter = px.scatter(
        cluster_df,
        x="crime_rate_latest",
        y="zhvi_latest",
        color="cluster",
        hover_name="Label",
        size="population",
        labels={
            "crime_rate_latest": "Crime Rate (per 1000)",
            "zhvi_latest": "Home Value (ZHVI)",
            "cluster": "Cluster",
        },
        title="Clusters in Safety-Value Space",
        color_continuous_scale="Viridis",
    )
    fig_scatter.update_layout(
        template="simple_white",
        height=520,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    map_df = gdf.merge(cluster_df[["ZIP", "cluster"]], on="ZIP", how="inner")
    geojson = json.loads(map_df[["ZIP", "geometry"]].to_json())
    fig_map = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="ZIP",
        featureidkey="properties.ZIP",
        color="cluster",
        mapbox_style="carto-positron",
        center={"lat": center_lat, "lon": center_lon},
        zoom=9.5,
        opacity=0.75,
        hover_name="ZIP",
        labels={"cluster": "Cluster"},
    )
    fig_map.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption(
        "每一类代表一种“社区原型”，例如：高房价低犯罪、人口密集但治安一般、边缘低房价高风险区域等。"
    )


def render_relationship_lab():
    st.markdown("### 🔍 Relationship Lab · 关系探索实验室")
    st.caption("通过联合人口、房价与犯罪数据，寻找模式与故事。")

    tabs = st.tabs(
        [
            "Value vs Safety",
            "Trajectories",
            "Neighborhood Archetypes",
        ]
    )
    with tabs[0]:
        render_value_safety_lab()
    with tabs[1]:
        render_trajectory_lab()
    with tabs[2]:
        render_cluster_lab()


def render_radar_and_ranking():
    st.markdown("### 📊 ZIP Radar & Ranking · 综合雷达与排名")

    if zip_df.empty:
        st.warning("No ZIP-level statistics available.")
        return

    col1, col2 = st.columns([1, 1.2])

    # Radar
    with col1:
        st.subheader("📍 ZIP Radar")
        choices = zip_df.sort_values("ZIP_label")
        label_to_zip = dict(zip(choices["ZIP_label"], choices["ZIP"]))
        selected_label = st.selectbox("Choose ZIP:", choices["ZIP_label"].tolist())
        selected_zip = label_to_zip[selected_label]
        row = zip_df[zip_df["ZIP"] == selected_zip].iloc[0]

        categories = ["Population Score", "Safety Score", "Home Value Score"]
        r_vals = [row["pop_score"], row["safety_score"], row["value_score"]]
        r_vals.append(r_vals[0])
        theta_vals = categories + [categories[0]]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                line_color="#0ea5e9",
                fillcolor="rgba(14,165,233,0.2)",
                name=f"ZIP {selected_label}",
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=420,
            template="simple_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Overall score for **{selected_label}**: `{row['overall_score']:.3f}`"
        )

    # Ranking
    with col2:
        st.subheader("🏆 Top ZIPs by Overall Score")
        top_n = st.slider("Show Top N ZIPs:", 5, 25, 10)
        top_df = (
            zip_df.sort_values("overall_score", ascending=False)
            .head(top_n)
            .copy()
        )
        top_df = top_df.iloc[::-1].copy()

        fig_rank = px.bar(
            top_df,
            x="overall_score",
            y="ZIP_label",
            orientation="h",
            hover_data={
                "population": ":,",
                "crime_rate_latest": ":.2f",
                "zhvi_latest": ":,.0f",
                "overall_score": ":.3f",
            },
            labels={"overall_score": "Overall Score", "ZIP_label": "ZIP"},
            title="Highest scoring ZIPs · 综合表现最佳区域",
            color="overall_score",
            color_continuous_scale="Blues",
        )
        fig_rank.update_layout(
            height=420,
            margin=dict(l=80, r=30, t=50, b=40),
            template="simple_white",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        best_row = zip_df.loc[zip_df["overall_score"].idxmax()]
        best_label = best_row["ZIP_label"]
        st.success(
            f"🏅 Current best ZIP: **{best_label}** (score = {best_row['overall_score']:.3f})"
        )


def render_bubble_playground():
    st.markdown("### 🔵 Bubble Physics Playground · 球体物理游乐场")
    st.caption(
        "每个 ZIP 是一颗弹跳的“行星”：半径代表人口或房价，颜色代表安全度或综合得分，"
        "在物理世界中不断碰撞、挤压。用鼠标悬停查看信息，拖拽推动小球。"
    )

    size_metric = st.selectbox(
        "Bubble size metric / 球体大小指标",
        ["Population", "Home Value (latest)"],
    )
    color_metric = st.selectbox(
        "Bubble color metric / 球体颜色指标",
        ["Safety Score", "Overall Score"],
    )

    bubbles = []
    for _, row in zip_df.iterrows():
        bubbles.append(
            {
                "zip": row["ZIP"],
                "label": row["ZIP_label"],
                "population": float(row["population"]),
                "zhvi_latest": float(row["zhvi_latest"]),
                "safety_score": float(row["safety_score"]),
                "overall_score": float(row["overall_score"]),
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
        body {
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at top, #e0f2fe, #0f172a);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #f9fafb;
        }
        #worldContainer {
            width: 100%;
            height: 520px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #worldCanvas {
            border-radius: 24px;
            overflow: hidden;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.8);
            background: radial-gradient(circle at 10% 20%, #1d4ed8 0%, #020617 55%);
        }
        #tooltip {
            position: absolute;
            pointer-events: none;
            background: rgba(15,23,42,0.96);
            color: #e5e7eb;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 11px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6);
            opacity: 0;
            transform: translate(-50%, -150%);
            white-space: nowrap;
        }
    </style>
</head>
<body>
<div id="worldContainer">
    <canvas id="worldCanvas" width="960" height="480"></canvas>
    <div id="tooltip"></div>
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

    const sizeVals = bubbles.map(b => sizeMetric === "Population" ? b.population : b.zhvi_latest);
    const sizeMin = Math.min(...sizeVals);
    const sizeMax = Math.max(...sizeVals);

    const colorVals = bubbles.map(b => colorMetric === "Safety Score" ? b.safety_score : b.overall_score);
    const colorMin = Math.min(...colorVals);
    const colorMax = Math.max(...colorVals);

    function colorForScore(s) {
        const t = (s - colorMin) / (colorMax - colorMin || 1);
        const h = 140 * t;
        return `hsl(${h}, 85%, 55%)`;
    }

    const bodies = [];

    bubbles.forEach((b, i) => {
        const metricValue = sizeMetric === "Population" ? b.population : b.zhvi_latest;
        const radius = scaleValue(metricValue, sizeMin, sizeMax, 12, 40);

        const angle = (2 * Math.PI / bubbles.length) * i;
        the_r = Math.min(width, height) / 3;
        const x = width/2 + the_r * Math.cos(angle);
        const y = height/2 + the_r * Math.sin(angle);

        const body = Bodies.circle(x, y, radius, {
            restitution: 0.9,
            frictionAir: 0.03,
            friction: 0.01,
            render: {
                fillStyle: colorForScore(
                    colorMetric === "Safety Score" ? b.safety_score : b.overall_score
                ),
                strokeStyle: "rgba(15,23,42,0.9)",
                lineWidth: 1.5
            }
        });
        body.customData = b;
        bodies.push(body);
    });

    World.add(world, bodies);

    bodies.forEach((body, i) => {
        const angle = (2 * Math.PI / bodies.length) * i;
        const speed = 0.7;
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
            tooltip.style.opacity = 1;
            tooltip.style.left = mousePos.x + 'px';
            tooltip.style.top = mousePos.y + 'px';
            tooltip.innerHTML =
                d.label +
                '<br/>Pop ' + d.population.toLocaleString() +
                '<br/>Zhvi ' + (d.zhvi_latest > 0 ? '$' + d.zhvi_latest.toLocaleString() : 'N/A') +
                '<br/>Safety ' + d.safety_score.toFixed(2) +
                ' · Overall ' + d.overall_score.toFixed(2);
        } else {
            tooltip.style.opacity = 0;
        }
    });

    Render.run(render);
    const runner = Runner.create();
    Runner.run(runner, engine);
</script>
</body>
</html>
"""
    html = (
        html_template.replace("__BUBBLES__", bubbles_json)
        .replace("__SIZE_METRIC__", size_metric)
        .replace("__COLOR_METRIC__", color_metric)
    )
    components.html(html, height=540, scrolling=False)


def render_bar_chart_race():
    st.markdown("### 🏃‍♀️ Bar Chart Race · 排名赛跑")

    race_type = st.selectbox(
        "Race variable / 比赛变量",
        ["Home Value (ZHVI)", "Crime Rate"],
    )

    if race_type == "Home Value (ZHVI)":
        df = race_zhvi_df.copy()
        df = df[df["Zhvi"] > 0]
        st.caption("房价随时间的排名变化 · ZHVI bar chart race")
        fig = px.bar(
            df,
            x="Zhvi",
            y="ZIP",
            color="ZIP",
            orientation="h",
            animation_frame="Year",
            range_x=[0, df["Zhvi"].max() * 1.1],
            labels={"Zhvi": "Average Home Value (USD)", "ZIP": "ZIP Code"},
            title="Home Value Race · 房价排名赛跑",
        )
    else:
        df = race_crime_df.copy()
        st.caption("犯罪率随时间的排名变化 · Crime Rate bar chart race")
        fig = px.bar(
            df,
            x="CrimeRate",
            y="ZIP",
            color="ZIP",
            orientation="h",
            animation_frame="Year",
            range_x=[0, df["CrimeRate"].max() * 1.1],
            labels={"CrimeRate": "Crime Rate (per 1000 residents)", "ZIP": "ZIP Code"},
            title="Crime Rate Race · 犯罪率排名赛跑",
        )

    fig.update_layout(
        template="simple_white",
        height=620,
        margin=dict(l=80, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_orbit_view():
    st.markdown("### 🪐 Orbit View · 城市“行星系统”")
    st.caption(
        "以市中心为“太阳”，每个 ZIP 是一颗行星：距离代表与市中心的距离，大小代表房价，颜色代表安全度。"
    )

    orbit_df = gdf.merge(
        zip_df[["ZIP", "safety_score", "zhvi_latest"]], on="ZIP", how="left"
    )
    orbit_df = orbit_df.dropna(subset=["safety_score", "zhvi_latest"])

    if orbit_df.empty:
        st.warning("No data for orbit view.")
        return

    orbit_df = orbit_df.sort_values("ZIP")
    orbit_df["angle_index"] = range(len(orbit_df))
    orbit_df["theta"] = orbit_df["angle_index"] / len(orbit_df) * 360

    fig = px.scatter_polar(
        orbit_df,
        r="distance_from_loop",
        theta="theta",
        size="zhvi_latest",
        color="safety_score",
        hover_name="ZIP",
        color_continuous_scale="Blues",
        size_max=40,
        title="Chicago ZIPs Orbiting the Loop",
    )
    fig.update_layout(
        template="simple_white",
        height=680,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_adjacency_network():
    st.markdown("### 🕸️ Adjacency Network · 邻接关系网络图")
    st.caption(
        "每个节点是 ZIP，相邻的 ZIP 用线连接；节点大小代表综合得分，颜色代表安全度。"
    )

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>ZIP Adjacency Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at top left, #e5e7eb, #020617);
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        #network {
            width: 100%;
            height: 620px;
        }
        .node text {
            pointer-events: none;
            font-size: 10px;
            fill: #e5e7eb;
        }
        .tooltip {
            position: absolute;
            pointer-events: none;
            background: rgba(15,23,42,0.96);
            color: #e5e7eb;
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 11px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.6);
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
        .range([8, 28]);

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
        .force('collision', d3.forceCollide().radius(d => sizeScale(d.overall_score) + 3));

    const link = svg.append('g')
        .attr('stroke', '#64748b')
        .attr('stroke-opacity', 0.45)
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
        .attr('stroke', '#020617')
        .attr('stroke-width', 1.2);

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
                '<br/>Overall: ' + d.overall_score.toFixed(3) +
                '<br/>Safety: ' + d.safety_score.toFixed(3)
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
        html_template.replace("__NODES_DATA__", nodes_json)
        .replace("__LINKS_DATA__", links_json)
    )
    components.html(html, height=640, scrolling=False)


# ============ 页面路由 ============

st.sidebar.title("🏙️ Chicago ZIP Visual Playground")

view_mode = st.sidebar.radio(
    "Choose a view / 选择视图",
    [
        "Overview Map",
        "Relationship Lab",
        "Radar & Ranking",
        "Bubble Playground",
        "Bar Chart Race",
        "Orbit View",
        "Adjacency Network",
    ],
)

st.title("🌆 Chicago ZIP Visual Playground")
st.write(
    "将人口、房价和犯罪数据结合，用地图与艺术感图形探索芝加哥不同 ZIP 社区的结构与故事。"
)

if view_mode == "Overview Map":
    render_overview_map()
elif view_mode == "Relationship Lab":
    render_relationship_lab()
elif view_mode == "Radar & Ranking":
    render_radar_and_ranking()
elif view_mode == "Bubble Playground":
    render_bubble_playground()
elif view_mode == "Bar Chart Race":
    render_bar_chart_race()
elif view_mode == "Orbit View":
    render_orbit_view()
elif view_mode == "Adjacency Network":
    render_adjacency_network()



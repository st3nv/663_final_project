# Home.py - Main entry point for Chicago Dashboard
import streamlit as st

st.set_page_config(
    page_title="Chicago Housing and Crime Dashboard",
    page_icon="📌",
    layout="wide"
)

st.title("📌 Chicago Housing and Crime Dashboard")

st.markdown("""
Welcome to the Chicago Housing and Crime Dashboard! This interactive application allows you to explore:

### 🧭 How the dashboard is organized

#### 1. Core maps
- **📊 Population** – View population distribution across Chicago ZIP codes (2021 Census).
- **🚨 Crime Rate** – Explore crime counts and crime rates per 1,000 residents over time.
- **🏠 Housing Prices** – Track ZIP-level home value changes (ZHVI, 2000–2024).

#### 2. Relationship & insight tools
- **💡 Relationship Lab** – Experiment with how crime, population, and home values move together across ZIP codes and over time.
- **🎯 Statistical Analysis** – Run pooled regression models with year fixed effects, ZIP clustering, and time-series diagnostics.
- **🌆 Other Cool Visualizations** – Explore scoring-based neighborhood rankings, network-style connectivity, and orbit-style views around the Loop.
- **💬 Chatbox** – Ask natural-language questions about crime, home values, and our statistical findings; answers are grounded in the same data and models.

### 🔧 Shared controls
- Use the **Settings** panel in the sidebar to:
  - Exclude specific ZIP codes (e.g., airport or PO-box ZIPs).
  - Optionally exclude **year 2025** to keep analyses focused on stable historical data.

---

👈 **Use the sidebar to navigate between different views**

### Data Sources
- Chicago Population Counts (2021)
- Chicago Crime Data (preprocessed)
- Zillow Home Value Index (ZHVI) Data
""")

# Display some quick stats if data is available
st.markdown("---")
st.subheader("Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "🗺️ **Core Maps**\n\nBegin with Population, Crime Rate, and Housing Prices to get a feel for the data across ZIP codes and over time."
    )
    
with col2:
    st.info(
        "💡 **Relationship Lab & Stats**\n\nThen open the Relationship Lab and Statistical Analysis tabs to see how crime and home values relate, including regression results."
    )
    
with col3:
    st.info(
        "💬 **Insight Chatbox**\n\nUse the Chatbox to ask follow-up questions in plain language, powered by the same data and statistical summaries."
    )

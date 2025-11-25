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

### 📊 Key Variables (Maps)
- **Population** - View population distribution across Chicago ZIP codes (2021 data)
- **Crime Rate** - Explore crime rates per 1,000 residents with animated timeline (2001-2024)
- **Housing Prices** - Track home value changes over time (2000-2024)

### 💡 Insights
- **Insight 1** - Coming soon
- **Insight 2** - Coming soon  
- **Insight 3** - Coming soon

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
    st.info("🗺️ **Population Map**\n\nStart with the population distribution to understand Chicago's demographics.")
    
with col2:
    st.info("🚨 **Crime Analysis**\n\nExplore crime trends over 20+ years with interactive animations.")
    
with col3:
    st.info("🏠 **Housing Prices**\n\nTrack how home values have changed across different neighborhoods.")

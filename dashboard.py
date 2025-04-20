# MUST BE FIRST Streamlit command
import streamlit as st
st.set_page_config(layout="wide", page_title="US ZIP Code Economic Dashboard")

# 🔧 Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import ttest_ind

# Cache the dataset loading
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    return df

# Load dataset
df = load_data()

# Page Header
st.title("Unlocking Business Success Through Strategic Location Analysis Dashboard")
st.markdown("Explore economic data at the ZIP code level across the United States.")

# 🎛 Sidebar Navigation
section = st.sidebar.radio("Navigate to Section", [
    "Home",
    "Map View: AGI by State + ZIP Explorer",
    "Population Overview by State",
    "Predictive Model Explorer",
    "ZIP Code Recommender",
    "Hypothesis Testing Viewer"
])

# The rest of the code remains unchanged
# --------------------------------------------------------
# SECTION 1: MAP VIEW + ZIP EXPLORER
if section == "Home":
    col1, col2, col3 = st.columns([1,2,1])  # Create 3 columns for centering
    with col2:  # Use middle column
        st.image("usmap.jpg", use_container_width=True, width=400)  # Set width to 400 pixels

    st.title("ZIP Code Economic Opportunity Dashboard")
    st.markdown("""
    Welcome to the **ZIP Code Economic Analysis Dashboard**!  
    This interactive tool uses real U.S. data from the IRS, Census Bureau, and Business Statistics to:

    - Visualize economic activity by ZIP code and state  
    - Explore how income, population, and education affect business density  
    - Build and test predictive models for economic potential  
    - Recommend ZIP codes based on user preferences  
    - Perform statistical hypothesis testing to drive real insights  

    ---
    """)

    st.markdown("### 🔗 Connect with Me")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/sai-pranam-reddy-chillakuru/)")
    with col2:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/ChillakuruSaipranam)")

elif section == "Map View: AGI by State + ZIP Explorer":
    st.header("AGI by State + ZIP Explorer")

    # AGI by state choropleth
    state_agi = df.groupby("STATE")["Total_AGI"].sum().reset_index()

    fig = px.choropleth(
        state_agi,
        locations="STATE",
        locationmode="USA-states",
        color="Total_AGI",
        color_continuous_scale="Viridis",
        scope="usa",
        title="Total Adjusted Gross Income (AGI) by State"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ZIP Code Table
    st.subheader("Explore ZIP Codes by State")
    selected_state = st.selectbox("Select a State to view ZIP codes", sorted(df["STATE"].unique()))
    filtered_zip_df = df[df["STATE"] == selected_state]

    st.dataframe(
        filtered_zip_df[["ZIP", "Total_AGI", "Total_Population", "Median_Income",
                         "Percent_Bachelor_or_Higher", "Total_Businesses"]]
        .sort_values(by="Total_AGI", ascending=False)
        .reset_index(drop=True)
    )

# The rest of the sections remain unchanged

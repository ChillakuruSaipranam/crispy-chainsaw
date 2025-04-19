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

# Load dataset
df = pd.read_csv("cleaned_data.csv")
df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)

#  Page Header
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

# --------------------------------------------------------
# SECTION 2: PREDICTIVE MODEL
elif section == "Predictive Model Explorer":
    st.header("Predictive Model: Classify High AGI ZIPs")

    df['AGI_Binary'] = df['AGI_Level'].apply(lambda x: 1 if x in ['High', 'Mid-High'] else 0)
    features = ['Total_Population', 'Median_Income', 'Percent_Bachelor_or_Higher',
                'Total_Businesses', 'AGI_Per_Capita', 'Business_Density']

    X = StandardScaler().fit_transform(df[features])
    y = df['AGI_Binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model_type = st.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
    model = LogisticRegression(max_iter=1000) if model_type == "Logistic Regression" else RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Classification Report")
    st.json(classification_report(y_test, y_pred, output_dict=True))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------
# SECTION 3: ZIP RECOMMENDER
elif section == "ZIP Code Recommender":
    st.header("ZIP Code Recommender")

    pop_min, pop_max = st.slider("Population range", 0, int(df['Total_Population'].max()), (5000, 50000))
    income_min, income_max = st.slider("Median income range", 0, int(df['Median_Income'].max()), (30000, 90000))
    edu_min = st.slider("Minimum % Bachelor's Degree", 0, 100, 20)

    filtered = df[
        (df['Total_Population'] >= pop_min) &
        (df['Total_Population'] <= pop_max) &
        (df['Median_Income'] >= income_min) &
        (df['Median_Income'] <= income_max) &
        (df['Percent_Bachelor_or_Higher'] >= edu_min)
    ]

    st.subheader(f"Matching ZIP Codes ({len(filtered)} found)")
    st.dataframe(filtered[["ZIP", "STATE", "Total_AGI", "Median_Income",
                           "Percent_Bachelor_or_Higher", "Total_Businesses"]].head(20))

# --------------------------------------------------------
# SECTION 4: HYPOTHESIS TESTING
elif section == "Hypothesis Testing Viewer":
    st.header("Hypothesis: Does Business Density Affect AGI?")

    q25 = df['Business_Density'].quantile(0.25)
    q75 = df['Business_Density'].quantile(0.75)

    low_density = df[df['Business_Density'] <= q25]
    high_density = df[df['Business_Density'] >= q75]

    t_stat, p_val = ttest_ind(high_density['Total_AGI'], low_density['Total_AGI'])

    st.write(f"**T-statistic**: {t_stat:.2f}")
    st.write(f"**P-value**: {p_val:.4f}")
    if p_val < 0.05:
        st.success("Statistically significant: Business density *impacts* AGI.")
    else:
        st.warning("Not statistically significant.")

# --------------------------------------------------------

# SECTION 5: POPULATION OVERVIEW BY STATE
elif section == "Population Overview by State":
    st.header("Population Overview by State")

    state_population = df.groupby("STATE")["Total_Population"].sum().reset_index()
    fig = px.bar(state_population, x="STATE", y="Total_Population",
                 title="Total Population by State", color="Total_Population",
                 color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

    
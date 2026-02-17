"""
Streamlit app for Iris dataset exploratory data analysis.
Provides visualization and statistical analysis of the Iris dataset.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.datasets import load_iris

# Configure Streamlit page settings
st.set_page_config(
    page_title="Iris EDA",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description
st.title("🌸 Iris Dataset Exploratory Data Analysis")
st.markdown("Interactive dashboard for exploring the classic Iris dataset")

# Load the Iris dataset
@st.cache_data
def load_data():
    """Load the Iris dataset from scikit-learn."""
    iris_data = load_iris()
    iris_df = pd.DataFrame(
        data=iris_data.data,
        columns=iris_data.feature_names
    )
    iris_df['species'] = iris_data.target_names[iris_data.target]
    return iris_df

# Load data
iris_df = load_data()

# ============================================================================
# Section 1: Display the first rows of data
# ============================================================================
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", len(iris_df))
with col2:
    st.metric("Total Columns", len(iris_df.columns))
with col3:
    st.metric("Species", iris_df['species'].nunique())

st.subheader("First Few Rows")
num_rows_display = st.slider("Number of rows to display", min_value=1, max_value=20, value=5)
st.dataframe(iris_df.head(num_rows_display), use_container_width=True)

# ============================================================================
# Section 2: Summary statistics
# ============================================================================
st.header("📈 Summary Statistics")
st.subheader("Descriptive Statistics")
st.dataframe(iris_df.describe(), use_container_width=True)

# Display data types
st.subheader("Data Types")
st.dataframe(iris_df.dtypes.to_frame("Data Type"), use_container_width=True)

# ============================================================================
# Section 3: Column selection and visualization
# ============================================================================
st.header("🎨 Visualization & Analysis")

# Get numeric columns (exclude species)
numeric_columns = iris_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# User selection
selected_column = st.selectbox(
    "Select a numeric column for histogram",
    options=numeric_columns,
    index=0
)

# ============================================================================
# Section 4: Histogram visualization
# ============================================================================
st.subheader("Histogram")
histogram_fig = px.histogram(
    iris_df,
    x=selected_column,
    nbins=20,
    color='species',
    title=f"Distribution of {selected_column}",
    labels={selected_column: selected_column.replace('(cm)', '(cm)')},
    barmode='overlay',
    opacity=0.7,
    hover_data={'species': True}
)
histogram_fig.update_layout(height=400, hovermode='x unified')
st.plotly_chart(histogram_fig, use_container_width=True)

# ============================================================================
# Section 5: Scatter plot visualization
# ============================================================================
st.subheader("Scatter Plot Analysis")

# Allow user to select x and y axes for scatter plot
col1, col2 = st.columns(2)
with col1:
    x_column = st.selectbox("Select X-axis column", options=numeric_columns, index=0)
with col2:
    y_column = st.selectbox("Select Y-axis column", options=numeric_columns, index=1)

# Create scatter plot
scatter_fig = px.scatter(
    iris_df,
    x=x_column,
    y=y_column,
    color='species',
    title=f"{x_column} vs {y_column}",
    labels={x_column: x_column, y_column: y_column},
    hover_data={'species': True},
    size_max=8,
    opacity=0.7
)
scatter_fig.update_layout(height=500, hovermode='closest')
st.plotly_chart(scatter_fig, use_container_width=True)

# ============================================================================
# Section 6: Additional insights
# ============================================================================
st.header("🔍 Additional Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Species Distribution")
    species_counts = iris_df['species'].value_counts()
    species_fig = px.bar(
        x=species_counts.index,
        y=species_counts.values,
        labels={'x': 'Species', 'y': 'Count'},
        title="Number of Samples per Species",
        color=species_counts.index
    )
    st.plotly_chart(species_fig, use_container_width=True)

with col2:
    st.subheader("Correlation Matrix")
    correlation_matrix = iris_df[numeric_columns].corr()
    corr_fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns.tolist(),
        y=correlation_matrix.index.tolist(),
        colorscale='RdBu',
        showscale=True
    )
    st.plotly_chart(corr_fig, use_container_width=True)

# Footer
st.divider()
st.markdown(
    """
    **About this app:**
    - Dataset: [Iris dataset (scikit-learn)](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
    - Built with: [Streamlit](https://streamlit.io/) for interactive visualization
    - Designed for GitHub Codespaces compatibility
    """
)

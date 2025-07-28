import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set Streamlit Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("🛍️ Customer Segmentation using k-Means")

# Sidebar for uploading dataset
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load default sample dataset
@st.cache_data
def load_sample_data():
    return pd.DataFrame({
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 137, 200),
        'Spending Score (1-100)': np.random.randint(1, 101, 200)
    })

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded!")
else:
    df = load_sample_data()
    st.info("ℹ️ Using sample dataset (Mall Customers)")

st.subheader("📋 Preview Data")
st.dataframe(df.head())

# Sidebar for feature selection
st.sidebar.header("2. Select Features")
default_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
available_features = df.select_dtypes(include=[np.number]).columns.tolist()
features = st.sidebar.multiselect("Choose numeric features for clustering", options=available_features, default=default_features)

if len(features) < 2:
    st.warning("⚠️ Please select at least 2 features to proceed.")
    st.stop()

X = df[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
st.sidebar.header("3. Choose Number of Clusters")
max_k = st.sidebar.slider("Max k for Elbow Method", 1, 15, 10)
inertia = []

for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
st.subheader("📈 Elbow Method to Find Optimal k")
fig, ax = plt.subplots()
ax.plot(range(1, max_k + 1), inertia, marker='o')
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Curve")
st.pyplot(fig)

# Select final k
k = st.sidebar.slider("Select Final Number of Clusters (k)", 2, max_k, 5)

# Fit final model
kmeans_final = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA for 2D Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Visualize Clusters
st.subheader("🧠 Clusters Visualization (PCA 2D Projection)")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', ax=ax2)
plt.title("Customer Segments (via PCA)")
st.pyplot(fig2)

# Cluster Summary
st.subheader("📊 Cluster Summary")
st.dataframe(df.groupby('Cluster')[features].mean().style.highlight_max(axis=0))

# Download segmented data
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df)
st.download_button(
    "📥 Download Segmented Data as CSV",
    csv,
    "segmented_customers.csv",
    "text/csv",
    key='download-csv'
)

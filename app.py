import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained models
pca = joblib.load("pca_model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")  # Save your scaler earlier with joblib.dump()

st.set_page_config(page_title="🎬 Movie User Taste Clustering", layout="wide")

# Title
st.title("🎬 Movie User Taste Clustering")
st.markdown("Upload user ratings to find which **taste cluster** they belong to!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your ratings CSV file", type=["csv"])

if uploaded_file:
    # Load ratings
    user_ratings = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Uploaded User Ratings")
    st.write(user_ratings.head())

    # Scale + PCA
    scaled = scaler.transform(user_ratings)
    reduced = pca.transform(scaled)
    cluster = kmeans.predict(reduced)

    st.success(f"✅ This user belongs to **Cluster {cluster[0]}**")

    # Visualization (PCA 2D)
    pca_2d = reduced[:, :2]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=cluster, cmap="Set1", s=80, alpha=0.8, edgecolors="k")
    ax.set_title("User Cluster Projection (PCA 2D)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file with ratings (movies as columns, ratings as rows).")

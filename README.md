# 🎬 Movie User Taste Clustering

## 📌 Project Overview
This project applies **unsupervised learning (clustering)** to group movie users based on their rating patterns using the **MovieLens 100k dataset**. The goal is to identify hidden user segments for better **personalization and recommendation** strategies.

---

## 🚀 Workflow
### Phase 1: Problem Understanding
- Objective: Cluster users into taste groups for OTT personalization.

### Phase 2: Data Collection & Preparation
- Dataset: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- Processed into **user–movie rating matrix (943 × 1682)**

### Phase 3: EDA
- Ratings distribution
- Sparsity check
- Basic summary statistics

### Phase 4: Clustering (Baseline)
- Standardization with `StandardScaler`
- PCA (95% variance retained)
- KMeans clustering
- Evaluation: Silhouette ≈ 0.34, DBI ≈ 1.74

### Phase 5: Model Tuning
- GridSearch over k and PCA components
- Best params: **k = 3, PCA = 100**
- Silhouette ≈ 0.45 (better!)

<img width="698" height="506" alt="image" src="https://github.com/user-attachments/assets/25abf347-53f9-4608-8e3a-a2fb8dac1f7d" />


### Phase 6: Visualization
- 2D PCA scatter plot
- 3D PCA scatter plot
- t-SNE projection

### Phase 7: Insights & Recommendations
- **Cluster 0**: Sci-Fi & Fantasy lovers
- **Cluster 1**: Romance & Family-oriented viewers
- **Cluster 2**: Drama & Crime buffs

<img width="552" height="486" alt="image" src="https://github.com/user-attachments/assets/77097bc7-dd7a-4006-ad79-44d0f6dda9fa" />


### Phase 8: Deployment
- Serialize trained models (joblib.dump)
- Build Streamlit dashboard for interactive demo


**Business Actions**:
- Personalize OTT homepages
- Cluster-based marketing campaigns
- Retention via personalized bundles

---

## 🛠 Tech Stack
- Python, Jupyter Notebook
- Libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`

---

## 📈 Results
- Found 3 meaningful clusters of movie users
- Stronger clustering after PCA tuning
- Insights support **personalized recommendations & marketing strategies**

<img width="1850" height="829" alt="image" src="https://github.com/user-attachments/assets/12c6cee2-e392-4898-81a4-a726af61b1ec" />


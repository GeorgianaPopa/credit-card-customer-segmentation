"""
Credit Card Customer Segmentation - Exam Project
Student: Popa Georgiana-Daniela
Group: AIAC 1
Description: Unsupervised Learning Pipeline comparing K-Means & Agglomerative Clustering
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

# Ensure directories exist for saving figures
os.makedirs('outputs', exist_ok=True)
print("=== Step 1: Environment and Dependencies Initialized Successfully ===")

# =====================================================================
# 1. DATASET LOADING & EXPLORATORY DATA ANALYSIS (EDA) [Slides S4->S5]
# =====================================================================
print("\n=== Step 2: Loading Dataset & Running Preprocessing ===")
data_path = 'data/CC GENERAL.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing dataset file! Please place 'CC GENERAL.csv' inside the 'data/' folder.")

df = pd.read_csv(data_path)
print(f"Dataset Loaded Successfully. Rows: {df.shape[0]}, Features: {df.shape[1]}")

# Display initial data state info
print("\n--- Missing Values Before Imputation ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Preprocessing: Impute missing entries with the feature median to keep distribution intact
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())

# Drop customer unique identifier as it contains no predictive behavioral patterns
X_raw = df.drop(columns=['CUST_ID'])

# Feature Scaling: Apply standard normalization (Mean=0, Var=1) since distance metrics 
# are highly sensitive to magnitude variations across distinct financial metrics.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print("Data Preprocessing and Feature Standardization Complete.")

# =====================================================================
# 2. BASELINE MODELING & VALIDATION (K-MEANS) [Slides S6->S8]
# =====================================================================
print("\n=== Step 3: Running Baseline K-Means & Elbow Optimization ===")
wcss = []
k_range = range(2, 11)

# Compute Within-Cluster Sum of Squares (WCSS) to locate the geometric 'Elbow'
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Generate and Save the Elbow Method Chart
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal K (Baseline Model)', fontsize=14)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Inertia)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/s6_elbow_plot.png', dpi=300)
plt.close()
print("Saved: outputs/s6_elbow_plot.png")

# Instantiate and fit the optimal Baseline Model (Choosing K=4 based on Elbow assessment)
optimal_k = 4
baseline_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans_labels = baseline_kmeans.fit_transform(X_scaled)
kmeans_cluster_assignments = baseline_kmeans.labels_

# Calculate validation scores for the baseline K-Means algorithm
kmeans_silhouette = silhouette_score(X_scaled, kmeans_cluster_assignments, sample_size=5000, random_state=42)
kmeans_db_index = davies_bouldin_score(X_scaled, kmeans_cluster_assignments)

print(f"-> Baseline K-Means (K={optimal_k}) Silhouette Score: {kmeans_silhouette:.4f}")
print(f"-> Baseline K-Means (K={optimal_k}) Davies-Bouldin Index: {kmeans_db_index:.4f}")

# =====================================================================
# 3. IMPROVED MODEL & COMPARATIVE ANALYSIS [Slides S9->S12]
# =====================================================================
print("\n=== Step 4: Running Improved Hierarchical Clustering & Linkage ===")

# Generate and Save a Dendrogram using Ward's Linkage
# We sample 200 rows to maintain clean visibility without cluttering structural branches
np.random.seed(42)
sample_indices = np.random.choice(X_scaled.shape[0], size=200, replace=False)
X_dendrogram_sample = X_scaled[sample_indices]

plt.figure(figsize=(10, 6))
dendrogram = sch.dendrogram(sch.linkage(X_dendrogram_sample, method='ward'))
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage, Sampled)', fontsize=14)
plt.xlabel('Customer Data Point Indices', fontsize=12)
plt.ylabel('Euclidean Distance Threshold', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/s9_dendrogram.png', dpi=300)
plt.close()
print("Saved: outputs/s9_dendrogram.png")

# Fit the Agglomerative Hierarchical Clustering Model
hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical_model.fit_predict(X_scaled)

# Calculate validation scores for the improved hierarchical model
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels, sample_size=5000, random_state=42)
hierarchical_db_index = davies_bouldin_score(X_scaled, hierarchical_labels)

print(f"-> Hierarchical Model Silhouette Score: {hierarchical_silhouette:.4f}")
print(f"-> Hierarchical Model Davies-Bouldin Index: {hierarchical_db_index:.4f}")

# =====================================================================
# 4. DIMENSIONALITY REDUCTION & VISUALIZATION [Slides S13->S14]
# =====================================================================
print("\n=== Step 5: Dimensionality Reduction via PCA & Final Profiles ===")

# Reduce 17 dimensional space down to 2 principal components for scatter plot visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create a clear side-by-side cluster distribution chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot K-Means Clusters
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_cluster_assignments, palette='viridis', alpha=0.6, ax=axes[0])
axes[0].set_title(f'K-Means Space Distribution (K={optimal_k})', fontsize=14)
axes[0].set_xlabel('Principal Component 1', fontsize=11)
axes[0].set_ylabel('Principal Component 2', fontsize=11)
axes[0].legend(title='Clusters')

# Plot Hierarchical Clusters
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hierarchical_labels, palette='plasma', alpha=0.6, ax=axes[1])
axes[1].set_title('Hierarchical Space Distribution (Ward Linkage)', fontsize=14)
axes[1].set_xlabel('Principal Component 1', fontsize=11)
axes[1].set_ylabel('Principal Component 2', fontsize=11)
axes[1].legend(title='Clusters')

plt.suptitle('Comparison of Customer Segments in 2D PCA Space', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/s13_pca_comparison.png', dpi=300)
plt.close()
print("Saved: outputs/s13_pca_comparison.png")

# Group original dataset features by K-Means labels to extract business personas
df['Cluster'] = kmeans_cluster_assignments
profile_features = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
cluster_profiles = df.groupby('Cluster')[profile_features].mean()

# Export a cluster distribution heat map showing feature deviations across segments
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profiles.T, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
plt.title('Financial Feature Distribution Profile per Customer Segment', fontsize=14, pad=15)
plt.ylabel('Financial Behavioral Features', fontsize=12)
plt.xlabel('Identified Customer Segments', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/s14_cluster_profiles.png', dpi=300)
plt.close()
print("Saved: outputs/s14_cluster_profiles.png")

# =====================================================================
# PRINT COMPLETE PERFORMANCE REPORT SUMMARY
# =====================================================================
print("\n" + "="*55)
print("        FINAL PRESENTATION METRIC SUMMARY REPORT        ")
print("="*55)
print(f"{'Metric Used':<25} | {'Baseline (K-Means)':<20} | {'Hierarchical':<15}")
print("-"*55)
print(f"{'Silhouette Score (↑)':<25} | {kmeans_silhouette:<20.4f} | {hierarchical_silhouette:<15.4f}")
print(f"{'Davies-Bouldin Index (↓)':<25} | {kmeans_db_index:<20.4f} | {hierarchical_db_index:<15.4f}")
print("="*55)
print("Pipeline complete. All graphics are generated in the 'outputs/' folder for your slides.")
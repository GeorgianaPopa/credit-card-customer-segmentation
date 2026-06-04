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

# create outputs folder 
os.makedirs('outputs', exist_ok=True)

# define where the dataset is located 
data_path = 'data/CC GENERAL.csv'

# make sure the file is actually there before trying to read it
if not os.path.exists(data_path):
    raise FileNotFoundError("Missing dataset file in data/ folder.")

# read the csv file into a pandas dataframe
df = pd.read_csv(data_path)

# fill in missing minimum payments with the median value so we don't lose rows
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

# do the exact same thing for any missing credit limits
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())

# drop the customer id column because it's just an identifier, not a behavioral feature
X = df.drop(columns=['CUST_ID'])

# initialize the standard scaler to normalize our features
scaler = StandardScaler()

# fit the scaler to our data and transform it so everything is on the same mathematical scale
X_scaled = scaler.fit_transform(X)

# prepare an empty list to store the within-cluster sum of squares (WCSS)
wcss = []

# we'll test k values from 2 up to 10 to find the elbow
k_range = range(2, 11)

# loop through each k value to see which one works best
for k in k_range:
    # create a kmeans model for the current k
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    
    # fit the model to our scaled data
    kmeans.fit(X_scaled)
    
    # add the inertia (WCSS) to our list for plotting later
    wcss.append(kmeans.inertia_)

# set up the figure for our elbow plot
plt.figure(figsize=(8, 5))

# plot the k values against their corresponding wcss
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')

# add a title to explain the chart
plt.title('Elbow Method')

# label the x and y axes so it makes sense
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')

# turn on the grid for easier visual reading
plt.grid(True)

# tidy up the layout so nothing overlaps
plt.tight_layout()

# save the plot directly to our outputs folder
plt.savefig('outputs/s6_elbow_plot.png', dpi=300)

# close the plot to free up memory
plt.close()

# based on the elbow plot, 4 looks like a solid choice for k
optimal_k = 4

# set up our final baseline kmeans model using the optimal k=4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)

# fit the model and get the cluster assignments for each customer
kmeans_labels = kmeans.fit_predict(X_scaled)

# calculate the silhouette score to see how well-separated our clusters are
# a sample size is used to speed things up since it's computationally heavy
kmeans_sil = silhouette_score(X_scaled, kmeans_labels, sample_size=5000, random_state=42)

# calculate the davies-bouldin score for another perspective on cluster quality
kmeans_db = davies_bouldin_score(X_scaled, kmeans_labels)

# set a random seed so our data sampling is reproducible next time we run this
np.random.seed(42)

# grab 200 random indices from our data for the dendrogram (so the chart isn't a solid block of black lines)
sample_idx = np.random.choice(X_scaled.shape[0], size=200, replace=False)

# filter our scaled data down using those random indices
X_sample = X_scaled[sample_idx]

# set up the figure for the dendrogram
plt.figure(figsize=(10, 6))

# build and plot the dendrogram using the ward linkage method
sch.dendrogram(sch.linkage(X_sample, method='ward'))

# add a title and label the axes for the dendrogram
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Distance')

# tidy up the layout
plt.tight_layout()

# save the dendrogram plot
plt.savefig('outputs/s9_dendrogram.png', dpi=300)

# close the plot
plt.close()

# set up our improved model: agglomerative hierarchical clustering using the same optimal k=4
hc = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')

# fit the model and grab the cluster labels
hc_labels = hc.fit_predict(X_scaled)

# calculate the silhouette score for the hierarchical model
hc_sil = silhouette_score(X_scaled, hc_labels, sample_size=5000, random_state=42)

# calculate the davies-bouldin score for the hierarchical model
hc_db = davies_bouldin_score(X_scaled, hc_labels)

# set up pca to reduce our 17 features down to just 2 dimensions so we can visualize it on a flat screen
pca = PCA(n_components=2, random_state=42)

# apply pca to transform our scaled data into the 2d space
X_pca = pca.fit_transform(X_scaled)

# create a side-by-side subplot setup
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# draw a scatter plot for the kmeans clusters on the left side
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis', alpha=0.6, ax=axes[0])
axes[0].set_title('K-Means Clusters')

# draw a scatter plot for the hierarchical clusters on the right side
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=hc_labels, palette='plasma', alpha=0.6, ax=axes[1])
axes[1].set_title('Hierarchical Clusters')

# tidy up the layout for the side-by-side plots
plt.tight_layout()

# save the final pca comparison plot
plt.savefig('outputs/s13_pca_comparison.png', dpi=300)

# close the plot
plt.close()

# add the kmeans cluster labels back to our original dataframe so we can profile them in human terms
df['Cluster'] = kmeans_labels

# list the key business features we want to look at for our cluster profiles
features = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 
            'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']

# group by cluster and calculate the average value of each feature
profiles = df.groupby('Cluster')[features].mean()

# set up a figure for our final heatmap
plt.figure(figsize=(10, 6))

# draw a heatmap to visually show the average feature values per cluster
sns.heatmap(profiles.T, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)

# add a title to the heatmap
plt.title('Cluster Profiles')

# tidy up the layout
plt.tight_layout()

# save the heatmap plot
plt.savefig('outputs/s14_cluster_profiles.png', dpi=300)

# close the plot
plt.close()

# Finally, print out a quick summary to the console so we know it finished and can see the scores
print("=== Final Evaluation ===")
print(f"K-Means (k={optimal_k})      - Silhouette: {kmeans_sil:.4f}, Davies-Bouldin: {kmeans_db:.4f}")
print(f"Hierarchical (k={optimal_k}) - Silhouette: {hc_sil:.4f}, Davies-Bouldin: {hc_db:.4f}")
print("All pipeline steps complete. Output plots saved to the 'outputs' directory.")

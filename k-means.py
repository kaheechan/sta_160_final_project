import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = 'double_df.csv'

data = pd.read_csv(file_path)

data['Total_Pot'] = data['SUMMARY'].str.extract(r'Total pot (\d+)').astype(float)
data['Fold_Count'] = data['HOLE.CARDS'].str.count('folds')
data['Raise_Count'] = data['HOLE.CARDS'].str.count('raises')
data['Check_Count'] = data['HOLE.CARDS'].str.count('checks')
data['Call_Count'] = data['HOLE.CARDS'].str.count('calls')

numerical_features = data[['Total_Pot', 'Fold_Count', 'Raise_Count', 'Check_Count', 'Call_Count']].fillna(0)

scaler = StandardScaler()
normalized_features = scaler.fit_transform(numerical_features)

inertia = []
k_values = range(1, 11)
for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(normalized_features)

    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# Based on the elbow plot, let's choose an optimal number of clusters (e.g., 3)
optimal_k = 3

# Performing k-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

# Adding cluster labels to the original numerical features
numerical_features['Cluster'] = clusters

# Visualizing the clustering results
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    cluster_data = numerical_features[numerical_features['Cluster'] == cluster]
    plt.scatter(cluster_data['Total_Pot'], cluster_data['Raise_Count'], label=f'Cluster {cluster}')

plt.title('Clusters Based on Total Pot and Raise Count')
plt.xlabel('Total Pot')
plt.ylabel('Raise Count')
plt.legend()
plt.grid()
plt.show()

cluster_analysis = numerical_features.groupby('Cluster').mean()

import ace_tools as tools; tools.display_dataframe_to_user(name="Cluster Analysis Summary", dataframe=cluster_analysis)

cluster_analysis

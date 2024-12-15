import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class KmeansClustering:
    def __init__(self, selected_features_matrix, i):
        self.user_item_matrix = selected_features_matrix
        self.i = i

    def kmeans_cluster(self):
        # Step 1: Standardize the data
        scaler = StandardScaler()
        user_item_matrix_std = scaler.fit_transform(self.user_item_matrix)

        # Step 2: Run k-means clustering
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(user_item_matrix_std)

        # Step 3: Get cluster labels
        labels = kmeans.labels_

        # Optionally, plot the clusters if needed
        plt.figure(figsize=(10, 6))
        plt.hist(labels, bins=np.arange(n_clusters+1) - 0.5, edgecolor='black')
        plt.xticks(range(n_clusters))
        plt.xlabel('Cluster')
        plt.ylabel('Number of Users')
        plt.title('Distribution of Users Across Clusters')
        plt.savefig(f'/Users/reza/University/thesis_final_codes/data_set/user_clusters_distribution{self.i}.png', format='png')

        return labels

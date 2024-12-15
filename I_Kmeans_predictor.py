import numpy as np


class KmeansPredictor:
    def __init__(self, path, training_file_name, kmean_labels):
        self.path = path
        self.training_file_name = training_file_name
        self.kmean_labels = kmean_labels
        file_path = f'{path}{training_file_name}'
        self.base_matrix = np.genfromtxt(file_path, delimiter=',')

    def kmeans_weighted_collaborative_filtering(self):
        # Initialize a matrix to store the predicted ratings
        predicted_ratings = np.zeros(self.base_matrix.shape)

        # Iterate through the user-item matrix to predict missing values
        for i in range(self.base_matrix.shape[0]):
            for j in range(self.base_matrix.shape[1]):
                if self.base_matrix[i][j] == 0:  # Predict only for missing values
                    # Find users in the same cluster
                    cluster_users = np.where(self.kmean_labels == self.kmean_labels[i])[0]

                    # Exclude the current user from the cluster users
                    cluster_users = cluster_users[cluster_users != i]

                    # Calculate the weighted prediction for user i on item j
                    weighted_sum = 0
                    sum_of_weights = 0
                    for u in cluster_users:
                        if self.base_matrix[u][j] != 0:
                            similarity = 1  # All users in the same cluster are treated with equal similarity
                            weighted_sum += similarity * self.base_matrix[u][j]
                            sum_of_weights += abs(similarity)

                    if sum_of_weights != 0:
                        predicted_ratings[i][j] = weighted_sum / sum_of_weights

        print(f'****Predicted rating matrix for {self.training_file_name} has been saved****')
        return predicted_ratings

import numpy as np


class Predictor:
    def __init__(self, path, training_file_name, sim_matrix, k_sim_neighbor):
        self.path = path
        self.training_file_name = training_file_name
        file_path = f'{path}{training_file_name}'
        self.base_matrix = np.genfromtxt(file_path, delimiter=',')
        self.sim_matrix = sim_matrix
        self.k_sim_neighbor = k_sim_neighbor

    def weighted_collaborative_filtering(self):
        # Initialize a matrix to store the predicted ratings
        predicted_ratings = np.zeros(self.base_matrix.shape)

        # Iterate through the user-item matrix to predict missing values
        for i in range(self.base_matrix.shape[0]):
            for j in range(self.base_matrix.shape[1]):
                if self.base_matrix[i][j] == 0:  # Predict only for missing values
                    # Find the top-k most similar users to user i
                    similar_users = np.argsort(self.sim_matrix[i])[::-1][
                                    1:self.k_sim_neighbor + 1]  # Exclude self-similarity

                    # Calculate the weighted prediction for user i on item j
                    weighted_sum = 0
                    sum_of_weights = 0
                    for u in similar_users:
                        if self.base_matrix[u][j] != 0:
                            similarity = self.sim_matrix[i][u]
                            weighted_sum += similarity * self.base_matrix[u][j]
                            sum_of_weights += abs(similarity)

                    if sum_of_weights != 0:
                        predicted_ratings[i][j] = weighted_sum / sum_of_weights

        # file_path = f'{self.path}predicted_matrix_{self.training_file_name}'
        # np.savetxt(file_path, predicted_ratings, delimiter=',', fmt='%.4f')
        print(f'****Predicted rating matrix for {self.training_file_name} has been saved****')
        return predicted_ratings

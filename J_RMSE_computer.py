import numpy as np
from sklearn.metrics import root_mean_squared_error


class RMSEComputer:
    def __init__(self, path, testing_file_name, predicted_matrix):
        self.testing_file_name = testing_file_name
        file_path = f'{path}{testing_file_name}'
        self.base_matrix = np.genfromtxt(file_path, delimiter=',')
        self.predicted_matrix = predicted_matrix

    def calculate_RMSE(self):
        # Create a mask to filter out non-zero entries in the test matrix
        mask = self.base_matrix != 0

        # Extract the actual ratings from the test matrix
        actual_values = self.base_matrix[mask]
        # print(actual_values)

        # Extract the predicted ratings for the same positions as in the test matrix
        predicted_values = self.predicted_matrix[mask]
        # print(predicted_values)

        # Calculate the mean squared error (MSE) for all values
        rmse_all = root_mean_squared_error(actual_values, predicted_values)

        return rmse_all

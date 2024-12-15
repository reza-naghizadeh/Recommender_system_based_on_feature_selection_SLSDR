import numpy as np
from sklearn.metrics import precision_score, recall_score


class PrecisionRecallComputer:
    def __init__(self, path, testing_file_name, predicted_matrix):
        file_path = f'{path}{testing_file_name}'
        self.base_matrix = np.genfromtxt(file_path, delimiter=',')
        self.predicted_matrix = predicted_matrix

    def precision_recall_computer(self):
        # Create a mask to filter out non-zero entries in the test matrix
        mask = self.base_matrix != 0

        # Extract the actual ratings from the test matrix
        actual_values = self.base_matrix[mask]

        # Extract the predicted ratings for the same positions as in the test matrix
        predicted_values = self.predicted_matrix[mask]

        # Binarize actual and predicted
        threshold = 4
        actual_bin = np.where(actual_values >= threshold, 1, 0)
        predicted_bin = np.where(predicted_values >= threshold, 1, 0)

        precision = precision_score(actual_bin, predicted_bin)
        recall = recall_score(actual_bin, predicted_bin)

        return precision, recall


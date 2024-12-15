import numpy as np
from sklearn.neighbors import NearestNeighbors
import math


class SLSDR:
    def __init__(self, file_name, num_feature, K, steps, alpha, beta, lambda_value, sigma, path):
        np.random.seed(1001)
        self.path = path
        self.file_name = file_name
        file_path = f'{self.path}{self.file_name}'
        self.X = np.genfromtxt(file_path, delimiter=',')
        self.num_feature = num_feature
        self.K = K
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.lambda_value = lambda_value
        self.sigma = sigma

    def _KNN(self, XX):
        knn = NearestNeighbors(algorithm='auto', n_neighbors=self.K).fit(XX)
        distances, indices = knn.kneighbors(XX)
        return distances, indices

    # We have two W formulas uncomment the one you like to use
    # This _W function is the original W formula (#10 in paper)
    def _W(self, XX):
        n = len(XX)
        w = np.zeros((n, n))
        distances, indices = self._KNN(XX)
        for i in range(n):
            b = indices[i]
            for j in range(self.K):
                w[i, b[j]] = math.exp(-((distances[i, j]) ** 2) / (self.sigma ** 2))
                w[b[j], i] = math.exp(-((distances[i, j]) ** 2) / (self.sigma ** 2))
        return w
    # This is the modified W function using cosine similarity
    # def _W(self, XX):
    #     n = len(XX)
    #     w = np.zeros((n, n))
    #     distances, indices = self._KNN(XX)
    #
    #     epsilon = 1e-8  # Small value to prevent division by zero
    #
    #     for i in range(n):
    #         b = indices[i]
    #         for j in range(self.K):
    #             norm_i = np.linalg.norm(XX[i])
    #             norm_j = np.linalg.norm(XX[b[j]])
    #
    #             if norm_i > epsilon and norm_j > epsilon:
    #                 cos_sim = np.dot(XX[i], XX[b[j]]) / (norm_i * norm_j)
    #             else:
    #                 cos_sim = 0.0  # Handle the case where the norm is zero
    #
    #             w[i, b[j]] = cos_sim
    #             w[b[j], i] = cos_sim
    #     return w

    @staticmethod
    def _Diagonal_Matrix(first_matrix):
        D_norm_list = [np.linalg.norm(row) for row in first_matrix]
        return np.diag(D_norm_list)

    @staticmethod
    def _Check_Zero(Number):
        return max(Number, 10 ** (-8))

    @staticmethod
    def _U_Value(XX, V, S):
        E = XX - (np.dot(XX, np.dot(S, V)))
        U = np.diag(1 / np.maximum(np.linalg.norm(E, axis=1), 10**(-8)))
        return U

    @staticmethod
    def _Index_Select(matrix, i):
        return [row[i] for row in matrix]

    def _SLSDR_algorithm(self):
        XX = np.array(self.X)
        column_number = len(XX[0])

        S = np.random.rand(column_number, self.num_feature)
        V = np.random.rand(self.num_feature, column_number)

        epsilon = 1e-8

        for i in range(self.steps):
            W_S = self._W(XX)
            D_S = self._Diagonal_Matrix(W_S)

            W_V = self._W(XX.T)
            D_V = self._Diagonal_Matrix(W_V)

            U = self._U_Value(XX, V, S)

            S_up = np.dot(np.dot(np.dot(XX.T, U), XX), V.T) + np.dot(
                self.alpha * (np.dot(np.dot(XX.T, W_S), XX)) + (
                    (np.identity(column_number)) * (self.beta + self.lambda_value)), S)

            S_down1 = np.dot(np.dot(np.dot(np.dot(np.dot(XX.T, U), XX), S), V), V.T)
            S_down2 = np.dot((self.alpha * (np.dot(np.dot(XX.T, D_S), XX))) + (
                self.beta * np.ones(column_number)) + self.lambda_value * np.dot(S, S.T), S)
            S_down = S_down1 + S_down2

            S = S * (S_up / (S_down + epsilon))

            V_UP = np.dot(np.dot(np.dot(S.T, XX.T), U), XX) + self.alpha * (np.dot(V, W_V))

            V_DOWN = np.dot(np.dot(np.dot(np.dot(np.dot(S.T, XX.T), U), XX), S), V) + self.alpha * (
                np.dot(V, D_V))

            V = V * (V_UP / (V_DOWN + epsilon))

        S_final_norm = [np.linalg.norm(S[i]) for i in range(column_number)]
        S_final_norm_index = list(range(1, column_number + 1))

        S_norm_index = np.array([S_final_norm, S_final_norm_index]).T
        S_sorted = S_norm_index[np.argsort(S_norm_index[:, 0])]

        final_index = self._Index_Select(S_sorted, 1)
        final_index.reverse()

        final_index_list = [int(final_index[j] - 1) for j in range(self.num_feature)]

        return final_index_list

    def reducer(self, selected_features):
        reduced_matrix = self.X[:, selected_features]
        return reduced_matrix

    def activator(self):
        selected_features = self._SLSDR_algorithm()
        selected_features_matrix = self.reducer(selected_features)

        # save_file_path = f'{self.path}selected_features_matrix_{self.file_name}'
        # np.savetxt(save_file_path, selected_features_matrix, delimiter=',', fmt='%d')

        print(f"****Selected Features Matrix for {self.file_name} has been saved****")
        print(f'The shape of selected features matrix: {selected_features_matrix.shape}')

        return selected_features_matrix

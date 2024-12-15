import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.stats import pearsonr


class SimilarityCalculator:
    def __init__(self, ratings_matrix):
        self.ratings_matrix = np.array(ratings_matrix)
        self.num_users = self.ratings_matrix.shape[0]
        self.user_similarity_matrix_cosine = None
        self.user_similarity_matrix_jaccard = None
        self.user_similarity_matrix_pearson = None
        self.user_similarity_matrix_cosine_adjusted_cosine = None

    def _cosine_similarity(self):
        self.user_similarity_matrix_cosine = cosine_similarity(self.ratings_matrix, dense_output=True)

    def _adjusted_cosine_similarity(self):
        mean_ratings = np.nanmean(self.ratings_matrix, axis=1).reshape(-1, 1)
        ratings_centered = self.ratings_matrix - mean_ratings
        ratings_centered[np.isnan(ratings_centered)] = 0  # Replace NaNs with 0 for cosine computation
        self.user_similarity_matrix_cosine_adjusted_cosine = cosine_similarity(ratings_centered, dense_output=True)

    def _jaccard_similarity(self):
        binary_matrix = np.where(self.ratings_matrix > 0, 1, 0).astype(np.bool_)
        jaccard_distances = pairwise_distances(binary_matrix, metric='jaccard')
        self.user_similarity_matrix_jaccard = 1 - jaccard_distances

    def _pearson_correlation(self):
        sim_matrix = np.zeros((self.num_users, self.num_users))

        for i in range(self.num_users):
            for j in range(i, self.num_users):
                common_items_mask = ~np.isnan(self.ratings_matrix[i]) & ~np.isnan(self.ratings_matrix[j])

                if np.any(common_items_mask):
                    ratings_i = self.ratings_matrix[i, common_items_mask]
                    ratings_j = self.ratings_matrix[j, common_items_mask]

                    if len(ratings_i) > 1 and len(ratings_j) > 1:
                        similarity, _ = pearsonr(ratings_i, ratings_j)
                        similarity = similarity if not np.isnan(similarity) else 0  # Handle NaN case
                    else:
                        similarity = 0
                else:
                    similarity = 0

                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity

        self.user_similarity_matrix_pearson = sim_matrix

    def get_similarity_matrix(self, method='cosine'):
        if method == 'cosine':
            if self.user_similarity_matrix_cosine is None:
                self._cosine_similarity()
            print('****The cosine similarity has been computed ****')
            return pd.DataFrame(self.user_similarity_matrix_cosine, index=range(self.num_users), columns=range(self.num_users))
        elif method == 'jaccard':
            if self.user_similarity_matrix_jaccard is None:
                self._jaccard_similarity()
            print('****The Jaccard similarity has been computed ****')
            return pd.DataFrame(self.user_similarity_matrix_jaccard, index=range(self.num_users), columns=range(self.num_users))
        elif method == 'pearson':
            if self.user_similarity_matrix_pearson is None:
                self._pearson_correlation()
            print('****The Pearson similarity has been computed ****')
            return pd.DataFrame(self.user_similarity_matrix_pearson, index=range(self.num_users), columns=range(self.num_users))
        elif method == 'adjusted_cosine':
            if self.user_similarity_matrix_cosine_adjusted_cosine is None:
                self._adjusted_cosine_similarity()
            print('****The adjusted cosine similarity has been computed ****')
            return pd.DataFrame(self.user_similarity_matrix_cosine_adjusted_cosine, index=range(self.num_users), columns=range(self.num_users))
        else:
            raise ValueError("Unknown method: choose from 'cosine', 'jaccard', 'pearson', or 'adjusted_cosine'")

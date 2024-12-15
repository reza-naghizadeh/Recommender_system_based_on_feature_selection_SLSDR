import numpy as np
from sklearn.decomposition import PCA


class PCAFeatureSelector:

    def __init__(self, path, file_name, num_feature):
        np.random.seed(1001)
        self.path = path
        self.file_name = file_name
        file_path = f'{self.path}{self.file_name}'
        self.user_item_matrix = np.genfromtxt(file_path, delimiter=',')
        self.number_feature = num_feature

    def PCA_runner(self):
        pca = PCA(n_components=self.number_feature)
        selected_features_matrix = pca.fit_transform(self.user_item_matrix)

        print(f"****Selected Features Matrix for {self.file_name} has been saved****")
        print(f'The shape of selected features matrix: {selected_features_matrix.shape}')
        return selected_features_matrix


import numpy as np
import pandas as pd


class UserItemMaker:
    def __init__(self, path):
        self.path = path
        # The name of train and test files from MovieLense 100k dataset
        self.train_list = ['u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base', 'ua.base', 'ub.base']
        self.test_list = ['u1.test', 'u2.test', 'u3.test', 'u4.test', 'u5.test', 'ua.test', 'ub.test']

    def train(self):
        # Loop for changing training files to user-item matrix form and saving them in .csv files
        for i in range(len(self.train_list)):
            # print(train_list[i])
            column_names = ['user_id', 'item_id', 'rating', 'timestamp']
            file_path = f'{self.path}original_files/{self.train_list[i]}'

            data = pd.read_csv(file_path, sep='\t', names=column_names)

            # Based on documentation
            num_users = 943
            num_items = 1682

            user_item_matrix = np.zeros((num_users, num_items))

            for row in data.itertuples():
                user_item_matrix[row.user_id - 1, row.item_id - 1] = row.rating

            # Save the matrix to a file
            output_filename = f'{self.path}{self.train_list[i].replace(".base", "_train.csv")}'
            np.savetxt(output_filename, user_item_matrix, delimiter=',', fmt='%d')
        print('****The training datasets have been saved****')

    def test(self):
        # Loop for changing test files to user-item matrix form and saving them in .csv files
        for i in range(len(self.test_list)):
            # print(self.test_list[i])
            column_names = ['user_id', 'item_id', 'rating', 'timestamp']
            file_path = f'{self.path}original_files/{self.test_list[i]}'

            data = pd.read_csv(file_path, sep='\t', names=column_names)

            # Based on documentation
            num_users = 943
            num_items = 1682

            user_item_matrix = np.zeros((num_users, num_items))

            for row in data.itertuples():
                user_item_matrix[row.user_id - 1, row.item_id - 1] = row.rating

            # Save the matrix to a file
            output_filename = f'{self.path}{self.test_list[i].replace(".test", "_test.csv")}'
            np.savetxt(output_filename, user_item_matrix, delimiter=',', fmt='%d')
        print('****The testing datasets have been saved****')

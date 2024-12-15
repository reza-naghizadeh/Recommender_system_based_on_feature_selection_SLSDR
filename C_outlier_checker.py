import numpy as np


class OutlierChecker:
    def __init__(self, path):
        self.path = path
        # The name of train and test files from user-item matrices
        self.train_list = ['u1_train.csv', 'u2_train.csv', 'u3_train.csv', 'u4_train.csv', 'u5_train.csv', 'ua_train.csv', 'ub_train.csv']
        self.test_list = ['u1_test.csv', 'u2_test.csv', 'u3_test.csv', 'u4_test.csv', 'u5_test.csv', 'ua_test.csv', 'ub_test.csv']
        # Define the valid range of values
        self.valid_values = set(range(6))  # This will create a set with values {0, 1, 2, 3, 4, 5}

    def training_checker(self):
        # Loop through each train file in the train list
        for file_name in self.train_list:
            # Load the user-item matrix from the CSV file
            file_path = f'{self.path}{file_name}'
            data = np.genfromtxt(file_path, delimiter=',')

            # Check if there are any values outside the valid range
            if not np.all(np.isin(data, list(self.valid_values))):
                print(f"Warning: {file_path} contains outliers!")

            # Check for users with less than 10 ratings
            user_rating_counts = np.sum(data > 0, axis=1)  # Count non-zero ratings for each user
            users_with_less_than_10_ratings = np.where(user_rating_counts < 10)[0]  # Get users with less than 10 ratings

            if users_with_less_than_10_ratings.size > 0:
                    print(f"Warning: {file_name} has users with less than 10 ratings!")
                    print(f"Users with less than 10 ratings: {users_with_less_than_10_ratings}")

        print('****The training datasets has been checked****')

    def testing_checker(self):
        # Loop through each test file in the train list
        for file_name in self.test_list:
            # Load the user-item matrix from the CSV file
            file_path = f'{self.path}{file_name}'
            data = np.genfromtxt(file_path, delimiter=',')

            # Check if there are any values outside the valid range
            if not np.all(np.isin(data, list(self.valid_values))):
                print(f"Warning: {file_path} contains outliers!")

        print('****The testing datasets has been checked****')

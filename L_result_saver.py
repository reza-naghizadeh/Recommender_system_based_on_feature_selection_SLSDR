import csv
import os


class Results:
    def __init__(self, path, file_name, num_features, min_value, max_value, avg_value):
        self.path = path
        self.num_features = num_features
        self.min_value = min_value
        self.max_value = max_value
        self.avg_value = avg_value
        self.file_path = f'{path}results/{file_name}_results.csv'

    def initialize_csv(self):
        # Check if the file exists
        if not os.path.isfile(self.file_path):
            # If not, create it and write the header
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header row
                writer.writerow(['Num_Features', 'Min', 'Max', 'Avg'])

    def save_results_to_csv(self):
        # Open the file in append mode
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the data as a single row
            writer.writerow([self.num_features, self.min_value, self.max_value, self.avg_value])

    def activator(self):
        # Initialize the CSV file with header if it doesn't exist
        Results.initialize_csv(self)
        Results.save_results_to_csv(self)



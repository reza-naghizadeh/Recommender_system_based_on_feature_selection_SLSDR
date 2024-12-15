import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
file_path = "/Users/reza/University/thesis_final_codes/data_set/adjusted_cosine_RMSE_results.csv"  # Update this with the actual path to your CSV file
df = pd.read_csv(file_path)

# Step 2: Plot the data
plt.figure(figsize=(10, 6))

plt.plot(df['Num_Features'], df['Min_RMSE'], marker='o', label='Min RMSE')
plt.plot(df['Num_Features'], df['Max_RMSE'], marker='o', label='Max RMSE')
plt.plot(df['Num_Features'], df['Avg_RMSE'], marker='o', label='Avg RMSE')

plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Features')
plt.legend()
plt.grid(True)

# Step 3: Save the plot
output_file = "/Users/reza/University/thesis_final_codes/data_set/output_results_kmeans_slsdrcos.png"  # Update this with the desired path to save the plot
plt.savefig(output_file)

# If you still want to display the plot, you can uncomment the following line
# plt.show()

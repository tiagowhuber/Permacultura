import pandas as pd

# Define the file paths
fixedoutput_path = 'fixedoutput.csv'  # Update with your actual path
trainedoutput_path = 'trainedoutput.csv'  # Update with your actual path

# Define column names since trainedoutput.csv does not have a header
column_names = ['Yield (tonne/ha)', 'Seasonal irrigation (mm)']

# Read the fixedoutput.csv file
fixed_data = pd.read_csv(fixedoutput_path)

# Read the trainedoutput.csv file without a header
trained_data = pd.read_csv(trainedoutput_path, header=None, names=column_names)

# Calculate the average of Yield (tonne/ha) and Seasonal irrigation (mm) for fixedoutput.csv
average_yield_fixed = fixed_data['Yield (tonne/ha)'].mean()
average_irrigation_fixed = fixed_data['Seasonal irrigation (mm)'].mean()

# Print the results for fixedoutput.csv
print(f"Fixed Output - Average Yield (tonne/ha): {average_yield_fixed}")
print(f"Fixed Output - Average Seasonal Irrigation (mm): {average_irrigation_fixed}")

# Calculate the average of Yield (tonne/ha) and Seasonal irrigation (mm) for trainedoutput.csv
average_yield_trained = trained_data['Yield (tonne/ha)'].mean()
average_irrigation_trained = trained_data['Seasonal irrigation (mm)'].mean()

# Print the results for trainedoutput.csv
print(f"Trained Output - Average Yield (tonne/ha): {average_yield_trained}")
print(f"Trained Output - Average Seasonal Irrigation (mm): {average_irrigation_trained}")

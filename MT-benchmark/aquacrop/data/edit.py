import pandas as pd

# Define the file path
file_path = 'champion_climate.txt'  # Update with the correct path

# Read the champion_climate.txt file
data = pd.read_csv(file_path, delim_whitespace=True)

# Set all values in the 'Prcp(mm)' column to zero
data['Prcp(mm)'] = 0.0

# Save the modified dataframe to a new file
output_path = 'champion_climate_no_rain.txt'  # Update with the desired output path
data.to_csv(output_path, index=False, sep='\t')

print(f"Modified file saved to {output_path}")

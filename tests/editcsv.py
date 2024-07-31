import pandas as pd

# Create a list of dates from 1979/10/01 to 1985/05/30
dates = pd.date_range(start='1972-10-01', end='2024-05-30', freq='D')

# Create a DataFrame with "Date" and "Depth" columns
df = pd.DataFrame({'Date': dates, 'Depth': 0})

# Convert the "Date" column to pd.Datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Save the DataFrame to a CSV file
df.to_csv('D:/CLASES/Memoria de Titulo/Simulator/aquacrop/testsirrigatenever.csv', index=False)
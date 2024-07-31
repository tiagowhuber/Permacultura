import csv
from datetime import datetime, timedelta

start_date = datetime(1979, 9, 1)
end_date = datetime(1982, 5, 30)

# Generate a list of dates within the specified range
dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date)
    current_date += timedelta(days=1)

# Generate a list of irrigation amounts for each date
irrigation_amounts = [10] * len(dates)  # Replace 10 with the desired irrigation amount

# Write the data to a CSV file
filename = 'irrigation_data.csv'
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Amount'])
    for date, amount in zip(dates, irrigation_amounts):
        writer.writerow([date.strftime('%Y-%m-%d'), amount])

print(f"CSV file '{filename}' created successfully.")
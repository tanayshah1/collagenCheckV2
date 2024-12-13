import csv

with open('cleaned_test_data.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Read the header row
    for row in reader:
        print(row[0])  # Access the first column

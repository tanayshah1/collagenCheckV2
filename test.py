import pandas as pd

# Read and print CSV file content
df = pd.read_csv('cleaned_control_data.txt')
#print(df.head())
print(f'Number of columns: {df.shape[1]}')
print(f'Number of rows: {df.shape[0]}')

import numpy as np
import csv
import torch
import torch.nn as nn
from autoencoder import Autoencoder

def load_test_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader, None)  # Read the header if present
        print(f"CSV Header: {header}")  # Print header for debugging
        for row in reader:
            print(f"Row read: {row}")  # Print each row for debugging
            if len(row) < 2:
                continue  # Skip rows that don't have enough data
            try:
                values = list(map(float, row[1:]))  # Convert the rest of the row to floats
                data.append(values)
            except ValueError:
                print(f"Skipping row due to value error: {row}")
                continue
    
    # Print the data before converting to numpy array
    print(f"Data before conversion: {data}")

    # Convert to numpy array
    data_array = np.array(data)
    
    # Print the shape of the numpy array
    print(f"Data array shape: {data_array.shape}")
    
    return data_array

def detect_anomalies(autoencoder, data):
    autoencoder.eval()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Ensure the input dimension of the autoencoder matches the data
    if data_tensor.ndim < 2 or data_tensor.shape[1] != autoencoder.encoder[0].in_features:
        raise ValueError(f"Input dimension {data_tensor.shape[1] if data_tensor.ndim >= 2 else 'undefined'} does not match model's expected dimension {autoencoder.encoder[0].in_features}")

    with torch.no_grad():
        reconstructed = autoencoder(data_tensor)
        loss_fn = nn.MSELoss()
        loss = loss_fn(reconstructed, data_tensor).numpy()
    
    # Define a threshold for anomaly detection
    threshold = 0.1
    anomalies = np.where(loss > threshold)[0]
    return anomalies

# Load the test data
test_data = load_test_data('cleaned_test_data.csv')  # Ensure this is the correct file

# Print the shape and content of the loaded data
print(f"Test data shape: {test_data.shape}")
print(f"Test data sample: {test_data[:5]}")

# Initialize the autoencoder with the correct input dimension
if test_data.size == 0:
    raise ValueError("Test data is empty. Please check the file content.")

input_dim = test_data.shape[1]  # Number of features
autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load('autoencoder.pth'))

# Detect anomalies
anomalies = detect_anomalies(autoencoder, test_data)

# Save anomalies to a text file
with open('anomalies_detected.txt', 'w') as file:
    for index in anomalies:
        file.write(f"Line Number: {index + 1}\n")  # Line number adjustment (1-based)
        file.write(f"Anomaly Data: {' '.join(map(str, test_data[index]))}\n")
print("Anomalies detected and saved to 'anomalies_detected.txt'.")

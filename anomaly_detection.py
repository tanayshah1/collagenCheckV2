import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
from dataset import customDataset  # Ensure this import matches your file structure

def detect_anomalies(autoencoder, data_loader):
    autoencoder.eval()
    losses = []

    # Define loss function
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for data_batch in data_loader:
            reconstructed = autoencoder(data_batch)
            loss = loss_fn(reconstructed, data_batch)
            losses.extend(loss.numpy())

    # Define a threshold for anomaly detection
    threshold = 0.1
    anomalies = [i for i, loss in enumerate(losses) if loss > threshold]
    return anomalies

# Load the dataset
dataset = customDataset('cleaned_test_data.csv')
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust batch_size as needed

# Print the shape and content of the loaded data (for debugging)
print(f"Number of samples in dataset: {len(dataset)}")

# Check the first sample
if len(dataset) > 0:
    print(f"First sample: {dataset[0]}")

# Initialize the autoencoder with the correct input dimension
input_dim = dataset[0].shape[0] if len(dataset) > 0 else 0  # Number of features from the first sample
if input_dim == 0:
    raise ValueError("Dataset is empty. Cannot determine input dimension.")

autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load('autoencoder.pth'))

# Detect anomalies
anomalies = detect_anomalies(autoencoder, data_loader)

# Save anomalies to a text file
with open('anomalies_detected.txt', 'w') as file:
    for index in anomalies:
        file.write(f"Line Number: {index + 1}\n")  # Line number adjustment (1-based)
        file.write(f"Anomaly Data: {' '.join(map(str, dataset[index].numpy()))}\n")
print("Anomalies detected and saved to 'anomalies_detected.txt'.")

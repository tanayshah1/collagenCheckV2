import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from autoencoder import RatioAutoencoder

class RatioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.values[:, 1:]  # Assuming the first column is the line number
        self.X = torch.tensor(self.X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

def train_model(train_csv, model_save_path, epochs=20, batch_size=64):
    dataset = RatioDataset(train_csv)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RatioAutoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), model_save_path)

# Example usage
train_model('cleaned_control_data.csv', 'autoencoder_model.pth')

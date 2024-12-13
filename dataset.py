import torch
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        print(f"Loading data from {file_path}")
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as file:
                for line in file:
                    try:
                        # Split the line by spaces and convert to float
                        values = list(map(float, line.strip().split()))
                        self.data.append(values)
                    except ValueError as e:
                        print(f"Skipping line due to value error: {line}. Error: {e}")
        
        except Exception as e:
            print(f"Error reading file: {e}")

        # Convert to tensor
        if not self.data:
            print("Warning: No data was loaded.")
        
        self.data = torch.tensor(self.data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

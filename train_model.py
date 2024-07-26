import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load the cleaned control data
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip header line
        for line in file:
            parts = line.strip().split()
            line_number = int(parts[0])  # Extract the line number
            values = list(map(float, parts[1:]))  # Convert the rest of the parts to floats
            data.append(values)
    return np.array(data)

# Load and prepare data
control_data = load_data('cleaned_control_data.csv')  # Ensure this is the correct file

# Check if the data has at least one feature
if control_data.shape[1] <= 0:
    raise ValueError("The cleaned control data does not contain enough features.")

X = control_data  # Features
y = np.zeros(X.shape[0])  # Dummy target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model
model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'trained_model.pkl')
print("Model trained and saved as 'trained_model.pkl'.")

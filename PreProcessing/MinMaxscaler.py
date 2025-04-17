from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Sample data (3 samples, 2 features)
data = np.array([[10, 200], [15, 300], [20, 400]])

# Initialize MinMaxScaler (default feature range is 0 to 1)
scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(data)

# Print the scaled data
print("Original Data:\n", data)
print("\nScaled Data:\n", scaled_data)

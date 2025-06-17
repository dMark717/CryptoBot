import pandas as pd

# Load the data (replace 'btc_data.csv' with the actual file path if necessary)
data = pd.read_csv('btc_data.csv')

# Convert the 'Timestamp' column to datetime format
data['Datetime'] = pd.to_datetime(data['Timestamp'], unit='s')

# Split the data into training and testing sets
train_data = data[(data['Datetime'].dt.year >= 2020) & (data['Datetime'].dt.year <= 2024)]
test_data = data[data['Datetime'].dt.year == 2025]

# Check the splits
print(f"Training Data:\n{train_data.head()}")
print(f"Testing Data:\n{test_data.head()}")

# Save the data splits to separate CSVs (optional)
train_data.to_csv('btc_data_train.csv', index=False)
test_data.to_csv('btc_data_test.csv', index=False)

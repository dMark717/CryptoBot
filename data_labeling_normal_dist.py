import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('btc_data_train.csv')

# Ensure sorted by timestamp
df = df.sort_values('Timestamp').reset_index(drop=True)

# Parameters
lookahead = 60
gain_threshold = 0.05

# Compute future max high and min low (shifted to look forward)
future_max_high = df['High'].shift(-1).rolling(window=lookahead, min_periods=1).max()
future_min_low = df['Low'].shift(-1).rolling(window=lookahead, min_periods=1).min()

# Current close price
current_close = df['Close']

# Compute gains and drops
gain = (future_max_high - current_close) / current_close
drop = (future_min_low - current_close) / current_close

# Normalize to 0–1 (up) and 0–(-1) (down)
upward_potential = gain.clip(lower=0, upper=gain_threshold) / gain_threshold
downward_potential = drop.clip(upper=0, lower=-gain_threshold) / gain_threshold  # negative

# Fill potential NaNs
upward_potential = upward_potential.fillna(0)
downward_potential = downward_potential.fillna(0)

# Calculate final Score
df['Score'] = upward_potential - abs(downward_potential)

# Drop rows without full future window
df = df.iloc[:-lookahead].reset_index(drop=True)

# Save only original columns + Score
original_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']
df[original_cols + ['Score']].to_csv('btc_data_train_labeled.csv', index=False)

print("Saved labeled data with Score column.")

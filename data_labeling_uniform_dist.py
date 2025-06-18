import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# Load the dataset
df = pd.read_csv('btc_data_test.csv')

# Ensure sorted by timestamp
df = df.sort_values('Timestamp').reset_index(drop=True)

# Parameters
lookahead = 120
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

unscaled_score = upward_potential - abs(downward_potential)

# Scale to uniform distribution using 200 buckets
n_buckets = 200

# Remove NaN values for ranking
valid_scores = unscaled_score.dropna()

# Calculate percentile ranks (0 to 1)
percentile_ranks = valid_scores.rank(method='average') / len(valid_scores)

# Scale to bucket numbers (0 to 199)
bucket_indices = (percentile_ranks * n_buckets).clip(0, n_buckets - 1).astype(int)

# The scaled_score is the bucket number, then normalized to -1 to 1 range
bucket_scores = bucket_indices
normalized_scores = (bucket_scores - 99.5) / 100

scaled_score = pd.Series(index=unscaled_score.index, dtype=float)
scaled_score.loc[valid_scores.index] = normalized_scores

# Fill any remaining NaN values with 0
scaled_score = scaled_score.fillna(0)

# Calculate final Score
df['Score'] = scaled_score

# Drop rows without full future window
df = df.iloc[:-lookahead].reset_index(drop=True)

# Save only original columns + Score
original_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']
df[original_cols + ['Score']].to_csv('btc_data_test_labeled.csv', index=False)

print("Saved labeled data.")
print(f"Score range: {df['Score'].min():.3f} to {df['Score'].max():.3f}")
print(f"Score mean: {df['Score'].mean():.3f}")
print(f"Score std: {df['Score'].std():.3f}")


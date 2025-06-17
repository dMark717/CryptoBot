import pandas as pd
import numpy as np
import ta

# Load labeled data
df = pd.read_csv('btc_data_train_labeled.csv', parse_dates=['Datetime'])

# Ensure data is sorted
df = df.sort_values('Timestamp').reset_index(drop=True)

# Define rolling windows
windows = [5, 15, 30, 60, 120, 180, 360]

# Lag features
for lag in [1, 2, 3, 5, 10, 30, 60, 120, 180, 360]:
    df[f'lag_{lag}'] = df['Close'].shift(lag)

# Rolling statistics
for window in windows:
    df[f'roll_mean_{window}'] = df['Close'].rolling(window).mean()
    df[f'roll_std_{window}'] = df['Close'].rolling(window).std()
    df[f'roll_min_{window}'] = df['Close'].rolling(window).min()
    df[f'roll_max_{window}'] = df['Close'].rolling(window).max()
    df[f'roll_median_{window}'] = df['Close'].rolling(window).median()
    df[f'roll_skew_{window}'] = df['Close'].rolling(window).skew()
    df[f'roll_kurt_{window}'] = df['Close'].rolling(window).kurt()

# Price change features
for period in windows:
    df[f'pct_change_{period}'] = df['Close'].pct_change(periods=period)

# Volume-based features
for window in windows:
    df[f'vol_mean_{window}'] = df['Volume'].rolling(window).mean()
    df[f'vol_std_{window}'] = df['Volume'].rolling(window).std()
    df[f'vol_spike_{window}'] = df['Volume'] / df[f'vol_mean_{window}']

# Time-based features
df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['day_of_month'] = df['Datetime'].dt.day
df['month'] = df['Datetime'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Technical indicators
df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
macd = ta.trend.MACD(close=df['Close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()
boll = ta.volatility.BollingerBands(close=df['Close'], window=20)
df['bollinger_mavg'] = boll.bollinger_mavg()
df['bollinger_hband'] = boll.bollinger_hband()
df['bollinger_lband'] = boll.bollinger_lband()

# Drop rows with insufficient history
df = df.iloc[360:].reset_index(drop=True)

# Select feature columns (excluding raw OHLCV and datetime fields)
drop_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Datetime']
feature_cols = [col for col in df.columns if col not in drop_cols]
df[feature_cols].to_csv('btc_data_train_labeled_features.csv', index=False)

print("Saved training data.")

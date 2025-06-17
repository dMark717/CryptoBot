import pandas as pd
import numpy as np
from xgboost import XGBRegressor, DMatrix
import ta

# Load and train
df_train = pd.read_csv("btc_data_train_labeled_features.csv")
X_train = df_train.drop(columns=["Score"])
y_train = df_train["Score"]

model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# Save the model as a binary file
model.save_model("btc_xgb_model.bin")

# Load test data
df_raw = pd.read_csv("btc_data_test_labeled.csv", parse_dates=["Datetime"])
lookback = 360
windows = [5, 15, 30, 60, 120, 180, 360]

# Causal loop
for t in range(lookback, len(df_raw)):
    window = df_raw.iloc[t - lookback : t].copy()
    current = df_raw.iloc[t]
    features = {}

    # Lag features
    for lag in [1, 2, 3, 5, 10, 30, 60, 120, 180, 360]:
        features[f"lag_{lag}"] = window["Close"].iloc[-lag]

    # Rolling stats
    for w in windows:
        features[f"roll_mean_{w}"] = window["Close"].rolling(w).mean().iloc[-1]
        features[f"roll_std_{w}"] = window["Close"].rolling(w).std().iloc[-1]
        features[f"roll_min_{w}"] = window["Close"].rolling(w).min().iloc[-1]
        features[f"roll_max_{w}"] = window["Close"].rolling(w).max().iloc[-1]
        features[f"roll_median_{w}"] = window["Close"].rolling(w).median().iloc[-1]
        features[f"roll_skew_{w}"] = window["Close"].rolling(w).skew().iloc[-1]
        features[f"roll_kurt_{w}"] = window["Close"].rolling(w).kurt().iloc[-1]

    # Percentage changes
    for p in windows:
        features[f"pct_change_{p}"] = window["Close"].pct_change(p).iloc[-1]

    # Volume-based
    for w in windows:
        v_mean = window["Volume"].rolling(w).mean().iloc[-1]
        v_std = window["Volume"].rolling(w).std().iloc[-1]
        features[f"vol_mean_{w}"] = v_mean
        features[f"vol_std_{w}"] = v_std
        features[f"vol_spike_{w}"] = window["Volume"].iloc[-1] / v_mean if v_mean != 0 else 0

    # Time-based
    dt = current["Datetime"]
    features["hour"] = dt.hour
    features["day_of_week"] = dt.dayofweek
    features["day_of_month"] = dt.day
    features["month"] = dt.month
    features["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
    features["day_of_week_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

    # Technical indicators
    features["rsi"] = ta.momentum.RSIIndicator(window["Close"], window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(window["Close"])
    features["macd"] = macd.macd().iloc[-1]
    features["macd_signal"] = macd.macd_signal().iloc[-1]
    features["macd_diff"] = macd.macd_diff().iloc[-1]
    boll = ta.volatility.BollingerBands(close=window["Close"], window=20)
    features["bollinger_mavg"] = boll.bollinger_mavg().iloc[-1]
    features["bollinger_hband"] = boll.bollinger_hband().iloc[-1]
    features["bollinger_lband"] = boll.bollinger_lband().iloc[-1]

    # Predict
    X_point = pd.DataFrame([features], columns=X_train.columns)
    pred = model.predict(X_point)[0]
    true = current["Score"]
    err = abs(pred - true)
    rel_err = err / abs(true) if true != 0 else np.nan

    # Confidence (SHAP strength)
    contribs = model.get_booster().predict(DMatrix(X_point), pred_contribs=True)
    shap_strength = np.sum(np.abs(contribs), axis=1)[0]

    # Output
    print(f"[{current['Datetime']}]")
    print(f"  Predicted Score : {pred:.5f}")
    print(f"  Actual Score    : {true:.5f}")
    print(f"  Abs Error       : {err:.5f}")
    print(f"  Rel Error       : {rel_err:.2%}" if not np.isnan(rel_err) else "  Rel Error: NaN")
    print(f"  Confidence (SHAP strength): {shap_strength:.5f}")
    print("-" * 50)

import pandas as pd
import numpy as np
import xgboost as xgb
import ta

# === USER CONFIGURABLE PARAMETERS ===

TEST_BEGIN = 0
TEST_END = 40000

INITIAL_BALANCE = 100.0
LEVERAGE = 5.0
TP_PROFIT_USD = 2.0
SL_SCALE = 0.5

PRED_LOWER_LIMIT = 0.9
PRED_MAX = 1.0
TP_MIN = 0.001
TP_MAX = 0.001
PRED_UPPER_LIMIT = 1.0
SHAP_THRESHOLD = 0.9
SHAP_UPPER_LIMIT = 1.0

CAUTION_DURATION = 360
MIN_TRADE_INTERVAL_MINUTES = 30

# === PERMANENT PARAMETERS ===

FEE_PCT = 0.000315
MAX_TRADE_DURATION_MINUTES = 8
LOOKBACK = 360
WINDOWS = [5, 15, 30, 60, 120, 180, 360]

# === LOAD MODEL ===
model = xgb.Booster()
model.load_model("trading_ai.bin")

# === LOAD DATA ===
df = pd.read_csv("btc_data_test.csv", parse_dates=["Datetime"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df = df.iloc[TEST_BEGIN:min(TEST_END, len(df))].copy()

balance = INITIAL_BALANCE
open_margin = 0.0
trades = []
open_trades = []
last_stop_time = None
last_trade_time = None
trade_counter = 1

def build_features(window, current):
    features = {}
    for w in WINDOWS:
        features[f"roll_mean_{w}"] = window["Close"].rolling(w).mean().iloc[-1]
        features[f"roll_std_{w}"] = window["Close"].rolling(w).std().iloc[-1]
        features[f"roll_min_{w}"] = window["Close"].rolling(w).min().iloc[-1]
        features[f"roll_max_{w}"] = window["Close"].rolling(w).max().iloc[-1]
        features[f"roll_median_{w}"] = window["Close"].rolling(w).median().iloc[-1]
        features[f"roll_skew_{w}"] = window["Close"].rolling(w).skew().iloc[-1]
        features[f"roll_kurt_{w}"] = window["Close"].rolling(w).kurt().iloc[-1]
    for p in WINDOWS:
        features[f"pct_change_{p}"] = window["Close"].pct_change(p).iloc[-1]
    for w in WINDOWS:
        v_mean = window["Volume"].rolling(w).mean().iloc[-1]
        features[f"vol_mean_{w}"] = v_mean
        features[f"vol_std_{w}"] = window["Volume"].rolling(w).std().iloc[-1]
        features[f"vol_spike_{w}"] = window["Volume"].iloc[-1] / v_mean if v_mean != 0 else 0
    dt_feat = current["Datetime"]
    features["rsi"] = ta.momentum.RSIIndicator(window["Close"], window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(window["Close"])
    features["macd"] = macd.macd().iloc[-1]
    features["macd_signal"] = macd.macd_signal().iloc[-1]
    features["macd_diff"] = macd.macd_diff().iloc[-1]
    boll = ta.volatility.BollingerBands(close=window["Close"], window=20)
    features["bollinger_mavg"] = boll.bollinger_mavg().iloc[-1]
    features["bollinger_hband"] = boll.bollinger_hband().iloc[-1]
    features["bollinger_lband"] = boll.bollinger_lband().iloc[-1]
    return features

# Removed calculate_shap_scaled_tp_profit function

for t in range(LOOKBACK, len(df) - 1):
    current = df.iloc[t]
    dt = current["Datetime"]

    # --- Close open trades ---
    to_close = []
    for trade in open_trades:
        idx_now = t
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        direction = trade["direction"]
        entry_type = trade["type"]
        tp_pct = trade["tp_pct"]
        sl_pct = trade["sl_pct"]
        trade_id = trade["trade_id"]

        high = df.iloc[idx_now]["High"]
        low = df.iloc[idx_now]["Low"]
        close = df.iloc[idx_now]["Close"]
        now_time = df.iloc[idx_now]["Datetime"]

        duration = (now_time - entry_time).total_seconds() / 60
        resolved = False
        pnl_pct_val = None
        result_type = None

        if direction == 1:
            if high >= entry_price * (1 + tp_pct + FEE_PCT):
                pnl_pct_val = tp_pct - FEE_PCT
                result_type = f"LONG TP {trade_id}"
                resolved = True
            elif low <= entry_price * (1 + sl_pct - FEE_PCT):
                pnl_pct_val = sl_pct - FEE_PCT
                result_type = f"LONG SL {trade_id}"
                resolved = True
        else:
            if low <= entry_price * (1 - tp_pct - FEE_PCT):
                pnl_pct_val = tp_pct - FEE_PCT
                result_type = f"SHORT TP {trade_id}"
                resolved = True
            elif high >= entry_price * (1 - sl_pct + FEE_PCT):
                pnl_pct_val = sl_pct - FEE_PCT
                result_type = f"SHORT SL {trade_id}"
                resolved = True
        
        if not resolved and duration > MAX_TRADE_DURATION_MINUTES:
            exit_price = close
            exit_price_adj = exit_price * (1 - FEE_PCT) if direction == 1 else exit_price * (1 + FEE_PCT)
            pnl_pct_val = direction * ((exit_price_adj - entry_price) / entry_price)
            result_type = f"{entry_type.upper()} TIMEOUT {trade_id}"
            resolved = True

        if resolved:
            profit = position_size * pnl_pct_val * LEVERAGE
            balance += profit
            open_margin -= position_size
            trades.append({
                "type": entry_type,
                "pnl": pnl_pct_val,
                "profit": profit,
                "size": position_size,
                "leverage": LEVERAGE,
                "entry_shap": trade["entry_shap"],
                "entry_pred": trade["entry_pred"],
                "entry_time": entry_time,
                "exit_time": now_time,
                "balance": balance,
                "result": result_type
            })
            if 'SL' in result_type:
                print(f"[{now_time}] {result_type} | PnL: {pnl_pct_val:.4f} | Loss: {profit:.2f} | New Balance: ${balance:.2f}")
                trade_duration = (now_time - entry_time).total_seconds() / 60
                remaining_caution = max(0, CAUTION_DURATION - trade_duration)
                if remaining_caution > 0:
                    last_stop_time = now_time
                else:
                    last_stop_time = None
            else:
                print(f"[{now_time}] {result_type} | PnL: {pnl_pct_val:.4f} | Profit: {profit:.2f} | New Balance: ${balance:.2f}")
            to_close.append(trade)

    open_trades = [tr for tr in open_trades if tr not in to_close]

    # === Pre-Trade Checks ===

    # 1. Minimum Trade Interval Check
    if last_trade_time is not None and \
       (dt - last_trade_time).total_seconds() / 60 < MIN_TRADE_INTERVAL_MINUTES:
        continue

    # 2. Caution Period Check & Block
    in_caution_period = False
    if last_stop_time is not None:
        minutes_since_stop = (dt - last_stop_time).total_seconds() / 60
        if minutes_since_stop < CAUTION_DURATION:
            in_caution_period = True
        else:
            last_stop_time = None # Caution period ended
    
    if in_caution_period:
        continue 

    # --- Feature Extraction & Prediction (only if not in caution) ---
    window = df.iloc[t - LOOKBACK:t].copy()
    features = build_features(window, current)
    X_point = pd.DataFrame([features])
    dmatrix = xgb.DMatrix(X_point)
    pred = model.predict(dmatrix)[0]
    
    # 3. Prediction Range Check
    if not (PRED_LOWER_LIMIT <= abs(pred) <= PRED_UPPER_LIMIT):
        continue

    pred_magnitude_for_tp = min(abs(pred), PRED_MAX)
    if (PRED_MAX - PRED_LOWER_LIMIT) == 0: 
        scale_for_tp = 0.0 if PRED_LOWER_LIMIT == PRED_MAX else 0.5 
    else:
        scale_for_tp = (pred_magnitude_for_tp - PRED_LOWER_LIMIT) / (PRED_MAX - PRED_LOWER_LIMIT)
    
    current_tp_pct = TP_MIN + scale_for_tp * (TP_MAX - TP_MIN)
    current_net_tp_pct = current_tp_pct - FEE_PCT

    # --- SHAP Calculation (Most Expensive Step) ---
    shap_strength = np.sum(np.abs(model.predict(dmatrix, pred_contribs=True)), axis=1)[0]
    
    # === Trade Entry Conditions (Post-SHAP) ===

    # 4. SHAP Strength Check
    if not (SHAP_THRESHOLD <= shap_strength <= SHAP_UPPER_LIMIT):
        continue

    direction = 1 if pred > 0 else -1
    entry_type = "long" if direction == 1 else "short"
    entry_price = current["Close"]

    # scaled_tp_profit_usd is now TP_PROFIT_USD directly
    scaled_tp_profit_usd = TP_PROFIT_USD 
    
    position_size = scaled_tp_profit_usd / (current_net_tp_pct * LEVERAGE)

    target_loss = -scaled_tp_profit_usd * SL_SCALE
    sl_pct = (target_loss / (position_size * LEVERAGE)) + FEE_PCT

    # 5. Available Balance Check
    available_balance = balance - open_margin
    if position_size > available_balance:
        continue

    # --- Execute Trade ---
    # Removed Scale from print statement as it's always 1.0x now
    print(f"[{dt}] OPENING {entry_type.upper()} {trade_counter} | Pred: {pred:.4f} | SHAP: {shap_strength:.4f}")
    
    open_margin += position_size
    open_trades.append({
        "type": entry_type,
        "direction": direction,
        "entry_price": entry_price,
        "entry_time": dt,
        "entry_idx": t,
        "position_size": position_size,
        "tp_pct": current_tp_pct,
        "sl_pct": sl_pct,
        "entry_shap": shap_strength,
        "entry_pred": pred,
        "trade_id": trade_counter
    })
    last_trade_time = dt
    trade_counter += 1

# === Final forced exits ===
if open_trades:
    last_dt_val = df.iloc[-1]["Datetime"]
    for trade in list(open_trades): 
        idx_now = len(df) - 1
        entry_time = trade["entry_time"]
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        direction = trade["direction"]
        entry_type = trade["type"]
        trade_id = trade["trade_id"]
        close = df.iloc[idx_now]["Close"]
        now_time = last_dt_val

        exit_price_adj = close * (1 - FEE_PCT) if direction == 1 else close * (1 + FEE_PCT)
        pnl_pct_val = direction * ((exit_price_adj - entry_price) / entry_price) 
        profit = position_size * pnl_pct_val * LEVERAGE
        balance += profit
        open_margin -= position_size
        trades.append({
            "type": entry_type,
            "pnl": pnl_pct_val, 
            "profit": profit,
            "size": position_size,
            "leverage": LEVERAGE,
            "entry_shap": trade["entry_shap"],
            "entry_pred": trade["entry_pred"],
            "entry_time": entry_time,
            "exit_time": now_time,
            "balance": balance,
            "result": f"{entry_type.upper()} FORCED EXIT {trade_id}"
        })
        print(f"[{now_time}] {entry_type.upper()} FORCED EXIT {trade_id} | PnL: {pnl_pct_val:.4f} | Profit: {profit:.2f} | New Balance: ${balance:.2f}")
    open_trades.clear()

# === Summary ===
if trades:
    total_return_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
    avg_profit = np.mean([t['profit'] for t in trades])
    win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades)
    longs = [t for t in trades if t['type'] == 'long']
    shorts = [t for t in trades if t['type'] == 'short']
    long_winrate = sum(1 for t in longs if t['profit'] > 0) / len(longs) if longs else 0
    short_winrate = sum(1 for t in shorts if t['profit'] > 0) / len(shorts) if shorts else 0
    print("\n--- Trading Summary ---")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total Return: {total_return_pct:.2%}")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Long Win Rate: {long_winrate:.2%} | Short Win Rate: {short_winrate:.2%}")
else:
    print("No trades executed.")
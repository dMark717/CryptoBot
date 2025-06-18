import pandas as pd
import numpy as np
import ta

# === CONFIG ===
INITIAL_BALANCE = 100.0
LEVERAGE = 5.0
FEE_PCT = 0.000315
POSITION_FRAC = 0.95
MAX_TRADE_DURATION_MINUTES = 240

TP_PCT = 0.02   # 2% take-profit
SL_PCT = 0.01   # 1% stop-loss

# === LOAD DATA ===
df = pd.read_csv("btc_data_test.csv", parse_dates=["Datetime"])
df = df.set_index("Datetime").sort_index().copy()

# Ensure High/Low columns exist
if "High" not in df.columns or "Low" not in df.columns:
    df["High"] = df["Close"]
    df["Low"] = df["Close"]

# === INDICATORS ===
df["ema_20"] = ta.trend.ema_indicator(df["Close"], window=20)
macd = ta.trend.MACD(df["Close"])
df["macd_hist"] = macd.macd_diff()
df["adx"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

# === STRATEGY ===
balance = INITIAL_BALANCE
open_trade = None
trades = []

for i in range(1, len(df)):
    row = df.iloc[i]
    now = row.name
    price = row["Close"]

    # --- Exit Logic ---
    if open_trade:
        duration = (now - open_trade["entry_time"]).total_seconds() / 60
        direction = open_trade["direction"]
        entry_price = open_trade["entry_price"]
        position_size = open_trade["position_size"]

        if direction == 1:  # LONG
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)
        else:  # SHORT
            tp_price = entry_price * (1 - TP_PCT)
            sl_price = entry_price * (1 + SL_PCT)

        high = row["High"]
        low = row["Low"]
        resolved = False

        if direction == 1:
            if high >= tp_price:
                pnl_pct = (tp_price - entry_price) / entry_price - FEE_PCT
                result = "LONG TP"
                resolved = True
            elif low <= sl_price:
                pnl_pct = (sl_price - entry_price) / entry_price - FEE_PCT
                result = "LONG SL"
                resolved = True
        else:
            if low <= tp_price:
                pnl_pct = (entry_price - tp_price) / entry_price - FEE_PCT
                result = "SHORT TP"
                resolved = True
            elif high >= sl_price:
                pnl_pct = (entry_price - sl_price) / entry_price - FEE_PCT
                result = "SHORT SL"
                resolved = True

        if not resolved and duration >= MAX_TRADE_DURATION_MINUTES:
            exit_price = price
            exit_adj = exit_price * (1 - FEE_PCT) if direction == 1 else exit_price * (1 + FEE_PCT)
            pnl_pct = direction * ((exit_adj - entry_price) / entry_price)
            result = f"{'LONG' if direction == 1 else 'SHORT'} TIMEOUT"
            resolved = True

        if resolved:
            profit = pnl_pct * position_size * LEVERAGE
            balance += profit
            trades.append({
                "type": "long" if direction == 1 else "short",
                "entry_time": open_trade["entry_time"],
                "exit_time": now,
                "entry_price": entry_price,
                "exit_price": price,
                "return_pct": pnl_pct,
                "profit": profit,
                "balance": balance,
                "result": result
            })
            print(f"[{now}] {result} | PnL: {pnl_pct:.4f} | Profit: {profit:.2f} | New Balance: ${balance:.2f}")
            open_trade = None

    # --- Entry Logic ---
    if not open_trade:
        if (
            row["Close"] > row["ema_20"]
            and row["macd_hist"] > 0
            and row["adx"] > 20
            and not np.isnan(row["atr"])
        ):
            direction = 1  # long only
            position_size = balance * POSITION_FRAC
            entry_price = price
            open_trade = {
                "direction": direction,
                "entry_price": entry_price,
                "entry_time": now,
                "position_size": position_size
            }
            print(f"[{now}] OPEN LONG | Price: {entry_price:.2f} | Size: {position_size:.2f}")

# === SUMMARY ===
if trades:
    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
    win_rate = sum(1 for t in trades if t["profit"] > 0) / len(trades)
    print("\n--- EMA/MACD/ADX Strategy Summary ---")
    print(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"Total Return:    {total_return:.2%}")
    print(f"Total Trades:    {len(trades)}")
    print(f"Win Rate:        {win_rate:.2%}")
else:
    print("No trades executed.")

import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------------------
# CONFIG
# ---------------------------------------
TRAIN_PERIODS = [
    ("2016-01-01", "2020-02-28"),
    ("2020-06-01", "2021-03-31"),
    ("2023-01-01", "2024-12-31")
]

TEST_PERIODS = [
    ("2021-04-01", "2022-12-31")
]

# Black-swan periods to remove
REMOVE_PERIODS = [
    ("2020-03-01", "2020-05-31"),   # COVID crash
    ("2021-05-01", "2021-06-30"),   # Mining ban
    ("2021-11-01", "2022-12-31")    # LUNA/3AC/FTX
]

# ---------------------------------------
# DOWNLOAD DATA
# ---------------------------------------

print("📥 Downloading BTC-USD from Yahoo...")
df = yf.download("BTC-USD", start="2014-01-01", end="2025-01-01")
df = df.dropna()

# ---------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------

# Log returns
df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))

# Rolling volatility
df['vol_7']  = df['log_ret'].rolling(7).std()
df['vol_14'] = df['log_ret'].rolling(14).std()
df['vol_30'] = df['log_ret'].rolling(30).std()

# Volatility momentum
df['vol_chg_7'] = df['vol_7'] - df['vol_7'].shift(7)

# Volume features
df['log_vol'] = np.log(df['Volume'] + 1)

# ---------------------------------------
# TARGET: 14-day realized volatility
# ---------------------------------------

def realized_vol(series, window=14):
    """
    Computes future realized volatility:
    sqrt( average of squared future returns )
    """
    return np.sqrt(series.rolling(window).apply(lambda x: np.mean(x**2), raw=True))

df['rv_14'] = realized_vol(df['log_ret'].shift(-1), window=14)

# Drop rows with missing values
df = df.dropna()

# ---------------------------------------
# REMOVE BLACK-SWAN PERIODS
# ---------------------------------------

def remove_periods(data, remove_list):
    for start, end in remove_list:
        mask = (data.index >= start) & (data.index <= end)
        data = data.loc[~mask]
    return data

df_clean = remove_periods(df, REMOVE_PERIODS)

# ---------------------------------------
# SPLIT INTO TRAIN / TEST
# ---------------------------------------

def select_periods(data, period_list):
    frames = []
    for start, end in period_list:
        mask = (data.index >= start) & (data.index <= end)
        frames.append(data.loc[mask])
    return pd.concat(frames)

train_df = select_periods(df_clean, TRAIN_PERIODS)
test_df  = select_periods(df_clean, TEST_PERIODS)

# ---------------------------------------
# SAVE DATASETS
# ---------------------------------------

train_df.to_csv("btc_train_14d.csv")
test_df.to_csv("btc_test_14d.csv")

print("✅ Dataset created!")
print("   → btc_train_14d.csv")
print("   → btc_test_14d.csv")
print()
print("Train shape:", train_df.shape)
print("Test  shape:", test_df.shape)

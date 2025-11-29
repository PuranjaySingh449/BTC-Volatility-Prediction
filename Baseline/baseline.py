import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ----------------------------
# Paths
# ----------------------------
train_path = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path  = r"D:\btc-volatility v2\btc_test_14d.csv"


# ----------------------------
# Load Data
# ----------------------------
def load_csv(path):
    df = pd.read_csv(path, skiprows=2)
    df.columns = [
        "Date", "Close", "High", "Low", "Open", "Volume", "log_ret",
        "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol", "rv_14"
    ]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


train_df = load_csv(train_path)
test_df = load_csv(test_path)


# ----------------------------
# Target Column: 14-day Realized Volatility (rv_14)
# ----------------------------
y_train = train_df["rv_14"].values
y_test = test_df["rv_14"].values
dates_test = test_df["Date"].values


# -------------------------------
# Baseline 1: Mean Volatility Predictor
# Predict rv_14 as the mean of training rv_14
# -------------------------------
mean_vol = np.mean(y_train)
mean_pred = np.full_like(y_test, mean_vol)


# -------------------------------
# Baseline 2: Random Walk (Naive Persistence)
# Predict next volatility = previous actual test volatility
# rv_14_pred[t] = rv_14_actual[t-1]
# First prediction uses last training volatility
# -------------------------------
random_walk_pred = np.zeros_like(y_test)
random_walk_pred[0] = y_train[-1]  # seed with last train volatility
random_walk_pred[1:] = y_test[:-1]  # naive: previous actual value


# -------------------------------
# Baseline 3: Last Known Volatility (Static)
# Predict all test volatility = last training volatility
# -------------------------------
last_train_vol = y_train[-1]
last_known_pred = np.full_like(y_test, last_train_vol)


# -------------------------------
# Metrics Function
# -------------------------------
def print_metrics(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print("\n" + "="*60)
    print(f"METRICS: {label}")
    print("="*60)
    print(f"MSE:         {mse:.6f}")
    print(f"RMSE:        {rmse:.6f}")
    print(f"MAE:         {mae:.6f}")
    print(f"MAPE:        {mape:.2f}%")
    print(f"R²:          {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
    print("="*60)
    return mae, rmse, mape, r2, corr


# -------------------------------
# Evaluate Baselines
# -------------------------------
print_metrics(y_test, mean_pred, "Mean Baseline (train mean volatility)")
print_metrics(y_test, random_walk_pred, "Random Walk Naive (previous volatility)")
print_metrics(y_test, last_known_pred, "Last Known Volatility (constant)")


# -------------------------------
# Plot Volatility Predictions
# -------------------------------
plt.figure(figsize=(14,6))
plt.plot(dates_test, y_test, label="Actual rv_14", color='blue', linewidth=2)
plt.plot(dates_test, mean_pred, label="Mean Baseline", color='green', linestyle="--")
plt.plot(dates_test, random_walk_pred, label="Random Walk Baseline", color='orange', linestyle=":")
plt.plot(dates_test, last_known_pred, label="Last Known Volatility", color='red', linestyle="-.")
plt.title("BTC 14-Day Realized Volatility (rv_14) - Baseline Comparison")
plt.xlabel("Date")
plt.ylabel("14-Day Realized Volatility")
plt.legend(fontsize=11)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.show()

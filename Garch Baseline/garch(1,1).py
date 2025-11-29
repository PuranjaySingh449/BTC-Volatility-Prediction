import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# -----------------------------
# Paths and load CSVs
# -----------------------------
train_path = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path  = r"D:\btc-volatility v2\btc_test_14d.csv"


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


# -----------------------------
# Use log returns (scaled by 100 for GARCH stability)
# -----------------------------
returns_train = train_df["log_ret"].dropna() * 100
returns_test = test_df["log_ret"].dropna() * 100


# -----------------------------
# Fit GARCH(1,1) on training returns
# -----------------------------
garch_model = arch_model(returns_train, vol='Garch', p=1, q=1, mean="Zero", rescale=False)
garch_fitted = garch_model.fit(disp="off")

print(garch_fitted.summary())


# -----------------------------
# Rolling 1-step ahead forecast on test set
# -----------------------------
garch_vol_forecasts = []

for i in range(len(returns_test)):
    # Use all data up to current test point (expanding window)
    train_plus_test = pd.concat([returns_train, returns_test.iloc[:i]])
    
    # Refit GARCH model (or use fixed parameters from initial fit for speed)
    model = arch_model(train_plus_test, vol='Garch', p=1, q=1, mean="Zero", rescale=False)
    fitted = model.fit(disp="off")
    
    # Forecast 1-step ahead conditional volatility
    forecast = fitted.forecast(horizon=1)
    vol_forecast = np.sqrt(forecast.variance.values[-1, 0])  # Get volatility (sqrt of variance)
    garch_vol_forecasts.append(vol_forecast)

# Convert to array and scale back to original units (divide by 100)
garch_vol = np.array(garch_vol_forecasts) / 100


# -------------------------------
# Target: rv_14 (14-day realized volatility)
# -------------------------------
y_test = test_df["rv_14"].values[:len(garch_vol)]  # Align lengths
dates_test = test_df["Date"].values[:len(garch_vol)]


# -------------------------------
# Metrics
# -------------------------------
def print_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print("\n" + "="*60)
    print(f"METRICS: {label}")
    print("="*60)
    print(f"MSE:         {mse:.7f}")
    print(f"RMSE:        {rmse:.7f}")
    print(f"MAE:         {mae:.7f}")
    print(f"MAPE:        {mape:.3f}%")
    print(f"R²:          {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
    print("="*60)


print_metrics(y_test, garch_vol, "GARCH(1,1) Rolling Forecast")


# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(14, 6))
plt.plot(dates_test, y_test, label="Actual rv_14", color='blue', linewidth=2)
plt.plot(dates_test, garch_vol, label="GARCH(1,1) Forecast", color='orange', linestyle="--")
plt.title("BTC 14-Day Realized Volatility: Actual vs GARCH(1,1) Rolling Forecast")
plt.xlabel("Date")
plt.ylabel("Volatility (rv_14)")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

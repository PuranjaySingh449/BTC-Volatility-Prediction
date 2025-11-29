import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Paths ---
test_path = r"D:\btc-volatility v2\btc_test_14d.csv"
model_path = r"D:\btc-volatility v2\LSTM\lstm_14d_model.h5"
scaler_path = r"D:\btc-volatility v2\LSTM\scaler.pkl"

# --- Feature columns (update as needed to match training features!) ---
FEATURE_COLUMNS = [
    "log_ret", "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol"
]

TARGET_COLUMN = "rv_14"

# --- Load test data ---
df_test = pd.read_csv(test_path, skiprows=2)
df_test.columns = [
    "Date", "Close", "High", "Low", "Open", "Volume", "log_ret",
    "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol", "rv_14"
]
df_test["Date"] = pd.to_datetime(df_test["Date"])
dates = df_test["Date"].values

X_test = df_test[FEATURE_COLUMNS].values.astype(np.float32)
y_test = df_test[TARGET_COLUMN].values.astype(np.float32)

# --- Load scaler and transform features ---
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)   # shape: (n_samples, n_features)

# --- Reshape for LSTM (samples, timesteps, features)
# If your model expects lag windows, e.g. shape (n_samples, lookback, n_features)
# If lookback=1 during training, shape is just (n_samples, 1, n_features)
X_test_lstm = np.expand_dims(X_test_scaled, axis=1)  # shape: (n_samples, 1, n_features)

# --- Load LSTM model ---
model = load_model(model_path)

# --- Make predictions ---
y_pred = model.predict(X_test_lstm).squeeze()

# --- Metrics ---
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
    print(f"MSE:         {mse:.6f}")
    print(f"RMSE:        {rmse:.6f}")
    print(f"MAE:         {mae:.6f}")
    print(f"MAPE:        {mape:.2f}%")
    print(f"R²:          {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
    print("="*60)

print_metrics(y_test, y_pred, "LSTM Forecast")

# --- Visualization ---
plt.figure(figsize=(14,6))
plt.plot(dates, y_test, label="Actual rv_14", color='blue')
plt.plot(dates, y_pred, label="LSTM Prediction", color='orange', linestyle="--")
plt.title("BTC 14-Day Realized Volatility: Actual vs LSTM Model Prediction")
plt.xlabel("Date")
plt.ylabel("Volatility (rv_14)")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

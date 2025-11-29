import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
test_csv = r"D:\btc-volatility v2\btc_test_14d.csv"
model_path = r"D:\btc-volatility v2\LSTM\lstm_14d_model.h5"
scaler_path = r"D:\btc-volatility v2\LSTM\scaler.pkl"

sequence_length = 60
target_column = "rv_14"

# ------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------
df = pd.read_csv(test_csv, skiprows=2)
df.columns = [
    "Date", "Close", "High", "Low", "Open", "Volume", "log_ret",
    "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol", "rv_14"
]
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna().reset_index(drop=True)

dates = df["Date"]
df_features = df.drop(["Date"], axis=1)

# ------------------------------------------------
# LOAD SCALER & SCALE
# ------------------------------------------------
scaler = joblib.load(scaler_path)
scaled_data = scaler.transform(df_features)

# ------------------------------------------------
# CREATE SEQUENCES
# ------------------------------------------------
def create_sequences(data, target_idx, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X), np.array(y)

target_idx = df_features.columns.get_loc(target_column)
X_test, y_test = create_sequences(scaled_data, target_idx, sequence_length)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ------------------------------------------------
# LOAD MODEL & PREDICT
# ------------------------------------------------
model = load_model(model_path, compile=False)
print("✓ Model loaded successfully")

y_pred_scaled = model.predict(X_test, verbose=1)

# ------------------------------------------------
# INVERSE SCALE PREDICTIONS
# ------------------------------------------------
dummy = np.zeros((len(y_pred_scaled), scaled_data.shape[1]))
dummy[:, target_idx] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy)[:, target_idx]

dummy_true = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_true[:, target_idx] = y_test
y_true = scaler.inverse_transform(dummy_true)[:, target_idx]

# ------------------------------------------------
# EVALUATION METRICS
# ------------------------------------------------
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
r2 = r2_score(y_true, y_pred)
corr = np.corrcoef(y_true, y_pred)[0, 1]

print("\n" + "="*60)
print("TEST SET EVALUATION METRICS")
print("="*60)
print(f"MSE:         {mse:.6f}")
print(f"RMSE:        {rmse:.6f}")
print(f"MAE:         {mae:.6f}")
print(f"MAPE:        {mape:.2f}%")
print(f"R²:          {r2:.4f}")
print(f"Correlation: {corr:.4f}")
print("="*60)

# ------------------------------------------------
# PLOT RESULTS
# ------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(dates.iloc[sequence_length:], y_true, label="Actual 14d Volatility", color='blue', linewidth=2)
plt.plot(dates.iloc[sequence_length:], y_pred, label="Predicted 14d Volatility", color='red', linewidth=2, alpha=0.8)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.title("BTC 14-Day Volatility: Actual vs Predicted (Fine-Tuned LSTM)", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------
output_df = pd.DataFrame({
    "Date": dates.iloc[sequence_length:].values,
    "Actual_rv14": y_true,
    "Predicted_rv14": y_pred
})
output_df.to_csv(r"D:\btc-volatility v2\LSTM\predictions_test.csv", index=False)
print("\n✓ Predictions saved to: D:\\btc-volatility v2\\LSTM\\predictions_test.csv")

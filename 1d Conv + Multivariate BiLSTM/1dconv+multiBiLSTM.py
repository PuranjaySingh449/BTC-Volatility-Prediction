import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Paths
# ----------------------------
train_path = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path  = r"D:\btc-volatility v2\btc_test_14d.csv"
model_path = r"D:\btc-volatility v2\conv1d_bilstm_model.keras"
scaler_path = r"D:\btc-volatility v2\scaler_conv1d.pkl"

# ----------------------------
# Load CSV
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
test_df  = load_csv(test_path)

# ----------------------------
# Multivariate features
# ----------------------------
features = ["Close", "High", "Low", "Open", "Volume", "log_ret",
            "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol", "rv_14"]

scaler = StandardScaler()
scaler.fit(train_df[features])

# Save scaler
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved.")

train_scaled = scaler.transform(train_df[features])
test_scaled  = scaler.transform(test_df[features])

# ----------------------------
# Create sequences
# ----------------------------
def create_sequences(data, n_steps=60):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, 0])  # predict 'Close'
    return np.array(X), np.array(y)

n_steps = 60
X_train, y_train = create_sequences(train_scaled, n_steps)
X_test, y_test   = create_sequences(test_scaled, n_steps)

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)

# ----------------------------
# Build Conv1D + BiLSTM model
# ----------------------------
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(n_steps, len(features))),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ----------------------------
# Train model
# ----------------------------
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# Save model
model.save(model_path)
print(f"Conv1D + BiLSTM model saved to: {model_path}")

# ----------------------------
# Predict on test set
# ----------------------------
y_pred_scaled = model.predict(X_test)

# Inverse scale
dummy = np.zeros((len(y_pred_scaled), len(features)))
dummy[:, 0] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(dummy)[:, 0]

y_test_dummy = np.zeros((len(y_test), len(features)))
y_test_dummy[:, 0] = y_test
y_test_inv = scaler.inverse_transform(y_test_dummy)[:, 0]

# ----------------------------
# Metrics
# ----------------------------
mae  = mean_absolute_error(y_test_inv, y_pred)
mse  = mean_squared_error(y_test_inv, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test_inv, y_pred)

print("\nTest Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# ----------------------------
# Plot actual vs predicted
# ----------------------------
plt.figure(figsize=(12,6))
plt.plot(test_df['Date'].iloc[n_steps:], y_test_inv, label="Actual Close", color='blue')
plt.plot(test_df['Date'].iloc[n_steps:], y_pred, label="Predicted Close", color='red')
plt.title("BTC-USD Actual vs Predicted Prices (Conv1D + Multivariate BiLSTM)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

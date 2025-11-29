import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib  # safer than pickle for sklearn

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
training_csv = r"D:\btc-volatility v2\btc_train_14d.csv"
model_save_path = r"D:\btc-volatility v2\lstm_14d_model.h5"
scaler_save_path = r"D:\btc-volatility v2\scaler.pkl"

sequence_length = 60
target_column = "rv_14"   # you are predicting 14-day realized volatility

# ------------------------------------------------
# LOAD & CLEAN DATA
# ------------------------------------------------

df = pd.read_csv(training_csv, skiprows=2)

df.columns = [
    "Date", "Close", "High", "Low", "Open", "Volume", "log_ret",
    "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol", "rv_14"
]

df["Date"] = pd.to_datetime(df["Date"])

# Drop rows with NaN
df = df.dropna().reset_index(drop=True)

# Save Date for future output (model does not use it)
dates = df["Date"]
df_features = df.drop(["Date"], axis=1)

# ------------------------------------------------
# SCALE FEATURES
# ------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_features)

# Save scaler safely using joblib
joblib.dump(scaler, scaler_save_path)
print(f"Scaler saved to: {scaler_save_path}")

# ------------------------------------------------
# BUILD SEQUENCES
# ------------------------------------------------

def create_sequences(data, target_idx, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X), np.array(y)

target_idx = df_features.columns.get_loc(target_column)

X, y = create_sequences(scaled_data, target_idx, sequence_length)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ------------------------------------------------
# TRAIN/VAL SPLIT (time series safe)
# ------------------------------------------------

split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

# ------------------------------------------------
# BUILD MODEL
# ------------------------------------------------

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# ------------------------------------------------
# TRAIN
# ------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    shuffle=False
)

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

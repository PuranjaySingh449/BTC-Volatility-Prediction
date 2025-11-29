import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ---------------------
# Paths & Directories
# ---------------------
output_dir = r"D:\btc-volatility v2\1dConv+multiBiLSTM"
os.makedirs(output_dir, exist_ok=True)
train_path = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path  = r"D:\btc-volatility v2\btc_test_14d.csv"
orig_model_path = os.path.join(output_dir, "conv1d_bilstm_volatility_14d.keras")
finetuned_model_path = os.path.join(output_dir, "conv1d_bilstm_volatility_14d_finetuned.keras")
scaler_path = os.path.join(output_dir, "scaler_volatility.pkl")
finetune_scaler_path = os.path.join(output_dir, "scaler_volatility_finetune.pkl")
results_csv = os.path.join(output_dir, "conv1d_bilstm_volatility_finetune_results.csv")
results_plot = os.path.join(output_dir, "conv1d_bilstm_volatility_finetune_results.png")

# ---------------------
# Load data util
# ---------------------
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

# ---------------------
# Load original scaler and save a (potentially re-fit) finetune scaler
# ---------------------
with open(scaler_path, "rb") as f:
    scaler_dict = pickle.load(f)
scaler_X = scaler_dict["scaler_X"]
scaler_y = scaler_dict["scaler_y"]
feature_cols = scaler_dict["feature_cols"]
target_col = scaler_dict["target_col"]
# If you want to re-fit on more data, do so here and overwrite scalers.

with open(finetune_scaler_path, "wb") as f:
    pickle.dump(scaler_dict, f)
print(f"✅ Finetune scaler saved: {finetune_scaler_path}")

X_train_scaled = scaler_X.transform(train_df[feature_cols])
X_test_scaled = scaler_X.transform(test_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[[target_col]]).flatten()
y_test_scaled = scaler_y.transform(test_df[[target_col]]).flatten()

def create_sequences(X_data, y_data, n_steps=14):
    X_seq, y_seq = [], []
    for i in range(n_steps, len(X_data)):
        X_seq.append(X_data[i-n_steps:i])
        y_seq.append(y_data[i])
    return np.array(X_seq), np.array(y_seq)

n_steps = 14
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, n_steps)
X_test, y_test   = create_sequences(X_test_scaled, y_test_scaled, n_steps)
test_dates = test_df['Date'].iloc[n_steps:].values

# ---------------------
# Load and fine-tune model
# ---------------------
model = load_model(orig_model_path)
model.compile(
    optimizer=Adam(learning_rate=0.0002),  # Lower LR for finetune
    loss='mse',
    metrics=['mae']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(finetuned_model_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("Starting fine-tuning (Conv1D+BiLSTM)...")
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)
print(f"✅ Fine-tuned model saved to: {finetuned_model_path}")

# ---------------------
# Predict and evaluate
# ---------------------
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)
print("\nFine-Tune Test Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_rv_14': y_test_actual,
    'Conv1D_BiLSTM_Finetune_Prediction': y_pred,
    'Residual': y_pred - y_test_actual,
    'Absolute_Error': np.abs(y_pred - y_test_actual),
    'Percentage_Error': ((y_pred - y_test_actual) / (y_test_actual) * 100)
})
results_df.to_csv(results_csv, index=False)
print(f"✅ Fine-tuned results saved: {results_csv}")

plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test_actual, label="Actual 14d Volatility (rv_14)", color='blue')
plt.plot(test_dates, y_pred, label="Predicted 14d Volatility (Conv1D+BiLSTM Finetune)", color='red', linestyle="--")
plt.title("BTC 14-Day Volatility: Actual vs Predicted (Conv1D+BiLSTM Fine-Tune)")
plt.xlabel("Date")
plt.ylabel("Volatility (rv_14)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Fine-tuned plot saved to: {results_plot}")

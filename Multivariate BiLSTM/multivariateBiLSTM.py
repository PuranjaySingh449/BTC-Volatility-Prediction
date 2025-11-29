import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set output directory
output_dir = r"D:\btc-volatility v2\multiBiLSTM"
os.makedirs(output_dir, exist_ok=True)

# Paths
train_path = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path = r"D:\btc-volatility v2\btc_test_14d.csv"
model_path = os.path.join(output_dir, "bilstm_volatility_14d.keras")
scaler_path = os.path.join(output_dir, "scaler_volatility.pkl")
results_csv = os.path.join(output_dir, "bilstm_volatility_results.csv")
results_plot = os.path.join(output_dir, "bilstm_volatility_results.png")

# Load CSVs
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

# Features for volatility prediction
feature_cols = ["Close", "High", "Low", "Open", "Volume", "log_ret",
                "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol"]
target_col = "rv_14"

# Scale features and target separately
scaler_X = StandardScaler()
scaler_X.fit(train_df[feature_cols].values)

scaler_y = StandardScaler()
scaler_y.fit(train_df[[target_col]].values)

# Save scalers
scaler_dict = {
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'feature_cols': feature_cols,
    'target_col': target_col
}
with open(scaler_path, "wb") as f:
    pickle.dump(scaler_dict, f)
print(f"✅ Scalers saved to: {scaler_path}")

# Scale the data
X_train_scaled = scaler_X.transform(train_df[feature_cols])
X_test_scaled = scaler_X.transform(test_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[[target_col]]).flatten()
y_test_scaled = scaler_y.transform(test_df[[target_col]]).flatten()

# Sequence creation for volatility
def create_sequences(X_data, y_data, n_steps=14):
    X_seq, y_seq = [], []
    for i in range(n_steps, len(X_data)):
        X_seq.append(X_data[i-n_steps:i])
        y_seq.append(y_data[i])
    return np.array(X_seq), np.array(y_seq)

n_steps = 14
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, n_steps)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, n_steps)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# Build BiLSTM model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(n_steps, len(feature_cols))),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Model saved to: {model_path}")

# Predict on test set
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
test_dates = test_df['Date'].iloc[n_steps:].values

# Metrics
def print_metrics(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    actual_dir = np.diff(y_true) > 0 if len(y_true) > 1 else []
    pred_dir = np.diff(y_pred) > 0 if len(y_pred) > 1 else []
    dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else np.nan
    print("\n" + "="*80)
    print(f"METRICS: {label}")
    print("="*80)
    print(f"MSE:        {mse:.7f}")
    print(f"RMSE:       {rmse:.7f}")
    print(f"MAE:        {mae:.7f}")
    print(f"MAPE:       {mape:.3f}%")
    print(f"R²:         {r2:.4f}")
    print(f"Correlation:{corr:.4f}")
    if not np.isnan(dir_acc):
        print(f"Directional Acc:{dir_acc:.2f}%")
    print("="*80)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Correlation': corr, 'Dir_Acc': dir_acc}

metrics = print_metrics(y_test_actual, y_pred, "BiLSTM 14-day Volatility")

# Save results to CSV
results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_rv_14': y_test_actual,
    'BiLSTM_Prediction': y_pred,
    'Residual': y_pred - y_test_actual,
    'Absolute_Error': np.abs(y_pred - y_test_actual),
    'Percentage_Error': ((y_pred - y_test_actual) / y_test_actual) * 100
})
results_df.to_csv(results_csv, index=False)
print(f"✅ Results saved: {results_csv}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Time series
axes[0, 0].plot(test_dates, y_test_actual, label="Actual rv_14", color='#2E86AB', linewidth=2)
axes[0, 0].plot(test_dates, y_pred, label="BiLSTM Prediction", color='#06A77D', linewidth=2, linestyle='--')
axes[0, 0].set_title('BiLSTM: Actual vs Predicted 14-Day Volatility', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Volatility (rv_14)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Scatter plot
axes[0, 1].scatter(y_test_actual, y_pred, alpha=0.6, s=50, color='#A23B72', edgecolor='black', linewidth=0.5)
z = np.polyfit(y_test_actual, y_pred, 1)
p = np.poly1d(z)
axes[0, 1].plot(y_test_actual, p(y_test_actual), "r--", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.4f}')
axes[0, 1].plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 
                'k-', linewidth=1.5, label='Perfect')
axes[0, 1].set_xlabel('Actual rv_14')
axes[0, 1].set_ylabel('Predicted rv_14')
axes[0, 1].set_title(f'Predictions vs Actual (R²={metrics["R2"]:.3f})', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals
residuals = y_pred - y_test_actual
axes[1, 0].plot(test_dates, residuals, color='#F18F01', linewidth=2, marker='o', markersize=4)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].axhline(y=np.mean(residuals), color='red', linestyle=':', linewidth=2, 
                   label=f'Mean={np.mean(residuals):.5f}')
axes[1, 0].fill_between(test_dates, 0, residuals, alpha=0.3, color='orange')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Residuals (Pred - Actual)')
axes[1, 0].set_title('Prediction Residuals Over Time', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Training history
axes[1, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss (MSE)')
axes[1, 1].set_title('Training History', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Results plot saved: {results_plot}")

# Optional: test saved model and scaler
with open(scaler_path, "rb") as f:
    loaded_scaler_dict = pickle.load(f)
loaded_model = load_model(model_path)
print(f"✅ Loaded model and scaler from: {output_dir}")

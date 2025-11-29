import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set output directory
output_dir = r"D:\btc-volatility v2\multiBiLSTM"
os.makedirs(output_dir, exist_ok=True)

train_path = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path = r"D:\btc-volatility v2\btc_test_14d.csv"
scaler_path = os.path.join(output_dir, "scaler_volatility.pkl")
model_path = os.path.join(output_dir, "bilstm_volatility_14d.keras")            # original model
finetuned_model_path = os.path.join(output_dir, "bilstm_volatility_14d_finetuned.keras")
results_csv = os.path.join(output_dir, "bilstm_volatility_finetune_results.csv")
results_plot = os.path.join(output_dir, "bilstm_volatility_finetune_results.png")

def load_csv(path):
    df = pd.read_csv(path, skiprows=2)
    df.columns = [
        "Date", "Close", "High", "Low", "Open", "Volume", "log_ret",
        "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol", "rv_14"
    ]
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Load scaler
with open(scaler_path, "rb") as f:
    scaler_dict = pickle.load(f)
scaler_X = scaler_dict['scaler_X']
scaler_y = scaler_dict['scaler_y']
feature_cols = scaler_dict['feature_cols']
target_col = scaler_dict['target_col']

# Load data
train_df = load_csv(train_path)
test_df = load_csv(test_path)

# Scale features and target
X_train_scaled = scaler_X.transform(train_df[feature_cols])
X_test_scaled = scaler_X.transform(test_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[[target_col]]).flatten()
y_test_scaled = scaler_y.transform(test_df[[target_col]]).flatten()

# Sequence creation for volatility prediction
def create_sequences(X_data, y_data, n_steps=14):
    X_seq, y_seq = [], []
    for i in range(n_steps, len(X_data)):
        X_seq.append(X_data[i-n_steps:i])
        y_seq.append(y_data[i])
    return np.array(X_seq), np.array(y_seq)

n_steps = 14
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, n_steps)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, n_steps)
test_dates = test_df['Date'].iloc[n_steps:].values

# Load and fine-tune model
model = load_model(model_path)
model.compile(
    optimizer=Adam(learning_rate=0.0002),  # Lower learning rate for fine-tuning
    loss='mse',
    metrics=['mae']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(finetuned_model_path, monitor='val_loss', save_best_only=True, verbose=1)
]

print("Starting fine-tuning...")
history = model.fit(
    X_train, y_train,
    epochs=60,               # Fewer epochs for fine-tuning
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)
print(f"\n✅ Fine-tuned model saved to: {finetuned_model_path}")

# Predict and inverse transform
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

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
metrics = print_metrics(y_test_actual, y_pred, "BiLSTM Fine-Tuned 14-day Volatility")

# Save fine-tuned results to CSV
results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_rv_14': y_test_actual,
    'BiLSTM_Finetune_Prediction': y_pred,
    'Residual': y_pred - y_test_actual,
    'Absolute_Error': np.abs(y_pred - y_test_actual),
    'Percentage_Error': ((y_pred - y_test_actual) / y_test_actual) * 100
})
results_df.to_csv(results_csv, index=False)
print(f"✅ Fine-tuned results saved: {results_csv}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(test_dates, y_test_actual, label="Actual rv_14", color='#2E86AB', linewidth=2)
axes[0, 0].plot(test_dates, y_pred, label="BiLSTM Finetune", color='#06A77D', linewidth=2, linestyle='--')
axes[0, 0].set_title('Fine-Tuned BiLSTM: Actual vs Predicted 14-Day Volatility', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Volatility (rv_14)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test_actual, y_pred, alpha=0.6, s=50, color='#A23B72', edgecolor='black', linewidth=0.5)
z = np.polyfit(y_test_actual, y_pred, 1)
p = np.poly1d(z)
axes[0, 1].plot(y_test_actual, p(y_test_actual), "r--", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.4f}')
axes[0, 1].plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 
                'k-', linewidth=1.5, label='Perfect')
axes[0, 1].set_xlabel('Actual rv_14')
axes[0, 1].set_ylabel('Finetuned Predicted rv_14')
axes[0, 1].set_title(f'Predictions vs Actual (R²={metrics["R2"]:.3f})', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

residuals = y_pred - y_test_actual
axes[1, 0].plot(test_dates, residuals, color='#F18F01', linewidth=2, marker='o', markersize=4)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].axhline(y=np.mean(residuals), color='red', linestyle=':', linewidth=2, label=f'Mean={np.mean(residuals):.5f}')
axes[1, 0].fill_between(test_dates, 0, residuals, alpha=0.3, color='orange')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Residuals (Pred - Actual)')
axes[1, 0].set_title('Prediction Residuals Over Time', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

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
print(f"✅ Fine-tuned results plot saved: {results_plot}")

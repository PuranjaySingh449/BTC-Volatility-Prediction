import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ----------------------------
# Attention Layer (reuse for loading model)
# ----------------------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="context_vector", shape=(input_shape[-1], 1),
                                initializer="random_normal", trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        att = tf.nn.softmax(tf.tensordot(u_it, self.u, axes=1), axis=1)
        output = tf.reduce_sum(x * att, axis=1)
        return output

# ----------------------------
# Paths
# ----------------------------
output_dir = r"D:\btc-volatility v2\attention+1dConv+MultiBiLSTM"
os.makedirs(output_dir, exist_ok=True)
train_path  = r"D:\btc-volatility v2\btc_train_14d.csv"
test_path   = r"D:\btc-volatility v2\btc_test_14d.csv"
orig_model_path = os.path.join(output_dir, "conv1d_bilstm_attention_14d.keras")
finetuned_model_path = os.path.join(output_dir, "conv1d_bilstm_attention_14d_finetuned.keras")
orig_scaler_path = os.path.join(output_dir, "scaler_attention.pkl")
finetune_scaler_path = os.path.join(output_dir, "scaler_attention_finetune.pkl")
results_csv = os.path.join(output_dir, "attention_conv1d_bilstm_volatility_finetune_results.csv")
results_plot = os.path.join(output_dir, "attention_conv1d_bilstm_volatility_finetune_results.png")

# ----------------------------
# Load CSVs
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
# Features for volatility prediction
# ----------------------------
feature_cols = ["Close", "High", "Low", "Open", "Volume", "log_ret",
                "vol_7", "vol_14", "vol_30", "vol_chg_7", "log_vol"]
target_col = "rv_14"

# ----------------------------
# Optionally, (re)fit scalers for fine-tuning
# ----------------------------
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_X.fit(train_df[feature_cols])
scaler_y = StandardScaler()
scaler_y.fit(train_df[[target_col]])
scaler_dict = {'scaler_X': scaler_X, 'scaler_y': scaler_y,
               'feature_cols': feature_cols, 'target_col': target_col}
with open(finetune_scaler_path, "wb") as f:
    pickle.dump(scaler_dict, f)
print(f"✅ Finetune scaler saved: {finetune_scaler_path}")

X_train_scaled = scaler_X.transform(train_df[feature_cols])
X_test_scaled  = scaler_X.transform(test_df[feature_cols])
y_train_scaled = scaler_y.transform(train_df[[target_col]]).flatten()
y_test_scaled  = scaler_y.transform(test_df[[target_col]]).flatten()

# ----------------------------
# Sequence creation for volatility
# ----------------------------
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

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# ----------------------------
# Load and finetune model
# ----------------------------
model = load_model(orig_model_path, custom_objects={"Attention": Attention})
model.compile(
    optimizer=Adam(learning_rate=0.0002),  # lower LR for finetune
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
    epochs=40,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)
print(f"✅ Fine-tuned model saved to: {finetuned_model_path}")

# ----------------------------
# Predict and evaluate
# ----------------------------
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(y_test_actual, y_pred)
mse  = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test_actual, y_pred)
print("\nFine-Tune Test Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# ----------------------------
# Save results to CSV
# ----------------------------
results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_rv_14': y_test_actual,
    'Attention_Conv1D_BiLSTM_Finetune_Prediction': y_pred,
    'Residual': y_pred - y_test_actual,
    'Absolute_Error': np.abs(y_pred - y_test_actual),
    'Percentage_Error': ((y_pred - y_test_actual) / (y_test_actual) * 100)
})
results_df.to_csv(results_csv, index=False)
print(f"✅ Fine-tuned results saved: {results_csv}")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12,6))
plt.plot(test_dates, y_test_actual, label="Actual 14d Volatility (rv_14)", color='blue')
plt.plot(test_dates, y_pred, label="Predicted 14d Volatility (Attn+Conv1D+BiLSTM Finetune)", color='red', linestyle="--")
plt.title("BTC 14-Day Volatility: Actual vs Predicted (Attn+Conv1D+BiLSTM Finetune)")
plt.xlabel("Date")
plt.ylabel("Volatility (rv_14)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_plot, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Fine-tuned plot saved: {results_plot}")

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# -----------------------------
# Load Data and Paths
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

returns_train = train_df["log_ret"].astype(float) * 100
returns_test = test_df["log_ret"].astype(float) * 100


# -----------------------------
# Fit TARCH(2,2)
# -----------------------------
print("\n" + "="*80)
print("FITTING TARCH(2,2) MODEL")
print("="*80)

tarch = arch_model(returns_train, vol='GARCH', p=2, o=2, q=2, power=2.0, mean="Zero", rescale=False)
tarch_fitted = tarch.fit(disp="off")

print("\nModel Summary:")
print(tarch_fitted.summary())


# -----------------------------
# Walk-Forward 14-Step Forecasts
# -----------------------------
print("\n" + "="*80)
print("WALK-FORWARD 14-STEP FORECASTS")
print("="*80)

forecast_points = range(0, len(test_df), 14)
tarch_vol_14step = []
forecast_dates = []
forecast_details = []

for start_idx in forecast_points:
    returns_up_to_idx = pd.concat([returns_train, returns_test[:start_idx]]) if start_idx > 0 else returns_train
    
    tarch_temp = arch_model(returns_up_to_idx, vol='GARCH', p=2, o=2, q=2, power=2.0, mean="Zero", rescale=False)
    tarch_temp_fitted = tarch_temp.fit(disp="off")
    
    forecast_temp = tarch_temp_fitted.forecast(horizon=14, start=None, method='simulation', simulations=5000)
    vol_temp = forecast_temp.variance.values[-1, :]**0.5 / 100
    
    avg_vol = np.mean(vol_temp)
    tarch_vol_14step.append(avg_vol)
    forecast_dates.append(test_df["Date"].iloc[start_idx] if start_idx < len(test_df) else test_df["Date"].iloc[-1])
    
    # Store detailed forecast
    forecast_details.append({
        'start_idx': start_idx,
        'date': test_df["Date"].iloc[start_idx] if start_idx < len(test_df) else test_df["Date"].iloc[-1],
        'forecast': avg_vol,
        'actual': test_df["rv_14"].iloc[start_idx] if start_idx < len(test_df) else np.nan,
        'error': avg_vol - test_df["rv_14"].iloc[start_idx] if start_idx < len(test_df) else np.nan
    })
    
    print(f"Forecast from index {start_idx:3d} ({forecast_dates[-1].strftime('%Y-%m-%d')}): "
          f"Pred={avg_vol:.6f}, Actual={test_df['rv_14'].iloc[start_idx]:.6f}, "
          f"Error={avg_vol - test_df['rv_14'].iloc[start_idx]:.6f}")

tarch_vol_14step = np.array(tarch_vol_14step)


# -----------------------------
# Evaluation Metrics with Enhanced Stats
# -----------------------------
def print_enhanced_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Additional metrics
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_ae = np.median(np.abs(errors))
    max_error = np.max(np.abs(errors))
    
    # Directional accuracy (did we predict increase/decrease correctly?)
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_acc = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_acc = np.nan
    
    print("\n" + "="*80)
    print(f"ENHANCED METRICS: {label}")
    print("="*80)
    print(f"MSE:                {mse:.7f}")
    print(f"RMSE:               {rmse:.7f}")
    print(f"MAE:                {mae:.7f}")
    print(f"Median AE:          {median_ae:.7f}")
    print(f"Max Error:          {max_error:.7f}")
    print(f"MAPE:               {mape:.3f}%")
    print(f"R²:                 {r2:.4f}")
    print(f"Correlation:        {corr:.4f}")
    print(f"Mean Error (Bias):  {mean_error:.7f}")
    print(f"Std Error:          {std_error:.7f}")
    if not np.isnan(directional_acc):
        print(f"Directional Acc:    {directional_acc:.2f}%")
    print("="*80)
    
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2,
        'Correlation': corr, 'Mean_Error': mean_error, 'Std_Error': std_error,
        'Median_AE': median_ae, 'Max_Error': max_error, 'Directional_Acc': directional_acc
    }


y_test = test_df["rv_14"].values
y_test_14step = test_df["rv_14"].iloc[::14].values[:len(tarch_vol_14step)]

metrics = print_enhanced_metrics(y_test_14step, tarch_vol_14step, "TARCH(2,2) 14-Step Ahead Forecasts")


# -----------------------------
# Comprehensive Visualization
# -----------------------------
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Time series comparison
ax1 = fig.add_subplot(gs[0, :])
dates_14step = test_df["Date"].iloc[::14].values[:len(tarch_vol_14step)]
ax1.plot(dates_14step, y_test_14step, label="Actual rv_14", color='#2E86AB', linewidth=2.5, marker='o', markersize=6)
ax1.plot(dates_14step, tarch_vol_14step, label="TARCH 14-Step Forecast", color='#06A77D', linewidth=2.5, marker='s', markersize=6, linestyle='--')
ax1.fill_between(dates_14step, y_test_14step, tarch_vol_14step, alpha=0.2, color='red')
ax1.set_title('TARCH(2,2) 14-Day Ahead Volatility Forecast', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Volatility (rv_14)', fontsize=12)
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot with regression line
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_test_14step, tarch_vol_14step, alpha=0.7, s=80, color='#A23B72', edgecolor='black', linewidth=0.5)
z = np.polyfit(y_test_14step, tarch_vol_14step, 1)
p = np.poly1d(z)
ax2.plot(y_test_14step, p(y_test_14step), "r--", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.4f}')
ax2.plot([y_test_14step.min(), y_test_14step.max()], [y_test_14step.min(), y_test_14step.max()], 
         'k-', linewidth=1.5, label='Perfect Forecast')
ax2.set_xlabel('Actual rv_14', fontsize=12)
ax2.set_ylabel('Predicted rv_14', fontsize=12)
ax2.set_title(f'Forecast vs Actual (R²={metrics["R2"]:.3f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals over time
ax3 = fig.add_subplot(gs[1, 1])
residuals = tarch_vol_14step - y_test_14step
ax3.plot(dates_14step, residuals, color='#F18F01', linewidth=2, marker='o', markersize=5)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.axhline(y=np.mean(residuals), color='red', linestyle=':', linewidth=2, label=f'Mean={np.mean(residuals):.5f}')
ax3.fill_between(dates_14step, 0, residuals, alpha=0.3, color='orange')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Residuals (Pred - Actual)', fontsize=12)
ax3.set_title('Forecast Residuals Over Time', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Residual distribution
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(residuals, bins=15, color='#C73E1D', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax4.axvline(x=np.mean(residuals), color='blue', linestyle=':', linewidth=2, label=f'Mean={np.mean(residuals):.5f}')
ax4.set_xlabel('Residuals', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Distribution of Forecast Errors', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Metrics summary as table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('tight')
ax5.axis('off')

metrics_table = [
    ['Metric', 'Value'],
    ['MSE', f"{metrics['MSE']:.7f}"],
    ['RMSE', f"{metrics['RMSE']:.7f}"],
    ['MAE', f"{metrics['MAE']:.7f}"],
    ['MAPE', f"{metrics['MAPE']:.2f}%"],
    ['R²', f"{metrics['R2']:.4f}"],
    ['Correlation', f"{metrics['Correlation']:.4f}"],
    ['Mean Error', f"{metrics['Mean_Error']:.7f}"],
    ['Directional Acc', f"{metrics['Directional_Acc']:.2f}%" if not np.isnan(metrics['Directional_Acc']) else 'N/A']
]

table = ax5.table(cellText=metrics_table, cellLoc='left', loc='center',
                  colWidths=[0.5, 0.5],
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(metrics_table)):
    if i % 2 == 0:
        table[(i, 0)].set_facecolor('#E8F4F8')
        table[(i, 1)].set_facecolor('#E8F4F8')

ax5.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)

plt.savefig('tarch_enhanced_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Enhanced analysis plot saved: tarch_enhanced_analysis.png")


# -----------------------------
# Export Results to CSV
# -----------------------------
results_df = pd.DataFrame({
    'Date': dates_14step,
    'Actual_rv_14': y_test_14step,
    'TARCH_Forecast': tarch_vol_14step,
    'Residual': residuals,
    'Absolute_Error': np.abs(residuals),
    'Percentage_Error': (residuals / y_test_14step) * 100
})

results_df.to_csv('tarch_14step_results.csv', index=False)
print("✅ Results exported to: tarch_14step_results.csv")

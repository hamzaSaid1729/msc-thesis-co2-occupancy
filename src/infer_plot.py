import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.dates as mdates

# -------------------------------
# Define the trained LSTM model
# -------------------------------
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# -------------------------------
# Load model and scaler
# -------------------------------
input_dim = 4
hidden_dim = 64
output_dim = 1
num_layers = 3
look_back = 180

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
model.load_state_dict(torch.load('outputs/trained_lstm_regression_model.pth', map_location=device))
model.eval()

scaler = joblib.load('outputs/scaler.save')

# -------------------------------
# Process single Excel file
# -------------------------------
file_path = r"C:\Users\User\Desktop\Experiments\test1\Book1.xlsx"  # <-- change path if needed

df = pd.read_excel(file_path)

# Drop unnecessary columns
df.drop(columns=[
    'PM25 (MICROGRAMS_PER_CUBIC_METER)',
    'PM10 (MICROGRAMS_PER_CUBIC_METER)',
    'TVOC (PARTS_PER_BILLION)'
], errors='ignore', inplace=True)

# Match schema
df.columns = ['utc_dates', 'local_dates', 'co2', 'iaqi', 'temp', 'humidity',
              'people', 'door', 'ventilation', 'window'][:len(df.columns)]

# Convert & clean
df['co2'] = pd.to_numeric(df['co2'], errors='coerce')
df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
df['people'] = pd.to_numeric(df['people'], errors='coerce')
df['local_dates'] = pd.to_datetime(df['local_dates'], errors='coerce')
df = df.dropna(subset=['local_dates', 'co2', 'temp', 'humidity', 'people'])

if len(df) < look_back:
    raise ValueError("Not enough data after cleaning.")

# Keep original values
original_humidity = df['humidity'].values.copy()
original_temp = df['temp'].values.copy()
original_co2 = df['co2'].values.copy()

# Scale
features = ['co2', 'iaqi', 'temp', 'humidity']
X_scaled = scaler.transform(df[features].values)

# Sequences
X_seq = [X_scaled[i:i + look_back] for i in range(len(X_scaled) - look_back + 1)]
X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    outputs = model(X_tensor)
    predicted_values = outputs.squeeze().cpu().numpy()
    predicted_labels = np.clip(np.round(predicted_values), 0, 3).astype(int)

# Align lengths
df_trimmed = df.iloc[look_back - 1:].reset_index(drop=True)
y_real = df_trimmed['people'].values
co2_series = original_co2[look_back - 1:]
temp_series = original_temp[look_back - 1:]
humidity_series = original_humidity[look_back - 1:]
time_series = df_trimmed['local_dates'].dt.tz_localize(None).values

# -------------------------------
# Evaluation Metrics
# -------------------------------
acc = accuracy_score(y_real, predicted_labels)
prec = precision_score(y_real, predicted_labels, average='weighted', zero_division=0)
rec = recall_score(y_real, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(y_real, predicted_labels, average='weighted', zero_division=0)

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_real, predicted_labels, zero_division=0))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_real, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_real), yticklabels=np.unique(y_real))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -------------------------------
# Plots (CO₂, Temp, Humidity vs Occupancy)
# -------------------------------
plt.figure(figsize=(18, 12))

# 1. CO₂ vs Occupancy
plt.subplot(3, 1, 1)
ax1 = plt.gca()
ax1.plot(time_series, co2_series, color='tab:blue', label='CO₂ (ppm)')
ax1.set_ylabel('CO₂ (ppm)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(time_series, y_real, 'kx-', label='Real Occupancy')
ax2.plot(time_series, predicted_labels, 'ro--', label='Predicted Occupancy')
ax2.set_ylabel('Occupancy', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.legend(loc='upper left')
plt.title("CO₂ vs Occupancy")

# 2. Temperature vs Occupancy
plt.subplot(3, 1, 2)
ax3 = plt.gca()
ax3.plot(time_series, temp_series, color='tab:green', label='Temperature (°C)')
ax3.set_ylabel('Temperature (°C)', color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')
ax3.grid(True)

ax4 = ax3.twinx()
ax4.plot(time_series, y_real, 'kx-', label='Real Occupancy')
ax4.plot(time_series, predicted_labels, 'ro--', label='Predicted Occupancy')
ax4.set_ylabel('Occupancy', color='tab:red')
ax4.tick_params(axis='y', labelcolor='tab:red')
ax4.legend(loc='upper left')
plt.title("Temperature vs Occupancy")

# 3. Humidity vs Occupancy
plt.subplot(3, 1, 3)
ax5 = plt.gca()
ax5.plot(time_series, humidity_series, color='tab:purple', label='Humidity (%)')
ax5.set_ylabel('Humidity (%)', color='tab:purple')
ax5.tick_params(axis='y', labelcolor='tab:purple')
ax5.grid(True)
ax5.axhline(30, color='gray', linestyle=':', alpha=0.5)
ax5.axhline(60, color='gray', linestyle=':', alpha=0.5)

ax6 = ax5.twinx()
ax6.plot(time_series, y_real, 'kx-', label='Real Occupancy')
ax6.plot(time_series, predicted_labels, 'ro--', label='Predicted Occupancy')
ax6.set_ylabel('Occupancy', color='tab:red')
ax6.tick_params(axis='y', labelcolor='tab:red')
ax6.legend(loc='upper left')
plt.title("Humidity vs Occupancy (Comfort Range: 30–60%)")

plt.tight_layout()
plt.show()

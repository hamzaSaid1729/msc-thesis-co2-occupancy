# LSTM-based Regression Model for Occupancy Prediction
# reads data from ../data, saves model/scaler/plot to ../outputs

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from librosa.util import frame
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# paths (relative to this file)
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")       # input data lives here
OUT_DIR  = os.path.join(ROOT_DIR, "outputs")    # artifacts go here
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# dataset wrapper
# -------------------------------
class TorchDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        x = torch.tensor(self.frames[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# -------------------------------
# LSTM (regression)
# -------------------------------
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last time step
        out = self.fc(out)           # regression output
        return out

# -------------------------------
# load + clean all CSVs from data/
# -------------------------------
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.csv')]

all_data = pd.DataFrame()
for file in files:
    df = pd.read_csv(os.path.join(DATA_DIR, file))

    # drop extras if present
    df.drop(columns=[
        'PM25 (MICROGRAMS_PER_CUBIC_METER)',
        'PM10 (MICROGRAMS_PER_CUBIC_METER)',
        'TVOC (PARTS_PER_BILLION)'
    ], errors='ignore', inplace=True)

    # standardize column names/order
    df.columns = ['utc_dates', 'local_dates', 'co2', 'iaqi', 'temp', 'humidity',
                  'people', 'door', 'ventilation', 'window'][:len(df.columns)]

    # types + basic cleaning
    df['co2'] = pd.to_numeric(df['co2'], errors='coerce')
    df['people'] = pd.to_numeric(df['people'], errors='coerce')
    df.dropna(subset=['co2', 'temp', 'humidity', 'iaqi', 'people'], inplace=True)

    all_data = pd.concat([all_data, df], axis=0)

# features/target
features = ['co2', 'iaqi', 'temp', 'humidity']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(all_data[features])
target = all_data['people'].values

# save scaler
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.save'))

# -------------------------------
# frame the sequence (hop=1)
# -------------------------------
frame_length = 50
X = frame(X_scaled, frame_length=frame_length, hop_length=1, axis=0)
y = frame(target,     frame_length=frame_length, hop_length=1, axis=0)

# use last label in each frame
class FrameLabel:
    def __init__(self, labels_frames):
        self.labels_frames = labels_frames
    def frame_label(self):
        return self.labels_frames[:, -1].reshape(-1, 1)

y = FrameLabel(y).frame_label()

# -------------------------------
# split (random 80/20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = TorchDataset(X_train, y_train)
test_dataset  = TorchDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# -------------------------------
# training setup
# -------------------------------
input_dim = len(features)
hidden_dim = 64
num_layers = 3
n_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[device] Using {'CUDA' if torch.cuda.is_available() else 'CPU'}")

model = LSTM(input_dim, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

train_losses, test_losses = [], []

# -------------------------------
# train loop
# -------------------------------
for epoch in range(n_epochs):
    model.train()
    tr = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.squeeze())
        loss.backward()
        optimizer.step()
        tr += loss.item()

    model.eval()
    te = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.squeeze())
            te += loss.item()

    tr /= max(1, len(train_loader))
    te /= max(1, len(test_loader))
    train_losses.append(tr)
    test_losses.append(te)

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {tr:.4f} | Test Loss: {te:.4f}")

# -------------------------------
# save model + plot
# -------------------------------
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'trained_lstm_regression_model.pth'))

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.title('Training and Test Loss (Regression)')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
plt.show()

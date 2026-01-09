import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import os

# --- 1. CONFIGURATION ---
# UPDATE THESE PATHS IF NEEDED
TRAIN_PATH = "/home/tom_mscds24/data/tencent dataset/train_data.csv"
TEST_PATH = "/home/tom_mscds24/data/tencent dataset/test_data.csv"
OUTPUT_FILE = "/home/tom_mscds24/data/tencent dataset/Final_Model_Comparison_Results.xlsx"

# Paper Settings: Compare different Lookbacks for 1-Day Prediction
LOOKBACKS = [96, 32, 16]   
PRED_LEN = 1               

# Input Features
FEATURE_COLS = [
    'Close', 'SMA', 'RSI', 'OBV', 'ADX',
    'Gold_Close', 'USD_JPY_Close', 'sentiment_score'
]

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20  # Increased to let Informer converge
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, flatten=False):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flatten = flatten

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[r_begin:r_end, 0] # Target: Close Price

        if self.flatten:
            return x.flatten(), y
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- 3. MODELS ---

# A. INFORMER & TRANSFORMER
class Informer(nn.Module):
    def __init__(self, enc_in, out_len, d_model=64, n_heads=4, e_layers=2, distil=True):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.enc_embedding = nn.Linear(enc_in, d_model)
        
        # Encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, e_layers)
        
        # Distillation (This is what makes it an Informer in this comparison)
        self.distil = distil
        if self.distil:
            self.distil_layer = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Generative Style Decoder (Simplified Projection)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.enc_embedding(x)
        x = self.encoder(x)
        
        # Apply Distilling if it's the Informer model
        if self.distil:
            x = x.permute(0, 2, 1)
            x = self.distil_layer(x)
            x = x.permute(0, 2, 1)
        
        # Project last state to prediction
        last_state = x[:, -1, :]
        pred = self.projection(last_state).unsqueeze(1).repeat(1, self.pred_len, 1)
        return pred.squeeze(-1)

# B. LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# C. TCN
class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels=[32, 64]):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_c = input_size if i == 0 else num_channels[i-1]
            out_c = num_channels[i]
            layers += [nn.Conv1d(in_c, out_c, 3, padding=dilation_size, dilation=dilation_size), nn.ReLU()]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.net(x)
        return self.fc(y[:, :, -1])

# --- 4. TRAINING ENGINES ---

def run_dl_model(model, train_loader, test_loader, criterion, optimizer):
    # Add Scheduler to improve Informer convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Step the scheduler
        scheduler.step(train_loss)
    
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            pred = model(batch_x)
            preds.append(pred.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    return np.mean((preds - actuals)**2)

def run_ml_model(model_name, train_set, test_set):
    X_train, y_train = [], []
    for i in range(len(train_set)):
        x, y = train_set[i]
        X_train.append(x)
        y_train.append(y)
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = [], []
    for i in range(len(test_set)):
        x, y = test_set[i]
        X_test.append(x)
        y_test.append(y)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Flatten Target for ML models to avoid warnings
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train_flat = y_train.ravel()
    else:
        y_train_flat = y_train

    if model_name == "Naive":
        preds = []
        for i in range(len(X_test)):
            # Predict t+1 = t (Last observed Close Price)
            last_close = X_test[i][-len(FEATURE_COLS)] 
            preds.append([last_close] * y_test.shape[1])
        preds = np.array(preds)

    elif model_name == "SVR":
        if y_train.shape[1] == 1:
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            model.fit(X_train, y_train_flat)
            preds = model.predict(X_test).reshape(-1, 1)
        else:
            model = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1))
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

    elif model_name == "RF":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_flat)
        preds = model.predict(X_test)
        if preds.ndim == 1: preds = preds.reshape(-1, 1)

    return mean_squared_error(y_test, preds)

# --- 5. MAIN EXECUTION ---
def main():
    if not os.path.exists(TRAIN_PATH):
        print("? Error: Files not found.")
        return

    print("?? Loading Data & Running Comparison...")
    df_train_full = pd.read_csv(TRAIN_PATH)
    df_test_full = pd.read_csv(TEST_PATH)

    # Scale Data
    scaler = MinMaxScaler()
    train_vals = df_train_full[FEATURE_COLS].values
    scaler.fit(train_vals)
    train_data_full = scaler.transform(train_vals)
    
    final_results = {}

    print(f"\n{'Lookback':<10} | {'Model':<12} | {'MSE (%)':<8}")
    print("-" * 40)

    # Loop 96 -> 32 -> 16
    for lookback in LOOKBACKS:
        final_results[lookback] = {}
        
        # Stitch Data (Train tail -> Test head) to keep predictions valid
        overlap = df_train_full.iloc[-lookback:][FEATURE_COLS].values
        test_vals_raw = df_test_full[FEATURE_COLS].values
        test_vals_stitched = np.concatenate([overlap, test_vals_raw], axis=0)
        test_data_stitched = scaler.transform(test_vals_stitched)
        
        # Prepare Datasets
        train_dl = TimeSeriesDataset(train_data_full, lookback, PRED_LEN, flatten=False)
        test_dl = TimeSeriesDataset(test_data_stitched, lookback, PRED_LEN, flatten=False)
        train_loader = DataLoader(train_dl, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dl, batch_size=BATCH_SIZE, shuffle=False)
        
        train_ml = TimeSeriesDataset(train_data_full, lookback, PRED_LEN, flatten=True)
        test_ml = TimeSeriesDataset(test_data_stitched, lookback, PRED_LEN, flatten=True)
        
        input_dim = len(FEATURE_COLS)

        # 1. Transformer (Baseline: No Distillation)
        model = Informer(input_dim, PRED_LEN, distil=False).to(DEVICE)
        res = run_dl_model(model, train_loader, test_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR))
        final_results[lookback]["Transformer"] = res

        # 2. Informer (Full: With Distillation)
        model = Informer(input_dim, PRED_LEN, distil=True).to(DEVICE)
        res = run_dl_model(model, train_loader, test_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR))
        final_results[lookback]["Informer"] = res

        # 3. LSTM
        model = LSTMModel(input_dim, 64, PRED_LEN).to(DEVICE)
        res = run_dl_model(model, train_loader, test_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR))
        final_results[lookback]["LSTM"] = res

        # 4. TCN
        model = TCNModel(input_dim, PRED_LEN).to(DEVICE)
        res = run_dl_model(model, train_loader, test_loader, nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR))
        final_results[lookback]["TCN"] = res

        # 5. SVR
        res = run_ml_model("SVR", train_ml, test_ml)
        final_results[lookback]["SVR"] = res

        # 6. Random Forest
        res = run_ml_model("RF", train_ml, test_ml)
        final_results[lookback]["RF"] = res

        # 7. Naive
        res = run_ml_model("Naive", train_ml, test_ml)
        final_results[lookback]["Naive"] = res
        
        print(f"{lookback:<10} | All Models   | Done")

    # --- SAVE RESULTS ---
    df_res = pd.DataFrame(final_results).T 
    
    # Convert MSE to Percentage (%)
    df_res = df_res * 100 
    
    # Reorder columns
    cols_order = ["Transformer", "Informer", "LSTM", "TCN", "SVR", "RF", "Naive"]
    df_res = df_res[cols_order]
    
    print("\n? Final Comparison Table (MSE %):")
    print(df_res.round(3))
    
    df_res.to_excel(OUTPUT_FILE)
    print(f"\n?? Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. CONFIGURATION ---
TRAIN_PATH = "/home/tom_mscds24/data/apple dataset/train_data.csv"
TEST_PATH = "/home/tom_mscds24/data/apple dataset/test_data.csv"
OUTPUT_FILE = "/home/tom_mscds24/data/apple dataset/Informer_Paper_Results.xlsx"

# The 9 Scenarios from the Paper (Input Window, Output Horizon)
WINDOW_CONFIGS = [
    (16, 1), (16, 8), (16, 16),
    (32, 1), (32, 8), (32, 16),
    (96, 1), (96, 8), (96, 16)
]

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match columns in your CSV
FEATURE_COLS = [
    'Close', 'SMA', 'RSI', 'OBV', 'ADX',
    'Gold_Close', 'USD_JPY_Close', 'sentiment_score'
]

# --- 2. INFORMER MODEL ARCHITECTURE ---
class DistillingLayer(nn.Module):
    def __init__(self, c_in):
        super(DistillingLayer, self).__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.max_pool(self.activation(self.norm(self.conv(x))))
        x = x.permute(0, 2, 1)
        return x

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=512, dropout=0.05):
        super(InformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        new_x, _ = self.attn(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class InformerEncoder(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, e_layers=2):
        super(InformerEncoder, self).__init__()
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.layers = nn.ModuleList()
        self.distill_layers = nn.ModuleList()
        for i in range(e_layers):
            self.layers.append(InformerEncoderLayer(d_model, n_heads))
            if i < e_layers - 1:
                self.distill_layers.append(DistillingLayer(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.enc_embedding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.distill_layers):
                x = self.distill_layers[i](x)
        return self.norm(x)

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=512, dropout=0.05):
        super(InformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, cross, tgt_mask=None):
        x = x + self.dropout(self.self_attn(x, x, x, attn_mask=tgt_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attn(x, cross, cross)[0])
        x = self.norm2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

class InformerDecoder(nn.Module):
    def __init__(self, dec_in, d_model, n_heads, d_layers=1):
        super(InformerDecoder, self).__init__()
        self.dec_embedding = nn.Linear(dec_in, d_model)
        self.layers = nn.ModuleList([InformerDecoderLayer(d_model, n_heads) for _ in range(d_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x, cross, tgt_mask=None):
        x = self.dec_embedding(x)
        for layer in self.layers:
            x = layer(x, cross, tgt_mask=tgt_mask)
        x = self.norm(x)
        x = self.projection(x)
        return x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, out_len, d_model=64, n_heads=4, e_layers=2, d_layers=1):
        super(Informer, self).__init__()
        self.out_len = out_len
        self.encoder = InformerEncoder(enc_in, d_model, n_heads, e_layers)
        self.decoder = InformerDecoder(dec_in, d_model, n_heads, d_layers)

    def forward(self, x_enc, x_dec):
        enc_out = self.encoder(x_enc)
        dec_seq_len = x_dec.shape[1]
        tgt_mask = torch.triu(torch.ones(dec_seq_len, dec_seq_len) * float('-inf'), diagonal=1).to(x_dec.device)
        dec_out = self.decoder(x_dec, enc_out, tgt_mask=tgt_mask)
        return dec_out[:, -self.out_len:, :]

# --- 3. DATA LOADER ---
class InformerDataset(Dataset):
    def __init__(self, data_scaled, seq_len, label_len, pred_len):
        self.data = data_scaled
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
       
        enc_input = self.data[s_begin:s_end]
        dec_input = self.data[r_begin:r_end].copy()
        dec_input[-self.pred_len:, :] = 0 # Mask future
        target = self.data[s_end : s_end + self.pred_len, 0:1]
       
        return (torch.tensor(enc_input, dtype=torch.float32),
                torch.tensor(dec_input, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))

# --- 4. EXPERIMENT RUNNER ---
def run_all_experiments():
    print("?? Loading Datasets...")
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print("? Error: Train/Test CSVs not found. Run Step 1 first.")
        return

    # Load & Scale Data
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
   
    scaler = MinMaxScaler()
    train_vals = df_train[FEATURE_COLS].values
    test_vals = df_test[FEATURE_COLS].values
   
    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals)
    test_scaled = scaler.transform(test_vals)
   
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_vals[:, 0:1]) # Close price is index 0

    writer = pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter')

    # LOOP THROUGH SCENARIOS
    for (inp_len, out_len) in WINDOW_CONFIGS:
        sheet_name = f"In_{inp_len}_Out_{out_len}"
        print(f"\n?? Scenario: Input {inp_len} -> Predict {out_len}")

        label_len = inp_len // 2
        train_set = InformerDataset(train_scaled, inp_len, label_len, out_len)
        test_set = InformerDataset(test_scaled, inp_len, label_len, out_len)
       
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

        model = Informer(
            enc_in=len(FEATURE_COLS),
            dec_in=len(FEATURE_COLS),
            out_len=out_len,
            d_model=64,
            n_heads=4
        ).to(DEVICE)
       
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(EPOCHS):
            for enc_in, dec_in, target in train_loader:
                enc_in, dec_in, target = enc_in.to(DEVICE), dec_in.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(enc_in, dec_in)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
       
        # Testing
        model.eval()
        preds_real = []
        actuals_real = []
        scaled_diffs_sq = [] # For storing scaled squared errors
       
        with torch.no_grad():
            for enc_in, dec_in, target in test_loader:
                enc_in, dec_in, target = enc_in.to(DEVICE), dec_in.to(DEVICE), target.to(DEVICE)
                output = model(enc_in, dec_in)
               
                # 1. Capture SCALED metrics (For Paper Comparison)
                # output and target are tensors on GPU, shape [Batch, Out_Len, 1]
                diff = (output - target).cpu().numpy()
                scaled_diffs_sq.append(diff**2)

                # 2. Capture REAL metrics (For Excel readability)
                pred_np = output.cpu().numpy()
                act_np = target.cpu().numpy()
               
                for i in range(pred_np.shape[0]):
                    p_real = target_scaler.inverse_transform(pred_np[i])
                    a_real = target_scaler.inverse_transform(act_np[i])
                    preds_real.append(p_real.flatten())
                    actuals_real.append(a_real.flatten())

        # Calculate Final Metrics
        # Flatten the list of arrays to get one big mean
        all_scaled_sq = np.concatenate(scaled_diffs_sq, axis=0)
        mse_scaled = np.mean(all_scaled_sq) # THIS should match the paper (~0.03)

        mse_real = np.mean((np.array(preds_real) - np.array(actuals_real))**2)
        mae_real = np.mean(np.abs(np.array(preds_real) - np.array(actuals_real)))

        # Save to Excel
        cols_p = [f"Pred_Day_{i+1}" for i in range(out_len)]
        cols_a = [f"Actual_Day_{i+1}" for i in range(out_len)]
       
        df_res = pd.DataFrame(preds_real, columns=cols_p)
        df_act = pd.DataFrame(actuals_real, columns=cols_a)
        final_sheet = pd.concat([df_act, df_res], axis=1)
       
        final_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
       
        # Write Metrics at bottom
        worksheet = writer.sheets[sheet_name]
        row_idx = len(final_sheet) + 2
       
        worksheet.write(row_idx, 0, "PAPER METRIC (Scaled MSE):")
        worksheet.write(row_idx, 1, mse_scaled)
       
        worksheet.write(row_idx + 1, 0, "Real Price MSE:")
        worksheet.write(row_idx + 1, 1, mse_real)
       
        worksheet.write(row_idx + 2, 0, "Real Price MAE:")
        worksheet.write(row_idx + 2, 1, mae_real)
       
        print(f"   ? Done. Scaled MSE: {mse_scaled:.5f} | Real MSE: {mse_real:.2f}")
       
        torch.cuda.empty_cache()

    writer.close()
    print(f"\n?? All Experiments Completed! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_all_experiments()
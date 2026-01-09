import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = "/home/tom_mscds24/data/Toyota_Informer_Sentiment_Ollama.csv" # The file with sentiment_score you created earlier
TRAIN_FILE = "/home/tom_mscds24/data/toyota dataset/train_data.csv"
TEST_FILE = "/home/tom_mscds24/data/toyota dataset/test_data.csv"

def split_dataset():
    print(f"?? Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"? Error: {INPUT_FILE} not found. Please upload it.")
        return

    df = pd.read_csv(INPUT_FILE)
   
    # 1. Ensure Time Order
    # We convert the Date column to datetime to sort correctly, then drop it if needed
    # or keep it for reference. The model uses integer indexing, so sorting is key.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
   
    # 2. Split Index (70% Train, 30% Test)
    total_rows = len(df)
    train_size = int(total_rows * 0.7)
   
    # 3. Create DataFrames
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
   
    # 4. Save
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
   
    print(f"? Data Preparation Complete!")
    print(f"   Train Set: {len(train_df)} rows -> Saved to {TRAIN_FILE}")
    print(f"   Test Set:  {len(test_df)} rows  -> Saved to {TEST_FILE}")

if __name__ == "__main__":
    split_dataset()
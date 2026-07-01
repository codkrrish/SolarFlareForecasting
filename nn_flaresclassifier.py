import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# 1. GLOBAL PARAMETERS & CONFIGURATION
# =====================================================================
DATA_PATH = r"data\out_builddata\windows_features.parquet"
OUT_DIR = r"lastrun\nn_model_output"

RUN_MODE = "binary"      # Options: "binary" (Flare vs No Flare) or "multiclass" (A/B/C/M/X)
SEQ_LENGTH = 12          # Past data lookback context window size
HIDDEN_DIM = 64          # Number of units in GRU hidden layers
NUM_LAYERS = 2           # Stacking depth of the GRU
BATCH_SIZE = 64
EPOCHS = 50               # Kept short for quick validation; scale up as needed
LR = 0.001
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# =====================================================================
# 2. DATASET DEFINITION
# =====================================================================
class SequentialFlareDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col, seq_length=12):
        self.seq_length = seq_length
        self.X_seqs = []
        self.y_labels = []
        
        # Chronological processing per independent segment_id
        grouped = dataframe.groupby('segment_id')
        for _, group in grouped:
            group = group.sort_values('window_start_time')
            features = group[feature_cols].to_numpy(dtype="float32")
            labels = group[target_col].to_numpy()
            
            if len(features) >= seq_length:
                for i in range(len(features) - seq_length + 1):
                    self.X_seqs.append(features[i : i + seq_length])
                    self.y_labels.append(labels[i + seq_length - 1])
                    
        self.X_seqs = torch.tensor(np.array(self.X_seqs), dtype=torch.float32)
        
        if RUN_MODE == "multiclass":
            self.y_labels = torch.tensor(np.array(self.y_labels), dtype=torch.long)
        else:
            self.y_labels = torch.tensor(np.array(self.y_labels), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        return self.X_seqs[idx], self.y_labels[idx]


# =====================================================================
# 3. NEURAL NETWORK ARCHITECTURE
# =====================================================================
class SequentialFlareClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(SequentialFlareClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        gru_out, _ = self.gru(x) 
        last_step_out = gru_out[:, -1, :] 
        out = self.dropout(last_step_out)
        return self.fc(out)


# =====================================================================
# 4. EVALUATION FUNCTION
# =====================================================================
def evaluate_model(model, data_loader, device, run_mode, dataset_name="Dataset"):
    """
    Evaluates the model on a given dataloader and prints a scikit-learn classification report.
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    print(f"\n⏳ Running inference for performance evaluation on: {dataset_name.upper()}...")
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            
            if run_mode == "binary":
                # Convert logits to probabilities, then threshold at 0.5
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).int().cpu().numpy()
                trues = batch_y.int().cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_trues.extend(trues.flatten())
            else:
                # Multi-class: select index containing the highest logit energy
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                trues = batch_y.cpu().numpy()
                
                all_preds.extend(preds)
                all_trues.extend(trues)
                
    print(f"\n=======================================================")
    print(f"📊 PERFORMANCE METRICS: {dataset_name.upper()} ({run_mode.upper()} MODE)")
    print(f"=======================================================")
    
    if run_mode == "binary":
        target_names = ["No Flare", "Flare"]
    else:
        # Generate generic or known string class labels safely
        unique_labels = np.unique(np.concatenate([all_trues, all_preds]))
        target_names = [f"Class {int(lbl)}" for lbl in unique_labels]
        
    print(classification_report(all_trues, all_preds, target_names=target_names, zero_division=0, digits=4))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_trues, all_preds))
    print(f"=======================================================\n")


# =====================================================================
# 5. MAIN EXECUTION FUNCTION
# =====================================================================
def main():
    print(f"🎬 Initializing Solar Flare Deep Learning Pipeline [{RUN_MODE.upper()} MODE]")
    
    # --- Step 1: Load Raw Parquet Table ---
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing feature file at target location: {DATA_PATH}")
        
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows with {df.shape[1]} columns.")
    
    NON_FEATURE_COLS = ["window_start_time", "window_end_time", "segment_id", "source_file",
                        "label_binary", "label_class", "flare_frac_in_window"]
    FEATURE_COLS = [c for c in df.columns if c not in NON_FEATURE_COLS]
    
    # Handle mathematical anomalies safely
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().fillna(0.0)

    # --- Step 2: Chronological Segment Split ---
    seg_order = df.groupby("segment_id")["window_start_time"].min().sort_values().index.tolist()
    
    train_cutoff = int(len(seg_order) * 0.70)
    val_cutoff = int(len(seg_order) * 0.85)
    
    train_segs = seg_order[:train_cutoff]
    val_segs = seg_order[train_cutoff:val_cutoff]
    test_segs = seg_order[val_cutoff:]
    
    train_df = df[df.segment_id.isin(train_segs)].copy()
    val_df = df[df.segment_id.isin(val_segs)].copy()
    test_df = df[df.segment_id.isin(test_segs)].copy()

    # --- Step 3: Sequence Normalization ---
    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])
    test_df[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

    # --- Step 4: Setup Configurations Dynamically Based on Run Mode ---
    if RUN_MODE == "binary":
        TARGET_COLUMN = "label_binary"
        NUM_CLASSES = 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
    else:
        TARGET_COLUMN = "label_class"
        if train_df[TARGET_COLUMN].dtype == object:
            mapping = {"no_flare": 0, "B": 1, "C": 2, "M": 3, "X": 4}
            train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].map(mapping)
            val_df[TARGET_COLUMN] = val_df[TARGET_COLUMN].map(mapping)
            test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].map(mapping)
        NUM_CLASSES = len(df[TARGET_COLUMN].dropna().unique())
        criterion = nn.CrossEntropyLoss()

    # --- Step 5: Instantiate Sequence Arrays & Loaders ---
    print("⏳ Transforming data frames into 3D sequential history chunks...")
    train_dataset = SequentialFlareDataset(train_df, FEATURE_COLS, TARGET_COLUMN, SEQ_LENGTH)
    val_dataset = SequentialFlareDataset(val_df, FEATURE_COLS, TARGET_COLUMN, SEQ_LENGTH)
    test_dataset = SequentialFlareDataset(test_df, FEATURE_COLS, TARGET_COLUMN, SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # Shuffle is turned off for evaluation loaders to retain execution visibility
    eval_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Step 6: Initialize Hardware Accelerator and Models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution Hardware Device Target: {device}")
    
    criterion = criterion.to(device)
    model = SequentialFlareClassifier(
        input_dim=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- Step 7: Training Loop ---
    print("🚀 Training loop initialized.")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation validation check
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item() * bx.size(0)
        epoch_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # --- Step 8: Comprehensive Pipeline Performance Evaluation ---
    # Running evaluation on training data sets checking for variance baseline
    evaluate_model(model, eval_train_loader, device, RUN_MODE, dataset_name="Training Set")
    
    # Running evaluation on held-out testing partitions to confirm true validation generalization
    evaluate_model(model, test_loader, device, RUN_MODE, dataset_name="Test Set")

    # --- Step 9: Save Operational Outputs ---
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "sequential_flare_model.pth"))
    print(f"💾 Model weight matrices successfully stored in {OUT_DIR}")


if __name__ == "__main__":
    main()
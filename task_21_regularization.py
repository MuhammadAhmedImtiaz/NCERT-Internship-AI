# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 21: REGULARIZATION TECHNIQUES IN DEEP LEARNING
# =====================================================================

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility Setup
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

# =====================================================================
# PART A: DATASET SELECTION & PREPROCESSING (20 Marks)
# =====================================================================
print("""
===================================================================
TASK 21: PART A – DATASET PREPARATION & THREE-WAY SPLIT
===================================================================
""")

# Load Dataset (Breast Cancer Wisconsin Classification)
data = load_breast_cancer()
X_raw, y_raw = data.data, data.target

# 1. Three-way Split (70% Train, 15% Validation, 15% Test)
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X_raw, y_raw, test_size=0.30, random_state=SEED, stratify=y_raw
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

# 2. Leakage-Free Scaling (fit on Train ONLY)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# 3. Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val_raw, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test_raw, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

print(f"Dataset Details: {X_raw.shape[1]} features, Binary Classification Target")
print(f"Split Ratios:")
print(f"  - Training Set:   {X_train_t.shape[0]} samples (70%)")
print(f"  - Validation Set: {X_val_t.shape[0]} samples (15%)")
print(f"  - Testing Set:    {X_test_t.shape[0]} samples (15%)\n")

# =====================================================================
# PART B & C: MODEL ARCHITECTURES & REGULARIZATION EXPERIMENTS (65 Marks)
# =====================================================================
print("""
===================================================================
TASK 21: PARTS B & C – REGULARIZATION EXPERIMENTS
===================================================================
""")

# 1. Baseline Network (Deep & Intentionally Prone to Overfitting)
class BaselineNet(nn.Module):
    def __init__(self, input_dim=30):
        super(BaselineNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# 2. Network with Dropout (p = 0.3)
class DropoutNet(nn.Module):
    def __init__(self, input_dim=30):
        super(DropoutNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# 3. Network with Batch Normalization
class BatchNormNet(nn.Module):
    def __init__(self, input_dim=30):
        super(BatchNormNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# Training Pipeline Function
def train_model(model, epochs=120, lr=0.01, early_stopping=False, patience=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
            
        tr_loss /= len(X_train_t)
        
        # Validation Pass
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * bx.size(0)
        v_loss /= len(X_val_t)
        
        train_loss_hist.append(tr_loss)
        val_loss_hist.append(v_loss)
        
        # Early Stopping Logic
        if early_stopping:
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at Epoch {epoch+1}!")
                    model.load_state_dict(best_weights)
                    break
                    
    return train_loss_hist, val_loss_hist

# Run Experiments
experiments = {
    "Baseline (No Reg)": (BaselineNet(), False),
    "Dropout (p=0.3)": (DropoutNet(), False),
    "Batch Normalization": (BatchNormNet(), False),
    "Early Stopping": (BaselineNet(), True)
}

results = {}
loss_curves = {}

for name, (model_inst, es_flag) in experiments.items():
    torch.manual_seed(SEED)
    tr_hist, val_hist = train_model(model_inst, epochs=120, lr=0.01, early_stopping=es_flag, patience=15)
    
    # Evaluation on Test Set
    model_inst.eval()
    with torch.no_grad():
        test_logits = model_inst(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1).numpy()
        
    y_true = y_test_raw
    acc  = accuracy_score(y_true, test_preds)
    prec = precision_score(y_true, test_preds, zero_division=0)
    rec  = recall_score(y_true, test_preds, zero_division=0)
    f1   = f1_score(y_true, test_preds, zero_division=0)
    
    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1,
        "CM": confusion_matrix(y_true, test_preds)
    }
    loss_curves[name] = (tr_hist, val_hist)

# Plot Loss Curves Comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (name, (tr_h, val_h)) in enumerate(loss_curves.items()):
    axes[idx].plot(tr_h, label="Train Loss", color="blue", linewidth=2)
    axes[idx].plot(val_h, label="Val Loss", color="orange", linestyle="--", linewidth=2)
    axes[idx].set_title(f"Loss Curve: {name}", fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Epochs")
    axes[idx].set_ylabel("Loss")
    axes[idx].legend()

plt.tight_layout()
plt.show()

# =====================================================================
# PART D: COMPARATIVE ANALYSIS & SUMMARY (15 Marks)
# =====================================================================
print("""
===================================================================
PART D: COMPARATIVE ANALYSIS & RECOMMENDATION
===================================================================
""")

summary_table = []
for name, metrics in results.items():
    summary_table.append({
        "Regularization Technique": name,
        "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
        "Precision": f"{metrics['Precision']:.4f}",
        "Recall": f"{metrics['Recall']:.4f}",
        "F1-Score": f"{metrics['F1-Score']:.4f}"
    })

df_summary = pd.DataFrame(summary_table)
print("--- Comparative Performance Matrix ---")
print(df_summary.to_string(index=False))

# Persist Results
df_summary.to_csv("task_21_regularization_summary.csv", index=False)
print("\nSaved summary report to 'task_21_regularization_summary.csv'!")
# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 20: END-TO-END FEEDFORWARD PIPELINE ON FASHION-MNIST / MNIST
# =====================================================================

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

# =====================================================================
# PART A: DATA PIPELINE & PREPROCESSING (20 Marks)
# =====================================================================
print("""
===================================================================
TASK 20: PART A – DATA PIPELINE & LEAKAGE-FREE PREPROCESSING
===================================================================
""")

print(f"PyTorch Version:     {torch.__version__}")
print(f"Torchvision Version: {torch.utils.__name__}")

# 1. Download Raw Dataset (Fashion-MNIST)
raw_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

# 2. Three-Way Split (80% Train, 20% Val from 60,000 train instances)
train_size = int(0.80 * len(raw_dataset))
val_size = len(raw_dataset) - train_size
train_subset, val_subset = random_split(raw_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

print(f"Three-Way Split Instance Counts:")
print(f"  - Training Set:   {len(train_subset)} samples (80% of Train Set)")
print(f"  - Validation Set: {len(val_subset)} samples (20% of Train Set)")
print(f"  - Test Set:       {len(test_dataset)} samples (Held-out Test Set)")

# 3. Computing Leakage-Free Normalization Statistics (Train Split ONLY)
train_loader_raw = DataLoader(train_subset, batch_size=len(train_subset), shuffle=False)
train_images, _ = next(iter(train_loader_raw))

mean_train = train_images.mean().item()
std_train  = train_images.std().item()

print(f"\nLeakage-Free Training Normalization Stats:")
print(f"  - Mean: {mean_train:.4f}")
print(f"  - Std:  {std_train:.4f}")
print("Explanation: Computing stats only on train_subset prevents data leakage from validation/test distributions.")

# 4. Pipeline Tensor Shape Verification
sample_img, sample_lbl = train_subset[0]
flattened_img = sample_img.view(-1)
batched_img = sample_img.unsqueeze(0)

print(f"\nTensor Shape Sanity Checks:")
print(f"  - Raw Image Shape:       {sample_img.shape} (1x28x28)")
print(f"  - Flattened Vector:      {flattened_img.shape} (784 dimensions)")
print(f"  - Batched Tensor Shape:   {batched_img.shape}")

# Data Loaders
BATCH_SIZE = 64
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean_train,), (std_train,))
])

# Re-apply transform pipeline
dataset_norm = torchvision.datasets.FashionMNIST(root="./data", train=True, download=False, transform=transform_norm)
test_dataset_norm = torchvision.datasets.FashionMNIST(root="./data", train=False, download=False, transform=transform_norm)
train_set, val_set = random_split(dataset_norm, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset_norm, batch_size=BATCH_SIZE, shuffle=False)

# =====================================================================
# PART B: MODEL ARCHITECTURE & JUSTIFICATION (20 Marks)
# =====================================================================
print("""
===================================================================
TASK 20: PART B – ARCHITECTURE & DESIGN JUSTIFICATION
===================================================================
""")

class FashionMLP(nn.Module):
    def __init__(self, input_dim=784, hidden1=128, hidden2=64, output_dim=10):
        super(FashionMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = FashionMLP()

# Hand Parameter Calculation Verification
# Layer 1: (784 * 128) + 128 = 100,480
# Layer 2: (128 * 64) + 64   = 8,256
# Layer 3: (64 * 10) + 10    = 650
# Total: 100,480 + 8,256 + 650 = 109,386
hand_params = (784 * 128 + 128) + (128 * 64 + 64) + (64 * 10 + 10)
torch_params = sum(p.numel() for p in model.parameters())

print(f"Model Parameter Verification:")
print(f"  - Calculated Hand Count: {hand_params:,} parameters")
print(f"  - PyTorch numel() Count: {torch_params:,} parameters")
print(f"  - Parameter Counts Match Perfectly: {hand_params == torch_params}\n")

# Loss Function Equivalency Check
sample_batch_x, sample_batch_y = next(iter(train_loader))
logits = model(sample_batch_x)

loss_ce = nn.CrossEntropyLoss()(logits, sample_batch_y)

softmax_probs = torch.softmax(logits, dim=1)
log_probs = torch.log(softmax_probs)
loss_nll = nn.NLLLoss()(log_probs, sample_batch_y)

loss_diff = abs(loss_ce.item() - loss_nll.item())
print(f"Loss Pairing Check (CrossEntropyLoss vs Softmax+NLLLoss):")
print(f"  - CrossEntropyLoss: {loss_ce.item():.6f}")
print(f"  - NLLLoss:          {loss_nll.item():.6f}")
print(f"  - Difference:       {loss_diff:.8e} (Match: {loss_diff < 1e-6})")

# =====================================================================
# PART C: TRAINING, EVALUATION & EXPERIMENTS (40 Marks)
# =====================================================================
print("""
===================================================================
TASK 20: PART C – TRAINING PIPELINE & ABLATION EXPERIMENTS
===================================================================
""")

def train_and_evaluate(model_instance, lr=0.001, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_instance.parameters(), lr=lr)
    
    train_loss_hist, val_loss_hist, val_acc_hist = [], [], []
    
    for epoch in range(epochs):
        model_instance.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_instance(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_set)
        
        # Validation Pass
        model_instance.eval()
        running_val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model_instance(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.numpy())
                val_targets.extend(labels.numpy())
                
        epoch_val_loss = running_val_loss / len(val_set)
        epoch_val_acc  = accuracy_score(val_targets, val_preds)
        
        train_loss_hist.append(epoch_train_loss)
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
        
    return train_loss_hist, val_loss_hist, val_acc_hist

print("--- Training Base Model (784 -> 128 -> 64 -> 10) ---")
torch.manual_seed(SEED)
base_model = FashionMLP()
base_tr_loss, base_val_loss, base_val_acc = train_and_evaluate(base_model, lr=0.001, epochs=10)

# Final Test Set Evaluation
base_model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = base_model(images)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.numpy())
        test_targets.extend(labels.numpy())

test_acc  = accuracy_score(test_targets, test_preds)
test_prec = precision_score(test_targets, test_preds, average='macro')
test_rec  = recall_score(test_targets, test_preds, average='macro')
test_f1   = f1_score(test_targets, test_preds, average='macro')

print(f"\n--- Final Test Set Evaluation (Base Model) ---")
print(f"Test Accuracy:  {test_acc * 100:.2f}%")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall:    {test_rec:.4f}")
print(f"Test F1-Score:  {test_f1:.4f}\n")

# Ablation Study: Model Configuration Comparison (Without Hidden Layer 2: 784 -> 128 -> 10)
class ShallowMLP(nn.Module):
    def __init__(self):
        super(ShallowMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

print("--- Running Controlled Ablation Model (Shallow Architecture: 784 -> 128 -> 10) ---")
torch.manual_seed(SEED)
shallow_model = ShallowMLP()
_, _, shallow_val_acc = train_and_evaluate(shallow_model, lr=0.001, epochs=10)

shallow_model.eval()
s_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        s_preds.extend(torch.argmax(shallow_model(images), dim=1).numpy())
shallow_acc = accuracy_score(test_targets, s_preds)

print(f"\nAblation Comparison:")
print(f"  - Base Model (2 Hidden Layers) Test Accuracy:    {test_acc * 100:.2f}%")
print(f"  - Shallow Model (1 Hidden Layer) Test Accuracy: {shallow_acc * 100:.2f}%")

# Plotting Training Curves & Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(base_tr_loss, label="Training Loss", color="blue", linewidth=2)
axes[0].plot(base_val_loss, label="Validation Loss", color="red", linestyle="--", linewidth=2)
axes[0].set_title("Base Model Loss Curves", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cm = confusion_matrix(test_targets, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title("Test Set Confusion Matrix", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# =====================================================================
# PART D: PERSISTENCE & DOCUMENTATION (20 Marks)
# =====================================================================
print("""
===================================================================
TASK 20: PART D – PERSISTENCE & RESULTS SUMMARY
===================================================================
""")

# Save State Dict
torch.save(base_model.state_dict(), "fashion_mlp_state_dict.pth")
print("Saved base model state dict to 'fashion_mlp_state_dict.pth'.")

# Reload Verification
reloaded_model = FashionMLP()
reloaded_model.load_state_dict(torch.load("fashion_mlp_state_dict.pth"))
reloaded_model.eval()

held_out_x, held_out_y = next(iter(test_loader))
with torch.no_grad():
    orig_p = torch.argmax(base_model(held_out_x), dim=1)
    reld_p = torch.argmax(reloaded_model(held_out_x), dim=1)

match_check = torch.equal(orig_p, reld_p)
print(f"Reloaded Model Predictions Exactly Match Original: {match_check}\n")

# Summary Table Output
results_df = pd.DataFrame([{
    "Model Architecture": "Base MLP (784-128-64-10)",
    "Test Accuracy": f"{test_acc * 100:.2f}%",
    "Macro Precision": f"{test_prec:.4f}",
    "Macro Recall": f"{test_rec:.4f}",
    "Macro F1-Score": f"{test_f1:.4f}",
    "Trainable Parameters": f"{torch_params:,}"
}, {
    "Model Architecture": "Shallow MLP (784-128-10)",
    "Test Accuracy": f"{shallow_acc * 100:.2f}%",
    "Macro Precision": "N/A",
    "Macro Recall": "N/A",
    "Macro F1-Score": "N/A",
    "Trainable Parameters": f"{sum(p.numel() for p in shallow_model.parameters()):,}"
}])

print("--- Final Performance Summary Table ---")
print(results_df.to_string(index=False))
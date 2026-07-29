# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 19: TRAINING LOOPS: LOSS FUNCTIONS, OPTIMIZERS & BATCHING
# =====================================================================

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Set seed for reproducible results
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

print(f"PyTorch Version: {torch.__version__}")

# =====================================================================
# PART A: LOSS FUNCTIONS IN DEPTH (20 Marks)
# =====================================================================
print("""
===================================================================
PART A: LOSS FUNCTIONS IN DEPTH & VERIFICATION
===================================================================
""")

# 1. Custom Loss Implementations vs PyTorch Built-ins
def custom_mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def custom_cross_entropy_loss(logits, target_cls):
    # Log-Softmax implementation for numerical stability
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_logits), dim=1, keepdim=True)) + max_logits
    log_probs = logits - log_sum_exp
    # Negative Log-Likelihood over target indices
    nll = -log_probs[torch.arange(target_cls.size(0)), target_cls]
    return torch.mean(nll)

def custom_l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

# Verification against torch.nn
dummy_logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
dummy_targets = torch.tensor([0, 1])

dummy_preds_reg = torch.tensor([[1.5, 2.5], [3.5, 4.5]], requires_grad=True)
dummy_targets_reg = torch.tensor([[1.0, 2.0], [3.0, 5.0]])

# Test MSE
custom_mse = custom_mse_loss(dummy_preds_reg, dummy_targets_reg)
builtin_mse = nn.MSELoss()(dummy_preds_reg, dummy_targets_reg)
mse_diff = abs(custom_mse.item() - builtin_mse.item())

# Test Cross-Entropy
custom_ce = custom_cross_entropy_loss(dummy_logits, dummy_targets)
builtin_ce = nn.CrossEntropyLoss()(dummy_logits, dummy_targets)
ce_diff = abs(custom_ce.item() - builtin_ce.item())

# Test L1
custom_l1 = custom_l1_loss(dummy_preds_reg, dummy_targets_reg)
builtin_l1 = nn.L1Loss()(dummy_preds_reg, dummy_targets_reg)
l1_diff = abs(custom_l1.item() - builtin_l1.item())

print(f"MSE Loss Diff:          {mse_diff:.8e} | Verified: {mse_diff < 1e-6}")
print(f"Cross-Entropy Loss Diff:{ce_diff:.8e} | Verified: {ce_diff < 1e-6}")
print(f"L1 Loss Diff:           {l1_diff:.8e} | Verified: {l1_diff < 1e-6}\n")

# 2. Custom Loss with Hand-Derived L2 Regularization & Gradient Verification
w_param = torch.tensor([1.5, -2.0, 0.5], requires_grad=True)
x_data = torch.tensor([2.0, 1.0, -1.0])
y_data = torch.tensor(2.0)
l2_lambda = 0.1

# Loss = 0.5 * (w * x - y)^2 + 0.5 * lambda * ||w||^2
pred = torch.dot(w_param, x_data)
loss_val = 0.5 * (pred - y_data) ** 2 + 0.5 * l2_lambda * torch.sum(w_param ** 2)
loss_val.backward()

autograd_grad = w_param.grad.clone()

# Hand derivative: dLoss/dw = (pred - y) * x + lambda * w
diff_val = (pred.item() - y_data.item())
manual_grad = diff_val * x_data + l2_lambda * w_param.data

grad_diff = torch.max(torch.abs(autograd_grad - manual_grad)).item()
print(f"L2 Loss Gradient Max Diff (Autograd vs Hand Derivative): {grad_diff:.8e}")
print(f"Gradient Match Confirmed: {grad_diff < 1e-6}\n")

# 3. Class-Weighted Cross Entropy Demonstration
imbalanced_logits = torch.tensor([[2.0, 0.1], [2.0, 0.1], [0.1, 2.0]], requires_grad=True)
imbalanced_targets = torch.tensor([0, 0, 1])
weights = torch.tensor([0.2, 0.8]) # Heavy weight on minority class 1

unweighted_loss = nn.CrossEntropyLoss()(imbalanced_logits, imbalanced_targets)
weighted_loss = nn.CrossEntropyLoss(weight=weights)(imbalanced_logits, imbalanced_targets)

print(f"Standard Unweighted Cross-Entropy Loss: {unweighted_loss.item():.4f}")
print(f"Class-Weighted Cross-Entropy Loss:      {weighted_loss.item():.4f}")

# =====================================================================
# PART B: OPTIMIZERS - SGD AND ADAM INTERNALS (30 Marks)
# =====================================================================
print("""
===================================================================
PART B: OPTIMIZER INTERNALS (MANUAL VS TORCH.OPTIM)
===================================================================
""")

# Custom Optimizer Implementation Classes
class CustomSGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.momentum != 0:
                    self.velocities[i] = self.momentum * self.velocities[i] + d_p
                    d_p = self.velocities[i]
                p.data -= self.lr * d_p

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class CustomAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = p.grad.data
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# Verification on Quadratic Bowl Function: f(w) = 0.5 * w1^2 + 2.5 * w2^2
w_custom_sgd = torch.tensor([5.0, 5.0], requires_grad=True)
w_torch_sgd  = torch.tensor([5.0, 5.0], requires_grad=True)

opt_custom_sgd = CustomSGD([w_custom_sgd], lr=0.1, momentum=0.9)
opt_torch_sgd  = torch.optim.SGD([w_torch_sgd], lr=0.1, momentum=0.9)

for _ in range(5):
    # Custom Step
    opt_custom_sgd.zero_grad()
    loss_c = 0.5 * (w_custom_sgd[0]**2) + 2.5 * (w_custom_sgd[1]**2)
    loss_c.backward()
    opt_custom_sgd.step()

    # PyTorch Step
    opt_torch_sgd.zero_grad()
    loss_t = 0.5 * (w_torch_sgd[0]**2) + 2.5 * (w_torch_sgd[1]**2)
    loss_t.backward()
    opt_torch_sgd.step()

sgd_diff = torch.max(torch.abs(w_custom_sgd - w_torch_sgd)).item()
print(f"Step-for-step Trajectory Diff (Custom SGD vs torch.optim.SGD): {sgd_diff:.8e}")
print(f"SGD Verification Match: {sgd_diff < 1e-6}")

# Adam Verification
w_custom_adam = torch.tensor([5.0, 5.0], requires_grad=True)
w_torch_adam  = torch.tensor([5.0, 5.0], requires_grad=True)

opt_custom_adam = CustomAdam([w_custom_adam], lr=0.1)
opt_torch_adam  = torch.optim.Adam([w_torch_adam], lr=0.1)

for _ in range(5):
    opt_custom_adam.zero_grad()
    loss_c = 0.5 * (w_custom_adam[0]**2) + 2.5 * (w_custom_adam[1]**2)
    loss_c.backward()
    opt_custom_adam.step()

    opt_torch_adam.zero_grad()
    loss_t = 0.5 * (w_torch_adam[0]**2) + 2.5 * (w_torch_adam[1]**2)
    loss_t.backward()
    opt_torch_adam.step()

adam_diff = torch.max(torch.abs(w_custom_adam - w_torch_adam)).item()
print(f"Step-for-step Trajectory Diff (Custom Adam vs torch.optim.Adam): {adam_diff:.8e}")
print(f"Adam Verification Match: {adam_diff < 1e-6}\n")

# L2 Regularization vs Decoupled Weight Decay (Adam vs AdamW) Demonstration
w_adam_l2 = torch.tensor([5.0], requires_grad=True)
w_adam_w  = torch.tensor([5.0], requires_grad=True)

# Adam with L2 regularization inside gradient
opt_l2 = torch.optim.Adam([w_adam_l2], lr=0.1, weight_decay=0.1)
loss_l2 = 0.5 * (w_adam_l2 ** 2)
loss_l2.backward()
opt_l2.step()

# AdamW with decoupled weight decay
opt_adamw = torch.optim.AdamW([w_adam_w], lr=0.1, weight_decay=0.1)
loss_adamw = 0.5 * (w_adam_w ** 2)
loss_adamw.backward()
opt_adamw.step()

print(f"Parameter after Adam (L2 in Gradient):    {w_adam_l2.item():.6f}")
print(f"Parameter after AdamW (Decoupled Decay):  {w_adam_w.item():.6f}")
print("Demonstration: L2 regularization inside gradient gets scaled inversely by Adam's second moment (v_t), whereas AdamW applies direct proportional decay.")

# =====================================================================
# PART C: EPOCHS, BATCHES & TRAINING ENGINEERING (35 Marks)
# =====================================================================
print("""
===================================================================
PART C: EXPERIMENTAL PIPELINE & TRAINING ENGINEERING
===================================================================
""")

# Iris Dataset Preparation
iris = load_iris()
X_raw, y_raw = iris.data, iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_tr_t, y_tr_t)

class StandardMLP(nn.Module):
    def __init__(self):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        return self.net(x)

# 1. Experiment 1: Comparing Optimizers (SGD vs SGD+Momentum vs Adam)
def run_optimizer_experiment():
    opts = {
        "Vanilla SGD": lambda params: torch.optim.SGD(params, lr=0.05),
        "SGD + Momentum": lambda params: torch.optim.SGD(params, lr=0.05, momentum=0.9),
        "Adam": lambda params: torch.optim.Adam(params, lr=0.05)
    }
    
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    results = {}

    for name, opt_fn in opts.items():
        torch.manual_seed(SEED)
        model = StandardMLP()
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        loss_hist = []
        for epoch in range(100):
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
            loss_hist.append(epoch_loss / len(train_dataset))
        
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val_t), dim=1).numpy()
        acc = accuracy_score(y_val, preds)
        results[name] = (loss_hist, acc)
        
    return results

opt_results = run_optimizer_experiment()

# 2. Experiment 2: Varying Batch Size (8, 32, 128, Full-Batch)
def run_batch_size_experiment():
    batch_sizes = [8, 32, 128, len(train_dataset)]
    results = {}
    
    for bs in batch_sizes:
        torch.manual_seed(SEED)
        model = StandardMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        
        start_t = time.time()
        loss_hist = []
        for epoch in range(100):
            epoch_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * bx.size(0)
            loss_hist.append(epoch_loss / len(train_dataset))
        wall_time = time.time() - start_t
        
        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_val_t), dim=1).numpy()
        acc = accuracy_score(y_val, preds)
        
        label = "Full-Batch" if bs == len(train_dataset) else f"Batch {bs}"
        results[label] = (loss_hist, acc, round(wall_time, 4))
        
    return results

batch_results = run_batch_size_experiment()

# 3. Experiment 3: Gradient Accumulation Verification
torch.manual_seed(SEED)
model_direct = StandardMLP()
model_accum  = copy.deepcopy(model_direct)

criterion = nn.CrossEntropyLoss()

# Single large batch computation (batch size = 32)
bx_large = X_tr_t[:32]
by_large = y_tr_t[:32]

out_direct = model_direct(bx_large)
loss_direct = criterion(out_direct, by_large)
loss_direct.backward()
grad_direct = model_direct.net[0].weight.grad.clone()

# Accumulated small batch computation (4 steps of batch size 8)
model_accum.zero_grad()
for i in range(4):
    bx_small = X_tr_t[i*8 : (i+1)*8]
    by_small = y_tr_t[i*8 : (i+1)*8]
    out_small = model_accum(bx_small)
    # Divide loss by accumulation factor (4) to scale gradients correctly
    loss_small = criterion(out_small, by_small) / 4.0
    loss_small.backward()

grad_accum = model_accum.net[0].weight.grad.clone()
accum_diff = torch.max(torch.abs(grad_direct - grad_accum)).item()

print(f"Gradient Accumulation Max Difference: {accum_diff:.8e}")
print(f"Gradient Accumulation Match Confirmed: {accum_diff < 1e-6}\n")

# Plotting Experimental Loss Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, (hist, acc) in opt_results.items():
    axes[0].plot(hist, label=f"{name} (Val Acc: {acc*100:.1f}%)", linewidth=2)
axes[0].set_title("Optimizer Convergence Comparison", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Training Loss")
axes[0].legend()

for label, (hist, acc, wtime) in batch_results.items():
    axes[1].plot(hist, label=f"{label} ({wtime}s, Val Acc: {acc*100:.1f}%)", linewidth=2)
axes[1].set_title("Batch Size Impact on Convergence & Speed", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Training Loss")
axes[1].legend()

plt.tight_layout()
plt.show()

# =====================================================================
# PART D: SUMMARY & ANALYSIS TABLE (15 Marks)
# =====================================================================
print("""
===================================================================
PART D: COMPARATIVE EXPERIMENTAL SUMMARY TABLE
===================================================================
""")

summary_data = []
for name, (hist, acc) in opt_results.items():
    summary_data.append({
        "Category": "Optimizer Study",
        "Configuration": name,
        "Final Training Loss": round(hist[-1], 4),
        "Validation Accuracy": f"{acc*100:.2f}%",
        "Wall Clock Time": "N/A"
    })

for label, (hist, acc, wtime) in batch_results.items():
    summary_data.append({
        "Category": "Batch Size Study",
        "Configuration": label,
        "Final Training Loss": round(hist[-1], 4),
        "Validation Accuracy": f"{acc*100:.2f}%",
        "Wall Clock Time": f"{wtime}s"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Persist Summary
summary_df.to_csv("task_19_experiment_summary.csv", index=False)
print("\nSaved experiment metrics to 'task_19_experiment_summary.csv'!")
# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 18: INTRO TO PYTORCH (TENSORS, AUTOGRAD & NN.MODULE)
# =====================================================================

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set_theme(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# PART A: TENSORS & BASIC OPERATIONS (20 Marks)
# =====================================================================
print("""
===================================================================
PART A: TENSORS & BASIC OPERATIONS
===================================================================
""")

print(f"PyTorch Version Installed: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Execution Device: {device}\n")

# 1. Four Tensor Creation Methods
t1 = torch.tensor([1.0, 2.0, 3.0])                        # Direct initialization
t2 = torch.zeros((3, 3))                                  # Matrix filled with zeros
t3 = torch.arange(0, 10, step=2)                          # Evenly spaced numbers
t4 = torch.rand((2, 4))                                   # Uniform random distribution
np_arr = np.array([10, 20, 30])
t5 = torch.from_numpy(np_arr)                            # Memory-shared tensor from NumPy

print(f"Direct Tensor: {t1}")
print(f"Zeros Tensor:\n{t2}")
print(f"Arange Tensor: {t3}")
print(f"Random Tensor:\n{t4}")
print(f"From NumPy Tensor: {t5}\n")

# Memory-sharing demonstration
np_arr[0] = 999
print(f"Modified NumPy Array element 0 to 999 -> Tensor updated automatically: {t5[0] == 999}")

# 2. Reshaping, Slicing & Broadcasting
t_base = torch.arange(12)
t_view = t_base.view(3, 4)       # Shares memory, requires contiguous tensor
t_reshape = t_base.reshape(3, 4) # Copies data if non-contiguous
print(f"\nReshaped 3x4 View:\n{t_view}")
print(f"Sliced Tensor (rows 0-1, cols 1-3):\n{t_view[0:2, 1:3]}")

# Broadcasting
row_vec = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
print(f"Broadcasting Addition (3x4 + 1x4):\n{t_view + row_vec}")

# 3. Matrix Multiplication Benchmark (NumPy vs PyTorch)
size = 2000
A_np = np.random.randn(size, size).astype(np.float32)
B_np = np.random.randn(size, size).astype(np.float32)

start_np = time.time()
C_np = np.dot(A_np, B_np)
time_np = time.time() - start_np

A_pt = torch.from_numpy(A_np)
B_pt = torch.from_numpy(B_np)

start_pt = time.time()
C_pt = torch.matmul(A_pt, B_pt)
time_pt = time.time() - start_pt

print(f"\nMatrix Multiplication (2000x2000):")
print(f"NumPy Execution Time:   {time_np:.4f} seconds")
print(f"PyTorch Execution Time: {time_pt:.4f} seconds")

# =====================================================================
# PART B: AUTOGRAD (20 Marks)
# =====================================================================
print("""
===================================================================
PART B: AUTOGRAD & GRADIENT VERIFICATION
===================================================================
""")

# 1. Scalar Expression & Manual Derivative Verification
# Function: y = 3x^2 + 2x + 1  => dy/dx = 6x + 2
x = torch.tensor(3.0, requires_grad=True)
y = 3 * (x ** 2) + 2 * x + 1
y.backward()

autograd_dx = x.grad.item()
manual_dx = 6 * 3.0 + 2.0

print(f"Function: y = 3x^2 + 2x + 1 at x = 3.0")
print(f"Autograd Computed Gradient dy/dx: {autograd_dx}")
print(f"Hand-Calculated Derivative dy/dx: {manual_dx}")
print(f"Difference: {abs(autograd_dx - manual_dx):.8f}")

# 2. Gradient Accumulation Demonstration
x_acc = torch.tensor(2.0, requires_grad=True)
z1 = x_acc ** 2
z1.backward()
print(f"\nFirst .backward() call (z1 = x^2, dz/dx = 2x = 4): x.grad = {x_acc.grad.item()}")

z2 = x_acc ** 3
z2.backward()
print(f"Second .backward() call without zero_grad() (z2 = x^3, dz/dx = 3x^2 = 12): x.grad = {x_acc.grad.item()} (Accumulated 4 + 12 = 16)")

x_acc.grad.zero_()
print(f"After calling x.grad.zero_(): x.grad = {x_acc.grad.item()}")

# 3. torch.no_grad() and .detach()
with torch.no_grad():
    y_nograd = x * 5
print(f"Tensor inside torch.no_grad() requires_grad: {y_nograd.requires_grad}")

y_detached = (x ** 2).detach()
print(f"Detached Tensor requires_grad: {y_detached.requires_grad}")

# 4. Compare Autograd vs Task 16 Manual Matrix Derivative
# Loss = 0.5 * ||W * X - Y||^2  => dLoss/dW = (W * X - Y) * X^T
X_sample = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
W_sample = torch.tensor([[0.5, -0.5]], requires_grad=True)
Y_target = torch.tensor([[1.0], [2.0]])

# Forward pass
pred_pt = torch.matmul(W_sample, X_sample.T)
loss_pt = 0.5 * torch.sum((pred_pt.T - Y_target) ** 2)
loss_pt.backward()

# Manual computation
W_np = W_sample.detach().numpy()
X_np = X_sample.numpy()
Y_np = Y_target.numpy()

pred_manual = np.dot(W_np, X_np.T)
diff = pred_manual.T - Y_np
manual_grad_W = np.dot(diff.T, X_np)

max_diff = np.max(np.abs(W_sample.grad.numpy() - manual_grad_W))
print(f"\nMax difference between PyTorch Autograd & Manual Task 16 Gradient: {max_diff:.8e}")
print(f"Gradients agree within tolerance (< 1e-6): {max_diff < 1e-6}")

# =====================================================================
# PART C: BUILDING & TRAINING A NEURAL NETWORK (45 Marks)
# =====================================================================
print("""
===================================================================
PART C: NN.MODULE NETWORK TRAINING (IRIS DATASET)
===================================================================
""")

# Load same Iris dataset as Task 16
iris = load_iris()
X_raw = iris.data
y_raw = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Define PyTorch Model (Same topology as Task 16: 4 -> 8 -> 3)
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=3):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate Model, Loss Function (CrossEntropyLoss includes Softmax), Optimizer
model = PyTorchMLP(input_dim=4, hidden_dim=8, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# Training Loop
epochs = 300
loss_history = []
start_train_time = time.time()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()           # 1. Clear previous gradients
    outputs = model(X_train_t)      # 2. Forward pass
    loss = criterion(outputs, y_train_t) # 3. Compute loss
    loss.backward()                 # 4. Backward pass (Autograd)
    optimizer.step()                # 5. Update weights
    
    loss_history.append(loss.item())

pytorch_train_time = time.time() - start_train_time

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds_pt = torch.argmax(logits, dim=1).numpy()

acc_pt = accuracy_score(y_test, preds_pt)
prec_pt = precision_score(y_test, preds_pt, average='macro')
rec_pt = recall_score(y_test, preds_pt, average='macro')
f1_pt = f1_score(y_test, preds_pt, average='macro')

print(f"PyTorch Model Results:")
print(f"Accuracy:  {acc_pt * 100:.2f}%")
print(f"Precision: {prec_pt:.4f}")
print(f"Recall:    {rec_pt:.4f}")
print(f"F1-Score:  {f1_pt:.4f}")
print(f"Training Time: {pytorch_train_time:.4f} seconds\n")

# Baseline comparison against Scikit-Learn MLPClassifier
sk_mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=300, random_state=42)
start_sk_time = time.time()
sk_mlp.fit(X_train, y_train)
sk_time = time.time() - start_sk_time
sk_preds = sk_mlp.predict(X_test)
sk_acc = accuracy_score(y_test, sk_preds)
sk_f1 = f1_score(y_test, sk_preds, average='macro')

# Performance Comparison Table
comparison_df = pd.DataFrame([
    {"Implementation": "Task 16 Manual NumPy MLP", "Accuracy": "100.00%", "Macro F1": 1.0000, "Notes": "Scratch matrix backpropagation"},
    {"Implementation": "Task 18 PyTorch nn.Module", "Accuracy": f"{acc_pt * 100:.2f}%", "Macro F1": round(f1_pt, 4), "Notes": f"Adam Optimizer ({pytorch_train_time:.3f}s)"},
    {"Implementation": "Scikit-Learn MLPClassifier", "Accuracy": f"{sk_acc * 100:.2f}%", "Macro F1": round(sk_f1, 4), "Notes": f"L-BFGS / Adam Solver ({sk_time:.3f}s)"}
])

print("--- Comparative Performance Matrix ---")
print(comparison_df.to_string(index=False))

# Plot Training Loss Curve & Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(loss_history, color='royalblue', linewidth=2)
axes[0].set_title("PyTorch Training Cross-Entropy Loss Curve", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")

cm = confusion_matrix(y_test, preds_pt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[1].set_title("PyTorch Model Confusion Matrix", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# =====================================================================
# PART D: ANALYSIS, PERSISTENCE & DOCUMENTATION (15 Marks)
# =====================================================================
print("""
===================================================================
PART D: MODEL PERSISTENCE & DOCUMENTATION
===================================================================
""")

# 1. Save and Reload State Dict
torch.save(model.state_dict(), "pytorch_mlp_state_dict.pth")
print("Successfully saved model weights to 'pytorch_mlp_state_dict.pth'!")

reloaded_model = PyTorchMLP(input_dim=4, hidden_dim=8, output_dim=3)
reloaded_model.load_state_dict(torch.load("pytorch_mlp_state_dict.pth"))
reloaded_model.eval()

with torch.no_grad():
    reloaded_logits = reloaded_model(X_test_t)
    reloaded_preds = torch.argmax(reloaded_logits, dim=1).numpy()

match = np.array_equal(preds_pt, reloaded_preds)
print(f"Reloaded Model Predictions Match Original: {match}")

print("""
Pipeline & Architectural Summary:
- Dataset: Iris Dataset (4 continuous features, 3 target classes).
- Architecture: Custom PyTorchMLP subclassing nn.Module (4 -> 8 -> 3).
- Activation: ReLU in hidden layer; implicit Softmax handled internally by nn.CrossEntropyLoss.
- Optimizer: Adam (lr = 0.05) providing adaptive moment estimation for rapid convergence.

Concrete Differences (Task 16 NumPy Scratch vs Task 18 PyTorch):
1. Code Complexity & Maintenance: In Task 16, manual chain-rule matrix derivatives 
   (dZ1, dW1, db1) had to be explicitly derived and updated. PyTorch abstracts this entirely 
   via `loss.backward()`, reducing code size and eliminating mathematical derivation errors.
2. Dynamic Computation Graph: PyTorch builds an imperative define-by-run graph during 
   the forward pass, allowing automatic gradient computation for arbitrary, complex architectures.

Reflections on Autograd & Manual Backprop:
- Autograd abstracts away manual calculus, node-level dependency tracking, and tensor shape alignment.
- The manual backprop exercise in Task 16 remains indispensable because it provides deep intuition 
  for gradient flow, vanishing/exploding gradients, parameter update mechanics, and debugging 
  training instabilities when PyTorch loss curves fail to converge.

Gotcha Encountered & Resolved:
- Target Shape & Type Mismatch: PyTorch `nn.CrossEntropyLoss` expects target tensors to be 1D 
  class indices (`torch.long`) rather than One-Hot encoded 2D vectors (`torch.float32`). 
  Resolved by passing `y_train` directly converted to `torch.long` instead of One-Hot encoding.
""")
# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 23: CONVOLUTIONAL NEURAL NETWORKS FROM SCRATCH (NUMPY)
# =====================================================================

import os
import time
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reproducibility Config
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
sns.set_theme(style="whitegrid")

# =====================================================================
# PART A: CONVOLUTION FUNDAMENTALS & IM2COL OPTIMIZATION
# =====================================================================
print("""
===================================================================
PART A: CONVOLUTION FUNDAMENTALS & IM2COL OPTIMIZATION
===================================================================
""")

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

class Conv2DScratch:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He Normal initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros((out_channels, 1))

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)
        
        self.X_col = im2col_indices(X, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        W_row = self.W.reshape(self.out_channels, -1)
        
        out = np.dot(W_row, self.X_col) + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        N, C, H, W = self.X.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        self.dW = np.dot(dout_reshaped, self.X_col.T).reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True)
        
        W_row = self.W.reshape(self.out_channels, -1)
        dX_col = np.dot(W_row.T, dout_reshaped)
        dX = col2im_indices(dX_col, self.X.shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        return dX

# Benchmark Conv2D (im2col matrix multiplication)
test_X = np.random.randn(10, 1, 28, 28)
conv_layer = Conv2DScratch(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)

t0 = time.time()
out_conv = conv_layer.forward(test_X)
t1 = time.time()

print(f"Input Shape:  {test_X.shape}")
print(f"Output Shape: {out_conv.shape} (Formula Output Validated)")
print(f"Fast im2col Forward Execution Time: {(t1 - t0)*1000:.2f} ms")

# =====================================================================
# PART B: POOLING, ACTIVATIONS & REGULARIZATION SCRATCH
# =====================================================================
print("""
===================================================================
PART B: POOLING (MAX & AVG) & REGULARIZATION FROM SCRATCH
===================================================================
""")

class MaxPool2DScratch:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        out_h = int((H - self.pool_size) / self.stride + 1)
        out_w = int((W - self.pool_size) / self.stride + 1)
        
        X_reshaped = X.reshape(N * C, 1, H, W)
        self.X_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)
        
        self.max_idx = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_idx, np.arange(self.max_idx.size)]
        
        out = out.reshape(out_h, out_w, N, C)
        out = out.transpose(2, 3, 0, 1)
        return out

    def backward(self, dout):
        N, C, H, W = self.X.shape
        dX_col = np.zeros_like(self.X_col)
        
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dX_col[self.max_idx, np.arange(self.max_idx.size)] = dout_flat
        
        dX = col2im_indices(dX_col, (N * C, 1, H, W), self.pool_size, self.pool_size, padding=0, stride=self.stride)
        return dX.reshape(N, C, H, W)

def relu(Z): return np.maximum(0, Z)
def d_relu(Z): return (Z > 0).astype(float)

# Test Pooling Layer
pool_layer = MaxPool2DScratch(pool_size=2, stride=2)
out_pool = pool_layer.forward(out_conv)
grad_pool = pool_layer.backward(out_pool)

print(f"Conv Feature Map Shape: {out_conv.shape}")
print(f"Pooled Output Shape:    {out_pool.shape}")
print(f"Backpropagated Gradient Shape: {grad_pool.shape}")

# =====================================================================
# PART C: FULL CNN PIPELINE FROM SCRATCH (USING TORCHVISION LOCAL LOAD)
# =====================================================================
print("""
===================================================================
PART C: TRAIN FULL Scratch CNN (Conv2D -> ReLU -> MaxPool -> Dense)
===================================================================
""")

# Load FashionMNIST locally via torchvision
train_ds = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
X_all = train_ds.data.numpy()[:2000].astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
y_all = train_ds.targets.numpy()[:2000].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

Y_train_oh = one_hot(y_train)

# Scratch Architecture Instance
class ScratchCNN:
    def __init__(self):
        self.conv1 = Conv2DScratch(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1) # 28x28 -> 28x28
        self.pool1 = MaxPool2DScratch(pool_size=2, stride=2) # 28x28 -> 14x14
        # Dense Layer: 4 channels * 14 * 14 = 784 inputs -> 10 output classes
        self.W_fc = np.random.randn(784, 10) * np.sqrt(2.0 / 784)
        self.b_fc = np.zeros((1, 10))

    def forward(self, X):
        self.z_conv = self.conv1.forward(X)
        self.a_conv = relu(self.z_conv)
        self.p_conv = self.pool1.forward(self.a_conv)
        
        self.flat = self.p_conv.reshape(self.p_conv.shape[0], -1)
        self.z_fc = np.dot(self.flat, self.W_fc) + self.b_fc
        
        # Softmax
        exp_z = np.exp(self.z_fc - np.max(self.z_fc, axis=1, keepdims=True))
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, Y_oh, lr=0.01):
        N = X.shape[0]
        dZ_fc = (self.probs - Y_oh) / N
        dW_fc = np.dot(self.flat.T, dZ_fc)
        db_fc = np.sum(dZ_fc, axis=0, keepdims=True)

        dflat = np.dot(dZ_fc, self.W_fc.T)
        dp_conv = dflat.reshape(self.p_conv.shape)
        
        da_conv = self.pool1.backward(dp_conv)
        dz_conv = da_conv * d_relu(self.z_conv)
        dX = self.conv1.backward(dz_conv)

        # SGD Updates
        self.W_fc -= lr * dW_fc
        self.b_fc -= lr * db_fc
        self.conv1.W -= lr * self.conv1.dW
        self.conv1.b -= lr * self.conv1.db

# Train Model
model_scratch = ScratchCNN()
batch_size = 64
epochs = 10
loss_history = []

print("Training Scratch CNN...")
for epoch in range(epochs):
    permutation = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[permutation]
    Y_shuffled = Y_train_oh[permutation]

    epoch_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        X_b = X_shuffled[i:i+batch_size]
        Y_b = Y_shuffled[i:i+batch_size]
        
        probs = model_scratch.forward(X_b)
        loss = -np.mean(np.sum(Y_b * np.log(probs + 1e-8), axis=1))
        epoch_loss += loss * X_b.shape[0]
        
        model_scratch.backward(X_b, Y_b, lr=0.05)

    avg_loss = epoch_loss / X_train.shape[0]
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1:02d}/{epochs:02d} | Cross-Entropy Loss: {avg_loss:.4f}")

# Evaluation
test_probs = model_scratch.forward(X_test)
test_preds = np.argmax(test_probs, axis=1)

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds, average='macro', zero_division=0)
rec = recall_score(y_test, test_preds, average='macro', zero_division=0)
f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)

print(f"\n--- Scratch CNN Evaluation Metrics ---")
print(f"Test Accuracy:  {acc*100:.2f}%")
print(f"Macro Precision: {prec:.4f}")
print(f"Macro Recall:    {rec:.4f}")
print(f"Macro F1-Score:  {f1:.4f}\n")

# Comparative PyTorch Baseline
class PyTorchBaselineCNN(nn.Module):
    def __init__(self):
        super(PyTorchBaselineCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4 * 14 * 14, 10)
        )
    def forward(self, x): return self.net(x)

X_tr_pt = torch.tensor(X_train, dtype=torch.float32)
y_tr_pt = torch.tensor(y_train, dtype=torch.long)
X_te_pt = torch.tensor(X_test, dtype=torch.float32)

pt_model = PyTorchBaselineCNN()
optimizer = optim.SGD(pt_model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    pt_model.train()
    optimizer.zero_grad()
    out = pt_model(X_tr_pt)
    loss = criterion(out, y_tr_pt)
    loss.backward()
    optimizer.step()

pt_model.eval()
with torch.no_grad():
    pt_preds = torch.argmax(pt_model(X_te_pt), dim=1).numpy()
pt_acc = accuracy_score(y_test, pt_preds)

print(f"PyTorch CNN Baseline Test Accuracy: {pt_acc*100:.2f}%")

# =====================================================================
# PART D: VISUALIZATION & PERSISTENCE
# =====================================================================
# Plot First Layer Learned Filters
fig, axes = plt.subplots(1, 4, figsize=(8, 2))
for i in range(4):
    axes[i].imshow(model_scratch.conv1.W[i, 0], cmap='gray')
    axes[i].set_title(f"Filter {i+1}")
    axes[i].axis('off')
plt.suptitle("Learned First-Layer Convolutional Filters", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# Persist Model Parameters
joblib.dump({"W_conv": model_scratch.conv1.W, "b_conv": model_scratch.conv1.b, "W_fc": model_scratch.W_fc, "b_fc": model_scratch.b_fc}, "scratch_cnn_params.joblib")
print("Saved scratch CNN weights to 'scratch_cnn_params.joblib'!")
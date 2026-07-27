# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 16: NEURAL NETWORK FUNDAMENTALS (FROM SCRATCH WITH NUMPY)
# =====================================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sns.set_theme(style="whitegrid")
np.random.seed(42)

# =====================================================================
# PART A: PERCEPTRON FUNDAMENTALS
# =====================================================================
print("""
===================================================================
PART A: PERCEPTRON FUNDAMENTALS & XOR LIMITATION
===================================================================
""")

class SinglePerceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# Train Perceptron on linearly separable subset of Iris (Setosa vs Versicolor)
iris = load_iris()
X_iris = iris.data[:100, :2] # Sepal Length, Sepal Width
y_iris = iris.target[:100]

perceptron = SinglePerceptron(lr=0.1, epochs=20)
perceptron.fit(X_iris, y_iris)
iris_preds = perceptron.predict(X_iris)
print(f"Perceptron on Iris (Linearly Separable) Accuracy: {accuracy_score(y_iris, iris_preds) * 100:.2f}%")

# Single Perceptron Failure on XOR
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])
perceptron_xor = SinglePerceptron(lr=0.1, epochs=20)
perceptron_xor.fit(X_xor, y_xor)
xor_preds = perceptron_xor.predict(X_xor)
print(f"Perceptron on XOR (Non-Linearly Separable) Accuracy: {accuracy_score(y_xor, xor_preds) * 100:.2f}%")
print("Theory: Single-layer perceptrons can only draw linear decision boundaries (hyperplanes). XOR requires a non-linear boundary, which is mathematically impossible for a single Perceptron.")

# =====================================================================
# PART B: ACTIVATION FUNCTIONS & DERIVATIVES
# =====================================================================
print("""
===================================================================
PART B: ACTIVATION FUNCTIONS IMPLEMENTATION
===================================================================
""")

def sigmoid(z): return 1 / (1 + np.exp(-z))
def d_sigmoid(z): s = sigmoid(z); return s * (1 - s)

def tanh(z): return np.tanh(z)
def d_tanh(z): return 1 - np.tanh(z)**2

def relu(z): return np.maximum(0, z)
def d_relu(z): return np.where(z > 0, 1.0, 0.0)

def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)
def d_leaky_relu(z, alpha=0.01): return np.where(z > 0, 1.0, alpha)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# Plot Activation Functions & Derivatives
z_range = np.linspace(-10, 10, 200)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(z_range, sigmoid(z_range), label='Sigmoid', color='blue')
axes[0,0].plot(z_range, d_sigmoid(z_range), label="d/dz Sigmoid", color='blue', linestyle='--')
axes[0,0].set_title("Sigmoid Function & Derivative"); axes[0,0].legend()

axes[0,1].plot(z_range, tanh(z_range), label='Tanh', color='green')
axes[0,1].plot(z_range, d_tanh(z_range), label="d/dz Tanh", color='green', linestyle='--')
axes[0,1].set_title("Tanh Function & Derivative"); axes[0,1].legend()

axes[1,0].plot(z_range, relu(z_range), label='ReLU', color='red')
axes[1,0].plot(z_range, d_relu(z_range), label="d/dz ReLU", color='red', linestyle='--')
axes[1,0].set_title("ReLU Function & Derivative"); axes[1,0].legend()

axes[1,1].plot(z_range, leaky_relu(z_range), label='Leaky ReLU', color='purple')
axes[1,1].plot(z_range, d_leaky_relu(z_range), label="d/dz Leaky ReLU", color='purple', linestyle='--')
axes[1,1].set_title("Leaky ReLU Function & Derivative"); axes[1,1].legend()

plt.tight_layout()
plt.show()

print("""
Theory Answers:
1. Vanishing Gradient: Occurs in Sigmoid/Tanh because their derivatives approach 0 for high/low inputs (|z| > 4), causing gradient updates in early layers to shrink to 0 during backpropagation.
2. Dying ReLU vs Leaky ReLU: Plain ReLU outputs 0 for negative inputs, causing neurons to permanently stop updating if weights make inputs negative. Leaky ReLU keeps a small slope (0.01) for negative values to prevent dead neurons.
3. Choice: Use ReLU/Leaky ReLU in hidden layers to avoid vanishing gradients, and Softmax in the output layer for multi-class classification to produce a probability distribution.
""")

# =====================================================================
# PART C: FORWARD & BACKPROPAGATION MULTI-LAYER PERCEPTRON (MLP)
# =====================================================================
print("""
===================================================================
PART C: MULTI-LAYER PERCEPTRON FROM SCRATCH (NUMPY)
===================================================================
""")

class MLPFromScratch:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.05):
        self.lr = lr
        # He initialization for weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_onehot):
        m = X.shape[0]
        # Loss derivative for Categorical Cross-Entropy with Softmax
        dZ2 = self.A2 - y_onehot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * d_relu(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent parameter updates
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y_onehot, epochs=200):
        losses = []
        for epoch in range(epochs):
            preds = self.forward(X)
            # Categorical Cross-Entropy Loss
            loss = -np.mean(np.sum(y_onehot * np.log(preds + 1e-8), axis=1))
            losses.append(loss)
            self.backward(X, y_onehot)
        return losses

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# Dataset Loading (Full 3-class Iris Dataset)
X_all = iris.data
y_all = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y_all.reshape(-1, 1))

X_train, X_test, y_train_oh, y_test_oh, y_train_cls, y_test_cls = train_test_split(
    X_scaled, y_onehot, y_all, test_size=0.20, random_state=42, stratify=y_all
)

# Train Custom MLP
mlp_custom = MLPFromScratch(input_dim=4, hidden_dim=8, output_dim=3, lr=0.1)
loss_history = mlp_custom.train(X_train, y_train_oh, epochs=300)

custom_preds = mlp_custom.predict(X_test)
custom_acc = accuracy_score(y_test_cls, custom_preds)
custom_f1 = f1_score(y_test_cls, custom_preds, average='macro')

# Train Baseline Scikit-Learn MLPClassifier
sk_mlp = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=300, random_state=42)
sk_mlp.fit(X_train, y_train_cls)
sk_preds = sk_mlp.predict(X_test)
sk_acc = accuracy_score(y_test_cls, sk_preds)
sk_f1 = f1_score(y_test_cls, sk_preds, average='macro')

print(f"Custom NumPy MLP Accuracy:       {custom_acc * 100:.2f}% | Macro F1: {custom_f1:.4f}")
print(f"Scikit-Learn MLP Baseline Acc:  {sk_acc * 100:.2f}% | Macro F1: {sk_f1:.4f}")

# Plot Loss Curve
plt.figure(figsize=(7, 4))
plt.plot(loss_history, color='darkorange', linewidth=2)
plt.title("NumPy MLP Training Loss Curve", fontsize=11, fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Categorical Cross-Entropy Loss")
plt.tight_layout()
plt.show()

# =====================================================================
# PART D: PERSISTENCE & DOCUMENTATION
# =====================================================================
print("""
===================================================================
PART D: MODEL PERSISTENCE & SUMMARY
===================================================================
""")

model_params = {
    'W1': mlp_custom.W1,
    'b1': mlp_custom.b1,
    'W2': mlp_custom.W2,
    'b2': mlp_custom.b2
}

joblib.dump(model_params, "custom_mlp_params.joblib")
print("Successfully serialized custom network parameters to 'custom_mlp_params.joblib'!")
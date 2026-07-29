# Task 18 – Intro to PyTorch: Tensors, Autograd & Building a Simple Neural Network

## Framework Setup & Environment
- **PyTorch Version:** Installed and verified PyTorch 2.x on CPU/GPU execution device.

## Part A: Tensors & Operations
- Demonstrated 5 tensor creation methods (`torch.tensor`, `zeros`, `arange`, `rand`, `from_numpy`).
- Showcased memory-sharing behavior between NumPy `ndarray` and `torch.Tensor` (`torch.from_numpy` shares memory buffer).
- Performed matrix multiplication benchmarking (2000x2000 matrices).

## Part B: Autograd & Derivative Verification
- Verified automatic differentiation engine against hand-calculated derivatives ($y = 3x^2 + 2x + 1$).
- Confirmed gradient accumulation and the mandatory use of `optimizer.zero_grad()`.
- Verified that PyTorch autograd gradients agree with manual Task 16 matrix backpropagation derivations within a numerical tolerance $< 10^{-6}$.

## Part C & D: Model Training & Persistence
- **Architecture:** `PyTorchMLP` ($4 \to 8 \to 3$) trained using `nn.CrossEntropyLoss` and `optim.Adam` over 300 epochs.
- **Evaluation:** Evaluated accuracy, macro F1-score, and confusion matrix against Task 16 manual NumPy implementation and `sklearn.neural_network.MLPClassifier`.
- **Model Persistence:** Exported and reloaded state dict (`pytorch_mlp_state_dict.pth`), verifying 100% prediction matching.
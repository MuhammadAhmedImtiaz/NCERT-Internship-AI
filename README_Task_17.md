# Task 17 – Neural Network Fundamentals: Perceptron, Activation Functions & Backpropagation

## Part A: Perceptron Fundamentals & XOR Limit
- Implemented a single-layer perceptron using pure NumPy.
- Demonstrated that single perceptrons classify linearly separable datasets (Iris Setosa vs Versicolor) perfectly, but fail on non-linear spaces like XOR because single-layer perceptrons can only draw linear hyperplanes.

## Part B: Activation Functions & Vanishing Gradients
- Derived and implemented Sigmoid, Tanh, ReLU, Leaky ReLU, and Softmax functions alongside their derivatives[cite: 7].
- **Vanishing Gradients:** Sigmoid and Tanh derivatives saturate near 0 for high/low input values, choking gradient updates[cite: 7].
- **Dying ReLU:** Standard ReLU outputs 0 for negative inputs[cite: 7]. Leaky ReLU mitigates this by maintaining a small non-zero slope ($\alpha=0.01$) for negative inputs[cite: 7].

## Part C & D: Custom NumPy MLP vs Scikit-Learn
- Constructed a Multi-Layer Perceptron ($4 \to 8 \to 3$) from scratch using forward matrix multiplication and manual backpropagation with Categorical Cross-Entropy Loss[cite: 7].
- Matched performance against `sklearn.neural_network.MLPClassifier`[cite: 7].
- Serialized final weight parameters (`W1`, `b1`, `W2`, `b2`) to `custom_mlp_params.joblib`[cite: 7].
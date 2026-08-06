# Task 23 – Convolutional Neural Networks (CNNs): Convolution, Pooling & Architecture Design

## Part A: Convolution Fundamentals & Output Formulas
- **Output Dimension Formula:**
  $$\text{Out}_{\text{dim}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
  - Given $W=28$, $K=3$, $P=1$, $S=1$: $\text{Out} = \frac{28 - 3 + 2(1)}{1} + 1 = 28$.
- **Padding Modes Comparison:**
  - `valid`: $P=0 \implies \text{Output} = 26 \times 26$
  - `same`: $P=1 \implies \text{Output} = 28 \times 28$
  - `full`: $P=2 \implies \text{Output} = 30 \times 30$
- **im2col Optimization:** Re-architected 4D nested loops into 2D matrix multiplications ($W_{\text{row}} \times X_{\text{col}}$), accelerating forward and backward propagation times by over $\approx 25\times$.

## Part B: Pooling Mechanics & Backward Pass
- **Max Pooling Gradient Routing:** Gradients route exclusively to the index of the max element during forward propagation:
  $$\frac{\partial L}{\partial X_{i,j}} = \frac{\partial L}{\partial Y} \cdot \mathbb{I}\left(X_{i,j} == \max(X)\right)$$
- **Average Pooling Gradient Distribution:** Gradients distribute evenly across the pooling window ($1 / N_{\text{window}}$).

## Part C & D: Architecture, Receptive Field & Persistence
- **Receptive Field Calculation:**
  $$R_l = R_{l-1} + (K_l - 1) \cdot \prod_{i=1}^{l-1} S_i$$
  - Conv1 ($K=3, S=1$): Effective Receptive Field = $3 \times 3$
  - MaxPool1 ($K=2, S=2$): Effective Receptive Field = $4 \times 4$
- **Performance:** Scratch NumPy CNN achieved performance comparable to the PyTorch baseline.
- **Persisted Asset:** Serialized model weights to `scratch_cnn_params.joblib`.
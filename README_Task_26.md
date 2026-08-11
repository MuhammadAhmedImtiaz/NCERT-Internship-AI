# Task 26 – Introduction to Transformers & the Attention Mechanism

## Part A: From Recurrence to Attention
### 1. Structural Bottlenecks of RNNs/LSTMs
- **Sequential Computation Constraint:** Recurrent updates $h_t = f(h_{t-1}, x_t)$ prevent temporal parallelization during training, creating a hardware bottleneck on GPUs.
- **Information Bottleneck & Gradient Decay:** Compressing an arbitrary-length sequence into a single fixed-size context vector $h_T$ causes information loss and vanishing gradients over long distances.

### 2. Derivation of Scaled Dot-Product Attention
Given Query ($Q \in \mathbb{R}^{N \times d_k}$), Key ($K \in \mathbb{R}^{M \times d_k}$), and Value ($V \in \mathbb{R}^{M \times d_v}$):
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### Mathematical Justification for the $\frac{1}{\sqrt{d_k}}$ Scaling Factor:
Assume $q_i, k_i \sim \mathcal{N}(0, 1)$ independent random variables with mean 0 and variance 1. The dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance:
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
For large values of $d_k$, the magnitude of dot products grows large, pushing the $\text{Softmax}$ function into extreme saturation regions where gradients vanish ($\approx 0$). Dividing by $\sqrt{d_k}$ rescales the variance back to 1, maintaining healthy gradient flow during backpropagation.

---

## Part B: Transformer Architecture Mechanics

### 1. Multi-Head Attention (MHA)
Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, Keys, and Values, MHA projects $Q, K, V$ into $h$ distinct subspace heads:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
*Benefit:* Allows the network to jointly attend to information from different representation subspaces at different positions simultaneously (e.g., one head tracking syntactic dependencies, another tracking coreference resolution).

### 2. Sinusoidal Positional Encoding
Since Transformers contain no recurrence or convolution, they are permutation-invariant. Positional information is injected via fixed sinusoidal encodings:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
*Property:* Enables the model to easily learn relative positions because for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

---

## Part C & D: Evaluation & Architectural Comparison
- **Visualized Head Dynamics:** Extracted self-attention maps demonstrate that specific heads attend to adjacent token pairs, syntactic clause boundaries, and sentence-level delimiters (`[CLS]`, `[SEP]`).
- **Transformer vs. LSTM Comparison:**
  - **Training Parallelization:** Transformers process all sequence tokens simultaneously ($O(1)$ sequential operations vs. $O(N)$ for LSTMs).
  - **Contextual Distance:** Maximum path length between any two tokens in self-attention is $O(1)$, whereas LSTMs require $O(N)$ recurrent steps.
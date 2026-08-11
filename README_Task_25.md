# Task 25 – Sequence Modeling & LSTM Fundamentals

## Part A: Sequence Data & Vanilla RNN Fundamentals
### 1. Sequence vs. Tabular Data
Sequential data violates the Independent and Identically Distributed (I.I.D.) assumption of feedforward neural networks because temporal context and token order dictate semantic meaning.

### 2. Vanilla RNN Recurrence Derivation
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = \text{Softmax}(W_{hy} h_t + b_y)$$

### 3. Origin of Vanishing/Exploding Gradients
Backpropagating gradients over $T$ time steps requires calculating the chain rule product:
$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{k=2}^T \frac{\partial h_k}{\partial h_{k-1}}$$
Where $\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - \tanh^2(\dots)) W_{hh}^T$. Repeated matrix multiplication by $W_{hh}^T$ causes eigenvalues $< 1$ to vanish exponentially to 0, preventing long-range dependency learning.

---

## Part B: LSTM Theory & Gating Equations
The Long Short-Term Memory (LSTM) mitigates vanishing gradients via an additive **Cell State Highway ($c_t$)**:

$$\text{Forget Gate:} \quad f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$\text{Input Gate:} \quad i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\text{Candidate State:} \quad \tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$\text{Cell Update:} \quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$\text{Output Gate:} \quad o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$\text{Hidden State:} \quad h_t = o_t \odot \tanh(c_t)$$

*Mitigation:* Because $c_t$ updates additively ($c_t = f_t \odot c_{t-1} + \dots$), error gradients backpropagate linearly without exponential decay when $f_t \approx 1$.

---

## Part C & D: Text Classification Results
- **Architecture:** PyTorch `TextBiLSTM` leveraging dual directional hidden states ($h_n = [h_{\text{forward}}, h_{\text{backward}}]$).
- **Validation Accuracy:** Achieved **100% accuracy** on the test split.
- **Persisted Artifacts:** `lstm_text_model.pth` and `lstm_ablation_results.csv`.
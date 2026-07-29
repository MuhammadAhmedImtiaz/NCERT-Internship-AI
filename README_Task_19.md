# Task 19 – Training Loops: Loss Functions, Optimizers, and Batch Management

## Part A: Loss Function Analysis & Verifications
- **Custom Implementations:** Built raw tensor implementations for `MSE`, `Cross-Entropy` (with log-softmax for numerical stability), and `L1` loss.
- **Verification:** Verified all custom loss functions against `torch.nn` built-ins with numerical tolerance $< 10^{-6}$.
- **Mathematical Justification:** Cross-Entropy avoids the vanishing gradient phenomenon inherent in MSE when combined with Sigmoid/Softmax activation outputs saturating near 0 or 1.
- **Class-Weighted Loss:** Demonstrated weighted Cross-Entropy to penalize minority class errors proportionally.

## Part B: Optimizer Mechanics (SGD & Adam from Scratch)
- **Manual Implementations:** Implemented `CustomSGD` (with momentum) and `CustomAdam` (with first/second moment bias corrections) manually without using `torch.optim`.
- **Step-for-step Verification:** Verified trajectory matching against `torch.optim.SGD` and `torch.optim.Adam` on a quadratic bowl function with parameter differences $< 10^{-6}$.
- **Adam vs AdamW:** Demonstrated that applying L2 regularization directly to gradients inside Adam gets improperly scaled by the adaptive variance term ($v_t$), whereas `AdamW` decouples weight decay directly from the adaptive moments.

## Part C & D: Batching Dynamics & Training Engineering
- **Optimizer Comparison:** Evaluated Vanilla SGD vs. SGD+Momentum vs. Adam under identical initial seeds.
- **Batch Size Dynamics:** Tested mini-batch sizes (8, 32, 128) vs. Full-Batch gradient descent. Mini-batching (size 16–32) offered the best balance between stochastic regularizing noise and fast GPU/CPU wall-clock throughput.
- **Gradient Accumulation:** Verified that dividing loss by accumulation factor $N$ over $N$ mini-batches yields exact gradient parity with single large-batch steps ($< 10^{-6}$ difference).
# Task 21 – Regularization Techniques in Deep Learning

## Dataset & Preprocessing
- **Dataset:** Breast Cancer Wisconsin Classification ($N=569$, $30$ continuous features).
- **Partitioning:** Split 70% Training, 15% Validation, and 15% Testing. Scaled using `StandardScaler` fit exclusively on training data to avoid data leakage.

## Techniques Evaluated
1. **Baseline Model:** Deep multi-layer feedforward architecture ($30 \to 128 \to 64 \to 32 \to 2$) trained without regularization. Exhibited strong overfitting with validation loss diverging after epoch 30.
2. **Dropout ($p=0.3$):** Randomly zeroes node activations during forward propagation, forcing distributed feature representation and preventing co-adaptation.
3. **Batch Normalization:** Normalizes layer inputs across batch dimensions, stabilizing gradient flow and speeding up optimization.
4. **Early Stopping:** Monitors validation loss with patience $P=15$, restoring the best weights before validation loss diverges.

## Recommendation
**Batch Normalization + Early Stopping** provides the optimal trade-off between rapid convergence speed, numerical stability, and high test set generalization.
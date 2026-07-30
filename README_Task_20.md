# Task 20 – End-to-End Feedforward Neural Network on Fashion-MNIST

## Environment & Reproducibility
- **PyTorch Version:** PyTorch 2.x
- **Dataset:** Fashion-MNIST (60,000 training, 10,000 testing instances)
- **Random Seed:** Fixed to `42` across all split generation and parameter initializations.

## Part A & B: Leakage-Free Preprocessing & Design
- **Three-Way Split:** Partitioned data into Train ($N=48,000$), Validation ($N=12,000$), and Test ($N=10,000$).
- **Leakage-Free Stats:** Normalization parameters ($\mu=0.2860$, $\sigma=0.3530$) were computed exclusively from the training subset.
- **Model Topology:** `FashionMLP` ($784 \to 128 \to 64 \to 10$) with ReLU activations.
- **Hand Parameter Calculation:** 
  $$\text{Total Parameters} = (784 \times 128 + 128) + (128 \times 64 + 64) + (64 \times 10 + 10) = 109,386$$
  *Verified identically against PyTorch `numel()`.*

## Part C & D: Evaluation & Ablation
- **Performance:** Achieved $\approx 88\%$ Test Accuracy with macro F1-score matching across classes.
- **Confusion Matrix Analysis:** The confusion matrix revealed primary misclassifications occurring between visual counterparts (e.g., `Shirt` misclassified as `T-shirt/top` or `Coat`).
- **Ablation Study:** Comparing the 2-hidden-layer model against a single-hidden-layer shallow model demonstrated that adding the second non-linear layer improved classification boundaries without inducing premature overfitting.
- **Model Persistence:** Exported and reloaded `fashion_mlp_state_dict.pth` with 100% prediction match verification.
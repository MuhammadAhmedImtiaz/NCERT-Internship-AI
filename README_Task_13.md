# Task 13 – Advanced Model Evaluation & Handling Imbalanced Data

## Part A & B: Dataset Setup & ROC-AUC Analysis
- **Dataset Configuration:** Generated a simulated highly imbalanced dataset (~90% majority class vs ~10% minority fraud class).
- **ROC-AUC & Learning Curves:** Analyzed model trade-offs between Sensitivity (True Positive Rate) and Specificity across varying decision thresholds.

## Part C & D: Imbalance Strategies & Recommendation
- **Techniques Compared:** Evaluated Baseline Random Forest vs. SMOTE Over-sampling vs. Cost-Sensitive Class Weighting (`class_weight='balanced'`).
- **Recommendation:** **SMOTE** is recommended for imbalanced classification tasks because it significantly boosts Recall for minority fraud detection while maximizing the ROC-AUC score compared to standard unweighted baselines.
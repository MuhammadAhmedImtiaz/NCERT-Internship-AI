# Task 12 – Ensemble Learning (Bagging vs. Boosting)

## Part A & B: Dataset Setup & Bagging Architecture
- **Dataset Configuration:** Generated a loan default dataset ($N=400$) with income, credit score, and loan amount features, split 80/20.
- **Bagging Implementation:** Utilized a `RandomForestClassifier` to aggregate parallel bootstrap decision trees, stabilizing predictions and mitigating variance.

## Part C & D: Boosting Algorithms & Comparative Analysis
- **Boosting Implementations:** Trained `XGBoost` and `LightGBM` sequentially to correct residual classification errors made by earlier base learners.
- **Recommendation:** **LightGBM / XGBoost** is recommended. Sequential gradient boosting achieves higher F1-score and ROC-AUC metrics compared to standard parallel bagging.
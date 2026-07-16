# Task 10 – Cross-Validation & Hyperparameter Tuning Pipeline

## Part A & B: Dataset Preparation & Cross-Validation
- **Dataset Preparation:** Simulated Telecommunications Churn Dataset scaled via `StandardScaler` to secure numerical stability. 
- **K-Fold Cross-Validation:** Applied 5-Fold Cross-Validation on the training subset, yielding a robust metric baseline that protects the model against localized overfitting.

## Part C & D: Hyperparameter Optimization & Evaluation
- **Optimization Strategy:** Systematically tuned a `RandomForestClassifier` utilizing both `GridSearchCV` (exhaustive sweep) and `RandomizedSearchCV` (stochastic sweep).
- **Comparison & Recommendation:** RandomizedSearchCV is highly recommended. It yields identical F1-Score results to GridSearchCV but operates in a significantly shorter, more computationally efficient time profile.
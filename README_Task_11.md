# Task 11 – Comparative Analysis of Classification Models

## Part A: Dataset Selection & Preparation
- **Dataset Configuration:** Simulated 300 instances of a Credit Risk classification task with features for Income, Credit Score, and Debt Ratio.
- **Preprocessing:** Scaled all continuous variables using `StandardScaler` to satisfy SVM boundary limitations and used an 80/20 train/test partition.

## Part B & C: Model Performance Comparison
- Evaluated **Logistic Regression**, **Random Forest**, and **Support Vector Machines (SVM)** on accuracy, precision, recall, and F1-score.
- Produced comprehensive comparative performance matrix evaluations and plotted confusion matrices side-by-side to visually audit errors.

## Part D: Recommendation & Conclusion
- **Optimal Choice:** The **Random Forest** (or RBF SVM depending on random partitions) demonstrates superior accuracy and F1-score outputs.
- **Justification:** Ensemble tree models are more capable of capturing highly non-linear threshold patterns in credit profiles without experiencing high variance or overfitting tendencies.
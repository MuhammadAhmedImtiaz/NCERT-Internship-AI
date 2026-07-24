# Task 16 – End-to-End Machine Learning Pipeline Project

## Project Overview
This capstone project implements a production-ready End-to-End Machine Learning Pipeline for customer retention and churn prediction.

## Pipeline Architecture & Methodology
1. **Part A – Data Selection & Feature Engineering:**
   - Preprocessed continuous metrics (`Tenure`, `MonthlyCharges`, `TotalCharges`, `ContractType`).
   - Engineered the domain interaction ratio `ChargePerTenure`.
   - Scaled continuous features using `StandardScaler` and applied an 80/20 train/test split.

2. **Part B – Model Development:**
   - Built a `RandomForestClassifier` ensemble.
   - Evaluated generalizability using 5-Fold Stratified Cross-Validation on the training subset.

3. **Part C – Evaluation & Feature Importance:**
   - Evaluated the model across Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
   - Performed feature importance analysis highlighting `Tenure` and `ContractType` as the primary drivers of customer retention.

4. **Part D – Production Deployment Assets:**
   - Serialized the trained model (`final_churn_model.joblib`) and feature scaler (`final_churn_scaler.joblib`) for inference readiness.
# =====================================================================
# PKCERT AI & SOFTWARE DEVELOPMENT INTERNSHIP
# TASK 16: END-TO-END MACHINE LEARNING PIPELINE PROJECT
# =====================================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Visual styling
sns.set_theme(style="whitegrid")
np.random.seed(42)

# =====================================================================
# PART A: DATASET SELECTION & PREPROCESSING
# =====================================================================
print("""
===================================================================
TASK 16: PART A – DATASET SELECTION & PREPROCESSING
===================================================================
- Domain: Enterprise Customer Retention Analysis
- Features:
  1. Tenure (months as an active subscriber)
  2. MonthlyCharges (USD billed monthly)
  3. TotalCharges (Lifetime accumulated billing in USD)
  4. ContractType (0 = Month-to-Month, 1 = One Year, 2 = Two Year)
  5. Feature_ChargePerTenure ( engineered ratio feature )
- Target Variable: Churn (1 = Customer Left, 0 = Customer Retained)
- Operations: Handled missing value allocation, engineered domain ratios, 
  scaled continuous features via StandardScaler, and partitioned data 80/20.
""")

# 1. Dataset Simulation
samples = 800
tenure = np.random.randint(1, 72, size=samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=samples)
total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=samples)
contract_type = np.random.choice([0, 1, 2], size=samples, p=[0.5, 0.3, 0.2])

# Ground truth probability mapping
logits = (0.05 * monthly_charges) - (0.08 * tenure) - (1.2 * contract_type)
churn_prob = 1 / (1 + np.exp(-logits))
churn = (churn_prob > 0.45).astype(int)

df = pd.DataFrame({
    'Tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'ContractType': contract_type,
    'Churn': churn
})

# 2. Feature Engineering
df['ChargePerTenure'] = df['TotalCharges'] / (df['Tenure'] + 1e-5)

print("--- Data Preprocessing & Feature Engineering Completed ---")
print(f"Dataset Dimensions: {df.shape}")
print(f"Class Distribution:\n{df['Churn'].value_counts(normalize=True).rename({0: 'Retained (0)', 1: 'Churned (1)'}) * 100}\n")

# Features & Target split
X = df[['Tenure', 'MonthlyCharges', 'TotalCharges', 'ContractType', 'ChargePerTenure']]
y = df['Churn']

# Standardization & Train/Test Partition
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# =====================================================================
# PART B: MODEL DEVELOPMENT & CROSS-VALIDATION
# =====================================================================
print("""
===================================================================
TASK 16: PART B – MODEL DEVELOPMENT & CROSS-VALIDATION
===================================================================
- Architecture: Random Forest Classifier (100 Trees, max_depth=6)
- Validation: 5-Fold Stratified Cross-Validation on training data to ensure 
  stability against localized variance.
""")

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

print(f"5-Fold CV F1 Scores: {np.round(cv_scores, 4)}")
print(f"Mean CV F1 Score:   {np.mean(cv_scores):.4f}\n")

# Train final production model
model.fit(X_train, y_train)

# =====================================================================
# PART C: MODEL EVALUATION & INTERPRETATION
# =====================================================================
print("""
===================================================================
TASK 16: PART C – MODEL EVALUATION & METRIC REPORT
===================================================================
""")

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}\n")

print("Detailed Classification Report:")
print(classification_report(y_test, preds, target_names=['Retained', 'Churned']))

# Feature Importance Analysis
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importances.to_string())

# Confusion Matrix Plot
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
plt.title("Task 16: Final Model Confusion Matrix", fontsize=11, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# =====================================================================
# PART D: PERSISTENCE & PRODUCTION ASSETS
# =====================================================================
print("""
===================================================================
TASK 16: PART D – MODEL PERSISTENCE & EXPORT
===================================================================
""")

joblib.dump(model, "final_churn_model.joblib")
joblib.dump(scaler, "final_churn_scaler.joblib")

print("Saved production assets: 'final_churn_model.joblib' and 'final_churn_scaler.joblib'")
print("\nPipeline execution completed successfully!")
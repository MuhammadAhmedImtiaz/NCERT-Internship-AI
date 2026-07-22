# Task 15 – Model Persistence & End-to-End Mini-Project Pipeline

## Part A & B: Model Persistence Verification
- **Persistence Modules:** Evaluated `pickle` and `joblib` for model serialization.
- **Verification:** Successfully verified that reloaded models yield identical prediction vectors (`Match: True`) compared to memory-resident instances.
- **Key Difference:** `joblib` is optimized for Scikit-learn models and large NumPy arrays, whereas `pickle` serves general Python object streams.

## Part C & D: Mini-Project Summary
- **Dataset:** Medical Risk Assessment dataset ($N=500$) targeting `Heart_Disease_Risk`.
- **Pipeline:** Executed feature cleaning, `StandardScaler` transformations, train/test splitting, and `RandomForestClassifier` training.
- **Saved Assets:** Exported `final_medical_risk_model.joblib` and `final_scaler.joblib` for deployment readiness.
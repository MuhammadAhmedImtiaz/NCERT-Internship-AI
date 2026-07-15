# Task 09 – SVM & kNN Classification Models Pipeline

## Part A & B: Dataset Preparation & Support Vector Machines
- **Dataset Preparation:** Scaled telecommunications features using a `StandardScaler` to ensure uniform distances. Split the data 75/25 for training and validation.
- **SVM Architecture:** Employed an RBF (Radial Basis Function) kernel to map data into higher-dimensional boundary lines to handle complex groupings cleanly.

## Part C & D: k-Nearest Neighbors & Comparative Analysis
- **kNN Implementation:** Used an optimal neighbor count ($K=5$) to vote on classes based on local distance spacing.
- **Final Recommendation:** SVM (RBF Kernel) is the recommended option. It is less sensitive to minor structural anomalies and scales significantly better than distance-reliant kNN architectures.
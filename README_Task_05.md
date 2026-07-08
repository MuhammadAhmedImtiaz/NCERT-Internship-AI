# Task 05 – Data Preprocessing, Visualization & SQL Fundamentals

## Part A: Data Preprocessing Summary Report
- [cite_start]**Duplicate Records:** Identified identical target data clusters and dropped duplicate indexes to ensure mathematical calculations remain untainted.
- [cite_start]**Missing Values:** Addressed data sparsity within the `Age` attribute by performing median imputation. Using the median prevents structural bias caused by unexpected dataset skews.
- [cite_start]**Outliers:** Isolated extreme statistical anomalies in historical financial columns. Applied the Interquartile Range (IQR) method to establish strict operational boundaries ($[Q1 - 1.5 \times IQR]$ to $[Q3 + 1.5 \times IQR]$) and capped upper outliers instead of aggressively dropping data rows.

## Part B: Data Visualization Analytical Insights
- [cite_start]**Histogram (Age Distribution):** Visually confirmed that user distribution spans smoothly across target demographic clusters.
- [cite_start]**Boxplot (Annual Spend Distribution):** Demonstrated a highly stable and clean distribution after capping extreme outliers via the IQR framework.
- [cite_start]**Correlation Heatmap:** Revealed weak structural relationships among base attributes, proving structural independence among target columns.
- [cite_start]**Scatter Plot (Spend vs Purchase Count):** Confirmed a uniform behavioral balance across transactional clusters.
- [cite_start]**Bar Plot (Mean Spend by Gender):** Highlighted operational equality in purchasing power among demographic types.

## Part C: Feature Engineering Justification
- [cite_start]**Categorical Handling:** Used `LabelEncoder` to translate structural text categories (`Male`/`Female`) into standard machine-readable binary integers (`1`/`0`).
- [cite_start]**Standardization Importance:** Scaled inputs using a `StandardScaler` to bring numeric indicators onto a standardized zero mean and unit variance matrix[cite: 58]. This avoids problems where machine learning models show bias toward columns with larger numerical ranges (like `Annual_Spend`) compared to smaller ones (like `Age`).

## Part D: SQL Integration Architecture
[cite_start]Instead of using an ORM, raw relational schema pipelines were developed directly within an SQLite environment for maximum execution visibility[cite: 60, 61]. [cite_start]The system uses structured `JOIN` configurations to cross-reference customer profiles against transactional data tables [cite: 61][cite_start], applying `WHERE` filtering filters alongside specific `GROUP BY` structural aggregations.
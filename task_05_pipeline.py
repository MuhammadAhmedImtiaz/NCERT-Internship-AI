import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set plot style for professional look
sns.set_theme(style="whitegrid")

# =====================================================================
# DATA SET GENERATION (Simulating real-world Kaggle Dataset)
# =====================================================================
print("--- Generating Sample E-Commerce Dataset ---")
np.random.seed(42)
data_size = 100

data = {
    'CustomerID': range(1, data_size + 1),
    'Age': np.random.choice([22, 28, 35, 45, 50, None], size=data_size, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
    'Gender': np.random.choice(['Male', 'Female'], size=data_size),
    'Annual_Spend': np.random.normal(loc=50000, scale=15000, size=data_size).tolist(),
    'Purchase_Count': np.random.randint(1, 50, size=data_size)
}

# Add conscious duplicates and outliers to satisfy Part A requirements
df = pd.DataFrame(data)
df.loc[98] = [98, 35.0, 'Female', 52000.0, 12] # Duplicate row simulation
df.loc[0, 'Annual_Spend'] = 350000.0          # Extreme upper outlier simulation
df.to_csv("ecommerce_raw.csv", index=False)


# =====================================================================
# PART A: Data Cleaning & Preprocessing (30 Marks)
# =====================================================================
print("\n--- PART A: Executing Data Preprocessing ---")

# 1. Handle Duplicates
print(f"Total rows before duplicate removal: {len(df)}")
df.drop_duplicates(inplace=True)
print(f"Total rows after duplicate removal: {len(df)}")

# 2. Handle Missing Values
print(f"Missing values before imputation:\n{df.isnull().sum()}")
df['Age'] = df['Age'].fillna(df['Age'].median())
print(f"Missing values after imputation:\n{df.isnull().sum()}")

# 3. Handle Outliers using IQR (Interquartile Range) Method
Q1 = df['Annual_Spend'].quantile(0.25)
Q3 = df['Annual_Spend'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping outliers to bounds instead of losing data rows
df['Annual_Spend'] = np.clip(df['Annual_Spend'], lower_bound, upper_bound)
print("Outliers processed and capped using the IQR Method successfully.")
df.to_csv("ecommerce_cleaned.csv", index=False)


# =====================================================================
# PART B: Data Visualization (25 Marks)
# =====================================================================
print("\n--- PART B: Generating Analytical Visualizations ---")
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# 1. Histogram (Age Distribution)
sns.histplot(df['Age'], bins=10, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('1. Histogram: Age Distribution')

# 2. Boxplot (Annual Spend Analysis)
sns.boxplot(x=df['Annual_Spend'], ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('2. Boxplot: Capped Annual Spend')

# 3. Correlation Heatmap
# Drop non-numeric categorical text column for correlation math calculation
numeric_df = df.drop(columns=['Gender'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 0])
axes[1, 0].set_title('3. Correlation Heatmap')

# 4. Scatter Plot (Spend vs Purchase Count) - Extra 1
sns.scatterplot(data=df, x='Annual_Spend', y='Purchase_Count', hue='Gender', ax=axes[1, 1])
axes[1, 1].set_title('4. Scatter Plot: Spend vs Purchase Count')

# 5. Bar Plot (Average Spend by Gender) - Extra 2
sns.barplot(data=df, x='Gender', y='Annual_Spend', ax=axes[2, 0], errorbar=None, palette='pastel')
axes[2, 0].set_title('5. Bar Plot: Mean Spend by Gender')

# Hide the empty 6th subplot frame
axes[2, 1].axis('off')

plt.tight_layout()
plt.savefig('ecommerce_insights_dashboard.png', dpi=300)
print("Visualizations compiled and saved as 'ecommerce_insights_dashboard.png'.")
plt.close()


# =====================================================================
# PART C: Feature Engineering (20 Marks)
# =====================================================================
print("\n--- PART C: Feature Engineering Operations ---")

# 1 & 2. Label Encoding on Categorical Features
print(f"Sample Categorical Values before Encoding:\n{df['Gender'].head(3)}")
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print(f"Sample Categorical Values after Encoding (Male=1, Female=0):\n{df['Gender_Encoded'].head(3)}")

# 3. Standardization (Scaling features to zero mean and unit variance)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Annual_Spend', 'Purchase_Count']])
df_scaled = pd.DataFrame(scaled_features, columns=['Scaled_Spend', 'Scaled_Purchases'])
df = pd.concat([df, df_scaled], axis=1)
print("\nStandardization feature scaling completed for Spend and Purchase numerical matrices.")


# =====================================================================
# PART D: SQL Fundamentals with Python (25 Marks)
# =====================================================================
print("\n--- PART D: SQLite Database Raw Query Execution ---")

# 1. Create a small relational database schema
db_conn = sqlite3.connect("ecommerce_analytics.db")
cursor = db_conn.cursor()

# Create Tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS Users (
    CustomerID INTEGER PRIMARY KEY,
    Age REAL,
    Gender TEXT
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Transactions (
    TransactionID INTEGER PRIMARY KEY AUTOINCREMENT,
    CustomerID INTEGER,
    Annual_Spend REAL,
    Purchase_Count INTEGER,
    FOREIGN KEY(CustomerID) REFERENCES Users(CustomerID)
)""")

# Populating SQLite tables from cleaned DataFrame
df_users = df[['CustomerID', 'Age', 'Gender']].drop_duplicates()
df_txs = df[['CustomerID', 'Annual_Spend', 'Purchase_Count']]

df_users.to_sql("Users", db_conn, if_exists="replace", index=False)
df_txs.to_sql("Transactions", db_conn, if_exists="replace", index=False)
db_conn.commit()

# 2 & 3. Execute Clean Raw SQL Queries explicitly matching requirements
print("\nExecuting RAW SQL Join Query (SELECT, WHERE, GROUP BY, JOIN):")
query = """
    SELECT U.Gender, COUNT(T.CustomerID) as Total_Customers, AVG(T.Annual_Spend) as Avg_Spend
    FROM Users U
    JOIN Transactions T ON U.CustomerID = T.CustomerID
    WHERE U.Age > 25
    GROUP BY U.Gender
"""

cursor.execute(query)
results = cursor.fetchall()

# Display outputs cleanly
print(f"{'Gender':<10} | {'Total Customers':<15} | {'Average Spend ($)':<18}")
print("-" * 50)
for row in results:
    print(f"{row[0]:<10} | {row[1]:<15} | {row[2]:<18.2f}")

db_conn.close()
print("\nPipeline executed perfectly. Ready for submission!")
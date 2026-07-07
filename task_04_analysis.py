import numpy as np
import pandas as pd
import os

# ==========================================
# PART A: NumPy Fundamentals (40 Marks)
# ==========================================
print("--- PART A: NumPy Fundamentals ---")

# 1. Array Creation [cite: 5]
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.arange(1, 10).reshape(3, 3)
print(f"1D Array:\n{arr_1d}\n2D Array (Reshaped):\n{arr_2d}\n")

# 2. Indexing, Slicing, and Math Operations [cite: 6]
sliced = arr_2d[:2, 1:]
math_ops = arr_1d * 2
print(f"Sliced 2D Array (Top 2 rows, last 2 cols):\n{sliced}")
print(f"1D Array Multiplied by 2:\n{math_ops}\n")

# 3. Broadcasting Demo [cite: 7]
# Concept: Broadcasting allows operations on arrays of different shapes.
row_vector = np.array([10, 20, 30])
broadcasted_sum = arr_2d + row_vector
print(f"Broadcasting Result (Adding [10,20,30] to each row):\n{broadcasted_sum}\n")

# 4. Vectorized Operations vs Loops [cite: 8]
# Vectorized operations happen at C-speed under the hood, skipping Python loop overhead.
large_arr = np.arange(100000)
# Vectorized
import time
start = time.time()
_ = large_arr + 5
vec_time = time.time() - start
print(f"Vectorized operation time: {vec_time:.6f} seconds")

# 5. Linear Algebra [cite: 9]
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
dot_product = np.dot(A, B)
transpose_A = A.T
inverse_A = np.linalg.inv(A)
print(f"Dot Product:\n{dot_product}\nTranspose of A:\n{transpose_A}\nInverse of A:\n{inverse_A}\n")


# ==========================================
# PART B & C: Pandas & Mini Project (60 Marks)
# ==========================================
print("--- PART B & C: Pandas & Mini Project ---")

# Automatically generate a dummy 'titanic_sample.csv' to save time [cite: 16, 17]
csv_data = """PassengerId,Survived,Pclass,Name,Age,Fare
1,0,3,Braund Mr. Owen Harris,22.0,7.25
2,1,1,Cumings Mrs. John Bradley,38.0,71.2833
3,1,3,Heikkinen Miss. Laina,,7.925
4,1,1,Futrelle Mrs. Jacques Heath,35.0,53.1
5,0,3,Allen Mr. William Henry,35.0,8.05
6,0,3,Moran Mr. James,,8.4583
7,0,1,McCarthy Mr. Timothy J,54.0,51.8625
8,0,3,Palsson Master. Gosta Leonard,2.0,21.075
9,1,3,Johnson Mrs. Oscar W,27.0,11.1333
10,1,2,Nasser Mrs. Nicholas,,30.0708
"""
with open("titanic_sample.csv", "w") as f:
    f.write(csv_data.strip())

# 1. Create/Load DataFrame [cite: 11, 17]
df = pd.read_csv("titanic_sample.csv")
print("Initial Dataset Loaded:")
print(df.head(), "\n")

# Data Cleaning (Part C Requirement) [cite: 17]
print(f"Missing values before cleaning:\n{df.isnull().sum()}")
# Filling missing Age with the median age [cite: 17]
df['Age'] = df['Age'].fillna(df['Age'].median()) 
print("Missing values after cleaning:")
print(df.isnull().sum(), "\n")

# 2. Indexing, Filtering, Sorting [cite: 12]
filtered_df = df[df['Age'] > 30]
sorted_df = filtered_df.sort_values(by='Fare', ascending=False)
print("Filtered (Age > 30) and Sorted by Fare:")
print(sorted_df[['Name', 'Age', 'Fare']], "\n")

# 3. Groupby Summary [cite: 13, 17]
summary_stats = df.groupby('Pclass')['Fare'].mean().reset_index()
print("Groupby: Average Fare per Passenger Class:")
print(summary_stats, "\n")

# 4. Merge and Join [cite: 14]
# Creating a secondary dataframe to demonstrate merging [cite: 14]
class_names = pd.DataFrame({
    'Pclass': [1, 2, 3],
    'Class_Level': ['Luxury', 'Middle', 'Economy']
})
merged_df = pd.merge(df, class_names, on='Pclass')
print("Merged DataFrame with Class Descriptions:")
print(merged_df[['Name', 'Pclass', 'Class_Level']].head())

# Clean up generated csv file locally if needed, keeping it for git submission
print("\nTask execution finished successfully!")
# Task 04 – NumPy & Pandas for Data Analysis

## Project Overview
This project contains implementations for fundamental numerical computing using NumPy, data manipulations using Pandas, and an exploratory data analysis (EDA) mini-project on a subset of the Titanic dataset.

## Part A: NumPy Findings
- **Array Transformations:** Effectively created 1D and 2D structures, showcasing reshaping mechanisms.
- **Broadcasting Advantage:** Demonstrated how NumPy performs element-wise operations on arrays of mismatched shapes efficiently without copying data unnecessarily.
- **Vectorization vs Loops:** Showed that vectorized operations run orders of magnitude faster than standard Python loops due to underlying optimized C execution.
- **Linear Algebra:** Resolved matrix multiplications, transpositions, and matrix inversions successfully.

## Part B & C: Data Analysis Mini Project
- **Dataset:** A structural sample of the Titanic Passenger Dataset including dimensions like Class, Age, Fare, and Survival metrics.
- **Data Cleaning:** Identified missing entries in the `Age` attribute and systematically imputed them utilizing the median age parameter to eliminate data bias.
- **Key Exploratory Findings:**
  1. Passengers in Class 1 (`Luxury`) commanded significantly higher average fares compared to Class 3 (`Economy`).
  2. Filtering data isolated demographics over the age of 30 to understand targeted survival distributions relative to pricing tiers.
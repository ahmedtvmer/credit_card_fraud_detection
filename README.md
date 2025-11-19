# Credit Card Fraud Detection

## 1. Overview
This module covers the **Numerical Dataset** requirement for the Machine Learning Projects 2025 assignment.
* **Goal:** Implement **Linear Regression** and **KNN** as regressors.
* **Objective:** Predict the probability of a transaction being fraudulent.

## 2. Dataset Information
* **Dataset Name:** Credit Card Fraud Detection Dataset 2023
* **Source:** Kaggle
* **Task Type:** Binary Classification framed as Regression (Probability Estimation)
* **Target Variable:** `Class` (0 = Legitimate, 1 = Fraud)

## 3. Data Preprocessing Pipeline

### A. Cleaning
* **Handling Missing Values:**
  * Identified columns with `NaN` values.
  * **Action:** Removed all rows containing `NaN` values (`df.dropna()`).
  * **Logic:** Rows with missing data were found to be completely empty/corrupt, ensuring no valuable information was lost.

### B. Data Splitting Strategy
A **90/10** primary split was selected to maximize training data while maintaining a statistically significant validation/test set, given the large dataset size.

* **Logic:** The 10% holdout set contains ~23,000+ rows, sufficient for robust evaluation.

**Split Hierarchy:**
1. **Primary Split:** 90% Training / 10% Holdout
2. **Secondary Split:** The 10% Holdout was split 50/50 into Validation and Testing sets

**Final Distribution:**
* **Training Set:** 90% of total data
* **Validation Set:** 5% of total data (used for hyperparameter tuning)
* **Testing Set:** 5% of total data (used for final metrics)

## 4. Requirements Checklist (Pending)
* [ ] **Feature Scaling:** Normalize/Standardize numerical features (Critical for KNN).
* [ ] **Model Implementation:**
  * Linear Regression (Regressor)
  * K-Nearest Neighbors (Regressor)
* [ ] **Hyperparameter Tuning:** Record learning rates, optimizers, and $k$ values.
* [ ] **Evaluation:** Generate Loss Curves, Confusion Matrices, and ROC Curves.

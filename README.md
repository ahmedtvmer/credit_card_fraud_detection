# Project: Credit Card Fraud Detection (Numerical Component)

## 1. Overview
This module covers the **Numerical Dataset** requirement for the Machine Learning Projects 2025 assignment.
* **Goal:** Implement **Linear Regression** and **KNN** as regressors.
* **Objective:** Predict the probability of a transaction being fraudulent

## 2. Dataset Information
* **Dataset Name:** Credit Card Fraud Detection Dataset 2023.
* **Source:** Kaggle.
* **Total Samples:** ~284,807 (approx).
* **Features:** 29 numerical features (V1-V28, Amount).
* **Target Variable:** `Class` (0 = Legitimate, 1 = Fraud).

### Data Splitting Strategy
A **90/10** primary split was selected to maximize training data while maintaining a statistically significant validation/test set.

* **Training Set:** 90%.
* **Validation Set:** 5%.
* **Testing Set:** 5%.

## 3. Implementation Details

### A. Preprocessing
* **Cleaning:** Rows containing `NaN` values were removed (minimal data loss).
* **Feature Extraction:** All 29 original numerical features were used. No dimensionality reduction (PCA) was applied as features are already PCA-transformed (V1-V28).
* **Scaling:** `StandardScaler` was applied to all features.
    * *Note:* Fit on Training data only, then transformed Validation and Test data to prevent leakage.

### B. Model 1: Linear Regression (via SGD)
To satisfy the requirement for a **Loss Curve** and **Hyperparameter Tuning**, `SGDRegressor` was used instead of OLS.

* **Optimizer:** Stochastic Gradient Descent (SGD).
* **Hyperparameters:**
    * `penalty`: 'l2' (Ridge Regularization).
    * `learning_rate`: 'constant'.
    * `eta0` (Initial LR): `1e-6` (Tuned for smooth convergence visualization).
    * `epochs`: 15.
* **Training Strategy:**
    * Manual epoch loop used to record training and validation loss at each step.
    * Weights initialized randomly to visualize the "learning" curve over time.
* **Results:**
    * **Loss Curve:** Generated successfully (L-shape).
    * **Accuracy:** ~98.97% (on Test set).
    * **ROC AUC:** 1.00.

### C. Model 2: K-Nearest Neighbors (Regressor)
* **Algorithm:** `KNeighborsRegressor`.
* **Prediction Output:** Continuous probability score (mean of neighbors).
* **Optimization Strategy:**
    * **Subset Tuning:** Hyperparameter `k` (n_neighbors) was tuned using a subset of 10,000 samples to save computation time.
    * **Final Training:** The optimal `k` is applied to the full Training set using parallel processing (`n_jobs=-1`).
* **Hyperparameters:**
    * `n_neighbors (k)`: *[Pending results]*
    * `n_jobs`: -1 (Parallel processing).

## 4. Results Summary

| Model | Accuracy | ROC AUC | Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Linear Regression (SGD)** | 98.97% | 1.00 | `eta0=1e-6`, `epochs=15` |
| **KNN Regressor** | *[Pending]* | *[Pending]* | `k=` *[Pending]* |

## 5. Deliverables Checklist
* [x] **Source Code:** Uploaded to GitHub/Colab.
* [x] **Dataset Info:** Documented split and size.
* [x] **Feature Details:** 29 Features, Standard Scaling used.
* [x] **Hyperparameters:** Learning rates and optimizers listed.
* [x] **Results (Linear Regression):** Loss Curve, Accuracy, Confusion Matrix, ROC.
* [ ] **Results (KNN):** Accuracy, Confusion Matrix, ROC.

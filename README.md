# Project: Credit Card Fraud Detection

## 1. Overview
* **Goal:** Implement **Linear Regression** and **KNN** as regressors.
* **Objective:** Predict the probability of a transaction being fraudulent.

## 2. Dataset Information
* **Dataset Name:** Credit Card Fraud Detection Dataset 2023.
* **Source:** Kaggle.
* **Total Samples:** ~284,807 (after cleaning).
* **Features:** 29 numerical features (V1-V28, Amount).
* **Target Variable:** `Class` (0 = Legitimate, 1 = Fraud).

### Data Splitting Strategy
A **90/10** primary split was selected to maximize training data while maintaining a statistically significant validation/test set (~23k samples).

* **Training Set:** 90% (Used for model fitting).
* **Validation Set:** 5% (Used for hyperparameter tuning and loss tracking).
* **Testing Set:** 5% (Used for final evaluation: Accuracy, ROC, Confusion Matrix).

## 3. Implementation Details

### A. Preprocessing
* **Cleaning:** Rows containing `NaN` values were removed (minimal data loss observed).
* **Feature Extraction:** All 29 original numerical features were used. No dimensionality reduction (PCA) was applied as features are already PCA-transformed (V1-V28).
* **Scaling:** `StandardScaler` was applied to all features.
    * *Note:* Scaler was fit on **Training** data only, then applied to Validation and Test sets to prevent data leakage.

### B. Model 1: Linear Regression (via SGD)
To satisfy the requirement for a **Loss Curve** and **Hyperparameter Tuning**, `SGDRegressor` was used instead of OLS.

* **Optimizer:** Stochastic Gradient Descent (SGD).
* **Hyperparameters:**
    * `penalty`: 'l2'.
    * `learning_rate`: 'constant'.
    * `eta0`: `1e-6`.
    * `epochs`: 15.
* **Training Strategy:**
    * Manual epoch loop used to record training and validation loss at each step.
    * Weights initialized randomly to visualize the "learning" curve over time.
* **Results:**
    * **Loss Curve:** Generated successfully.
    * **Accuracy:** ~98.97%.
    * **ROC AUC:** ~1.00.

### C. Model 2: K-Nearest Neighbors (Regressor)
* **Algorithm:** `KNeighborsRegressor`.
* **Prediction Output:** Continuous probability score (mean of neighbors).
* **Optimization Strategy:**
    * **Subset Tuning:** Hyperparameter `k` (n_neighbors) was tuned using a subset of 10,000 samples to save computation time.
    * **Final Training:** The optimal `k` was applied to the full Training set using parallel processing (`n_jobs=-1`).
* **Hyperparameters:**
    * `n_neighbors (k)`: 3.
    * `n_jobs`: -1.
* **Results:**
    * **Accuracy:** 99.97%.
    * **ROC AUC:** ~1.00.

## 4. Results Summary (on Test set)

| Model | Accuracy | ROC AUC | Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Linear Regression (SGD)** | 98.97% | 1.00 | `eta0=1e-6`, `epochs=15` |
| **KNN Regressor** | 99.97% | 1.00 | `k=3` |

## 5. Checklist
* [x] **Source Code:** Uploaded to GitHub/Colab.
* [x] **Dataset Info:** Documented split, size, and cleaning steps.
* [x] **Feature Details:** 29 Features, Standard Scaling implemented.
* [x] **Hyperparameters:** Learning rates, optimizers, and $k$ values listed.
* [x] **Results:** Loss Curve, Accuracy, Confusion Matrix, and ROC Curve generated for relevant models.

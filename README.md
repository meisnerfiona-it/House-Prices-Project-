# 🏠 House Prices Prediction — Advanced Regression Pipeline

## 📌 Overview

This project tackles the Kaggle **House Prices: Advanced Regression Techniques** problem using a structured machine learning pipeline focused on robust preprocessing, feature engineering, and gradient boosting.

The goal is to predict housing prices with high accuracy while maintaining a clean, reproducible workflow.

---

## 🚀 Key Highlights

* 📊 Comprehensive data cleaning and missing value handling
* 🧠 Feature engineering based on domain intuition
* 📉 Skewness correction using log transformations
* 🔢 Consistent encoding across train/test datasets
* 🌲 XGBoost model with early stopping
* 📏 Proper evaluation using RMSE on log-transformed targets

---

---

## 🧹 Data Preprocessing

### Missing Values

* Categorical features filled with mode or "No" where appropriate
* Numerical features filled using median
* Ensures no leakage from target variable

### Feature Engineering

Key engineered features:

* `AllSF` → Total square footage
* `PorchSF` → Combined porch/deck area
* `Total_Bathrooms` → Weighted bathroom count
* `BackyardSF` → Lot area minus living space

These features aim to better capture real-world property value drivers.

---

## 🔢 Encoding Strategy

* One-hot encoding applied using `pd.get_dummies`
* Train and test sets aligned to ensure identical feature space:

---

## 🤖 Model

### XGBoost Regressor

Chosen for its strong performance on tabular data.

Key configuration:

* Learning rate: 0.01
* Large number of estimators with early stopping
* Objective: `reg:squarederror`

---

## 📏 Evaluation

The model is evaluated using RMSE on log-transformed prices:

---

## 🔁 Training Strategy

* Train/validation split used for early stopping
* Final model retrained on full dataset before submission

---

## 🧠 What I Learned

* The importance of correct evaluation metrics
* How feature engineering impacts model performance
* Why aligning datasets is critical after encoding
* The value of early stopping in boosting models

---

## 🔮 Future Improvements

* Cross-validation for more stable evaluation
* Feature selection based on importance
* Model stacking / ensembling
* Hyperparameter optimization

---

# Obesity Level Prediction — ML Capstone (23CSE301)

**B.Tech CSE — III Year | Machine Learning Capstone Project | AY 2026–27**

An end-to-end machine learning pipeline covering Regression, Classification, and Clustering tracks, built on the UCI "Estimation of Obesity Levels Based on Eating Habits and Physical Condition" dataset.

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

- **Rows:** 2,111
- **Features:** 17 (demographic, dietary, and lifestyle attributes — e.g. `Age`, `Height`, `Weight`, `Gender`, `family_history_with_overweight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`)
- **Targets used across tracks:**
  - `Weight` (continuous) — Regression track
  - `NObeyesdad` (7-class obesity category) — Classification track

## 🎯 Problem Tracks

| Track | Target | Status |
|---|---|---|
| **Regression** | `Weight` (kg) | ✅ Complete — 10/10 algorithms (Review 1) |
| **Classification** | `NObeyesdad` | 🔄 Part A complete (5/10 algorithms, Review 1) — Part B due Review 2 |
| **Clustering** | Unsupervised | ⏳ Due Review 2 |

## 🧠 Algorithms Implemented

**Regression:** Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression (deg 2 & 3), Decision Tree, Random Forest, Gradient Boosting, SVR, KNN Regressor

**Classification (Part A):** Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVM (SVC)

**Classification (Part B — Review 2):** Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP *(pending)*

**Clustering (Review 2):** K-Means, Agglomerative Hierarchical Clustering *(pending)*

## 📈 Results Summary

### Regression Track (test set, 80:20 split)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| _fill in from `results_df` in regression.ipynb_ | | | |

**Best model:** _TBD_ · **Tuned via GridSearchCV:** Random Forest, Gradient Boosting

### Classification Track — Part A (test set, stratified 80:20 split)

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC (OvR) |
|---|---|---|---|---|---|
| _fill in from `results_df` in classification.ipynb_ | | | | | |

**Best model:** _TBD_



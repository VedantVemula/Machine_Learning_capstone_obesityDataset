# 23CSE301 Machine Learning Capstone: Regression & Classification

**B.Tech. Computer Science and Engineering, III Year | Academic Year 2026-27**

**Status:** Review 1 complete (Regression track + Classification Part A). Review 2 (Classification Part B + Clustering) upcoming.

---

## 1. Project Overview

An end-to-end machine learning pipeline covering dataset audit, EDA, cleaning, feature engineering, model training and comparison, hyperparameter tuning, and visualisation. All experiments use an 80:20 train/test split and `random_state=42`. Every scaler and encoder is fitted on the training set only, to avoid data leakage.

| Track | Dataset | Target | Problem type | Review |
|---|---|---|---|---|
| Regression | Estimation of Obesity Levels (UCI) | `Weight` (kg) | Continuous | 1 |
| Classification | Telco Customer Churn (IBM / Kaggle) | `Churn` (Yes / No) | Binary | Part A: 1, Part B: 2 |
| Clustering | To be added | n/a | n/a | 2 |

---

## 2. Datasets

### 2.1 Obesity Levels (Regression)
- File: `data/ObesityDataSet_raw_and_data_sinthetic.csv`
- 2,111 rows x 17 columns; no missing values; 24 exact duplicate rows removed, leaving **2,087 rows**.
- Target `Weight`: mean 86.6 kg, std 26.2, range 39 to 173 kg.
- Features cover demographics (Gender, Age, Height), eating habits (FAVC, FCVC, NCP, CAEC, CH2O), physical condition and activity (SCC, FAF, TUE), lifestyle (SMOKE, CALC, MTRANS), and family history of overweight.
- **Data-leakage note:** `NObeyesdad` (the obesity class) is computed directly from BMI = Weight / Height², so it is **dropped** from the feature set.
- The dataset is grouped by obesity class rather than row-shuffled, which matters for cross-validation (see 5.1).

### 2.2 Telco Customer Churn (Classification)
- File: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7,043 customers x 20 columns (after dropping the `customerID` identifier).
- Class distribution: **5,174 No (73%)** vs **1,869 Yes (27%)**, so the data is imbalanced.
- `TotalCharges` is read as text because 11 brand-new customers (`tenure == 0`) have a blank value; these were converted to numeric and filled with 0.

---

## 3. Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── ObesityDataSet_raw_and_data_sinthetic.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── notebooks/
    ├── regression.ipynb                # Regression track (Review 1)
    ├── classification_telco.ipynb      # Classification Part A (Review 1)
    └── classification_partB.ipynb      # Classification Part B (Review 2, upcoming)
```

Plots are saved as PNG files (`reg_*.png`, `clf_*.png`) in the notebook's working directory when the notebooks are run.

---

## 4. Methodology

### 4.1 Preprocessing

| Step | Regression (Obesity) | Classification (Telco) |
|---|---|---|
| Cleaning | 24 duplicate rows removed | `TotalCharges` blanks fixed; 22 rows (0.31%) share an attribute profile with another row but are different customers, so none were dropped |
| Outliers (IQR rule) | Age: 167 values capped to the fence [10.8, 35.1]. Height (1) and Weight (1) outliers left untreated because they are plausible body measurements and Weight is the target | None found in `tenure`, `MonthlyCharges`, or `TotalCharges` |
| Encoding | One-hot (`drop_first=True`) on 8 categorical columns, giving 23 features | One-hot on categorical columns, giving 31 features; target label-encoded (No = 0, Yes = 1) |
| Split | 80:20, **stratified on Weight quintiles** (1,669 train / 418 test) | 80:20, **stratified on Churn** (5,634 train / 1,409 test) |
| Scaling | `StandardScaler` fitted on the training set only | `StandardScaler` fitted on the training set only |

### 4.2 Feature Engineering
- **`Veg_Meal_Ratio = FCVC / NCP`** (Regression): vegetable-consumption frequency relative to the number of main meals. It captures dietary quality relative to eating frequency, which neither raw column shows alone. It uses only feature columns, so there is no target leakage.
- **`TotalServices`** (Classification): the number of the 8 core/add-on services a customer subscribes to (mean 3.36, range 0 to 8). It is a compact proxy for engagement and lock-in with the provider; churners subscribe to noticeably fewer services.

### 4.3 Evaluation Metrics
- **Regression:** R², RMSE, MAE on the held-out test set, plus 5-fold cross-validated R² for the two best models.
- **Classification:** Accuracy, weighted Precision / Recall / F1, confusion matrix, and ROC-AUC, plus 5-fold cross-validated accuracy for the two best models.

---

## 5. Results

### 5.1 Regression: Predicting Weight

All 10 required algorithms were trained on the same preprocessed data and evaluated on the same test set, ranked by R².

| Rank | Model | R² | RMSE (kg) | MAE (kg) |
|---|---|---|---|---|
| 1 | Random Forest Regressor (200 trees) | 0.8623 | 9.92 | 5.34 |
| 2 | K-Nearest Neighbors Regressor (k=7) | 0.8221 | 11.28 | 6.20 |
| 3 | Support Vector Regressor (RBF, C=10) | 0.8080 | 11.72 | 7.52 |
| 4 | Gradient Boosting Regressor | 0.7953 | 12.10 | 8.36 |
| 5 | Decision Tree Regressor (max_depth=6) | 0.6959 | 14.75 | 9.50 |
| 6 | Polynomial Regression (degree 2) | 0.6586 | 15.62 | 10.59 |
| 7 | Linear Regression | 0.5710 | 17.51 | 13.88 |
| 8 | Ridge Regression (alpha=1.0) | 0.5710 | 17.52 | 13.89 |
| 9 | Lasso Regression (alpha=0.1) | 0.5692 | 17.55 | 13.89 |
| 10 | ElasticNet (alpha=0.1, l1_ratio=0.5) | 0.5627 | 17.68 | 14.04 |
| n/a | Polynomial Regression (degree 3), compared against degree 2 | -6.6135 | 73.78 | 28.46 |

**Hyperparameter tuning** (`GridSearchCV`, 5-fold shuffled CV, scoring = R²):

| Model | Search space | Best parameters | Baseline test R² | Tuned test R² | Change |
|---|---|---|---|---|---|
| Random Forest | n_estimators [100, 200, 300], max_depth [None, 8, 12], min_samples_leaf [1, 2, 4] | n_estimators=300, max_depth=None, min_samples_leaf=1 | 0.8623 | 0.8610 | -0.0013 |
| Gradient Boosting | n_estimators [100, 200], learning_rate [0.05, 0.1, 0.2], max_depth [2, 3, 4] | n_estimators=200, learning_rate=0.1, max_depth=4 | 0.7953 | 0.8182 | +0.0229 |

**5-fold cross-validated R² (two best models):**

| Model | Mean R² | Std |
|---|---|---|
| Random Forest Regressor | 0.8951 | 0.0153 |
| KNN Regressor | 0.8321 | 0.0251 |

**Key findings**
- Non-linear models (Random Forest, KNN, SVR, Gradient Boosting) clearly outperform linear models (R² about 0.57), so the relationship between the features and Weight is non-linear.
- Linear Regression's largest coefficients are `Height` (+10.94), `Veg_Meal_Ratio` (-7.59), `FCVC` (+7.45), `Age` (+6.62), and `MTRANS_Public_Transportation` (+6.37).
- Lasso (alpha=0.1) zeroed out only 1 of 23 coefficients, so the feature set is not very sparse.
- Polynomial Regression at degree 3 collapses (R² = -6.61). About 23 features expanded to degree 3, fitted with unregularised linear regression on about 1,670 training rows, overfits severely. Degree 2 is the usable setting.
- Tuning gave Gradient Boosting a clear gain (+0.023 R²). For Random Forest the untuned baseline was already near its ceiling and was marginally better on this single test split.
- The raw CSV is ordered by obesity class. A default un-shuffled `KFold` gave each fold a very different Weight distribution and produced meaningless negative CV scores, so a shuffled `KFold(n_splits=5, random_state=42)` is used for all cross-validation and grid searches.

**Visualisations:** residual plot and predicted-vs-actual plot (tuned Random Forest), and a top-15 feature-importance plot (Random Forest).

### 5.2 Classification, Part A: Predicting Customer Churn

Five Part-A algorithms trained on the same stratified split, ranked by accuracy.

| Rank | Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | Logistic Regression | 0.8070 | 0.7996 | 0.8070 | 0.8019 | 0.8419 |
| 2 | SVM (SVC, RBF, C=10) | 0.7850 | 0.7699 | 0.7850 | 0.7705 | 0.7871 |
| 3 | Decision Tree (max_depth=8) | 0.7828 | 0.7803 | 0.7828 | 0.7815 | 0.7999 |
| 4 | K-Nearest Neighbors (k=7) | 0.7551 | 0.7513 | 0.7551 | 0.7531 | 0.7828 |
| 5 | Gaussian Naive Bayes | 0.6572 | 0.7928 | 0.6572 | 0.6763 | 0.8087 |

SVM probabilities for ROC-AUC come from probability calibration (`CalibratedClassifierCV`).

**Best model (Logistic Regression), per-class report on the test set (1,409 customers):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No (stayed) | 0.85 | 0.89 | 0.87 | 1,035 |
| Yes (churned) | 0.66 | 0.56 | 0.61 | 374 |
| Accuracy | | | 0.81 | 1,409 |

**5-fold cross-validated accuracy (two best models):**

| Model | Mean accuracy | Std |
|---|---|---|
| Logistic Regression | 0.8038 | 0.0080 |
| SVM (SVC) | 0.7907 | 0.0092 |

**Key findings**
- The strongest Logistic Regression coefficients are `tenure` (-1.26), `MonthlyCharges` (-0.94), `InternetService_Fiber optic` (+0.78), `Contract_Two year` (-0.59), and `TotalCharges` (+0.54). Short-tenure, month-to-month, fiber-optic customers are the highest churn risk.
- EDA showed churners concentrated at low tenure and higher monthly charges. Month-to-month contracts, fiber-optic internet, and no OnlineSecurity or TechSupport are associated with more churn, while `gender` and `PhoneService` show almost no effect.
- `tenure` and `TotalCharges` are strongly correlated (about 0.83), since charges accumulate over time.
- Because of the 73/27 imbalance, accuracy alone is misleading (predicting "No" for everyone gives about 73%). Recall on the churn class (0.56 for the best model) is the main limitation to address.
- Gaussian Naive Bayes scores lowest on accuracy because its conditional-independence and Gaussian assumptions are violated by one-hot and highly correlated billing features.

**Visualisations:** class distribution, numeric and categorical distributions by churn, correlation heatmap, feature-target scatter plots, engineered-feature boxplot, decision-tree diagram (top 3 levels), confusion matrices for all models, model comparison chart, and per-class precision/recall/F1 chart.

---

## 6. How to Run

**Requirements:** Python 3.10 or newer.

1. Download or clone this repository and open a terminal in the project folder.
2. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy the two CSV files from `data/` into the `notebooks/` folder. The notebooks load the datasets by filename.
4. Launch Jupyter and run each notebook top to bottom (**Kernel > Restart & Run All**):

```bash
cd notebooks
jupyter notebook
```

**`requirements.txt`**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 7. Reproducibility
- `random_state=42` and `np.random.seed(42)` are set in every notebook.
- The same 80:20 split is used for all algorithms within a track, so metrics are directly comparable.
- Scalers and encoders are fitted on the training set only.
- Cross-validation for regression uses a shuffled `KFold` with a fixed seed.

---

## 8. Roadmap (Review 2)
- [ ] Classification Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), MLP
- [ ] Consolidated 10-algorithm comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
- [ ] Final model selection with hyperparameter tuning and documented improvement
- [ ] Clustering: K-Means (elbow curve) and Agglomerative (dendrogram) with Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics, plus PCA and t-SNE visualisations
- [ ] Optional bonus: Streamlit/Gradio interface and public deployment

---

## 9. Academic Integrity & AI Disclosure
Generative AI tools were used for code scaffolding only. All analysis, interpretation, and feature-engineering decisions are the team's own. Any code adapted from external sources is cited in the notebook Markdown cells.

## 10. References
- Palechor, F. M., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.* UCI Machine Learning Repository.
- IBM Sample Data Sets: *Telco Customer Churn.*
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12.


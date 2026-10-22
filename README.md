# 23CSE301 Machine Learning Capstone: Regression, Classification & Clustering

**B.Tech. Computer Science and Engineering, III Year | Academic Year 2026–27**

**Status:** Review 1 completed (Regression and Classification Part A). Review 2 work completed in the supplied notebooks: Classification Part B (10-model comparison) and Clustering (customer segmentation).

---

## 1. Project Overview

This capstone implements three machine-learning tracks: regression, supervised classification, and unsupervised clustering. The workflow includes dataset auditing, exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model training and comparison, evaluation, and visualisation.

The notebooks use `random_state=42` where applicable. Preprocessing choices are described in each notebook. The clustering track is unsupervised and therefore does not use a target label.

| Track | Dataset | Target / objective | Problem type | Review |
|---|---|---|---|---|
| Regression | Estimation of Obesity Levels (UCI) | Predict `Weight` (kg) | Regression | 1 |
| Classification | Telco Customer Churn (IBM / Kaggle) | Predict `Churn` (`Yes` / `No`) | Binary classification | Part A: 1; Part B: 2 |
| Clustering | Credit Card Dataset for Clustering | Segment customers by credit-card behaviour | Unsupervised clustering | 2 |

## 2. Datasets

### 2.1 Obesity Levels — Regression

- **File:** `ObesityDataSet_raw_and_data_sinthetic.csv`
- The source dataset contains 2,111 rows and 17 columns. After removing 24 exact duplicate rows, 2,087 rows remain.
- Target: `Weight` (kg); observed range: 39–173 kg.
- Features include demographics, eating habits, physical activity, lifestyle, and family history of overweight.
- **Leakage prevention:** `NObeyesdad` is excluded because the obesity category is directly related to BMI, which is calculated using weight and height.
- The source rows are grouped by obesity class, so shuffled cross-validation is used.

### 2.2 Telco Customer Churn — Classification

- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7,043 customer records; `customerID` is excluded from the predictors.
- Target distribution: 5,174 `No` (about 73%) and 1,869 `Yes` (about 27%), so the target is imbalanced.
- `TotalCharges` contains blank values for 11 customers with `tenure == 0`; these are converted to numeric and filled with 0.
- Feature engineering adds `TotalServices`, a count of the eight selected phone/internet services.

### 2.3 Credit Card Customer Segmentation — Clustering

- **File:** `credit_card.csv`
- 8,950 rows and 18 columns in the supplied notebook; `CUST_ID` is removed, leaving 17 numeric behavioural features.
- The features describe balances, purchases, cash advances, purchase frequency, transactions, credit limit, payments, minimum payments, full-payment behaviour, and tenure.
- The audit found 313 missing values in `MINIMUM_PAYMENTS` and one in `CREDIT_LIMIT`; no exact duplicate rows were found.
- Missing numeric values are imputed with each feature's median. Non-negative, strongly right-skewed features are transformed with `log1p`, then the features are standardised.
- There is no target variable: the goal is to discover groups with similar credit-card behaviour.

---

## 3. Repository Structure

The following is the intended repository layout. Keep the CSV files in the locations expected by the notebooks, or update the notebook file paths if your folder layout differs.

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── ObesityDataSet_raw_and_data_sinthetic.csv
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── credit_card.csv
└── notebooks/
    ├── regression.ipynb
    ├── classification_telco.ipynb
    └── clustering.ipynb
```

Plots are displayed by the notebooks when executed. The notebooks may also save figures depending on the cell being run.

## 4. Methodology

### 4.1 Preprocessing summary

| Step | Regression — Obesity | Classification — Telco | Clustering — Credit Card |
|---|---|---|---|
| Cleaning | Remove 24 exact duplicate rows | Convert `TotalCharges` to numeric and fill 11 blanks with 0; retain distinct customers with repeated attribute profiles | Remove `CUST_ID`; median-impute missing values; replace infinities with missing values |
| Feature handling | Drop `NObeyesdad` to avoid target leakage; cap 167 `Age` values at the IQR fences | One-hot encode categorical predictors; encode `No` as 0 and `Yes` as 1 | Apply `log1p` to selected non-negative, strongly skewed features |
| Feature engineering | `Veg_Meal_Ratio = FCVC / NCP` | `TotalServices` counts eight selected services | No target-derived features; retain 17 numeric behavioural features |
| Scaling | `StandardScaler` fitted on training data only | Scale features for distance-/margin-based models; tree models use unscaled features | Standardise all cleaned features |
| Evaluation | R², RMSE, MAE; shuffled 5-fold CV R² for the two top models | Accuracy, weighted precision/recall/F1, ROC-AUC, confusion matrices, and 5-fold CV accuracy for the two leading Part-A models | Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index |

### 4.2 Feature engineering

- **Regression — `Veg_Meal_Ratio`:** vegetable-consumption frequency divided by the number of main meals (`FCVC / NCP`). It combines two input features without using the target.
- **Classification — `TotalServices`:** counts the selected eight phone/internet services. It provides a compact indicator of the breadth of a customer's subscribed services.

### 4.3 Evaluation metric interpretation

- **Regression:** higher R² is better; lower RMSE and MAE indicate smaller prediction errors.
- **Classification:** higher Accuracy, Precision, Recall, F1, and ROC-AUC are generally better. Because churn is imbalanced, accuracy alone is not enough; recall for customers who churn is especially important.
- **Clustering:** higher Silhouette and Calinski–Harabasz scores are preferred, while a lower Davies–Bouldin index is preferred. These metrics assess cluster separation/compactness without ground-truth labels.

---

## 5. Results

### 5.1 Regression — Predicting Weight

Ten required regression algorithms were evaluated on the same held-out test split. Values below are from the notebook's recorded run.

| Rank | Model | R² | RMSE (kg) | MAE (kg) |
|---:|---|---:|---:|---:|
| 1 | Random Forest Regressor (200 trees) | 0.8628 | 9.91 | 5.33 |
| 2 | K-Nearest Neighbors Regressor (k=7) | 0.8221 | 11.28 | 6.20 |
| 3 | Support Vector Regressor (RBF, C=10) | 0.8080 | 11.72 | 7.52 |
| 4 | Gradient Boosting Regressor | 0.7954 | 12.09 | 8.36 |
| 5 | Decision Tree Regressor (max_depth=6) | 0.6959 | 14.75 | 9.50 |
| 6 | Polynomial Regression (degree 2) | 0.6586 | 15.62 | 10.59 |
| 7 | Linear Regression | 0.5710 | 17.51 | 13.88 |
| 8 | Ridge Regression (alpha=1.0) | 0.5710 | 17.52 | 13.89 |
| 9 | Lasso Regression (alpha=0.1) | 0.5692 | 17.55 | 13.89 |
| 10 | ElasticNet (alpha=0.1, l1_ratio=0.5) | 0.5627 | 17.68 | 14.04 |
| — | Polynomial Regression (degree 3; diagnostic comparison) | -6.6135 | 73.78 | 28.46 |

**Hyperparameter tuning**

| Model | Best parameters from GridSearchCV | Baseline test R² | Tuned test R² | Change |
|---|---|---:|---:|---:|
| Random Forest | `n_estimators=300`, `max_depth=None`, `min_samples_leaf=1` | 0.8628 | 0.8618 | -0.0010 |
| Gradient Boosting | `n_estimators=200`, `learning_rate=0.1`, `max_depth=4` | 0.7954 | 0.8182 | +0.0227 |

**Five-fold cross-validation for the two top test-set models**

| Model | Mean R² | Standard deviation |
|---|---:|---:|
| Random Forest Regressor | 0.8951 | 0.0155 |
| KNN Regressor | 0.8321 | 0.0251 |

**Key findings**

- The non-linear models outperformed the linear models on this split; Random Forest achieved the highest test R².
- Gradient Boosting improved after tuning, while the tuned Random Forest was marginally below its baseline on the held-out test set.
- Degree-3 Polynomial Regression performed very poorly, indicating that the expanded feature space overfit or became numerically unstable without stronger regularisation.
- Shuffled cross-validation is important because the original dataset is ordered by obesity class.

**Visualisations:** residual plot, predicted-versus-actual plot, and Random Forest feature-importance plot.

### 5.2 Classification — Telco Customer Churn (Parts A and B)

The notebook evaluates ten algorithms using the same train/test split and reports Accuracy, weighted Precision, weighted Recall, weighted F1, and ROC-AUC. The table is sorted by test accuracy; figures are taken from the recorded notebook outputs.

| Rank by accuracy | Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Logistic Regression | 0.8070 | 0.7996 | 0.8070 | 0.8019 | 0.8419 |
| 2 | Bagging | 0.8027 | 0.7925 | 0.8027 | 0.7943 | 0.8409 |
| 3 | Gradient Boosting | 0.7984 | 0.7874 | 0.7984 | 0.7891 | 0.8435 |
| 4 | MLP Neural Network | 0.7970 | 0.7868 | 0.7970 | 0.7891 | 0.8337 |
| 5 | AdaBoost | 0.7963 | 0.7851 | 0.7963 | 0.7870 | 0.8423 |
| 6 | SVM (RBF) | 0.7850 | 0.7699 | 0.7850 | 0.7705 | 0.7871 |
| 7 | Decision Tree | 0.7828 | 0.7803 | 0.7828 | 0.7815 | 0.8000 |
| 8 | Random Forest | 0.7800 | 0.7677 | 0.7800 | 0.7709 | 0.8252 |
| 9 | K-Nearest Neighbors | 0.7551 | 0.7513 | 0.7551 | 0.7531 | 0.7833 |
| 10 | Gaussian Naive Bayes | 0.6572 | 0.7928 | 0.6572 | 0.6763 | 0.8087 |

**Best overall model by accuracy and weighted F1:** Logistic Regression (Accuracy 0.8070; weighted F1 0.8019).

For Logistic Regression, the recorded per-class test report is:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| No (stayed) | 0.85 | 0.89 | 0.87 | 1,035 |
| Yes (churned) | 0.66 | 0.56 | 0.61 | 374 |

**Part-A cross-validation**

| Model | Mean 5-fold CV accuracy | Standard deviation |
|---|---:|---:|
| Logistic Regression | 0.8038 | 0.0080 |
| SVM (SVC) | 0.7907 | 0.0092 |

**Key findings**

- Logistic Regression had the strongest overall test accuracy and weighted F1 in the recorded run. Bagging was a close second by accuracy.
- Gradient Boosting had the highest ROC-AUC in the ten-model table (about 0.8435), so the model ranking differs depending on the metric.
- Churn recall for the Logistic Regression model was 0.56. This means the model missed a meaningful portion of actual churners, despite its overall accuracy of about 81%.
- The dataset's class imbalance makes accuracy alone insufficient. A baseline that always predicts `No` would already be around 73% accurate.
- In the Part-A coefficient analysis, short tenure, higher monthly charges, fibre-optic internet, and month-to-month contracts were associated with churn risk. These are model associations, not proof of causation.

**Part-B algorithms:** Random Forest, AdaBoost, Gradient Boosting, Bagging with a Decision Tree base estimator, and MLP Neural Network.

**Visualisations:** class distributions, feature distributions and correlation heatmap, scatter plots, confusion matrices, decision-tree visualisation, model comparison, per-class metrics, feature importance, and ROC curves.

### 5.3 Clustering — Credit Card Customer Segmentation

Two clustering algorithms were evaluated with four clusters each: K-Means and Agglomerative Hierarchical Clustering (Ward linkage).

**Cluster-size counts from the recorded run**

| Algorithm | Cluster counts |
|---|---|
| K-Means (`k=4`) | Cluster 0: 1,923; Cluster 1: 3,105; Cluster 2: 2,393; Cluster 3: 1,529 |
| Agglomerative Hierarchical (`k=4`) | Cluster 0: 2,701; Cluster 1: 2,245; Cluster 2: 1,984; Cluster 3: 2,020 |

**Internal clustering metrics**

| Algorithm | Number of clusters | Silhouette Score ↑ | Davies–Bouldin Index ↓ | Calinski–Harabasz Index ↑ |
|---|---:|---:|---:|---:|
| K-Means | 4 | 0.2150 | 1.6559 | 2302.5672 |
| Agglomerative Hierarchical | 4 | 0.1784 | 1.8456 | 1916.3725 |

**Interpretation**

- K-Means scored better on all three reported internal metrics for this run: higher Silhouette and Calinski–Harabasz scores, and a lower Davies–Bouldin index.
- The Silhouette score of 0.215 indicates that the discovered groups overlap to some degree; the segments should not be treated as perfectly separated customer types.
- PCA was used to visualise the clusters in two dimensions. The first two components explain **56.27%** of the scaled-feature variance.
- The notebook also includes a dendrogram based on a sample of up to 1,000 customers and a t-SNE visualisation based on a sample of up to 2,000 customers. These are visual aids; the quantitative comparison uses the three internal clustering metrics above.

**Visualisations:** feature distributions, correlation heatmap, box plots, K-Means elbow curve, hierarchical dendrogram, PCA cluster plots, and t-SNE plot.

---

## 6. How to Run

**Requirements:** Python 3.10 or newer, Jupyter Notebook, and the packages listed below.

1. Place the three CSV files where the notebook paths expect them. In the current notebook versions, the datasets are loaded by filename, so running notebooks from a directory containing the corresponding CSV files is the simplest setup.
2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch Jupyter in the directory containing the notebooks and required CSV files:

```bash
jupyter notebook
```

5. Open each notebook and use **Kernel → Restart & Run All** to reproduce the workflow. Some outputs and metrics can vary with library versions or if the data or parameters change.

### `requirements.txt`

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

---

## 7. Reproducibility and Limitations

- Fixed random seeds are used where applicable in the notebooks.
- Models within a track use the same held-out split for comparison.
- The regression notebook fits its scaler on the training set only. The classification notebook uses scaled inputs for scale-sensitive models and unscaled inputs for tree-based models.
- The clustering notebook has no train/test split because it is an unsupervised segmentation exercise; median imputation, log transformation, and scaling are applied before clustering.
- Reported metrics describe the supplied notebook runs and should be regenerated after changing data, parameters, or library versions.
- Clustering labels such as `Cluster 0` and `Cluster 1` are arbitrary identifiers. Business interpretations should be supported by examining each cluster's feature profile before naming segments.

## 8. Review 2 Deliverables

- [x] Classification Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging, and MLP.
- [x] Consolidated comparison of all 10 classification algorithms with Accuracy, weighted Precision/Recall/F1, and ROC-AUC.
- [x] Confusion matrices and ROC-curve visualisation for classification.
- [x] Clustering preprocessing and exploratory analysis for the credit-card dataset.
- [x] K-Means elbow analysis and Agglomerative Hierarchical dendrogram.
- [x] Comparison using Silhouette, Davies–Bouldin, and Calinski–Harabasz metrics.
- [x] PCA and t-SNE cluster visualisations.
- [ ] Optional extension: Streamlit/Gradio interface and public deployment.

## 9. Academic Integrity & AI Disclosure

Generative AI tools were used for code scaffolding and documentation support. The team should verify, understand, and be able to explain all code, model choices, metric interpretations, and feature-engineering decisions. Any code adapted from external sources should be cited in the relevant notebook cells, in accordance with course requirements.

## 10. References

- Palechor, F. M., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.* UCI Machine Learning Repository.
- IBM Sample Data Sets. *Telco Customer Churn.*
- *Credit Card Dataset for Clustering* (`credit_card.csv`), as used in the clustering notebook.
- Pedregosa, F. et al. (2011). “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research*, 12.

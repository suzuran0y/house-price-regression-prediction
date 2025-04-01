# 🏠 House Prices: Advanced Regression Techniques
（[US English](README.md) | [CN 中文](README_CN.md)）

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-brightgreen)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-lightgrey)](https://xgboost.readthedocs.io/)
[![mlxtend](https://img.shields.io/badge/mlxtend-0.22.0-purple)](http://rasbt.github.io/mlxtend/)

[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/downloads/)
 [![Miniconda](https://img.shields.io/badge/Anaconda-Miniconda-violet.svg)](https://docs.anaconda.net.cn/miniconda/install/)

>Predicting house prices using advanced regression techniques
(A Kaggle competition solution integrating model stacking and feature engineering)

---

## ✨ Project Highlights
- 🏆 Achieved Top 15% best ranking in the Kaggle competition
- 🔁 Demonstrates a complete machine learning workflow (EDA → Feature Engineering → Model Ensembling → Visualization)
- 🧩 Supports both modular and integrated execution for flexible customization
- ✒️ Cleanly structured, easy to reuse and extend—ideal as a reference template for regression projects

---

## 📃 Project Overview

This project is based on the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) , which provides a dataset of home sales with 79 explanatory variables. The goal is to build a model that can predict house sale prices (**SalePrice**) as accurately as possible.

This competition is ongoing indefinitely and uses a **rolling leaderboard**, where rankings update in real time as participants submit new predictions. Anyone can join at any time.

Based on the integrated version [`integration_code.py`](source/integration_code.py), the project has been optimized into a modular architecture, covering the full pipeline—from data preprocessing and feature engineering to model training, ensemble prediction, and result visualization.

---

## 🎯 Core Objectives

- Analyze and model housing features to predict their final sale prices
- Use `Root Mean Squared Logarithmic Error (RMSLE)` as the evaluation metric
- Build a robust and generalizable ensemble model architecture to reduce overfitting risk

---

## 📂 Project Structure

This project adopts a modular architecture that implements the full pipeline from data preprocessing and feature engineering to model training, ensemble prediction, and result visualization. The structure is clear and easy to maintain or extend, making it a great reference template for regression tasks.

```bash
house-price-regression-prediction/
├── data/                    # 📚 Raw data, intermediate results, and final predictions
├── figure/                  # 📈 EDA & model evaluation figures (used in README)
├── models/                  # 🧠 Trained models saved in .pkl format
│   ├── ridge_model.pkl
│   ├── xgb_model.pkl
│   └── ... (total 7 models)
├── source/                  # 🧩 Integrated and modular code files
│   │   ⬇ Integrated version # One complete script (single-run)
│   ├── integration_code.py     
│   │   ⬇ Modular version    # Independent scripts (modular workflow)
│   ├── main.py              # Entry point that orchestrates the modules
│   └── ... (9 modules total) ...
│── requirements.txt         # 📦 Dependency list
└── README.md                # 📄 Project documentation
```

---

## 🧱 Modular Design

To improve readability, maintainability, and reusability, the complete modeling workflow is divided into multiple functional modules based on the Single Responsibility Principle. All modules are stored under the `source/` directory, and the main program is `main.py`, which orchestrates the full prediction pipeline.

The functions of each module are listed as follows:

| Module File              | Description                                                                                                                       |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `data_loader.py`         | Loads training and test datasets, removes `Id` column, and returns raw data and corresponding IDs                                 |
| `eda.py`                 | Exploratory Data Analysis (EDA), including distribution plots, scatter plots, heatmaps, and feature visualizations                |
| `preprocessing.py`       | Handles outliers, fills in missing values, log-transforms the target variable, and merges datasets                                |
| `feature_engineering.py` | Performs skewness correction, constructs combined features, applies log/square transformations, and encodes categorical variables |
| `model_builder.py`       | Defines base models and stacking model, including LGBM, XGBoost, SVR, etc.                                                        |
| `model_training.py`      | Wraps evaluation metrics, cross-validation, and training functions                                                                |
| `model_fusion.py`        | Handles ensemble strategies (Stacking and Blending)                                                                               |
| `utils.py`               | Utilities for saving models and predictions, plotting evaluation graphs, and exporting results                                    |
| `main.py`                | Main control flow that links all modules and outputs predictions and visualizations                                               |

Each module has a clear interface, allowing independent execution and testing. This design makes it easy to replace models, add new features, or extend functionality in the future.

---

## 🚀 Quick Start

You can follow the steps below to clone and run this project:

```bash
# 1. Clone the repository
git clone https://github.com/suzuran0y/house-price-regression-prediction.git
cd house-price-regression-prediction

# 2. Create a virtual environment and install dependencies
conda create -n house_price_prediction python=3.10
conda activate house_price_prediction
pip install -r requirements.txt

# 3. Run the main script
# python source/integration_code.py    # (Integrated version)
# python source/main.py                # (Modular version)

```
The dataset files (`train.csv` / `test.csv`) should be placed in the `data/` folder. Output results will be automatically saved under `models/` and `data/`.

---

## 🍏 Project Functionality Overview

Based on the integrated script [`integration_code.py`](source/integration_code.py), the project is divided into four major components:
`Data Loading & Exploratory Data Analysis (EDA)`, `Missing Value Handling & Data Cleaning`, `Feature Transformation & Construction`, and `Model Building & Ensembling`.

Compared with the modular code, the integrated version reveals each analytical and visualization step performed on the raw dataset, closely following the practical thought process. (Note: Some output statements are commented out to reduce verbosity.)

---

## 1. Data Loading & Exploratory Data Analysis (EDA)

We start by loading both the training and test datasets, and then perform a series of visual analyses on the target variable `SalePrice` and its relationship with other features.

---

### 📈 1.1 SalePrice Distribution

The target variable `SalePrice` is clearly right-skewed and does not follow a normal distribution. Therefore, we later apply a logarithmic transformation to make it more suitable for modeling.

<p align="center">
  <img src="figure/1 - SalePrice distribution (original state).png" alt="SalePrice distribution (original state)" width="360px">
  <br>
  <b>SalePrice distribution (original state)</b>
</p>

Additionally, we calculate the `skewness` and `kurtosis` of `SalePrice` to quantify its deviation from normality.

### 📊 1.2 Visualization of Numerical Feature Relationships

We visualize scatter plots of all numerical features against `SalePrice` to examine their correlation and detect potential outliers.

<p align="center">
  <img src="figure/2 - Visualize numerical features against SalePrice (scatter plots).png" alt="Visualize numerical features against SalePrice" width="1080px">
  <br>
  <b>Numerical features against SalePrice</b>
</p>

### 🔥 1.3 Feature Correlation Heatmap

By plotting the correlation matrix, we identify features that are strongly linearly correlated with `SalePrice`, such as `OverallQual`, `GrLivArea`, and `TotalBsmtSF`.

<p align="center">
  <img src="figure/3 - Plot heatmap of correlation between features.png" alt="Plot heatmap of correlation between features" width="360px">
  <br>
  <b>Correlation between features</b>
</p>

### 📦 1.4 Key Feature Distribution Analysis

We further analyze the relationships between several critical features and house prices—such as overall quality (`OverallQual`), year built (`YearBuilt`), and above-ground living area (`GrLivArea`).

- `YearBuilt` vs `SalePrice`

<p align="center">
  <img src="figure/5 - YearBuilt vs SalePrice relationship (boxplot).png" alt="YearBuilt vs SalePrice relationship (boxplot)" width="640px">
  <br>
  <b>YearBuilt vs SalePrice relationship</b>
</p>

- `OverallQual` vs `SalePrice` (boxplot) and  `GrLivArea` vs `SalePrice` (scatter plot)

<table style="margin: 0 auto;">
  <tr>
    <td align="center">
      <img src="figure/4 - Relationship between OverallQual and SalePrice (boxplot).png"  alt="Relationship between OverallQual and SalePrice (boxplot)" width="360px"><br>
      <b>OverallQual vs SalePrice relationship</b>
    </td>
    <td align="center">
      <img src="figure/8 - GrLivArea vs SalePrice relationship (scatter plot).png" alt="GrLivArea vs SalePrice relationship (scatter plot)" width="360px"><br>
      <b>GrLivArea vs SalePrice relationship</b>
    </td>
  </tr>
</table>

---

## 2. Missing Value Handling and Data Cleaning

Both training and test datasets contain missing values in several features. We begin by calculating the percentage of missing values for each feature and visualizing the distribution. Then, based on domain knowledge and logical reasoning, we adopt tailored imputation strategies.

---

### 📊 2.1 Visualization of Missing Values

We visualize the proportion of missing data for each feature to better understand the extent and distribution of missingness.

<p align="center"> 
  <img src="figure/10 - Visualize missing values distribution.png" alt="Visualize missing values distributionn" width="360px">
  <br>
  <b>Missing values distribution</b>
</p>

---

### 🛠️ 2.2 Imputation Strategies

We apply different filling strategies based on the nature of each feature:

- **Categorical Variables**：
  - `Functional`: Missing values imply normal functionality (`Typ`)
  - `Electrical`, `KitchenQual`, `Exterior1st/2nd`, `SaleType`: Filled with the mode (most frequent value)
  - `MSZoning`: Grouped and filled by mode based on `MSSubClass`
  - Garage-related and basement-related fields (e.g., `GarageType`, `BsmtQual`): NA indicates absence and is filled with `'None'`

- **Numerical Variables**：
  - `GarageYrBlt`, `GarageArea`, `GarageCars`: Missing values are filled with 0
  - `LotFrontage`: Filled using the median value within each `neighborhood`
  - All other numerical features are filled with 0

- **Special Handling**：
  - Fields such as `MSSubClass`, `YrSold`, and `MoSold` are treated as categorical variables and converted to string type
  - A final check ensures that all missing values are handled properly

---

### ✂️ 2.3 ID Removal and Target Variable Transformation

The `Id` column is removed as it only serves as a unique identifier and does not contribute to prediction. The target variable `SalePrice` is transformed using `log(1 + x)` to reduce skewness and improve model robustness.

<p align="center"> 
  <img src="figure/9 - View transformed SalePrice distribution + fit normal distribution.png" alt="Log-transformed SalePrice distribution" width="360px">
  <br>
  <b>Log-transformed SalePrice distribution</b>
</p>

---

### 🚫 2.4 Outlier Removal

Using scatter plots and logical rules, we manually remove several clear outliers:

- Houses with `OverallQual < 5` but unusually high prices
- Houses with `GrLivArea > 4500` but unexpectedly low prices

Such samples may mislead the model and are excluded from training.

---

### 🔗 2.5 Merging Training and Test Sets (for Unified Preprocessing)

- Extract the `SalePrice` label from the training set
- Concatenate training and test features into a single dataframe `all_features`

This enables unified preprocessing such as encoding, transformation, and feature engineering.

---
## 3. Feature Transformation and Construction

Feature engineering is one of the core components of this project. Its goal is to help the model better capture complex relationships among features, thereby improving predictive performance and generalization.

---

### 📝 3.1 Feature Merging

To streamline preprocessing, we concatenate the training and test feature sets (excluding labels) into a unified feature matrix `all_features`:

```bash
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
```

---

### 🔬 3.2 Skewness Detection and Normalization

Highly **skewed** numerical features can hurt model performance. Therefore, we identify features with `skewness > 0.5` and apply the **Box-Cox transformation** to normalize their distributions.

```bash
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
```

- **Before normalization**: Many features (e.g., `PoolArea`) are strongly right-skewed, with extreme outliers and long tails—problematic for modeling.

- **After normalization**: Most skewed features become more symmetric and centered, outliers are reduced or more reasonable—helping to stabilize the model. Some skewness may remain, but its impact is significantly reduced.

<div align="center">
<table>
  <tr>
    <td align="center">
      <img src="figure/11 - Visualize the distribution of numerical features (Boxplot).png" alt="distribution of numerical features (Boxplot)" width="360px"><br>
      <b>Numerical features distribution (origin)</b>
    </td>
    <td align="center">
      <img src="figure/12 - Visualize normalized skewed feature distributions.png" alt="Numerical features distribution" width="360px"><br>
      <b>Numerical features distribution (normalized)</b>
    </td>
  </tr>
</table>
</div>

---

### 🧱 3.3 Construction of Combined Features

Beyond raw variables, we introduce several **domain-informed combined features** to enhance the model’s understanding of structure, area, and overall quality:


- `Total_Home_Quality = OverallQual + OverallCond`: An indicator of overall home quality
- `YearsSinceRemodel = YrSold - YearRemodAdd`: Years since last renovation
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`: Total floor area
- `Total_sqr_footage = BsmtFinSF1 + BsmtFinSF2 + 1stFlrSF + 2ndFlrSF`: Effective square footage
- `Total_Bathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`: Combined count of full and half bathrooms
- `Total_porch_sf = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF`: Total porch and deck area
- `YrBltAndRemod = YearBuilt + YearRemodAdd`: Combined build and remodel year, representing house age

To strengthen the model's understanding of specific home features, we also add **binary flags**:


| Feature Name	    | Description                        |
|-------------------|------------------------------------|
| `haspool`         | Whether the house has a pool       |
| `has2ndfloor`     | Whether it has a second floor      |
| `hasgarage`       | Whether it has a garage            |
| `hasbsmt`         | Whether it has a basement          |
| `hasfireplace`    | Whether it has a fireplace         |
| ......            | ......                             |

These features help the model distinguish more feature-rich homes, improving pricing accuracy.

---

### 🔁 3.5 Feature Transformation: Log & Square

We apply nonlinear transformations to certain numerical features to enhance the model’s ability to fit nonlinear relationships:

  - #### 📐 Log Transformation

To reduce skewness and compress extreme values, we applied the `log(1.01 + x)` transformation to several features with skewed distributions.

```bash
log_features = [
  'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
  'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
  'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
  'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
  'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',
  'ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF'
]
```
Each variable generates a new derived column with the `*_log` suffix. After transformation, the distributions become more centralized, which facilitates stable model training.



- #### ⏹️ Square Transformation

We further applied a square transformation to some key `*_log` features (e.g., area-related variables) to enhance the model’s ability to capture second-order relationships.

```bash
squared_features = [
  'YearRemodAdd', 'LotFrontage_log', 'TotalBsmtSF_log',
  '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
  'GarageCars_log', 'GarageArea_log'
]
```

New columns with the `*_sq` suffix were created to represent the squared features.

---

### 🧮 3.6 One-Hot Encoding for Categorical Features
We use `pd.get_dummies()` to perform **one-hot encoding** on all categorical variables, transforming them into boolean dummy variables:

```bash
all_features = pd.get_dummies(all_features).reset_index(drop=True)
# Remove duplicated column names if any
all_features = all_features.loc[:, ~all_features.columns.duplicated()]
# Re-split into training and testing sets
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
```

---

### 🔬 3.7 Revisiting Feature–SalePrice Relationships
To validate the effectiveness of our feature engineering, we re-visualize the relationship between transformed numerical features and the target variable SalePrice.

<p align="center">
  <img src="figure/13 - Visualize relationships between numeric training features and SalePrice (scatter plots).png" alt="Visualize relationships between numeric training features and SalePrice" width="1080px">
  <br>
  <b>Numerical training features against SalePrice</b>
</p>

This step confirms whether the engineered features exhibit stable correlations with the target variable and are meaningful inputs for the model.

---

## 4. Modeling & Ensembling

To achieve better prediction performance and robustness, we construct a set of diverse regression models and integrate them using **Stacking** and **Blending** techniques.


---

### 📦 4.1 Model Definition
The following models are included in our ensemble pipeline:

| Model Name            |Description                                                                     |
|-----------------------|--------------------------------------------------------------------------------|
| `LGBMRegressor`       |LightGBM — a fast and high-performance gradient boosting framework              |
| `XGBRegressor`        |XGBoost — a powerful boosting model widely used in competitions                 |
| `SVR`                 |Support Vector Regression — suitable for small to medium datasets               |
| `RidgeCV`             |Ridge Regression with built-in cross-validation                                 |
| `GradientBoosting`    |Gradient Boosting Trees with robust loss function                               |
| `RandomForest`        |Random Forest — ensemble of decision trees with strong anti-overfitting ability |
| `StackingCVRegressor` |Stacked regressor that combines multiple base models                            |

---

### ⚙️ 4.2 Model Initialization

Below is the full parameter configuration used to initialize each model:

<details> <summary>📋 Model Definition Code (Click to expand)</summary>

```python
# LightGBM
lightgbm = LGBMRegressor(
    objective='regression',
    num_leaves=6,
    learning_rate=0.01,
    n_estimators=7000,
    max_bin=200,
    bagging_fraction=0.8,
    bagging_freq=4,
    bagging_seed=8,
    feature_fraction=0.2,
    feature_fraction_seed=8,
    min_sum_hessian_in_leaf=11,
    verbose=-1,
    random_state=42
)

# XGBoost
xgboost = XGBRegressor(
    learning_rate=0.01,
    n_estimators=6000,
    max_depth=4,
    min_child_weight=0,
    gamma=0.6,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear',
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.00006,
    random_state=42
)

# SVR
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))

# RidgeCV
ridge_alphas = [...]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=6000,
    learning_rate=0.01,
    max_depth=4,
    max_features='sqrt',
    min_samples_leaf=15,
    min_samples_split=10,
    loss='huber',
    random_state=42
)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=1200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features=None,
    oob_score=True,
    random_state=42
)

# Stacking Regressor
stack_gen = StackingCVRegressor(
    regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
    meta_regressor=xgboost,
    use_features_in_secondary=True
)
```
</details>

<br>

We use `StackingCVRegressor` to build a **stacking ensemble**, combining the predictions of base models to improve accuracy:

```bash
stack_gen = StackingCVRegressor(
    regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
    meta_regressor=xgboost,
    use_features_in_secondary=True
)
```

On top of **stacking**, we also design a **blending strategy** using manually set weights to generate the final predictions.

---

### 🚨 4.3 Model Evaluation & Cross-Validation

We use **K-Fold Cross-Validation** combined with the `Root Mean Squared Logarithmic Error (RMSLE)` as our evaluation metric to assess model performance and generalization.

```bash
# Define RMSLE and cross-validated RMSE
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return rmse
```

Each model (`LightGBM`, `XGBoost`, `SVR`, `Ridge`, `Gradient Boosting` and `Random Forest`) is evaluated using its average score and standard deviation over multiple folds, and the training time is also recorded:

```bash
# Example: LightGBM model
scores = {}
start = time.time()
score = cv_rmse(lightgbm) # model score
end = time.time()
print("lightgbm: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['lgb'] = (score.mean(), score.std())
```

This evaluation process allows us to compare the performance and efficiency of all models, which helps guide our final ensembling strategy.

---

### 🔀 4.4 Model Training & Ensembling (Stacking + Blending)

After evaluating all base models, we proceed to fully train them and integrate their predictions using two ensembling strategies:

- 1️⃣ **Stacking**

We train a stacked regression model using `StackingCVRegressor`, which combines multiple base model predictions via a secondary meta-model:

```bash
# Train stacked model
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
```
- 2️⃣ **Blending**

We manually assign weights to each model and compute a weighted average of their predictions to form the final blended output:

```bash
def blended_predictions(X):
    # Define model weights
    ridge_coefficient = 0.1; svr_coefficient = 0.2; gbr_coefficient = 0.1; xgb_coefficient = 0.1; lgb_coefficient = 0.1; rf_coefficient = 0.05; stack_gen_coefficient = 0.35;
    return (
        ridge_coefficient * ridge_model_full_data.predict(X) +
        svr_coefficient * svr_model_full_data.predict(X) +
        gbr_coefficient * gbr_model_full_data.predict(X) +
        xgb_coefficient * xgb_model_full_data.predict(X) +
        lgb_coefficient * lgb_model_full_data.predict(X) +
        rf_coefficient * rf_model_full_data.predict(X) +
        stack_gen_coefficient * stack_gen_model.predict(np.array(X))
    )
```
We then compute the RMSLE score of the blended model on the training set:

```bash
blended_score = rmsle(train_labels, blended_predictions(X))
print(f"RMSLE score on train data: {blended_score}")
```

---

### 📈 4.5 Model Comparison & Visualization

We visualize the cross-validation scores of all models to compare their performance:

<p align="center">
  <img src="figure/14 - Visualize model score comparison (pointplot).png" alt="Visualize model score comparison" width="640px">
  <br>
  <b>Model score comparison</b>
</p>

The vertical axis `Score (RMSE)` represents the model’s prediction error. A lower score indicates better predictive accuracy and a closer fit between predicted and actual values.

Additionally, we visualize the fitting results of the blended model on the training set:

<p align="center">
  <img src="figure/15 - Visualize true vs predicted values on the training set (in log space).png" alt="Visualize true vs predicted values on the training set" width="640px">
  <br>
  <b>True vs predicted values</b>
</p>

In the plot, the red diagonal line represents ideal predictions. The closer the blue dots are to the red line, the more accurate the model’s predictions.

---

### 📤 4.6 Test Prediction & Submission Saving

We apply the final blended model to predict house prices on the test set, and use `np.expm1()` to reverse the log transformation applied earlier to the target variable `SalePrice`:

```bash
final_predictions = np.expm1(blended_predictions(X_test))
```

Then, we create a submission file with the predicted results:

```bash
submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": final_predictions
})
submission.to_csv(model_save_dir/"submission.csv", index=False) # Final prediction result
```

---

### 💾 4.7 Model Saving

To support deployment and reuse, we save all trained models using `joblib`:

```bash
joblib.dump(stack_gen_model, model_save_dir/"stack_gen_model.pkl")
joblib.dump(ridge_model_full_data, model_save_dir/"ridge_model.pkl")
joblib.dump(svr_model_full_data, model_save_dir/"svr_model.pkl")
joblib.dump(gbr_model_full_data, model_save_dir/"gbr_model.pkl")
joblib.dump(xgb_model_full_data, model_save_dir/"xgb_model.pkl")
joblib.dump(lgb_model_full_data, model_save_dir/"lgb_model.pkl")
joblib.dump(rf_model_full_data, model_save_dir/"rf_model.pkl")
```

These saved `.pkl` files can be easily loaded for future predictions or deployment:

```bash
loaded_model = joblib.load("models/stack_gen_model.pkl")
preds = loaded_model.predict(X_test)
```

---

## 🏆 Kaggle Submission Result

This project was submitted to the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview), achieving a **Top 15%** ranking on the public leaderboard.

<p align="center">
  <img src="figure/16 - Final ranking results.png" alt="Final ranking results" width="1080px">
  <br>
  <b>Final ranking results</b>
</p>

---

## 📄 License
This project is licensed under the [MIT License](LICENSE)。
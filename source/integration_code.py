## Integration code
# From In[1] to In[43], all necessary operations for the project will be processed and output at once.
# To ensure concise output, some output functionalities have been commented out.

# In[1]: Import libraries (Essentials + Plots + Models + Stats + Misc)
# Essentials
import numpy as np
import pandas as pd
import time
import os

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
from pathlib import Path

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
# Misc
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
pd.set_option('display.max_columns', None)
# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

# In[2]: Read the data
project_root = Path(__file__).resolve().parent.parent
model_save_dir = project_root/"data"
train = pd.read_csv(model_save_dir/'train.csv')
test = pd.read_csv(model_save_dir/'test.csv')
train.shape, test.shape

# In[3]: SalePrice distribution plot (original state)
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
# View the original distribution of SalePrice
sns.distplot(train['SalePrice'], color="b")
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.grid(True, linestyle='--')
plt.show()

# In[4]: Output the skewness and kurtosis of SalePrice
# Skewness measures symmetry, kurtosis measures peakedness of the distribution
# print("Skewness: %f" % train['SalePrice'].skew())
# print("Kurtosis: %f" % train['SalePrice'].kurt())

# In[5]: Visualize numerical features against SalePrice (scatter plots)
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms', 'Total_porch_sf', 'haspool', 'hasgarage', 'hasbsmt', 'hasfireplace']:
            continue
        numeric.append(i)
valid_features = [f for f in numeric if f in train.columns and not train[f].isnull().all()]
rows = 3
columns = (len(valid_features) + rows - 1) // rows
fig, axs = plt.subplots(rows, columns, figsize=(5 * columns, 6 * rows))
axs = axs.flatten()
for i, feature in enumerate(valid_features):
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train, ax=axs[i])
    axs[i].set_xlabel(f'{feature}', size=12)
    axs[i].set_ylabel("SalePrice", size=12)
    axs[i].tick_params(axis='x', labelsize=10)
    axs[i].tick_params(axis='y', labelsize=10)
    axs[i].legend(prop={'size': 8})
    axs[i].grid(True, linestyle='--')
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

# In[6]: Plot heatmap to observe correlation between features
# Compute correlation matrix using only numerical features
corr = train.select_dtypes(include=[np.number]).corr()
# Visualize the heatmap
plt.subplots(figsize=(15, 12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
plt.show()

# In[7]: Relationship between OverallQual and SalePrice (boxplot)
# Visualize relationship between OverallQual and SalePrice (high correlation)
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.grid(True, linestyle='--')
plt.show()

# In[8]: YearBuilt vs SalePrice relationship (boxplot)
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['YearBuilt'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--')
plt.show()

# In[9]: TotalBsmtSF vs SalePrice relationship (scatter plot)
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', alpha=0.3, ylim=(0, 800000))
plt.grid(True, linestyle='--')
plt.show()

# In[10]: LotArea vs SalePrice relationship (scatter plot)
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.3, ylim=(0, 800000))
plt.grid(True, linestyle='--')
plt.show()

# In[11]: GrLivArea vs SalePrice relationship (scatter plot)
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.3, ylim=(0, 800000))
plt.grid(True, linestyle='--')
plt.show()

# In[12]: Drop Id column
# Remove the Ids from train and test, as they are unique for each row and hence not useful for the model
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
train.shape, test.shape

# In[13]: Apply log(1 + x) transformation to make SalePrice more normally distributed
train["SalePrice"] = np.log1p(train["SalePrice"])

# In[14]: View transformed SalePrice distribution + fit normal distribution
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
# Fit a normal distribution curve
sns.distplot(train['SalePrice'], fit=norm, color="b")
# Output fitted parameters
(mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.grid(True, linestyle='--')
plt.show()

# In[15]: Remove outliers
# Drop obvious outlier samples
train.drop(train[(train['OverallQual'] < 5) & (train['SalePrice'] > 200000)].index, inplace=True)
train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)].index, inplace=True)
# Reset index
train.reset_index(drop=True, inplace=True)

# In[16]: Split training labels & merge train/test features for unified preprocessing
# Extract labels
train_labels = train['SalePrice'].reset_index(drop=True)
# Extract training set features
train_features = train.drop(['SalePrice'], axis=1)
# Keep test set unchanged
test_features = test
# Merge training and test features for unified preprocessing
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape

# In[17]: Detect missing value percentages & print the top 10 features with the most missing values
# Calculate the percentage of missing values for each column
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(data)
    missing_dict = {}
    for col in df_cols:
        missing_dict[col] = round(data[col].isnull().mean() * 100, 2)
    return missing_dict
# Get the top 10 features with the highest missing value ratio
missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
# print("Percent of missing data of fisrt 10 terms:")
# print(df_miss[0:10])

# In[18]: Visualize missing values distribution (using training set)
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean() * 100, 2)
missing = missing[missing > 0].sort_values()
missing.plot.bar(color="b")
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
plt.grid(True, linestyle='--')
plt.show()

# In[19]: Some features are essentially categorical and should be converted to strings
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)

# In[20]: Main function to fill in missing values
def handle_missing(features):
    # 1. For functional features, NA means 'Typ'
    features['Functional'] = features['Functional'].fillna('Typ')
    # 2. Use mode to fill in selected fields
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    # 3. For fields related to presence of certain house features, NA means absence
    features["PoolQC"] = features["PoolQC"].fillna("None")
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
    # 4. Fill LotFrontage with median grouped by Neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    # 5. Fill remaining categorical features with 'None'
    object_cols = [col for col in features.columns if features[col].dtype == 'object']
    features[object_cols] = features[object_cols].fillna('None')
    # 6. Fill numeric features with 0
    numeric_cols = [col for col in features.columns if features[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features
# Apply missing value handler to the combined dataset
all_features = handle_missing(all_features)

# In[21]: Check again whether all missing values have been filled
missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
# print("Percent of missing data:")
# print(df_miss) #  all outputs should show np.float64(0.0), meaning no missing values

# In[22]: Extract numerical feature column names
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)

# In[23]: Visualize the distribution of numerical features to detect outliers and skewness (Boxplot)
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[numeric], orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
plt.grid(True, linestyle='--')
plt.show()

# In[24]: Find features with high skewness (Skew > 0.5)
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
# print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)

# In[25]: Normalize skewed features using Box-Cox transformation (with error handling)
failed_boxcox_cols = []  # Store failed columns for Box-Cox transformation
for i in skew_index:
    try:
        all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
    except:
        failed_boxcox_cols.append(i)
        # print(f"skipped features that cannot be processed: {i}")

# In[26]: Visualize normalized skewed feature distributions
# Confirm that skewness has been mostly corrected
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[skew_index], orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)
plt.grid(True, linestyle='--')
plt.show()
# Comparison before and after normalization:
# Before: Many features (e.g. PoolArea) exhibit strong right skew, high left concentration, extreme outliers, long tails — problematic for modeling.
# After: Most originally skewed features become more symmetric, tails are shortened, distributions are more centralized, and outliers are reduced or more reasonable — helpful for model stability. Some skew still exists, but overall impact is much smaller.

# In[27]: Construct derived features based on existing columns
all_features['BsmtFinType1_Unf'] = 1 * (all_features['BsmtFinType1'] == 'Unf')
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1
all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
# Drop irrelevant features
all_features = all_features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
# Create area-related derived features
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']
all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] + all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + 0.5 * all_features['HalfBath'] + all_features['BsmtFullBath'] + 0.5 * all_features['BsmtHalfBath'])
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] + all_features['EnclosedPorch'] + all_features['ScreenPorch'] + all_features['WoodDeckSF'])
# Fix some zero or negative values to avoid issues in log/modeling
all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
# Create boolean flags for presence of certain house features
all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# In[28]: Apply log(1 + x) transformation on selected features to create new *_log features
def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01 + res[l])).values)
        res.columns.values[m] = l + '_log'
        m += 1
    return res
log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea', 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF', 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']
all_features = logs(all_features, log_features)

# In[29]: Apply square transformation on selected features to create new *_sq features
def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l] * res[l]).values)
        res.columns.values[m] = l + '_sq'
        m += 1
    return res
squared_features = ['YearRemodAdd', 'LotFrontage_log', 'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log', 'GarageCars_log', 'GarageArea_log']
all_features = squares(all_features, squared_features)

# In[30]: Apply one-hot encoding to categorical variables
all_features = pd.get_dummies(all_features).reset_index(drop=True)
# Remove duplicated column names (if any)
all_features = all_features.loc[:, ~all_features.columns.duplicated()]
# Re-split training and testing feature sets based on label size
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]

# In[31]: Visualize relationships between numeric training features and SalePrice (scatter plots)
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            continue
        numeric.append(i)
sns.color_palette("husl", 8)
valid_features = [f for f in numeric if f in train.columns and not train[f].isnull().all()]
columes = (len(valid_features) + 2) // 3
fig, axs = plt.subplots(3, columes, figsize=(5 * columes, 18))
axs = axs.flatten()
for i, feature in enumerate(valid_features):
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train, ax=axs[i])
    axs[i].set_xlabel(f'{feature}', size=12)
    axs[i].set_ylabel("SalePrice", size=12)
    axs[i].tick_params(axis='x', labelsize=10)
    axs[i].tick_params(axis='y', labelsize=10)
    axs[i].legend(prop={'size': 8})
    axs[i].grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# In[32]: Set up 12-fold cross-validation strategy
kf = KFold(n_splits=12, random_state=42, shuffle=True)

# In[33]: Define RMSLE evaluation metric and cross-validated RMSE function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return rmse

# In[34]: Define multiple regression models (including ensemble and stacking models)
# Initialize all models (LightGBM, XGBoost, SVR, Ridge, GBR, RF, StackingCV)
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
# Ridge Regression (RobustScaler)
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
# SVR (Support Vector Regression)
svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
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
# Stacked regression (base models + meta model)
stack_gen = StackingCVRegressor(
    regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
    meta_regressor=xgboost,
    use_features_in_secondary=True
)

# In[35]: Evaluate all models using cross-validation and record scores
scores = {}
# lightgbm
start = time.time()
score = cv_rmse(lightgbm)
end = time.time()
print("lightgbm: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['lgb'] = (score.mean(), score.std())
# xgboost
start = time.time()
score = cv_rmse(xgboost)
end = time.time()
print("xgboost: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['xgb'] = (score.mean(), score.std())
# SVR
start = time.time()
score = cv_rmse(svr)
end = time.time()
print("SVR: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['svr'] = (score.mean(), score.std())
# Ridge
start = time.time()
score = cv_rmse(ridge)
end = time.time()
print("Ridge: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['ridge'] = (score.mean(), score.std())
# Random Forest
start = time.time()
score = cv_rmse(rf)
end = time.time()
print("Random Forest: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['rf'] = (score.mean(), score.std())
# Gradient Boosting
start = time.time()
score = cv_rmse(gbr)
end = time.time()
print("Gradient Boosting: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['gbr'] = (score.mean(), score.std())

# In[36]: Fit all models using the full training set
print('Fitting Stacking Regressor...')
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
print('Fitting LightGBM...')
lgb_model_full_data = lightgbm.fit(X, train_labels)
print('Fitting XGBoost...')
xgb_model_full_data = xgboost.fit(X, train_labels)
print('Fitting SVR...')
svr_model_full_data = svr.fit(X, train_labels)
print('Fitting Ridge...')
ridge_model_full_data = ridge.fit(X, train_labels)
print('Fitting Random Forest...')
rf_model_full_data = rf.fit(X, train_labels)
print('Fitting Gradient Boosting...')
gbr_model_full_data = gbr.fit(X, train_labels)

# In[37]: Blended model prediction function (weighted average of multiple models)
def blended_predictions(X):
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

# In[38]: Compute blended model score on training set (RMSLE)
blended_score = rmsle(train_labels, blended_predictions(X))
scores['blended'] = (blended_score, 0)
print(f"RMSLE score on train data: {blended_score}")


# In[39]: Visualize model score comparison (pointplot)
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))
ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()],markers='o', linestyles='-')
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')
plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)
plt.title('Scores of Models', size=20)
plt.grid(True, linestyle='--')
plt.show()

# In[40]: Predict test set house prices using the blended model (reverse log transform)
# Use the blended model to predict on test features and apply inverse log (np.expm1)
final_predictions = np.expm1(blended_predictions(X_test))

# In[41]: Save prediction results as submission file (submission.csv)
project_root = Path(__file__).resolve().parent.parent
model_save_dir = project_root/"data"
submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": final_predictions
})
submission.to_csv(model_save_dir/"submission.csv", index=False)
print("Prediction results are saved as submission.csv")

# In[42]: Visualize true vs predicted values on the training set (in log space)
y_true = train_labels
y_pred = blended_predictions(X)
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red', linewidth=2)
plt.xlabel("Actual Log(SalePrice)", fontsize=14)
plt.ylabel("Predicted Log(SalePrice)", fontsize=14)
plt.title("Actual vs Predicted SalePrice (Log Space)", fontsize=16)
plt.grid(True, linestyle='--')
plt.show()

# In[43]: Save all trained models

# Get the root path of the project
project_root = Path(__file__).resolve().parent.parent
model_save_dir = project_root/"models"
os.makedirs(model_save_dir, exist_ok=True)
# Save the main model
joblib.dump(stack_gen_model, model_save_dir/"stack_gen_model.pkl")
joblib.dump(ridge_model_full_data, model_save_dir/"ridge_model.pkl")
joblib.dump(svr_model_full_data, model_save_dir/"svr_model.pkl")
joblib.dump(gbr_model_full_data, model_save_dir/"gbr_model.pkl")
joblib.dump(xgb_model_full_data, model_save_dir/"xgb_model.pkl")
joblib.dump(lgb_model_full_data, model_save_dir/"lgb_model.pkl")
joblib.dump(rf_model_full_data, model_save_dir/"rf_model.pkl")
print("Models have been saved！")
# Example of loading saved model (to be used elsewhere)
# loaded_model = joblib.load(model_save_dir/"stack_gen_model.pkl")
# predictions = loaded_model.predict(X_test)
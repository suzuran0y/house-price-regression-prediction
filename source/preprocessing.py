# Data preprocessing module, including the following key functionalities:
# 1. Detect and fill missing values;
# 2. Convert some categorical-type features to strings;
# 3. Remove outliers;
# 4. Log transform the target variable;
# 5. Merge training and test features for unified processing.

import numpy as np
import pandas as pd

def log_transform_target(train):
    # Apply log(1 + x) transformation to the target variable SalePrice to make it more normally distributed.
    train["SalePrice"] = np.log1p(train["SalePrice"])
    return train

def drop_outliers(train):
    # Remove outlier samples based on the combination of OverallQual and GrLivArea.
    train = train.drop(train[(train['OverallQual'] < 5) & (train['SalePrice'] > 200000)].index)
    train = train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)].index)
    train = train.reset_index(drop=True)
    return train

def split_features_labels(train, test):
    # Separate the SalePrice label and merge train/test features into all_features.
    train_labels = train['SalePrice'].reset_index(drop=True)
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test.copy()
    all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
    return train_labels, all_features

def convert_categorical_to_string(all_features):
    # Convert numerical fields that are actually categorical into string type.
    all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
    all_features['YrSold'] = all_features['YrSold'].astype(str)
    all_features['MoSold'] = all_features['MoSold'].astype(str)
    return all_features

def handle_missing(features):
    # Perform missing value imputation based on feature type.
    # 1. Functional fields: missing values imply 'Typ' (normal)
    features['Functional'] = features['Functional'].fillna('Typ')
    # 2. Fields filled with mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    # 3. NA means the feature is not present
    features["PoolQC"] = features["PoolQC"].fillna("None")
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
    # 4. Fill LotFrontage with the median within each Neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    # 5. Fill other categorical fields with 'None'
    object_cols = [col for col in features.columns if features[col].dtype == 'object']
    features[object_cols] = features[object_cols].fillna('None')
    # 6. Fill numerical fields with 0
    numeric_cols = [col for col in features.columns if features[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']]
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features

def check_missing(features):
    # Check the missing value ratio of all features.
    missing_percent = features.isnull().mean() * 100
    return missing_percent[missing_percent > 0].sort_values(ascending=False)
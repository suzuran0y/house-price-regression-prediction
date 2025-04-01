# Feature construction and transformation module, including the following key functionalities:
# 1. Box-Cox transformation (skewness correction);
# 2. Custom feature construction (derived composite features);
# 3. Logarithmic and square transformations;
# 4. One-hot encoding and dataset splitting.

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

def fix_skewness(all_features, skew_threshold=0.5):
    # Perform Box-Cox normalization on skewed features.
    # Return processed data and a list of features that couldn't be transformed.
    numeric_feats = all_features.select_dtypes(include=[np.number]).columns
    skewed_feats = all_features[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[skewed_feats > skew_threshold]
    skew_index = high_skew.index
    failed_boxcox = []
    for feature in skew_index:
        try:
            all_features[feature] = boxcox1p(all_features[feature], boxcox_normmax(all_features[feature] + 1))
        except:
            failed_boxcox.append(feature)
    return all_features, failed_boxcox

def create_combined_features(all_features):
    # Construct domain-driven combined features and boolean flags.
    all_features['BsmtFinType1_Unf'] = (all_features['BsmtFinType1'] == 'Unf').astype(int)
    all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0).astype(int)
    all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0).astype(int)
    all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0).astype(int)
    all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0).astype(int)
    all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0).astype(int)
    all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
    all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
    all_features = all_features.drop(['Utilities', 'Street', 'PoolQC'], axis=1)
    all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
    all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']
    all_features['Total_sqr_footage'] = all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
    all_features['Total_Bathrooms'] = all_features['FullBath'] + 0.5 * all_features['HalfBath'] + all_features['BsmtFullBath'] + 0.5 * all_features['BsmtHalfBath']
    all_features['Total_porch_sf'] = all_features['OpenPorchSF'] + all_features['3SsnPorch'] + all_features['EnclosedPorch'] + all_features['ScreenPorch'] + all_features['WoodDeckSF']
    # Replace extreme values with small default values
    all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0 else x)
    all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0 else x)
    all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0 else x)
    all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0 else x)
    all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0 else x)
    all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0 else x)
    all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0 else x)
    # Binary flag features
    all_features['haspool'] = (all_features['PoolArea'] > 0).astype(int)
    all_features['has2ndfloor'] = (all_features['2ndFlrSF'] > 0).astype(int)
    all_features['hasgarage'] = (all_features['GarageArea'] > 0).astype(int)
    all_features['hasbsmt'] = (all_features['TotalBsmtSF'] > 0).astype(int)
    all_features['hasfireplace'] = (all_features['Fireplaces'] > 0).astype(int)
    return all_features

def apply_log_transform_features(all_features, feature_list):
    # Apply log(1.01 + x) transformation to specified features to generate *_log derived features.
    for f in feature_list:
        all_features[f + '_log'] = np.log1p(all_features[f] + 0.01)
    return all_features

def apply_square_transform_features(all_features, feature_list):
    # Apply square transformation to specified features to generate *_sq derived features.
    for f in feature_list:
        all_features[f + '_sq'] = all_features[f] ** 2
    return all_features

def one_hot_encode(all_features, train_labels):
    # Perform one-hot encoding and re-split training and test feature sets.
    all_features = pd.get_dummies(all_features).reset_index(drop=True)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    X = all_features.iloc[:len(train_labels), :]
    X_test = all_features.iloc[len(train_labels):, :]
    return X, X_test
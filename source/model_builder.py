# Model building module
# Initialize various base models (LightGBM, XGBoost, SVR, Ridge, RF, GBR);
# Define the function for building the StackingCVRegressor ensemble model.

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

def define_base_models(kf):
    # Define multiple regression models and return as a dictionary.
    # Ridge Regression + Automatic hyperparameter tuning
    ridge_alphas = [
        1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4,
        1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18,
        20, 30, 50, 75, 100
    ]
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
    # SVR (Support Vector Regression)
    svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
    # LightGBM model
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
    # XGBoost model
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
    return {
        'ridge': ridge,
        'svr': svr,
        'lightgbm': lightgbm,
        'xgboost': xgboost,
        'gbr': gbr,
        'rf': rf
    }

def build_stacking_model(base_models, meta_model):
    # Build a stacking model as an ensemble.
    # Parameters:
    #     base_models (dict): basic model dictionary
    #     meta_model: meta-model (e.g., xgboost)
    # return:
    #     stack_model (StackingCVRegressor): stacked regressor
    stack_model = StackingCVRegressor(
        regressors=(
            base_models['xgboost'],
            base_models['lightgbm'],
            base_models['svr'],
            base_models['ridge'],
            base_models['gbr'],
            base_models['rf']
        ),
        meta_regressor=meta_model,
        use_features_in_secondary=True
    )
    return stack_model
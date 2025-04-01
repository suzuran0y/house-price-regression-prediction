# Main control module that orchestrates all components and runs the code.

import numpy as np
from sklearn.model_selection import KFold
from data_loader import load_data
from preprocessing import (
    log_transform_target, drop_outliers, split_features_labels,
    convert_categorical_to_string, handle_missing
)
from feature_engineering import (
    fix_skewness, create_combined_features,
    apply_log_transform_features, apply_square_transform_features,
    one_hot_encode
)
from model_builder import define_base_models, build_stacking_model
from model_training import evaluate_models, train_all_models, rmsle
from model_fusion import train_stacking_model, blended_predictions
from utils import save_models, save_submission, plot_model_comparison, plot_prediction_vs_actual
from pathlib import Path

# Get the project root path
project_root = Path(__file__).resolve().parent.parent
model_save_dir = project_root/"models"

def main():
    # Step 1: Load the data
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"
    train, test, train_ID, test_ID = load_data(
        train_path=str(data_path / "train.csv"),
        test_path=str(data_path / "test.csv")
    )
    # Step 2: Transform target variable + Remove outliers
    train = log_transform_target(train)
    train = drop_outliers(train)
    # Step 3: Extract labels and features
    train_labels, all_features = split_features_labels(train, test)
    all_features = convert_categorical_to_string(all_features)
    all_features = handle_missing(all_features)
    # Step 4: Feature engineering
    all_features, _ = fix_skewness(all_features)
    all_features = create_combined_features(all_features)
    # Step 5: Feature transformation (log & square)
    log_features = [
        'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
        'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
        'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
        'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF',
        'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
        'MiscVal','YearRemodAdd','TotalSF'
    ]
    all_features = apply_log_transform_features(all_features, log_features)
    squared_features = [
        'YearRemodAdd', 'LotFrontage_log', 'TotalBsmtSF_log',
        '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
        'GarageCars_log', 'GarageArea_log'
    ]
    all_features = apply_square_transform_features(all_features, squared_features)
    # Step 6: One-hot encoding + Re-split features
    X, X_test = one_hot_encode(all_features, train_labels)
    # Step 7: Model definition and training
    kf = KFold(n_splits=12, random_state=42, shuffle=True)
    base_models = define_base_models(kf)
    scores = evaluate_models(base_models, X, train_labels, kf)
    trained_models = train_all_models(base_models, X, train_labels)
    # Step 8: Model fusion (stacking + blending)
    stack_model = build_stacking_model(base_models, base_models['xgboost'])
    stack_model = train_stacking_model(stack_model, X, train_labels)
    final_log_preds = blended_predictions(X_test, trained_models, stack_model)
    # Step 9: Save submission results
    final_predictions = np.expm1(final_log_preds)  # Inverse transformation of log1p
    save_submission(test_ID, final_predictions)
    save_models({**trained_models, 'stack': stack_model}, model_save_dir)
    # Step 10: Visualization
    plot_model_comparison(scores)
    plot_prediction_vs_actual(train_labels, blended_predictions(X, trained_models, stack_model))

if __name__ == "__main__":
    main()
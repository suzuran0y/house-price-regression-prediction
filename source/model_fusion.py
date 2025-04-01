# Model fusion module
# Train the StackingCVRegressor model;
# Use Blending strategy to combine model predictions.

import numpy as np

def train_stacking_model(stacking_model, X, y):
    # Fit a stacking model (StackingCVRegressor).
    # Parameters:
    #     stacking_model: Unfitted stacking model
    #     X: Training features
    #     y: Labels
    # Returns:
    #     Fitted stacking model
    print("Fitting Stacking Regressor...")
    return stacking_model.fit(np.array(X), np.array(y))

def blended_predictions(X, trained_models, stack_model, weights=None):
    # Compute prediction results using weighted Blending.
    # Parameters:
    #     X (DataFrame): Test set features
    #     trained_models (dict): Trained models (including ridge, svr, gbr, xgb, lgb, rf)
    #     stack_model: Trained stacking model
    #     weights (dict): Optional, custom weight dictionary
    # Returns:
    #     Blended prediction results (np.array)
    if weights is None: # Default blending weights (sum should be 1)
        weights = {
            'ridge': 0.1,
            'svr': 0.2,
            'gbr': 0.1,
            'xgboost': 0.1,
            'lightgbm': 0.1,
            'rf': 0.05,
            'stack': 0.35
        }
    blend = (
            weights['ridge'] * trained_models['ridge'].predict(X) +
            weights['svr'] * trained_models['svr'].predict(X) +
            weights['gbr'] * trained_models['gbr'].predict(X) +
            weights['xgboost'] * trained_models['xgboost'].predict(X) +
            weights['lightgbm'] * trained_models['lightgbm'].predict(X) +
            weights['rf'] * trained_models['rf'].predict(X) +
            weights['stack'] * stack_model.predict(np.array(X))
    )
    return blend
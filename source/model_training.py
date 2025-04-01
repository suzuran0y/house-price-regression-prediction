# Model training and evaluation module, including the following key functionalities:
# 1. Define RMSLE evaluation function;
# 2. Define cross-validated RMSE computation function;
# 3. Evaluate and summarize all models;
# 4. Implement complete training logic for all models.

import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

def rmsle(y, y_pred):
    # Root Mean Squared Logarithmic Error (RMSLE) evaluation metric.
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X, y, kf):
    # Perform RMSE evaluation with K-Fold cross-validation.
    rmse = np.sqrt(-cross_val_score(
        model, X, y,
        scoring="neg_mean_squared_error",
        cv=kf
    ))
    return rmse

def evaluate_models(models, X, y, kf):
    # Evaluate each model’s cross-validation score and output the result with time taken.
    # Parameters:
    #     models (dict): Dictionary of model names and instances
    #     X (DataFrame): Feature matrix
    #     y (Series): Labels
    #     kf (KFold): K-fold cross-validation object
    # Returns:
    #     scores (dict): Model name -> (mean_score, std_dev)
    scores = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        start = time.time()
        score = cv_rmse(model, X, y, kf)
        end = time.time()
        print(f"{name}: {score.mean():.4f} ({score.std():.4f}) | Time: {end - start:.2f} sec")
        scores[name] = (score.mean(), score.std())
    return scores

def train_all_models(models, X, y):
    # Fit all models on the full training dataset.
    # Parameters:
    #     models (dict): Model names and instances
    #     X, y: Training data
    # Returns:
    #     trained_models (dict): Model name -> fitted model
    trained_models = {}
    for name, model in models.items():
        print(f"Fitting {name}...")
        trained_models[name] = model.fit(X, y)
    return trained_models
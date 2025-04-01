# Utility functions module, including the following:
# Save model files;
# Output prediction results as CSV;
# Model comparison visualization;
# Plotting fit performance (predicted vs actual).

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_models(model_dict, save_dir):
    # Save all trained models to the specified directory.
    # Parameters:
    #     model_dict (dict): Dictionary of trained models
    #     save_dir (str): Path to save the models
    os.makedirs(save_dir, exist_ok=True)
    for name, model in model_dict.items():
        path = os.path.join(save_dir, f"{name}_model.pkl")
        joblib.dump(model, path)
    print(f"All models saved to: {save_dir}")

def save_submission(test_ID, predictions, path="submission.csv"):
    # Output prediction results as a CSV file (for submission format).
    # Parameters:
    #     test_ID (Series): ID column of the test set
    #     predictions (array-like): Predicted SalePrice values
    #     path (str): Output file path
    submission = pd.DataFrame({
        "Id": test_ID,
        "SalePrice": predictions
    })
    submission.to_csv(path, index=False)
    print(f"Predictions saved to: {path}")

def plot_model_comparison(scores):
    # Visualize model score comparison (lower RMSE is better).
    sns.set_style("white")
    fig = plt.figure(figsize=(24, 12))
    ax = sns.pointplot(
        x=list(scores.keys()),
        y=[score for score, _ in scores.values()],
        markers='o',
        linestyles='-'
    )
    for i, score in enumerate(scores.values()):
        ax.text(i, score[0] + 0.002,
                '{:.6f}'.format(score[0]),
                horizontalalignment='left',
                size='large',
                color='black',
                weight='semibold')
    plt.ylabel('Score (RMSE)', size=20)
    plt.xlabel('Model', size=20)
    plt.title('Scores of Models', size=20)
    plt.grid(True, linestyle='--')
    plt.show()

def plot_prediction_vs_actual(y_true, y_pred):
    # Plot comparison between predicted and actual values (in log space).
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='royalblue')
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        '--', color='red', linewidth=2
    )
    plt.xlabel("Actual Log(SalePrice)", fontsize=14)
    plt.ylabel("Predicted Log(SalePrice)", fontsize=14)
    plt.title("Actual vs Predicted SalePrice (Log Space)", fontsize=16)
    plt.grid(True, linestyle='--')
    plt.show()
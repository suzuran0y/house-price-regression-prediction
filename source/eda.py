# Exploratory Data Analysis (EDA) module, visualizing the following data characteristics:
# 1. Distribution of the target variable;
# 2. Relationship between numerical features and SalePrice;
# 3. Heatmap of feature correlations;
# 4. Key features.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm

def plot_saleprice_distribution(train):
    # Visualize the original distribution of SalePrice and the fitted normal distribution.
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(8, 7))
    sns.distplot(train['SalePrice'], fit=norm, color="b")
    mu, sigma = norm.fit(train['SalePrice'])
    plt.legend(['Normal dist. ($\mu=$ {:.2f}, $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best')
    ax.set(title="SalePrice Distribution", xlabel="SalePrice", ylabel="Frequency")
    plt.grid(True, linestyle='--')
    plt.show()

def visualize_numeric_feature_relationships(train):
    # Draw scatter plots between all numerical features and SalePrice to observe correlation.
    numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    if 'SalePrice' in numeric:
        numeric.remove('SalePrice')
    valid_features = [f for f in numeric if not train[f].isnull().all()]
    rows = 3
    columns = (len(valid_features) + rows - 1) // rows
    fig, axs = plt.subplots(rows, columns, figsize=(5 * columns, 6 * rows))
    axs = axs.flatten()
    for i, feature in enumerate(valid_features):
        sns.scatterplot(x=feature, y='SalePrice', data=train, ax=axs[i], hue='SalePrice', palette='Blues')
        axs[i].set_title(feature)
        axs[i].grid(True, linestyle='--')
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(train):
    # Draw a heatmap of correlations between numeric features.
    corr = train.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_feature_vs_price(train, feature, kind='box'):
    # Visualize the relationship between a feature and SalePrice (boxplot or scatter plot).
    # Parameters:
    #     feature (str): name of the feature
    #     kind (str): figure type ('box' or 'scatter')
    data = pd.concat([train['SalePrice'], train[feature]], axis=1)
    plt.figure(figsize=(10, 6))
    if kind == 'box':
        sns.boxplot(x=feature, y="SalePrice", data=data)
    else:
        sns.scatterplot(x=feature, y="SalePrice", data=data)
    plt.title(f'{feature} vs SalePrice')
    plt.grid(True, linestyle='--')
    plt.show()

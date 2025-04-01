# 🏠 House Prices: Advanced Regression Techniques
（[US English](README.md) | [CN 中文](README_CN.md)）

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-brightgreen)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-lightgrey)](https://xgboost.readthedocs.io/)
[![mlxtend](https://img.shields.io/badge/mlxtend-0.22.0-purple)](http://rasbt.github.io/mlxtend/)

[![Python](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/downloads/)
 [![Miniconda](https://img.shields.io/badge/Anaconda-Miniconda-violet.svg)](https://docs.anaconda.net.cn/miniconda/install/)

>使用高级回归技术预测房价（结合模型堆叠和特征工程的 Kaggle 竞赛解决方案）

---

## ✨ 项目亮点

- 🏆 该项目在 Kaggle 竞赛中最佳排名 **Top 15%**
- 🔁 展示完整机器学习建模流程（EDA → 特征工程 → 多模型集成 → 输出可视化）
- 🧩 支持模块化与一体化两种运行方式，便于适配自定义模型
- ✒️ 项目结构清晰，便于复用与扩展，适合作为机器学习回归类项目的参考模板。

---

## 📃 项目概述

这是基于 Kaggle 竞赛 [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) 的房价预测项目。该比赛提供了一个包含 79 个特征的房屋销售数据集，目标是构建一个能尽可能准确预测房价（SalePrice）的模型。

该比赛是无限期进行的，其成绩采用 **“滚动排行榜”（rolling leaderboard）** 会随着参赛者提交新的结果，排名实时发生变化。参赛者可以随时参与。

在一体化代码[`integration_code.py`](source/integration_code.py)基础上优化为模块化架构，实现了从数据预处理、特征工程、模型训练、融合预测到结果可视化的全流程。

---

## 🎯 核心目标

- 对房屋属性进行分析和建模，预测其最终售价。
- 使用 `Root Mean Squared Logarithmic Error (RMSLE)` 作为评估指标。
- 构建鲁棒、可泛化的集成模型架构，减少过拟合风险。

---

## 📂 项目结构

本项目采用模块化架构实现了从数据预处理、特征工程、模型训练、融合预测到结果可视化的全流程。项目结构清晰，便于复用与扩展，适合作为机器学习回归类项目的参考模板。

```bash
house-price-regression-prediction/
├── data/                    # 📚 原始数据 & 中间处理结果 & 最终预测输出
├── figure/                  # 📈 EDA & 模型评估图像（用于 README 插图）
├── models/                  # 🧠 保存训练好的模型（.pkl 格式）
│   ├── ridge_model.pkl
│   ├── xgb_model.pkl
│   └── ...（共 7 个模型）
├── source/                  # 🧩 一体化代码与模块化代码文件
│   │   ⬇ 一体化代码          # 单一完整代码：一体化运行程序
│   ├── integration_code.py     
│   │   ⬇ 模块化代码          # 多项独立代码：模块化运行程序
│   ├── main.py              # 主程序入口（按模块顺序调度）
│   └── ...（共 9 个模块）    ...
│── requirements.txt         # 📦 依赖列表
└── README.md                # 📄 项目文档
```

---

## 🧱 模块化设计

为了提升项目的可读性、可维护性与复用性，我们将完整的模型流程划分为多个功能模块，遵循“单一职责”的设计原则。各模块文件均位于 `source/` 文件夹下，主程序为 `main.py`，用于统一调度整个预测流程。

各模块功能如下所示：

| 模块文件                  | 功能说明 |
|--------------------------|---------|
| `data_loader.py`         | 加载训练集和测试集，移除 `Id` 字段，返回原始数据和对应 ID |
| `eda.py`                 | 探索性数据分析，包括分布图、散点图、热力图、特征可视化 |
| `preprocessing.py`       | 异常值处理、缺失值填补、目标变量对数变换与数据整合 |
| `feature_engineering.py` | 偏态矫正、组合特征构造、对数/平方变换、类别编码等 |
| `model_builder.py`       | 定义基础模型与堆叠模型，包括 LGBM、XGBoost、SVR 等 |
| `model_training.py`      | 模型评估指标计算、交叉验证、训练函数封装 |
| `model_fusion.py`        | 模型融合策略（Stacking 与 Blending 权重集成） |
| `utils.py`               | 保存模型与结果、绘制评估图、提交结果保存工具 |
| `main.py`                | 主控制流程，按步骤串联所有模块，输出预测与可视化 |

每个模块文件都具备明确功能接口，可以独立调用与测试，方便后期扩展（如替换模型、添加新特征等）。

---

## 🚀 快速开始

你可以按照以下步骤克隆本项目并运行：

```bash
# 1. 克隆项目仓库
git clone https://github.com/suzuran0y/house-price-regression-prediction.git
cd house-price-regression-prediction
# 2. 创建虚拟环境并安装依赖
conda create -n house_price_prediction python=3.10
conda activate house_price_prediction
pip install -r requirements.txt
# 3. 运行主程序
# python source/integration_code.py    #（一体化版本）
# python source/main.py                #（模块化版本）
```

数据文件 (`train.csv` / `test.csv`) 储存在 `data/` 文件夹中。输出结果将自动保存到 `models/` 和 `data/` 中。

---

## 🍏 项目功能介绍

基于一体化代码 [`integration_code.py`](source/integration_code.py)，我们将项目分为了4块内容：`数据读取与探索性数据分析`、`缺失值处理与数据清洗`、`特征变换与构造` 和 `模型构建与集成`。

相较于模块化代码，前者展现了对原始数据的每一步分析与可视化过程（代码中为了减少输出内容而对部分分析结果的输出功能进行了注释），更为贴近实际思考过程。

---

## 1. 数据读取与探索性数据分析（EDA）

我们首先加载训练集和测试集数据，并对目标变量 `SalePrice` 的分布情况和特征间的关系进行初步可视化分析。

---

### 📈 1.1 SalePrice 分布

目标变量 `SalePrice` 明显右偏，不服从正态分布，因此后续我们将对其进行对数变换以便于建模。
<p align="center">
  <img src="figure/1 - SalePrice distribution (original state).png" alt="SalePrice distribution (original state)" width="360px">
  <br>
  <b>SalePrice 原始分布</b>
</p>

此外我们也计算了 `SalePrice` 的偏度（`Skewness`）和峰度（`Kurtosis`）来量化其非正态性。

### 📊 1.2 数值型特征关系可视化

将数据集中所有数值型特征与 `SalePrice` 进行散点图可视化，观察其相关性和异常值：

<p align="center">
  <img src="figure/2 - Visualize numerical features against SalePrice (scatter plots).png" alt="Visualize numerical features against SalePrice" width="1080px">
  <br>
  <b>数值特征与 SalePrice 的关系</b>
</p>

### 🔥 1.3 特征相关性热力图

通过相关系数矩阵和热力图直观显示 `SalePrice` 与各数值型特征的线性相关性，发现如 `OverallQual`, `GrLivArea`, `TotalBsmtSF` 等与价格高度相关。

<p align="center">
  <img src="figure/3 - Plot heatmap of correlation between features.png" alt="Plot heatmap of correlation between features" width="360px">
  <br>
  <b>特征间的相关性</b>
</p>

### 📦 1.4 部分重要特征的分布分析

进一步分析影响房价的重要变量，如房屋整体质量（`OverallQual`）、建造年份（`YearBuilt`）、地上面积（`GrLivArea`）与房价之间的关系。

- `YearBuilt` vs `SalePrice`
<p align="center">
  <img src="figure/5 - YearBuilt vs SalePrice relationship (boxplot).png" alt="YearBuilt vs SalePrice relationship (boxplot)" width="640px">
  <br>
  <b>YearBuilt 与 SalePrice 的关系</b>
</p>

- `OverallQual` vs `SalePrice` (boxplot) and  `GrLivArea` vs `SalePrice` (scatter plot)
<table style="margin: 0 auto;">
  <tr>
    <td align="center">
      <img src="figure/4 - Relationship between OverallQual and SalePrice (boxplot).png"  alt="Relationship between OverallQual and SalePrice (boxplot)" width="360px"><br>
      <b>OverallQual 与 SalePrice 的关系</b>
    </td>
    <td align="center">
      <img src="figure/8 - GrLivArea vs SalePrice relationship (scatter plot).png" alt="GrLivArea vs SalePrice relationship (scatter plot)" width="360px"><br>
      <b>GrLivArea 与 SalePrice 的关系</b>
    </td>
  </tr>
</table>

---

## 2. 缺失值处理与数据清洗

在训练和测试数据中，有部分特征存在缺失值。我们首先统计每个特征的缺失比例，并绘图展示，随后根据字段含义和数据逻辑选择合适的填补方式。

---

### 📊 2.1 缺失值可视化

通过可视化每个特征的缺失数据比例，了解数据的缺失程度和分布情况。

<p align="center"> 
  <img src="figure/10 - Visualize missing values distribution.png" alt="Visualize missing values distributionn" width="360px">
  <br>
  <b>缺失值分布</b>
</p>

---

### 🛠️ 2.2 缺失值填补策略

我们针对不同类型的特征采用了不同的填补方式：

- **分类变量**：
  - `Functional`: 缺失表示正常（`Typ`）
  - `Electrical`, `KitchenQual`, `Exterior1st/2nd`, `SaleType`: 使用众数填补
  - `MSZoning`: 分组后按众数填充（基于 `MSSubClass`）
  - 车库和地下室相关特征（如 `GarageType`, `BsmtQual`）等：NA 表示无车库，填为 `'None'`

- **数值变量**：
  - `GarageYrBlt`, `GarageArea`, `GarageCars`: 缺失填 0
  - `LotFrontage`: 按 `Neighborhood` 分组后使用中位数填补
  - 其余数值特征统一填充为 0

- **特殊处理**：
  - 将 `MSSubClass`, `YrSold`, `MoSold` 等看作类别变量，转为字符串
  - 最终检查确保所有缺失值均已处理完成

---

### ✂️ 2.3 ID 去除与目标变量变换

我们去除了 `Id` 字段，因为它是样本唯一标识，对预测没有实际意义。同时，对目标变量 `SalePrice` 应用了 `log(1 + x)` 变换，以减少右偏、提升模型的稳定性。

<p align="center"> 
  <img src="figure/9 - View transformed SalePrice distribution + fit normal distribution.png" alt="Log-transformed SalePrice distribution" width="360px">
  <br>
  <b>SalePrice 对数变换后的分布</b>
</p>

---

### 🚫 2.4 异常值处理

通过散点图和逻辑判断，手动移除了明显离群值：

- `OverallQual < 5` 却售价高于 200000（质量与价格不符）
- `GrLivArea > 4500` 但售价低于 300000（面积巨大但售价反常）

这些样本可能干扰模型训练，因此剔除。

---

### 🔗 2.5 合并训练集与测试集（用于统一预处理）

- 提取训练集的标签 `SalePrice`
- 将训练集特征与测试集特征合并为 `all_features`

如此处理便于统一执行后续处理流程（如编码、归一化、特征构造等），为后续特征工程奠定了结构基础。

---
## 3. 特征变换与构造（Feature Engineering）

特征工程是本项目的核心部分之一，目的是增强模型感知特征之间复杂关系的能力，提高预测性能与泛化能力。

---

### 📝 3.1 特征整合

为统一处理方式，我们将训练集和测试集的特征进行拼接（不含标签），构建总特征矩阵 `all_features`：

```bash
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
```

---

### 🔬 3.2 数值特征偏态分析与校正
模型在面对高度偏态（**skewed**）的数据时性能较差，因此我们首先找出偏态值较大的特征（`skewness > 0.5`），随后使用 **Box-Cox 变换**校正其分布：

```bash
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
```

- **归一化前**： 许多特征（如 `PoolArea`）表现出明显的右偏态，数据大多集中在左侧，存在极端离群点，长尾现象严重 —— 这对建模非常不利。

- **归一化后**： 大多数原本偏态严重的特征变得更加对称，长尾变短，分布更集中，离群点也减少或变得更加合理 —— 这有助于模型的稳定性。虽然部分偏态仍然存在，但整体影响已显著减小。

<div align="center">
<table>
  <tr>
    <td align="center">
      <img src="figure/11 - Visualize the distribution of numerical features (Boxplot).png" alt="distribution of numerical features (Boxplot)" width="360px"><br>
      <b>数值特征分布（归一化前）</b>
    </td>
    <td align="center">
      <img src="figure/12 - Visualize normalized skewed feature distributions.png" alt="Numerical features distribution" width="360px"><br>
      <b>数值特征分布（归一化后）</b>
    </td>
  </tr>
</table>
</div>

---

### 🧱 3.3 构造组合型派生特征

除了原始变量外，我们基于领域知识与常识，设计了一些组合型变量，增强模型捕捉房屋整体结构、面积、质量等方面的能力。

主要构造特征如下：

- `Total_Home_Quality = OverallQual + OverallCond`：表示房屋综合质量评分；
- `YearsSinceRemodel = YrSold - YearRemodAdd`：房屋翻新后至今的年限；
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`：房屋总面积；
- `Total_sqr_footage = BsmtFinSF1 + BsmtFinSF2 + 1stFlrSF + 2ndFlrSF`：有效建筑面积；
- `Total_Bathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`：综合卫浴数量；
- `Total_porch_sf = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF`：门廊/露台总面积；
- `YrBltAndRemod = YearBuilt + YearRemodAdd`：建造与翻新时间合并，用于体现房屋整体年份属性。

此外，为了增强模型对“是否存在某项功能”的判断力，我们引入了一系列二值标记变量：

| 特征名            | 含义描述               |
|-------------------|------------------------|
| `haspool`         | 是否有游泳池           |
| `has2ndfloor`     | 是否有二楼             |
| `hasgarage`       | 是否有车库             |
| `hasbsmt`         | 是否有地下室           |
| `hasfireplace`    | 是否有壁炉             |
| ......    | ......            |

这些特征可以帮助模型更好地区分出功能丰富的房屋，从而提高对价格的判断能力。

---

### 🔁 3.5 特征转换：对数 & 平方变换

我们对部分数值型特征进行了非线性变换，以提升模型对复杂关系的拟合能力。

  - #### 📐 对数变换

为了减少偏态并压缩极端值，我们对多个具有偏态分布的特征应用了 `log(1.01 + x)` 变换：

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
每个变量将生成新的 `*_log` 派生列。变换后分布更集中，有助于模型稳定训练。

- #### ⏹️ 平方变换

我们进一步对部分关键的 `*_log` 特征应用平方变换（例如面积类变量），增强模型对二阶关系的拟合能力：

```bash
squared_features = [
  'YearRemodAdd', 'LotFrontage_log', 'TotalBsmtSF_log',
  '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
  'GarageCars_log', 'GarageArea_log'
]
```
生成新列 `*_sq` 表示平方特征。

---

### 🧮 3.6 类别变量编码（One-Hot Encoding）

使用 `pd.get_dummies()` 对所有类别变量执行 **One-Hot 编码**，生成布尔型的哑变量（dummy variables）：

```bash
all_features = pd.get_dummies(all_features).reset_index(drop=True)
# 确保删除重复列名：
all_features = all_features.loc[:, ~all_features.columns.duplicated()]
# 重新划分训练集与测试集：
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
```

---

### 🔬 3.7 特征与 SalePrice 的关系复检

为验证特征构造与转换是否有效，我们将构造后的数值特征重新与目标变量 SalePrice 进行可视化分析。

<p align="center">
  <img src="figure/13 - Visualize relationships between numeric training features and SalePrice (scatter plots).png" alt="Visualize relationships between numeric training features and SalePrice" width="1080px">
  <br>
  <b>训练集中数值特征与 SalePrice 的关系</b>
</p>

该步骤帮助确认模型输入与目标变量之间是否存在稳定关系。

---

## 4. 模型构建与集成

为了获得更优预测性能与稳健性，我们构建了多种回归模型，并使用 **堆叠（Stacking）** 和 **调和（Blending）** 技术进行集成。

---

### 📦 4.1 定义模型

我们引入的模型包括：

| 模型名                 |简要说明                             |
|-----------------------|-------------------------------------|
| `LGBMRegressor`       |LightGBM，梯度提升框架，速度快、性能好 |
| `XGBRegressor`        |XGBoost，性能优异的 boosting 模型    |
| `SVR`                 |支持向量回归，适用于中小型数据集       |
| `RidgeCV`             |岭回归，自动交叉验证调参               |
| `GradientBoosting`    |梯度提升树，加入鲁棒损失函数           |
| `RandomForest`        |随机森林，多树集成，抗过拟合能力强     |
| `StackingCVRegressor` |模型堆叠，提高预测性能                |

---

### ⚙️ 4.2 模型初始化

以下是模型初始化的完整参数配置：

<details>
<summary>📋 模型定义代码 （点击展开） </summary>

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

随后，我们使用 `StackingCVRegressor` 实现**模型堆叠（stacking）**：组合多个基模型预测最佳结果

```bash
stack_gen = StackingCVRegressor(
    regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
    meta_regressor=xgboost,
    use_features_in_secondary=True
)
```

在 **stacking** 的基础上，我们又设计了 **Blending 权重策略**，用于生成最终的预测。

---

### 🚨 4.3 模型评估与交叉验证

我们使用 **K折交叉验证（K-Fold Cross-Validation）** 结合 `Root Mean Squared Logarithmic Error (RMSLE)` 作为评估指标，比较各个模型的泛化能力。

```bash
# 定义交叉验证评估指标
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return rmse
```

随后分别评估了 `LightGBM`、`XGBoost`、`SVR`、`Ridge`、`Gradient Boosting`、`Random Forest` 等模型的平均得分与标准差，并记录模型的训练时间：

```bash
# 以LightGBM模型为例
scores = {}
start = time.time()
score = cv_rmse(lightgbm) # 模型得分
end = time.time()
print("lightgbm: {:.4f} ({:.4f}) | Time: {:.2f} sec".format(score.mean(), score.std(), end - start))
scores['lgb'] = (score.mean(), score.std())
```

通过评估得分与训练速度，我们可以对各个模型的表现进行比较分析，为融合建模提供依据。

---

### 🔀 4.4 模型训练与融合（Stacking + Blending）

我们在评估基础上，对所有模型进行完整训练，并通过两种方式进行模型融合：

- 1️⃣ **Stacking（堆叠）**
  
使用 `StackingCVRegressor` 对多个基础模型的输出进行二级建模，提升泛化能力：

```bash
# 训练Stacked regression模型
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))
```

- 2️⃣ **Blending（加权融合）**

人为设定权重，对各模型预测结果进行加权平均生成最终输出：

```bash
def blended_predictions(X):
    # 设定各模型权重
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

最终在训练集上计算融合模型的 RMSLE 得分：

```bash
blended_score = rmsle(train_labels, blended_predictions(X))
print(f"RMSLE score on train data: {blended_score}")
```

---

### 📈 4.5 模型对比与可视化

我们将所有模型在交叉验证下的得分进行了可视化比较：

<p align="center">
  <img src="figure/14 - Visualize model score comparison (pointplot).png" alt="Visualize model score comparison" width="640px">
  <br>
  <b>模型评分对比</b>
</p>

图中纵坐标 `Score(RMSE)` 表示模型预测的准确度，其数值越接近0表示模型预测值与实际值差异越小，即预测越准确。

此外，我们还绘制了融合模型在训练集上的拟合效果：

<p align="center">
  <img src="figure/15 - Visualize true vs predicted values on the training set (in log space).png" alt="Visualize true vs predicted values on the training set" width="640px">
  <br>
  <b>真实值与预测值对比</b>
</p>

图中红线表示理想预测，蓝点越接近红线表示模型预测越准确。

---

### 📤 4.6 测试集预测与结果保存

我们应用最终的加权融合模型（**blended model**）来预测测试集中的房价，并使用 `np.expm1()` 函数来还原之前对目标变量 `SalePrice` 所做的对数变换。

```bash
final_predictions = np.expm1(blended_predictions(X_test))
```

最终将预测结果保存为提交文件：

```bash
submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": final_predictions
})
submission.to_csv(model_save_dir/"submission.csv", index=False) # 最终预测结果
```

---

### 💾 4.7 模型保存

为便于部署与复用，我们使用 `joblib` 保存了全部模型文件：

```bash
joblib.dump(stack_gen_model, model_save_dir/"stack_gen_model.pkl")
joblib.dump(ridge_model_full_data, model_save_dir/"ridge_model.pkl")
joblib.dump(svr_model_full_data, model_save_dir/"svr_model.pkl")
joblib.dump(gbr_model_full_data, model_save_dir/"gbr_model.pkl")
joblib.dump(xgb_model_full_data, model_save_dir/"xgb_model.pkl")
joblib.dump(lgb_model_full_data, model_save_dir/"lgb_model.pkl")
joblib.dump(rf_model_full_data, model_save_dir/"rf_model.pkl")
```

保存后的模型可直接加载使用：

```bash
loaded_model = joblib.load("models/stack_gen_model.pkl")
preds = loaded_model.predict(X_test)
```

---

## 🏆 Kaggle 提交成绩

本项目基于 Kaggle 房价预测竞赛（[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)），提交后在公共排行榜上取得了 **Top 15%** 的成绩。

<p align="center">
  <img src="figure/16 - Final ranking results.png" alt="Final ranking results" width="1080px">
  <br>
  <b>最终成绩</b>
</p>

---

## 📄 许可证

本项目采用 [MIT License 许可证](LICENSE)。
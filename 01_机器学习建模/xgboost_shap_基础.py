# -*- coding: utf-8 -*-
# =============================================================================
# 功能: XGBoost 回归 + SHAP 分析
# 输入: Excel (第一列为 y, 后续列为 X)
# 输出: 控制台 MSE + SHAP summary_plot / force_plot / dependence_plot
# 使用: 修改 file_path 路径 → 设置 NORMALIZE=True/False → 运行
# 依赖: xgboost, shap, sklearn, pandas, matplotlib
# =============================================================================
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ========== 配置参数 ==========
NORMALIZE = False  # 是否对特征做归一化处理
file_path = r"F:\AAA\python\sample.xlsx"
sheet_name = "1990"
# ============================

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Step 1: 加载数据
data = pd.read_excel(file_path, index_col=0, sheet_name=sheet_name)

# Step 2: 数据预处理
target = data.columns[0]
X = data.iloc[:, 1:]
y = data[target]

if NORMALIZE:
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: 训练 XGBoost 模型
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "eta": 0.1,
    "seed": 42
}
model = xgb.train(params, dtrain, num_boost_round=100)

# Step 4: 模型评估
y_pred = model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.4f}")

# Step 5: SHAP 解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Step 6: 可视化
shap.summary_plot(shap_values, X_test)

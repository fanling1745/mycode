# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 7 种回归模型对比 (RF/GBDT/SVR/LR/KNN/XGBoost/LightGBM), 自动选最优
# 输入: Excel (第二列为 y, 后续列为 X)
# 输出: 各模型 R²/RMSE/MAE 排名 + 最优模型 pkl 文件 + 特征重要性
# 使用: 修改 excel_path → 运行
# 依赖: sklearn, xgboost, lightgbm, joblib, pandas, numpy
# =============================================================================
import pandas as pd
import numpy as np
import warnings
import joblib  # 用于保存模型

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance  # 用于解析黑盒模型特征重要性

# 导入各类机器学习模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# 尝试导入 XGBoost 和 LightGBM (空间预测的两大神器)
try:
    from xgboost import XGBRegressor

    has_xgb = True
except ImportError:
    has_xgb = False

try:
    from lightgbm import LGBMRegressor

    has_lgbm = True
except ImportError:
    has_lgbm = False

warnings.filterwarnings('ignore')

# ----------------- 1. 配置参数 -----------------
excel_path = r"H:\fang\浙江居民收入空间化\分区统计结果.xlsx"
model_save_path = r"H:\fang\浙江居民收入空间化\best_spatial_model.pkl"  # 模型保存路径


# ------------------------------------------------

def optimize_and_compare_models():
    print("正在加载与清洗数据...")

    try:
        df = pd.read_excel(excel_path, sheet_name="平均值")
    except Exception as e:
        print(f"[错误] 无法读取 Excel 文件。错误信息: {e}")
        return

    df = df.dropna()
    if len(df) < 10:
        print("[错误] 有效样本量过少，无法进行机器学习建模！")
        return

    y = df.iloc[:, 1].values
    y_name = df.columns[1]
    X = df.iloc[:, 2:]

    # 去掉“曲率”变量
    if '曲率' in X.columns:
        X = X.drop(columns=['曲率'])

    feature_names = X.columns.tolist()

    print(f"\n成功加载 {len(df)} 个有效样本。")
    print(f"目标变量: {y_name}")
    print(f"参与建模特征数: {len(feature_names)} 个\n")

    # 划分训练集和测试集 (70% 训练，30% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ================= 2. 定义模型字典与超参数网格 =================
    models = {
        "RandomForest (随机森林)": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))]),
            "param_grid": {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 10, 20]}
        },
        "GradientBoosting (梯度提升树)": {
            "estimator": Pipeline(
                [("scaler", StandardScaler()), ("model", GradientBoostingRegressor(random_state=42))]),
            "param_grid": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.01, 0.05, 0.1],
                           "model__max_depth": [3, 5]}
        },
        "SVR (支持向量机)": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
            "param_grid": {"model__kernel": ['rbf', 'linear'], "model__C": [0.1, 1.0, 10.0, 50.0]}
        },
        "Multiple Linear Regression (多元线性回归)": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "param_grid": {"model__fit_intercept": [True, False]}
        },
        "KNN (K近邻回归)": {
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
            "param_grid": {"model__n_neighbors": [3, 5, 7], "model__weights": ['uniform', 'distance']}
        }
    }

    # 动态添加 XGBoost
    if has_xgb:
        models["XGBoost (极端梯度提升)"] = {
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(random_state=42))]),
            "param_grid": {"model__n_estimators": [50, 100, 200], "model__max_depth": [3, 5],
                           "model__learning_rate": [0.01, 0.05, 0.1]}
        }
    else:
        print("[提示] 未检测到 xgboost 库，跳过该模型。")

    # 动态添加 LightGBM
    if has_lgbm:
        models["LightGBM (轻量级梯度提升)"] = {
            "estimator": Pipeline(
                [("scaler", StandardScaler()), ("model", LGBMRegressor(random_state=42, verbose=-1))]),
            "param_grid": {"model__n_estimators": [50, 100, 200], "model__num_leaves": [15, 31],
                           "model__learning_rate": [0.01, 0.05, 0.1]}
        }
    else:
        print("[提示] 未检测到 lightgbm 库，跳过该模型。")

    # ================= 3. 训练与评估 (10折交叉验证) =================
    results = []
    best_models_dict = {}

    print("\n开始进行模型训练与超参数优化 (10折交叉验证)，请耐心等待...\n")

    for model_name, config in models.items():
        print(f">>> 正在训练: {model_name}")

        # 【重要修改】：此处 cv=10 表示 10 折交叉验证
        grid_search = GridSearchCV(
            config["estimator"], config["param_grid"],
            cv=10, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        results.append({
            "模型名称": model_name, "最优参数": grid_search.best_params_,
            "R²": r2, "RMSE": rmse, "MAE": mae
        })
        best_models_dict[model_name] = best_model

    # ================= 4. 输出评估报告 =================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R²", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(" 【多模型测试集性能对比报告】")
    print("=" * 70)
    for i, row in results_df.iterrows():
        print(f"Top {i + 1}: {row['模型名称']}")
        print(f"   R²: {row['R²']:.4f} | RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f}")
    print("=" * 70)

    # ================= 5. 提取特征重要性 =================
    best_model_name = results_df.iloc[0]['模型名称']
    best_pipeline = best_models_dict[best_model_name]

    print(f"\n 【最佳模型 ({best_model_name}) 特征贡献度分析】")
    print("-" * 70)

    final_estimator = best_pipeline.named_steps['model']

    # 1. 优先尝试直接获取树模型的特征重要性
    if hasattr(final_estimator, 'feature_importances_'):
        importances = final_estimator.feature_importances_
        # 归一化到 100%
        importances = importances / importances.sum() * 100
        feat_imp = list(zip(feature_names, importances))
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        for idx, (feat, imp) in enumerate(feat_imp, start=1):
            print(f"Top {idx:02d} | 特征: {feat:<15} | 相对贡献度: {imp:>5.2f} %")

    # 2. 对于线性模型提取系数
    elif hasattr(final_estimator, 'coef_') and not isinstance(final_estimator, SVR):
        coefs = final_estimator.coef_
        if coefs.ndim > 1: coefs = coefs[0]
        feat_coef = list(zip(feature_names, coefs))
        feat_coef.sort(key=lambda x: abs(x[1]), reverse=True)
        for idx, (feat, coef) in enumerate(feat_coef, start=1):
            print(f"Top {idx:02d} | 特征: {feat:<15} | 系数大小(绝对值排序): {coef:>8.4f}")

    # 3. 对于核SVM、KNN等黑盒模型，使用 Permutation Importance (排列重要性)
    else:
        print("[采用排列重要性 (Permutation Importance) 解析特征影响度]")
        r = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=42)
        importances = r.importances_mean
        importances = np.maximum(importances, 0)
        if importances.sum() > 0:
            importances = importances / importances.sum() * 100

        feat_imp = list(zip(feature_names, importances))
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        for idx, (feat, imp) in enumerate(feat_imp, start=1):
            print(f"Top {idx:02d} | 特征: {feat:<15} | 影响度权重: {imp:>5.2f} %")

    print("=" * 70)

    # ================= 6. 保存用于网格预测的最佳模型 =================
    joblib.dump(best_pipeline, model_save_path)
    print(f" 已将表现最好的模型保存至: {model_save_path}")


if __name__ == "__main__":
    optimize_and_compare_models()
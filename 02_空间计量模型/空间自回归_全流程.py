# -*- coding: utf-8 -*-
# =============================================================================
# 功能: OLS → LM 诊断 → SAR → SEM → SDM 全流程空间计量建模
# 输入: Shapefile (含因变量和自变量字段)
# 输出: LM 检验结果 + 模型推荐 + SAR/SEM/SDM 估计 + Model_Comparison 表
# 使用: 修改 file_path / y_col / x_cols → 运行
# 依赖: libpysal, spreg, geopandas, scipy, numpy, pandas
# =============================================================================
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN
import spreg
from scipy.sparse import eye as sparse_eye

# ==============================================================================
# 1. 配置参数
# ==============================================================================
file_path = r"F:\项目\城乡聚落三维形态\数据\空间计量模型分析\样本文件\样本筛选12w.shp"
y_col = 'CS'  # 因变量
# 自变量列表 (9个)
x_cols = ['BCD', 'MBH', 'HBH', 'HBV', 'HCV', 'MBV', 'SPLIT', 'SHDI', 'LSI']

# ==============================================================================
# 2. 数据准备与权重构建
# ==============================================================================
print(f">>> 1. 读取数据: {file_path} ...")
gdf = gpd.read_file(file_path)
gdf = gdf.dropna(subset=[y_col] + x_cols).reset_index(drop=True)
N = len(gdf)
print(f"   样本量: {N}")

print(">>> 2. 构建空间权重 (KNN k=8)...")
coords = list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y))
w = KNN.from_array(coords, k=8)
w.transform = 'r'  # 行标准化
W_sparse = w.sparse

# 准备矩阵
y = gdf[y_col].values.reshape(-1, 1)
x = gdf[x_cols].values

# ==============================================================================
# 3. 第一步：OLS 回归与 LM 检验 (基准)
# ==============================================================================
print("\n>>> 3. 正在运行 OLS 并进行 LM 检验...")
# spat_diag=True 会自动计算 LM-Error, LM-Lag 等统计量
ols = spreg.OLS(y, x, w=w, name_y=y_col, name_x=x_cols, spat_diag=True)

# 提取 LM 检验结果
lm_data = {
    '检验指标': ['LM Error (误差)', 'LM Lag (滞后)', 'Robust LM Error', 'Robust LM Lag'],
    '统计量': [ols.lm_error[0], ols.lm_lag[0], ols.rlm_error[0], ols.rlm_lag[0]],
    'P值': [ols.lm_error[1], ols.lm_lag[1], ols.rlm_error[1], ols.rlm_lag[1]]
}
df_lm = pd.DataFrame(lm_data)
print(df_lm)

# 判断逻辑
print("-" * 50)
if ols.rlm_lag[1] < 0.05 and ols.rlm_error[1] < 0.05:
    print("【结论】 两个 Robust LM 均显著，统计上最支持 SDM 模型。")
elif ols.rlm_lag[1] < 0.05:
    print("【结论】 Robust LM Lag 显著，建议使用 SAR 模型。")
elif ols.rlm_error[1] < 0.05:
    print("【结论】 Robust LM Error 显著，建议使用 SEM 模型。")
print("-" * 50)

# ==============================================================================
# 4. 第二步：估计 SAR 模型 (空间滞后)
# ==============================================================================
print("\n>>> 4. 正在估计 SAR 模型 (GM_Lag)...")
# SAR: y = rho*Wy + X*beta + e
# 使用 GM 方法处理大样本
sar = spreg.GM_Lag(y, x, w=w, name_y=y_col, name_x=x_cols)
print(f"   SAR 计算完成! Rho: {sar.betas[-1][0]:.4f}, Pseudo R2: {sar.pr2:.4f}")

# ==============================================================================
# 5. 第三步：估计 SEM 模型 (空间误差)
# ==============================================================================
print("\n>>> 5. 正在估计 SEM 模型 (GM_Error)...")
# SEM: y = X*beta + u, u = lambda*Wu + e
# 使用 GM_Error_Het (允许异方差)
sem = spreg.GM_Error_Het(y, x, w=w, name_y=y_col, name_x=x_cols)
print(f"   SEM 计算完成! Lambda: {sem.betas[-1][0]:.4f}, Pseudo R2: {sem.pr2:.4f}")

# ==============================================================================
# 6. 第四步：估计 SDM 模型 (空间杜宾)
# ==============================================================================
print("\n>>> 6. 正在估计 SDM 模型 (TSLS 手动构造)...")
# 复用之前的 TSLS 逻辑以避免崩溃

# A. 构造变量
wy = (W_sparse @ y).reshape(-1, 1)
wx_list, w2x_list, valid_wx_names = [], [], []
W2_sparse = W_sparse @ W_sparse

for i, col in enumerate(x_cols):
    x_vec = x[:, i].reshape(-1, 1)
    wx_vec = (W_sparse @ x_vec)

    # 简单的共线性过滤
    if abs(np.corrcoef(x_vec.flatten(), wx_vec.flatten())[0, 1]) < 0.98:
        wx_list.append(wx_vec)
        w2x_list.append(W2_sparse @ x_vec)
        valid_wx_names.append(f"W_{col}")

x_sdm = np.hstack((x, np.hstack(wx_list)))
all_x_names = x_cols + valid_wx_names
q_inst = np.hstack(w2x_list)

# B. 运行 TSLS
sdm = spreg.TSLS(y, x_sdm, yend=wy, q=q_inst, name_y=y_col, name_x=all_x_names, name_yend=['W_CS'])
print(f"   SDM 计算完成! Rho: {sdm.betas[-1][0]:.4f}, Pseudo R2: {sdm.pr2:.4f}")

# ==============================================================================
# 7. 最终汇总与对比
# ==============================================================================
print("\n" + "=" * 60)
print("【全流程模型对比汇总】")
print("=" * 60)

summary_data = {
    '模型': ['OLS (基准)', 'SAR (滞后)', 'SEM (误差)', 'SDM (杜宾)'],
    '拟合优度 (R2)': [
        f"{ols.r2:.4f}",
        f"{sar.pr2:.4f}",
        f"{sem.pr2:.4f}",
        f"{sdm.pr2:.4f}"
    ],
    '空间系数': [
        "-",
        f"Rho: {sar.betas[-1][0]:.4f}",
        f"Lambda: {sem.betas[-1][0]:.4f}",
        f"Rho: {sdm.betas[-1][0]:.4f}"
    ],
    '自变量数量': [len(x_cols), len(x_cols), len(x_cols), len(all_x_names)]
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))
print("-" * 60)
print("注：对于大样本 GMM/TSLS 估计，使用 Pseudo R2 进行比较。")
print("    通常 SDM 的 R2 最高，且能提供溢出效应分解，故为最优选择。")

# 导出汇总
df_summary.to_csv("Model_Comparison_12w.csv", encoding='utf-8-sig', index=False)
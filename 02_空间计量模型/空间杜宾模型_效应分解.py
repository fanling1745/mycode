# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 空间杜宾模型 (SDM) + LeSage & Pace 直接/间接/总效应分解
# 输入: Shapefile
# 输出: 效应分解表 (系数/效应 + 显著性星号 + 间接效应占比) → SDM_Final_Report.csv
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
# 1. 参数配置 (请确认无误)
# ==============================================================================
file_path = r"F:\项目\城乡聚落三维形态\数据\空间计量模型分析\样本文件\样本筛选12w.shp"
y_col = 'CS'  # 因变量
x_cols = ['BCD', 'MBH', 'HBH', 'HBV', 'HCV', 'MBV', 'SPLIT', 'SHDI', 'LSI']  # 9个自变量

print(f">>> 1. 读取数据: {file_path} ...")
gdf = gpd.read_file(file_path)
gdf = gdf.dropna(subset=[y_col] + x_cols).reset_index(drop=True)
N = len(gdf)
print(f"   样本量: {N}")

# ==============================================================================
# 2. 构建模型所需矩阵
# ==============================================================================
print(">>> 2. 构建空间权重 (KNN k=8)...")
coords = list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y))
w = KNN.from_array(coords, k=8)
w.transform = 'r'  # 行标准化
W_sparse = w.sparse

y = gdf[y_col].values.reshape(-1, 1)
x = gdf[x_cols].values

print(">>> 3. 构造 SDM 变量 (X, WX, Wy, W^2X)...")
# 1. 内生变量 Wy
wy = (W_sparse @ y).reshape(-1, 1)

# 2. 自变量 WX 和 工具变量 W^2X
wx_list = []
w2x_list = []
valid_wx_names = []
W2_sparse = W_sparse @ W_sparse  # 二阶矩阵

for i, col in enumerate(x_cols):
    x_vec = x[:, i].reshape(-1, 1)
    # 计算 WX
    wx_vec = (W_sparse @ x_vec)
    # 简单的共线性过滤 (阈值 0.98)
    if abs(np.corrcoef(x_vec.flatten(), wx_vec.flatten())[0, 1]) < 0.98:
        wx_list.append(wx_vec)
        # 计算 W^2X (工具)
        w2x_list.append(W2_sparse @ x_vec)
        valid_wx_names.append(f"W_{col}")
    else:
        print(f"   [提示] 剔除高共线性变量: W_{col}")

x_sdm = np.hstack((x, np.hstack(wx_list)))
all_x_names = x_cols + valid_wx_names
q_inst = np.hstack(w2x_list)  # 工具变量矩阵

# ==============================================================================
# 3. 估计 SDM 模型 (TSLS)
# ==============================================================================
print(f">>> 4. 正在估计 SDM 模型 (变量数: {len(all_x_names)})...")
model = spreg.TSLS(y, x_sdm, yend=wy, q=q_inst, name_y=y_col, name_x=all_x_names, name_yend=['W_CS'])

rho = model.betas[-1][0]
r2 = model.pr2
print(f"   模型估计完成! Rho: {rho:.4f}, R2: {r2:.4f}")

# ==============================================================================
# 4. 效应分解 (核心步骤)
# ==============================================================================
print(">>> 5. 正在进行效应分解 (Direct, Indirect, Total)...")

# 准备系数
beta_map = dict(zip(['CONSTANT'] + all_x_names + ['W_CS'], model.betas.flatten()))
z_stat_map = dict(zip(['CONSTANT'] + all_x_names + ['W_CS'], model.z_stat))

# 准备近似计算所需的 Trace
# Direct ≈ Beta + (Theta + Beta*Rho)*tr(W)/N + (Beta*Rho^2 + Theta*Rho)*tr(W^2)/N
# 因为 tr(W)=0, 所以一阶项消失，只计算二阶项
tr_W2 = np.sum((W_sparse @ W_sparse).diagonal())

results = []

for var in x_cols:
    # --- 1. 获取系数 ---
    beta = beta_map.get(var, 0)
    theta = beta_map.get(f"W_{var}", 0)

    # 获取 P值 (用于判断显著性)
    p_beta = z_stat_map.get(var, [0, 1.0])[1]
    p_theta = z_stat_map.get(f"W_{var}", [0, 1.0])[1]

    # --- 2. 计算效应 (LeSage & Pace 近似法) ---
    # 总效应 Total = (Beta + Theta) / (1 - Rho)
    total_effect = (beta + theta) / (1 - rho)

    # 直接效应 Direct (保留二阶近似)
    # 修正项 = (Beta * Rho^2 + Theta * Rho) * (tr(W^2) / N)
    correction = (beta * (rho ** 2) + theta * rho) * (tr_W2 / N)
    direct_effect = beta + correction

    # 间接效应 Indirect
    indirect_effect = total_effect - direct_effect


    # --- 3. 格式化输出 (加星号) ---
    def star(p):
        if p < 0.01: return '***'
        if p < 0.05: return '**'
        if p < 0.1: return '*'
        return ''


    row = {
        '变量': var,
        # 回归系数部分
        '直接系数 (Beta)': beta,
        'Sig_Beta': star(p_beta),
        '滞后系数 (Theta)': theta,
        'Sig_Theta': star(p_theta),
        # 效应分解部分
        '直接效应 (Direct)': direct_effect,
        '间接效应 (Indirect)': indirect_effect,
        '总效应 (Total)': total_effect,
        # 占比分析 (可选)
        '间接效应占比(%)': (indirect_effect / total_effect) * 100 if abs(total_effect) > 1e-6 else 0
    }
    results.append(row)

# ==============================================================================
# 5. 生成最终表格并保存
# ==============================================================================
df_final = pd.DataFrame(results)

# 创建一个用于展示的“漂亮版”表格 (保留4位小数)
df_display = df_final.copy()
num_cols = ['直接系数 (Beta)', '滞后系数 (Theta)', '直接效应 (Direct)', '间接效应 (Indirect)', '总效应 (Total)']
for col in num_cols:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

# 合并星号显示 (例如: 0.555***)
df_display['直接系数'] = df_display['直接系数 (Beta)'] + df_display['Sig_Beta']
df_display['滞后系数'] = df_display['滞后系数 (Theta)'] + df_display['Sig_Theta']

# 整理列顺序
final_cols = ['变量', '直接系数', '滞后系数', '直接效应 (Direct)', '间接效应 (Indirect)', '总效应 (Total)',
              '间接效应占比(%)']
df_display = df_display[final_cols]

print("\n" + "=" * 80)
print(f"【空间杜宾模型 (SDM) 最终结果表】 N={N}, Rho={rho:.4f}***, R2={r2:.4f}")
print("=" * 80)
print(df_display.to_string(index=False))
print("-" * 80)

# 保存文件
file_name = "SDM_Final_Report_12w.csv"
df_display.to_csv(file_name, encoding='utf-8-sig', index=False)
print(f"表格已保存至: {file_name}")
print("提示：该文件可直接用 Excel 打开，复制到论文中即可。")
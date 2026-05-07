# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 读取地理探测器 CSV → Python 重绘学术风格交互热图
# 输入: result_dir 文件夹 (自动搜索 交互探测结果_*.csv)
# 输出: Python版_交互热图_{名称}.png (600 DPI, *双因子增强, **非线性增强)
# 使用: 修改 result_dir → 运行
# 依赖: pandas, numpy, matplotlib
# =============================================================================
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

# ==========================================
# 0. 基础设置与环境准备
# ==========================================
# 【请修改这里】：设定你之前保存“地理探测器结果”的总文件夹路径
result_dir = r"E:\数据分析\地理探测器\20260305\001_山山而川\结果\地理探测器结果"

# 设定热图的颜色配方 (与 R 代码完全一致)
colors = ["#0c5496", "#73a9d1", "#e3eff6", "#f49695", "#e63536"]
custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

# 字体精准控制：设定 Times New Roman 和 宋体 的 FontProperties
# (这里设置了备用字体，确保在 Windows/Mac 上都能成功调用)
font_tnr_text = FontProperties(family=['Times New Roman', 'serif'], size=10)
font_tnr_star = FontProperties(family=['Times New Roman', 'serif'], size=14)
font_tnr_axis = FontProperties(family=['Times New Roman', 'serif'], size=12)
font_st_caption = FontProperties(family=['SimSun', 'Songti SC', 'STSong', 'sans-serif'], size=11)

# ==========================================
# 1. 自动搜索并配对结果文件
# ==========================================
# 穿透子文件夹，找到所有交互探测的 CSV
inter_files = glob.glob(os.path.join(result_dir, "**", "交互探测结果_*.csv"), recursive=True)

if not inter_files:
    raise FileNotFoundError("未能在指定路径下找到任何 '交互探测结果_xxx.csv' 文件，请检查路径！")

print(f"\n发现 {len(inter_files)} 个结果文件，开始使用 Python 批量重绘热图...\n")

# ==========================================
# 2. 循环读取数据并重绘
# ==========================================
for inter_file in inter_files:
    # 提取纯粹的文件名标识
    file_name_only = os.path.basename(inter_file)
    base_name = file_name_only.replace("交互探测结果_", "").replace(".csv", "")

    # 推导因子探测文件路径
    factor_file = inter_file.replace("交互探测结果_", "因子探测结果_")

    if not os.path.exists(factor_file):
        print(f"警告：找不到 {base_name} 对应的单因子探测结果，跳过！")
        continue

    print(f"正在渲染并重绘: {base_name} ...")

    # --- 步骤 A：读取并解析 CSV 数据 ---
    df_factor = pd.read_csv(factor_file)
    df_inter = pd.read_csv(inter_file)

    # 自动识别变量名
    var_names = df_factor.iloc[:, 0].tolist()
    var_num = len(var_names)
    var_to_idx = {v: i for i, v in enumerate(var_names)}

    # 自动寻找 q 值的列名
    q_col_factor = [c for c in df_factor.columns if 'q' in c.lower()][0]

    # 自动寻找交互探测中的列名
    cols = df_inter.columns
    col_v1, col_v2 = cols[0], cols[1]
    col_q12 = [c for c in cols if '12' in c][0]
    col_type = [c for c in cols if 'inter' in c.lower()][0]

    # 构造空矩阵用于热图映射 (包含 NaN 值，用于留白)
    q_matrix = np.full((var_num, var_num), np.nan)
    symbols = {}

    # 填充单因子探测数据 (对角线)
    for _, row in df_factor.iterrows():
        v_idx = var_to_idx[row[df_factor.columns[0]]]
        q_matrix[v_idx, v_idx] = row[q_col_factor]

    # 填充交互探测数据 (非对角线，构建下三角矩阵)
    for _, row in df_inter.iterrows():
        v1_idx = var_to_idx[row[col_v1]]
        v2_idx = var_to_idx[row[col_v2]]

        # 确保 x >= y 以构建特定的下三角/上三角形状，与原 R 代码一致
        x_idx = max(v1_idx, v2_idx)
        y_idx = min(v1_idx, v2_idx)

        q_matrix[y_idx, x_idx] = row[col_q12]

        # 生成显著性星号
        stype = str(row[col_type]).lower()
        if 'nonlinear' in stype:
            symbols[(x_idx, y_idx)] = "**"
        elif 'bi' in stype:
            symbols[(x_idx, y_idx)] = "*"

    # --- 步骤 B：使用 Matplotlib 绘制热图 ---
    fig, ax = plt.subplots(figsize=(8, 7), dpi=600)

    # 设置网格边界，使用 pcolormesh 绘制方块（自动剔除 NaN 并绘制白色边界）
    x_edges = np.arange(0.5, var_num + 1.5)
    y_edges = np.arange(0.5, var_num + 1.5)

    im = ax.pcolormesh(x_edges, y_edges, q_matrix, cmap=custom_cmap,
                       edgecolors='white', linewidth=1)

    # 将坐标轴原点锁死，强制为 1:1 的正方形网格
    ax.set_aspect('equal')
    ax.set_xlim(0.5, var_num + 0.5)
    ax.set_ylim(0.5, var_num + 0.5)

    # 添加文字标注 (数字和星号)
    for i in range(var_num):  # i 等价于 y_idx
        for j in range(var_num):  # j 等价于 x_idx
            q_val = q_matrix[i, j]
            if not np.isnan(q_val):
                # 矩阵坐标到画板坐标的中心点映射
                cx = j + 1
                cy = i + 1

                # 绘制 Q 值数字 (Times New Roman)
                ax.text(cx, cy, f"{q_val:.4f}", ha='center', va='center',
                        fontproperties=font_tnr_text)

                # 绘制星号并调整偏移量 (Times New Roman)
                sym = symbols.get((j, i), "")
                if sym:
                    ax.text(cx + 0.28, cy + 0.34, sym, ha='center', va='center',
                            fontproperties=font_tnr_star)

    # 设置坐标轴外观
    ax.set_xticks(np.arange(1, var_num + 1))
    ax.set_xticklabels(var_names, fontproperties=font_tnr_axis)
    ax.set_yticks(np.arange(1, var_num + 1))
    ax.set_yticklabels(var_names, fontproperties=font_tnr_axis)
    ax.tick_params(axis='both', which='major', length=4, colors='black')

    # 为画板外侧添加黑色边框 (模仿 ggplot2 theme_minimal + panel.border)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

    # 绘制内部悬浮图例 (Colorbar)
    # [left, bottom, width, height]
    cbar_ax = ax.inset_axes([0.08, 0.60, 0.04, 0.30])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.8)

    # 图例标题与刻度的字体控制
    cbar.ax.set_title('q_value', fontproperties=font_tnr_text, pad=10)
    for t in cbar.ax.get_yticklabels():
        t.set_fontproperties(font_tnr_text)

    # 添加中文图注 (宋体)，放置于左下角
    caption_text = "注：*为双因子增强，**为非线性增强"
    ax.text(0, -0.12, caption_text, transform=ax.transAxes, ha='left', va='top',
            fontproperties=font_st_caption)

    # --- 步骤 C：保存图片 ---
    output_png = os.path.join(os.path.dirname(inter_file), f"Python版_交互热图_{base_name}.png")

    plt.savefig(output_png, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"✓ 已成功保存高精度字体热图至: {output_png}")

print("\n全部渲染完成！")
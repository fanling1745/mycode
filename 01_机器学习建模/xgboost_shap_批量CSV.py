# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 批量处理 CSV 文件夹, 逐文件训练 XGBoost + 生成 SHAP 图
# 输入: CSV 文件夹 (第一列为 y, 第 2-17 列为 X)
# 输出: {文件名}_shap.png + {文件名}_importance.png
# 使用: 修改 input_folder 和 output_folder → 运行
# 依赖: xgboost, shap, pandas, matplotlib
# =============================================================================
import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局可视化细节调整
# ==========================================
# 设置为学术常用字体 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 解决坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 设置输入和输出文件夹路径
# ==========================================
# 输入路径：存放 CSV 文件的文件夹
input_folder = r'E:\数据分析\地理探测器\20260318\一觉睡到小时候\GD'

# 输出路径：生成的图片保存位置
output_folder = r'E:\数据分析\地理探测器\20260318\一觉睡到小时候\GD\输出结果1'

# 检查并自动创建输出文件夹（如果不存在的话）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建输出文件夹：{output_folder}")

# 自动匹配输入文件夹下所有的 .csv 文件
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

if not csv_files:
    print(f"在 {input_folder} 中没有找到任何 CSV 文件，请检查路径。")
else:
    print(f"共找到 {len(csv_files)} 个 CSV 文件，开始批量处理...\n")

# ==========================================
# 2. 开始批量循环处理
# ==========================================
for file_path in csv_files:
    base_name = os.path.basename(file_path).replace('.csv', '')

    print("=" * 50)
    print(f"▶ 正在处理当前文件: {base_name}.csv")
    print("=" * 50)

    # --- 数据加载与清洗 ---
    print("  - 加载和清洗数据...")
    df = pd.read_csv(file_path, encoding='gbk')

    # 检查 y 和 x 中的 <空> 或 NaN，并删除整行
    df = df.replace({'<空>': np.nan, '': np.nan, ' ': np.nan})
    df = df.dropna()

    # 样本量控制：提升 SHAP 计算效率
    if len(df) > 200000:
        print(f"  - 数据量为 {len(df)}，超过 200,000，进行随机降采样...")
        df = df.sample(n=200000, random_state=42)

    # --- 变量划分与模型训练 ---
    # 第 0 列为目标变量 y，第 1 到 16 列为自变量 X
    y = df.iloc[:, 0].astype(float)
    X = df.iloc[:, 1:17].astype(float)

    print("  - 训练 XGBoost 模型...")
    model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    # --- SHAP 分析 ---
    print("  - 计算 SHAP 值...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # ==========================================
    # 可视化图 1：SHAP Detailed Summary (Beeswarm Plot)
    # ==========================================
    print("  - 生成并保存 SHAP Summary Plot...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    shap.summary_plot(shap_values.values, X, show=False, max_display=16)
    # ==========================================
    # 【新增这里】：修改 SHAP 图例（颜色条）的字体大小
    # 1. 获取当前图像的所有坐标轴
    cb_axes = plt.gcf().get_axes()
    # 2. SHAP 绘制的颜色条默认是最后一个坐标轴 (cb_axes[-1])
    if len(cb_axes) > 1:
        cb_ax = cb_axes[-1]
        # 修改侧边文字 "Feature value" 的字体大小（比如改为 15）
        cb_ax.set_ylabel("Feature value", fontsize=16)
        # 修改两端文字 "Low" 和 "High" 的字体大小（比如改为 14）
        cb_ax.tick_params(labelsize=14)

    plt.title(f'SHAP Detailed Summary', fontsize=18, pad=20)
    plt.xlabel('SHAP value (impact on model output)', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    fig1_path = os.path.join(output_folder, f'{base_name}_shap.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # ==========================================
    # 可视化图 2：全局特征重要性排序图 (按重要性排序)
    # ==========================================
    print("  - 生成并保存特征重要性排序图...")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = X.columns.tolist()

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    y_pos = np.arange(len(importance_df))

    # 【关键修复】显式设置 x 轴范围，为数据标签预留 10% 的缓冲空间，防止标注越界
    max_val = importance_df['Importance'].max()
    ax2.set_xlim(0, max_val * 1.1)

    ax2.barh(y_pos, importance_df['Importance'], align='center', color='#3b73a3', height=0.6, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(importance_df['Feature'], fontsize=14)
    ax2.set_xlabel('Mean |SHAP| (magnitude)', fontsize=14)
    ax2.set_title(f'Feature Importance', fontsize=16)
    ax2.tick_params(axis='x', labelsize=14)


    # 添加数据标签
    for i, v in enumerate(importance_df['Importance']):
        ax2.text(v + (max_val * 0.01), i, f"{v:.4f}", va='center', fontsize=12, color='black')

    plt.tight_layout()
    fig2_path = os.path.join(output_folder, f'{base_name}_importance.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"✔ 文件 {base_name}.csv 处理完成！\n")

print(f"所有文件处理完毕！图片已全部保存在：{output_folder}")
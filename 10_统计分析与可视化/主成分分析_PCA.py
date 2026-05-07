# -*- coding: utf-8 -*-
# =============================================================================
# 功能: PCA 降维 + Biplot 可视化 (散点图 + 特征向量箭头 + 解释方差表)
# 输入: Shapefile + 分析字段列表 + n_components
# 输出: 控制台解释方差比 + PCA Biplot 图
# 使用: 修改 input_path 和 fields → 运行
# 依赖: geopandas, sklearn, matplotlib, seaborn, pandas, numpy
# =============================================================================
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# ========== 1. 数据读取与预处理 ==========
def load_and_preprocess(input_path):
    """加载矢量数据并预处理"""
    gdf = gpd.read_file(input_path)

    # 选择分析字段
    fields = ["MBV", "VCV", "HBV", "MBH", "HCV", "HBH", "FRAC_MN", "IJI", "SPLIT", "BCD"]
    data = gdf[fields]

    # 处理缺失值（将N/A转为0）
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 筛选非全0行（可选，根据需求保留）
    mask_all_zero = (data == 0).all(axis=1)
    valid_data = data[~mask_all_zero]

    return gdf, data, valid_data, mask_all_zero


# ========== 2. 主成分分析 ==========
def perform_pca(data, n_components=3):
    """执行PCA分析"""
    # 标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # PCA分析
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled_data)

    return pca, pcs, scaler


# ========== 3. 可视化 ==========
def visualize_results(pca, pcs, fields):
    """可视化PCA结果"""
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 10))

    # 1. 主成分得分散点图
    scatter = sns.scatterplot(
        x=pcs[:, 0],
        y=pcs[:, 1],
        alpha=0.6,
        edgecolor='w',
        s=100
    )

    # 2. 特征向量箭头
    feature_vectors = pca.components_.T
    scale = 1.5 * np.max(np.abs(pcs))  # 动态缩放箭头长度

    for i, var in enumerate(fields):
        plt.arrow(0, 0,
                  feature_vectors[i, 0] * scale,
                  feature_vectors[i, 1] * scale,
                  color='red', alpha=0.8,
                  width=0.002 * scale,
                  head_width=0.02 * scale)
        plt.text(feature_vectors[i, 0] * scale * 1.15,
                 feature_vectors[i, 1] * scale * 1.15,
                 var, color='darkred', fontsize=12,
                 ha='center', va='center')

    # 3. 图形装饰
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA Biplot of Urban-Rural Settlement 3D Metrics", pad=20)

    # 4. 添加解释方差表格
    plt.table(cellText=[['PC' + str(i + 1), f'{var * 100:.1f}%']
                        for i, var in enumerate(pca.explained_variance_ratio_)],
              colLabels=['Component', 'Variance Explained'],
              loc='lower right',
              bbox=[0.7, 0.1, 0.25, 0.2])

    plt.tight_layout()
    return plt


# ========== 主程序 ==========
if __name__ == "__main__":
    # 1. 数据准备
    input_path = r"F:\项目\城乡聚落三维形态\数据\主成分分析\data\三维指标_CS_816.shp"
    gdf, data, valid_data, mask_all_zero = load_and_preprocess(input_path)

    # 2. 执行PCA
    n_components = 3
    pca, valid_pcs, scaler = perform_pca(valid_data, n_components)

    # 3. 合并结果
    pcs_full = np.zeros((data.shape[0], n_components))
    pcs_full[~mask_all_zero] = valid_pcs
    pc_df = pd.DataFrame(pcs_full, columns=[f"PC{i + 1}" for i in range(n_components)])
    gdf = pd.concat([gdf, pc_df], axis=1)

    # 4. 输出解释方差
    print("=== Explained Variance Ratio ===")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i + 1}: {ratio:.4f} ({ratio * 100:.1f}%)")

    # 5. 可视化
    plt = visualize_results(pca, valid_pcs, data.columns)

    # 6. 保存结果（可选）
    output_path = r"F:\项目\城乡聚落三维形态\数据\主成分分析\data\pca_results.shp"
    # gdf.to_file(output_path, encoding='utf-8')

    plt.show()
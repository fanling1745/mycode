# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 批量计算点/面数据的全局 Moran's I
# 输入: 存放 Shapefile 的文件夹 + target_field 字段名
# 输出: Moran_Results.csv (Moran's I / Z分数 / P值 / 显著性)
# 使用: 修改 input_folder 和 target_field → 运行
# 依赖: libpysal, esda, geopandas, pandas
# =============================================================================
import geopandas as gpd
import pandas as pd
import libpysal
from esda.moran import Moran
import os
import glob

# ================= 配置参数 =================
input_folder = r"E:\frag\20260204\空间自相关\局部空间自相关"  # 输入：存放所有Shapefile的文件夹路径
output_csv = "Moran_Results.csv"  # 输出：结果表格保存路径
target_field = "grid_code"  # 计算的目标字段名
k_neighbors = 8  # 定义邻居数量 (点数据常用KNN算法)

# ================= 主程序 =================

# 1. 获取所有 .shp 文件
shp_files = glob.glob(os.path.join(input_folder, "*.shp"))
results_list = []

print(f"找到 {len(shp_files)} 个文件，开始处理...")

for file_path in shp_files:
    file_name = os.path.basename(file_path)
    print(f"正在处理: {file_name} ...")

    try:
        # 2. 读取矢量数据
        gdf = gpd.read_file(file_path)

        # 检查字段是否存在
        if target_field not in gdf.columns:
            print(f"  [跳过] 字段 '{target_field}' 不存在")
            continue

        # 检查数据量是否足够计算 (至少需要 k+1 个点)
        if len(gdf) <= k_neighbors:
            print(f"  [跳过] 点数量不足 (仅 {len(gdf)} 个，需要 > {k_neighbors})")
            continue

        # 3. 创建空间权重矩阵 (使用KNN算法，适用于点数据)
        # k=8 是常用的默认值
        w = libpysal.weights.KNN.from_dataframe(gdf, k=k_neighbors)

        # 行标准化 (Row-standardization)，这对莫兰指数计算很重要
        w.transform = 'R'

        # 4. 计算全局莫兰指数
        # permutations=999 表示进行999次置换检验来计算P值
        y = gdf[target_field].values
        moran = Moran(y, w, permutations=999)

        # 5. 收集结果
        # 0.05 或 0.01 的 p-value 通常表示显著聚集
        significance = "显著" if moran.p_sim < 0.05 else "不显著"

        results_list.append({
            "文件名": file_name,
            "点数量": len(gdf),
            "Moran_Index": round(moran.I, 4),  # 莫兰指数 (-1 到 1)
            "Z_Score": round(moran.z_sim, 4),  # Z分数 (>1.96 或 <-1.96 表示显著)
            "P_Value": round(moran.p_sim, 4),  # 显著性水平
            "结果解释": significance
        })

    except Exception as e:
        print(f"  [错误] 处理失败: {e}")

# ================= 保存结果 =================
if results_list:
    df_result = pd.DataFrame(results_list)

    # 保存为 CSV
    df_result.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("-" * 30)
    print(f"处理完成！结果已保存至: {output_csv}")
    print(df_result)
else:
    print("未生成任何结果，请检查输入数据。")
# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 按人工区/自然区逐像元统计 NDVI (最大/最小/均值/标准差)
# 输入: NDVI 栅格文件夹 + 人工区矢量
# 输出: NDVI_Stats_Unified.csv
# 使用: 修改 ndvi_folder / artificial_shp_path → 运行
# 依赖: rasterio, geopandas, numpy, pandas, tqdm
# =============================================================================
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from tqdm import tqdm


def main():
    # ================= 1. 参数设置 =================
    # NDVI 栅格文件夹路径
    ndvi_folder = r"D:\GD\结果\裁剪NDVI"

    # 统一的人工区矢量文件路径 (请修改为具体的 .shp 文件路径)
    artificial_shp_path = r"D:\GD\更新结果\人工区范围\人工区范围.shp"

    # 结果保存路径
    output_csv_path = r"D:\GD\更新结果\人工区范围\NDVI_Stats_Unified.csv"

    # 时间范围
    start_year = 2000
    end_year = 2025
    years = range(start_year, end_year + 1)

    # ================= 2. 读取矢量数据 =================
    print(f"正在读取人工区矢量: {artificial_shp_path}")
    if not os.path.exists(artificial_shp_path):
        raise FileNotFoundError(f"找不到矢量文件: {artificial_shp_path}")

    gdf_artificial = gpd.read_file(artificial_shp_path)

    # 检查矢量是否为空
    if gdf_artificial.empty:
        raise ValueError("输入的矢量文件为空，请检查数据！")

    # 预处理：合并矢量内的所有多边形为一个整体（以防有重叠或多个要素）
    # 这步可选，但为了生成Mask更准确，建议合并
    # 此时暂时不转坐标系，等读取栅格时根据栅格投影动态转
    unified_geom = gdf_artificial.unary_union

    # 放入 GeoDataFrame 方便后续重投影
    # 注意：这里我们只保留 geometry 列
    gdf_base = gpd.GeoDataFrame({'geometry': [unified_geom]}, crs=gdf_artificial.crs)

    print("矢量读取完成。开始逐年统计...")

    # ================= 3. 统计函数 =================
    def calc_stats(data_array):
        """计算统计值 (最大, 最小, 均值, 标准差)"""
        # 移除 NaN 值
        valid_data = data_array[~np.isnan(data_array)]

        if valid_data.size == 0:
            return np.nan, np.nan, np.nan, np.nan

        return (
            np.max(valid_data),
            np.min(valid_data),
            np.mean(valid_data),
            np.std(valid_data)
        )

    results = []

    # ================= 4. 循环处理每一年的NDVI =================
    for year in tqdm(years, desc="Processing Years"):
        tif_name = f"Landsat_{year}_NDVI.tif"
        tif_path = os.path.join(ndvi_folder, tif_name)

        if not os.path.exists(tif_path):
            print(f"  [警告] 缺失文件: {tif_name}，跳过...")
            continue

        with rasterio.open(tif_path) as src:
            # 4.1 读取栅格数据
            ndvi_data = src.read(1).astype(np.float32)

            # 处理 NoData (如果源文件有定义)
            if src.nodata is not None:
                ndvi_data[ndvi_data == src.nodata] = np.nan

            # 简单过滤：假设 NDVI 范围在 -1 到 1 之间，过滤掉极端的异常背景值
            # 如果你的数据没有异常值，这行可以注释掉
            ndvi_data[(ndvi_data < -1) | (ndvi_data > 1)] = np.nan

            # 4.2 坐标系对齐
            # 检查矢量的投影是否与当前栅格一致，不一致则转换
            if gdf_base.crs != src.crs:
                gdf_current = gdf_base.to_crs(src.crs)
            else:
                gdf_current = gdf_base

            # 4.3 创建掩膜 (Mask)
            # invert=True: 矢量内部为 True (1), 外部为 False (0)
            mask_artificial = features.geometry_mask(
                gdf_current.geometry,
                out_shape=src.shape,
                transform=src.transform,
                invert=True
            )

            # 4.4 数据分区
            # 提取人工区数据 (Mask 为 True 的地方)
            data_artificial = ndvi_data[mask_artificial]

            # 提取自然区数据 (Mask 为 False 的地方)
            # 注意：必须同时也是非 NaN 的有效像素
            data_natural = ndvi_data[~mask_artificial]

            # 4.5 计算统计指标
            art_stats = calc_stats(data_artificial)
            nat_stats = calc_stats(data_natural)

            # 4.6 记录结果
            results.append({
                "Year": year,

                # 人工区统计
                "Art_Max": art_stats[0],
                "Art_Min": art_stats[1],
                "Art_Mean": art_stats[2],
                "Art_Std": art_stats[3],

                # 自然区统计
                "Nat_Max": nat_stats[0],
                "Nat_Min": nat_stats[1],
                "Nat_Mean": nat_stats[2],
                "Nat_Std": nat_stats[3]
            })

    # ================= 5. 保存结果 =================
    if results:
        df = pd.DataFrame(results)

        # 格式化数值：保留4位小数
        float_cols = [c for c in df.columns if c != "Year"]
        for col in float_cols:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")

        output_dir = os.path.dirname(output_csv_path)
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print("-" * 50)
        print(f"处理完成！统计结果已保存至:\n{output_csv_path}")
    else:
        print("未生成任何结果，请检查输入路径或年份设置。")


if __name__ == "__main__":
    main()
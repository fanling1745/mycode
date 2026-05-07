# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 多时段植被面积转移矩阵 (重投影到 Albers 等面积投影)
# 输入: 植被分类栅格 (0=无,1=有)
# 输出: 转移栅格 (0=无→无,1=无→有,2=有→无,3=有→有) + CSV 面积统计表
# 使用: 修改 input_folder → 运行
# 依赖: rasterio, numpy, pandas
# =============================================================================
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import pandas as pd


def main():
    # ================= 1. 参数配置 =================

    # 输入路径：这里应该是你上一步输出合成数据的文件夹
    # 例如：D:\GD\结果\植被无植被栅格裁剪\Composite_5Years
    input_folder = r"D:\GD\更新结果\重新合成五期植被无植被区域"

    # 输出路径
    output_folder = os.path.join(input_folder, "Transition_Matrix_Results")
    os.makedirs(output_folder, exist_ok=True)

    # 输出表格名称
    output_csv_path = os.path.join(output_folder, "Vegetation_Periods_Transition.csv")

    # 定义投影 (Albers 等面积，确保面积准确)
    proj_string = '+proj=aea +lat_1=25 +lat_2=47 +lat_0=30 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    TARGET_CRS = CRS.from_string(proj_string)

    NODATA_VAL = 255

    # ================= 2. 定义时期与文件名 =================
    # 文件名模板，对应上一步生成的 Composite 文件
    # 例如: Vegetation_Composite_2000_2004.tif
    filename_template = "Vegetation_Composite_{}_{}.tif"

    # 定义 5 个时期 (Period 1 到 Period 5)
    periods_info = {
        1: (2000, 2004),
        2: (2005, 2009),
        3: (2010, 2014),
        4: (2015, 2019),
        5: (2020, 2024)
    }

    # 定义需要计算的转移对 (起始期ID, 结束期ID)
    # 1->2, 2->3, 3->4, 4->5 以及 1->5
    transition_pairs = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (1, 5)  # 整体变化
    ]

    results = []
    print(f"开始计算多期转移矩阵... 目标投影: {TARGET_CRS}")
    print(f"结果将保存在: {output_folder}\n")

    # ================= 3. 循环计算转移矩阵 =================
    for p_start_id, p_end_id in transition_pairs:
        # 获取年份信息
        s_years = periods_info[p_start_id]  # e.g. (2000, 2004)
        e_years = periods_info[p_end_id]  # e.g. (2005, 2009)

        label_str = f"P{p_start_id}_to_P{p_end_id}"
        desc_str = f"{s_years[0]}-{s_years[1]} 到 {e_years[0]}-{e_years[1]}"
        print(f"正在处理: {desc_str} ({label_str})")

        # 构建文件路径
        file_start = os.path.join(input_folder, filename_template.format(s_years[0], s_years[1]))
        file_end = os.path.join(input_folder, filename_template.format(e_years[0], e_years[1]))

        if not os.path.exists(file_start) or not os.path.exists(file_end):
            print(f"  [错误] 文件缺失，跳过: {os.path.basename(file_start)} 或 {os.path.basename(file_end)}")
            continue

        # --- A. 读取并重投影起始年份 ---
        with rasterio.open(file_start) as src:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, TARGET_CRS, src.width, src.height, *src.bounds
            )
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({
                'crs': TARGET_CRS, 'transform': dst_transform,
                'width': dst_width, 'height': dst_height,
                'nodata': NODATA_VAL, 'dtype': rasterio.uint8
            })

            data_start_proj = np.zeros((dst_height, dst_width), dtype=rasterio.uint8)
            reproject(
                source=rasterio.band(src, 1), destination=data_start_proj,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=TARGET_CRS,
                resampling=Resampling.nearest, dst_nodata=NODATA_VAL
            )

        # --- B. 读取并重投影结束年份 (强制对齐) ---
        with rasterio.open(file_end) as src:
            data_end_proj = np.zeros((dst_height, dst_width), dtype=rasterio.uint8)
            reproject(
                source=rasterio.band(src, 1), destination=data_end_proj,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform,  # 关键：使用起始年的网格
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest, dst_nodata=NODATA_VAL
            )

        # --- C. 计算转移代码 ---
        # 0=无, 1=有
        # Code: 0->0=0, 0->1=1, 1->0=2, 1->1=3
        valid_mask = np.isin(data_start_proj, [0, 1]) & np.isin(data_end_proj, [0, 1])
        transition_grid = np.full((dst_height, dst_width), NODATA_VAL, dtype=np.uint8)

        s = data_start_proj[valid_mask].astype(np.uint8)
        e = data_end_proj[valid_mask].astype(np.uint8)
        codes = s * 2 + e
        transition_grid[valid_mask] = codes

        # --- D. 输出栅格 ---
        raster_name = f"Transition_{label_str}.tif"
        with rasterio.open(os.path.join(output_folder, raster_name), 'w', **dst_kwargs) as dst:
            dst.write(transition_grid, 1)

        # --- E. 统计面积 ---
        pixel_size_x = abs(dst_transform[0])
        pixel_size_y = abs(dst_transform[4])
        pixel_area_m2 = pixel_size_x * pixel_size_y

        counts = np.bincount(codes, minlength=4)
        area_km2 = counts * pixel_area_m2 / 1_000_000.0

        # 记录结果
        results.append({
            "Transfer_Pair": label_str,
            "Description": desc_str,
            "无-无 (0)": area_km2[0],
            "无-有 (1,增加)": area_km2[1],
            "有-无 (2,减少)": area_km2[2],
            "有-有 (3)": area_km2[3],
            "总面积": np.sum(area_km2)
        })
        print(f"  [统计完成] 总面积: {np.sum(area_km2):.3f} km²")

    # ================= 4. 保存 CSV =================
    if results:
        df = pd.DataFrame(results)
        cols = ["Transfer_Pair", "Description", "有-有 (3)", "无-无 (0)", "无-有 (1,增加)", "有-无 (2,减少)", "总面积"]
        df = df[cols]
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format='%.3f')
        print(f"\n全部完成！表格已保存至: {output_csv_path}")
    else:
        print("未生成任何结果，请检查输入路径或文件名是否匹配。")


if __name__ == "__main__":
    main()
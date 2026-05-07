# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 逐年植被数据合成 5 年期二值栅格 (5年内 ≥3 年有植被 → 1)
# 输入: Vegetation_Class_{year}.tif
# 输出: Vegetation_Composite_{start}_{end}.tif
# 使用: 修改 input_folder → 运行
# 依赖: rasterio, numpy, tqdm
# =============================================================================
import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm


def main():
    # ================= 1. 参数设置 =================
    # 输入文件夹路径
    input_folder = r"D:\GD\结果\植被无植被栅格裁剪"

    # 输出文件夹路径 (自动创建)
    output_folder = os.path.join(input_folder, "Composite_5Years")
    os.makedirs(output_folder, exist_ok=True)

    # 文件名模板
    filename_template = "Vegetation_Class_{}.tif"

    # 定义时间段
    periods = [
        (2000, 2004),
        (2005, 2009),
        (2010, 2014),
        (2015, 2019),
        (2020, 2024)
    ]

    # 定义 NoData 值 (输出时背景值)
    NODATA_VAL = 255

    print("开始合成 5 年期植被栅格 (规则: 5年内 >= 3年为植被)...")
    print(f"结果将保存在: {output_folder}")
    print("-" * 50)

    # ================= 2. 循环处理每个时段 =================
    for start_year, end_year in periods:
        period_name = f"{start_year}-{end_year}"
        print(f"正在处理时段: {period_name}")

        # 构建当前时段的文件列表
        years_in_period = range(start_year, end_year + 1)
        file_list = []
        for y in years_in_period:
            f_path = os.path.join(input_folder, filename_template.format(y))
            if os.path.exists(f_path):
                file_list.append(f_path)
            else:
                print(f"  [警告] 缺失年份: {y}，该年将不参与计算。")

        if not file_list:
            print(f"  [跳过] 该时段没有找到任何文件。")
            continue

        # ================= 3. 读取并累加数据 =================
        # 以第一张图为基准 (Base)
        base_file = file_list[0]

        with rasterio.open(base_file) as src_base:
            # 获取基准元数据
            base_meta = src_base.meta.copy()
            base_crs = src_base.crs
            base_transform = src_base.transform
            base_w, base_h = src_base.width, src_base.height

            # 初始化累加矩阵 (Sum Array)
            # 使用 float 以便处理 nan，但这里数据是 0/1，用 int 也可以
            # 这里初始化为0
            sum_grid = np.zeros((base_h, base_w), dtype=np.uint8)

            # 创建一个用于记录有效区域的掩膜 (Mask)
            # 只有当基准影像不是 NoData 的地方，我们才认为有效
            # 假设 NoData 是 255 或其他，这里先读取第一年的数据作为参考
            base_data = src_base.read(1)
            # 如果原始数据里有 NoData (比如255)，需要将其视为0不参与累加，或者完全屏蔽
            # 这里生成一个全局有效掩膜：只要基准图不是NoData，就计算
            valid_mask = (base_data != NODATA_VAL)

        # 循环读取该时段的所有文件并累加
        for f_path in tqdm(file_list, desc=f"  Synthesizing {period_name}", leave=False):
            with rasterio.open(f_path) as src:
                # 使用 WarpedVRT 强制对齐到基准影像
                with WarpedVRT(src,
                               crs=base_crs,
                               transform=base_transform,
                               width=base_w,
                               height=base_h,
                               resampling=Resampling.nearest) as vrt:

                    data = vrt.read(1)

                    # 数据清洗：确保只有 0 和 1 参与计算
                    # 将 NoData (如255) 临时转为 0，以免影响求和
                    # 假设有效值只有 0 和 1
                    clean_data = data.copy()

                    # 如果数据中有 NoData 标记，将其设为 0
                    if src.nodata is not None:
                        clean_data[data == src.nodata] = 0
                    else:
                        # 如果没有定义nodata但有255这种大数，也强制归零
                        clean_data[data > 1] = 0

                    sum_grid += clean_data.astype(np.uint8)

        # ================= 4. 应用规则 (阈值 >= 3) =================
        # 规则：5年里有3年及以上是1，则结果为1，否则为0
        # 如果你缺失了年份（比如只有4个文件），这个逻辑依然适用（即总和>=3）

        final_grid = np.where(sum_grid >= 3, 1, 0).astype(np.uint8)

        # 将背景区域还原为 NoData
        # (即：如果基准影像在这个位置是无效的，结果也设为无效)
        if 'valid_mask' in locals():
            final_grid[~valid_mask] = NODATA_VAL

        # ================= 5. 输出结果 =================
        out_filename = f"Vegetation_Composite_{start_year}_{end_year}.tif"
        out_path = os.path.join(output_folder, out_filename)

        # 更新元数据
        out_meta = base_meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": base_h,
            "width": base_w,
            "transform": base_transform,
            "crs": base_crs,
            "dtype": rasterio.uint8,
            "count": 1,
            "nodata": NODATA_VAL,
            "compress": "lzw"
        })

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(final_grid, 1)

        print(f"  [完成] 已生成: {out_filename}")

    print("-" * 50)
    print("所有时段合成完毕！")


if __name__ == "__main__":
    main()
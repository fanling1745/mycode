# -*- coding: utf-8 -*-
# =============================================================================
# 功能: Sen's Slope + Mann-Kendall 趋势检验 (NDVI 长时间序列)
# 输入: NDVI 栅格时间序列 Landsat_{year}_NDVI.tif (如 2000-2025)
# 输出: MK_Trend_Class.tif (趋势分级 -4~4) + Sen_Slope_Value.tif (斜率值)
# 使用: 修改 input_folder / start_year / end_year → 运行
# 依赖: rasterio, numpy, tqdm
# =============================================================================
import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm


def main():
    # ================= 参数设置 =================
    input_folder = r"D:\GD\结果\NDVI"  # 请修改路径
    output_trend_path = os.path.join(input_folder, "MK_Trend_Class.tif")
    output_slope_path = os.path.join(input_folder, "Sen_Slope_Value.tif")

    start_year = 2000
    end_year = 2025
    years = range(start_year, end_year + 1)
    n_years = len(years)

    filename_template = "Landsat_{}_NDVI.tif"

    # ================= 1. 智能读取与对齐 =================
    print("正在读取影像数据 (自动对齐中)...")
    raster_data = []

    # 1. 读取基准影像 (使用第一年 2000 作为基准)
    base_path = os.path.join(input_folder, filename_template.format(start_year))
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"基准文件未找到: {base_path}")

    # 读取基准影像的属性
    with rasterio.open(base_path) as src_base:
        base_meta = src_base.meta.copy()
        base_transform = src_base.transform
        base_crs = src_base.crs
        base_width = src_base.width
        base_height = src_base.height

        # 读取第一年的数据并处理
        data = src_base.read(1).astype(np.float32)
        data[data <= 0] = np.nan
        raster_data.append(data)
        print(f"基准年份 {start_year} 读取完成 (尺寸: {base_height}x{base_width})")

    # 2. 读取其余年份，并强制对齐到基准影像
    for year in tqdm(years[1:], desc="Aligning & Reading"):
        tif_path = os.path.join(input_folder, filename_template.format(year))

        if not os.path.exists(tif_path):
            print(f"警告: 年份 {year} 缺失，跳过...")
            # 注意：如果缺失年份，整个时间序列长度会变，这会影响MK计算
            # 简单处理：如果必须连续，这里应该报错；如果允许缺失，需要插值填充
            # 这里简单填充全NaN层，以免堆叠报错
            nan_layer = np.full((base_height, base_width), np.nan, dtype=np.float32)
            raster_data.append(nan_layer)
            continue

        with rasterio.open(tif_path) as src:
            # 使用 WarpedVRT 强制将当前影像重采样/裁剪到与基准影像一致
            with WarpedVRT(src,
                           crs=base_crs,
                           transform=base_transform,
                           width=base_width,
                           height=base_height,
                           resampling=Resampling.nearest) as vrt:
                data = vrt.read(1).astype(np.float32)
                data[data <= 0] = np.nan
                raster_data.append(data)

    # 堆叠数据
    try:
        stack_data = np.stack(raster_data, axis=0)
    except ValueError as e:
        print("错误: 即使经过对齐，数据堆叠仍然失败。请检查是否有年份完全读取失败。")
        raise e

    rows, cols = stack_data.shape[1], stack_data.shape[2]
    print(f"数据堆叠完成，尺寸: {stack_data.shape}")

    # ================= 2. 计算 Sen's Slope 和 MK S统计量 =================
    print("正在计算斜率和 S 统计量...")

    s_stat = np.zeros((rows, cols), dtype=np.float32)
    slope_candidates = []

    # 向量化计算
    for i in tqdm(range(n_years), desc="Calculating Pairs"):
        for j in range(i + 1, n_years):
            data_late = stack_data[j]
            data_early = stack_data[i]

            # 如果某一年全是NaN（缺失），这里结果也是NaN，后续需要处理
            diff = data_late - data_early
            time_span = j - i

            sgn = np.sign(diff)
            sgn = np.nan_to_num(sgn, nan=0)  # 把NaN视为0，不影响S统计量
            s_stat += sgn

            val_slope = diff / time_span
            slope_candidates.append(val_slope)

    # 计算 Sen Slope 中位数
    print("计算中位数斜率 (可能占用较大内存)...")
    slope_stack = np.stack(slope_candidates, axis=0)
    # 使用 nanmedian 忽略缺失值
    sen_slope = np.nanmedian(slope_stack, axis=0)

    del slope_stack, slope_candidates  # 释放内存

    # ================= 3. 计算 Z 统计量 =================
    print("正在进行显著性检验...")

    n = n_years
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    z_score = np.zeros((rows, cols), dtype=np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        mask_pos = s_stat > 0
        z_score[mask_pos] = (s_stat[mask_pos] - 1) / np.sqrt(var_s)

        mask_neg = s_stat < 0
        z_score[mask_neg] = (s_stat[mask_neg] + 1) / np.sqrt(var_s)

    # ================= 4. 分级赋值 =================
    print("正在分类...")
    trend_class = np.zeros((rows, cols), dtype=np.int8)
    abs_z = np.abs(z_score)

    # 上升
    pos_slope = sen_slope > 0
    trend_class[pos_slope & (abs_z >= 2.58)] = 4
    trend_class[pos_slope & (abs_z >= 1.96) & (abs_z < 2.58)] = 3
    trend_class[pos_slope & (abs_z >= 1.645) & (abs_z < 1.96)] = 2
    trend_class[pos_slope & (abs_z < 1.645)] = 1

    # 下降
    neg_slope = sen_slope < 0
    trend_class[neg_slope & (abs_z >= 2.58)] = -4
    trend_class[neg_slope & (abs_z >= 1.96) & (abs_z < 2.58)] = -3
    trend_class[neg_slope & (abs_z >= 1.645) & (abs_z < 1.96)] = -2
    trend_class[neg_slope & (abs_z < 1.645)] = -1

    # 无效值掩膜处理
    valid_mask = np.any(~np.isnan(stack_data), axis=0)
    trend_class[~valid_mask] = 0
    sen_slope[~valid_mask] = -9999

    # ================= 5. 保存结果 =================
    print("正在保存...")

    # 更新元数据以匹配基准影像
    profile_class = base_meta.copy()
    profile_class.update(dtype=rasterio.int8, count=1, nodata=0, compress='lzw')
    with rasterio.open(output_trend_path, 'w', **profile_class) as dst:
        dst.write(trend_class, 1)

    profile_slope = base_meta.copy()
    profile_slope.update(dtype=rasterio.float32, count=1, nodata=-9999, compress='lzw')
    with rasterio.open(output_slope_path, 'w', **profile_slope) as dst:
        dst.write(sen_slope, 1)

    print("完成！")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 调试脚本, 逐网格检查景观指数计算过程
# 输入: 土地利用 TIFF + 渔网 SHP + 目标像素值 + DEBUG_INDEX
# 输出: 控制台详细中间过程输出
# 使用: 修改 input_folder / fishnet_path / DEBUG_INDEX → 运行
# 依赖: rasterio, pylandstats, geopandas, numpy
# =============================================================================
import os
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import pylandstats
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 请确保路径正确
input_folder = r"D:\frag\0129\tif"
fishnet_path = r"D:\frag\0129\shp\Urban_1km.shp"
year = 2015
target_raw_val = 8

# 【关键】这里填写之前诊断脚本里显示“找到了8”的那个网格的 ID (索引号)
# 如果你不确定，就写一个大概在城市中心的ID，或者保持 1733 试试
DEBUG_INDEX = 1733


# ===========================================

def debug_single_grid():
    print(f"🔬 正在对第 {DEBUG_INDEX} 号网格进行‘开颅手术’检查...")

    # 1. 读取数据
    base_fishnet = gpd.read_file(fishnet_path)
    row = base_fishnet.iloc[DEBUG_INDEX]  # 只取这一行
    geom_original = row['geometry']
    print(f"📍 网格原始中心点: {geom_original.centroid}")

    tif_path = os.path.join(input_folder, f"CLCD_v01_{year}_albert.tif")

    with rasterio.open(tif_path) as src:
        print(f"🗺️ 栅格投影: {src.crs.to_string()[:50]}...")

        # 2. 坐标投影 (模拟主程序的逻辑)
        # 我们来看看这一步转出来的几何到底对不对
        if base_fishnet.crs != src.crs:
            print("🔄 执行坐标转换 (Vector -> Raster CRS)...")
            # 创建单行 GDF 进行转换
            gdf_single = gpd.GeoDataFrame({'geometry': [geom_original]}, crs=base_fishnet.crs)
            gdf_proj = gdf_single.to_crs(src.crs)
            geom_proj = gdf_proj.geometry.iloc[0]
            print(f"   转换后网格边界: {geom_proj.bounds}")
        else:
            geom_proj = geom_original

        # 3. 计算窗口
        window = from_bounds(*geom_proj.bounds, transform=src.transform)
        print(f"🪟 读取窗口 (Window): {window}")

        # 4. 读取原始数据
        # 强制 int32 以防 uint8 溢出问题
        arr_raw = src.read(1, window=window, boundless=True, fill_value=0).astype('int32')
        print(f"🔢 原始数组形状: {arr_raw.shape}")
        print(f"   原始数组里的值: {np.unique(arr_raw)}")

        if target_raw_val in arr_raw:
            print(f"   ✅ 在原始数组里找到了 {target_raw_val}！第一步通过。")
        else:
            print(f"   ❌ 原始数组里没找到 {target_raw_val}！问题出在 Window 读取偏移。")
            return  # 既然读不到，后面不用看了

        # 5. 二值化
        FOREGROUND = 1
        BACKGROUND = 0
        NODATA = 999

        binary_arr = np.full(arr_raw.shape, BACKGROUND, dtype='int32')
        binary_arr[arr_raw == target_raw_val] = FOREGROUND
        print(f"0️⃣1️⃣ 二值化后包含的值: {np.unique(binary_arr)}")

        # 6. 掩膜
        window_transform = src.window_transform(window)
        mask = geometry_mask([geom_proj], out_shape=binary_arr.shape, transform=window_transform, invert=True)

        # 打印掩膜情况
        true_count = np.sum(mask)
        total_count = mask.size
        print(f"🎭 掩膜遮挡比例: {true_count}/{total_count} 像素被标记为几何外部")

        binary_arr[mask] = NODATA

        unique_final = np.unique(binary_arr)
        print(f"🏁 最终送入计算的数组值: {unique_final}")

        if FOREGROUND not in unique_final:
            print("   ❌ 糟糕！目标像元在掩膜这一步被切掉了！")
            print("   原因可能是：坐标转换虽成功，但只有边缘重叠，中心其实没对上。")
            return

        # 7. 强行计算 pylandstats
        print("🧮 正在调用 pylandstats...")
        ls = pylandstats.Landscape(binary_arr, res=src.res, nodata=NODATA)

        # 打印一下它识别出的地类
        print(f"   pylandstats 识别到的地类列表: {ls.classes}")

        # 计算
        metrics = ['proportion_of_landscape', 'number_of_patches', 'aggregation_index']
        df = ls.compute_class_metrics_df(metrics=metrics)
        print("\n📊 【计算结果直接打印】:")
        print(df)

        # 8. 模拟提取
        val = 0
        if FOREGROUND in df.index:
            val = df.loc[FOREGROUND, 'proportion_of_landscape']
            print(f"\n✅ 成功提取到 PLAND: {val}")
        else:
            print(f"\n❌ 结果里没有 Class {FOREGROUND}。")


if __name__ == "__main__":
    debug_single_grid()
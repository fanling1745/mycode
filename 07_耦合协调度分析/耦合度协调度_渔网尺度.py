# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 渔网尺度的耦合度 C 与协调度 D 计算
# 输入: 破碎化和 TVDI 栅格 (自动重投影到 UTM 48N)
# 输出: Coupling_Coordination_{year}.shp (含 C_Value/D_Value/C_Class/D_Class)
# 使用: 修改 input_folder / output_folder / years → 运行
# 依赖: rasterio, rasterstats, geopandas, shapely, numpy, pandas
# =============================================================================
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import box
from rasterstats import zonal_stats
import pandas as pd
import tempfile


def reproject_raster(input_path, target_epsg=32648):
    """
    将栅格重投影到 UTM Zone 48N (EPSG:32648)
    返回临时文件路径
    """
    dst_crs = f'EPSG:{target_epsg}'

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata
        })

        temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)

    return temp_path


def create_fishnet(raster_path, resolution=120):
    """
    基于栅格范围创建全域渔网 (300m x 300m)
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    xmin, ymin, xmax, ymax = bounds
    geometries = []

    # 从左下角开始生成网格
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            p = box(x, y, x + resolution, y + resolution)
            geometries.append(p)
            y += resolution
        x += resolution

    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=crs)
    # 添加一个唯一ID列，方便后续查看
    gdf['Grid_ID'] = range(1, len(gdf) + 1)
    return gdf


def classify_value(series, bins, labels):
    """
    辅助函数：基于区间进行分级
    right=False 表示区间为 [a, b) 左闭右开
    """
    # 稍微扩展最后一个bin的右边界，以包含最大值 1.0 (因为 right=False 默认不包含右边界)
    # 例如 bins=[0, 0.3, 0.5, 0.8, 1.0] -> 实际上最后一段需要是 1.0001 才能包含 1.0
    adjusted_bins = bins.copy()
    adjusted_bins[-1] = adjusted_bins[-1] + 0.0001

    return pd.cut(series, bins=adjusted_bins, labels=labels, right=False)


def calculate_ccdm_vector(year, input_dir, output_dir):
    print(f"=== 正在处理 {year} 年数据 ===")

    frag_file = os.path.join(input_dir, f"masked_fragmentation_{year}.tif")
    tvdi_file = os.path.join(input_dir, f"TVDI_Cropland_{year}.tif")

    if not os.path.exists(frag_file) or not os.path.exists(tvdi_file):
        print(f"跳过 {year}: 文件缺失")
        return

    # 1. 重投影
    print("  正在重投影到 UTM 48N...")
    frag_temp = reproject_raster(frag_file, target_epsg=32648)
    tvdi_temp = reproject_raster(tvdi_file, target_epsg=32648)

    try:
        # 2. 生成全域渔网
        print(f"  正在生成全域渔网 (300m)...")
        gdf_grid = create_fishnet(frag_temp, resolution=120)

        # 3. 分区统计 (保留所有网格，即使结果为None)
        print("  正在统计栅格均值...")

        # 统计 U (破碎化)
        stats_frag = zonal_stats(gdf_grid, frag_temp, stats="mean", nodata=-9999)
        gdf_grid['U_Val'] = [x['mean'] for x in stats_frag]

        # 统计 E (TVDI)
        stats_tvdi = zonal_stats(gdf_grid, tvdi_temp, stats="mean", nodata=-9999)
        gdf_grid['E_Val'] = [x['mean'] for x in stats_tvdi]

        # 4. 计算逻辑 (使用 Pandas 矢量化计算，自动处理 NaN)
        print("  正在计算耦合度与协调度...")

        # 提取数据便于书写公式
        U = gdf_grid['U_Val']
        E = gdf_grid['E_Val']

        # 计算 C (耦合度)
        # 只有当 U 和 E 都有值时才计算，否则保持 NaN
        sum_ue = U + E

        # 避免除以0，如果是0则设为NaN
        denom = sum_ue ** 2
        term = (U * E) / denom

        # C = 2 * sqrt( (UE) / (U+E)^2 )
        gdf_grid['C_Value'] = 2 * np.sqrt(term)

        # 处理可能的计算异常（如除零导致的inf），将其转回NaN
        gdf_grid['C_Value'] = gdf_grid['C_Value'].replace([np.inf, -np.inf], np.nan)

        # 计算 D (协调度)
        # T = 0.5U + 0.5E
        T = 0.5 * U + 0.5 * E
        gdf_grid['D_Value'] = np.sqrt(gdf_grid['C_Value'] * T)

        # 5. 分级 (Classification)
        print("  正在进行分级分类...")

        # --- C 分级 ---
        # 1: [0, 0.3), 2: [0.3, 0.5), 3: [0.5, 0.8), 4: [0.8, 1.0]
        c_bins = [0, 0.3, 0.5, 0.8, 1.0]
        c_labels = [1, 2, 3, 4]
        gdf_grid['C_Class'] = classify_value(gdf_grid['C_Value'], c_bins, c_labels)

        # --- D 分级 ---
        # 1: [0, 0.3), 2: [0.3, 0.6), 3: [0.6, 0.8), 4: [0.8, 1.0]
        # 注意：这里中间阈值是 0.6
        d_bins = [0, 0.3, 0.6, 0.8, 1.0]
        d_labels = [1, 2, 3, 4]
        gdf_grid['D_Class'] = classify_value(gdf_grid['D_Value'], d_bins, d_labels)

        # 6. 保存结果
        # 将分类列转为数字(float或int)，因为Shapefile对Categorical支持不好
        # NaN会自动变为空
        gdf_grid['C_Class'] = gdf_grid['C_Class'].astype(float)
        gdf_grid['D_Class'] = gdf_grid['D_Class'].astype(float)

        output_name = f"Coupling_Coordination_{year}.shp"
        output_path = os.path.join(output_dir, output_name)

        # 即使某些列是空的，也会被写入 Shapefile
        gdf_grid.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"  已保存: {output_name} (包含 {len(gdf_grid)} 个网格)")

    except Exception as e:
        print(f"  处理 {year} 年时发生错误: {e}")
    finally:
        # 清理
        if os.path.exists(frag_temp):
            try:
                os.remove(frag_temp)
            except:
                pass
        if os.path.exists(tvdi_temp):
            try:
                os.remove(tvdi_temp)
            except:
                pass


# ================= 运行配置 =================
input_folder = r"E:\方\fragststs\空间自相关\shuju\shuju"
output_folder = r"E:\方\fragststs\空间自相关\shuju\CCDM_Vector_120m"
years = [2000, 2005, 2010, 2015, 2020, 2022]

if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for year in years:
        calculate_ccdm_vector(year, input_folder, output_folder)
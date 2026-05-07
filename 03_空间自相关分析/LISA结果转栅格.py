# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 将局部 Moran's I 分析的点 Shapefile 结果转为分类栅格
# 输入: 含 COType 字段的 LISA 结果 SHP + 模板栅格
# 输出: 分类 TIFF (LL=1,LH=2,HH=3,HL=4, 背景=0)
# 使用: 修改 input_folder / template_raster_path / output_folder → 运行
# 依赖: geopandas, rasterio, numpy
# =============================================================================
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import numpy as np
import os
import glob

# ================= 1. 配置路径 =================
input_folder = r"E:\frag\20260204\空间自相关\局部空间自相关"  # 输入：存放所有点shapefile的文件夹
template_raster_path = r"E:\frag\20260204\景观格局尺度指数\c2000_contag.tif"  # 输入：栅格模板 (仅用于获取范围和坐标系)
output_folder = r"E:\frag\20260204\空间自相关\栅格计算结果"  # 输出：结果保存文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# ================= 2. 准备栅格参数 (只做一次) =================
print(f"正在读取模板信息: {template_raster_path}...")
with rasterio.open(template_raster_path) as src:
    template_crs = src.crs
    # 获取模板的物理边界 (左, 下, 右, 上)
    minx, miny, maxx, maxy = src.bounds

    # 检查坐标系是否为投影坐标系 (单位必须是米)
    if src.crs.is_geographic:
        print("【警告】模板坐标系似乎是经纬度 (EPSG:4326等)。")
        print("       分辨率设置为 1000 会被视为 1000度！请确保模板使用投影坐标系 (如 UTM, EPSG:3857)。")

# 定义分辨率 (单位：米)
resolution = 1000

# 计算输出栅格的行列数 (向上取整，确保覆盖整个范围)
out_width = int(np.ceil((maxx - minx) / resolution))
out_height = int(np.ceil((maxy - miny) / resolution))

# 定义仿射变换 (左上角锚点 + 分辨率)
# 注意：rasterio的transform原点通常是左上角 (minx, maxy)
out_transform = from_origin(minx, maxy, resolution, resolution)

print(f"--- 统一栅格定义 ---")
print(f"坐标系: {template_crs}")
print(f"分辨率: {resolution} 米")
print(f"栅格尺寸: {out_width} x {out_height}")
print("-" * 30)

# ================= 3. 批量处理循环 =================
# 获取文件夹内所有 .shp 文件
shp_files = glob.glob(os.path.join(input_folder, "*.shp"))

if not shp_files:
    print(f"错误：在 {input_folder} 中未找到 .shp 文件")

for shp_path in shp_files:
    # --- A. 获取文件名 ---
    # 例如: "C:/data/points_2020.shp" -> "points_2020"
    filename = os.path.splitext(os.path.basename(shp_path))[0]
    output_path = os.path.join(output_folder, f"{filename}.tif")

    print(f"正在处理: {filename} ...")

    # --- B. 读取与重投影 ---
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"  读取失败，跳过: {e}")
        continue

    if gdf.crs != template_crs:
        gdf = gdf.to_crs(template_crs)

    # --- C. 字段映射 ---
    # 定义 COType 到 数字 的映射
    type_mapping = {
        'LL': 1,
        'LH': 2,
        'HH': 3,
        'HL': 4
    }

    # 映射字段，如果有点的 COType 不在字典里，会变成 NaN
    gdf['val_code'] = gdf['COType'].map(type_mapping)

    # 去除无法映射的点 (可选)
    valid_points = gdf.dropna(subset=['val_code'])

    if valid_points.empty:
        print(f"  警告: {filename} 中没有有效的 COType 数据，生成全0栅格。")
        # 即使没有点，我们也需要生成一个全0的背景
        shapes = []
    else:
        # 准备烧录数据: (geometry, value)
        shapes = ((geom, val) for geom, val in zip(valid_points.geometry, valid_points['val_code']))

    # --- D. 栅格化 (核心步骤) ---
    # fill=0 实现了你的需求："范围内的所有NODATA填充0值"
    burned_array = features.rasterize(
        shapes=shapes,
        out_shape=(out_height, out_width),
        transform=out_transform,
        fill=0,  # <--- 关键：没有点的地方填充 0
        all_touched=True,  # 如果点落在像素边缘也算入
        dtype=rasterio.uint8
    )

    # --- E. 写入结果 ---
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=out_height,
            width=out_width,
            count=1,
            dtype=rasterio.uint8,
            crs=template_crs,
            transform=out_transform,
            nodata=0,  # 将 0 标记为 NoData (根据你的需求，也可以不设)
            compress='lzw'
    ) as dst:
        dst.write(burned_array, 1)

    print(f"  已保存: {output_path}")

print("\n所有文件处理完成！")
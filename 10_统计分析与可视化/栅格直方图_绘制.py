# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 支持单栅格或多栅格 N×1 组图的直方图绘制
# 输入: 单张或多张 TIFF 路径
# 输出: 直方图 PNG
# 使用: 调用 plot_tif_histogram(tif_path) 或 plot_multiple_tif_histograms(tif_paths)
# 依赖: rasterio, numpy, matplotlib
# =============================================================================
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib全局字体和字体大小（支持中文）
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号“-”显示问题
plt.rcParams['font.weight'] = 'bold'  # 设置全局字体加粗
plt.rcParams['xtick.labelsize'] = 25  # 设置x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 25  # 设置y轴刻度标签字体大小

# 定义读取和绘制直方图的函数
def plot_multiple_tif_histograms(tif_paths, band=1, bins=200, output_path=None):
    """
    读取多个tif栅格数据，并绘制它们的直方图，排列格式为7行1列。

    参数:
        tif_paths (list): 包含7个tif文件路径的列表。
        band (int): 栅格文件的波段号 (默认为1)。
        bins (int): 直方图的柱数 (默认为256)。
        output_path (str): 保存图片的路径。
    """
    num_files = len(tif_paths)
    if num_files != 7:
        print("错误: 请确保提供7个tif文件路径！")
        return

    # 设置画布大小和子图布局
    fig, axes = plt.subplots(7, 1, figsize=(8, 18), tight_layout=True)  # 7行1列

    # 遍历每个tif文件，绘制直方图
    for i, (tif_path, ax) in enumerate(zip(tif_paths, axes)):
        try:
            with rasterio.open(tif_path) as src:
                # 读取指定波段数据
                raster_data = src.read(band)

            # 去掉nodata值
            raster_data = raster_data[raster_data != src.nodata]
            raster_data = raster_data[~np.isnan(raster_data)]

            # 绘制直方图
            ax.hist(raster_data.flatten(), bins=bins, color='#568497')

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='#568497', fontsize=8)
            print(f"Error processing file {tif_path}: {e}")

    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # 显示画布
    plt.show()

# 定义文件夹路径
path = r"F:\AAA\data\HQ\HQ7"

# 拼接tif文件路径列表
tif_files = [
    os.path.join(path, filename) for filename in [
        "1990quality_c.tif", "1995quality_c.tif", "2000quality_c.tif",
        "2005quality_c.tif", "2010quality_c.tif", "2015quality_c.tif",
        "2020quality_c.tif"
    ]
]

# 调用函数并保存图片
output_path = r"F:\AAA\img\生境质量\histograms.png"
plot_multiple_tif_histograms(tif_files, band=1, bins=200, output_path=output_path)
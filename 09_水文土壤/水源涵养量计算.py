# -*- coding: utf-8 -*-
# =============================================================================
# 功能: InVEST 水源涵养模型 (Retention = V_factor × T1_factor × K_factor × wyield)
# 输入: T1.tif / K.tif (静态) + v_{year}.tif / wyield_{year}.tif (动态)
# 输出: Retention_{year}.tif
# 使用: 修改 arcpy.env.workspace → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
from arcpy.sa import *
import os

# 1. 设置工作空间（请修改为你的实际数据所在文件夹路径）
arcpy.env.workspace = r"E:\数据分析\水源涵养\20260306_para\Cal\Cal"

# 允许覆盖已有输出文件
arcpy.env.overwriteOutput = True

# 检查并获取空间分析许可
arcpy.CheckOutExtension("Spatial")

print("环境设置完毕，开始处理...")

# 2. 读取静态栅格（三个年份共用）
# 为了保证严格对齐，可以将环境变量的范围、像元大小和捕捉栅格设置为 T1.tif
arcpy.env.extent = "K.tif"
arcpy.env.cellSize = "K.tif"
arcpy.env.snapRaster = "K.tif"

T1_raster = Raster("T1.tif")
K_raster = Raster("K.tif")

# 3. 提取循环外可以预先计算的静态因子，避免重复运算，提高计算效率
print("正在计算静态地形和土壤因子...")
# T1 修正因子：Min(1, (0.9 * T1) / 3)
T1_factor = Con((0.9 * T1_raster) / 3.0 > 1.0, 1.0, (0.9 * T1_raster) / 3.0)

# K 修正因子：Min(1, K / 300)
K_factor = Con(K_raster / 300.0 > 1.0, 1.0, K_raster / 300.0)

# 将两个静态因子相乘合并
static_combined_factor = T1_factor * K_factor

# 4. 设置需要循环的年份列表
years = [2022]

# 5. 开始批量循环计算多年份水源涵养量
for year in years:
    print(f"开始计算 {year} 年水源涵养量...")

    # 构建动态输入文件名
    V_name = f"v_{year}.tif"
    wyield_name = f"wyield_{year}.tif"
    out_name = f"Retention_{year}.tif"

    # 检查当前年份的数据是否存在
    if not arcpy.Exists(V_name) or not arcpy.Exists(wyield_name):
        print(f"警告：找不到 {year} 年的 V 或 wyield 栅格，跳过该年份。")
        continue

    # 读取动态栅格
    V_raster = Raster(V_name)
    wyield_raster = Raster(wyield_name)

    # 计算当前年份的流速 V 修正因子：Min(1, 249 / V)
    V_factor = Con(249.0 / V_raster > 1.0, 1.0, 249.0 / V_raster)

    # 将动态因子、静态合并因子与产水量相乘
    retention_result = V_factor * static_combined_factor * wyield_raster

    # 保存结果
    retention_result.save(out_name)
    print(f"--> {year} 年计算完成，已保存为 {out_name}")

print("所有年份水源涵养量计算全部完成！")
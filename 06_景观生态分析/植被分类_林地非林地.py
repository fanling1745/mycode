# -*- coding: utf-8 -*-
# =============================================================================
# 功能: CLCD 重分类为林地(1)/非林地(2), 林地=编码 2/3/4
# 输入: 存放多年份 CLCD TIFF 的文件夹
# 输出: 重分类后的 TIFF
# 使用: 修改 in_folder / out_folder / year_list → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
from arcpy.sa import *
import os

# ------------------- 参数设置 -------------------
# 输入数据的文件夹路径
in_folder = r"D:\frag\002\土地利用"

# 输出数据的文件夹路径 (建议新建一个空文件夹，避免覆盖原文件)
out_folder = r"D:\frag\002\002\raster"

# 需要处理的年份列表
year_list = [2000, 2005, 2010, 2015, 2020, 2024]

# CLCD 地类编码 (根据武汉大学CLCD标准)
# 2=林地, 3=灌木, 4=草地
target_values = [2, 3, 4]

# ------------------- 环境设置 -------------------
arcpy.CheckOutExtension("Spatial")  # 检出空间分析扩展模块
arcpy.env.overwriteOutput = True  # 允许覆盖同名文件

# 确保输出文件夹存在，不存在则创建
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

print("开始处理...")

# ------------------- 主循环 -------------------
for year in year_list:
    # 构建文件名
    file_name = f"CLCD_v01_{year}_albert.tif"
    in_path = os.path.join(in_folder, file_name)
    out_path = os.path.join(out_folder, file_name)

    # 检查文件是否存在
    if os.path.exists(in_path):
        try:
            print(f"正在处理: {file_name} ...")

            # 加载栅格
            raster = Raster(in_path)

            # --- 核心重分类逻辑 ---
            # 逻辑：如果像素值在 target_values (2,3,4) 中，则为 1，否则为 2
            # 注意：原始数据中的 NoData 依然会保持为 NoData

            # 方法：使用 Con 函数结合布尔运算
            # (raster == 2) | (raster == 3) | (raster == 4)
            out_raster = Con((raster == 2) | (raster == 3) | (raster == 4), 1, 2)

            # 如果你的数据里还有不需要参与计算的背景值(如0或255)，可能需要先处理NoData
            # 但通常CLCD是标准的，上述代码会将非2/3/4的所有有效值转为2

            # 保存结果
            out_raster.save(out_path)
            print(f"成功导出: {out_path}")

        except Exception as e:
            print(f"处理 {file_name} 时出错: {e}")
    else:
        print(f"跳过: 找不到文件 {in_path}")

print("所有处理完成！")
# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 根据 CLCD 编码首位数字计算土地利用强度 (1→3, 2/3/4→2, 5→4)
# 输入: LUCC 栅格
# 输出: 土地利用强度栅格
# 使用: 修改 in_raster 和 out_raster → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
from arcpy.sa import *

# 1. 设置环境
arcpy.env.workspace = r"F:\项目\大理\大理\大理.gdb"  # 替换为你的文件夹
arcpy.CheckOutExtension("Spatial")  # 启用 Spatial Analyst

# 2. 输入输出路径
in_raster = "LUCC__2020_tif"  # 或完整路径 r"C:\...\LUCC_2000.tif"
out_raster = r"F:\项目\大理\data\大理市土地利用强度\土地利用强度_2020.tif"  # 必须写完整路径

# 3. 如果文件已存在，先删除
if arcpy.Exists(out_raster):
    arcpy.Delete_management(out_raster)

# 4. 计算首位数字并重分类
first_digit = Int(Raster(in_raster) / 10)  # 提取首位数字
reclassified = Con(first_digit == 1, 3,
                  Con((first_digit == 2) | (first_digit == 3) | (first_digit == 4), 2,
                      Con(first_digit == 5, 4, 0)))

# 5. 保存结果
reclassified.save(out_raster)
print("重分类完成！输出路径：", out_raster)
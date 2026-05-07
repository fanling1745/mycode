# -*- coding: utf-8 -*-
# =============================================================================
# 功能: CLCD 9 类重分类为 7 类 (冰雪→水体, 湿地→水体, 裸地→未利用地)
# 输入: 地理数据库中的 CLCD 栅格
# 输出: 重分类后的栅格
# 使用: 修改 arcpy.env.workspace / input_raster / output_raster → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
from arcpy.sa import *

# 设置工作环境和输入输出路径
arcpy.env.workspace = r"F:\项目\城乡聚落三维形态\工程文件\城乡聚落三维形态\城乡聚落三维形态.gdb"
input_raster = "云南省CLCD"
output_raster = "云南省CLCD_reclass"

# 确保空间分析扩展模块可用
arcpy.CheckOutExtension("Spatial")

# 定义重分类规则
# 原始值: 1 2 3 4 5 6 7 8 9
# 新值:   1 2 3 4 5 5 6 7 5
reclass_field = "Value"
remap = RemapValue([[1, 1],  # 耕地保持不变
                    [2, 2],  # 林地保持不变
                    [3, 3],  # 灌木保持不变
                    [4, 4],  # 草地保持不变
                    [5, 5],  # 水体保持不变
                    [6, 5],  # 冰雪地→水域
                    [7, 6],  # 裸地→未利用地
                    [8, 7],  # 不透水面→建设用地
                    [9, 5]]) # 湿地→水域

# 执行重分类
reclassified = Reclassify(input_raster, reclass_field, remap, "NODATA")

# 保存结果
reclassified.save(output_raster)

# 可选：为输出栅格添加属性描述
# 首先将栅格转为整型(如果是浮点型)
# int_raster = Int(output_raster)
# int_output = "输出重分类整型.tif"
# int_raster.save(int_output)

# 添加栅格属性表
#arcpy.BuildRasterAttributeTable_management(int_output, "Overwrite")

print("重分类完成！")
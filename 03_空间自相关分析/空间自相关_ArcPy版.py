# -*- coding: utf-8 -*-
# =============================================================================
# 功能: ArcPy 版栅格→点→局部异常值分析→全局 Moran 统计
# 输入: 存放 TIFF 栅格的文件夹
# 输出: LISA 分类栅格 + Global_Morans_I_Results.csv
# 使用: 修改 in_workspace → 运行 (需 ArcPy Spatial Analyst)
# 依赖: arcpy
# =============================================================================
import arcpy
import os
import csv

# ================= 环境变量配置 =================
# 1. 定义输入TIF栅格所在的文件夹路径
in_workspace = r"E:\数据分析\地理探测器\20260310\安多芬\重采样300"

# 2. 定义保存矢量点数据（.shp）的输出文件夹路径
out_vector_folder = r"E:\数据分析\地理探测器\20260310\安多芬\栅格转点"

# 3. 定义最终输出LISA分类栅格（.tif）的文件夹
out_raster_folder = r"E:\数据分析\地理探测器\20260310\安多芬\LISA栅格"

# 4. 定义全局空间自相关结果的CSV表格保存路径
csv_table_path = r"E:\数据分析\地理探测器\20260310\安多芬\LISA栅格\Global_Morans_I_Results.csv"

# 设置工作空间和允许覆盖输出
arcpy.env.workspace = in_workspace
arcpy.env.overwriteOutput = True
# ================================================

# 获取文件夹下所有的 tif 文件
tif_files = arcpy.ListRasters("*", "TIF")

# 初始化 CSV 表格并写入表头
with open(csv_table_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Raster_Name", "Morans_Index", "Expected_Index", "Variance", "Z_Score", "P_Value"])

for tif in tif_files:
    base_name = os.path.splitext(tif)[0]
    print(f"正在处理栅格: {base_name} ...")

    # 获取原栅格的像元大小并设置捕捉栅格
    desc = arcpy.Describe(tif)
    cell_size = desc.meanCellWidth
    arcpy.env.snapRaster = tif

    # ---------------------------------------------------------
    # 1. 栅格转点 (保存为 Shapefile)
    # ---------------------------------------------------------
    # 加上 .shp 后缀，确保保存在普通文件夹中
    out_point = os.path.join(out_vector_folder, f"{base_name}_Pts.shp")
    arcpy.conversion.RasterToPoint(tif, out_point, "Value")
    print(f"  -> 已保存栅格转点结果: {base_name}_Pts.shp")

    # ---------------------------------------------------------
    # 2. 优化异常值分析 (Anselin Local Moran's I)
    # ---------------------------------------------------------
    out_outlier = os.path.join(out_vector_folder, f"{base_name}_LISA.shp")
    arcpy.stats.OptimizedOutlierAnalysis(out_point, out_outlier, "grid_code")

    # ---------------------------------------------------------
    # 3. 重分类 COType 以备转回栅格
    # ---------------------------------------------------------
    # 注意：Shapefile 字段名最多 10 个字符，所以用 RCode
    reclass_field = "RCode"
    arcpy.management.AddField(out_outlier, reclass_field, "SHORT")

    code_block = """
def get_code(cotype):
    if cotype == 'HH': return 1
    elif cotype == 'HL': return 2
    elif cotype == 'LL': return 3
    elif cotype == 'LH': return 4
    else: return 5  # 包括空值在内的所有不显著区域
"""
    expression = "get_code(!COType!)"
    arcpy.management.CalculateField(out_outlier, reclass_field, expression, "PYTHON3", code_block)

    # ---------------------------------------------------------
    # 4. 分析结果点转回栅格
    # ---------------------------------------------------------
    out_outlier_raster = os.path.join(out_raster_folder, f"{base_name}_LISA_Res.tif")
    arcpy.conversion.PointToRaster(out_outlier, reclass_field, out_outlier_raster, cellsize=cell_size)
    print(f"  -> 已生成局部异常值分类栅格: {base_name}_LISA_Res.tif")

    # ---------------------------------------------------------
    # 5 & 6. 全局空间自相关分析并记录到表格
    # ---------------------------------------------------------
    try:
        global_res = arcpy.stats.SpatialAutocorrelation(
            out_point, "grid_code", "NO_REPORT",
            "INVERSE_DISTANCE", "EUCLIDEAN_DISTANCE", "ROW"
        )

        morans_i = global_res.getOutput(0)
        expected = global_res.getOutput(1)
        variance = global_res.getOutput(2)
        z_score = global_res.getOutput(3)
        p_value = global_res.getOutput(4)

        with open(csv_table_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([base_name, morans_i, expected, variance, z_score, p_value])
        print(f"  -> 全局自相关分析完成，结果已写入 CSV 表格。")

    except Exception as e:
        print(f"  -> {base_name} 全局空间自相关计算失败，错误原因: {e}")

print("\n--- 所有任务处理完成！ ---")
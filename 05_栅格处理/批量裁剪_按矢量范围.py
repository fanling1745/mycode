# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 按矢量中每个要素的范围批量裁剪所有栅格
# 输入: 栅格文件夹 + 含多个多边形的矢量 + ID 命名字段
# 输出: {ID值}_{栅格名}.tif
# 使用: 修改 in_raster_workspace / in_vector / out_workspace → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
import os
from arcpy.sa import *

# ================= 配置区域 =================
# 请将以下路径替换为你自己的实际路径
in_raster_workspace = r"E:\frag\20260304\crop\crop"  # 存放原始栅格的文件夹
in_vector = r"E:\frag\20260304\crop\shp\省范围.shp"  # 省份矢量数据路径
out_workspace = r"E:\frag\20260304\crop\clip"  # 裁剪结果保存的文件夹
id_field_name = "ID"  # 矢量数据中用于命名的字段名

# ================= 环境设置 =================
arcpy.env.workspace = in_raster_workspace
arcpy.env.overwriteOutput = True  # 允许覆盖同名输出文件

# 检查并获取 Spatial Analyst 扩展许可
arcpy.CheckOutExtension("Spatial")

# 获取文件夹下的所有栅格数据
raster_list = arcpy.ListRasters()
if not raster_list:
    print("未在输入工作空间中找到栅格数据，请检查路径！")
else:
    print(f"找到 {len(raster_list)} 个栅格数据，准备开始裁剪...")

# ================= 核心处理 =================
# 1. 创建要素图层 (SelectLayerByAttribute 必须在图层上运行)
layer_name = "prov_layer"
arcpy.MakeFeatureLayer_management(in_vector, layer_name)

# 获取矢量数据的 OID/FID 字段名 (shp通常是FID，gdb通常是OBJECTID)
oid_field = arcpy.Describe(layer_name).OIDFieldName

# 2. 使用搜索游标遍历每个省份多边形
# 读取 OID 用于选择，读取指定的 ID 字段用于命名
with arcpy.da.SearchCursor(layer_name, [oid_field, id_field_name]) as cursor:
    for row in cursor:
        oid = row[0]
        prov_id = str(row[1])  # 确保转换为字符串格式

        # 3. 按 OID 选中当前省份
        where_clause = f"{oid_field} = {oid}"
        arcpy.SelectLayerByAttribute_management(layer_name, "NEW_SELECTION", where_clause)

        # 4. 遍历所有栅格并进行裁剪
        for ras in raster_list:
            print(f"正在处理: 区域ID [{prov_id}] -> 栅格 [{ras}]")

            try:
                # 执行按掩膜提取 (此时图层处于选中状态，工具只会按照选中的多边形进行掩膜)
                out_extract = ExtractByMask(ras, layer_name)

                # 构造输出文件名：分离原栅格后缀，避免出现 ID_name.tif.tif 的情况
                ras_name_only = os.path.splitext(ras)[0]
                out_name = f"{prov_id}_{ras_name_only}.tif"
                out_path = os.path.join(out_workspace, out_name)

                # 保存提取结果
                out_extract.save(out_path)

            except Exception as e:
                print(f"处理 区域ID [{prov_id}] 的栅格 [{ras}] 时出错: {e}")

# ================= 清理工作 =================
arcpy.SelectLayerByAttribute_management(layer_name, "CLEAR_SELECTION")
arcpy.CheckInExtension("Spatial")
print("所有批量裁剪任务已顺利完成！")
# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 按矢量属性表字段值逐个裁剪栅格
# 输入: 单个栅格 + 含多个要素的矢量 + 命名字段
# 输出: {栅格名}_{字段值}.tif
# 使用: 修改 in_raster / in_vector / field_col → 运行
# 依赖: arcpy
# =============================================================================
import arcpy
import os
import re


def clip_raster_by_attributes_arcpy(raster_path, vector_path, field_name, output_dir):
    """
    使用 Arcpy 根据矢量字段裁剪栅格。
    命名规则：栅格原名_字段值.tif
    """

    # 1. 环境与路径设置
    arcpy.env.overwriteOutput = True
    # 推荐开启压缩，防止裁剪出的小图体积过大 (对应 rasterio 的 compress='lzw')
    arcpy.env.compression = "LZW"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"输出目录: {output_dir}")

    # 2. 获取栅格的基础名称 (用于文件命名)
    # 例如: "C:/Data/Landuse_2020.tif" -> "Landuse_2020"
    raster_basename = os.path.splitext(os.path.basename(raster_path))[0]

    print(f"正在处理栅格: {raster_basename}")

    # 3. 创建矢量的临时图层 (用于后续的选择操作)
    temp_layer = "temp_clip_layer"
    if arcpy.Exists(temp_layer):
        arcpy.Delete_management(temp_layer)
    arcpy.MakeFeatureLayer_management(vector_path, temp_layer)

    # 4. 遍历矢量数据的每一个要素
    # 获取 OID (用于选择) 和 命名字段 (用于命名)
    with arcpy.da.SearchCursor(vector_path, ["OID@", field_name]) as cursor:
        for row in cursor:
            oid = row[0]
            raw_field_val = str(row[1])  # 获取字段值

            # --- 文件名清洗 ---
            # 替换非法字符 (空格, /, \, :, *, ?, ", <, >, |) 为下划线
            safe_field_val = re.sub(r'[\\/:*?"<>| ]', '_', raw_field_val)

            # 拼接输出文件名: 栅格名_字段值.tif
            out_name = f"{raster_basename}_{safe_field_val}.tif"
            out_path = os.path.join(output_dir, out_name)

            print(f"  正在处理要素 OID {oid}: {safe_field_val} -> {out_name}")

            try:
                # 5. 选中当前这一个要素
                # 判断数据源类型以构建正确的 SQL 语句
                desc = arcpy.Describe(vector_path)
                oid_field = desc.OIDFieldName  # 自动获取 ID 字段名 (FID 或 OBJECTID)

                sql = f"{arcpy.AddFieldDelimiters(vector_path, oid_field)} = {oid}"
                arcpy.SelectLayerByAttribute_management(temp_layer, "NEW_SELECTION", sql)

                # 6. 执行裁剪 (Clip Management)
                # 关键点: clipping_geometry="ClippingGeometry" 等同于 rasterio 的 mask(crop=True)
                arcpy.Clip_management(
                    in_raster=raster_path,
                    rectangle="#",  # 使用选定要素的边界
                    out_raster=out_path,
                    in_template_dataset=temp_layer,  # 输入刚才选中的图层
                    nodata_value="#",  # 保持原图 NoData，或者在此指定具体数值
                    clipping_geometry="ClippingGeometry",  # 【必须】按多边形形状裁剪
                    maintain_clipping_extent="NO_MAINTAIN_EXTENT"
                )

            except arcpy.ExecuteError:
                # 捕捉 ArcPy 特有的地理处理错误 (如多边形在栅格范围外)
                print(f"  [ArcPy 警告] 裁剪失败 (可能不重叠): {arcpy.GetMessages(1)}")
            except Exception as e:
                print(f"  [系统 错误] {e}")

    # 清理
    arcpy.Delete_management(temp_layer)
    print("--- 全部完成 ---")


# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    # 配置路径
    in_raster = r"D:\frag\004\frag (2)\frag\lucc_40.tif"  # 你的栅格
    in_vector = r"D:\frag\004\frag (2)\frag\边界.shp"  # 你的矢量
    field_col = "编号"  # 你的命名字段
    out_save = r"D:\frag\004\frag (2)\frag\clip"  # 输出位置

    clip_raster_by_attributes_arcpy(in_raster, in_vector, field_col, out_save)
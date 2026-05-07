# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 栅格 NoData 像元全部填充为 0
# 输入: 栅格文件夹
# 输出: 处理后的栅格
# 使用: 修改 input_folder 和 output_folder → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
from arcpy.sa import *
import os

# ================= 配置部分 =================
# 输入文件夹路径（存放原始栅格）
input_folder = r"E:\frag\20260204\空间自相关\栅格计算结果"

# 输出文件夹路径（存放处理后的结果）
# 建议输出到新文件夹，避免直接覆盖原始数据导致不可逆错误
output_folder = r"E:\frag\20260204\空间自相关\去除nodata"

# ================= 环境设置 =================
# 检查并创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置工作空间
arcpy.env.workspace = input_folder
arcpy.env.overwriteOutput = True

# 检查 Spatial Analyst 许可
arcpy.CheckOutExtension("Spatial")

# ================= 主程序 =================
try:
    # 获取文件夹下所有的栅格文件 (TIF, IMG, 等)
    # 如果只要特定格式，可以写 arcpy.ListRasters("*", "TIF")
    raster_list = arcpy.ListRasters("*")

    if not raster_list:
        print(f"在 {input_folder} 中未找到栅格数据。")

    for raster_name in raster_list:
        print(f"正在处理: {raster_name} ...")

        # 1. 定义输出路径
        out_path = os.path.join(output_folder, raster_name)

        # 2. 执行地图代数运算
        # 逻辑：如果是 NoData (IsNull)，则赋值 0，否则保持原值
        in_raster = arcpy.Raster(raster_name)
        out_raster = Con(IsNull(in_raster), 0, in_raster)

        # 3. 保存结果
        # 这里你可以设置压缩方式，例如 LZW，以减小文件体积
        arcpy.env.compression = "LZW"
        out_raster.save(out_path)

        print(f"  --> 已保存至: {out_path}")

    print("\n所有文件处理完成！")

except arcpy.ExecuteError:
    # 捕获 ArcPy 报错
    print(arcpy.GetMessages(2))
except Exception as e:
    # 捕获其他 Python 报错
    print(e)
finally:
    # 归还许可
    arcpy.CheckInExtension("Spatial")
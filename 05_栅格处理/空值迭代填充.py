# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 迭代填充栅格空值 (10×10 窗口焦点均值), 直到填满或达最大迭代次数
# 输入: GDB 中的栅格 + 矢量掩膜
# 输出: 填充完成后的 TIFF
# 使用: 修改 GDB_PATH / VECTOR_MASK_PATH / OUTPUT_DIR → 运行
# 依赖: arcpy (Spatial Analyst)
# =============================================================================
import arcpy
import os


def auto_iterative_fill(input_gdb, output_folder, vector_mask, max_iterations=200):
    """
    自动识别并循环填充矢量范围内的所有栅格空值，直到完全填满。
    """
    # 1. 检查空间分析许可与环境设置
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.workspace = input_gdb
    arcpy.env.overwriteOutput = True

    # 核心：将处理范围和分析掩膜严格限制为矢量数据
    arcpy.env.mask = vector_mask
    arcpy.env.extent = vector_mask

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    raster_list = arcpy.ListRasters()
    if not raster_list:
        print("在指定的 GDB 中没有找到栅格数据。")
        return

    print(f"共找到 {len(raster_list)} 个栅格数据，开始自动迭代填充...")

    # 2. 遍历处理每个栅格
    for ras_name in raster_list:
        print(f"--------------------------------------------------")
        print(f"正在处理: {ras_name} ...")

        try:
            # 初始读取，并先用矢量掩膜截取一次，确保只关注矢量范围内的数据
            current_raster = arcpy.sa.ExtractByMask(arcpy.Raster(ras_name), vector_mask)

            iteration = 0
            while iteration < max_iterations:
                # 步骤 A: 检查是否还有空值
                # IsNull 会将 NoData 赋值为 1，有数据的像元赋值为 0
                isnull_raster = arcpy.sa.IsNull(current_raster)

                # 获取 IsNull 栅格的最大值。如果最大值是 0，说明范围内已经没有 NoData 了
                max_isnull_val = int(arcpy.GetRasterProperties_management(isnull_raster, "MAXIMUM").getOutput(0))

                if max_isnull_val == 0:
                    print(f"  --> 自动填充完成！共迭代了 {iteration} 次，空缺已全部填满。")
                    break

                # 步骤 B: 如果还有空值，执行一次 3x3 焦点统计
                focal_mean = arcpy.sa.FocalStatistics(current_raster, arcpy.sa.NbrRectangle(10, 10, "CELL"), "MEAN",
                                                      "DATA")

                # 步骤 C: 将空缺部分用焦点统计结果替换，原有正常值保持不变
                current_raster = arcpy.sa.Con(isnull_raster, focal_mean, current_raster)

                iteration += 1

            # 安全机制提示
            if iteration == max_iterations:
                print(
                    f"  --> [警告] 达到了最大保护迭代次数 {max_iterations}，可能由于缺失面积过大，最中心极少部分仍未填满。")

            # 3. 严格按矢量边界再次提取并保存为 TIF
            final_raster = arcpy.sa.ExtractByMask(current_raster, vector_mask)
            out_tif_path = os.path.join(output_folder, f"{ras_name}.tif")
            final_raster.save(out_tif_path)

            print(f"  --> 成功导出至: {out_tif_path}")

        except Exception as e:
            print(f"  [错误] 处理 {ras_name} 时发生异常: {e}")

    arcpy.CheckInExtension("Spatial")
    print("所有栅格处理完毕！")


# ==========================================
# 用户自定义参数区域
# ==========================================
if __name__ == "__main__":
    # 1. 存放原始栅格数据的文件地理数据库路径
    GDB_PATH = r"E:\数据分析\水源涵养\重采样.gdb"

    # 2. 规定填充范围的矢量数据路径
    VECTOR_MASK_PATH = r"E:\数据分析\水源涵养\20260209_息澜\重采样\韩江流域.shp"

    # 3. 最终 TIF 文件保存的文件夹路径
    OUTPUT_DIR = r"E:\数据分析\水源涵养\20260209_息澜\填充空值"

    # 4. 最大迭代次数（防止死循环，默认 200 次，如果你的空值斑块极其巨大可以调高）
    MAX_ITER = 20

    # 执行函数
    auto_iterative_fill(GDB_PATH, OUTPUT_DIR, VECTOR_MASK_PATH, MAX_ITER)
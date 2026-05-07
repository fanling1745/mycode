# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 为地理探测器准备数据 (渔网内 PM2.5 均值 + 各地类面积占比)
# 输入: 渔网矢量 + PM2.5 栅格 + CLCD 土地利用栅格
# 输出: CSV (Zone_ID / y / x1-x8)
# 使用: 修改 fishnet_fc / pm25_raster / clcd_raster → 运行
# 依赖: arcpy, pandas
# =============================================================================
import arcpy
import pandas as pd
import os

# ================= 核心参数配置区 (请直接在此修改路径) =================
# 建议使用 File Geodatabase 存放全国尺度的中间结果，以保证读写速度和稳定性
temp_workspace = r"E:\数据分析\空间自相关\20260306\结果\NRSEI.gdb"
arcpy.env.workspace = temp_workspace
arcpy.env.overwriteOutput = True

# 输入数据路径
fishnet_fc = r"E:\数据分析\地理探测器\202605\0507\无名之辈\渔网10km.shp"  # 渔网矢量数据
zone_field = "FID"  # 渔网的唯一标识字段 (如 FID, OBJECTID 等)

pm25_raster = r"E:\数据分析\地理探测器\202605\0507\无名之辈\pm25.tif"  # 全国 PM2.4 栅格
clcd_raster = r"E:\数据分析\地理探测器\202605\0507\无名之辈\CLCD_2023.tif"  # 全国 CLCD 栅格

# 输出结果路径
out_csv = r"E:\数据分析\地理探测器\202605\0507\无名之辈\fishnet_result.csv"

# 环境变量：将处理的像元捕捉对齐到 CLCD 栅格，确保面积计算精度
arcpy.env.snapRaster = clcd_raster


# =====================================================================

def process_spatial_data():
    print("开始处理空间数据...")

    # 中间表名称
    pm25_table = "pm25_zonal_stats"
    lu_table = "lu_tabulate_area"

    # ---------------------------------------------------------
    # 步骤 1: 计算渔网内 PM2.5 的平均值 (y)
    # ---------------------------------------------------------
    print("1/4: 正在计算 PM2.5 平均值...")
    arcpy.sa.ZonalStatisticsAsTable(
        in_zone_data=fishnet_fc,
        zone_field=zone_field,
        in_value_raster=pm25_raster,
        out_table=pm25_table,
        ignore_nodata="DATA",
        statistics_type="MEAN"
    )

    # ---------------------------------------------------------
    # 步骤 2: 计算渔网内各土地利用类型的面积
    # ---------------------------------------------------------
    print("2/4: 正在统计 CLCD 土地利用分类面积...")
    arcpy.sa.TabulateArea(
        in_zone_data=fishnet_fc,
        zone_field=zone_field,
        in_class_data=clcd_raster,
        class_field="Value",
        out_table=lu_table
    )

    # ---------------------------------------------------------
    # 步骤 3: 读取内存字典并计算占比
    # ---------------------------------------------------------
    print("3/4: 正在读取表数据并计算 x1-x8 与 y...")

    # 构建 PM2.4 字典: {zone_id: mean_value}
    pm_dict = {}
    with arcpy.da.SearchCursor(pm25_table, [zone_field, "MEAN"]) as cursor:
        for row in cursor:
            pm_dict[row[0]] = row[1]

    # 定义 CLCD 编码到需求变量名的严格映射 (注意跳过 VALUE_6 冰雪)
    lu_mapping = {
        'VALUE_1': 'x1',  # 耕地
        'VALUE_2': 'x2',  # 林地
        'VALUE_3': 'x3',  # 灌木
        'VALUE_4': 'x4',  # 草地
        'VALUE_5': 'x5',  # 水体
        'VALUE_7': 'x6',  # 裸地
        'VALUE_8': 'x7',  # 建设用地
        'VALUE_9': 'x8'  # 湿地
    }

    result_data = []
    lu_fields = [f.name for f in arcpy.ListFields(lu_table)]

    with arcpy.da.SearchCursor(lu_table, "*") as cursor:
        for row in cursor:
            row_dict = dict(zip(lu_fields, row))
            zid = row_dict[zone_field]

            # 计算该网格内的有效像元总面积（作为计算占比的分母）
            # 必须包含 VALUE_6 (如果有)，保证总体物理面积占比为真实分布
            total_area = sum([v for k, v in row_dict.items() if k.startswith('VALUE_') and v is not None])

            if total_area == 0:
                continue  # 空网格跳过

            # 初始化该网格的结果字典
            cell_result = {
                'Zone_ID': zid,
                'y': pm_dict.get(zid, None)  # 提取 PM2.4 均值，若无则为 None
            }

            # 遍历映射表，计算 x1 - x8 的占比
            for val_field, x_name in lu_mapping.items():
                area = row_dict.get(val_field, 0)
                if area is None:
                    area = 0
                cell_result[x_name] = area / total_area

            result_data.append(cell_result)

    # ---------------------------------------------------------
    # 步骤 4: 导出格式化 CSV
    # ---------------------------------------------------------
    print("4/4: 正在格式化表格并导出 CSV 文件...")
    df = pd.DataFrame(result_data)

    # 规范化列名排序，处理特定类型在全网格完全缺失的极端异常情况
    cols = ['Zone_ID', 'y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[cols]

    # 强制以 utf-8 编码导出为 CSV
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"任务完成！结果已成功保存至: {out_csv}")


if __name__ == "__main__":
    process_spatial_data()
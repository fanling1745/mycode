# -*- coding: utf-8 -*-
# =============================================================================
# 功能: CSV 格式 POI 数据批量转为 ArcGIS 点要素类 (按大类分组)
# 输入: 含经度/纬度/大类字段的 CSV 文件夹
# 输出: 每类 POI 一个点要素类, 存入指定 GDB
# 使用: 修改 input_folder / output_gdb / 字段映射 → 运行
# 依赖: arcpy, pandas
# =============================================================================
import arcpy
import os
import pandas as pd
import re

# ================= 配置区域 =================
# 1. 存放所有地市Excel表格的文件夹路径
input_folder = r"E:\数据分析\论文\浙江人均居民收入空间化_乐\数据准备\原始数据\浙江省POI数据"

# 2. 输出的Geodatabase数据库路径 (建议提前在ArcGIS中建好)
output_gdb = r"D:\GD\MyProject\poi.gdb"

# 3. 字段名称匹配 (请根据你的 CSV 表头严格对应)
lon_field = "经度"
lat_field = "纬度"
category_field = "大类"

# ============================================

arcpy.env.workspace = output_gdb
arcpy.env.overwriteOutput = True  # 允许覆盖同名文件

# 定义空间参考：WGS84 地理坐标系 (EPSG: 4326)
sr = arcpy.SpatialReference(4326)
temp_dir = arcpy.env.scratchFolder  # 获取系统临时文件夹存放中间文件


def process_csv_poi():
    print("开始读取并合并 CSV 文件...")
    all_data = []

    # 1. 遍历并合并所有 CSV 表
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(input_folder, file)
            try:
                # 尝试用 utf-8 读取
                df = pd.read_csv(file_path, encoding='utf-8')
                all_data.append(df)
                print(f"  成功读取: {file} (共 {len(df)} 条, utf-8 编码)")
            except UnicodeDecodeError:
                try:
                    # 如果 utf-8 报错，尝试用 gbk (国内 Windows 常用编码) 读取
                    df = pd.read_csv(file_path, encoding='gbk')
                    all_data.append(df)
                    print(f"  成功读取: {file} (共 {len(df)} 条, gbk 编码)")
                except Exception as e:
                    print(f"  [错误] 读取 {file} 失败，请检查编码格式: {e}")

    if not all_data:
        print("未在文件夹中找到有效 CSV 文件，请检查路径！")
        return

    # 2. 合并全省数据
    merged_df = pd.concat(all_data, ignore_index=True)

    # 清理列名中的多余空格（防止爬虫数据带有隐形空格）
    merged_df.columns = merged_df.columns.str.strip()

    print(f"\n数据合并完成，全省 POI 总数: {len(merged_df)} 条")

    # 3. 获取所有不重复的“大类”名称
    if category_field not in merged_df.columns:
        print(f"[错误] 在合并后的数据中找不到字段名 '{category_field}'，请检查配置！")
        return

    unique_categories = merged_df[category_field].dropna().unique()
    print(f"识别出 {len(unique_categories)} 个 POI 大类。开始生成点要素...\n")

    # 4. 按大类筛选并生成点数据
    for category in unique_categories:
        # 清理名称中的非法字符（数据库要素类命名规范：只能包含字母、数字、下划线）
        clean_name = re.sub(r'[^\w\u4e00-\u9fa5]', '', str(category))
        fc_name = f"POI_{clean_name}"

        print(f"正在处理: {category} -> 准备输出: {fc_name}")

        # 筛选当前大类的数据
        category_df = merged_df[merged_df[category_field] == category]

        # 导出为临时 CSV 给 ArcPy 读取 (这一步能有效避开由于 Pandas 直接写 Shapefile 的各种依赖问题)
        temp_csv = os.path.join(temp_dir, f"temp_{clean_name}.csv")
        category_df.to_csv(temp_csv, index=False, encoding='utf-8-sig')

        out_feature_class = os.path.join(output_gdb, fc_name)
        event_layer = f"Layer_{clean_name}"

        try:
            # 根据临时 CSV 中的经纬度创建 XY 事件图层
            arcpy.MakeXYEventLayer_management(
                table=temp_csv,
                in_x_field=lon_field,
                in_y_field=lat_field,
                out_layer=event_layer,
                spatial_reference=sr
            )

            # 将事件图层保存为永久的要素类
            arcpy.CopyFeatures_management(event_layer, out_feature_class)

            # 清理内存与临时文件
            arcpy.Delete_management(event_layer)
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

            print(f"  [完成] 成功导出: {fc_name} (包含 {len(category_df)} 个点)\n")

        except Exception as e:
            print(f"  [失败] 处理 {category} 时出现 ArcPy 报错: {e}\n")

    print("-" * 30)
    print(f"所有操作执行完毕！请前往 {output_gdb} 查看生成的 {len(unique_categories)} 类 POI 点数据。")


if __name__ == "__main__":
    process_csv_poi()
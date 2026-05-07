# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 基于 5 级破碎化/干旱化等级, 计算空间耦合关系 (11-55) 和滞后类型 (同步/滞后)
# 支持栅格模式和矢量模式两种输入
# 输出: 空间关系栅格 / 滞后类型栅格 (或含 Type_A/Type_B 字段的矢量)
# 使用: 修改 input_folder / output_folder / process_years → 运行
# 依赖: rasterio, numpy (栅格模式) 或 geopandas, pandas (矢量模式)
# =============================================================================
import os
import numpy as np
import rasterio


def classify_data_5levels(data, thresholds):
    """
    基于5级标准进行重分类
    thresholds: [0.2, 0.4, 0.6, 0.8]

    返回等级:
    1: 0.0 - 0.2 (低)
    2: 0.2 - 0.4 (较低)
    3: 0.4 - 0.6 (一般)
    4: 0.6 - 0.8 (高)
    5: 0.8 - 1.0 (剧烈)
    """
    # 创建一个全0矩阵，保持mask状态
    classified = np.ma.zeros(data.shape, dtype=np.uint8)
    classified.mask = data.mask  # 继承掩膜

    # 获取有效数据区域
    valid_mask = ~data.mask

    # --- 开始分级 (1-5) ---

    # Level 1: < 0.2
    classified[valid_mask & (data < thresholds[0])] = 1

    # Level 2: 0.2 <= x < 0.4
    classified[valid_mask & (data >= thresholds[0]) & (data < thresholds[1])] = 2

    # Level 3: 0.4 <= x < 0.6
    classified[valid_mask & (data >= thresholds[1]) & (data < thresholds[2])] = 3

    # Level 4: 0.6 <= x < 0.8
    classified[valid_mask & (data >= thresholds[2]) & (data < thresholds[3])] = 4

    # Level 5: >= 0.8
    classified[valid_mask & (data >= thresholds[3])] = 5

    return classified


def analyze_overlay_types_5levels(data_dir, output_dir, years):
    """
    计算5级分类下的耦合关系类型(图a)和滞后类型(图b)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 设置新的5级阈值 ===
    # 对应: 0.2, 0.4, 0.6, 0.8
    thresholds = [0.2, 0.4, 0.6, 0.8]
    # ======================

    for year in years:
        print(f"正在分析 {year} 年 (5级分类)...")

        frag_file = os.path.join(data_dir, f"masked_fragmentation_{year}.tif")
        tvdi_file = os.path.join(data_dir, f"TVDI_Cropland_{year}.tif")

        if not os.path.exists(frag_file) or not os.path.exists(tvdi_file):
            print(f"跳过 {year}: 文件缺失")
            continue

        with rasterio.open(frag_file) as src_frag, rasterio.open(tvdi_file) as src_tvdi:
            # 读取数据
            data_frag = src_frag.read(1, masked=True).astype('float32')
            data_tvdi = src_tvdi.read(1, masked=True).astype('float32')

            meta = src_frag.meta.copy()
            meta.update(dtype='uint8', compress='lzw', nodata=0)

            # --- 步骤 1: 数据分级 (1-5) ---
            # 1=低, 2=较低, 3=一般, 4=高, 5=剧烈
            class_frag = classify_data_5levels(data_frag, thresholds)
            class_tvdi = classify_data_5levels(data_tvdi, thresholds)

            # --- 步骤 2: 生成图 6a (耦合关系类型图 - 组合矩阵) ---
            # 结果将是两位数: XY
            # X (十位) = 破碎化等级 (1-5)
            # Y (个位) = 石漠化/干旱等级 (1-5)
            # 范围从 11 (低-低) 到 55 (剧烈-剧烈)

            type_a = np.ma.zeros(class_frag.shape, dtype=np.uint8)
            type_a.mask = class_frag.mask | class_tvdi.mask

            valid_locs = ~type_a.mask
            # 计算组合码
            type_a[valid_locs] = class_frag[valid_locs] * 10 + class_tvdi[valid_locs]

            # --- 步骤 3: 生成图 6b (同步/滞后类型图) ---
            # 1: 同步型 (等级相同)
            # 2: 石漠化滞后型 (TVDI < Frag)
            # 3: 破碎化滞后型 (TVDI > Frag)

            type_b = np.ma.zeros(class_frag.shape, dtype=np.uint8)
            type_b.mask = type_a.mask

            # 判断逻辑
            type_b[valid_locs & (class_frag == class_tvdi)] = 1  # 同步
            type_b[valid_locs & (class_tvdi < class_frag)] = 2  # 石漠化滞后
            type_b[valid_locs & (class_tvdi > class_frag)] = 3  # 破碎化滞后

            # --- 保存结果 ---

            # 保存 Type A (组合关系 11-55)
            out_a_path = os.path.join(output_dir, f"空间关系_{year}.tif")
            with rasterio.open(out_a_path, 'w', **meta) as dst:
                dst.write(type_a.filled(0), 1)

            # 保存 Type B (滞后关系 1,2,3)
            out_b_path = os.path.join(output_dir, f"滞后类型_{year}.tif")
            with rasterio.open(out_b_path, 'w', **meta) as dst:
                dst.write(type_b.filled(0), 1)

            print(f"完成 {year} 年: 结果已保存。")


# ================= 配置与运行 =================
# 请修改为您的实际路径
input_folder = r"E:\方\fragststs\空间自相关\shuju\shuju"
output_folder = r"E:\方\fragststs\空间自相关\shuju\空间关系判断"
process_years = [2000, 2005, 2010,2015,2020,2022]

if __name__ == "__main__":
    analyze_overlay_types_5levels(input_folder, output_folder, process_years)
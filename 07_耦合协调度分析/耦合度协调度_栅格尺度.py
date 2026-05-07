# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 基于破碎化和干旱化栅格, 计算耦合度 C 和协调度 D (含分级)
# 输入: masked_fragmentation_{year}.tif + TVDI_Cropland_{year}.tif
# 输出: C/D 数值栅格 + C/D 分级栅格 (1-4级)
# 使用: 修改 input_folder / output_folder / years_to_process → 运行
# 依赖: rasterio, numpy
# =============================================================================
import os
import numpy as np
import rasterio


def classify_raster(data, thresholds, values):
    """
    根据阈值对栅格数据进行分级
    data: 输入的numpy数组 (masked array)
    thresholds: 阈值列表 [t1, t2, t3]
    values: 对应区间的值列表 [v1, v2, v3, v4]

    区间规则:
    < t1 -> v1
    t1 <= x < t2 -> v2
    t2 <= x < t3 -> v3
    >= t3 -> v4
    """
    # 创建一个全为0的数组，大小与输入一致，类型为整数
    classified = np.zeros(data.shape, dtype=np.uint8)

    # 获取有效数据的掩膜 (非 NoData)
    valid_mask = ~data.mask

    # 提取有效数据
    valid_data = data[valid_mask]

    # 使用 numpy.select 或条件索引进行分级
    # 这里的条件是互斥的
    cond1 = valid_data < thresholds[0]
    cond2 = (valid_data >= thresholds[0]) & (valid_data < thresholds[1])
    cond3 = (valid_data >= thresholds[1]) & (valid_data < thresholds[2])
    cond4 = valid_data >= thresholds[2]

    # 将分级值填入对应的位置
    # 创建临时数组存放分类结果
    temp_res = np.zeros(valid_data.shape, dtype=np.uint8)
    temp_res[cond1] = values[0]
    temp_res[cond2] = values[1]
    temp_res[cond3] = values[2]
    temp_res[cond4] = values[3]

    # 将分类结果填回原始形状的数组中
    classified[valid_mask] = temp_res

    return classified, valid_mask


def process_year_data(year, data_dir, output_dir):
    """处理单一年份的数据"""
    print(f"正在处理 {year} 年的数据...")

    frag_file = os.path.join(data_dir, f"masked_fragmentation_{year}.tif")
    tvdi_file = os.path.join(data_dir, f"TVDI_Cropland_{year}.tif")

    if not os.path.exists(frag_file) or not os.path.exists(tvdi_file):
        print(f"警告: 找不到 {year} 年的文件，跳过。")
        return

    with rasterio.open(frag_file) as src_u, rasterio.open(tvdi_file) as src_e:
        # 读取数据 (自动处理 NoData)
        U = src_u.read(1, masked=True).astype('float32')
        E = src_e.read(1, masked=True).astype('float32')

        meta = src_u.meta.copy()

        # 1. 计算耦合度 C
        sum_ue = U + E
        denom = sum_ue ** 2

        # 初始化 C
        C = np.ma.zeros(U.shape, dtype='float32')
        C.mask = U.mask | E.mask | (denom == 0)  # 更新掩膜

        valid_locs = ~C.mask

        # 计算 C 值
        term = (U[valid_locs] * E[valid_locs]) / denom[valid_locs]
        C[valid_locs] = 2 * np.sqrt(term)

        # 2. 计算耦合协调度 D
        alpha, beta = 0.5, 0.5
        T = alpha * U + beta * E
        D = np.ma.sqrt(C * T)

        # --- 保存数值结果 (Float) ---
        meta.update(dtype='float32', compress='lzw', nodata=None)  # 这里假设输出不需要特定的nodata值，或者沿用输入的

        # 写入 C 值
        out_c_val = os.path.join(output_dir, f"Coupling_Degree_C_Value_{year}.tif")
        with rasterio.open(out_c_val, 'w', **meta) as dst:
            dst.write(C.filled(np.nan), 1)  # 用NaN填充无效值

        # 写入 D 值
        out_d_val = os.path.join(output_dir, f"Coordination_Degree_D_Value_{year}.tif")
        with rasterio.open(out_d_val, 'w', **meta) as dst:
            dst.write(D.filled(np.nan), 1)

        # --- 进行分级 (Classify) ---

        # C 的分级阈值: 0.3, 0.5, 0.8
        c_class, mask_c = classify_raster(C, [0.3, 0.5, 0.8], [1, 2, 3, 4])

        # D 的分级阈值: 0.3, 0.6, 0.8  <-- 注意这里第二个阈值是 0.6
        d_class, mask_d = classify_raster(D, [0.3, 0.6, 0.8], [1, 2, 3, 4])

        # 更新元数据为整数型 (Byte) 用于保存分类结果
        meta.update(dtype='uint8', compress='lzw', nodata=0)

        # 写入 C 分级
        out_c_class = os.path.join(output_dir, f"Coupling_Degree_C_Class_{year}.tif")
        with rasterio.open(out_c_class, 'w', **meta) as dst:
            # 这里的 0 是背景/NoData
            dst.write(c_class, 1)

        # 写入 D 分级
        out_d_class = os.path.join(output_dir, f"Coordination_Degree_D_Class_{year}.tif")
        with rasterio.open(out_d_class, 'w', **meta) as dst:
            dst.write(d_class, 1)

        print(f"完成 {year} 年: 结果已保存。")


# ================= 配置路径并运行 =================

# 输入数据文件夹路径
input_folder = r"E:\方\fragststs\空间自相关\shuju\shuju"
# 输出结果文件夹路径
output_folder = r"E:\方\fragststs\空间自相关\shuju\耦合度协调度计算"
years_to_process = [2000, 2005, 2010,2015,2020,2022]

if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for year in years_to_process:
        process_year_data(year, input_folder, output_folder)
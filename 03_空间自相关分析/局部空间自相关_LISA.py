# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 栅格级全局 + 局部 Moran's I (LISA) → 分类栅格输出
# 输入: 存放 TIFF 栅格的文件夹
# 输出: {文件名}_LISA_Clust.tif (1=HH,2=LH,3=LL,4=HL) + Global_Moran_Stats.xlsx
# 使用: 修改 INPUT_FOLDER 和 OUTPUT_FOLDER → 运行
# 依赖: libpysal, esda, rasterio, numpy, pandas
# =============================================================================
import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 1. 参数配置区域 =================

# 输入数据文件夹
INPUT_FOLDER = r'E:\frag\20260204\景观格局尺度指数'

# 输出结果文件夹 (生成的tif都在这里)
OUTPUT_FOLDER = r'E:\frag\20260204\空间自相关'

# 判定为 0 的阈值 (用于处理浮点数的 -0.00000...1 这种情况)
# 如果您的数据是整型，这个逻辑也兼容
ZERO_THRESHOLD = 1e-9


# ================= 2. 核心功能函数 =================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_year(filename):
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None


def write_raster(output_path, array, profile):
    """
    写入 TIF，强制将 NodData 设为 None (因为用户要求背景也是 0，即有效值)
    或者根据需要设为 -9999，但在本需求中，用户希望 0 代表无意义
    """
    profile.update({
        'driver': 'GTiff',
        'dtype': rasterio.float32,
        'count': 1,
        'compress': 'lzw',
        'nodata': None  # 关键：不设置 NoData，让 0 成为普通值
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(rasterio.float32), 1)


# ================= 3. 主程序逻辑 =================

def main():
    print("=== 开始空间自相关分析 (严格剔除 0/-0 版) ===")
    ensure_dir(OUTPUT_FOLDER)

    tif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    if not tif_files:
        print("未找到 .tif 文件")
        return

    global_results_list = []

    for idx, tif_path in enumerate(tif_files):
        filename = os.path.basename(tif_path)
        year = extract_year(filename)
        print(f"\n[{idx + 1}/{len(tif_files)}] 正在处理: {filename}")

        with rasterio.open(tif_path) as src:
            data = src.read(1)  # 读取第一波段
            profile = src.profile
            transform = src.transform
            src_nodata = src.nodata

            # --- 步骤 1: 严格的数据清洗 (Mask构建) ---
            # 逻辑：保留 (非NoData) 且 (非NaN) 且 (绝对值大于阈值) 的像素

            # 1. 基础有效性 (非 NaN, 非 Inf)
            mask = np.isfinite(data)

            # 2. 剔除原始文件的 NoData 标记
            if src_nodata is not None:
                # 注意浮点数比较
                mask = mask & (~np.isclose(data, src_nodata))

            # 3. 严格剔除 0 和 -0 (通过绝对值判断)
            # 凡是 abs(value) <= 0.000000001 的都视为 0，不参与计算
            mask = mask & (np.abs(data) > ZERO_THRESHOLD)

            # 获取有效数据的索引 (Row, Col)
            rows, cols = np.where(mask)
            values = data[rows, cols]

            count = len(values)
            print(f"   -> 原始像元数: {data.size}")
            print(f"   -> 剔除 0/-0/NoData 后的参与计算像元数: {count}")

            if count < 10:
                print("   -> [跳过] 有效样本太少")
                continue

            # --- 步骤 2: 构建权重矩阵 (KNN) ---
            # 计算有效像素的中心坐标
            xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
            coords = list(zip(xs, ys))

            try:
                # k=8 表示周围8个邻居
                w = KNN.from_array(np.array(coords), k=8)
                w.transform = 'r'  # 行标准化
            except Exception as e:
                print(f"   -> [错误] 权重构建失败: {e}")
                continue

            # --- 步骤 3: 全局莫兰指数 ---
            try:
                mi = Moran(values, w)
                global_results_list.append({
                    '文件名': filename,
                    '年份': year,
                    'Global_Moran_I': mi.I,
                    'Z_Score': mi.z_norm,
                    'P_Value': mi.p_norm
                })
                print(f"   -> Global Moran's I: {mi.I:.4f}")
            except Exception as e:
                print(f"   -> [错误] 全局计算失败: {e}")
                continue

            # --- 步骤 4: 局部莫兰指数 (LISA) 与 结果重构 ---
            try:
                # 计算 LISA
                lisa = Moran_Local(values, w)

                # 提取结果
                sigs = lisa.p_sim < 0.05  # 显著性筛选 (p < 0.05)
                quadrants = lisa.q  # 象限 (1=HH, 2=LH, 3=LL, 4=HL)

                # === 生成分类编码 ===
                # 默认所有有效点先设为 0 (不显著)
                cluster_vals = np.zeros(count, dtype=np.int32)

                # 仅修改显著的点
                cluster_vals[sigs & (quadrants == 1)] = 1  # HH
                cluster_vals[sigs & (quadrants == 2)] = 2  # LH
                cluster_vals[sigs & (quadrants == 3)] = 3  # LL
                cluster_vals[sigs & (quadrants == 4)] = 4  # HL

                # === 关键：重构回二维栅格 ===
                # 1. 初始化一个全为 0 的矩阵 (满足您“背景为 0”的要求)
                #    这样，原本被 Mask 掉的 0/-0/NoData 区域，在这里天然就是 0
                out_array = np.zeros(data.shape, dtype=np.float32)

                # 2. 仅将计算出的分类结果填入有效位置
                out_array[rows, cols] = cluster_vals

                # === 保存结果 ===
                base_name = os.path.splitext(filename)[0]
                out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_LISA_Clust.tif")

                write_raster(out_path, out_array, profile)
                print(f"   -> 局部结果已保存: {os.path.basename(out_path)}")

            except Exception as e:
                print(f"   -> [错误] 局部计算失败: {e}")

    # --- 步骤 5: 保存统计表 ---
    if global_results_list:
        df = pd.DataFrame(global_results_list)
        df = df.sort_values(by=['年份', '文件名'])
        df.to_excel(os.path.join(OUTPUT_FOLDER, 'Global_Moran_Stats.xlsx'), index=False)
        print("\n=== 处理完成 ===")


if __name__ == "__main__":
    main()
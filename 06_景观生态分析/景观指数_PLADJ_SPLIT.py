# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 分块计算大栅格 PLADJ (相似邻接比) 和 SPLIT (分离指数)
# 输入: 土地利用栅格 + 目标地类编码 + output_res_meters 输出分辨率
# 输出: {地类}_SPLIT.tif / {地类}_PLADJ.tif
# 使用: 修改 input_raster 和 target_val → 运行
# 依赖: rasterio, numpy, scipy, tqdm
# =============================================================================
import rasterio
import numpy as np
from rasterio.windows import Window
from scipy.ndimage import label
from tqdm import tqdm
import os


# --- 核心算法 (保持不变，针对内存块计算) ---

def calculate_split_numpy(data_chunk, target_val):
    """
    计算 SPLIT (分离指数)
    只关注 data_chunk 中等于 target_val 的像素
    """
    # 1. 生成二值掩膜 (目标类=1, 其他=0)
    mask = (data_chunk == target_val)

    # 如果窗口里没有这个地类，直接返回 NaN
    if not np.any(mask):
        return np.nan

        # 2. 标记连通区域 (8邻域)
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(mask, structure=structure)

    if num_features == 0:
        return np.nan

    # 3. 计算每个斑块的面积
    # bincount统计每个ID出现的次数(即面积)
    patch_areas = np.bincount(labeled_array.ravel())[1:]  # [1:]去掉背景0

    # 4. 景观总面积 (窗口总面积)
    total_area = data_chunk.size

    # 5. SPLIT 公式: A^2 / sum(ai^2)
    sum_squared_patch_area = np.sum(patch_areas ** 2)
    if sum_squared_patch_area == 0:
        return np.nan

    split_index = (total_area ** 2) / sum_squared_patch_area
    return split_index


def calculate_pladj_numpy(data_chunk, target_val):
    """
    计算 PLADJ (相似邻接比)
    """
    # 1. 转为二值图 (目标类=1, 其他类/背景=0)
    # 这样我们可以复用之前的逻辑：
    # 耕地vs耕地 = 1+1=2 (Like Adjacency)
    # 耕地vs林地 = 1+0=1 (Non-Like Adjacency)
    binary_map = (data_chunk == target_val).astype(int)

    if np.sum(binary_map) == 0:
        return np.nan

    # 2. 矩阵错位计算相邻关系
    # 水平相邻
    h_diff = binary_map[:, :-1] + binary_map[:, 1:]
    # 垂直相邻
    v_diff = binary_map[:-1, :] + binary_map[1:, :]

    # 3. 统计边数
    # 同类相邻 (Like Adjacencies): 值等于2的位置
    like_adj = np.sum(h_diff == 2) + np.sum(v_diff == 2)

    # 总相邻 (Total Adjacencies involving class i):
    # 即 (Target, Target) 和 (Target, Non-Target)
    # 在二值图中，这就是所有值 >= 1 的情况
    total_adj = np.sum(h_diff >= 1) + np.sum(v_diff >= 1)

    if total_adj == 0:
        return 0

    pladj = (like_adj / total_adj) * 100
    return pladj


def process_metric(input_path, output_path, target_val, metric='SPLIT', output_res_meters=1000):
    """
    通用处理函数: 读取大图 -> 分块计算 -> 写入结果
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到文件 {input_path}")
        return

    print(f"🚀 开始处理: {metric} | 目标像素值: {target_val} | 输出: {os.path.basename(output_path)}")

    with rasterio.open(input_path) as src:
        # 1. 计算窗口参数
        pixel_size_x = src.transform[0]  # 假设单位是米
        window_size_px = int(output_res_meters / pixel_size_x)

        out_width = src.width // window_size_px
        out_height = src.height // window_size_px

        # 防止分辨率设置过大导致输出为空
        if out_width == 0 or out_height == 0:
            print("❌ 错误: 计算窗口比原图还大，请检查分辨率设置。")
            return

        # 计算新的地理变换参数
        new_transform = src.transform * src.transform.scale(window_size_px, window_size_px)

        # 更新元数据
        profile = src.profile
        profile.update({
            'driver': 'GTiff',
            'height': out_height,
            'width': out_width,
            'transform': new_transform,
            'count': 1,
            'dtype': rasterio.float32,
            'nodata': -9999
        })

        # 2. 执行计算并写入
        with rasterio.open(output_path, 'w', **profile) as dst:
            # 遍历每一行块
            for i in tqdm(range(out_height), desc=f"计算中 (Class {target_val})"):
                row_data = np.full(out_width, -9999, dtype=np.float32)

                for j in range(out_width):
                    # 定义读取窗口
                    win = Window(j * window_size_px, i * window_size_px, window_size_px, window_size_px)

                    # 读取数据 (这里读到的是原始分类数据，包含1, 2, 3...)
                    data = src.read(1, window=win)

                    # 如果读到的块尺寸不对（边缘），跳过
                    if data.shape != (window_size_px, window_size_px): continue

                    # 根据指标调用不同函数
                    if metric == 'SPLIT':
                        val = calculate_split_numpy(data, target_val)
                    elif metric == 'PLADJ':
                        val = calculate_pladj_numpy(data, target_val)
                    else:
                        val = -9999

                    if not np.isnan(val):
                        row_data[j] = val

                # 写入这一行 (修复了之前的 indexes 错误)
                dst.write(row_data.reshape(1, out_width), 1, window=Window(0, i, out_width, 1))


# --- 主程序入口 ---
if __name__ == "__main__":

    # ================= 配置区域 =================
    # 1. 输入文件路径 (请修改为你的完整土地利用数据路径)
    input_raster = r"D:\frag\003\HLJ2007CLCDPROJECT\HLJ2007CLCDPROJECT\hlj2007clcdPROJECT1.tif"

    # 2. 输出文件夹
    output_dir = r"D:\frag\003\results_1km"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # ================= 任务执行 =================

    # --- 任务 1: 耕地 (Class 1) ---
    print("\n--- 正在计算 耕地 (Class 1) ---")
    process_metric(input_raster, os.path.join(output_dir, "Cropland_SPLIT.tif"), target_val=1, metric='SPLIT')
    process_metric(input_raster, os.path.join(output_dir, "Cropland_PLADJ.tif"), target_val=1, metric='PLADJ')

    # --- 任务 2: 林地 (Class 2) ---
    print("\n--- 正在计算 林地 (Class 2) ---")
    process_metric(input_raster, os.path.join(output_dir, "Forest_SPLIT.tif"), target_val=2, metric='SPLIT')
    process_metric(input_raster, os.path.join(output_dir, "Forest_PLADJ.tif"), target_val=2, metric='PLADJ')

    print("\n✅ 所有任务已完成！")
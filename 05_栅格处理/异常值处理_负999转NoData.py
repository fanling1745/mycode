# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 批量将栅格中的 -999 替换为真正的 NoData 标记 (多进程 + 分块读写)
# 输入: 存放 TIFF 的文件夹
# 输出: 处理后的 TIFF (同名输出, LZW 压缩)
# 使用: 修改 INPUT_DIR 和 OUTPUT_DIR → 运行
# 依赖: rasterio, numpy
# =============================================================================
import os
from pathlib import Path
import numpy as np
import rasterio
from concurrent.futures import ProcessPoolExecutor

# ================= 配置参数 =================
INPUT_DIR = r"F:\frag\202604\0423\tif\合并"  # 替换为你的输入文件夹路径
OUTPUT_DIR = r"F:\frag\202604\0423\结果\contag"  # 输出文件夹 (建议设为新文件夹，脚本会自动保持同名)
SOURCE_VALUE = -999  # 需要被替换的异常值


# ============================================

def process_single_tif(file_path, out_path):
    """处理单个 TIF 文件的核心函数"""
    try:
        with rasterio.open(file_path) as src:
            meta = src.meta.copy()

            # 获取原数据的 NoData 设定值
            target_nodata = src.nodata

            # 如果原图根本没有注册 NoData 属性，我们将其注册为 -9999
            if target_nodata is None:
                target_nodata = SOURCE_VALUE
                meta.update(nodata=target_nodata)

            # 优化点 1：开启 LZW 无损压缩，加快磁盘写入速度，减小文件体积
            meta.update(compress='lzw')

            # 创建输出文件
            with rasterio.open(out_path, 'w', **meta) as dst:

                # 优化点 2：分块读取 (Block windows)，完美应对超大体积的高分辨率栅格
                for ji, window in src.block_windows(1):
                    # 读取该块内所有波段的数据
                    block_data = src.read(window=window)

                    # 优化点 3：NumPy 向量化条件替换
                    # 只有当目标 nodata 和当前值不一样时才需要物理替换像素值
                    if target_nodata != SOURCE_VALUE:
                        block_data = np.where(block_data == SOURCE_VALUE, target_nodata, block_data)

                    # 写入处理后的块
                    dst.write(block_data, window=window)

        return f"成功: {file_path.name}"
    except Exception as e:
        return f"失败: {file_path.name}, 错误信息: {e}"


def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    # 自动创建输出文件夹（如果不存在）
    output_path.mkdir(parents=True, exist_ok=True)

    # 抓取文件夹下所有的 .tif 和 .TIF 文件
    tif_files = list(input_path.glob("*.[tT][iI][fF]"))
    if not tif_files:
        print("⚠未在输入文件夹中找到 TIF 文件，请检查路径。")
        return

    print(f" 找到 {len(tif_files)} 个 TIF 文件，开始高速批量处理...")

    # 优化点 4：使用多进程池并行处理
    # max_workers 默认会自动使用机器所有的 CPU 核心
    with ProcessPoolExecutor() as executor:
        # 构建任务队列
        futures = [
            executor.submit(process_single_tif, tif, output_path / tif.name)
            for tif in tif_files
        ]

        # 实时打印进度
        for future in futures:
            print(future.result())

    print("所有栅格数据处理完毕！")


if __name__ == '__main__':
    # Windows 环境下多进程必须在 __main__ 保护块内运行
    main()
# -*- coding: utf-8 -*-
# =============================================================================
# 功能: MGWR 多带宽地理加权回归, 自动区分全局/局部变量
# 输入: 多年份 Shapefile (通过 my_tasks 配置) + max_sample_size 控制抽样
# 输出: 多 Sheet Excel (地理变异/全局变量/局部变量统计 + R²/AICc)
# 使用: 修改 my_tasks 列表和 output_file → 运行
# 依赖: mgwr, geopandas, numpy, pandas
# =============================================================================
import geopandas as gpd
import pandas as pd
import numpy as np
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR
import warnings
import time
import gc

warnings.filterwarnings("ignore")


def run_mixed_gwr(tasks, output_excel_path, max_sample_size=10000):
    print(f"准备开始运行 MGWR, 结果将保存至: {output_excel_path}\n" + "=" * 50)

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for task in tasks:
            start_time = time.time()
            shapefile_path = task["path"]
            y_field = task["y_field"]
            year_label = task["year"]

            print(f"\n正在处理: {shapefile_path} (因变量: {y_field})")

            try:
                gdf = gpd.read_file(shapefile_path)
                gdf = gdf.dropna()
                original_N = len(gdf)

                if original_N > max_sample_size:
                    print(f"[{year_label}] 原始样本 {original_N}, 触发抽样 -> {max_sample_size}")
                    gdf = gdf.sample(n=max_sample_size, random_state=42)
                else:
                    print(f"[{year_label}] 样本量 {original_N}, 使用全量数据")

                N = len(gdf)
                coords = np.column_stack((gdf.centroid.x, gdf.centroid.y))

                y = gdf[y_field].values.reshape((-1, 1)).astype(np.float32)
                exclude_cols = ['geometry', 'FID', 'shape', 'Shape', 'fid', y_field]
                x_cols = [col for col in gdf.columns if col not in exclude_cols]
                X = gdf[x_cols].values.astype(np.float32)

                X_std = (X - X.mean(axis=0)) / X.std(axis=0)
                y_std = (y - y.mean(axis=0)) / y.std(axis=0)

                var_names = ['Intercept (常数项)'] + x_cols

                print(f"[{year_label}] 正在计算最优带宽...")
                selector = Sel_BW(coords, y_std, X_std, multi=True)
                bws = selector.search(multi_bw_min=[100], tol_multi=1e-4, max_iter_multi=30)

                print(f"[{year_label}] 正在拟合模型...")
                model = MGWR(coords, y_std, X_std, selector)
                results = model.fit()

                bws_array = results.model.bws

                df_geo_variation = pd.DataFrame({
                    '变量名称': var_names,
                    '最优带宽': bws_array,
                    '变量类型': ['全局变量' if bw >= N - 2 else '局部变量' for bw in bws_array]
                })
                df_geo_variation.to_excel(writer, sheet_name=f'{year_label}_地理变异', index=False)

                global_indices = [i for i, bw in enumerate(bws_array) if bw >= N - 2]
                local_indices = [i for i, bw in enumerate(bws_array) if bw < N - 2]

                if global_indices:
                    global_stats = [
                        {'解释变量': var_names[idx],
                         '回归系数 (Estimate)': round(results.params[:, idx].mean(), 6),
                         'T值 (t-value)': round(results.tvalues[:, idx].mean(), 6)}
                        for idx in global_indices
                    ]
                    pd.DataFrame(global_stats).to_excel(writer, sheet_name=f'{year_label}_全局变量', index=False)
                else:
                    pd.DataFrame([{'提示': '未检测到全局变量'}]).to_excel(
                        writer, sheet_name=f'{year_label}_全局变量', index=False)

                if local_indices:
                    local_stats = []
                    for idx in local_indices:
                        coefs = results.params[:, idx]
                        local_stats.append({
                            '解释变量': var_names[idx],
                            '平均值 (Mean)': round(np.mean(coefs), 6),
                            '标准差 (STD)': round(np.std(coefs), 6),
                            '最小值 (Min)': round(np.min(coefs), 6),
                            '最大值 (Max)': round(np.max(coefs), 6),
                            '下四分位 (Lwr)': round(np.percentile(coefs, 25), 6),
                            '中位数 (Median)': round(np.median(coefs), 6),
                            '上四分位 (Upr)': round(np.percentile(coefs, 75), 6)
                        })
                    df_local = pd.DataFrame(local_stats)
                    df_metrics = pd.DataFrame({
                        '解释变量': ['模型指标 ->', 'R2', '调整 R2', 'AICc'],
                        '平均值 (Mean)': ['', round(results.R2, 4), round(results.adj_R2, 4), round(results.aicc, 4)]
                    })
                    pd.concat([df_local, df_metrics], ignore_index=True).to_excel(
                        writer, sheet_name=f'{year_label}_局部变量', index=False)

                elapsed = (time.time() - start_time) / 60
                print(f"[{year_label}] 完成! 耗时: {elapsed:.2f} 分钟")

                del gdf, X, y, model, results
                gc.collect()

            except Exception as e:
                print(f"处理 {year_label} 时出错: {e}")

    print(f"所有任务完成! 结果保存为: {output_excel_path}")


# ================= 运行配置 =================
if __name__ == "__main__":
    my_tasks = [
        {"year": "2000", "path": r"路径/因子筛选2000.shp", "y_field": "a2000_FFI"},
        {"year": "2010", "path": r"路径/因子筛选2010.shp", "y_field": "a2010_FFI"},
        {"year": "2020", "path": r"路径/因子筛选2020.shp", "y_field": "a2020_FFI"}
    ]

    output_file = "GWR_MGWR_结果.xlsx"
    run_mixed_gwr(my_tasks, output_file, max_sample_size=10000)

# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 空间栅格偏相关分析 (FVC 与各因子的净相关, 控制其他因子)
# 输入: 多年份目标变量和因子栅格
# 输出: Spatial_Pcor.csv (Year/Factor/Partial_Correlation/P_Value/Significance)
# 使用: 修改 data_dir / out_file / years → 运行
# 依赖: rasterio, pingouin, scipy, pandas, numpy
# =============================================================================
import os
import numpy as np
import pandas as pd
import rasterio
import pingouin as pg
from scipy import stats

# ================= 1. 配置参数 =================
data_dir = r"E:\数据分析\地理探测器\20260318\沉\偏相关分析\数据"  # 替换为你的数据所在文件夹
out_file = r"E:\数据分析\地理探测器\20260318\沉\偏相关分析\结果\Spatial_Pcor.csv"  # 结果保存路径

years = [2004, 2014, 2024]
target_var = 'FVC'

# 所有需要分析的因子
dynamic_vars = ['pet', 'pop', 'NTL', 'pre', 'tmp']
static_vars = ['高程', '坡度']
all_factors = dynamic_vars + static_vars

# 初始化一个空列表，用于存放所有年份的计算结果
all_results = []

# ================= 2. 循环处理各个年份 =================
for year in years:
    print(f"\n[{year}年] 正在读取栅格数据并构建空间样本池...")
    data_dict = {}
    nodata_val = None

    # 2.1 读取目标变量 (FVC)
    fvc_path = os.path.join(data_dir, f"{target_var}_{year}.tif")
    try:
        with rasterio.open(fvc_path) as src:
            data_dict[target_var] = src.read(1).flatten()
            nodata_val = src.nodata
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {fvc_path}，跳过该年份。")
        continue

    # 2.2 读取所有因子变量
    for factor in all_factors:
        # 静态变量（高程、坡度）不带年份后缀
        if factor in static_vars:
            filepath = os.path.join(data_dir, f"{factor}.tif")
        else:
            filepath = os.path.join(data_dir, f"{factor}_{year}.tif")

        try:
            with rasterio.open(filepath) as src:
                data_dict[factor] = src.read(1).flatten()
        except FileNotFoundError:
            print(f"❌ 错误：找不到因子文件 {filepath}，请检查数据完整性。")

    # 2.3 转换为 Pandas DataFrame
    df = pd.DataFrame(data_dict)

    # ================= 3. 数据清洗与采样 =================
    print(f"[{year}年] 正在清洗无效像元及空值数据...")

    # 将栅格的 NoData 值替换为 NaN
    if nodata_val is not None:
        df = df.replace(nodata_val, np.nan)

    # 严格剔除含有任何空值的行（无论是目标变量还是因子变量为空，整行剔除）
    df = df.dropna(how='any')

    n_samples = len(df)
    print(f"[{year}年] 清洗后共有 {n_samples} 个有效空间像元。")

    # 样本量控制：超过 200,000 则进行随机抽样
    if n_samples > 200000:
        print(f"[{year}年] 样本量大于 200,000，正在随机抽取 200,000 个样本进行计算...")
        # random_state=42 保证每次运行抽样的结果是固定的（可复现）
        df = df.sample(n=200000, random_state=42)

        # ================= 4. 偏相关分析 =================
    print(f"[{year}年] 开始计算偏相关系数...")

    for main_factor in all_factors:
        # 找出除了当前主因子以外的所有其他因子，作为控制变量(covariates)
        covariates = [f for f in all_factors if f != main_factor]

        # 1. 使用 pingouin 计算偏相关，只提取相关系数 r (这个列名永远是 'r')
        pcor_stats = pg.partial_corr(data=df, x=target_var, y=main_factor, covar=covariates)
        r_value = pcor_stats['r'].values[0]

        # 2. 手动计算 P 值 (彻底无视 pingouin 库的 P 值列名以防报错)
        n_obs = len(df)  # 当前用于计算的样本量
        k_covars = len(covariates)  # 控制变量的数量
        df_err = n_obs - 2 - k_covars  # 计算自由度

        # 计算 t 统计量 (分母加 1e-8 是为了防止极端情况下 r=1 导致除以 0 报错)
        t_stat = r_value * np.sqrt(df_err / (1.0 - r_value ** 2 + 1e-8))

        # 计算双侧 P 值
        p_value = 2 * stats.t.sf(np.abs(t_stat), df_err)

        # 3. 划分显著性等级标识
        if p_value < 0.001:
            sig_level = "***"
        elif p_value < 0.01:
            sig_level = "**"
        elif p_value < 0.05:
            sig_level = "*"
        else:
            sig_level = "ns"  # not significant

        # 将单次计算结果追加到总列表中
        all_results.append({
            'Year': year,
            'Target_Variable': target_var,
            'Factor': main_factor,
            'Partial_Correlation (r)': round(r_value, 4),
            'P_Value': p_value,
            'Significance': sig_level
        })

    print(f"[{year}年] 分析完成。")

# ================= 5. 输出最终表格 =================
print("\n================ 所有年份处理完毕，正在导出表格 =================")
# 将所有结果转换为 DataFrame
results_df = pd.DataFrame(all_results)

# 打印预览
print(results_df.to_string(index=False))

# 自动创建输出文件夹（如果不存在）
out_dir_path = os.path.dirname(out_file)
if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)

# 导出为 CSV 文件 (使用 utf-8-sig 编码防止在 Excel 中打开时中文乱码)
results_df.to_csv(out_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 结果已成功保存至: {out_file}")
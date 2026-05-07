# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 自动识别因子基础名称, 按年份分别计算 Pearson 相关性
# 输入: Shapefile (含多年份变量, 如 tmp_2000, tmp_2010, tmp_2020)
# 输出: Correlation_Results.xlsx (多层表头: 年份×[Pearson,显著性])
# 使用: 修改 input_file 和 output_file → 运行
# 依赖: geopandas, scipy, pandas, re
# =============================================================================
import geopandas as gpd
import pandas as pd
from scipy.stats import pearsonr
import re

# ================= 1. 自定义设置区 =================
# 在这里填入你的输入文件路径
input_file = r"E:\frag\20260317\001\地理加权回归\渔网赋值\YW_赋值.shp"

# 在这里自定义你的输出表格路径（务必以 .xlsx 结尾）
output_file = r"E:\frag\20260317\001\地理加权回归\结果\Correlation_Results.xlsx"
# ===================================================

print("正在读取数据...")
gdf = gpd.read_file(input_file)
df = pd.DataFrame(gdf.drop(columns='geometry'))

# 2. 修正字段名
# 满足你的需求，将 铁路_202 改为 铁路_2020。顺便把 200/201 这种明显的后缀截断也一并修复。
rename_dict = {
    '铁路_202': '铁路_2020', '铁路_201': '铁路_2010', '铁路_200': '铁路_2000',
    '公路_202': '公路_2020', '公路_201': '公路_2010', '公路_200': '公路_2000'
}
df.rename(columns=rename_dict, inplace=True)

# 3. 提取所有自变量的基础名称（剔除年份后缀，如 tmp_2020 变为 tmp）
years = ['2000', '2010', '2020']
dependent_vars = [f'a{year}_FFI' for year in years]

# 获取所有的数值列
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in dependent_vars]

base_factor_names = []
for col in numeric_cols:
    # 使用正则去掉 _2000, _2010, _2020 后缀来获取因子基础名称
    base_name = re.sub(r'_(2000|2010|2020)$', '', col)
    if base_name not in base_factor_names:
        base_factor_names.append(base_name)

# 4. 构建与图片一致的多层表头表格
# 顶层表头：2000, 2010, 2020；底层表头：Pearson相关性, 显著性
columns_multi = pd.MultiIndex.from_product([years, ['Pearson相关性', '显著性']])
result_df = pd.DataFrame(index=base_factor_names, columns=columns_multi)
result_df.index.name = '因子名称'

print("正在计算相关性...")
# 5. 分别计算不同年份的相关性
for factor in base_factor_names:
    for year in years:
        dep_col = f'a{year}_FFI'  # 当前年份的因变量

        # 寻找对应的自变量列：优先找带年份的（如 tmp_2020），找不到则用静态的（如 DEM）
        factor_with_year = f"{factor}_{year}"
        if factor_with_year in df.columns:
            indep_col = factor_with_year
        elif factor in df.columns:
            indep_col = factor
        else:
            continue  # 如果都没有，跳过

        # 提取数据并清洗空值
        valid_data = df[[indep_col, dep_col]].dropna()

        if len(valid_data) > 1:
            r_val, p_val = pearsonr(valid_data[indep_col], valid_data[dep_col])

            # 显著性星号判断
            stars = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')

            # 填入数据：Pearson保留三位小数+星号，显著性保留三位小数
            result_df.loc[factor, (year, 'Pearson相关性')] = f"{r_val:.3f}{stars}"
            result_df.loc[factor, (year, '显著性')] = f"{p_val:.3f}"

# 6. 保存为 Excel 表格
result_df.to_excel(output_file)
print(f"计算完成！精美表格已成功保存至: {output_file}")
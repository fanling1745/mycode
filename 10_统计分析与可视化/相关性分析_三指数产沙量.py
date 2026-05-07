# -*- coding: utf-8 -*-
# =============================================================================
# 功能: CSI/CCI/SPI 与水蚀/风蚀产沙量的 Pearson 相关性分析
# 输入: 含多个流域 Sheet 的 Excel 文件
# 输出: 流域三指数与产沙量相关性分析结果.xlsx
# 使用: 修改 excel_file → 运行
# 依赖: pandas, scipy, numpy
# =============================================================================
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# 1. 定义包含所有表单的Excel文件路径（请确保文件名和路径与你的实际情况一致）
excel_file = r"F:\数据分析\数据处理\20260427\三指数和产沙量关系.xlsx"


# 2. 定义相关系数和显著性星号的格式化函数
def format_corr(r, p):
    if np.isnan(r):
        return "-"
    elif p < 0.01:
        return f"{r:.3f}**"
    elif p < 0.05:
        return f"{r:.3f}*"
    else:
        return f"{r:.3f}"


# 3. 初始化结果列表和参数
results = []
years = ['2023', '2025']
factors = ['CSI', 'CCI', 'SPI']

# 你的Excel文件中对应的三个表单名称
sheet_names = ['海勒斯太', '六道沟', '杨家沟']

# 4. 遍历处理每个表单（流域）的数据
for basin in sheet_names:
    try:
        # 读取指定Excel文件中的特定表单(sheet)
        df = pd.read_excel(excel_file, sheet_name=basin)

        for year in years:
            for factor in factors:
                # 拼接列名，例如 "2023年CSI"
                factor_col = f"{year}年{factor}"
                water_col = f"{year}年水蚀产沙量"
                wind_col = f"{year}年风蚀产沙量"

                # --- 计算水蚀相关性 ---
                if factor_col in df.columns and water_col in df.columns:
                    # 过滤掉缺失值
                    mask_water = ~df[factor_col].isna() & ~df[water_col].isna()
                    if mask_water.sum() > 2:  # 至少需要3个样本点才能计算相关性
                        r_water, p_water = pearsonr(df.loc[mask_water, factor_col], df.loc[mask_water, water_col])
                        water_str = format_corr(r_water, p_water)
                    else:
                        water_str = "-"
                else:
                    water_str = "-"

                # --- 计算风蚀相关性（杨家沟所在表单没有此列，会自动跳过并填充"-"） ---
                if wind_col in df.columns:
                    mask_wind = ~df[factor_col].isna() & ~df[wind_col].isna()
                    if mask_wind.sum() > 2:
                        r_wind, p_wind = pearsonr(df.loc[mask_wind, factor_col], df.loc[mask_wind, wind_col])
                        wind_str = format_corr(r_wind, p_wind)
                    else:
                        wind_str = "-"
                else:
                    wind_str = "-"

                # 保存当前因子的结果
                results.append({
                    '流域': basin,
                    '年份': year,
                    '因子': factor,
                    '水蚀产沙量': water_str,
                    '风蚀产沙量': wind_str
                })

    except Exception as e:
        print(f"处理表单【{basin}】时出错，请检查表单名是否完全匹配或数据是否异常。错误信息: {e}")

# 5. 生成最终的数据框格式
result_df = pd.DataFrame(results)

# 打印预览结果
print("================== 结果预览 ==================")
print(result_df.to_string(index=False))
print("==============================================")

# 6. 导出到新的Excel表格
output_filename = "流域三指数与产沙量相关性分析结果.xlsx"
result_df.to_excel(output_filename, index=False)
print(f"\n计算完成！结果已成功保存为同一个文件夹下的：{output_filename}")
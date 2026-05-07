# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 按年份全变量 OLS 回归 + VIF 诊断
# 输入: Shapefile + 按年份定义的变量列表
# 输出: 多 Sheet Excel (系数/T值/P值/VIF + 模型 R²/F值)
# 使用: 修改 input_file 和 models_config → 运行
# 依赖: geopandas, statsmodels, pandas, numpy
# =============================================================================
import geopandas as gpd
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 自定义设置区 =================
input_file = r"E:\frag\20260317\001\地理加权回归\渔网赋值\YW_赋值.shp"
output_file = r"E:\frag\20260317\001\地理加权回归\结果\全变量回归汇总结果.xlsx"

print("正在读取并清洗数据...")
gdf = gpd.read_file(input_file)
df = pd.DataFrame(gdf.drop(columns='geometry'))

# 修正被 shp 截断的字段名
rename_dict = {
    '铁路_202': '铁路_2020', '铁路_201': '铁路_2010', '铁路_200': '铁路_2000',
    '公路_202': '公路_2020', '公路_201': '公路_2010', '公路_200': '公路_2000'
}
df.rename(columns=rename_dict, inplace=True)

# ================= 2. 显式定义各年份全变量列表 =================
# 这里定义的所有因子，都将100%进入最终的回归表格中
models_config = {
    '2000': {
        'y': 'a2000_FFI',
        'x': ['铁路_2000', '公路_2000', 'tmp_2000', 'Slope', 'pre_2000',
              'pop_2000', 'pet_2000', 'NTL_2000', 'NDVI_2000', 'gdp_2000',
              'DEM', 'CLCD_2000']
    },
    '2010': {
        'y': 'a2010_FFI',
        'x': ['铁路_2010', '公路_2010', 'tmp_2010', 'Slope', 'pre_2010',
              'pop_2010', 'pet_2010', 'NTL_2010', 'NDVI_2010', 'gdp_2010',
              'DEM', 'CLCD_2010']
    },
    '2020': {
        'y': 'a2020_FFI',
        'x': ['铁路_2020', '公路_2020', 'tmp_2020', 'Slope', 'pre_2020',
              'pop_2020', 'pet_2020', 'NTL_2020', 'NDVI_2020', 'gdp_2020',
              'DEM', 'CLCD_2020']
    }
}

# ================= 3. 逐年执行全变量回归并写入同一个 Excel =================
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for year, config in models_config.items():
        print(f"\n--- 正在处理 {year} 年的数据 ---")
        dep_col = config['y']

        # 只提取在数据集中实际存在的 X 变量
        current_year_factors = [col for col in config['x'] if col in df.columns]

        if dep_col not in df.columns:
            print(f"警告：未找到 {year} 年的因变量 {dep_col}，跳过该年份。")
            continue

        analysis_cols = current_year_factors + [dep_col]

        # 步骤 A：清理空值与控制样本量
        valid_data = df[analysis_cols].copy()
        valid_data.replace("<空>", np.nan, inplace=True)
        valid_data.dropna(inplace=True)

        if len(valid_data) > 200000:
            valid_data = valid_data.sample(n=200000, random_state=42)
            print(f"数据量过大，已随机抽取 200,000 个样本进行计算。")

        valid_data = valid_data.astype(float)
        y = valid_data[dep_col]
        X = valid_data[current_year_factors]

        # 步骤 B：直接执行全变量 OLS 回归 (不再进行逐步筛选)
        print(f"强行纳入以下所有变量进行回归: {current_year_factors}")

        # 必须保留常数项，否则回归模型的 R² 和系数会严重失真
        X_final = sm.add_constant(X)
        final_model = sm.OLS(y, X_final).fit()

        # 提取结果表
        results_df = pd.DataFrame({
            '系数 (Coef)': final_model.params,
            'T值 (T-stat)': final_model.tvalues,
            'P值 (P-value)': final_model.pvalues
        })

        # 步骤 C：多重共线性检测 (VIF)
        vif_data = pd.DataFrame()
        vif_data["变量"] = X_final.columns
        vif_data["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]

        # 匹配 VIF 值，剔除常数项的 VIF 显示
        vif_dict = dict(zip(vif_data["变量"], vif_data["VIF"]))
        results_df['VIF'] = results_df.index.map(lambda x: vif_dict.get(x, None) if x != 'const' else np.nan)

        # 步骤 D：汇总模型整体指标并排版
        summary_info = pd.DataFrame({
            '系数 (Coef)': [None, final_model.rsquared, final_model.fvalue],
            'T值 (T-stat)': [None, 'Adj R-squared:', final_model.rsquared_adj],
            'P值 (P-value)': [None, 'F-prob:', final_model.f_pvalue],
            'VIF': [None, None, None]
        }, index=['---', '模型 R²', '模型 F值'])

        final_output = pd.concat([results_df, summary_info])

        # 步骤 E：保存至单独的 Sheet
        sheet_name = f"{year}年全变量结果"
        final_output.to_excel(writer, sheet_name=sheet_name)
        print(f"========== 【{year} 年全变量回归结果已写入表单：{sheet_name}】 ==========")

print(f"\n✅ 所有年份处理完毕！汇总文件已保存至:\n{output_file}")
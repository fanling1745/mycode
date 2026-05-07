# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 从宽表 Excel 按年份提取 y 和 x 因子列, 输出地理探测器所需 CSV
# 输入: Excel 文件 (列名如 esv_{year}, pop_{year}, DEM, SLOPE)
# 输出: data_{year}.csv (列名 y / x1-x8)
# 使用: 修改 file_path → 运行
# 依赖: pandas
# =============================================================================
import pandas as pd

def clean_and_split_data(input_file):
    # 1. 读取Excel文件
    print(f"正在读取文件: {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"读取文件失败，请检查文件路径。错误信息: {e}")
        return

    # 2. 定义年份和固定变量
    years = [ '2010', '2015', '2020', '2023']
    constant_vars = ['DEM', 'SLOPE']

    # 3. 循环处理每个年份
    for year in years:
        # 定义当前年份需要提取的列名及其对应的重命名规则
        # 严格按照你要求的顺序: DEM, Slope, tem, pre, pet, EMP, NTL, pop, CLCD -> x1~x9
        columns_mapping = {
            f'esv_{year}': 'y',
            f'pop_{year}': 'x1',
            f'GDP_{year}': 'x2',
            f'公路_{year}': 'x3',
            f'NTL_{year}': 'x4',
            f'tmp_{year}': 'x5',
            f'pre_{year}': 'x6',
            'DEM': 'x7',
            'SLOPE': 'x8'
        }

        # 获取当前年份需要的所有原始列名
        cols_to_extract = list(columns_mapping.keys())

        # 安全检查：检查所需的列是否都在原始数据中存在
        missing_cols = [col for col in cols_to_extract if col not in df.columns]
        if missing_cols:
            print(f"⚠ 警告：年份 {year} 缺少以下列 {missing_cols}，将跳过该年份的提取。")
            continue

        # 4. 提取数据并生成新的 DataFrame
        df_year = df[cols_to_extract].copy()

        # 5. 重命名列
        df_year.rename(columns=columns_mapping, inplace=True)

        # 6. 数据清洗 (可选：剔除含有缺失值的行)
        # 如果你需要删除有缺失值的行，取消下面这行代码的注释
        # df_year.dropna(inplace=True)

        # 7. 导出为 CSV 文件
        output_filename = f'data_{year}.csv'
        # 使用 utf-8-sig 编码防止在 Excel 中打开 CSV 时出现中文乱码（如果有中文字符的话）
        df_year.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f" 成功生成文件: {output_filename} (包含 {len(df_year)} 条数据)")

    print(" 所有年份数据拆分与清洗完毕！")


# ==========================================
# 运行代码
# ==========================================
# 请将 'your_data.xlsx' 替换为你实际的 Excel 文件路径
file_path = r"F:\数据分析\地理探测器\202605\0504\ESV_ta\分区1km\新建 XLSX 工作表.xlsx"
clean_and_split_data(file_path)
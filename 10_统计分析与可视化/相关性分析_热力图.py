# -*- coding: utf-8 -*-
# =============================================================================
# 功能: Pearson 相关系数矩阵 + 学术热力图 (含显著性 *p<0.05, **p<0.01)
# 输入: Excel 或 Shapefile + 分析指标列表
# 输出: 热力图 PNG + 相关系数矩阵 Excel
# 使用: 修改 data_path 和 indicators → 运行
# 依赖: pandas, scipy, seaborn, matplotlib, numpy
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 2. 加载您的实际数据
data_path = r'F:\项目\城乡聚落三维形态\数据\驱动力分析\筛选特征\sample.xlsx'
data = pd.read_excel(data_path, sheet_name="Sheet1")

# 3. 选择需要分析的指标（请确保列名完全匹配）
indicators = ['碳储量', 'SPLIT', 'HCV', 'SDBV', 'HBH', 'HBV',
             'FRAC_MN', 'MBV', 'MBH', 'BCD', 'BD', 'VCV', 'IJI']
df = data[indicators].copy()

# 4. 数据预处理：检查并处理缺失值
print("数据缺失值统计：")
print(df.isnull().sum())

# 用中位数填充缺失值（可根据数据特点选择其他填充方式）
df = df.fillna(df.median())

# 5. 计算相关系数和p值（稳健版本）
def safe_pearsonr(x, y):
    """处理可能出现的统计计算异常"""
    try:
        return pearsonr(x, y)
    except:
        return (np.nan, 1.0)  # 返回不显著的结果

def calculate_pvalues(df):
    cols = df.columns
    n = len(cols)
    p_matrix = np.ones((n, n))  # 默认p=1（不显著）
    for i in range(n):
        for j in range(n):
            if i == j:
                p_matrix[i, j] = 0  # 对角线设为0
            else:
                # 跳过常数列或包含NaN的列
                if df.iloc[:, i].nunique() > 1 and df.iloc[:, j].nunique() > 1:
                    _, p_matrix[i, j] = safe_pearsonr(df.iloc[:, i].dropna(),
                                                    df.iloc[:, j].dropna())
    return pd.DataFrame(p_matrix, columns=cols, index=cols)

corr_matrix = df.corr()
p_values = calculate_pvalues(df)

# 6. 创建显著性标记函数
def format_annotation(val, pval):
    if pd.isna(val) or pd.isna(pval):
        return ""
    if pval < 0.01:
        return f"{val:.2f}**"
    elif pval < 0.05:
        return f"{val:.2f}*"
    else:
        return f"{val:.2f}"

# 7. 构建标注矩阵
annot_matrix = np.empty_like(corr_matrix, dtype=object)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        annot_matrix[i, j] = format_annotation(corr_matrix.iloc[i, j],
                                             p_values.iloc[i, j])

# 8. 绘制热力图（专业学术风格）
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角

heatmap = sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=annot_matrix,
    fmt="",
    cmap='RdBu_r',  # 红蓝渐变色，更适合相关性展示
    center=0,
    vmin=-1,
    vmax=1,
    annot_kws={"size": 9, "color": "black"},
    cbar_kws={"shrink": 0.8, "label": "相关系数"},
    linewidths=0.5,
    linecolor='white'
)

# 9. 图表美化
plt.title("景观指标与碳储量相关性分析\n(*p<0.05, **p<0.01)",
         fontsize=16, pad=20, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
heatmap.figure.axes[-1].yaxis.label.set_size(12)  # 调整colorbar标签大小

# 10. 保存高质量图片
output_path = r'F:\项目\城乡聚落三维形态\python\驱动力分析\碳储量相关性热图.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"热力图已保存至：{output_path}")
plt.close()

# 11. 输出相关系数矩阵（可选）
corr_matrix.to_excel(output_path.replace('.png', '_相关系数矩阵.xlsx'))
print("分析完成！")
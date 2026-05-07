# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 随机森林 10 折交叉验证
# 输入: Excel (第二列为 y, 第三列起为 X)
# 输出: 控制台输出每折 R² 和平均 R²
# 使用: 修改 file_path → 运行
# 依赖: sklearn, pandas, numpy
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# 设置Excel文件路径
file_path = r"F:\项目\城乡聚落三维形态\数据\驱动力分析\sample.xlsx"  # ← 请替换为你实际的路径

# 读取数据
df = pd.read_excel(file_path, sheet_name="Sheet2", header=None)

# 打印原始数据量
original_rows = df.shape[0]
print(f"原始数据量：{original_rows} 条")

# 去除包含空值（NaN）、字符串“<空>”或0的行（第二列为因变量，之后为自变量）
df.replace('<空>', np.nan, inplace=True)  # 把 '<空>' 替换为 NaN
df = df.dropna()  # 删除包含 NaN 的行

# 将所有列转换为数值型
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()  # 再次删除转换失败变成 NaN 的行

# 删除因变量或任一自变量为0的行
df = df[(df.iloc[:, 1] != 0) & (df.iloc[:, 2:] != 0).all(axis=1)]

# 打印清洗后的数据量
cleaned_rows = df.shape[0]
print(f"清洗后的数据量：{cleaned_rows} 条")

# 准备特征和目标变量
y = df.iloc[:, 1].values  # 第二列为因变量
X = df.iloc[:, 2:].values  # 从第三列开始为自变量

# 构建随机森林模型
model = RandomForestRegressor(random_state=42)

# 十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# 打印每折得分和平均得分
print("每折 R² 分数：", scores)
print("平均 R² 分数：", scores.mean())

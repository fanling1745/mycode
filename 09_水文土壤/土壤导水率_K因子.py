# -*- coding: utf-8 -*-
# =============================================================================
# 功能: Saxton (2006) 土壤饱和导水率 Ks 计算
# 输入: Excel (含 SAND/CLAY/OM 列)
# 输出: soil_data_with_K_cmd.xlsx (新增 K_cm_d 列, 单位 cm/d)
# 使用: 修改 file_path → 运行
# 依赖: pandas, numpy
# =============================================================================
import pandas as pd
import numpy as np

# 1. 读取你的 Excel 数据表 (请修改为实际文件路径)
file_path = r"E:\数据分析\水源涵养\20260209_息澜\HWSD2_Weighted_Average_Final.xlsx"
df = pd.read_excel(file_path)

# 2. 过滤无效值：只对 SAND, CLAY, OM 不为 -999 的行进行计算
valid_mask = (df['SAND'] != -999) & (df['CLAY'] != -999) & (df['OM'] != -999)

# 3. 提取变量并进行严格的单位换算
S = df.loc[valid_mask, 'SAND'] / 100.0
C = df.loc[valid_mask, 'CLAY'] / 100.0
OM = df.loc[valid_mask, 'OM']

# --- 以下为 Saxton (2006) 核心计算过程 ---

# 第一步：计算 1500 kPa 处的土壤水分（萎蔫系数 θ_1500）
t1500_t = -0.024*S + 0.487*C + 0.006*OM + 0.005*(S*OM) - 0.013*(C*OM) + 0.068*(S*C) + 0.031
t1500 = t1500_t + (0.14 * t1500_t - 0.02)

# 第二步：计算 33 kPa 处的土壤水分（田间持水量 θ_33）
t33_t = -0.251*S + 0.195*C + 0.011*OM + 0.006*(S*OM) - 0.027*(C*OM) + 0.452*(S*C) + 0.299
t33 = t33_t + (1.283 * t33_t**2 - 0.374 * t33_t - 0.015)

# 第三步：计算孔隙度中间变量 θ_(S-33)
ts33_t = 0.278*S + 0.034*C + 0.022*OM - 0.018*(S*OM) - 0.027*(C*OM) - 0.584*(S*C) + 0.078
ts33 = ts33_t + (0.636 * ts33_t - 0.107)

# 第四步：计算土壤饱和含水量/孔隙度（θ_S）
ts = t33 + ts33 - 0.097*S + 0.043

# 第五步：计算土壤饱和导水率（Ks），标准单位为 mm/h
B = (np.log(1500) - np.log(33)) / (np.log(t33) - np.log(t1500))
lam = 1.0 / B
Ks_mm_h = 1930 * (ts - t33)**(3 - lam)

# --- 结束核心计算 ---

# 4. 转换为最终单位：cm/d (厘米/天)
# 换算逻辑：mm/h * 24小时 / 10 = cm/d，也就是直接乘以 2.4
Ks_cm_d = Ks_mm_h * 2.4

# 5. 将结果写回数据表
# 先将整列默认赋值为 -999，然后用正常计算的 cm/d 结果覆盖有效行
df['K_cm_d'] = -999.0
df.loc[valid_mask, 'K_cm_d'] = Ks_cm_d

# 6. 保存为新的 Excel 文件
output_path = "soil_data_with_K_cmd.xlsx"
df.to_excel(output_path, index=False)
print(f"计算完成！结果已保存至 {output_path}，单位为 cm/d，原始无效值的计算结果均已设为 -999。")
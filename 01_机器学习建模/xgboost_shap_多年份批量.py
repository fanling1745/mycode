# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 加载多年份预训练 XGBoost 模型, 批量生成 SHAP 解释图 + PDP 部分依赖图
# 输入: 预训练模型 pkl 文件 + Excel 多 Sheet 数据
# 输出: SHAP summary plot / PDP 3×4 子图 (pdp_plot_{year}.png)
# 使用: 修改 excel_file_path 和 model_save_folder → 运行
# 依赖: shap, xgboost, sklearn, pandas, matplotlib
# =============================================================================
import os
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# ---------------------------
# 全局设置（仅用于默认设置，后续可按需在局部覆盖）
plt.rcParams['font.sans-serif'] = ['SimSun']  # 默认中文字体：宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ---------------------------

# ---------------------------
# 配置文件路径及模型年份
excel_file_path = r"F:\研究生\毕业论文\区域热岛\驱动力分析\值提取至点\sample.xlsx"
model_save_folder = r"F:\研究生\毕业论文\区域热岛\驱动力分析\训练模型\模型存放路径"
sheet_names = ["2001", "2005","2010","2015","2020"]  # 可根据需要添加更多年份
# ---------------------------

# 用于存储每个年份的模型和特征数据
models = {}
X_data = {}

# ---------------------------
# 加载模型与数据
for sheet in sheet_names:
    # 加载模型（假设模型文件命名为 xgb_model_{sheet}.pkl）
    model_file = os.path.join(model_save_folder, f"xgb_model_{sheet}.pkl")
    with open(model_file, "rb") as f:
        models[sheet] = pickle.load(f)

    # 读取 Excel 中对应 Sheet 数据，并剔除缺失值
    df = pd.read_excel(excel_file_path, sheet_name=sheet)
    df_clean = df.dropna()
    # 根据约定：第三列及以后的数据为自变量
    X_data[sheet] = df_clean.iloc[:, 2:]

# ---------------------------
# 针对每个模型生成两幅图：
# 1. SHAP解释图
# 2. 部分依赖图（PDP），每个子图的 y 轴单独适应数据，并将 y 轴标签修改为中文
# ---------------------------
for sheet in sheet_names:
    model = models[sheet]
    X = X_data[sheet]

    # ===========================
    # 1. SHAP 解释图（单独出图）
    # ===========================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    # 绘制 SHAP summary plot（dot 类型），show=False 避免自动调用 plt.show()
    shap.summary_plot(shap_values, X, plot_type="dot", show=False)
    # 设置 SHAP 图的标题和 x 轴标签的字体样式
    plt.title(f"{sheet} 年的SHAP 解释图", fontsize=24, fontweight='bold')
    plt.xlabel("SHAP值", fontsize=24)
    plt.ylabel("特征名称", fontsize=24)  # 去掉 y 轴标注

    # 设置当前坐标轴的刻度标注字体大小
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=22)
    plt.setp(ax.get_yticklabels(), fontsize=22)

    # 修改右侧色带（colorbar）的属性
    fig = plt.gcf()
    if len(fig.axes) > 1:
        # 色带一般在最后一个 Axes 中
        cbar_ax = fig.axes[-1]
        cbar_ax.set_title("")            # 去掉色带标题
        cbar_ax.tick_params(labelsize=22)  # 修改色带刻度字体大小

    plt.tight_layout()
    # 保存 SHAP 值解释图
    #plt.savefig(f'F:\研究生\毕业论文\区域热岛\IMG\驱动力分析\SHAP解释图_{sheet}.png',dpi=300)  # 按年份保存图像
    #plt.show()

    # ===========================
    # 2. 部分依赖图（PDP，单独出图）
    # ===========================
    # 选择前 12 个特征进行展示（可根据实际情况调整）
    features_to_plot = X.columns.tolist()[:12]

    # 创建 3 行 4 列的子图布局，禁用 y 轴共享
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharey=False)
    axes = axes.flatten()  # 将子图展平为一维列表

    # 利用 PartialDependenceDisplay.from_estimator 绘制 PDP
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features_to_plot,
        ax=axes,
        kind="average"  # 计算平均 PDP
    )

    # 为整个 PDP 图设置整体标题，并指定字体样式
    plt.suptitle(f"Partial Dependence Plots for {sheet}", fontsize=22, fontweight='bold')

    # 遍历每个子图，重新计算 y 轴范围并设置中文 y 轴标签
    for ax in axes:
        ax.relim()  # 根据当前数据重新计算坐标轴范围
        ax.autoscale_view(scalex=False, scaley=True)  # 仅调整 y 轴（保持 x 轴不变）
        # 设置子图 x、y 轴标签及其字体
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel("偏依赖值", fontsize=22)  # 将 y 轴标签修改为中文“偏依赖值”
        # 设置子图刻度标注字体
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.show()
    plt.savefig(f'F:\研究生\毕业论文\区域热岛\IMG\驱动力分析\临时\pdp_plot_{sheet}.png',dpi=300)  # 按年份保存图像

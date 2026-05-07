# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 高阶 SHAP 可视化 (复合蜂群图 + 全特征 GAM 依赖图)
# 输入: CSV (通过 csv_path 和 TARGET_COL 配置)
# 输出: Enhanced_Summary.png + GAM_Dependence_All.png
# 使用: 修改 csv_path 和 TARGET_COL → 运行
# 依赖: xgboost, shap, pygam, sklearn, pandas, matplotlib
# =============================================================================
import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pygam import LinearGAM, s

# ==========================================
# 0. 功能性配置：设置新罗马字体和自定义 SHAP 色带
# ==========================================
print("\n--- 配置科研图表风格 ---")
print("1. 启用 Times New Roman 字体序列...")
times_new_roman_fonts = ['Times New Roman', 'Times', 'serif']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = times_new_roman_fonts

print("2. 定义自定义色带...")
shap_low_blue = '#008BFB'
shap_high_pink = '#FF0051'
shap_cmap_colors = [shap_low_blue, '#FFFFFF', shap_high_pink]
shap_cmap = LinearSegmentedColormap.from_list("shap_standard_diverging", shap_cmap_colors)

# 右侧 Dependence Plot 使用的红蓝发散色带
prob_cmap = plt.get_cmap('RdBu_r')

# ==========================================
# 1. 参数设置与数据路径定义
# ==========================================

csv_path = r"F:\数据分析\数据处理\20260428\CS2020-new.csv"
output_dir = r"F:\数据分析\数据处理\20260428\输出结果"
os.makedirs(output_dir, exist_ok=True)

# 目标变量列名
TARGET_COL = "Y_CS"


# ==========================================
# 2. 表格数据读取与清洗函数
# ==========================================
def load_and_prepare_csv(file_path, target_col):
    print(f"\n正在读取 CSV 数据: {file_path} ...")
    df = pd.read_csv(file_path)

    # 1. 设置 UID 为索引
    if 'UID' in df.columns:
        df.set_index('UID', inplace=True)
        print("已将 'UID' 列设置为数据索引。")
    else:
        print("警告：未检测到 'UID' 列，将使用默认数字索引。")

    if target_col not in df.columns:
        raise ValueError(f"数据中找不到目标变量列 '{target_col}'，请检查表格首行表头！")

    print("正在清洗数据：处理无穷大(inf)与缺失值(NaN)...")
    # 拦截无穷大与浮点型异常极值
    df = df.replace([np.inf, -np.inf], np.nan)

    # 直接剔除含有缺失值的行（XGBoost虽能处理缺失值，但为了后续 GAM 拟合和 SHAP 绘图的稳定性，建议剔除）
    before_drop = len(df)
    df.dropna(inplace=True)
    after_drop = len(df)
    print(f"清洗完毕。剔除了 {before_drop - after_drop} 条含有缺失值的样本，剩余有效样本量: {after_drop}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 如果你的 Y_CS 数据分布比较均匀，不需要取对数，可以注释掉下面两行！
    print("执行 log1p 对数转换以消除目标变量极值偏态...")
    y = np.where(y < 0, 0, y)
    y = pd.Series(np.log1p(y), index=X.index)

    return X, y


# ==========================================
# 3. XGBoost 建模、评估与 SHAP 分析
# ==========================================
def run_xgboost_shap(X, y):
    print("\n训练 XGBoost 模型...")
    # 可以根据你的 CSV 数据量微调 n_estimators 和 max_depth
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"  - 决定系数 (R²): {r2:.4f} | 均方根误差 (RMSE): {rmse:.4f}")

    print("计算全量 SHAP 值 (可能需要一些时间)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    return shap_values, y_pred


# ==========================================
# 4. 高阶可视化函数
# ==========================================
def plot_enhanced_summary(shap_values, X, output_folder, target_name):
    print("\n绘制复合型全局特征重要性蜂群图...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sort_inds = np.argsort(mean_abs_shap)

    X_plot = X.copy()
    new_cols = []
    for i, col in enumerate(X.columns):
        new_cols.append(f"{col} ({mean_abs_shap[i]:.3f})")
    X_plot.columns = new_cols

    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    fig, ax1 = plt.subplots(figsize=(10, 8))

    shap.summary_plot(shap_values, X_plot, show=False)
    ax1 = plt.gca()
    ax1.set_xlabel("SHAP value (Impact on Output)", fontsize=14, fontfamily='serif')

    ax2 = ax1.twiny()
    sorted_means = mean_abs_shap[sort_inds]
    y_pos = np.arange(len(sorted_means))

    # ==========================================
    # [修改点 1]：在这里修改条形图的颜色 (color) 和透明度 (alpha)
    # 例如：改用浅灰色 'lightgray' 或者 '#D3D3D3'、#A9C6D9
    # ==========================================
    ax2.barh(y_pos, sorted_means, color='#A9C6D9', alpha=0.2, height=0.6, zorder=0)

    ax2.set_xlabel("Mean(|SHAP|) - Global Importance", fontsize=14, fontfamily='serif')
    ax2.set_xlim(0, max(sorted_means) * 1.1)

    # ==========================================
    # [修改点 2]：在这里修改图例的字体大小
    # ==========================================
    cb_ax = fig.axes[-1]
    # fontsize=16 控制 "Feature Value (Low → High)" 的大小
    cb_ax.set_ylabel("Feature Value (Low → High)", fontsize=16, fontfamily='serif')
    # labelsize=14 控制图例两端 "High" 和 "Low" 的大小
    cb_ax.tick_params(axis='y', labelsize=15)

    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{target_name}_Enhanced_Summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.rcdefaults()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = times_new_roman_fonts


def plot_advanced_dependence_gam_all(shap_values, X, y_pred, output_folder, target_name):
    print(f"\n正在为所有 {X.shape[1]} 个因子绘制带 GAM 平滑的依赖图...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_features = X.columns[sorted_indices]

    num_features = len(sorted_features)
    cols = 3
    rows = int(np.ceil(num_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    if num_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 随机抽样以提升 GAM 拟合和绘图速度
    sample_size = min(10000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_samp = X.iloc[idx].values
    shap_samp = shap_values[idx]
    y_pred_samp = y_pred[idx]

    for i, (feat_idx, feat_name) in enumerate(zip(sorted_indices, sorted_features)):
        ax = axes[i]
        x_val = X_samp[:, feat_idx]
        s_val = shap_samp[:, feat_idx]

        # 散点 (zorder=2)
        sc = ax.scatter(x_val, s_val, c=y_pred_samp, cmap=prob_cmap, s=8, alpha=0.7, edgecolors='none', zorder=2)

        try:
            # GAM 拟合 (zorder=3)
            gam = LinearGAM(s(0, n_splines=10)).fit(x_val, s_val)
            XX = np.linspace(x_val.min(), x_val.max(), 500)
            YY = gam.predict(XX)
            ax.plot(XX, YY, color='dimgray', linewidth=2.5, label='GAM Fit', zorder=3)

            # 寻找 Threshold (zorder=10)
            zero_crossings = np.where(np.diff(np.sign(YY)))[0]
            if len(zero_crossings) > 0:
                thresh_x = XX[zero_crossings[0]]
                ax.axvline(x=thresh_x, color='darkred', linestyle='--', linewidth=1.5,
                           label=f'Thresh: {thresh_x:.2f}', zorder=10)
        except Exception as e:
            pass

        # 寻找 Median (zorder=10)
        median_x = np.median(x_val)
        ax.axvline(x=median_x, color='teal', linestyle='-.', linewidth=1.5, label=f'Median: {median_x:.2f}', zorder=10)

        # 零轴参考线 (zorder=0)
        ax.axhline(y=0, color='lightgray', linestyle=':', linewidth=1, zorder=0)

        # 字体及修饰
        ax.set_title(feat_name, fontsize=18, fontweight='bold')
        ax.set_ylabel("SHAP Value", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='best', fontsize=12.5, framealpha=0.9)

    # 隐藏多余空白网格
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    # 拉开子图间距
    fig.subplots_adjust(hspace=0.35, wspace=0.3, right=0.88)

    # 全局 Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Predicted Value (Low → High)', fontsize=18, fontfamily='serif')
    cbar.ax.tick_params(labelsize=16)

    out_path = os.path.join(output_folder, f"{target_name}_GAM_Dependence_All.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 5. 主程序执行入口
# ==========================================
if __name__ == "__main__":
    print(f"{'=' * 50}")
    print(f"🚀 开始处理 CSV 模型任务")
    print(f"{'=' * 50}")

    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}，请检查路径！")
    else:
        # 1. 读取并清洗表格数据
        X, y = load_and_prepare_csv(csv_path, TARGET_COL)

        # 2. 训练并计算 SHAP
        shap_values, y_pred = run_xgboost_shap(X, y)

        # 3. 绘制带有背景全局重要性的复合蜂群图
        plot_enhanced_summary(shap_values, X, output_dir, TARGET_COL)

        # 4. 为全部特征绘制带有 GAM 拟合的全家福依赖图
        plot_advanced_dependence_gam_all(shap_values, X, y_pred, output_dir, TARGET_COL)

        print(f"\n✅ 基于 CSV 表格的高阶学术图表生成完毕！保存在: {output_dir}")
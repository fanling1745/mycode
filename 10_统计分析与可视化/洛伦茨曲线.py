# -*- coding: utf-8 -*-
# =============================================================================
# 功能: 绘制文化遗产/设施空间分布的洛伦茨曲线 (累积面积 vs 累积密度)
# 输入: 城市矢量 + 分类点数据
# 输出: Lorenz_Curve_{类型}.png
# 使用: 修改 cities / points 路径和字段名 → 运行
# 依赖: geopandas, pandas, numpy, matplotlib
# =============================================================================
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ==========================================
# 0. 全局可视化风格设置 & 城市中英文映射字典
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']

# 【新增】：城市中英文映射字典
CITY_EN_MAP = {
    "杭州市": "Hangzhou",
    "宁波市": "Ningbo",
    "温州市": "Wenzhou",
    "嘉兴市": "Jiaxing",
    "湖州市": "Huzhou",
    "绍兴市": "Shaoxing",
    "金华市": "Jinhua",
    "衢州市": "Quzhou",
    "舟山市": "Zhoushan",
    "台州市": "Taizhou",
    "丽水市": "Lishui"
}

# ==========================================
# 1. 读取数据
# ==========================================
print("正在加载数据...")
# 请替换为实际路径
cities = gpd.read_file(r'E:\数据分析\数据处理\测试\市矢量.shp')
points = gpd.read_file(r'E:\数据分析\数据处理\测试\五类点prj.shp')

city_field = 'name'  # 城市名称字段
area_field = 'Shape_Area'  # 面积字段
type_field = '文化'  # 遗产类型字段

# 预处理：汇总各市总面积
city_stats = cities.groupby(city_field)[area_field].sum().reset_index()

# ==========================================
# 2. 空间连接与分类统计
# ==========================================
pts_in_cities = gpd.sjoin(points, cities[[city_field, 'geometry']], how="left", predicate="intersects")
# 提取有效点位
valid_pts = pts_in_cities.dropna(subset=[type_field])
pt_counts = valid_pts.groupby([city_field, type_field]).size().reset_index(name='count')

# 获取所有的文化类型列表
culture_types = valid_pts[type_field].unique()

# ==========================================
# 3. 循环为每种文化类型单独绘图 (复刻图片样式)
# ==========================================
for c_type in culture_types:
    print(f"正在绘制类型: {c_type}...")

    subset = pt_counts[pt_counts[type_field] == c_type]
    # 保留所有城市，没有点的城市 count 填 0
    df = city_stats.merge(subset, on=city_field, how='left').fillna(0)

    # 计算密度。没有点的城市密度为 0。
    # 按照密度升序排序；如果密度相同（如都为0），则按面积升序排序
    df['density'] = df['count'] / df[area_field]
    df.sort_values(by=['density', area_field], ascending=[True, True], inplace=True)

    # 计算累积占比
    df['cum_area_pct'] = df[area_field].cumsum() / df[area_field].sum()
    df['cum_count_pct'] = df['count'].cumsum() / df['count'].sum()

    plot_x = [0] + df['cum_area_pct'].tolist()
    plot_y = [0] + df['cum_count_pct'].tolist()

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(9, 7))  # 设置为接近正方形的比例

    # 1. 绘制图片中的红色虚线 (Uniform distribution line)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#df4132', linewidth=2.5, zorder=1)
    # 在原点和终点添加橙色圆点
    ax.plot([0, 1], [0, 1], marker='o', color='none', markeredgecolor='#e67e22', markerfacecolor='#e67e22',
            markersize=8, zorder=2)

    # 2. 绘制实际数据曲线 (Teal / 青蓝色实线带圆点)
    ax.plot(plot_x, plot_y, color='#1f9fa0', marker='o', markersize=6, linewidth=1.8, zorder=3)

    # 3. 设置 X 轴的城市标注
    prev_x = 0
    for idx, row in df.iterrows():
        curr_x = row['cum_area_pct']
        mid_x = (prev_x + curr_x) / 2

        # 绘制浅色垂直分隔线，帮助区分城市区间
        ax.axvline(curr_x, color='#eeeeee', linestyle='-', zorder=0)

        # 【核心修改 4】：获取中文名并转换为英文
        city_name_cn = row[city_field]
        # 使用字典匹配，如果字典里没有（防止出现脏数据），默认保留原中文名
        city_name_en = CITY_EN_MAP.get(city_name_cn, city_name_cn)

        # 写入英文字符
        ax.text(mid_x, 0.02, city_name_en, rotation=90, va='bottom', ha='center', fontsize=14, color='#555555')

        prev_x = curr_x

    # 4. 坐标轴及刻度样式 (0%, 20%, 40%...)
    ax.set_xlabel('Cumulative Area Proportion', fontsize=16, labelpad=10)
    ax.set_ylabel('Cumulative Density Weight', fontsize=16, labelpad=10)

    ticks = np.arange(0, 1.05, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{int(x * 100)}%' for x in ticks], fontsize=13, color='#333333')
    ax.set_yticklabels([f'{int(x * 100)}%' for x in ticks], fontsize=13, color='#333333')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)  # Y轴稍微留一点余量

    # 5. 图表边框样式 (浅灰色边框，无内部网格)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.5)
    ax.grid(False)  # 关闭网格，和图片保持一致

    # 6. 自定义底部图例 (完美复刻原图图例)
    teal_line = mlines.Line2D([], [], color='#1f9fa0', marker='o', markersize=6, label='Cumulative Density Weight')
    red_dashed = mlines.Line2D([], [], color='#df4132', linestyle='--', marker='o', markerfacecolor='#e67e22',
                               markeredgecolor='#e67e22', label='uniform distribution line')

    ax.legend(handles=[teal_line, red_dashed],
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2,
              frameon=False,
              fontsize=11)

    # 7. 保存文件 (文件名体现差异，图片内不体现)
    plt.tight_layout()
    # 清理文件名中可能导致保存失败的特殊字符
    safe_type_name = str(c_type).replace('/', '_').replace('\\', '_')
    output_filename = f'Lorenz_Curve_{safe_type_name}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"[{c_type}] 绘制完成，已保存为: {output_filename}")

print("所有图表处理完毕！")
# -*- coding: utf-8 -*-
# =============================================================================
# 功能: VIF 共线性诊断 → LassoCV 特征选择 → 随机森林建模
# 输入: Excel + optimal_features 变量列表
# 输出: vif_analysis.png / lasso_mse_vs_alpha.png / rf_feature_importance.png + CSV 结果
# 使用: 修改 Excel 路径和 optimal_features → 运行
# 依赖: sklearn, statsmodels, matplotlib, seaborn, pandas, numpy
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# 假设我们已经有了这些数据
optimal_features = ['BCD', 'BD', 'MBH', 'HBH', 'HBV', 'HCV', 'SDBH', 'MBV', 'SDBV', 'VCV',
                   'IJI', 'CONTAG', 'PD', 'FRAC_MN', 'SPLIT', 'MSIDI', 'SHEI', 'SIEI', 'AI']

# 1. 数据准备
# 示例数据加载方式(请根据实际情况修改):
data = pd.read_excel(r'F:\项目\城乡聚落三维形态\数据\驱动力分析\筛选特征\sample.xlsx', sheet_name="Sheet1")
X = data[optimal_features]
y = data.iloc[:, 0]  # 假设第一列是目标变量

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=optimal_features)

# ==============================================
# 新增部分1: 共线性诊断
# ==============================================
print("\n正在进行共线性诊断...")

# 计算VIF(方差膨胀因子)
vif_data = pd.DataFrame()
vif_data["feature"] = optimal_features
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(len(optimal_features))]

# 按VIF值排序
vif_data = vif_data.sort_values(by="VIF", ascending=False)

# 打印VIF结果
print("\nVIF共线性诊断结果:")
print(vif_data)

# 可视化VIF结果
plt.figure(figsize=(10, 6))
sns.barplot(x="VIF", y="feature", data=vif_data, palette="viridis")
plt.axvline(x=5, color='r', linestyle='--', label='VIF=5')
plt.axvline(x=10, color='orange', linestyle='--', label='VIF=10 ')
#plt.title("特征VIF值(共线性诊断)")
plt.xlabel("VIF")
plt.ylabel("feature")
plt.legend()
plt.tight_layout()
plt.savefig('vif_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 使用LassoCV进行特征选择
print("\n正在进行Lasso共线性诊断与特征选择...")

# 设置LassoCV参数
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 0, 100),  # 创建从10^-4到10^0的100个alpha值
    cv=5,                            # 5折交叉验证
    max_iter=10000,                  # 最大迭代次数
    random_state=42,                 # 随机种子
    n_jobs=-1                        # 使用所有CPU核心
)

# 拟合模型
lasso_cv.fit(X_scaled, y)

# 4. 分析结果
# 获取每个alpha对应的MSE
mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
mse_std = np.std(lasso_cv.mse_path_, axis=1)

# 可视化alpha与MSE的关系
plt.figure(figsize=(10, 6))
plt.semilogx(lasso_cv.alphas_, mse_mean, 'b-', label='Mean MSE')
plt.fill_between(lasso_cv.alphas_,
                mse_mean - mse_std,
                mse_mean + mse_std,
                alpha=0.2, color='b')
plt.axvline(lasso_cv.alpha_, color='r', linestyle='--',
            label=f'Optimal alpha: {lasso_cv.alpha_:.4f}')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('LassoCV MSE vs Alpha')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('lasso_mse_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 选择非零系数特征
# 获取系数绝对值大于1e-5的特征(可根据需要调整阈值)
selected_mask = np.abs(lasso_cv.coef_) > 0.1
final_features = np.array(optimal_features)[selected_mask].tolist()

print("\n共线性诊断与特征选择结果:")
print(f"- 最优alpha值: {lasso_cv.alpha_:.6f}")
print(f"- 原始特征数: {len(optimal_features)}")
print(f"- 筛选后特征数: {len(final_features)}")
print(f"- 剔除的特征: {set(optimal_features) - set(final_features)}")
print(f"- 最终保留的特征: {final_features}")

# 6. 保存结果
result_df = pd.DataFrame({
    'Feature': optimal_features,
    'Coefficient': lasso_cv.coef_,
    'Selected': selected_mask,
    'VIF': vif_data['VIF'].values
}).sort_values('Coefficient', key=abs, ascending=False)

result_df.to_csv('lasso_feature_selection_results.csv', index=False)

# ==============================================
# 新增部分2: 随机森林回归模型
# ==============================================
print("\n构建随机森林回归模型...")

# 使用筛选后的特征
X_final = X[final_features]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf_model = RandomForestRegressor(
    n_estimators=100,  # 树的数量
    max_depth=None,    # 树的最大深度
    min_samples_split=2, # 分裂内部节点所需的最小样本数
    min_samples_leaf=1,  # 叶节点所需的最小样本数
    random_state=42,
    n_jobs=-1
)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测
y_pred = rf_model.predict(X_test)

# 计算模型精度
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n随机森林模型性能:")
print(f"- 均方误差(MSE): {mse:.4f}")
print(f"- 均方根误差(RMSE): {rmse:.4f}")
print(f"- R平方值(R²): {r2:.4f}")

# 特征重要性可视化
feature_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存特征重要性结果
feature_importance.to_csv('rf_feature_importance.csv', index=False)

print("\n分析完成！所有结果已保存。")
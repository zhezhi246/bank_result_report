import os
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 0. 环境设置与防报错处理
# ==========================================
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 消除 Windows 系统下 joblib 核心查询的警告
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

input_file = r"C:\Users\折纸\Desktop\研究生\研一下\机器学习\Data Cleaning\Cleaned_Churn_Modelling.csv"
project_root = os.path.dirname(os.path.dirname(input_file))
output_dir = os.path.join(project_root, "Ultimate_Benchmark")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[环境设置] 已创建终极对比实验目录：{output_dir}")

# ==========================================
# 1. 数据加载与基础处理
# ==========================================
print("[数据加载] 正在读取数据集...")
df = pd.read_csv(input_file)
cols_to_drop = [col for col in ['Gender_Label', 'Country_Label', 'Status'] if col in df.columns]
df = df.drop(columns=cols_to_drop)


# 临时还原国家标签用于计算群体统计特征
def get_country(row):
    if row.get('Geography_Germany', 0) == 1:
        return 'Germany'
    elif row.get('Geography_Spain', 0) == 1:
        return 'Spain'
    else:
        return 'France'


df['Country_Temp'] = df.apply(get_country, axis=1)

X = df.drop('Exited', axis=1)
y = df['Exited']

# 划分训练集与测试集 (严格防数据泄露)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ==========================================
# 2. 四大数据集场景的精密构造
# ==========================================
print("[特征工程] 正在构造 4 种实验场景的数据集...")
datasets = {}

# 场景 1: Baseline (纯 XGBoost)
X_tr_base = X_train.drop(columns=['Country_Temp'])
X_te_base = X_test.drop(columns=['Country_Temp'])
datasets['1. Pure XGBoost'] = (X_tr_base, X_te_base)

# 场景 2: 仅加群体统计特征 (Group_Stat Only)
country_mean = X_train.groupby('Country_Temp')['Balance'].mean().to_dict()
X_tr_gs = X_tr_base.copy()
X_te_gs = X_te_base.copy()
X_tr_gs['Group_Stat'] = X_train['Balance'] - X_train['Country_Temp'].map(country_mean)
X_te_gs['Group_Stat'] = X_test['Balance'] - X_test['Country_Temp'].map(country_mean)
datasets['2. XGBoost + Group_Stat'] = (X_tr_gs, X_te_gs)

# 场景 3: 仅加 K-Means 聚类特征 (KMeans Only)
cluster_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'IsActiveMember']
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train[cluster_features])
X_te_scaled = scaler.transform(X_test[cluster_features])

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
tr_ids = kmeans.fit_predict(X_tr_scaled)
te_ids = kmeans.predict(X_te_scaled)
tr_dists = kmeans.transform(X_tr_scaled)
te_dists = kmeans.transform(X_te_scaled)

X_tr_km = X_tr_base.copy()
X_te_km = X_te_base.copy()
X_tr_km['KMeans_Cluster_ID'] = tr_ids
X_te_km['KMeans_Cluster_ID'] = te_ids
for i in range(n_clusters):
    X_tr_km[f'Dist_to_Cluster_{i}'] = tr_dists[:, i]
    X_te_km[f'Dist_to_Cluster_{i}'] = te_dists[:, i]
datasets['3. XGBoost + KMeans'] = (X_tr_km, X_te_km)

# 场景 4: 混合叠加 (Group_Stat + KMeans)
X_tr_both = X_tr_gs.copy()
X_te_both = X_te_gs.copy()
X_tr_both['KMeans_Cluster_ID'] = tr_ids
X_te_both['KMeans_Cluster_ID'] = te_ids
for i in range(n_clusters):
    X_tr_both[f'Dist_to_Cluster_{i}'] = tr_dists[:, i]
    X_te_both[f'Dist_to_Cluster_{i}'] = te_dists[:, i]
datasets['4. XGBoost + Both'] = (X_tr_both, X_te_both)

# ==========================================
# 3. 模型训练与评估收集
# ==========================================
print("\n[模型训练] 开始执行四组交叉训练...")
results = []
confusion_matrices = {}

for scenario_name, (X_tr, X_te) in datasets.items():
    # 统一使用带有正负样本权重调整的 XGBoost
    model = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    # 提取三大核心指标
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        'Model Scenario': scenario_name,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    })
    confusion_matrices[scenario_name] = cm

    print(f"{scenario_name:25} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

# 转换为供绘图使用的长格式(melt)
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir, "Ultimate_Benchmark_Metrics.csv"), index=False)
df_melted = df_results.melt(id_vars='Model Scenario', var_name='Metric', value_name='Score')

# ==========================================
# 4. 可视化：同框展示三大指标 (柱状图)
# ==========================================
print("\n[可视化] 正在生成综合性能对比图 (Precision / Recall / F1) ...")
plt.figure(figsize=(14, 7))

# 画在一张图里，X轴为三个指标，不同颜色代表四种模型场景
sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model Scenario', palette='Set1')
plt.title("XGBoost Performance Benchmark: Baseline vs Advanced Feature Engineering", fontsize=16, weight='bold')
plt.ylabel("Score", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)

# 动态调整 y 轴上限以便放置图例和文本
min_score = df_melted['Score'].min() - 0.1
plt.ylim(max(0, min_score), 1.05)

# 在柱子上标注具体数值
for p in plt.gca().patches:
    height = p.get_height()
    if height > 0:
        plt.gca().annotate(f"{height:.3f}",
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=11,
                           xytext=(0, 5), textcoords='offset points')

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
plt.tight_layout()
metrics_plot_path = os.path.join(output_dir, "01_Benchmark_Metrics_Comparison.png")
plt.savefig(metrics_plot_path, bbox_inches='tight')
plt.close()

# ==========================================
# 5. 可视化：2x2 混淆矩阵演进图
# ==========================================
print("[可视化] 正在生成 2x2 混淆矩阵演进图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

# 给四张图安排渐进式的色系
color_maps = ['Blues', 'Greens', 'Purples', 'Oranges']

for idx, (scenario_name, cm) in enumerate(confusion_matrices.items()):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap=color_maps[idx], ax=ax, cbar=False, annot_kws={"size": 16})

    ax.set_title(scenario_name, fontsize=15, weight='bold')
    ax.set_xlabel('Predicted Label (0:Retained, 1:Exited)', fontsize=12)
    ax.set_ylabel('True Label (0:Retained, 1:Exited)', fontsize=12)

plt.suptitle("Confusion Matrix Evolution: Synergy of K-Means and Group Statistics", fontsize=18, y=1.03)
plt.tight_layout()
cm_plot_path = os.path.join(output_dir, "02_Benchmark_Confusion_Matrices.png")
plt.savefig(cm_plot_path, bbox_inches='tight')
plt.close()

print(f"\n[大功告成] 终极对比实验完成！图表已保存至: {output_dir}")
import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# 0. 环境设置与数据加载
# ==========================================
# 请确保路径与您之前的代码一致
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


input_file = r"C:\Users\折纸\Desktop\研究生\研一下\数据挖掘\数据挖掘作业二—新\Data Cleaning\Cleaned_Churn_Modelling.csv"
project_root = os.path.dirname(os.path.dirname(input_file))
output_dir = os.path.join(project_root, "Customer_Segmentation")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("[数据加载] 正在读取清洗后的数据集...")
df = pd.read_csv(input_file)

# 聚类时，我们需要去掉目标变量 'Exited' 以及不具备业务解释性的临时标签
cols_to_drop = [col for col in ['Exited', 'Gender_Label', 'Country_Label', 'Status'] if col in df.columns]
X_cluster = df.drop(columns=cols_to_drop)

# 为了雷达图好看，我们提取最具代表性的连续/等级变量
radar_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'IsActiveMember']
X_radar_base = X_cluster[radar_features].copy()

# ==========================================
# 1. 数据标准化与 K-Means 聚类
# ==========================================
print("[模型训练] 正在进行 StandardScaler 标准化与 K-Means 聚类...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_radar_base)

# 假设通过肘部法则(Elbow Method)，我们选择 K=4 个客群
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ==========================================
# 2. 计算各群体的特征均值，并缩放用于雷达图
# ==========================================
# 计算每个簇的真实均值（用于业务解释）
cluster_means = df.groupby('Cluster')[radar_features].mean()
print("\n[业务画像] 各客群真实特征均值：")
print(cluster_means.round(2))
cluster_means.to_csv(os.path.join(output_dir, "Cluster_Real_Means.csv"))

# 为了画雷达图，将均值缩放到 0-1 之间
minmax_scaler = MinMaxScaler()
cluster_means_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index
)

# ==========================================
# 3. 绘制客户画像雷达图 (Radar Chart)
# ==========================================
print("\n[可视化] 正在生成客户画像雷达图...")
labels = np.array(radar_features)
num_vars = len(labels)

# 计算角度
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
cluster_names = ['Segment 0', 'Segment 1', 'Segment 2', 'Segment 3']  # 后续可根据业务命名

for i in range(4):
    values = cluster_means_scaled.iloc[i].tolist()
    values += values[:1]  # 闭合图形

    ax.plot(angles, values, color=colors[i], linewidth=2, label=cluster_names[i])
    ax.fill(angles, values, color=colors[i], alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)

# 去掉径向标签（0-1的刻度），让图更清爽
ax.set_yticklabels([])

plt.title('Customer Segmentation Profiles (Radar Chart)', size=20, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

radar_path = os.path.join(output_dir, "Customer_Segments_Radar.png")
plt.savefig(radar_path, bbox_inches='tight')
plt.close()

print(f"[大功告成] 聚类分析结束！雷达图已保存至: {radar_path}")
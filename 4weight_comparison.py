import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 环境设置与数据准备
# ==========================================
plt.rcParams['figure.dpi'] = 300

input_file = r"C:\Users\折纸\Desktop\研究生\研一下\数据挖掘\数据挖掘作业二—新\Data Cleaning\Cleaned_Churn_Modelling.csv"
project_root = os.path.dirname(os.path.dirname(input_file))
output_dir = os.path.join(project_root, "Model_Evaluation")

print("[数据加载] 正在读取清洗后的数据集...")
df = pd.read_csv(input_file)
cols_to_drop = [col for col in ['Gender_Label', 'Country_Label', 'Status'] if col in df.columns]
df = df.drop(columns=cols_to_drop)

X = df.drop('Exited', axis=1)
y = df['Exited']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 计算正负样本比例（用于 XGBoost 加权）
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# ==========================================
# 1. 定义四组对比模型
# ==========================================
models = {
    "RF (Unweighted)": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8),
    "RF (Weighted)": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100, max_depth=8),
    "XGBoost (Unweighted)": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    "XGBoost (Weighted)": xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
}

# ==========================================
# 2. 训练与评估
# ==========================================
print("\n[对比实验] 开始训练并评估有无权重调整的模型...")
results = []
cms = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 我们重点关注流失类（1）的召回率和精确率
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    cms[name] = cm
    results.append({
        'Model': name,
        'Recall (查全率)': recall,
        'Precision (精确率)': precision,
        'F1-Score': f1
    })
    print(f" > {name:20} | 召回率: {recall:.4f}, 精确率: {precision:.4f}, F1: {f1:.4f}")

df_results = pd.DataFrame(results)

# ==========================================
# 3. 可视化对比结果
# ==========================================
print("\n[可视化] 正在生成对比图表...")

# 3.1 绘制核心指标柱状对比图
# 转换数据格式以适应 seaborn 多重柱状图
df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='viridis')
plt.title("Impact of Class Weights on Model Performance (Focus on Exited=1)", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.0)
plt.xticks(rotation=15)
plt.legend(loc='upper right')

# 在柱子上标注数值
for p in plt.gca().patches:
    if p.get_height() > 0:
        plt.gca().annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_Weight_Adjustment_Comparison_Bar.png"))
plt.close()

# 3.2 绘制 2x2 混淆矩阵直观对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd' if 'Weighted' in name else 'Blues',
                ax=axes[idx], cbar=False, annot_kws={"size": 14})
    axes[idx].set_title(name, fontsize=14, weight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=11)
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_xticklabels(['Retained(0)', 'Exited(1)'])
    axes[idx].set_yticklabels(['Retained(0)', 'Exited(1)'])

plt.tight_layout()
plt.suptitle("Confusion Matrix: Unweighted vs Weighted", fontsize=16, y=1.02)
plt.savefig(os.path.join(output_dir, "05_Weight_Adjustment_Confusion_Matrix.png"))
plt.close()

print(f"\n[实验完成] 对比图表已保存至：{output_dir}")
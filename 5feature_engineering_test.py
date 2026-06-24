import os
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix

# ==========================================
# 0. 环境设置与数据加载
# ==========================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 确保路径正确
input_file = r"C:\Users\折纸\Desktop\研究生\研一下\机器学习\Data Cleaning\Cleaned_Churn_Modelling.csv"
project_root = os.path.dirname(os.path.dirname(input_file))
output_dir = os.path.join(project_root, "Tree_Ablation_Study")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("[数据加载] 正在读取清洗后的数据集...")
df = pd.read_csv(input_file)
cols_to_drop = [col for col in ['Gender_Label', 'Country_Label', 'Status'] if col in df.columns]
df = df.drop(columns=cols_to_drop)


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)


# ==========================================
# 1. 定义特征工程处理函数
# ==========================================
def apply_feature_engineering(X_tr, X_te, scenario):
    X_train_new = X_tr.copy()
    X_test_new = X_te.copy()

    # 零余额强标记
    if scenario in ['Zero_Balance', 'All']:
        X_train_new['Is_Zero_Balance'] = (X_train_new['Balance'] == 0).astype(int)
        X_test_new['Is_Zero_Balance'] = (X_test_new['Balance'] == 0).astype(int)

    # 高危客群强交叉
    if scenario in ['High_Risk', 'All']:
        X_train_new['Is_Germany_Female'] = (
                    (X_train_new['Geography_Germany'] == 1) & (X_train_new['Gender'] == 0)).astype(int)
        X_test_new['Is_Germany_Female'] = ((X_test_new['Geography_Germany'] == 1) & (X_test_new['Gender'] == 0)).astype(
            int)

    # 群体统计特征 (相对本地平均余额差)
    if scenario in ['Group_Stat', 'All']:
        country_mean = X_train_new.groupby('Country_Temp')['Balance'].mean().to_dict()
        X_train_new['Country_Mean_Balance'] = X_train_new['Country_Temp'].map(country_mean)
        X_test_new['Country_Mean_Balance'] = X_test_new['Country_Temp'].map(country_mean)

        X_train_new['Balance_Diff_Country_Mean'] = X_train_new['Balance'] - X_train_new['Country_Mean_Balance']
        X_test_new['Balance_Diff_Country_Mean'] = X_test_new['Balance'] - X_test_new['Country_Mean_Balance']

        X_train_new = X_train_new.drop(columns=['Country_Mean_Balance'])
        X_test_new = X_test_new.drop(columns=['Country_Mean_Balance'])

    X_train_new = X_train_new.drop(columns=['Country_Temp'])
    X_test_new = X_test_new.drop(columns=['Country_Temp'])

    return X_train_new, X_test_new


scenarios = ['Baseline', 'Group_Stat', 'Zero_Balance', 'High_Risk', 'All']

# ==========================================
# 2. 核心实验循环
# ==========================================
results = []
confusion_matrices = {}

print("\n[开始实验] 正在遍历模型与特征场景...")
for scenario in scenarios:
    print(f"\n--- 特征场景: {scenario} ---")
    X_train_proc, X_test_proc = apply_feature_engineering(X_train, X_test, scenario)

    models = {
        "XGBoost": xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    }

    for model_name, model in models.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        y_pred_proba = model.predict_proba(X_test_proc)[:, 1]

        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        results.append({'Model': model_name, 'Scenario': scenario, 'Recall': round(recall, 4), 'F1-Score': round(f1, 4),
                        'AUC': round(auc, 4)})
        print(f"[{model_name}] Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # 收集这三个关键节点的混淆矩阵用于画图
        if scenario in ['Baseline', 'Group_Stat', 'All']:
            key = f"{model_name}_{scenario}"
            confusion_matrices[key] = confusion_matrix(y_test, y_pred)

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir, "Ablation_Study_Results.csv"), index=False)

# ==========================================
# 3. 绘图：全面评估柱状图
# ==========================================
print("\n[可视化] 正在生成综合评估指标柱状图...")
fig, axes = plt.subplots(3, 1, figsize=(14, 18))
metrics = ['Recall', 'F1-Score', 'AUC']
for i, metric in enumerate(metrics):
    sns.barplot(data=df_results, x='Model', y=metric, hue='Scenario', ax=axes[i], palette="Set2")
    axes[i].set_title(f"{metric} Comparison across Feature Scenarios", fontsize=15, weight='bold')
    axes[i].set_ylabel(metric, fontsize=12)
    axes[i].set_xlabel("")
    min_val = df_results[metric].min() - 0.05
    axes[i].set_ylim(max(0, min_val), 1.05)
    axes[i].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    for p in axes[i].patches:
        height = p.get_height()
        if height > 0:
            axes[i].annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom',
                             fontsize=10, xytext=(0, 4), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_Metrics_Comparison_Bars.png"), bbox_inches='tight')
plt.close()

# ==========================================
# 4. 绘图：集成混淆矩阵对比 (2x3 面板)
# ==========================================
print("[可视化] 正在生成 集成混淆矩阵面板 (Baseline -> Group_Stat -> All)...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 定义绘图矩阵：(数据Key, 坐标轴位置, 图表标题)
plot_keys_all = [
    ("XGBoost_Baseline", axes[0], "XGB: Baseline"),
    ("XGBoost_Group_Stat", axes[1], "XGB: Group_Stat Only"),
    ("XGBoost_All", axes[2], "XGB: All Features")
]

for key, ax, title in plot_keys_all:
    cm = confusion_matrices[key]

    # 根据所处阶段动态分配颜色
    if 'Baseline' in title:
        cmap = 'Blues'
    elif 'Group_Stat' in title:
        cmap = 'Greens'
    else:
        cmap = 'Oranges'

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False, annot_kws={"size": 16})
    ax.set_title(title, fontsize=15, weight='bold')
    ax.set_xlabel('Predicted Label (0:Retained, 1:Exited)', fontsize=12)
    ax.set_ylabel('True Label (0:Retained, 1:Exited)', fontsize=12)
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])

plt.suptitle("Evolution of Confusion Matrix: Baseline vs Group_Stat vs All Features", fontsize=20, y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_Confusion_Matrix_Evolution.png"), bbox_inches='tight')
plt.close()

print(f"\n[大功告成] 所有实验结束！全新的2x3混淆矩阵演进图已保存至: 02_Confusion_Matrix_Evolution.png")
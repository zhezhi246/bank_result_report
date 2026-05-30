import os
import pandas as pd
import numpy as np

# 强制使用静态后端，避免绘图报错
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score, recall_score
import shap

# ==========================================
# 0. 环境设置与数据准备
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

input_file = r"C:\Users\折纸\Desktop\研究生\研一下\数据挖掘\数据挖掘作业二—新\Data Cleaning\Cleaned_Churn_Modelling.csv"
project_root = os.path.dirname(os.path.dirname(input_file))
output_dir = os.path.join(project_root, "Model_Evaluation")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[环境设置] 已创建模型评估输出目录：{output_dir}")

print("[数据加载] 正在读取清洗后的数据集...")
df = pd.read_csv(input_file)

# 确保所有为了 EDA 画图临时生成的非数值列被剔除（防呆设计）
cols_to_drop = [col for col in ['Gender_Label', 'Country_Label', 'Status'] if col in df.columns]
df = df.drop(columns=cols_to_drop)

# 划分特征矩阵 (X) 和目标变量 (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# 划分训练集和测试集 (80% 训练，20% 测试，启用分层抽样保持流失比例一致)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[数据划分] 训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# ==========================================
# 1. 模型初始化与独立归一化处理
# ==========================================
# 计算用于 XGBoost 的正负样本比例权重
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=6),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100, max_depth=8),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
}

# 单独为逻辑回归准备标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. 模型训练与评估
# ==========================================
print("\n[模型训练] 开始训练并评估模型...")
results = {}

for name, model in models.items():
    # 根据模型名称决定是否使用标准化数据
    curr_X_train = X_train_scaled if name == "Logistic Regression" else X_train
    curr_X_test = X_test_scaled if name == "Logistic Regression" else X_test

    # 模型训练
    model.fit(curr_X_train, y_train)

    # 预测概率（用于 ROC-AUC）和预测类别（用于混淆矩阵、召回率、F1）
    y_pred = model.predict(curr_X_test)
    y_pred_proba = model.predict_proba(curr_X_test)[:, 1]

    # 计算指标
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    # 存储结果
    results[name] = {
        'Recall': recall, 'F1-Score': f1, 'AUC': roc_auc,
        'FPR': fpr, 'TPR': tpr, 'CM': cm,
        'Model_Obj': model  # 保存模型对象用于后续 SHAP 分析
    }
    print(f" > {name} 完成 | 召回率: {recall:.4f}, F1分数: {f1:.4f}, AUC: {roc_auc:.4f}")

# ==========================================
# 3. 可视化：综合指标对比 (ROC-AUC & 混淆矩阵)
# ==========================================
print("\n[可视化] 正在生成模型对比图表...")

# 3.1 绘制多模型 ROC 曲线比较
plt.figure(figsize=(10, 8))
for name, metrics in results.items():
    plt.plot(metrics['FPR'], metrics['TPR'], lw=2, label=f"{name} (AUC = {metrics['AUC']:.3f})")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-AUC Curve Comparison', fontsize=16)
plt.legend(loc="lower right", fontsize=11)
plt.savefig(os.path.join(output_dir, "01_ROC_Curve_Comparison.png"))
plt.close()

# 3.2 绘制 2x2 混淆矩阵面板
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, metrics) in enumerate(results.items()):
    sns.heatmap(metrics['CM'], annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                annot_kws={"size": 14})
    axes[idx].set_title(f"{name}", fontsize=14)
    axes[idx].set_xlabel('Predicted Label', fontsize=11)
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_xticklabels(['Retained (0)', 'Exited (1)'])
    axes[idx].set_yticklabels(['Retained (0)', 'Exited (1)'])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_Confusion_Matrix_Grid.png"))
plt.close()

# ==========================================
# 4. 模型解释性：SHAP 分析 (基于 XGBoost)
# ==========================================
print("[模型解释] 正在生成 XGBoost 模型的 SHAP 解释图...")
xgb_model = results['XGBoost']['Model_Obj']

# 构建树模型的 SHAP 解释器
explainer = shap.TreeExplainer(xgb_model)
# 计算测试集的 SHAP 值
shap_values = explainer.shap_values(X_test)

# 绘制 SHAP Summary Plot
plt.figure(figsize=(10, 8))
# show=False 防止控制台卡住
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot (XGBoost Feature Impacts)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_SHAP_Summary_Plot.png"))
plt.close()

print(f"\n[流程结束] 模型构建与评估全部完成！报告图表已保存至：{output_dir}")
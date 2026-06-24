import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
matplotlib.use('Agg')
# ==========================================
# 0. 环境设置与数据准备
# ==========================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


input_file = r"C:\Users\折纸\Desktop\研究生\研一下\机器学习\Data Cleaning\Cleaned_Churn_Modelling.csv"
output_dir = os.path.dirname(input_file)
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


# 复用之前的特征工程逻辑
def get_features(X_tr, X_te, mode):
    X_tr_new, X_te_new = X_tr.copy(), X_te.copy()
    if mode == 'Baseline':
        return X_tr_new.drop('Country_Temp', axis=1), X_te_new.drop('Country_Temp', axis=1)

    # 加入特征1 (Group_Stat)
    country_mean = X_tr_new.groupby('Country_Temp')['Balance'].mean().to_dict()
    X_tr_new['Group_Stat'] = X_tr_new['Balance'] - X_tr_new['Country_Temp'].map(country_mean)
    X_te_new['Group_Stat'] = X_te_new['Balance'] - X_te_new['Country_Temp'].map(country_mean)

    if mode == 'One_Feature':
        return X_tr_new.drop('Country_Temp', axis=1), X_te_new.drop('Country_Temp', axis=1)

    # 加入特征2 & 3
    X_tr_new['Is_Zero_Balance'] = (X_tr_new['Balance'] == 0).astype(int)
    X_te_new['Is_Zero_Balance'] = (X_te_new['Balance'] == 0).astype(int)
    X_tr_new['Is_Germany_Female'] = ((X_tr_new['Geography_Germany'] == 1) & (X_tr_new['Gender'] == 0)).astype(int)
    X_te_new['Is_Germany_Female'] = ((X_te_new['Geography_Germany'] == 1) & (X_te_new['Gender'] == 0)).astype(int)

    return X_tr_new.drop('Country_Temp', axis=1), X_te_new.drop('Country_Temp', axis=1)


# ==========================================
# 1. 训练模型并获取预测概率
# ==========================================
plt.figure(figsize=(8, 6))
modes = ['Baseline', 'One_Feature', 'All_Features']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

for mode, color in zip(modes, colors):
    X_tr, X_te = get_features(X_train, X_test, mode)
    model = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_te)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{mode} (AUC = {roc_auc:.4f})')

# ==========================================
# 2. 绘图设置
# ==========================================
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve: Feature Engineering Ablation')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_dir, "XGBoost_AUC_Comparison.png"))
print(f"[完成] AUC曲线图已保存至: {output_dir}")
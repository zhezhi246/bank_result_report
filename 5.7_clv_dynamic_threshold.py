import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

C_COST = 0.1
GAMMA = 0.3

# ============================
# 1. 数据加载
# ============================
input_file = r"C:\Users\折纸\Desktop\研究生\研一下\机器学习\Data Cleaning\Cleaned_Churn_Modelling.csv"
df = pd.read_csv(input_file)
cols_to_drop = [col for col in ['Gender_Label', 'Country_Label', 'Status', 'RowNumber', 'CustomerId', 'Surname'] if
                col in df.columns]
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

y_train_arr = y_train.values
y_test_arr = y_test.values

# ============================
# 2. 特征工程：真 Baseline（无特征工程）vs Group_Stat
# ============================

# --- 2a. 真 Baseline: 仅去掉 Country_Temp, 不做任何特征工程 ---
X_train_baseline = X_train.drop(columns=['Country_Temp'])
X_test_baseline = X_test.drop(columns=['Country_Temp'])

# --- 2b. Group_Stat: 群体统计特征 (Balance_Diff_Country_Mean) ---
def apply_group_stat_features(X_tr, X_te):
    X_train_new = X_tr.copy()
    X_test_new = X_te.copy()

    country_mean = X_train_new.groupby('Country_Temp')['Balance'].mean().to_dict()
    X_train_new['Country_Mean_Balance'] = X_train_new['Country_Temp'].map(country_mean)
    X_test_new['Country_Mean_Balance'] = X_test_new['Country_Temp'].map(country_mean)

    X_train_new['Balance_Diff_Country_Mean'] = X_train_new['Balance'] - X_train_new['Country_Mean_Balance']
    X_test_new['Balance_Diff_Country_Mean'] = X_test_new['Balance'] - X_test_new['Country_Mean_Balance']

    X_train_new = X_train_new.drop(columns=['Country_Temp', 'Country_Mean_Balance'])
    X_test_new = X_test_new.drop(columns=['Country_Temp', 'Country_Mean_Balance'])

    return X_train_new, X_test_new


X_train_gs, X_test_gs = apply_group_stat_features(X_train, X_test)

# ============================
# 3. 客户价值代理指标 Vi 计算（统一用于评估）
# ============================
def compute_vi(X_data, scaler_bal=None, scaler_sal=None, is_train=True):
    if is_train:
        scaler_bal = MinMaxScaler().fit(X_data[['Balance']])
        scaler_sal = MinMaxScaler().fit(X_data[['EstimatedSalary']])

    bal_norm = scaler_bal.transform(X_data[['Balance']]).flatten()
    sal_norm = scaler_sal.transform(X_data[['EstimatedSalary']]).flatten()
    is_active = X_data['IsActiveMember'].values
    vi_raw = 0.6 * bal_norm + 0.4 * sal_norm * (1 + 0.2 * is_active)
    Vi = 1.0 + 9.0 * (vi_raw - vi_raw.min()) / (vi_raw.max() - vi_raw.min() + 1e-5)
    return Vi, scaler_bal, scaler_sal


Vi_train, s_bal, s_sal = compute_vi(X_train, is_train=True)
Vi_test, _, _ = compute_vi(X_test, scaler_bal=s_bal, scaler_sal=s_sal, is_train=False)

# ============================
# 4. 训练三个模型
# ============================

# --- 模型 A: True Baseline — 无特征工程 + 标准 XGBoost, 固定阈值 0.5 ---
model_A = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
model_A.fit(X_train_baseline, y_train_arr)
probs_A = model_A.predict_proba(X_test_baseline)[:, 1]
y_pred_A = (probs_A >= 0.5).astype(int)

# --- 模型 B: Group_Stat + 标准 XGBoost, 固定阈值 0.5 ---
model_B = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
model_B.fit(X_train_gs, y_train_arr)
probs_B = model_B.predict_proba(X_test_gs)[:, 1]
y_pred_B = (probs_B >= 0.5).astype(int)

# --- 模型 C: Group_Stat + CLV-Modified XGBoost (样本权重 + 动态阈值) ---
clv_sample_weight = np.ones(len(y_train_arr))
clv_sample_weight[y_train_arr == 1] = Vi_train[y_train_arr == 1]

model_C = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
model_C.fit(X_train_gs, y_train_arr, sample_weight=clv_sample_weight)
probs_C = model_C.predict_proba(X_test_gs)[:, 1]

K = C_COST / GAMMA
dynamic_thresholds = np.clip(K / Vi_test, 0.01, 0.99)
y_pred_C = (probs_C >= dynamic_thresholds).astype(int)

# ============================
# 5. 标准指标汇总
# ============================
model_names = [
    "True Baseline\n(No FE + XGBoost)",
    "Group Stats\n+ XGBoost",
    "Group Stats\n+ CLV-XGBoost"
]

metrics = pd.DataFrame({
    "Model": model_names,
    "AUC": [
        roc_auc_score(y_test_arr, probs_A),
        roc_auc_score(y_test_arr, probs_B),
        roc_auc_score(y_test_arr, probs_C)
    ],
    "F1-Score": [
        f1_score(y_test_arr, y_pred_A),
        f1_score(y_test_arr, y_pred_B),
        f1_score(y_test_arr, y_pred_C)
    ],
    "Recall": [
        recall_score(y_test_arr, y_pred_A),
        recall_score(y_test_arr, y_pred_B),
        recall_score(y_test_arr, y_pred_C)
    ]
})

print("\n========== 标准指标对比 ==========")
print(metrics.to_string(index=False))

# ============================
# 6. 图 1: AUC / F1 / Recall 三模型对比柱状图
# ============================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#7FB3D8', '#5DADE2', '#E74C3C']
for i, m in enumerate(["AUC", "F1-Score", "Recall"]):
    sns.barplot(data=metrics, x="Model", y=m, ax=axes[i], palette=colors)
    axes[i].set_title(m, fontsize=13, weight='bold')
    axes[i].set_ylim(0, 1.05)
    for p in axes[i].patches:
        axes[i].annotate(f"{p.get_height():.3f}",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.02), ha='center', fontsize=10)
    axes[i].tick_params(axis='x', labelsize=8)

plt.suptitle("三模型标准指标对比: True Baseline vs Group Stats + XGBoost vs CLV-Modified XGBoost",
             weight='bold', fontsize=14)
plt.tight_layout()
plt.savefig("5.7_Metrics_Comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("图 1 已保存: 5.7_Metrics_Comparison.png")

# ============================
# 7. 代价矩阵计算
# ============================
def get_cost_matrix(y_true, y_pred, vi_arr):
    cost = np.zeros((2, 2))
    for yt, yp, v in zip(y_true, y_pred, vi_arr):
        if yt == 0 and yp == 1:
            cost[0, 1] += C_COST          # FP: 营销成本
        elif yt == 1 and yp == 0:
            cost[1, 0] += v                # FN: 流失价值损失
        elif yt == 1 and yp == 1:
            cost[1, 1] += (C_COST + (1 - GAMMA) * v)  # TP: 营销成本 + 未挽回部分
    return cost


# 三个模型的混淆矩阵与代价矩阵
cm_list = [
    confusion_matrix(y_test_arr, y_pred_A),
    confusion_matrix(y_test_arr, y_pred_B),
    confusion_matrix(y_test_arr, y_pred_C)
]
cost_list = [
    get_cost_matrix(y_test_arr, y_pred_A, Vi_test),
    get_cost_matrix(y_test_arr, y_pred_B, Vi_test),
    get_cost_matrix(y_test_arr, y_pred_C, Vi_test)
]

# ============================
# 8. 图 2: 混淆矩阵 + 银行营销代价敏感矩阵 (3行 x 2列)
# ============================
fig, axes = plt.subplots(3, 2, figsize=(13, 15))

row_titles = [
    "True Baseline (No FE + XGBoost)",
    "Group Stats + XGBoost",
    "Group Stats + CLV-Modified XGBoost"
]

for row_idx in range(3):
    # 左列: 标准混淆矩阵
    sns.heatmap(cm_list[row_idx], annot=True, fmt='d', cmap='Blues',
                ax=axes[row_idx, 0], cbar=False,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
                annot_kws={"size": 14})
    axes[row_idx, 0].set_title(f'{row_titles[row_idx]}\n标准混淆矩阵', fontsize=12, weight='bold')

    # 右列: 代价敏感矩阵
    sns.heatmap(cost_list[row_idx], annot=True, fmt='.1f', cmap='Reds',
                ax=axes[row_idx, 1], cbar=False,
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
                annot_kws={"size": 14})
    axes[row_idx, 1].set_title(f'{row_titles[row_idx]}\n银行代价矩阵 (总损失: {cost_list[row_idx].sum():.1f})',
                               fontsize=12, weight='bold')

plt.suptitle("混淆矩阵与银行营销代价敏感矩阵对比 (三模型)", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("5.7_Matrices_Comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("图 2 已保存: 5.7_Matrices_Comparison.png")

# ============================
# 9. 客户价值代理指标
# ============================
def compute_clv_proxy_metrics(y_true, y_pred, vi_arr, name):
    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    total_churn_value = vi_arr[y_true == 1].sum()

    tp_value = vi_arr[tp_mask].sum()
    fn_value = vi_arr[fn_mask].sum()

    clv_recall = tp_value / (total_churn_value + 1e-5)
    avg_tp_clv = vi_arr[tp_mask].mean() if tp_mask.sum() > 0 else 0.0
    avg_fn_clv = vi_arr[fn_mask].mean() if fn_mask.sum() > 0 else 0.0

    revenue = tp_value * GAMMA
    cost_marketing = (tp_mask.sum() + fp_mask.sum()) * C_COST
    nmr = revenue - cost_marketing
    efficiency = tp_value / (cost_marketing + 1e-5)

    return {
        "Model": name,
        "TP Count": tp_mask.sum(),
        "FN Count": fn_mask.sum(),
        "FP Count": fp_mask.sum(),
        "CLV-Recall": round(clv_recall, 4),
        "Avg TP CLV": round(avg_tp_clv, 2),
        "Avg FN CLV": round(avg_fn_clv, 2),
        "Total CLV Recovered": round(tp_value, 1),
        "Total CLV Missed": round(fn_value, 1),
        "NMR (Net Monetary Return)": round(nmr, 1),
        "Cost Efficiency": round(efficiency, 2)
    }


short_names = ["True Baseline", "Group+XGB", "Group+CLV-XGB"]
preds = [y_pred_A, y_pred_B, y_pred_C]

clv_results = []
for sname, yp in zip(short_names, preds):
    clv_results.append(compute_clv_proxy_metrics(y_test_arr, yp, Vi_test, sname))

clv_df = pd.DataFrame(clv_results).set_index("Model").T

print("\n========== 客户价值代理指标对比 ==========")
print(clv_df.to_string())

# ============================
# 10. 图 3: CLV 代理指标可视化 (三模型)
# ============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 子图 1: CLV-Recall + Avg TP CLV (归一化)
ax1 = axes[0]
clv_recall_vals = [r["CLV-Recall"] for r in clv_results]
avg_tp_vals = [r["Avg TP CLV"] / 10.0 for r in clv_results]
x_pos = np.arange(3)
width = 0.35
bars1 = ax1.bar(x_pos - width / 2, clv_recall_vals, width, label='CLV-Recall', color='#5DADE2')
bars2 = ax1.bar(x_pos + width / 2, avg_tp_vals, width, label='Avg TP CLV / 10', color='#E74C3C')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(short_names, fontsize=9)
ax1.set_title("CLV-Weighted Recall & Avg TP CLV", fontsize=12, weight='bold')
ax1.set_ylim(0, 1.05)
ax1.legend(loc='lower right', fontsize=8)
for bar in bars1:
    ax1.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                 ha='center', fontsize=8)
for bar in bars2:
    ax1.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                 ha='center', fontsize=8)

# 子图 2: 检测数量分布 (TP / FN / FP)
ax2 = axes[1]
count_data = pd.DataFrame({
    "Model": short_names,
    "TP (正确识别流失)": [r["TP Count"] for r in clv_results],
    "FN (漏检流失)": [r["FN Count"] for r in clv_results],
    "FP (误报)": [r["FP Count"] for r in clv_results]
})
count_data.set_index("Model").plot(kind='bar', ax=ax2, color=['#27AE60', '#E74C3C', '#F39C12'])
ax2.set_title("预测数量分布", fontsize=12, weight='bold')
ax2.legend(loc='upper right', fontsize=8)
ax2.tick_params(axis='x', labelsize=9)

# 子图 3: NMR 对比
ax3 = axes[2]
nmr_vals = [r["NMR (Net Monetary Return)"] for r in clv_results]
nmr_colors = ['#7FB3D8', '#5DADE2', '#E74C3C']
bars = ax3.bar(short_names, nmr_vals, color=nmr_colors, width=0.4)
ax3.set_title("Net Monetary Return (NMR)", fontsize=12, weight='bold')
ax3.tick_params(axis='x', labelsize=9)
for bar in bars:
    h = bar.get_height()
    ax3.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + max(0.5, abs(h) * 0.02)),
                 ha='center', fontsize=10)

plt.suptitle("客户价值代理指标对比 (CLV Proxy Metrics) — 三模型", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig("5.7_CLV_Proxy_Metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("图 3 已保存: 5.7_CLV_Proxy_Metrics.png")

# ============================
# 11. 汇总输出
# ============================
print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)
print(f"{'模型':<30} {'AUC':<8} {'F1':<8} {'Recall':<8} {'总损失':<10}")
print("-" * 60)
for i, name in enumerate(short_names):
    total_cost = cost_list[i].sum()
    print(f"{name:<30} {metrics['AUC'].iloc[i]:.4f}  {metrics['F1-Score'].iloc[i]:.4f}  "
          f"{metrics['Recall'].iloc[i]:.4f}  {total_cost:.1f}")

print("\n脚本 5.7 执行完毕，三张对比图表已保存。")

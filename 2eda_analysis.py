import os
import pandas as pd
import numpy as np

# 强制使用静态后端，避免 IDE 绘图弹窗冲突
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 环境设置与数据加载
# ==========================================
# 设置绘图风格与高分辨率
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# 定义输入文件路径
input_file = r"C:\Users\折纸\Desktop\研究生\研一下\机器学习\Data Cleaning\Cleaned_Churn_Modelling.csv"
print(f"[数据加载] 正在读取清洗后的数据文件：\n{input_file}")

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print("[错误] 未找到指定文件，请核对路径是否正确！")
    exit()

# 获取项目根目录，并在其下创建 EDA 文件夹
project_root = os.path.dirname(os.path.dirname(input_file))
output_dir = os.path.join(project_root, "EDA")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[环境设置] 已成功创建图表输出目录：{output_dir}")

# ==========================================
# 1. 还原可视化标签 (仅用于绘图，不改变原数据)
# ==========================================
# 为了让报告图表更易懂，将 0/1 和独热编码还原为文本
if 'Gender' in df.columns:
    df['Gender_Label'] = df['Gender'].map({0: 'Female', 1: 'Male'})

# 还原国家特征 (基于 drop_first=True 的独热编码逻辑：既不是德国也不是西班牙，就是法国)
def get_country(row):
    if row.get('Geography_Germany', 0) == 1:
        return 'Germany'
    elif row.get('Geography_Spain', 0) == 1:
        return 'Spain'
    else:
        return 'France'

if 'Geography_Germany' in df.columns and 'Geography_Spain' in df.columns:
    df['Country_Label'] = df.apply(get_country, axis=1)

# ==========================================
# 2. 单变量分析 (Univariate Analysis)
# ==========================================
print("[绘图进行中] 正在生成单变量分析图表...")

# 2.1 绘制 Age 与 CreditScore 的直方图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Age'], kde=True, ax=axes[0], color='#4C72B0', bins=30)
axes[0].set_title("Distribution of Age", fontsize=14)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")

sns.histplot(df['CreditScore'], kde=True, ax=axes[1], color='#55A868', bins=30)
axes[1].set_title("Distribution of Credit Score", fontsize=14)
axes[1].set_xlabel("Credit Score")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_Histogram_Age_CreditScore.png"))
plt.close()

# 2.2 绘制客户流失占比饼图
plt.figure(figsize=(6, 6))
exited_counts = df['Exited'].value_counts()
# 凸显流失客户那一块 (explode)
plt.pie(exited_counts, labels=['Retained (0)', 'Exited (1)'],
        autopct='%1.1f%%', startangle=90, colors=['#4C72B0', '#C44E52'],
        explode=(0, 0.1), shadow=True, textprops={'fontsize': 12})
plt.title("Proportion of Customer Churn", fontsize=15, weight='bold')
plt.savefig(os.path.join(output_dir, "02_PieChart_Churn_Proportion.png"))
plt.close()

# ==========================================
# 3. 双变量分析 (Bivariate Analysis)
# ==========================================
print("[绘图进行中] 正在生成双变量分析图表...")

# 3.1 分组柱状图：性别与国家的流失率差异
# barplot 默认计算 y 变量的平均值，因为 Exited 是 0/1，所以平均值正好等于"流失率"
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(x='Gender_Label', y='Exited', data=df, ax=axes[0], hue='Gender_Label', palette='pastel', errorbar=None, legend=False)
axes[0].set_title("Churn Rate by Gender", fontsize=14)
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Churn Rate")
# 在柱子上添加具体的百分比数值
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                     textcoords='offset points')

sns.barplot(x='Country_Label', y='Exited', data=df, ax=axes[1], hue='Country_Label', palette='pastel', errorbar=None, legend=False)
axes[1].set_title("Churn Rate by Geography", fontsize=14)
axes[1].set_xlabel("Country")
axes[1].set_ylabel("Churn Rate")
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 8),
                     textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_BarPlot_Categorical_ChurnRate.png"))
plt.close()

# 3.2 核密度估计图 (KDE Plot)：流失/未流失客户在年龄和余额上的分布差异
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 修改标签用于图例展示
df['Status'] = df['Exited'].map({0: 'Retained', 1: 'Exited'})

sns.kdeplot(data=df, x='Age', hue='Status', fill=True, common_norm=False,
            palette={'Retained': '#4C72B0', 'Exited': '#C44E52'}, alpha=0.5, ax=axes[0])
axes[0].set_title("KDE Plot: Age Distribution by Churn Status", fontsize=14)
axes[0].set_xlabel("Age")

sns.kdeplot(data=df, x='Balance', hue='Status', fill=True, common_norm=False,
            palette={'Retained': '#4C72B0', 'Exited': '#C44E52'}, alpha=0.5, ax=axes[1])
axes[1].set_title("KDE Plot: Balance Distribution by Churn Status", fontsize=14)
axes[1].set_xlabel("Balance")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_KDE_Age_Balance.png"))
plt.close()

print(f"[流程结束] EDA 可视化分析完成！所有图表已保存至：{output_dir}")
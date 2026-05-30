import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # 后台出图
# ==========================================
# 0. 全局样式与字体配置
# ==========================================
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 0. 初始化与环境设置
# ==========================================
# 创建输出目录
output_dir = "Data Cleaning"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"[环境设置] 已成功创建输出目录：{output_dir}")
else:
    print(f"[环境设置] 输出目录已存在：{output_dir}")

# 设置绘图风格，适用于正式报告
sns.set_theme(style="whitegrid")
# 注：为避免不同操作系统下中文字体缺失导致报错，图表标签暂采用标准英文，可根据需要调整
plt.rcParams['figure.dpi'] = 300  # 设置高分辨率以适应报告排版

# ==========================================
# 1. 数据加载与无关特征剔除
# ==========================================
file_path = "C:/Users/折纸/Desktop/研究生/研一下/数据挖掘/数据挖掘作业二—新/Churn_Modelling.csv"  # 请确保该文件与脚本在同一目录下
print(f"[数据加载] 正在读取文件：{file_path}")
df = pd.read_csv(file_path)
print(f"[数据审视] 原始数据集维度为：{df.shape[0]} 行, {df.shape[1]} 列。")

# 剔除行号、客户ID、姓氏等对建模无实际预测意义的标识符
columns_to_drop = ['CustomerId', 'Surname']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"[特征筛选] 已剔除无关特征：{columns_to_drop}。当前特征数量为：{df.shape[1]}。")

# ==========================================
# 2. 缺失值与异常值处理
# ==========================================
print("[数据校验] 正在执行缺失值检测...")
missing_counts = df.isnull().sum()
if missing_counts.sum() == 0:
    print("[数据校验] 经核查，数据集中不存在缺失值。")
else:
    print("[数据校验] 发现缺失值，处理策略需视具体情况而定（当前数据集默认无缺失值）。")

# 绘制清洗前的箱线图以观察异常值 (保存用于报告)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 第一组：单独绘制 Age (十位数级别)
sns.boxplot(x=df['Age'], ax=axes[0], color='skyblue')
axes[0].set_title("Distribution of Age", fontsize=12)
axes[0].set_xlabel("")  # 隐藏子图横坐标标签使排版更清爽

# 第二组：单独绘制 CreditScore (百位数级别)
sns.boxplot(x=df['CreditScore'], ax=axes[1], color='lightgreen')
axes[1].set_title("Distribution of Credit Score", fontsize=12)
axes[1].set_xlabel("")

# 第三组：共同绘制 Balance 和 EstimatedSalary (十万数级别)
sns.boxplot(data=df[['Balance', 'EstimatedSalary']], orient="h", ax=axes[2], palette="Set2")
axes[2].set_title("Distribution of Balance and Estimated Salary", fontsize=12)
axes[2].set_xlabel("Value", fontsize=10)

# 设置总标题并优化布局
plt.suptitle("Distribution of Numerical Features (Before Outlier Treatment)", fontsize=16)
plt.tight_layout()

# 保存图表至输出目录
plt.savefig(os.path.join(output_dir, "01_Boxplot_Before_Treatment.png"))
plt.close()
print("[可视化输出] 优化后的异常值检测箱线图（分组子图）已保存至目录。")

# 异常值处理：采用截断法（Winsorization）处理 Age 和 CreditScore 的极端离群点
# 将数据限制在 1% 到 99% 的分位数之间，避免直接删除导致样本流失
print("[数据修正] 正在对 'Age' 与 'CreditScore' 执行极值截断处理 (1st-99th Percentile)...")
for col in ['Age', 'CreditScore']:
    lower_limit = df[col].quantile(0.01)
    upper_limit = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower_limit, upper_limit)

# ==========================================
# 3. 类别变量编码 (Categorical Encoding)
# ==========================================
print("[变量编码] 正在执行类别变量的数字化转换...")
# 性别特征 (Gender)：二元类别，使用映射转换为 0 和 1
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# 地域特征 (Geography)：多元无序类别，使用独热编码 (One-Hot Encoding)
# 参数 drop_first=True 用于规避虚拟变量陷阱（多重共线性）
df = pd.get_dummies(df, columns=['Geography'], drop_first=True, dtype=int)
print("[变量编码] 编码操作完成。当前数据集已完全转换为数值形态。")


# ==========================================
# 4. 清洗效果展示与结果导出
# ==========================================
# 绘制清洗后的相关性热力图 (保存用于报告)
plt.figure(figsize=(12, 10))
# 计算皮尔逊相关系数矩阵
corr_matrix = df.corr()
# 使用 Seaborn 绘制热力图
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=.5)
plt.title("Correlation Matrix of Cleaned Features", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_Correlation_Heatmap_Cleaned.png"))
plt.close()
print("[可视化输出] 清洗后特征相关性热力图已保存至目录。")

# 导出最终的清洗数据集
cleaned_file_path = os.path.join(output_dir, "Cleaned_Churn_Modelling.csv")
df.to_csv(cleaned_file_path, index=False)
print(f"[流程结束] 数据预处理工作全部完成。清洗后的高质量数据集已保存至：{cleaned_file_path}")

# 打印处理后的数据概况作为最终确认
print("\n" + "="*40)
print("清洗后数据前三行概览 (Data Head):")
print("="*40)
print(df.head(3))
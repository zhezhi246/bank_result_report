import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. 加载数据
input_file = r"C:\Users\折纸\Desktop\研究生\研一下\数据挖掘\数据挖掘作业二—新\Data Cleaning\Cleaned_Churn_Modelling.csv"
df = pd.read_csv(input_file)

def get_country(row):
    if row.get('Geography_Germany', 0) == 1: return 'Germany'
    elif row.get('Geography_Spain', 0) == 1: return 'Spain'
    else: return 'France'
df['Country'] = df.apply(get_country, axis=1)

X = df.drop(columns=['Exited', 'Gender_Label', 'Country_Label', 'Status', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'RowNumber', 'CustomerId', 'Surname'], errors='ignore')
y = df['Exited']

# 2. 计算并保存国家均值字典 (用于计算 Group_Stat)
country_mean_dict = df.groupby('Country')['Balance'].mean().to_dict()
joblib.dump(country_mean_dict, 'country_mean_dict.pkl')

# 3. 构造 XGBoost 训练特征
X_xgb = X.copy()
X_xgb['Group_Stat'] = df['Balance'] - df['Country'].map(country_mean_dict)
X_xgb = X_xgb.drop(columns=['Country'])

# 训练 XGBoost
pos_weight = (len(y) - sum(y)) / sum(y)
model_xgb = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42)
model_xgb.fit(X_xgb, y)
model_xgb.save_model('xgboost_churn_model.json')

# 4. 训练 K-Means 与保存 Scaler
cluster_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'IsActiveMember']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')

print("✅ 模型与预处理文件导出成功！生成了 4 个文件。")
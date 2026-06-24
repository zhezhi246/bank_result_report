import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

C_COST = 0.1
GAMMA = 0.3

# 1. 加载数据
input_file = r"C:\Users\折纸\Desktop\研究生\研一下\机器学习\Data Cleaning\Cleaned_Churn_Modelling.csv"
df = pd.read_csv(input_file)

def get_country(row):
    if row.get('Geography_Germany', 0) == 1: return 'Germany'
    elif row.get('Geography_Spain', 0) == 1: return 'Spain'
    else: return 'France'
df['Country'] = df.apply(get_country, axis=1)

X = df.drop(columns=['Exited', 'Gender_Label', 'Country_Label', 'Status',
                     'Geography_France', 'Geography_Germany', 'Geography_Spain',
                     'RowNumber', 'CustomerId', 'Surname'], errors='ignore')
y = df['Exited']

# 2. 群体统计特征
country_mean_dict = df.groupby('Country')['Balance'].mean().to_dict()
joblib.dump(country_mean_dict, 'country_mean_dict.pkl')

X_xgb = X.copy()
X_xgb['Group_Stat'] = df['Balance'] - df['Country'].map(country_mean_dict)
X_xgb = X_xgb.drop(columns=['Country'])

# 3. 计算客户价值代理指标 Vi
scaler_bal = MinMaxScaler().fit(df[['Balance']])
scaler_sal = MinMaxScaler().fit(df[['EstimatedSalary']])

bal_norm = scaler_bal.transform(df[['Balance']]).flatten()
sal_norm = scaler_sal.transform(df[['EstimatedSalary']]).flatten()
is_active = df['IsActiveMember'].values
vi_raw = 0.6 * bal_norm + 0.4 * sal_norm * (1 + 0.2 * is_active)

vi_raw_min = vi_raw.min()
vi_raw_max = vi_raw.max()
Vi = 1.0 + 9.0 * (vi_raw - vi_raw_min) / (vi_raw_max - vi_raw_min + 1e-5)

joblib.dump(scaler_bal, 'scaler_bal.pkl')
joblib.dump(scaler_sal, 'scaler_sal.pkl')
joblib.dump({'vi_raw_min': vi_raw_min, 'vi_raw_max': vi_raw_max,
             'C_COST': C_COST, 'GAMMA': GAMMA}, 'vi_params.pkl')

# 4. 训练 CLV-Modified XGBoost（样本权重 + scale_pos_weight）
pos_weight = (len(y) - sum(y)) / sum(y)
clv_sample_weight = np.ones(len(y))
clv_sample_weight[y == 1] = Vi[y == 1]

model_xgb = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
model_xgb.fit(X_xgb, y, sample_weight=clv_sample_weight)
model_xgb.save_model('xgboost_churn_model.json')

# 5. 训练 K-Means 与保存 Scaler
cluster_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'IsActiveMember']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')

print("✅ 模型与预处理文件导出成功！")
print(f"   - xgboost_churn_model.json  (CLV-Modified XGBoost)")
print(f"   - country_mean_dict.pkl")
print(f"   - scaler_bal.pkl / scaler_sal.pkl / vi_params.pkl  (CLV 计算组件)")
print(f"   - scaler.pkl / kmeans_model.pkl  (客户分群)")
print(f"   Vi 范围: [{Vi.min():.2f}, {Vi.max():.2f}]")

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# ==========================================
# 0. 页面配置与缓存模型加载
# ==========================================
st.set_page_config(page_title="银行客户智能预警系统", page_icon="🏦", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_assets():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(BASE_DIR, 'xgboost_churn_model.json'))
    kmeans = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    country_dict = joblib.load(os.path.join(BASE_DIR, 'country_mean_dict.pkl'))
    scaler_bal = joblib.load(os.path.join(BASE_DIR, 'scaler_bal.pkl'))
    scaler_sal = joblib.load(os.path.join(BASE_DIR, 'scaler_sal.pkl'))
    vi_params = joblib.load(os.path.join(BASE_DIR, 'vi_params.pkl'))
    return xgb_model, kmeans, scaler, country_dict, scaler_bal, scaler_sal, vi_params


xgb_model, kmeans, scaler, country_dict, scaler_bal, scaler_sal, vi_params = load_assets()

C_COST = vi_params['C_COST']
GAMMA = vi_params['GAMMA']

CLUSTER_LABELS = {
    0: "优质活跃主力军 (High-Value Active)",
    1: "沉睡的高净值客群 (Wealthy Sleeping Dog)",
    2: "低频单产品青年 (Low-Balance Youth)",
    3: "高危敏感型客户 (High-Risk Sensitive)"
}

# ==========================================
# 1. 侧边栏：客户信息输入区
# ==========================================
st.sidebar.header("📝 客户信息录入")
st.sidebar.markdown("请调整下方参数，系统将实时更新预测结果。")

country = st.sidebar.selectbox("国家 (Geography)", ['France', 'Germany', 'Spain'])
gender = st.sidebar.radio("性别 (Gender)", ['Male', 'Female'])
age = st.sidebar.slider("年龄 (Age)", 18, 92, 40)
credit_score = st.sidebar.slider("信用评分 (Credit Score)", 300, 850, 650)
balance = st.sidebar.number_input("账户余额 (Balance)", value=50000.0, step=1000.0)
salary = st.sidebar.number_input("预估年薪 (Estimated Salary)", value=60000.0, step=1000.0)
tenure = st.sidebar.slider("网龄/年 (Tenure)", 0, 10, 5)
num_products = st.sidebar.slider("持有产品数 (Num Of Products)", 1, 4, 1)
has_crcard = st.sidebar.checkbox("拥有信用卡 (Has Credit Card)", value=True)
is_active = st.sidebar.checkbox("活跃会员 (Is Active Member)", value=True)

gender_val = 1 if gender == 'Male' else 0
card_val = 1 if has_crcard else 0
active_val = 1 if is_active else 0

# ==========================================
# 2. 后台推理计算
# ==========================================
# 2a. 群体统计特征
group_stat = balance - country_dict.get(country, 0)

# 2b. XGBoost 输入
xgb_features = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_val],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [card_val],
    'IsActiveMember': [active_val],
    'EstimatedSalary': [salary],
    'Group_Stat': [group_stat]
})

# 2c. 客户价值代理指标 Vi 与个性化阈值
bal_norm = scaler_bal.transform([[balance]])[0][0]
sal_norm = scaler_sal.transform([[salary]])[0][0]
vi_raw = 0.6 * bal_norm + 0.4 * sal_norm * (1 + 0.2 * active_val)
Vi = 1.0 + 9.0 * (vi_raw - vi_params['vi_raw_min']) / (vi_params['vi_raw_max'] - vi_params['vi_raw_min'] + 1e-5)
theta = np.clip(C_COST / (GAMMA * Vi), 0.01, 0.99)

# 2d. KMeans 输入
cluster_features = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'EstimatedSalary': [salary],
    'IsActiveMember': [active_val]
})

# 2e. 预测
churn_prob = xgb_model.predict_proba(xgb_features)[0][1]
scaled_features = scaler.transform(cluster_features)
cluster_id = kmeans.predict(scaled_features)[0]
persona = CLUSTER_LABELS[cluster_id]

# ==========================================
# 3. 主界面 UI 展示
# ==========================================
st.title("🏦 银行客户流失预警与策略系统（CLV 动态阈值）")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 流失风险评估")
    st.metric(label="流失概率 (Churn Probability)", value=f"{churn_prob * 100:.2f}%")

    # 基于个性化动态阈值的风险判定
    if churn_prob > theta:
        st.error(f"🚨 高危预警：流失概率超过个性化阈值 ({theta*100:.1f}%)，建议立即干预！")
    elif churn_prob > theta * 0.7:
        st.warning(f"⚠️ 中度风险：流失概率接近个性化阈值 ({theta*100:.1f}%)，建议关注。")
    else:
        st.success(f"✅ 安全状态：流失概率远低于个性化阈值 ({theta*100:.1f}%)，客户留存意愿良好。")

    st.progress(float(churn_prob))

with col2:
    st.subheader("🪞 客户画像分群")
    st.info(f"**归属客群：** \n### {persona}")

    st.markdown("**数据指标：**")
    st.write(f"- 客户价值代理指标 (Vi): **{Vi:.2f}** / 10.0")
    st.write(f"- 个性化预警阈值: **{theta*100:.1f}%**")
    st.write(f"- 与所在国平均余额差值 (Group_Stat): **€ {group_stat:,.2f}**")
    st.write(f"- 活跃状态: {'活跃' if is_active else '沉睡'}")

st.markdown("---")

st.subheader("💡 AI 智能营销策略建议")

if churn_prob > theta:
    if "高净值" in persona or balance > 100000:
        st.error("🎯 **VIP 挽回策略 (高风险 + 高价值):**\n\n"
                 f"该客户价值评分属于核心资产人群。"
                 "建议指派专属客户经理进行 1V1 电话回访，提供高收益理财产品体验券，增加退出壁垒。")
    elif "单产品" in persona or num_products == 1:
        st.warning("🎯 **交叉营销策略 (高风险 + 单一产品):**\n\n"
                   "流失原因大概率是产品粘性不足。建议立即推送包含'信用卡免年费+小额消费贷'的组合产品短信，不要过度打扰。")
    else:
        st.warning("🎯 **标准化挽留策略:**\n\n"
                   "向该客户邮箱发送自动化温情关怀邮件，附带常规积分兑换活动。")
else:
    if "沉睡" in persona:
        st.info("🛡️ **唤醒策略 (安全 + 沉睡):**\n\n"
                "客户暂无流失风险，但活跃度低。建议在节假日推送 App 登录抽奖活动，提升促活。")
    else:
        st.success("🛡️ **维护策略 (安全 + 活跃):**\n\n"
                   "客户忠诚度高，是银行的利润基石。适宜进行深度交叉销售（如推销高净值保险或基金）。")

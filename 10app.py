import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# ==========================================
# 0. 页面配置与缓存模型加载
# ==========================================
st.set_page_config(page_title="银行客户智能预警系统", page_icon="🏦", layout="wide")


@st.cache_resource
def load_assets():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(r'xgboost_churn_model.json')
    kmeans = joblib.load(r'kmeans_model.pkl')
    scaler = joblib.load(r'scaler.pkl')
    country_dict = joblib.load(r'country_mean_dict.pkl')
    return xgb_model, kmeans, scaler, country_dict


xgb_model, kmeans, scaler, country_dict = load_assets()

# 定义客群业务标签 (你可以根据之前算出的聚类均值修改这些名字)
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

# 转换输入格式
gender_val = 1 if gender == 'Male' else 0
card_val = 1 if has_crcard else 0
active_val = 1 if is_active else 0

# ==========================================
# 2. 后台推理计算
# ==========================================
# 计算群体统计特征 (Group_Stat)
group_stat = balance - country_dict.get(country, 0)

# 构建 XGBoost 输入
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

# 构建 KMeans 输入
cluster_features = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'EstimatedSalary': [salary],
    'IsActiveMember': [active_val]
})

# 进行预测
churn_prob = xgb_model.predict_proba(xgb_features)[0][1]
scaled_features = scaler.transform(cluster_features)
cluster_id = kmeans.predict(scaled_features)[0]
persona = CLUSTER_LABELS[cluster_id]

# ==========================================
# 3. 主界面 UI 展示
# ==========================================
st.title("🏦 银行客户流失预警与策略系统")
st.markdown("---")

col1, col2 = st.columns(2)

# 面板 1：预测结果
with col1:
    st.subheader("📊 流失风险评估")
    st.metric(label="流失概率 (Churn Probability)", value=f"{churn_prob * 100:.2f}%")

    if churn_prob > 0.6:
        st.error("🚨 高危预警：该客户流失风险极高，需立即干预！")
    elif churn_prob > 0.4:
        st.warning("⚠️ 中度风险：建议关注该客户近期动向。")
    else:
        st.success("✅ 安全状态：客户留存意愿良好。")

    st.progress(float(churn_prob))

# 面板 2：客户画像
with col2:
    st.subheader("🪞 客户画像分群")
    st.info(f"**归属客群：** \n### {persona}")

    # 根据客群给出不同的话术展示
    st.markdown("**数据指标：**")
    st.write(f"- 与所在国平均余额差值 (Group_Stat): **€ {group_stat:,.2f}**")
    st.write(f"- 活跃状态: {'活跃' if is_active else '沉睡'}")
    st.write(f"- 资产负债情况: {num_products} 款产品")

st.markdown("---")

# 面板 3：智能业务策略 (流失率 + 画像的联立闭环)
st.subheader("💡 AI 智能营销策略建议")

if churn_prob > 0.5:
    if "高净值" in persona or balance > 100000:
        st.error("🎯 **VIP 挽回策略 (高风险 + 高价值):**\n系统识别该客户为核心资产人群。建议指派专属客户经理进行 1V1 电话回访，提供高收益理财产品体验券，增加退出壁垒。")
    elif "单产品" in persona or num_products == 1:
        st.warning("🎯 **交叉营销策略 (高风险 + 单一产品):**\n流失原因大概率是产品粘性不足。建议立即推送包含‘信用卡免年费+小额消费贷’的组合产品短信，不要过度打扰。")
    else:
        st.warning("🎯 **标准化挽留策略:**\n向该客户邮箱发送自动化温情关怀邮件，附带常规积分兑换活动。")
else:
    if "沉睡" in persona:
        st.info("🛡️ **唤醒策略 (安全 + 沉睡):**\n客户暂无流失风险，但活跃度低。建议在节假日推送 App 登录抽奖活动，提升促活。")
    else:
        st.success("🛡️ **维护策略 (安全 + 活跃):**\n客户忠诚度高，是银行的利润基石。适宜进行深度交叉销售（如推销高净值保险或基金）。")

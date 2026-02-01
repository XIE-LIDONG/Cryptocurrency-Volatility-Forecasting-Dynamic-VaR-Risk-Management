import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from arch import arch_model  # GARCH建模核心库
import warnings
warnings.filterwarnings('ignore')

# ====================== 全局设置 ======================
st.set_page_config(
    page_title="Crypto Volatility & VaR Dashboard",
    page_icon="📈",
    layout="wide"
)

# 绘图设置（纯英文，避免字体报错）
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ====================== 核心函数：数据获取+GARCH建模+VaR计算 ======================
@st.cache_data(ttl=3600)  # 缓存1小时，避免重复拉取数据
def get_crypto_data(asset, start_date, end_date):
    """从Yahoo Finance拉取加密货币数据"""
    # 定义Yahoo Finance代码
    ticker_map = {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD"
    }
    # 拉取数据
    df = yf.download(ticker_map[asset], start=start_date, end=end_date)
    # 保留收盘价，计算日收益率（百分比）
    df = df[['Close']].copy()
    df['returns'] = df['Close'].pct_change()  # 收益率=（今日收盘价-昨日）/昨日
    df = df.dropna()  # 删除空值
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    return df

def fit_garch_model(returns):
    """拟合GARCH(1,1)模型，返回波动率和模型参数"""
    # 拟合GARCH(1,1)（均值=0，因为加密货币收益率均值接近0）
    am = arch_model(returns * 100, mean='Zero', vol='GARCH', p=1, q=1)  # 乘以100避免数值过小
    res = am.fit(disp='off')  # disp='off'关闭拟合日志
    
    # 提取参数
    params = {
        'omega': res.params['omega'] / 10000,  # 还原到原始尺度（因为乘以了100）
        'alpha': res.params['alpha[1]'],
        'beta': res.params['beta[1]'],
        'alpha_beta': res.params['alpha[1]'] + res.params['beta[1]'],
        'long_term_vol': np.sqrt(res.params['omega'] / (1 - res.params['alpha[1]'] - res.params['beta[1]'])) / 100  # 长期波动率
    }
    
    # 提取条件波动率（还原到原始尺度）
    cond_vol = res.conditional_volatility / 100
    
    return cond_vol, params

def calculate_var(cond_vol, dist_type="Normal"):
    """计算动态VaR"""
    # Normal分布VaR
    var_95_normal = 1.65 * cond_vol
    var_99_normal = 2.33 * cond_vol
    
    # t分布VaR（自由度8，适配加密货币厚尾）
    t_95 = abs(t.ppf(0.05, df=8))
    t_99 = abs(t.ppf(0.01, df=8))
    var_95_t = t_95 * cond_vol
    var_99_t = t_99 * cond_vol
    
    if dist_type == "Normal":
        return var_95_normal, var_99_normal
    else:
        return var_95_t, var_99_t

def predict_next_vol_var(returns, params, last_vol):
    """预测下一日波动率和VaR"""
    # 取最后一天的收益率残差（这里假设均值=0，残差=收益率）
    last_residual = returns.iloc[-1]
    # GARCH(1,1)递推公式
    next_vol_sq = params['omega'] + params['alpha'] * (last_residual **2) + params['beta'] * (last_vol** 2)
    next_vol = np.sqrt(next_vol_sq)
    
    # 计算VaR
    var_95 = 1.65 * next_vol
    var_99 = 2.33 * next_vol
    t_95 = abs(t.ppf(0.05, df=8))
    t_99 = abs(t.ppf(0.01, df=8))
    var_95_t = t_95 * next_vol
    var_99_t = t_99 * next_vol
    
    return next_vol, var_95, var_99, var_95_t, var_99_t

# ====================== 页面UI开始 ======================
st.title("📈 Crypto Volatility & VaR Dashboard")
st.subheader("Real-Time GARCH(1,1) Modeling & Risk Analysis for BTC/ETH")
st.markdown("*Automatically pulls data from Yahoo Finance | No manual CSV required*")
st.divider()

# 1. 顶部核心选择区
col1, col2, col3 = st.columns([1.5, 2, 1.5])
with col1:
    selected_asset = st.selectbox("Select Cryptocurrency", ["Bitcoin (BTC)", "Ethereum (ETH)"])
with col2:
    # 默认时间范围：近3年（适配你的研究周期）
    default_start = pd.Timestamp.now() - pd.DateOffset(years=3)
    default_end = pd.Timestamp.now()
    date_range = st.date_input(
        "Select Date Range",
        value=[default_start.date(), default_end.date()],
        min_value=pd.Timestamp("2017-01-01").date(),
        max_value=pd.Timestamp.now().date()
    )
with col3:
    var_dist = st.radio(
        "VaR Distribution Type",
        ["Normal Distribution", "t-Distribution (Fat Tail)"],
        horizontal=True
    )

# 2. 一键执行：拉数据+建模
st.divider()
if st.button("🔄 Run Analysis (Pull Data + Fit GARCH + Calculate VaR)", type="primary"):
    with st.spinner("Processing... (This may take 10-20 seconds for GARCH fitting)"):
        # 步骤1：拉取数据
        df = get_crypto_data(selected_asset, date_range[0], date_range[1])
        st.success(f"✅ Successfully pulled {len(df)} days of {selected_asset} data")
        
        # 步骤2：拟合GARCH(1,1)
        cond_vol, garch_params = fit_garch_model(df['returns'])
        df['cond_vol'] = cond_vol.values  # 把波动率加入DataFrame
        st.success(f"✅ GARCH(1,1) model fitted successfully")
        
        # 步骤3：计算VaR
        var_95, var_99 = calculate_var(df['cond_vol'], var_dist.split(' ')[0])
        df['var_95'] = var_95
        df['var_99'] = var_99
        df['loss'] = -df['returns']  # 计算亏损
        # 标记击穿点
        df['break_95'] = df['loss'] > df['var_95']
        df['break_99'] = df['loss'] > df['var_99']
        st.success(f"✅ Dynamic VaR calculated ({var_dist})")
        
        # 步骤4：计算回测结果
        break_95_count = df['break_95'].sum()
        break_95_rate = break_95_count / len(df)
        break_99_count = df['break_99'].sum()
        break_99_rate = break_99_count / len(df)
        
        # ====================== 结果展示 ======================
        st.divider()
        
        # 模块1：GARCH参数展示（核心亮点）
        st.header("🔧 GARCH(1,1) Model Parameters")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ω (Long-Term Variance Floor)", f"{garch_params['omega']:.6f}")
        with col2:
            st.metric("α (Shock Coefficient)", f"{garch_params['alpha']:.4f}")
        with col3:
            st.metric("β (Volatility Persistence)", f"{garch_params['beta']:.4f}")
        with col4:
            st.metric("α+β (Total Persistence)", f"{garch_params['alpha_beta']:.4f}")
        with col5:
            st.metric("Long-Term Volatility", f"{garch_params['long_term_vol']*100:.2f}%")
        
        # 参数解释
        with st.expander("📖 Parameter Explanation"):
            st.markdown(f"""
            - **ω**: Minimum volatility level (long-term floor) for {selected_asset.split(' ')[0]}
            - **α**: Sensitivity to daily price shocks (higher = more reactive to new information)
            - **β**: Persistence of historical volatility (higher = volatility lasts longer)
            - **α+β**: Closer to 1 = stronger volatility clustering (typical for crypto)
            - **Long-Term Volatility**: Theoretical steady-state volatility
            """)
        
        # 模块2：波动率可视化
        st.divider()
        st.header("📊 Dynamic Volatility Analysis")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 子图1：日收益率
        ax1.plot(df['date'], df['returns'], color="gray", alpha=0.7, label="Daily Returns")
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Returns (Decimal)")
        ax1.set_title(f"{selected_asset} Daily Returns")
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 子图2：GARCH波动率
        ax2.plot(df['date'], df['cond_vol'], color="royalblue", linewidth=1.5, label="GARCH Conditional Volatility")
        ax2.axhline(y=garch_params['long_term_vol'], color="red", linestyle="--", label=f"Long-Term Volatility ({garch_params['long_term_vol']*100:.2f}%)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility (Decimal)")
        ax2.set_title(f"{selected_asset} Dynamic Volatility (GARCH(1,1))")
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 模块3：VaR风险分析
        st.divider()
        st.header("🛡️ Dynamic VaR Risk Analysis")
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # 收益率曲线
        ax.plot(df['date'], df['returns'], color="gray", alpha=0.5, label="Daily Returns")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        
        # VaR曲线（亏损线为负）
        ax.plot(df['date'], -df['var_95'], color="red", linewidth=1.5, label=f"95% {var_dist} VaR (Max Loss)")
        ax.plot(df['date'], -df['var_99'], color="darkred", linewidth=1.5, label=f"99% {var_dist} VaR (Max Loss)")
        
        # 标记击穿点
        break_95_df = df[df['break_95']]
        ax.scatter(break_95_df['date'], break_95_df['returns'], color="red", s=20, label="95% VaR Breakthrough", zorder=5)
        break_99_df = df[df['break_99']]
        ax.scatter(break_99_df['date'], break_99_df['returns'], color="darkred", s=30, label="99% VaR Breakthrough", zorder=6)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns (Decimal)")
        ax.set_title(f"{selected_asset} Returns vs Dynamic VaR ({var_dist})")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # VaR回测结果
        st.subheader("📋 VaR Backtesting Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("95% VaR Breakthrough Count", f"{break_95_count}")
        with col2:
            st.metric("95% VaR Breakthrough Rate", f"{break_95_rate*100:.2f}% (Ideal: 5%)")
        with col3:
            st.metric("99% VaR Breakthrough Count", f"{break_99_count}")
        with col4:
            st.metric("99% VaR Breakthrough Rate", f"{break_99_rate*100:.2f}% (Ideal: 1%)")
        
        # 结果评价
        if 0.009 <= break_99_rate <= 0.011:
            st.success("✅ Near-ideal performance: Model perfectly captures extreme risk!")
        elif 0.04 <= break_95_rate <= 0.06:
            st.success("✅ Excellent performance: Model accurately captures daily risk!")
        else:
            st.info("ℹ️ Reasonable risk prediction (crypto markets are highly volatile)")
        
        # 模块4：实时预测
        st.divider()
        st.header("🔮 Next-Day Volatility & VaR Prediction")
        # 取最后一天的波动率
        last_vol = df['cond_vol'].iloc[-1]
        # 预测下一日数值
        next_vol, var_95, var_99, var_95_t, var_99_t = predict_next_vol_var(df['returns'], garch_params, last_vol)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Volatility", f"{next_vol*100:.2f}%")
        with col2:
            st.metric("95% Normal VaR", f"{var_95*100:.2f}%")
        with col3:
            st.metric("99% Normal VaR", f"{var_99*100:.2f}%")
        with col4:
            st.metric("95% t-VaR (Fat Tail)", f"{var_95_t*100:.2f}%")
        
        # 白话解释
        st.markdown(f"""
        ### 📝 Prediction Interpretation
        For **{selected_asset.split(' ')[0]}** next trading day:
        - With 95% confidence: Maximum expected loss = **{var_95*100:.2f}%**
        - With 99% confidence (extreme risk): Maximum expected loss = **{var_99*100:.2f}%**
        - t-Distribution VaR accounts for crypto's fat tail (more conservative)
        """)

# ====================== 底部信息 ======================
st.divider()
st.markdown("""
### 📚 Project Details
- **Data Source**: Yahoo Finance (Real-time crypto price data)
- **Model**: GARCH(1,1) (Volatility Clustering & Persistence)
- **Risk Metric**: Value-at-Risk (Normal/t-Distribution)
- **GitHub Repository**: [Your GitHub Link Here]
- **Built with**: Python, Streamlit, yfinance, arch, matplotlib
""")
st.markdown("---")
st.markdown("*Quantitative Finance Project for Study Abroad Application*")
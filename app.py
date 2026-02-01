import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from arch import arch_model
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ====================== 全局设置 ======================
st.set_page_config(
    page_title="Crypto Volatility & VaR Dashboard",
    page_icon="📈",
    layout="wide"
)

# 绘图设置
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 初始化session state（共享数据，避免重复计算）
if 'df' not in st.session_state:
    st.session_state.df = None
if 'garch_params' not in st.session_state:
    st.session_state.garch_params = None
if 'var_dist' not in st.session_state:
    st.session_state.var_dist = None
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = "Bitcoin (BTC)"
if 'var_95' not in st.session_state:
    st.session_state.var_95 = None
if 'var_99' not in st.session_state:
    st.session_state.var_99 = None
if 'cond_vol' not in st.session_state:
    st.session_state.cond_vol = None

# ====================== 核心函数 ======================
@st.cache_data(ttl=3600)
def get_crypto_data(asset, start_date, end_date):
    """从Yahoo Finance拉取加密货币数据（支持BTC/ETH）"""
    ticker_map = {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD"
    }
    df = yf.download(ticker_map[asset], start=start_date, end=end_date)
    # 保留核心列并处理
    df = df[['Close']].copy()
    df['returns'] = df['Close'].pct_change()  # 简单收益率
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))  # 对数收益率
    df['simple_vol'] = df['returns'].rolling(window=21).std()  # 21天滚动波动率
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    return df

def fit_garch_model(returns):
    """拟合GARCH(1,1)模型，返回波动率和参数"""
    am = arch_model(returns * 100, mean='Zero', vol='GARCH', p=1, q=1)
    res = am.fit(disp='off')
    
    params = {
        'omega': res.params['omega'] / 10000,
        'alpha': res.params['alpha[1]'],
        'beta': res.params['beta[1]'],
        'alpha_beta': res.params['alpha[1]'] + res.params['beta[1]'],
        'long_term_vol': np.sqrt(res.params['omega'] / (1 - res.params['alpha[1]'] - res.params['beta[1]'])) / 100
    }
    
    cond_vol = res.conditional_volatility / 100
    return cond_vol, params

def calculate_var(cond_vol, dist_type="Normal"):
    """计算动态VaR"""
    var_95_normal = 1.65 * cond_vol
    var_99_normal = 2.33 * cond_vol
    
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
    last_residual = returns.iloc[-1]
    next_vol_sq = params['omega'] + params['alpha'] * (last_residual **2) + params['beta'] * (last_vol** 2)
    next_vol = np.sqrt(next_vol_sq)
    
    var_95 = 1.65 * next_vol
    var_99 = 2.33 * next_vol
    t_95 = abs(t.ppf(0.05, df=8))
    t_99 = abs(t.ppf(0.01, df=8))
    var_95_t = t_95 * next_vol
    var_99_t = t_99 * next_vol
    
    return next_vol, var_95, var_99, var_95_t, var_99_t

def rolling_window_prediction(df, window_size):
    """滚动预测核心函数：按1/3数据设窗口，返回预测结果"""
    rolling_vol = []
    rolling_var_95 = []
    rolling_var_99 = []
    actual_vol = []
    actual_loss = []
    dates = []
    
    # 从window_size开始滚动
    for i in range(window_size, len(df)):
        # 仅用窗口内的真实数据训练
        train_returns = df['returns'].iloc[i-window_size:i]
        # 拟合GARCH
        am = arch_model(train_returns * 100, mean='Zero', vol='GARCH', p=1, q=1)
        res = am.fit(disp='off')
        # 提取参数和最后一天波动率
        params = {
            'omega': res.params['omega'] / 10000,
            'alpha': res.params['alpha[1]'],
            'beta': res.params['beta[1]']
        }
        last_vol = res.conditional_volatility.iloc[-1] / 100
        # 预测下一天波动率和VaR
        next_residual = train_returns.iloc[-1]
        next_vol_sq = params['omega'] + params['alpha'] * (next_residual **2) + params['beta'] * (last_vol** 2)
        next_vol = np.sqrt(next_vol_sq)
        var_95 = 1.65 * next_vol
        var_99 = 2.33 * next_vol
        
        # 存储结果（真实值+预测值）
        rolling_vol.append(next_vol)
        rolling_var_95.append(var_95)
        rolling_var_99.append(var_99)
        actual_vol.append(df['cond_vol'].iloc[i])  # 真实GARCH波动率
        actual_loss.append(-df['returns'].iloc[i])  # 真实亏损
        dates.append(df['date'].iloc[i])
    
    # 整理结果
    rolling_df = pd.DataFrame({
        'date': dates,
        'pred_vol': rolling_vol,
        'pred_var_95': rolling_var_95,
        'pred_var_99': rolling_var_99,
        'actual_vol': actual_vol,
        'actual_loss': actual_loss
    })
    return rolling_df

# ====================== 侧边导航栏 ======================
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio(
    "Select Function",
    ["🏠 Home", "📊 Data Visualization", "🧪 Model Validation", "🔮 Prediction"]
)

# ====================== 页面逻辑 ======================
# 1. 主页：核心选择区 + 数据加载
if page == "🏠 Home":
    st.title("📈 Crypto Volatility & VaR Dashboard")
    st.subheader("Real-Time GARCH(1,1) Modeling & Risk Analysis for BTC/ETH")
    st.divider()
    
    # 核心选择区
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col1:
        selected_asset = st.selectbox(
            "Select Cryptocurrency", 
            ["Bitcoin (BTC)", "Ethereum (ETH)"],
            index=["Bitcoin (BTC)", "Ethereum (ETH)"].index(st.session_state.selected_asset)
        )
        st.session_state.selected_asset = selected_asset
    with col2:
        # 时间范围：起始最早2017-01-01，结束默认当天
        min_start = pd.Timestamp("2017-01-01").date()
        max_end = pd.Timestamp.now().date()
        default_start = pd.Timestamp.now() - pd.DateOffset(years=3)
        date_range = st.date_input(
            "Select Date Range",
            value=[default_start.date(), max_end],
            min_value=min_start,
            max_value=max_end
        )
    with col3:
        var_dist = st.radio(
            "VaR Distribution Type",
            ["Normal Distribution", "t-Distribution (Fat Tail)"],
            horizontal=True
        )
        st.session_state.var_dist = var_dist
    
    # 一键运行按钮
    if st.button("🔄 Run Analysis (Pull Data + Fit GARCH + Calculate VaR)", type="primary"):
        with st.spinner("Processing... (This may take 10-20 seconds)"):
            # 拉取数据
            df = get_crypto_data(selected_asset, date_range[0], date_range[1])
            st.session_state.df = df
            st.success(f"✅ Successfully pulled {len(df)} days of {selected_asset} data")
            
            # 拟合GARCH模型
            cond_vol, garch_params = fit_garch_model(df['returns'])
            st.session_state.cond_vol = cond_vol
            st.session_state.garch_params = garch_params
            df['cond_vol'] = cond_vol.values
            st.success(f"✅ GARCH(1,1) model fitted successfully")
            
            # 计算VaR
            var_95, var_99 = calculate_var(cond_vol, var_dist.split(' ')[0])
            st.session_state.var_95 = var_95
            st.session_state.var_99 = var_99
            df['var_95'] = var_95
            df['var_99'] = var_99
            df['loss'] = -df['returns']
            df['break_95'] = df['loss'] > df['var_95']
            df['break_99'] = df['loss'] > df['var_99']
            st.session_state.df = df
            st.success(f"✅ Dynamic VaR calculated ({var_dist})")
            
            st.info("✅ All calculations completed! You can now navigate to other tabs to view results.")

# 2. 数据可视化页面
elif page == "📊 Data Visualization":
    st.title("📊 Data Visualization")
    st.divider()
    
    # 检查数据是否加载
    if st.session_state.df is None:
        st.warning("⚠️ Please run analysis first on the Home page!")
    else:
        df = st.session_state.df
        selected_asset = st.session_state.selected_asset
        
        # 绘制三张核心图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # 价格图
        ax1.plot(df['date'], df['Close'], color="darkblue", linewidth=1.2)
        ax1.set_ylabel("Closing Price (USD)")
        ax1.set_title(f"{selected_asset} Historical Price")
        ax1.grid(alpha=0.3)
        
        # 对数收益率图
        ax2.plot(df['date'], df['log_returns'], color="green", alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Log Returns (Decimal)")
        ax2.set_title(f"{selected_asset} Log Returns")
        ax2.grid(alpha=0.3)
        
        # 原始波动率图
        ax3.plot(df['date'], df['simple_vol'], color="orange", linewidth=1.2)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("21-Day Rolling Volatility (Decimal)")
        ax3.set_title(f"{selected_asset} Raw Volatility")
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# 3. 模型验证页面
elif page == "🧪 Model Validation":
    st.title("🧪 Model Validation")
    st.divider()
    
    # 检查数据是否加载
    if st.session_state.df is None:
        st.warning("⚠️ Please run analysis first on the Home page!")
    else:
        df = st.session_state.df
        selected_asset = st.session_state.selected_asset
        var_dist = st.session_state.var_dist
        var_95 = st.session_state.var_95
        var_99 = st.session_state.var_99
        
        # ========== Dynamic VaR Risk Analysis ==========
        st.subheader("🛡️ Dynamic VaR Risk Analysis")
        # 计算击穿率
        break_95_count = df['break_95'].sum()
        break_95_rate = break_95_count / len(df)
        break_99_count = df['break_99'].sum()
        break_99_rate = break_99_count / len(df)
        
        # 绘制VaR图
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df['date'], df['returns'], color="gray", alpha=0.5, label="Daily Returns")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.plot(df['date'], -df['var_95'], color="red", linewidth=1.5, label=f"95% {var_dist} VaR (Max Loss)")
        ax.plot(df['date'], -df['var_99'], color="darkred", linewidth=1.5, label=f"99% {var_dist} VaR (Max Loss)")
        
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("95% VaR Breakthrough Count", f"{break_95_count}")
        with col2:
            st.metric("95% VaR Breakthrough Rate", f"{break_95_rate*100:.2f}% ")
        with col3:
            st.metric("99% VaR Breakthrough Count", f"{break_99_count}")
        with col4:
            st.metric("99% VaR Breakthrough Rate", f"{break_99_rate*100:.2f}% ")
        
        # ========== 滚动预测 ==========
        st.divider()
        st.subheader("🎯 Rolling Window Prediction")
        # 自动计算窗口大小=数据长度的1/3（取整）
        window_size = int(len(df) / 3)
        st.info(f"🔍 Auto-set window size: {window_size} days (1/3 of total data: {len(df)} days)")
        
        with st.spinner("Running rolling prediction... (This may take 1-2 minutes)"):
            rolling_df = rolling_window_prediction(df, window_size)
            
            # 绘制滚动预测图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
            
            # 波动率对比
            ax1.plot(rolling_df['date'], rolling_df['pred_vol'], color="blue", linewidth=1.5, label="Predicted Volatility")
            ax1.plot(rolling_df['date'], rolling_df['actual_vol'], color="green", linewidth=1.5, alpha=0.7, label="Actual GARCH Volatility")
            start_pred_date = rolling_df['date'].iloc[0]
            ax1.axvline(x=start_pred_date, color="red", linestyle="--", label="Prediction Start Date")
            ax1.set_ylabel("Volatility (Decimal)")
            ax1.set_title(f"{selected_asset} Rolling Prediction: Volatility")
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # VaR对比
            ax2.plot(rolling_df['date'], rolling_df['pred_var_95'], color="red", linewidth=1.5, label="Predicted 95% VaR")
            ax2.plot(rolling_df['date'], rolling_df['pred_var_99'], color="darkred", linewidth=1.5, label="Predicted 99% VaR")
            ax2.plot(rolling_df['date'], rolling_df['actual_loss'], color="gray", alpha=0.7, label="Actual Loss")
            ax2.axvline(x=start_pred_date, color="red", linestyle="--")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Loss / VaR (Decimal)")
            ax2.set_title(f"{selected_asset} Rolling Prediction: VaR vs Actual Loss")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 滚动预测结果统计
            rolling_break_95 = (rolling_df['actual_loss'] > rolling_df['pred_var_95']).sum()
            rolling_break_95_rate = rolling_break_95 / len(rolling_df)
            rolling_break_99 = (rolling_df['actual_loss'] > rolling_df['pred_var_99']).sum()
            rolling_break_99_rate = rolling_break_99 / len(rolling_df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prediction Period Days", f"{len(rolling_df)}")
            with col2:
                st.metric("95% VaR Breakthrough Rate", f"{rolling_break_95_rate*100:.2f}% ")
            with col3:
                st.metric("99% VaR Breakthrough Count", f"{rolling_break_99}")
            with col4:
                st.metric("99% VaR Breakthrough Rate", f"{rolling_break_99_rate*100:.2f}% ")

# 4. 预测页面
elif page == "🔮 Prediction":
    st.title("🔮 Next-Day Prediction")
    st.divider()
    
    # 检查数据是否加载
    if st.session_state.df is None or st.session_state.garch_params is None:
        st.warning("⚠️ Please run analysis first on the Home page!")
    else:
        df = st.session_state.df
        selected_asset = st.session_state.selected_asset
        garch_params = st.session_state.garch_params
        
        # 计算下一个交易日
        last_date = df['date'].iloc[-1]
        next_date = last_date + timedelta(days=1)
        # 跳过周末（加密货币周末交易，保留逻辑兼容）
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")
        
        # 预测次日数据
        last_vol = df['cond_vol'].iloc[-1]
        next_vol, var_95, var_99, var_95_t, var_99_t = predict_next_vol_var(df['returns'], garch_params, last_vol)
        
        # 展示预测结果
        st.subheader(f"📅 Prediction for Next Trading Day: {next_date_str}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Volatility", f"{next_vol*100:.2f}%")
        with col2:
            st.metric("95% Normal VaR", f"{var_95*100:.2f}%")
        with col3:
            st.metric("99% Normal VaR", f"{var_99*100:.2f}%")
        with col4:
            st.metric("95% t-VaR (Fat Tail)", f"{var_95_t*100:.2f}%")
        
        # 预测解释
        st.markdown(f"""
        ### 📝 Prediction Interpretation
        For **{selected_asset.split(' ')[0]}** on {next_date_str}:
        - With 95% confidence: Maximum expected loss = **{var_95*100:.2f}%**
        - With 99% confidence (extreme risk): Maximum expected loss = **{var_99*100:.2f}%**
        - t-Distribution VaR accounts for crypto's fat tail (more conservative)
        """)




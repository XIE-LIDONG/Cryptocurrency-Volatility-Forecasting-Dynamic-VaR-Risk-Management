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

# 初始化所有session state（避免KeyError）
session_keys = [
    'df', 'garch_params', 'ewma_vol', 'ewma_var_95', 'ewma_var_99',
    'var_dist', 'selected_asset', 'var_95', 'var_99', 'cond_vol'
]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# ====================== 核心函数 ======================
@st.cache_data(ttl=3600)
def get_crypto_data(asset, start_date, end_date):
    """拉取加密货币数据"""
    ticker_map = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}
    df = yf.download(ticker_map[asset], start=start_date, end=end_date)
    if df.empty:
        st.error("❌ 未获取到数据，请检查日期范围！")
        return None
    
    # 核心数据处理
    df = df[['Close']].copy()
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['simple_vol'] = df['returns'].rolling(window=21).std()
    df = df.dropna()
    
    # 提前计算loss列（关键修复：避免后续切片丢失）
    df['loss'] = -df['returns']
    
    if len(df) < 50:
        st.error(f"❌ 有效数据仅{len(df)}天，至少需要50天！")
        return None
    
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    return df

def fit_garch_model(returns):
    """拟合GARCH模型"""
    try:
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
    except Exception as e:
        st.error(f"❌ GARCH拟合失败：{str(e)}")
        return None, None

def calculate_ewma_vol(returns, lambda_=0.94):
    """计算EWMA波动率"""
    if len(returns) < 21:
        st.error("❌ EWMA计算需要至少21天数据！")
        return pd.Series()
    
    initial_vol = returns.iloc[:21].std()
    vol_list = []
    for i in range(21, len(returns)):
        prev_vol_sq = initial_vol**2 if i == 21 else vol_list[-1]**2
        curr_return_sq = returns.iloc[i-1]**2
        ewma_vol_sq = lambda_ * prev_vol_sq + (1 - lambda_) * curr_return_sq
        vol_list.append(np.sqrt(ewma_vol_sq))
    
    ewma_vol = pd.Series(
        [np.nan]*21 + vol_list,
        index=returns.index[:len([np.nan]*21 + vol_list)]
    )
    return ewma_vol.dropna()

def calculate_var(cond_vol, dist_type="Normal"):
    """计算VaR"""
    if cond_vol is None or len(cond_vol) == 0:
        return None, None
    
    var_95_normal = 1.65 * cond_vol
    var_99_normal = 2.33 * cond_vol
    
    t_95 = abs(t.ppf(0.05, df=8))
    t_99 = abs(t.ppf(0.01, df=8))
    var_95_t = t_95 * cond_vol
    var_99_t = t_99 * cond_vol
    
    return (var_95_normal, var_99_normal) if dist_type == "Normal" else (var_95_t, var_99_t)

def predict_next_vol_var(returns, params, last_vol, model_type="GARCH"):
    """预测次日波动率和VaR"""
    if last_vol is None or pd.isna(last_vol):
        return None, None, None, None, None
    
    last_residual = returns.iloc[-1] if len(returns) > 0 else 0
    
    if model_type == "GARCH":
        if not params:
            return None, None, None, None, None
        next_vol_sq = params['omega'] + params['alpha'] * last_residual**2 + params['beta'] * last_vol**2
        next_vol = np.sqrt(next_vol_sq)
    else:  # EWMA
        next_vol_sq = 0.94 * last_vol**2 + 0.06 * last_residual**2
        next_vol = np.sqrt(next_vol_sq)
    
    var_95 = 1.65 * next_vol
    var_99 = 2.33 * next_vol
    t_95 = abs(t.ppf(0.05, df=8))
    t_99 = abs(t.ppf(0.01, df=8))
    var_95_t = t_95 * next_vol
    var_99_t = t_99 * next_vol
    
    return next_vol, var_95, var_99, var_95_t, var_99_t

def rolling_window_prediction(df, window_size, model_type="GARCH"):
    """滚动预测"""
    if window_size < 21:
        window_size = 21  # 最小窗口限制
    
    rolling_vol, rolling_var_95, rolling_var_99 = [], [], []
    actual_vol, actual_loss, dates = [], [], []
    
    for i in range(window_size, len(df)):
        train_returns = df['returns'].iloc[i-window_size:i]
        
        if model_type == "GARCH":
            am = arch_model(train_returns * 100, mean='Zero', vol='GARCH', p=1, q=1)
            res = am.fit(disp='off')
            params = {'omega': res.params['omega']/10000, 'alpha': res.params['alpha[1]'], 'beta': res.params['beta[1]']}
            last_vol = res.conditional_volatility.iloc[-1] / 100
            next_vol_sq = params['omega'] + params['alpha'] * train_returns.iloc[-1]**2 + params['beta'] * last_vol**2
            next_vol = np.sqrt(next_vol_sq)
        else:  # EWMA
            ewma_vol_train = calculate_ewma_vol(train_returns)
            last_vol = ewma_vol_train.iloc[-1] if len(ewma_vol_train) > 0 else train_returns.std()
            next_vol_sq = 0.94 * last_vol**2 + 0.06 * train_returns.iloc[-1]**2
            next_vol = np.sqrt(next_vol_sq)
        
        var_95 = 1.65 * next_vol
        var_99 = 2.33 * next_vol
        
        rolling_vol.append(next_vol)
        rolling_var_95.append(var_95)
        rolling_var_99.append(var_99)
        actual_vol.append(df['cond_vol'].iloc[i] if model_type == "GARCH" else df['simple_vol'].iloc[i])
        actual_loss.append(df['loss'].iloc[i])
        dates.append(df['date'].iloc[i])
    
    return pd.DataFrame({
        'date': dates, 'pred_vol': rolling_vol, 'pred_var_95': rolling_var_95,
        'pred_var_99': rolling_var_99, 'actual_vol': actual_vol, 'actual_loss': actual_loss
    })

# ====================== 导航栏 ======================
st.sidebar.title("📑 Navigation")
page = st.sidebar.radio(
    "Select Function",
    ["🏠 Home", "📊 Data Visualization", "🧪 GARCH Model Validation", 
     "📊 EWMA Model Validation", "🔍 Model Comparison", "🔮 Prediction"]
)

# ====================== 页面逻辑 ======================
# 1. Home页（核心修复：提前计算loss列，避免KeyError）
if page == "🏠 Home":
    st.markdown(
        """
        <div style='display: flex; justify-content: flex-end; align-items: center;'>
            <p style='color: #666666; font-size: 14px; margin: 0;'>By XIE LI DONG</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.title("📈 Crypto Volatility & VaR Dashboard")
    st.subheader("Real-Time GARCH/EWMA Modeling & Risk Analysis for BTC/ETH")
    st.divider()
    
    # 选择区
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col1:
        selected_asset = st.selectbox(
            "Select Cryptocurrency", ["Bitcoin (BTC)", "Ethereum (ETH)"],
            index=["Bitcoin (BTC)", "Ethereum (ETH)"].index(st.session_state.selected_asset)
        )
        st.session_state.selected_asset = selected_asset
    with col2:
        min_start = pd.Timestamp("2017-01-01").date()
        max_end = pd.Timestamp.now().date()
        default_start = pd.Timestamp.now() - pd.DateOffset(years=3)
        date_range = st.date_input(
            "Select Date Range",
            value=[default_start.date(), max_end],
            min_value=min_start, max_value=max_end
        )
    with col3:
        var_dist = st.radio("VaR Distribution Type", 
                           ["Normal Distribution", "t-Distribution (Fat Tail)"], horizontal=True)
        st.session_state.var_dist = var_dist
    
    # 运行分析
    if st.button("🔄 Run Analysis (Pull Data + Fit Models + Calculate VaR)", type="primary"):
        with st.spinner("Processing... (10-20 seconds)"):
            # 1. 拉取数据（已提前计算loss列）
            df = get_crypto_data(selected_asset, date_range[0], date_range[1])
            if df is None:
                st.stop()
            st.session_state.df = df
            st.success(f"✅ 成功拉取 {len(df)} 天 {selected_asset} 数据")
            
            # 2. 拟合GARCH
            cond_vol, garch_params = fit_garch_model(df['returns'])
            if cond_vol is None:
                st.stop()
            st.session_state.cond_vol = cond_vol
            st.session_state.garch_params = garch_params
            df['cond_vol'] = cond_vol.values
            st.success("✅ GARCH模型拟合完成")
            
            # 3. GARCH VaR
            var_95, var_99 = calculate_var(cond_vol, var_dist.split(' ')[0])
            st.session_state.var_95 = var_95
            st.session_state.var_99 = var_99
            df['var_95'] = var_95
            df['var_99'] = var_99
            df['break_95'] = df['loss'] > df['var_95']
            df['break_99'] = df['loss'] > df['var_99']
            
            # 4. EWMA计算（关键修复：确保df_ewma有loss列）
            ewma_vol = calculate_ewma_vol(df['returns'])
            if len(ewma_vol) == 0:
                st.stop()
            st.session_state.ewma_vol = ewma_vol
            
            # 对齐EWMA数据（切片时保留所有列）
            ewma_index = ewma_vol.index
            df['ewma_vol'] = np.nan
            df.loc[ewma_index, 'ewma_vol'] = ewma_vol.values
            
            # 5. EWMA VaR（关键修复：直接用df计算，避免切片）
            ewma_var_95, ewma_var_99 = calculate_var(ewma_vol, var_dist.split(' ')[0])
            st.session_state.ewma_var_95 = ewma_var_95
            st.session_state.ewma_var_99 = ewma_var_99
            df['ewma_var_95'] = np.nan
            df['ewma_var_99'] = np.nan
            df.loc[ewma_index, 'ewma_var_95'] = ewma_var_95.values
            df.loc[ewma_index, 'ewma_var_99'] = ewma_var_99.values
            
            # 6. EWMA击穿率（关键修复：直接在原df计算，避免KeyError）
            df['ewma_break_95'] = np.nan
            df['ewma_break_99'] = np.nan
            df.loc[ewma_index, 'ewma_break_95'] = df.loc[ewma_index, 'loss'] > df.loc[ewma_index, 'ewma_var_95']
            df.loc[ewma_index, 'ewma_break_99'] = df.loc[ewma_index, 'loss'] > df.loc[ewma_index, 'ewma_var_99']
            
            st.session_state.df = df
            st.success("✅ EWMA计算完成")
            st.info("✅ 所有计算完成！可切换到其他页面查看结果")

# 2. 数据可视化页（无修改）
elif page == "📊 Data Visualization":
    st.title("📊 Data Visualization")
    st.divider()
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ 请先在Home页运行分析！")
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.plot(df['date'], df['Close'], color="darkblue", linewidth=1.2)
        ax1.set_ylabel("Closing Price (USD)")
        ax1.set_title(f"{st.session_state.selected_asset} Historical Price")
        ax1.grid(alpha=0.3)
        
        ax2.plot(df['date'], df['log_returns'], color="green", alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Log Returns (Decimal)")
        ax2.set_title(f"{st.session_state.selected_asset} Log Returns")
        ax2.grid(alpha=0.3)
        
        ax3.plot(df['date'], df['simple_vol'], color="orange", linewidth=1.2)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("21-Day Rolling Volatility (Decimal)")
        ax3.set_title(f"{st.session_state.selected_asset} Raw Volatility")
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# 3. GARCH验证页（仅重命名，无修改）
elif page == "🧪 GARCH Model Validation":
    st.title("🧪 GARCH Model Validation")
    st.divider()
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ 请先在Home页运行分析！")
    else:
        var_dist = st.session_state.var_dist
        break_95_count = df['break_95'].sum()
        break_95_rate = break_95_count / len(df)
        break_99_count = df['break_99'].sum()
        break_99_rate = break_99_count / len(df)
        
        # VaR图
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df['date'], df['returns'], color="gray", alpha=0.5, label="Daily Returns")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.plot(df['date'], -df['var_95'], color="red", linewidth=1.5, label=f"95% {var_dist} VaR")
        ax.plot(df['date'], -df['var_99'], color="darkred", linewidth=1.5, label=f"99% {var_dist} VaR")
        break_95_df = df[df['break_95']]
        ax.scatter(break_95_df['date'], break_95_df['returns'], color="red", s=20, label="95% VaR Breakthrough")
        break_99_df = df[df['break_99']]
        ax.scatter(break_99_df['date'], break_99_df['returns'], color="darkred", s=30, label="99% VaR Breakthrough")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns (Decimal)")
        ax.set_title(f"{st.session_state.selected_asset} Returns vs GARCH VaR ({var_dist})")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("95% VaR击穿次数", f"{break_95_count}")
        col2.metric("95% VaR击穿率", f"{break_95_rate*100:.2f}%")
        col3.metric("99% VaR击穿次数", f"{break_99_count}")
        col4.metric("99% VaR击穿率", f"{break_99_rate*100:.2f}%")
        
        # 滚动预测
        st.divider()
        st.subheader("🎯 GARCH Rolling Window Prediction")
        window_size = max(int(len(df)/3), 21)
        st.info(f"🔍 自动窗口大小：{window_size} 天（总数据的1/3）")
        
        with st.spinner("运行滚动预测... (1-2分钟)"):
            rolling_df = rolling_window_prediction(df, window_size, "GARCH")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
            ax1.plot(rolling_df['date'], rolling_df['pred_vol'], color="blue", linewidth=1.5, label="Predicted Volatility")
            ax1.plot(rolling_df['date'], rolling_df['actual_vol'], color="green", linewidth=1.5, alpha=0.7, label="Actual Volatility")
            ax1.axvline(x=rolling_df['date'].iloc[0], color="red", linestyle="--", label="Prediction Start")
            ax1.set_ylabel("Volatility (Decimal)")
            ax1.set_title(f"{st.session_state.selected_asset} GARCH Rolling Prediction")
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.plot(rolling_df['date'], rolling_df['pred_var_95'], color="red", linewidth=1.5, label="Predicted 95% VaR")
            ax2.plot(rolling_df['date'], rolling_df['pred_var_99'], color="darkred", linewidth=1.5, label="Predicted 99% VaR")
            ax2.plot(rolling_df['date'], rolling_df['actual_loss'], color="gray", alpha=0.7, label="Actual Loss")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Loss / VaR (Decimal)")
            ax2.set_title(f"{st.session_state.selected_asset} GARCH VaR vs Actual Loss")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 滚动预测指标
            rolling_break_95 = (rolling_df['actual_loss'] > rolling_df['pred_var_95']).sum()
            rolling_break_95_rate = rolling_break_95 / len(rolling_df)
            rolling_break_99 = (rolling_df['actual_loss'] > rolling_df['pred_var_99']).sum()
            rolling_break_99_rate = rolling_break_99 / len(rolling_df)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("预测天数", f"{len(rolling_df)}")
            col2.metric("95% VaR击穿率", f"{rolling_break_95_rate*100:.2f}%")
            col3.metric("99% VaR击穿次数", f"{rolling_break_99}")
            col4.metric("99% VaR击穿率", f"{rolling_break_99_rate*100:.2f}%")

# 4. EWMA验证页（纯EWMA，无GARCH对比）
elif page == "📊 EWMA Model Validation":
    st.title("📊 EWMA Model Validation")
    st.divider()
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ 请先在Home页运行分析！")
    else:
        # 筛选有效EWMA数据
        df_ewma = df.dropna(subset=['ewma_vol']).copy()
        if len(df_ewma) == 0:
            st.error("❌ 无有效EWMA数据！")
            st.stop()
        
        var_dist = st.session_state.var_dist
        break_95_count = df_ewma['ewma_break_95'].sum()
        break_95_rate = break_95_count / len(df_ewma)
        break_99_count = df_ewma['ewma_break_99'].sum()
        break_99_rate = break_99_count / len(df_ewma)
        
        # EWMA VaR图
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(df_ewma['date'], df_ewma['returns'], color="gray", alpha=0.5, label="Daily Returns")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.plot(df_ewma['date'], -df_ewma['ewma_var_95'], color="orange", linewidth=1.5, label=f"95% {var_dist} VaR")
        ax.plot(df_ewma['date'], -df_ewma['ewma_var_99'], color="darkorange", linewidth=1.5, label=f"99% {var_dist} VaR")
        break_95_df = df_ewma[df_ewma['ewma_break_95']]
        ax.scatter(break_95_df['date'], break_95_df['returns'], color="orange", s=20, label="95% VaR Breakthrough")
        break_99_df = df_ewma[df_ewma['ewma_break_99']]
        ax.scatter(break_99_df['date'], break_99_df['returns'], color="darkorange", s=30, label="99% VaR Breakthrough")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns (Decimal)")
        ax.set_title(f"{st.session_state.selected_asset} Returns vs EWMA VaR ({var_dist})")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # EWMA指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("95% VaR击穿次数", f"{break_95_count}")
        col2.metric("95% VaR击穿率", f"{break_95_rate*100:.2f}%")
        col3.metric("99% VaR击穿次数", f"{break_99_count}")
        col4.metric("99% VaR击穿率", f"{break_99_rate*100:.2f}%")
        
        # EWMA滚动预测
        st.divider()
        st.subheader("🎯 EWMA Rolling Window Prediction")
        window_size = max(int(len(df_ewma)/3), 21)
        st.info(f"🔍 自动窗口大小：{window_size} 天（EWMA数据的1/3）")
        
        with st.spinner("运行EWMA滚动预测... (1-2分钟)"):
            rolling_df = rolling_window_prediction(df_ewma, window_size, "EWMA")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
            ax1.plot(rolling_df['date'], rolling_df['pred_vol'], color="orange", linewidth=1.5, label="Predicted EWMA Volatility")
            ax1.plot(rolling_df['date'], rolling_df['actual_vol'], color="green", linewidth=1.5, alpha=0.7, label="Actual Rolling Volatility")
            ax1.axvline(x=rolling_df['date'].iloc[0], color="red", linestyle="--", label="Prediction Start")
            ax1.set_ylabel("Volatility (Decimal)")
            ax1.set_title(f"{st.session_state.selected_asset} EWMA Rolling Prediction")
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            ax2.plot(rolling_df['date'], rolling_df['pred_var_95'], color="orange", linewidth=1.5, label="Predicted 95% VaR")
            ax2.plot(rolling_df['date'], rolling_df['pred_var_99'], color="darkorange", linewidth=1.5, label="Predicted 99% VaR")
            ax2.plot(rolling_df['date'], rolling_df['actual_loss'], color="gray", alpha=0.7, label="Actual Loss")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Loss / VaR (Decimal)")
            ax2.set_title(f"{st.session_state.selected_asset} EWMA VaR vs Actual Loss")
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # EWMA滚动预测指标
            rolling_break_95 = (rolling_df['actual_loss'] > rolling_df['pred_var_95']).sum()
            rolling_break_95_rate = rolling_break_95 / len(rolling_df)
            rolling_break_99 = (rolling_df['actual_loss'] > rolling_df['pred_var_99']).sum()
            rolling_break_99_rate = rolling_break_99 / len(rolling_df)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("预测天数", f"{len(rolling_df)}")
            col2.metric("95% VaR击穿率", f"{rolling_break_95_rate*100:.2f}%")
            col3.metric("99% VaR击穿次数", f"{rolling_break_99}")
            col4.metric("99% VaR击穿率", f"{rolling_break_99_rate*100:.2f}%")

# 5. 模型对比页
elif page == "🔍 Model Comparison":
    st.title("🔍 GARCH vs EWMA Model Comparison")
    st.divider()
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ 请先在Home页运行分析！")
    else:
        # 筛选同时有GARCH和EWMA的数据
        df_compare = df.dropna(subset=['cond_vol', 'ewma_vol']).copy()
        if len(df_compare) == 0:
            st.error("❌ 无对比数据！")
            st.stop()
        
        # 统计对比表格
        st.subheader("📋 Model Performance Statistics")
        stats_data = {
            'Metric': [
                'Average Volatility (%)',
                '95% VaR (Avg, %)',
                '99% VaR (Avg, %)',
                '95% VaR Breakthrough Rate (%)',
                '99% VaR Breakthrough Rate (%)',
                'Volatility Std Dev (%)'
            ],
            'GARCH Model': [
                f"{df_compare['cond_vol'].mean()*100:.2f}",
                f"{df_compare['var_95'].mean()*100:.2f}",
                f"{df_compare['var_99'].mean()*100:.2f}",
                f"{(df_compare['break_95'].sum()/len(df_compare)*100):.2f}",
                f"{(df_compare['break_99'].sum()/len(df_compare)*100):.2f}",
                f"{df_compare['cond_vol'].std()*100:.2f}"
            ],
            'EWMA Model': [
                f"{df_compare['ewma_vol'].mean()*100:.2f}",
                f"{df_compare['ewma_var_95'].mean()*100:.2f}",
                f"{df_compare['ewma_var_99'].mean()*100:.2f}",
                f"{(df_compare['ewma_break_95'].sum()/len(df_compare)*100):.2f}",
                f"{(df_compare['ewma_break_99'].sum()/len(df_compare)*100):.2f}",
                f"{df_compare['ewma_vol'].std()*100:.2f}"
            ]
        }
        st.table(pd.DataFrame(stats_data))
        
        # 波动率对比图
        st.divider()
        st.subheader("📈 Volatility Comparison")
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(df_compare['date'], df_compare['cond_vol'], color="royalblue", linewidth=1.2, label="GARCH Volatility")
        ax.plot(df_compare['date'], df_compare['ewma_vol'], color="orange", linewidth=1.2, alpha=0.8, label="EWMA Volatility (λ=0.94)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (Decimal)")
        ax.set_title(f"{st.session_state.selected_asset} GARCH vs EWMA Volatility")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # VaR对比图
        st.divider()
        st.subheader("🛡️ 95% VaR Comparison")
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(df_compare['date'], -df_compare['var_95'], color="royalblue", linewidth=1.2, label="GARCH 95% VaR")
        ax.plot(df_compare['date'], -df_compare['ewma_var_95'], color="orange", linewidth=1.2, alpha=0.8, label="EWMA 95% VaR")
        ax.plot(df_compare['date'], df_compare['returns'], color="gray", alpha=0.5, label="Daily Returns")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns / VaR (Decimal)")
        ax.set_title(f"{st.session_state.selected_asset} GARCH vs EWMA 95% VaR")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# 6. 预测页（双模型对比）
elif page == "🔮 Prediction":
    st.title("🔮 Next-Day Prediction (GARCH + EWMA)")
    st.divider()
    df = st.session_state.df
    garch_params = st.session_state.garch_params
    if df is None or garch_params is None:
        st.warning("⚠️ 请先在Home页运行分析！")
    else:
        # 计算下一个交易日
        last_date = df['date'].iloc[-1]
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        next_date_str = next_date.strftime("%Y-%m-%d")
        
        # GARCH预测
        last_garch_vol = df['cond_vol'].iloc[-1]
        garch_pred = predict_next_vol_var(df['returns'], garch_params, last_garch_vol, "GARCH")
        
        # EWMA预测
        last_ewma_vol = df['ewma_vol'].dropna().iloc[-1] if len(df['ewma_vol'].dropna()) > 0 else df['simple_vol'].iloc[-1]
        ewma_pred = predict_next_vol_var(df['returns'], {}, last_ewma_vol, "EWMA")
        
        # 预测表格
        st.subheader(f"📅 Prediction for {next_date_str}")
        pred_data = {
            'Metric': [
                'Predicted Volatility (%)',
                '95% Normal VaR (%)',
                '99% Normal VaR (%)',
                '95% t-VaR (%)',
                '99% t-VaR (%)'
            ],
            'GARCH Model': [
                f"{garch_pred[0]*100:.2f}" if garch_pred[0] else "N/A",
                f"{garch_pred[1]*100:.2f}" if garch_pred[1] else "N/A",
                f"{garch_pred[2]*100:.2f}" if garch_pred[2] else "N/A",
                f"{garch_pred[3]*100:.2f}" if garch_pred[3] else "N/A",
                f"{garch_pred[4]*100:.2f}" if garch_pred[4] else "N/A"
            ],
            'EWMA Model': [
                f"{ewma_pred[0]*100:.2f}" if ewma_pred[0] else "N/A",
                f"{ewma_pred[1]*100:.2f}" if ewma_pred[1] else "N/A",
                f"{ewma_pred[2]*100:.2f}" if ewma_pred[2] else "N/A",
                f"{ewma_pred[3]*100:.2f}" if ewma_pred[3] else "N/A",
                f"{ewma_pred[4]*100:.2f}" if ewma_pred[4] else "N/A"
            ]
        }
        st.table(pd.DataFrame(pred_data))
        
        # 解释
        st.divider()
        st.markdown(f"""
        ### 📝 Interpretation
        - **GARCH**: 保守型预测，更适合极端风险评估
        - **EWMA**: 对近期波动更敏感，适合短期预测
        - t分布VaR更适配加密货币的厚尾特性（推荐参考）
        """)

# 页脚
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'>Crypto Volatility Dashboard | Powered by Yahoo Finance & Streamlit</p>", unsafe_allow_html=True)

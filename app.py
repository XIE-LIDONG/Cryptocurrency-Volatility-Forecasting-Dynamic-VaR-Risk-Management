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

# 全局设置
st.set_page_config(page_title="Crypto Volatility Dashboard", page_icon="📈", layout="wide")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 初始化session state（极简）
if 'selected_asset' not in st.session_state:
    st.session_state.selected_asset = "Bitcoin (BTC)"
if 'var_dist' not in st.session_state:
    st.session_state.var_dist = "Normal Distribution"

# 核心函数（仅保留必要逻辑）
def get_crypto_data(asset, start_date, end_date):
    ticker_map = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD"}
    df = yf.download(ticker_map[asset], start=start_date, end=end_date)
    if df.empty or len(df) < 50:
        return None
    df = df[['Close']].copy()
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['simple_vol'] = df['returns'].rolling(window=21).std()
    df['loss'] = -df['returns']  # 提前计算loss
    df = df.dropna()
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    return df

def fit_garch_model(returns):
    try:
        am = arch_model(returns * 100, mean='Zero', vol='GARCH', p=1, q=1)
        res = am.fit(disp='off')
        params = {'omega': res.params['omega']/10000, 'alpha': res.params['alpha[1]'], 'beta': res.params['beta[1]']}
        return res.conditional_volatility / 100, params
    except:
        return None, None

def calculate_ewma_vol(returns):
    if len(returns) < 21:
        return pd.Series()
    initial_vol = returns.iloc[:21].std()
    vol_list = []
    for i in range(21, len(returns)):
        prev_vol_sq = initial_vol**2 if i == 21 else vol_list[-1]**2
        vol_list.append(np.sqrt(0.94 * prev_vol_sq + 0.06 * returns.iloc[i-1]**2))
    return pd.Series([np.nan]*21 + vol_list, index=returns.index)

def calculate_var(cond_vol, dist_type="Normal"):
    if cond_vol is None or len(cond_vol) == 0:
        return None, None
    if dist_type == "Normal":
        return 1.65 * cond_vol, 2.33 * cond_vol
    else:
        return abs(t.ppf(0.05, 8)) * cond_vol, abs(t.ppf(0.01, 8)) * cond_vol

def rolling_window_prediction(df, window_size, model_type="GARCH"):
    window_size = max(window_size, 21)
    rolling_vol, rolling_var_95, rolling_var_99 = [], [], []
    actual_vol, actual_loss, dates = [], [], []
    for i in range(window_size, len(df)):
        train_ret = df['returns'].iloc[i-window_size:i]
        if model_type == "GARCH":
            am = arch_model(train_ret*100, mean='Zero', vol='GARCH', p=1, q=1)
            res = am.fit(disp='off')
            last_vol = res.conditional_volatility.iloc[-1]/100
            next_vol = np.sqrt(res.params['omega']/10000 + res.params['alpha[1]']*train_ret.iloc[-1]**2 + res.params['beta[1]']*last_vol**2)
        else:
            ewma_vol = calculate_ewma_vol(train_ret)
            last_vol = ewma_vol.dropna().iloc[-1] if len(ewma_vol.dropna()) > 0 else train_ret.std()
            next_vol = np.sqrt(0.94 * last_vol**2 + 0.06 * train_ret.iloc[-1]**2)
        rolling_vol.append(next_vol)
        rolling_var_95.append(1.65 * next_vol)
        rolling_var_99.append(2.33 * next_vol)
        actual_vol.append(df['cond_vol'].iloc[i] if model_type=="GARCH" else df['simple_vol'].iloc[i])
        actual_loss.append(df['loss'].iloc[i])
        dates.append(df['date'].iloc[i])
    return pd.DataFrame({'date':dates, 'pred_vol':rolling_vol, 'pred_var_95':rolling_var_95, 
                         'pred_var_99':rolling_var_99, 'actual_vol':actual_vol, 'actual_loss':actual_loss})

# 导航栏
page = st.sidebar.radio("Select Page", ["Home", "GARCH Validation", "EWMA Validation", "Comparison", "Prediction"])

# Home页（核心修复点）
if page == "Home":
    st.title("Crypto Volatility & VaR Dashboard")
    col1, col2, col3 = st.columns([1.5,2,1.5])
    with col1:
        selected_asset = st.selectbox("Crypto", ["Bitcoin (BTC)", "Ethereum (ETH)"], 
                                     index=["Bitcoin (BTC)", "Ethereum (ETH)"].index(st.session_state.selected_asset))
        st.session_state.selected_asset = selected_asset
    with col2:
        min_date = pd.Timestamp("2017-01-01").date()
        max_date = pd.Timestamp.now().date()
        date_range = st.date_input("Date Range", [pd.Timestamp.now()-pd.DateOffset(years=3), max_date], min_date, max_date)
    with col3:
        var_dist = st.radio("VaR Distribution", ["Normal Distribution", "t-Distribution"], horizontal=True,
                           index=0 if st.session_state.var_dist=="Normal Distribution" else 1)
        st.session_state.var_dist = var_dist

    if st.button("Run Analysis"):
        with st.spinner("Processing..."):
            df = get_crypto_data(selected_asset, date_range[0], date_range[1])
            if df is None:
                st.error("No valid data!")
                st.stop()
            
            # GARCH
            cond_vol, garch_params = fit_garch_model(df['returns'])
            if cond_vol is None:
                st.error("GARCH fit failed!")
                st.stop()
            df['cond_vol'] = cond_vol
            var_95, var_99 = calculate_var(cond_vol, var_dist.split(' ')[0])
            df['var_95'] = var_95
            df['var_99'] = var_99
            df['break_95'] = df['loss'] > df['var_95']
            df['break_99'] = df['loss'] > df['var_99']
            
            # EWMA（核心修复：统一索引长度）
            ewma_vol = calculate_ewma_vol(df['returns'])
            df['ewma_vol'] = ewma_vol
            ewma_var_95, ewma_var_99 = calculate_var(ewma_vol.dropna(), var_dist.split(' ')[0])
            
            # 修复赋值逻辑：先创建等长数组，再赋值
            df['ewma_var_95'] = np.nan
            df['ewma_var_99'] = np.nan
            valid_ewma_idx = ewma_vol.dropna().index  # 只取非空的EWMA索引
            df.loc[valid_ewma_idx, 'ewma_var_95'] = ewma_var_95.values  # 用.values保证一维数组
            df.loc[valid_ewma_idx, 'ewma_var_99'] = ewma_var_99.values
            
            # 关键修复：确保比较结果是一维数组，长度匹配
            df['ewma_break_95'] = np.nan
            df['ewma_break_99'] = np.nan
            # 先计算布尔值，转为numpy数组保证长度匹配
            break_95_vals = (df.loc[valid_ewma_idx, 'loss'] > df.loc[valid_ewma_idx, 'ewma_var_95']).values
            break_99_vals = (df.loc[valid_ewma_idx, 'loss'] > df.loc[valid_ewma_idx, 'ewma_var_99']).values
            # 赋值
            df.loc[valid_ewma_idx, 'ewma_break_95'] = break_95_vals
            df.loc[valid_ewma_idx, 'ewma_break_99'] = break_99_vals
            
            st.session_state.df = df
            st.success("Analysis completed!")

# GARCH Validation
elif page == "GARCH Validation":
    st.title("GARCH Model Validation")
    df = st.session_state.get('df')
    if df is None:
        st.warning("Run analysis first!")
        st.stop()
    # 极简绘图
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(df['date'], df['returns'], 'gray', alpha=0.5, label='Returns')
    ax.plot(df['date'], -df['var_95'], 'red', label='95% VaR')
    ax.plot(df['date'], -df['var_99'], 'darkred', label='99% VaR')
    ax.legend()
    st.pyplot(fig)
    # 滚动预测
    rolling_df = rolling_window_prediction(df, int(len(df)/3), "GARCH")
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(rolling_df['date'], rolling_df['pred_vol'], 'blue', label='Pred Vol')
    ax.plot(rolling_df['date'], rolling_df['actual_vol'], 'green', label='Actual Vol')
    ax.legend()
    st.pyplot(fig)

# EWMA Validation
elif page == "EWMA Validation":
    st.title("EWMA Model Validation")
    df = st.session_state.get('df')
    if df is None:
        st.warning("Run analysis first!")
        st.stop()
    df_ewma = df.dropna(subset=['ewma_vol'])
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(df_ewma['date'], df_ewma['returns'], 'gray', alpha=0.5, label='Returns')
    ax.plot(df_ewma['date'], -df_ewma['ewma_var_95'], 'orange', label='95% VaR')
    ax.plot(df_ewma['date'], -df_ewma['ewma_var_99'], 'darkorange', label='99% VaR')
    ax.legend()
    st.pyplot(fig)
    # 滚动预测
    rolling_df = rolling_window_prediction(df_ewma, int(len(df_ewma)/3), "EWMA")
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(rolling_df['date'], rolling_df['pred_vol'], 'orange', label='Pred Vol')
    ax.plot(rolling_df['date'], rolling_df['actual_vol'], 'green', label='Actual Vol')
    ax.legend()
    st.pyplot(fig)

# Comparison
elif page == "Comparison":
    st.title("GARCH vs EWMA Comparison")
    df = st.session_state.get('df')
    if df is None:
        st.warning("Run analysis first!")
        st.stop()
    df_comp = df.dropna(subset=['cond_vol', 'ewma_vol'])
    # 波动率对比
    fig, ax = plt.subplots(figsize=(15,7))
    ax.plot(df_comp['date'], df_comp['cond_vol'], 'blue', label='GARCH Vol')
    ax.plot(df_comp['date'], df_comp['ewma_vol'], 'orange', label='EWMA Vol')
    ax.legend()
    st.pyplot(fig)

# Prediction
elif page == "Prediction":
    st.title("Next-Day Prediction")
    df = st.session_state.get('df')
    if df is None:
        st.warning("Run analysis first!")
         st.stop()
    # GARCH预测
    last_garch_vol = df['cond_vol'].iloc[-1]
    garch_next_vol = np.sqrt(st.session_state.get('garch_params')['omega'] + st.session_state.get('garch_params')['alpha']*df['returns'].iloc[-1]**2 + st.session_state.get('garch_params')['beta']*last_garch_vol**2)
    # EWMA预测
    last_ewma_vol = df['ewma_vol'].dropna().iloc[-1]
    ewma_next_vol = np.sqrt(0.94 * last_ewma_vol**2 + 0.06 * df['returns'].iloc[-1]**2)
    # 展示
    st.write(f"GARCH Pred Vol: {garch_next_vol*100:.2f}%")
    st.write(f"EWMA Pred Vol: {ewma_next_vol*100:.2f}%")



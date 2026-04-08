# ============================================================
# STOCK & PORTFOLIO ANALYTICS APP — Streamlit
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Stock & Portfolio Analytics",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS (for clean look)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.metric-value { font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTION
# ============================================================
def format_currency(val):
    return f"${val:,.2f}"

def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_annual_volatility(series, window=20):
    returns = series.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(252)
    return vol

# ============================================================
# APP TITLE & TABS
# ============================================================
st.title("📈 Stock & Portfolio Analytics App")
tab1, tab2 = st.tabs(["📊 Individual Stock Analysis", "💼 Portfolio Dashboard"])

# ============================================================
# TAB 1: INDIVIDUAL STOCK
# ============================================================
with tab1:
    st.header("Individual Stock Analysis")

    # --- Inputs ---
    ticker = st.text_input("Enter stock ticker", value="AAPL").upper()
    start_date = st.date_input("Start date", pd.to_datetime("today") - pd.Timedelta(days=180))
    end_date = st.date_input("End date", pd.to_datetime("today"))

    # --- Download data ---
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or 'Close' not in df.columns:
        st.error(f"No price data found for {ticker}. Check ticker or date range.")
    else:
        close = df['Close']
        current_price = close.iloc[-1]

        # --- Moving averages ---
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]

        # --- Trend ---
        if current_price > ma20 > ma50:
            trend = "Strong Uptrend"
        elif current_price < ma20 < ma50:
            trend = "Strong Downtrend"
        else:
            trend = "Mixed Trend"

        # --- RSI ---
        rsi_series = calculate_rsi(close)
        rsi = rsi_series.iloc[-1]
        if rsi > 70:
            rsi_signal = "Overbought (Sell)"
        elif rsi < 30:
            rsi_signal = "Oversold (Buy)"
        else:
            rsi_signal = "Neutral"

        # --- Volatility ---
        vol = calculate_annual_volatility(close).iloc[-1] * 100
        if vol > 40:
            vol_label = "High"
        elif vol >= 25:
            vol_label = "Medium"
        else:
            vol_label = "Low"

        # --- Recommendation ---
        if trend == "Strong Uptrend" and rsi_signal == "Oversold (Buy)":
            recommendation = "Buy"
        elif trend == "Strong Downtrend" or rsi_signal == "Overbought (Sell)":
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        # --- Display metrics ---
        st.subheader(f"Latest analysis for {ticker}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", format_currency(current_price))
        col2.metric("20-Day MA", format_currency(ma20))
        col3.metric("50-Day MA", format_currency(ma50))
        col4.metric("RSI", f"{rsi:.1f}", rsi_signal)
        col5.metric("Volatility", f"{vol:.1f}%", vol_label)

        st.markdown(f"**Trend:** {trend}  |  **Recommendation:** {recommendation}")

        # --- Chart ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(close.index, close, label="Close")
        ax.plot(close.index, close.rolling(20).mean(), label="20-day MA")
        ax.plot(close.index, close.rolling(50).mean(), label="50-day MA")
        ax.set_title(f"{ticker} Price & Moving Averages")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

# ============================================================
# TAB 2: PORTFOLIO
# ============================================================
with tab2:
    st.header("Portfolio Performance Dashboard")

    # --- Inputs ---
    default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    tickers = st.text_area("Enter 5 tickers (comma-separated)", value=",".join(default_stocks))
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    weights = st.text_area("Enter 5 weights (sum=1.0, comma-separated)", value="0.2,0.2,0.2,0.2,0.2")
    weights = [float(w) for w in weights.split(",")]

    benchmark = st.text_input("Benchmark ticker", value="^GSPC").upper()
    start_date_pf = st.date_input("Start date (portfolio)", pd.to_datetime("today") - pd.Timedelta(days=365))
    end_date_pf = st.date_input("End date (portfolio)", pd.to_datetime("today"))

    if len(tickers) != 5 or len(weights) != 5 or abs(sum(weights) - 1.0) > 0.001:
        st.error("Enter exactly 5 tickers and weights that sum to 1.0")
    else:
        # --- Download portfolio & benchmark data ---
        df_pf = yf.download(tickers + [benchmark], start=start_date_pf, end=end_date_pf)['Close']

        if df_pf.empty:
            st.error("No data found for portfolio tickers / benchmark.")
        else:
            # Portfolio returns
            returns = df_pf[tickers].pct_change().dropna()
            portfolio_returns = (returns * weights).sum(axis=1)
            benchmark_returns = df_pf[benchmark].pct_change().dropna()

            # Performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            benchmark_return = (1 + benchmark_returns).prod() - 1
            vol_pf = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))

            st.subheader("Portfolio Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Portfolio Total Return", f"{total_return*100:.1f}%")
            col2.metric(f"{benchmark} Return", f"{benchmark_return*100:.1f}%")
            col3.metric("Annual Volatility", f"{vol_pf*100:.1f}%")
            col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            # Comparison plot
            cum_pf = (1 + portfolio_returns).cumprod()
            cum_bench = (1 + benchmark_returns).cumprod()
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(cum_pf.index, cum_pf, label="Portfolio")
            ax2.plot(cum_bench.index, cum_bench, label=benchmark)
            ax2.set_title("Portfolio vs Benchmark Cumulative Returns")
            ax2.set_ylabel("Growth of $1")
            ax2.legend()
            st.pyplot(fig2)

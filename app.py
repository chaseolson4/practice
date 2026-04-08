# ============================================================
# STOCK ANALYTICS & PORTFOLIO DASHBOARD — Streamlit App
# ============================================================

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Stock Analytics & Portfolio Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS STYLING (matches personal finance example)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.metric-card {background: #f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem 1.5rem; text-align:center; margin-bottom:1rem;}
.metric-label {font-size:13px; color:#64748b; margin-bottom:4px; letter-spacing:0.03em; text-transform:uppercase;}
.metric-value {font-size:28px; font-weight:600; color:#0f172a; font-family:'DM Mono', monospace;}
.metric-value.green {color:#16a34a;}
.metric-value.red {color:#dc2626;}
.tip-box {background:#eff6ff; border-left:4px solid #2563eb; border-radius:0 8px 8px 0; padding:1rem 1.25rem; margin-top:1rem; font-size:15px; color:#1e3a5f; line-height:1.7;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_currency(value):
    return f"${value:,.2f}"

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def annualized_volatility(returns):
    return np.std(returns) * np.sqrt(252)

# ============================================================
# APP TITLE & NAVIGATION
# ============================================================
st.title("📈 Stock Analytics & Portfolio Dashboard")
st.markdown("*Analyze individual stocks and multi-asset portfolios with real data from Yahoo Finance.*")

tab1, tab2 = st.tabs([
    "📊 Individual Stock Analysis",
    "💹 Portfolio Performance Dashboard"
])

# ============================================================
# TAB 1: INDIVIDUAL STOCK ANALYSIS
# ============================================================
with tab1:

    st.header("Individual Stock Analysis")

    # --- STOCK INPUT ---
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()

    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)  # last 6 months

    # --- DATA DOWNLOAD ---
    df = yf.download(ticker, start=start_date, end=end_date)

   if df.empty or 'Close' not in df.columns:
    st.error(f"No price data available for {ticker}")
else:
    close = df['Close']
    current_price = close.iloc[-1]

        # --- TREND ANALYSIS ---
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        current_price = close[-1]

        if current_price > ma20[-1] > ma50[-1]:
            trend = "Strong Uptrend"
            trend_color = "green"
        elif current_price < ma20[-1] < ma50[-1]:
            trend = "Strong Downtrend"
            trend_color = "red"
        else:
            trend = "Mixed Trend"
            trend_color = "black"

        # --- RSI ANALYSIS ---
        rsi = calculate_rsi(close)
        current_rsi = rsi[-1]
        if current_rsi > 70:
            momentum = "Overbought (Sell Signal)"
            rsi_color = "red"
        elif current_rsi < 30:
            momentum = "Oversold (Buy Signal)"
            rsi_color = "green"
        else:
            momentum = "Neutral"
            rsi_color = "black"

        # --- VOLATILITY ---
        returns = close.pct_change().dropna()
        vol = annualized_volatility(returns)
        if vol > 0.40:
            vol_label = "High"
        elif vol > 0.25:
            vol_label = "Medium"
        else:
            vol_label = "Low"

        # --- TRADING RECOMMENDATION ---
        if trend == "Strong Uptrend" and momentum == "Neutral":
            recommendation = "Buy"
        elif trend == "Strong Downtrend" or momentum == "Overbought (Sell Signal)":
            recommendation = "Sell"
        else:
            recommendation = "Hold"

        # --- DISPLAY METRICS ---
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Current Price", format_currency(current_price))
        with col2: st.metric("Trend", trend, "", delta_color=trend_color)
        with col3: st.metric("RSI", f"{current_rsi:.1f}", momentum)
        with col4: st.metric("Volatility", f"{vol*100:.1f}%", vol_label)

        st.markdown(f"<div class='tip-box'><strong>Trading Recommendation:</strong> {recommendation}</div>", unsafe_allow_html=True)

        # --- PLOT PRICE & MOVING AVERAGES ---
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(close.index, close, label="Close Price")
        ax.plot(ma20.index, ma20, label="20-Day MA")
        ax.plot(ma50.index, ma50, label="50-Day MA")
        ax.set_title(f"{ticker} Price & Moving Averages (6 Months)")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ============================================================
# TAB 2: PORTFOLIO PERFORMANCE DASHBOARD
# ============================================================
with tab2:

    st.header("Portfolio Performance Dashboard")

    # --- DEFAULT PORTFOLIO ---
    default_portfolio = {
        "AAPL": 0.2,
        "MSFT": 0.2,
        "GOOGL": 0.2,
        "AMZN": 0.2,
        "META": 0.2
    }

    st.subheader("Portfolio Tickers & Weights")
    tickers = []
    weights = []

    cols = st.columns(len(default_portfolio))
    for i, (stock, w) in enumerate(default_portfolio.items()):
        tickers.append(cols[i].text_input(f"Ticker {i+1}", value=stock).upper())
        weights.append(cols[i].number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0, value=w, step=0.05))

    # Normalize weights if not sum to 1
    total_weight = sum(weights)
    if total_weight != 1.0:
        weights = [w/total_weight for w in weights]

    benchmark_ticker = st.text_input("Benchmark Ticker", value="^GSPC").upper()

    # --- DATA DOWNLOAD ---
    start_date_portfolio = datetime.today() - timedelta(days=365)
    all_tickers = tickers + [benchmark_ticker]
    df_port = yf.download(all_tickers, start=start_date_portfolio)['Close']

    if df_port.empty:
        st.error("Error fetching portfolio data.")
    else:
        # --- CALCULATE RETURNS ---
        returns = df_port[tickers].pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        benchmark_returns = df_port[benchmark_ticker].pct_change().dropna()

        # --- PERFORMANCE METRICS ---
        total_return = (1 + portfolio_returns).prod() - 1
        benchmark_return = (1 + benchmark_returns).prod() - 1
        vol_port = annualized_volatility(portfolio_returns)
        vol_bench = annualized_volatility(benchmark_returns)
        sharpe_ratio = (portfolio_returns.mean()/portfolio_returns.std())*np.sqrt(252)
        outperformance = total_return - benchmark_return

        # --- DISPLAY METRICS ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Portfolio Return", f"{total_return*100:.2f}%")
        c2.metric("Benchmark Return", f"{benchmark_return*100:.2f}%")
        c3.metric("Outperformance", f"{outperformance*100:.2f}%")
        c4.metric("Portfolio Volatility", f"{vol_port*100:.2f}%")
        c5.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        # --- PLOT CUMULATIVE RETURNS ---
        cumulative_port = (1 + portfolio_returns).cumprod()
        cumulative_bench = (1 + benchmark_returns).cumprod()
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(cumulative_port.index, cumulative_port, label="Portfolio")
        ax2.plot(cumulative_bench.index, cumulative_bench, label=benchmark_ticker)
        ax2.set_title("Cumulative Returns (1 Year)")
        ax2.set_ylabel("Cumulative Return ($1 invested)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # --- INTERPRETATION ---
        st.markdown(f"""
        <div class='tip-box'>
        <strong>Portfolio Insights:</strong><br><br>
        Your portfolio returned <strong>{total_return*100:.2f}%</strong> over the past year.<br>
        Benchmark ({benchmark_ticker}) returned <strong>{benchmark_return*100:.2f}%</strong>.<br>
        Outperformance: <strong>{outperformance*100:.2f}%</strong>.<br>
        Portfolio volatility: <strong>{vol_port*100:.2f}%</strong> (Benchmark: {vol_bench*100:.2f}%)<br>
        Sharpe Ratio: <strong>{sharpe_ratio:.2f}</strong> — measures risk-adjusted efficiency.<br><br>
        { "Your portfolio outperformed the benchmark!" if outperformance > 0 else "Your portfolio underperformed the benchmark." }
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Built with Python, Streamlit & yfinance · For educational purposes")

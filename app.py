# ============================================================
# STOCK ANALYTICS & PORTFOLIO DASHBOARD — Streamlit App
# ============================================================

# --- IMPORTS ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None
        return data["Close"].dropna()
    except:
        return None

def moving_average(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi

def volatility(returns):
    return returns.std() * np.sqrt(252)

def format_pct(x):
    return f"{x*100:.2f}%"

# ============================================================
# TITLE
# ============================================================
st.title("📈 Stock Analytics & Portfolio Dashboard")
st.markdown("*Analyze stocks using real market data.*")

tab1, tab2 = st.tabs(["📊 Stock Analysis", "💼 Portfolio Dashboard"])

# ============================================================
# TAB 1: STOCK ANALYSIS
# ============================================================
with tab1:

    st.header("Individual Stock Analysis")

    ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")

    if ticker:

        data = get_data(ticker, "6mo")

        if data is None or len(data) < 50:
            st.error("Invalid ticker or not enough data (need ~50 days).")
        else:

            # --- CALCULATIONS ---
            current_price = float(data.iloc[-1])

            ma20 = moving_average(data, 20)
            ma50 = moving_average(data, 50)

            latest_ma20 = ma20.dropna().iloc[-1]
            latest_ma50 = ma50.dropna().iloc[-1]

            # --- TREND (FIXED) ---
            if pd.notna(current_price) and pd.notna(latest_ma20) and pd.notna(latest_ma50):

                if current_price > latest_ma20 and latest_ma20 > latest_ma50:
                    trend = "Strong Uptrend 📈"
                elif current_price < latest_ma20 and latest_ma20 < latest_ma50:
                    trend = "Strong Downtrend 📉"
                else:
                    trend = "Mixed Trend ⚖️"
            else:
                trend = "Not enough data"

            # --- RSI (FIXED) ---
            rsi = calculate_rsi(data).dropna()
            latest_rsi = float(rsi.iloc[-1]) if not rsi.empty else None

            if latest_rsi is None:
                rsi_signal = "Not enough data"
            elif latest_rsi > 70:
                rsi_signal = "Overbought (Sell)"
            elif latest_rsi < 30:
                rsi_signal = "Oversold (Buy)"
            else:
                rsi_signal = "Neutral"

            # --- VOLATILITY ---
            returns = data.pct_change().dropna()
            vol = volatility(returns)

            if vol > 0.40:
                vol_level = "High"
            elif vol > 0.25:
                vol_level = "Medium"
            else:
                vol_level = "Low"

            # --- RECOMMENDATION ---
            if "Uptrend" in trend and (latest_rsi is not None and latest_rsi < 70):
                recommendation = "Buy ✅"
            elif "Downtrend" in trend and (latest_rsi is not None and latest_rsi > 30):
                recommendation = "Sell ❌"
            else:
                recommendation = "Hold ⚖️"

            # --- DISPLAY ---
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("20-Day MA", f"${latest_ma20:.2f}")
            col3.metric("50-Day MA", f"${latest_ma50:.2f}")
            col4.metric("RSI (14)", f"{latest_rsi:.2f}" if latest_rsi else "N/A")

            st.subheader("Analysis")
            st.write(f"**Trend:** {trend}")
            st.write(f"**RSI Signal:** {rsi_signal}")
            st.write(f"**Volatility:** {format_pct(vol)} ({vol_level})")

            st.subheader("Recommendation")
            st.success(recommendation)

            # --- CHART ---
            st.subheader("Price Chart")

            chart_df = pd.DataFrame({
                "Price": data,
                "MA20": ma20,
                "MA50": ma50
            })

            st.line_chart(chart_df)

# ============================================================
# TAB 2: PORTFOLIO DASHBOARD
# ============================================================
with tab2:

    st.header("Portfolio Performance")

    tickers = []
    weights = []

    cols = st.columns(5)

    defaults = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    for i in range(5):
        with cols[i]:
            t = st.text_input(f"Stock {i+1}", defaults[i])
            w = st.number_input(f"Weight {i+1}", 0.0, 1.0, 0.2)
            tickers.append(t)
            weights.append(w)

    if abs(sum(weights) - 1.0) > 0.01:
        st.error("Weights must sum to 1.0")

    else:
        try:
            prices = yf.download(tickers, period="1y", progress=False)["Close"].dropna()
            benchmark = yf.download("^GSPC", period="1y", progress=False)["Close"].dropna()

            if prices.empty or benchmark.empty:
                st.error("Error loading market data.")
            else:
                returns = prices.pct_change().dropna()
                benchmark_returns = benchmark.pct_change().dropna()

                weights_array = np.array(weights)

                portfolio_returns = returns.dot(weights_array)

                total_return = (1 + portfolio_returns).prod() - 1
                benchmark_return = (1 + benchmark_returns).prod() - 1

                volatility_port = volatility(portfolio_returns)
                sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)

                performance = total_return - benchmark_return

                # --- DISPLAY ---
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Portfolio Return", format_pct(total_return))
                c2.metric("Benchmark (^GSPC)", format_pct(benchmark_return))
                c3.metric("Outperformance", format_pct(performance))
                c4.metric("Volatility", format_pct(volatility_port))

                st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                # --- CHART ---
                st.subheader("Portfolio vs Benchmark")

                cumulative_port = (1 + portfolio_returns).cumprod()
                cumulative_bench = (1 + benchmark_returns).cumprod()

                chart_df = pd.DataFrame({
                    "Portfolio": cumulative_port,
                    "S&P 500 (^GSPC)": cumulative_bench
                })

                st.line_chart(chart_df)

                # --- INTERPRETATION ---
                st.subheader("Interpretation")

                if performance > 0:
                    st.success("Portfolio outperformed the market")
                else:
                    st.warning("Portfolio underperformed the market")

                if volatility_port > volatility(benchmark_returns):
                    st.write("Portfolio is more volatile (riskier)")
                else:
                    st.write("Portfolio is less volatile (safer)")

                st.write("Higher Sharpe ratio = better risk-adjusted return")

        except:
            st.error("Error processing portfolio data.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Built with Python, Streamlit, and Yahoo Finance data")

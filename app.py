# app.py
# Stock Analytics and Portfolio Dashboard
# Author: Your Name
# Streamlit version

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------

def calculate_moving_averages(df, windows=[20,50]):
    """Calculate moving averages for given windows"""
    for window in windows:
        df[f"MA_{window}"] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

def calculate_volatility(df, window=20):
    """Calculate annualized volatility based on returns"""
    df['Returns'] = df['Close'].pct_change()
    rolling_std = df['Returns'].rolling(window=window).std()
    df['Volatility'] = rolling_std * np.sqrt(252)  # Annualized
    return df

def trend_signal(df):
    """Determine trend based on price and moving averages"""
    price = df['Close'].iloc[-1]
    ma20 = df['MA_20'].iloc[-1]
    ma50 = df['MA_50'].iloc[-1]
    
    if price > ma20 > ma50:
        return "Strong Uptrend"
    elif price < ma20 < ma50:
        return "Strong Downtrend"
    else:
        return "Mixed Trend"

def rsi_signal(df):
    """Generate RSI signal"""
    rsi = df['RSI'].iloc[-1]
    if rsi > 70:
        return "Overbought (Sell)"
    elif rsi < 30:
        return "Oversold (Buy)"
    else:
        return "Neutral"

def volatility_level(df):
    """Classify volatility"""
    vol = df['Volatility'].iloc[-1]
    if vol > 0.40:
        return "High"
    elif 0.25 <= vol <= 0.40:
        return "Medium"
    else:
        return "Low"

def portfolio_metrics(portfolio_weights, price_data, benchmark_data):
    """Calculate portfolio performance metrics"""
    returns = price_data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    portfolio_returns = (returns * portfolio_weights).sum(axis=1)
    total_return = (1 + portfolio_returns).prod() - 1
    benchmark_return = (1 + benchmark_returns).prod() - 1
    excess_return = total_return - benchmark_return
    annualized_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    
    metrics = {
        "Total Return": total_return,
        "Benchmark Return": benchmark_return,
        "Excess Return": excess_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio
    }
    return metrics, portfolio_returns, benchmark_returns

# ---------------------------------------------
# Streamlit App Layout
# ---------------------------------------------

st.set_page_config(page_title="Stock Analytics & Portfolio Dashboard", layout="wide")

# Title
st.title("📊 Stock Analytics and Portfolio Dashboard")

# Sidebar - Navigation
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Individual Stock Analysis", "Portfolio Dashboard"])

# ---------------------------------------------
# Part 1: Individual Stock Analysis
# ---------------------------------------------

if section == "Individual Stock Analysis":
    st.header("🔹 Individual Stock Analysis")
    
    # Input
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", value="AAPL").upper()
    
    if stock_symbol:
        # Data download
        end_date = datetime.today()
        start_date = end_date - timedelta(days=180)
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if not df.empty:
            df = calculate_moving_averages(df)
            df = calculate_rsi(df)
            df = calculate_volatility(df)
            
            # Display Data
            st.subheader("Latest Data")
            st.dataframe(df.tail(5))
            
            # Charts
            st.subheader("📈 Price and Moving Averages")
            plt.figure(figsize=(10,5))
            plt.plot(df['Close'], label='Close Price')
            plt.plot(df['MA_20'], label='20-Day MA')
            plt.plot(df['MA_50'], label='50-Day MA')
            plt.legend()
            st.pyplot(plt)
            
            # Trend
            trend = trend_signal(df)
            rsi_status = rsi_signal(df)
            vol_status = volatility_level(df)
            
            st.subheader("Analysis")
            st.write(f"**Current Price:** ${df['Close'].iloc[-1]:.2f}")
            st.write(f"**Trend:** {trend}")
            st.write(f"**RSI Signal:** {rsi_status}")
            st.write(f"**Volatility Level:** {vol_status}")
            
            # Recommendation
            st.subheader("💡 Trading Recommendation")
            if trend == "Strong Uptrend" and rsi_status == "Oversold (Buy)":
                recommendation = "Buy – strong trend and oversold"
            elif trend == "Strong Downtrend" and rsi_status == "Overbought (Sell)":
                recommendation = "Sell – downtrend and overbought"
            else:
                recommendation = "Hold – mixed signals"
            st.write(recommendation)
        else:
            st.error("Invalid stock symbol or no data available.")

# ---------------------------------------------
# Part 2: Portfolio Performance Dashboard
# ---------------------------------------------

if section == "Portfolio Dashboard":
    st.header("🔹 Portfolio Performance Dashboard")
    
    # Portfolio input
    st.subheader("Portfolio Setup")
    tickers = []
    weights = []
    
    cols = st.columns(5)
    for i in range(5):
        tickers.append(cols[i].text_input(f"Stock {i+1}", value=f"AAPL" if i==0 else f"MSFT" if i==1 else f"GOOGL" if i==2 else f"AMZN" if i==3 else f"TSLA"))
        weights.append(cols[i].number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0, value=0.2, step=0.05))
    
    if round(sum(weights),2) != 1.0:
        st.warning("Weights must sum to 1.0")
    
    benchmark_symbol = st.text_input("Benchmark ETF (default SPY):", value="SPY").upper()
    
    if st.button("Calculate Portfolio Performance"):
        try:
            start_date = datetime.today() - timedelta(days=365)
            end_date = datetime.today()
            
            price_data = pd.DataFrame()
            for ticker in tickers:
                data = yf.download(ticker, start=start_date, end=end_date)['Close']
                price_data[ticker] = data
            
            benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)['Close']
            
            metrics, port_returns, bench_returns = portfolio_metrics(np.array(weights), price_data, benchmark_data)
            
            st.subheader("Portfolio Metrics")
            st.write(pd.DataFrame(metrics, index=[0]).T.rename(columns={0:"Value"}))
            
            st.subheader("📈 Portfolio vs Benchmark")
            plt.figure(figsize=(10,5))
            (1+port_returns).cumprod().plot(label="Portfolio")
            (1+bench_returns).cumprod().plot(label="Benchmark")
            plt.legend()
            st.pyplot(plt)
            
            st.subheader("Interpretation")
            if metrics["Excess Return"] > 0:
                st.write("✅ Portfolio outperformed the benchmark.")
            else:
                st.write("⚠️ Portfolio underperformed the benchmark.")
            
            st.write(f"Annualized Volatility: {metrics['Annualized Volatility']:.2%}")
            st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            if metrics['Sharpe Ratio'] > 1:
                st.write("The portfolio is efficient based on Sharpe ratio.")
            else:
                st.write("The portfolio may not be efficient based on Sharpe ratio.")
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# ---------------------------------------------
# End of App
# ---------------------------------------------

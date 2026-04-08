# ============================================================
# FINANCIAL DATA ANALYTICS APP — Built with Streamlit
# ============================================================
# Covers:
#   Part 1 — Individual Stock Analysis
#   Part 2 — Portfolio Performance Dashboard
#
# Dependencies:
#   pip install streamlit yfinance pandas numpy matplotlib
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG — must be the very first Streamlit command
# layout="wide" uses the full screen width — good for finance apps
# ============================================================
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS STYLING
# st.markdown() lets us inject raw HTML or CSS into the page.
# unsafe_allow_html=True is required to render actual HTML.
# Fonts: DM Sans (body) + DM Mono (numbers) — same as template
# ============================================================
st.markdown("""
<style>
    /* Import a clean, professional font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

    /* Apply font to the whole app */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Style the big metric number cards ── */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 4px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 600;
        color: #0f172a;
        font-family: 'DM Mono', monospace;
    }
    .metric-value.highlight { color: #2563eb; }
    .metric-value.green     { color: #16a34a; }
    .metric-value.red       { color: #dc2626; }
    .metric-value.amber     { color: #d97706; }
    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* ── Signal / badge pills ─────────────────────────────── */
    .badge {
        display: inline-block;
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        letter-spacing: 0.05em;
    }
    .badge-buy     { background: #dcfce7; color: #16a34a; border: 1px solid #86efac; }
    .badge-sell    { background: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
    .badge-hold    { background: #fef9c3; color: #ca8a04; border: 1px solid #fde68a; }
    .badge-neutral { background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }

    /* ── Tip / insight box — .tip-box ─────── */
    .tip-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        font-size: 15px;
        color: #1e3a5f;
        line-height: 1.7;
    }
    .tip-box .verdict {
        font-size: 1.05rem;
        color: #2563eb;
        font-weight: 600;
    }

    /* ── Allocation / data rows — matches template .alloc-row  */
    .alloc-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 15px;
    }
    .alloc-label { color: #374151; }
    .alloc-value { font-weight: 600; color: #0f172a; font-family: 'DM Mono', monospace; }

    /* ── Table ────────────────────────────────────────────── */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    .styled-table th {
        background: #f8fafc;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.72rem;
        padding: 0.65rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        text-align: left;
        font-family: 'DM Sans', sans-serif;
    }
    .styled-table td {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #f1f5f9;
        color: #374151;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
    }
    .styled-table tr:last-child td { border-bottom: none; }
    .styled-table tr:hover td { background: #f8fafc; }

    /* ── Section intro — .section-intro ───── */
    .section-intro {
        font-size: 16px;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.7;
    }

    /* ── Make Streamlit's built-in metric ───── */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MATPLOTLIB THEME 
# ============================================================
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#f8fafc",
    "axes.edgecolor":    "#e2e8f0",
    "axes.labelcolor":   "#64748b",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        "#e2e8f0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.8,
    "text.color":        "#0f172a",
    "font.family":       "sans-serif",
    "figure.dpi":        130,
})


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt_pct(v):
    """Format as percentage string, e.g. 0.1234 → '+12.34%'"""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"

def fmt_money(v):
    """Format as dollar string with two decimal places."""
    return f"${v:,.2f}"

@st.cache_data(ttl=300)   # Cache 5 minutes so repeated runs don't re-hit Yahoo Finance
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.
    Returns a DataFrame with a DatetimeIndex.
    period examples: '6mo', '1y'
    """
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    
    if df.empty:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    
    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Classic Wilder RSI.
    delta   = daily price change
    gain    = positive deltas only (losses set to 0)
    loss    = absolute negative deltas only (gains set to 0)
    avg_gain / avg_loss = exponential moving average over `window` periods
    RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def trend_label(price, ma20, ma50):
    """Classify price trend based on MA crossover logic."""
    if price > ma20 > ma50:
        return "Strong Uptrend", "green"
    elif price < ma20 < ma50:
        return "Strong Downtrend", "red"
    else:
        return "Mixed / Sideways", "amber"


def rsi_signal(rsi_val):
    """Map RSI value to a trading signal."""
    if rsi_val > 70:
        return "Overbought — Possible Sell", "red"
    elif rsi_val < 30:
        return "Oversold — Possible Buy", "green"
    else:
        return "Neutral", "neutral"


def volatility_label(vol):
    """Classify annualized volatility into three buckets."""
    if vol > 0.40:
        return "High  (>40%)", "red"
    elif vol >= 0.25:
        return "Medium  (25–40%)", "amber"
    else:
        return "Low  (<25%)", "green"


def final_recommendation(trend_col, rsi_col, vol_col):
    """
    Simple rule-based recommendation engine.
    Weights trend (primary), RSI (secondary), volatility (risk modifier).
    Returns 'BUY', 'SELL', or 'HOLD' plus an explanation string.
    """
    bullish, bearish = 0, 0
    notes = []

    if trend_col == "green":
        bullish += 2
        notes.append("Price is above both moving averages (uptrend).")
    elif trend_col == "red":
        bearish += 2
        notes.append("Price is below both moving averages (downtrend).")
    else:
        notes.append("Trend is mixed — no clear directional edge.")

    if rsi_col == "green":
        bullish += 1
        notes.append("RSI signals oversold conditions — potential bounce.")
    elif rsi_col == "red":
        bearish += 1
        notes.append("RSI signals overbought — caution on new entries.")

    if vol_col == "red":
        notes.append("High volatility adds risk; size positions carefully.")

    if bullish > bearish:
        rec = "BUY"
    elif bearish > bullish:
        rec = "SELL"
    else:
        rec = "HOLD"

    return rec, " ".join(notes)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.045) -> float:
    """
    Annualized Sharpe Ratio.
    risk_free default = 4.5% (approximate current T-bill rate).
    Multiply by √252 to annualize daily excess returns.
    """
    excess = returns - risk_free / 252
    if returns.std() == 0:
        return 0.0
    return (excess.mean() / returns.std()) * np.sqrt(252)


# ============================================================
# APP TITLE & NAVIGATION
# st.title() shows a big heading
# st.tabs() creates clickable tab panels — one per section
# ============================================================
st.title("📊 Stock Analytics Dashboard")
st.markdown("*Apply real-world financial data analysis to individual stocks and multi-asset portfolios.*")

tab1, tab2 = st.tabs([
    "🏦  Part 1 — Individual Stock Analysis",
    "📈  Part 2 — Portfolio Dashboard"
])


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 1 — INDIVIDUAL STOCK ANALYSIS                     ║
# ╚══════════════════════════════════════════════════════════╝
with tab1:

    st.header("Individual Stock Analysis")
    st.markdown("""
    <p class="section-intro">
    Enter any stock ticker to retrieve 6 months of real market data from Yahoo Finance.
    The app will calculate trend signals using moving averages, measure momentum with RSI,
    quantify risk through volatility, and generate a final trading recommendation.
    </p>
    """, unsafe_allow_html=True)

    # ── Step 1: Data Collection inputs ──────────────────────
    st.subheader("Step 1 — Data Collection")

    col_in1, col_in2 = st.columns([2, 1])
    with col_in1:
        ticker_input = st.text_input(
            "Enter a stock ticker (e.g. AAPL, TSLA, NVDA, MSFT)",
            value="AAPL",
            max_chars=10
        ).upper().strip()
    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("▶  Run Analysis", key="run1")

    if run_analysis:

        with st.spinner(f"Downloading 6 months of data for {ticker_input} from Yahoo Finance…"):
            try:
                df = fetch_data(ticker_input, "6mo")
            except Exception as e:
                st.error(f"Could not download data for '{ticker_input}'. Check the ticker and try again.")
                st.stop()

        if df.empty:
            st.error("No data returned for '{ticker_input}'. Try a different ticker.")
            st.stop()

        close = df["Close"]

        # ── Step 2: Trend Analysis ───────────────────────────
        # Current price and the two rolling moving averages
        current_price = float(close.iloc[-1])
        start_price   = float(close.iloc[0])
        period_return = (current_price - start_price) / start_price

        # rolling(n).mean() calculates the average of the last n closing prices
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        trend_text, trend_col = trend_label(current_price, ma20, ma50)

        # ── Step 3: RSI ──────────────────────────────────────
        rsi_series        = compute_rsi(close, 14)
        rsi_val           = float(rsi_series.iloc[-1])
        rsi_text, rsi_col = rsi_signal(rsi_val)

        # ── Step 4: Volatility ───────────────────────────────
        # Daily log returns, then annualize by multiplying std by √252
        # (252 = average number of trading days in a year)
        daily_returns  = np.log(close / close.shift(1)).dropna()
        rolling_20_vol = daily_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        vol_text, vol_col = volatility_label(rolling_20_vol)

        # ── Step 5: Recommendation ───────────────────────────
        rec, explanation = final_recommendation(trend_col, rsi_col, vol_col)
        badge_cls = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[rec]

        # ── Display: Snapshot metrics ─────────────────────────
        st.subheader(f"{ticker_input} — 6-Month Snapshot")
        pct_sign = "+" if period_return >= 0 else ""

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value highlight">{fmt_money(current_price)}</div>
                <div class="metric-sub">{pct_sign}{period_return*100:.2f}% over 6 months</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">20-Day MA</div>
                <div class="metric-value">{fmt_money(ma20)}</div>
                <div class="metric-sub">Short-term trend</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">50-Day MA</div>
                <div class="metric-value">{fmt_money(ma50)}</div>
                <div class="metric-sub">Medium-term trend</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">14-Day RSI</div>
                <div class="metric-value {rsi_col}">{rsi_val:.1f}</div>
                <div class="metric-sub">{rsi_text.split('—')[0].strip()}</div>
            </div>""", unsafe_allow_html=True)

        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">20-Day Vol (Ann.)</div>
                <div class="metric-value {vol_col}">{rolling_20_vol*100:.1f}%</div>
                <div class="metric-sub">{vol_text.split('(')[0].strip()}</div>
            </div>""", unsafe_allow_html=True)

        # ── Display: Signal summary row ───────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        sig1, sig2, sig3 = st.columns(3)

        with sig1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Trend Signal</div>
                <div class="metric-value {trend_col}" style="font-size:1.1rem; margin-bottom:0.4rem;">
                    {trend_text}
                </div>
                <div class="metric-sub">
                    Price {fmt_money(current_price)} · 20MA {fmt_money(ma20)} · 50MA {fmt_money(ma50)}
                </div>
            </div>""", unsafe_allow_html=True)

        with sig2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Momentum (RSI)</div>
                <div class="metric-value {rsi_col}" style="font-size:1.1rem; margin-bottom:0.4rem;">
                    {rsi_text}
                </div>
                <div class="metric-sub">RSI = {rsi_val:.1f} · thresholds: 30 oversold / 70 overbought</div>
            </div>""", unsafe_allow_html=True)

        with sig3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Recommendation</div>
                <span class="badge {badge_cls}" style="font-size:1rem; padding:0.4rem 1.2rem;">{rec}</span>
                <div class="metric-sub" style="margin-top:0.5rem;">{explanation}</div>
            </div>""", unsafe_allow_html=True)

        # ── Chart 1: Price + Moving Averages ──────────────────
        st.subheader("Price Chart with Moving Averages")
        st.caption("Closing price alongside 20-day and 50-day moving averages — used to identify trend direction")

        fig1, ax1 = plt.subplots(figsize=(12, 4.5))
        dates = close.index
        ax1.fill_between(dates, close, alpha=0.06, color="#2563eb")
        ax1.plot(dates, close,                    color="#2563eb", lw=1.8, label="Close Price")
        ax1.plot(dates, close.rolling(20).mean(), color="#f59e0b", lw=1.4, linestyle="--", label="20-Day MA")
        ax1.plot(dates, close.rolling(50).mean(), color="#7c3aed", lw=1.4, linestyle="--", label="50-Day MA")
        ax1.set_title(f"{ticker_input} — Daily Closing Price (6 Months)", fontsize=11, pad=12, color="#64748b")
        ax1.set_ylabel("Price (USD)", fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.legend(framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
        ax1.grid(True)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ── Chart 2: RSI ──────────────────────────────────────
        st.subheader("RSI — 14-Day Relative Strength Index")
        st.caption("Values above 70 indicate overbought conditions; below 30 indicates oversold")

        fig2, ax2 = plt.subplots(figsize=(12, 2.8))
        rsi_dates = rsi_series.dropna().index
        ax2.plot(rsi_dates, rsi_series.dropna(), color="#2563eb", lw=1.5, label="RSI")
        ax2.axhline(70, color="#dc2626", lw=1.1, linestyle="--", label="Overbought (70)")
        ax2.axhline(30, color="#16a34a", lw=1.1, linestyle="--", label="Oversold (30)")
        ax2.axhline(50, color="#cbd5e1", lw=0.8, linestyle=":")
        ax2.fill_between(rsi_dates, rsi_series.dropna(), 70,
                         where=rsi_series.dropna() > 70, alpha=0.12, color="#dc2626")
        ax2.fill_between(rsi_dates, rsi_series.dropna(), 30,
                         where=rsi_series.dropna() < 30, alpha=0.12, color="#16a34a")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", fontsize=9)
        ax2.set_title(f"{ticker_input} — RSI (14-Day)", fontsize=11, pad=10, color="#64748b")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.legend(framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
        ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Written Interpretation ─────────────────────────────
        st.subheader("Written Interpretation")

        trend_prose = {
            "green": f"The price is above both the 20-day MA ({fmt_money(ma20)}) and 50-day MA ({fmt_money(ma50)}), "
                     "forming a classic <strong>strong uptrend</strong> structure. Buyers are in control "
                     "and momentum is positive.",
            "red":   f"The price is below both the 20-day MA ({fmt_money(ma20)}) and 50-day MA ({fmt_money(ma50)}), "
                     "signalling a <strong>strong downtrend</strong>. Sellers dominate and the path of least "
                     "resistance is lower.",
            "amber": f"The price ({fmt_money(current_price)}) sits between or around its moving averages, indicating "
                     "a <strong>mixed / sideways market</strong>. No clear directional bias is present — "
                     "wait for a decisive breakout before committing."
        }[trend_col]

        rsi_prose = {
            "red":     f"With an RSI of <strong>{rsi_val:.1f}</strong>, the stock is in <strong>overbought</strong> "
                       "territory (above 70). This does not guarantee an immediate reversal but warns that the "
                       "rally may be extended and a pullback is possible.",
            "green":   f"With an RSI of <strong>{rsi_val:.1f}</strong>, the stock is in <strong>oversold</strong> "
                       "territory (below 30). Historically, oversold readings have preceded rebounds, "
                       "presenting a potential contrarian entry opportunity.",
            "neutral": f"RSI is at <strong>{rsi_val:.1f}</strong>, sitting in neutral territory (30–70). "
                       "No strong momentum extreme in either direction — the signal is inconclusive on its own."
        }[rsi_col]

        vol_prose = {
            "red":   f"Annualized volatility of <strong>{rolling_20_vol*100:.1f}%</strong> is <strong>high</strong> "
                     "(above 40%). Expect large daily swings. Risk management through smaller position sizing "
                     "or tighter stop-losses is strongly recommended.",
            "amber": f"Annualized volatility of <strong>{rolling_20_vol*100:.1f}%</strong> is <strong>moderate</strong> "
                     "(25–40%). The stock moves meaningfully but is not unusually wild. Standard position sizing applies.",
            "green": f"Annualized volatility of <strong>{rolling_20_vol*100:.1f}%</strong> is <strong>low</strong> "
                     "(below 25%). This is a relatively stable mover — suitable for risk-averse investors "
                     "or larger position sizes."
        }[vol_col]

        st.markdown(f"""
        <div class="tip-box">
            <strong>What does the trend suggest?</strong><br>
            {trend_prose}
            <br><br>
            <strong>What did RSI indicate?</strong><br>
            {rsi_prose}
            <br><br>
            <strong>What did volatility suggest?</strong><br>
            {vol_prose}
            <br><br>
            <strong>Final Recommendation:</strong>
            <span class="verdict">{rec}</span> — {explanation}
        </div>
        """, unsafe_allow_html=True)

        # ── Raw data table (expandable) ───────────────────────
        with st.expander("📋  View Raw OHLCV Data (last 30 trading days)"):
            display_df = df.copy()
            display_df.index = display_df.index.strftime("%Y-%m-%d")
            st.dataframe(
                display_df[["Open","High","Low","Close","Volume"]].tail(30).round(2),
                use_container_width=True
            )


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 2 — PORTFOLIO PERFORMANCE DASHBOARD               ║
# ╚══════════════════════════════════════════════════════════╝
with tab2:

    st.header("Portfolio Performance Dashboard")
    st.markdown("""
    <p class="section-intro">
    Build a custom 5-stock portfolio and compare its 1-year performance against a benchmark ETF.
    The dashboard calculates total return, annualized volatility, Sharpe ratio, and max drawdown
    — the same metrics used by professional portfolio managers.
    </p>
    """, unsafe_allow_html=True)

    # ── Step 1: Portfolio Setup ───────────────────────────────
    st.subheader("Step 1 — Portfolio Setup")
    st.caption("Enter 5 stock tickers and assign weights that sum to 1.00")

    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    DEFAULT_WEIGHTS = [0.25,    0.25,   0.20,    0.15,   0.15]
    DEFAULT_BENCH   = "SPY"

    port_cols = st.columns(5)
    tickers, weights = [], []

    for i, col in enumerate(port_cols):
        with col:
            t = st.text_input(f"Ticker {i+1}", value=DEFAULT_TICKERS[i], key=f"pt{i}").upper().strip()
            w = st.number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0,
                                value=DEFAULT_WEIGHTS[i], step=0.05, key=f"pw{i}")
            tickers.append(t)
            weights.append(w)

    bench_col, run_col = st.columns([2, 1])
    with bench_col:
        benchmark = st.text_input("Benchmark Ticker (e.g. SPY, QQQ, IWM)", value=DEFAULT_BENCH).upper().strip()
    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_portfolio = st.button("▶  Analyze Portfolio", key="run2")

    # Weight validation — remind user if weights don't sum to 1.00
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"⚠️ Weights currently sum to {total_weight:.2f}. Please adjust so they total 1.00.")

    if run_portfolio or True:   # auto-run on load with defaults

        with st.spinner("Downloading 1 year of price data for all tickers…"):
            try:
                price_data = {}
                for tk in tickers + [benchmark]:
                    raw = fetch_data(tk, "1y")
                    if not raw.empty:
                        price_data[tk] = raw["Close"]
            except Exception as e:
                st.error(f"Data download failed: {e}")
                st.stop()

        # Build aligned price DataFrame and drop rows with any NaN
        prices_df = pd.DataFrame(price_data).dropna()

        if prices_df.empty or len(prices_df) < 50:
            st.error("Not enough data. Check your tickers and try again.")
            st.stop()

        # ── Step 4: Return Calculations ──────────────────────
        # pct_change() computes daily percentage returns for each column
        returns_df = prices_df.pct_change().dropna()

        # Weighted portfolio daily returns: multiply each stock's return by its weight, then sum
        port_rets  = (returns_df[tickers] * weights).sum(axis=1)
        bench_rets = returns_df[benchmark]

        # Cumulative growth of $1 invested at the start
        port_cum  = (1 + port_rets).cumprod()
        bench_cum = (1 + bench_rets).cumprod()

        # ── Step 5: Performance Metrics ──────────────────────
        total_port_ret  = float(port_cum.iloc[-1]  - 1)
        total_bench_ret = float(bench_cum.iloc[-1] - 1)
        outperform      = total_port_ret - total_bench_ret

        # Annualized volatility = daily std × √252
        port_vol   = float(port_rets.std()  * np.sqrt(252))
        bench_vol  = float(bench_rets.std() * np.sqrt(252))

        port_sharpe  = sharpe_ratio(port_rets)
        bench_sharpe = sharpe_ratio(bench_rets)

        # Max drawdown = largest peak-to-trough decline over the period
        def max_drawdown(cum_ret_series):
            roll_max = cum_ret_series.cummax()
            drawdown = (cum_ret_series - roll_max) / roll_max
            return float(drawdown.min())

        port_dd  = max_drawdown(port_cum)
        bench_dd = max_drawdown(bench_cum)

        # ── Display: Performance Summary ──────────────────────
        st.subheader("Performance Summary")

        out_col  = "green" if outperform >= 0 else "red"
        out_sign = "+" if outperform >= 0 else ""

        pm1, pm2, pm3, pm4 = st.columns(4)

        with pm1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Return</div>
                <div class="metric-value highlight">{fmt_pct(total_port_ret)}</div>
                <div class="metric-sub">1-year total return</div>
            </div>""", unsafe_allow_html=True)

        with pm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{benchmark} Return</div>
                <div class="metric-value">{fmt_pct(total_bench_ret)}</div>
                <div class="metric-sub">Benchmark (1-year)</div>
            </div>""", unsafe_allow_html=True)

        with pm3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">vs Benchmark</div>
                <div class="metric-value {out_col}">{out_sign}{outperform*100:.2f}%</div>
                <div class="metric-sub">{'Outperformed ✓' if outperform >= 0 else 'Underperformed ✗'}</div>
            </div>""", unsafe_allow_html=True)

        with pm4:
            sharpe_col = "green" if port_sharpe >= 1.0 else "amber"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Sharpe</div>
                <div class="metric-value {sharpe_col}">{port_sharpe:.2f}</div>
                <div class="metric-sub">vs {benchmark}: {bench_sharpe:.2f}</div>
            </div>""", unsafe_allow_html=True)

        pm5, pm6, pm7, pm8 = st.columns(4)

        vol_col_cls = "red" if port_vol > bench_vol else "green"
        with pm5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Volatility</div>
                <div class="metric-value {vol_col_cls}">{port_vol*100:.1f}%</div>
                <div class="metric-sub">Annualized</div>
            </div>""", unsafe_allow_html=True)

        with pm6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{benchmark} Volatility</div>
                <div class="metric-value">{bench_vol*100:.1f}%</div>
                <div class="metric-sub">Annualized</div>
            </div>""", unsafe_allow_html=True)

        with pm7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value red">{port_dd*100:.1f}%</div>
                <div class="metric-sub">Portfolio worst peak-to-trough</div>
            </div>""", unsafe_allow_html=True)

        with pm8:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{benchmark} Max Drawdown</div>
                <div class="metric-value">{bench_dd*100:.1f}%</div>
                <div class="metric-sub">Benchmark worst drawdown</div>
            </div>""", unsafe_allow_html=True)

        # ── Chart: Cumulative Returns ──────────────────────────
        st.subheader("Cumulative Growth — Portfolio vs Benchmark")
        st.caption("Starting value = $1.00 invested at the beginning of the 1-year period")

        fig3, ax3 = plt.subplots(figsize=(12, 4.5))
        ax3.plot(port_cum.index,  port_cum,  color="#2563eb", lw=2.0,  label="Portfolio")
        ax3.plot(bench_cum.index, bench_cum, color="#7c3aed", lw=1.5,  linestyle="--", label=benchmark)
        ax3.fill_between(port_cum.index, port_cum, bench_cum,
                         where=port_cum >= bench_cum, alpha=0.08, color="#16a34a", label="Outperforms")
        ax3.fill_between(port_cum.index, port_cum, bench_cum,
                         where=port_cum <  bench_cum, alpha=0.08, color="#dc2626", label="Underperforms")
        ax3.axhline(1.0, color="#cbd5e1", lw=0.8, linestyle=":")
        ax3.set_title("Portfolio vs Benchmark — Cumulative Return (1 Year)", fontsize=11, pad=12, color="#64748b")
        ax3.set_ylabel("Growth of $1", fontsize=9)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax3.legend(framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
        ax3.grid(True)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ── Chart: Individual stock returns ────────────────────
        st.subheader("Individual Stock Returns (1 Year)")
        st.caption("Each bar shows the total return for that stock; dashed line marks the benchmark")

        ind_returns = {t: float((1 + returns_df[t]).cumprod().iloc[-1] - 1) for t in tickers}
        colors_bar  = ["#16a34a" if v >= 0 else "#dc2626" for v in ind_returns.values()]

        fig4, ax4 = plt.subplots(figsize=(9, 3.5))
        bars = ax4.bar(ind_returns.keys(), [v * 100 for v in ind_returns.values()],
                       color=colors_bar, width=0.5, edgecolor="#ffffff", linewidth=1.2)
        for bar, val in zip(bars, ind_returns.values()):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.8 if val >= 0 else -3),
                     f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9,
                     color="#16a34a" if val >= 0 else "#dc2626")
        ax4.axhline(total_bench_ret * 100, color="#7c3aed", lw=1.3, linestyle="--",
                    label=f"{benchmark} = {total_bench_ret*100:.1f}%")
        ax4.set_title("Individual Stock Total Returns vs Benchmark", fontsize=11, pad=10, color="#64748b")
        ax4.set_ylabel("Total Return (%)", fontsize=9)
        ax4.legend(framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
        ax4.grid(True, axis="y")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # ── Holdings summary table ─────────────────────────────
        st.subheader("Portfolio Holdings Summary")

        rows = []
        for t, w in zip(tickers, weights):
            if t not in returns_df.columns:
                continue
            ret = float((1 + returns_df[t]).cumprod().iloc[-1] - 1)
            vol = float(returns_df[t].std() * np.sqrt(252))
            shr = sharpe_ratio(returns_df[t])
            rows.append({
                "Ticker": t,
                "Weight": f"{w*100:.0f}%",
                "1-Yr Return": fmt_pct(ret),
                "Ann. Volatility": f"{vol*100:.1f}%",
                "Sharpe Ratio": f"{shr:.2f}",
                "Weighted Contrib.": fmt_pct(ret * w),
            })

        html_rows = ""
        for r in rows:
            ret_val   = float(r["1-Yr Return"].replace("+","").replace("%","")) / 100
            ret_color = "#16a34a" if ret_val >= 0 else "#dc2626"
            html_rows += f"""
            <tr>
                <td style="color:#2563eb; font-weight:600;">{r['Ticker']}</td>
                <td>{r['Weight']}</td>
                <td style="color:{ret_color};">{r['1-Yr Return']}</td>
                <td>{r['Ann. Volatility']}</td>
                <td>{r['Sharpe Ratio']}</td>
                <td style="color:{ret_color};">{r['Weighted Contrib.']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Ticker</th><th>Weight</th><th>1-Yr Return</th>
                    <th>Ann. Volatility</th><th>Sharpe Ratio</th><th>Weighted Contrib.</th>
                </tr>
            </thead>
            <tbody>{html_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── Step 6: Written Interpretation ────────────────────
        st.subheader("Step 6 — Written Interpretation")

        out_text = (f"outperformed {benchmark} by <strong>{fmt_pct(abs(outperform))}</strong>"
                    if outperform >= 0 else
                    f"underperformed {benchmark} by <strong>{fmt_pct(abs(outperform))}</strong>")

        risk_text = (f"<strong>more</strong> volatile than the benchmark "
                     f"({port_vol*100:.1f}% vs {bench_vol*100:.1f}%)"
                     if port_vol > bench_vol else
                     f"<strong>less</strong> volatile than the benchmark "
                     f"({port_vol*100:.1f}% vs {bench_vol*100:.1f}%)")

        sharpe_text = (
            "A Sharpe ratio above 1.0 is generally considered good — the portfolio "
            "is generating adequate return per unit of risk taken."
            if port_sharpe >= 1.0 else
            "A Sharpe ratio below 1.0 suggests the portfolio's return does not "
            "adequately compensate for its risk level. Consider rebalancing toward "
            "higher quality or lower-volatility positions."
        )

        takeaway = (
            "The portfolio demonstrates strong risk-adjusted performance. Continue "
            "monitoring individual position volatility and rebalance if any single weight "
            "drifts significantly from its target."
            if port_sharpe >= bench_sharpe and outperform >= 0 else
            "Consider reviewing the largest detractors and evaluating whether their "
            "risk contribution is justified by their expected return."
        )

        st.markdown(f"""
        <div class="tip-box">
            <strong>Did the portfolio outperform the benchmark?</strong><br>
            Over the past year, this portfolio returned <strong>{fmt_pct(total_port_ret)}</strong>
            vs the {benchmark} benchmark's <strong>{fmt_pct(total_bench_ret)}</strong>.
            The portfolio {out_text}.
            <br><br>
            <strong>Was it more or less risky?</strong><br>
            The portfolio was {risk_text}. The maximum drawdown reached
            <strong>{port_dd*100:.1f}%</strong> versus the benchmark's <strong>{bench_dd*100:.1f}%</strong>.
            {"Taking on additional risk was rewarded with higher returns." if port_vol > bench_vol and outperform > 0 else
             "Higher risk was not rewarded with proportionally higher returns — this is unfavorable." if port_vol > bench_vol and outperform <= 0 else
             "Achieving competitive returns with lower volatility is an attractive outcome."}
            <br><br>
            <strong>Was it efficient based on Sharpe ratio?</strong><br>
            The portfolio's Sharpe ratio is <strong>{port_sharpe:.2f}</strong>
            (benchmark: <strong>{bench_sharpe:.2f}</strong>). {sharpe_text}
            <br><br>
            <strong>Key Takeaway:</strong> {takeaway}
        </div>
        """, unsafe_allow_html=True)

        # ── Rolling correlation (expandable) ──────────────────
        with st.expander("📊  Rolling 30-Day Correlation: Portfolio vs Benchmark"):
            rolling_corr = port_rets.rolling(30).corr(bench_rets).dropna()
            fig5, ax5 = plt.subplots(figsize=(12, 3))
            ax5.plot(rolling_corr.index, rolling_corr, color="#2563eb", lw=1.5)
            ax5.axhline(0.7, color="#dc2626", lw=0.9, linestyle="--", label="0.70 threshold")
            ax5.set_ylim(-1, 1)
            ax5.set_title("30-Day Rolling Correlation — Portfolio vs Benchmark",
                          fontsize=10, pad=8, color="#64748b")
            ax5.set_ylabel("Correlation", fontsize=9)
            ax5.legend(framealpha=0.9, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=9)
            ax5.grid(True)
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
            st.caption("High correlation (>0.70) means the portfolio moves closely with the benchmark. "
                       "When correlation is high, the diversification benefit is limited.")


# ============================================================
# FOOTER
# st.divider() draws a horizontal line
# st.caption() shows small gray text
# ============================================================
st.divider()
st.caption("Built with Python & Streamlit · Real market data via Yahoo Finance · For educational purposes only · Not financial advice")

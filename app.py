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
# ============================================================
st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# APP TITLE & NAVIGATION
# st.title() shows a big heading
# ============================================================
st.title("📊 Stock Analytics Dashboard")
st.markdown("*Financial Analytics Dashboard - Real data from Yahoo Finance used.*")

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* Import a clean, professional font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* Apply font to the whole app */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Style the big metric number cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Card label — the small heading above the number */
    .card-label {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 6px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Card value — the big bold number */
    .card-value {
        font-size: 36px;
        font-weight: 700;
        color: #0f172a;
        font-family: 'DM Mono', monospace;
        line-height: 1.15;
        margin-bottom: 4px;
    }

    /* Color variants for card-value */
    .card-value.cyan    { color: #0284c7; }
    .card-value.green   { color: #16a34a; }
    .card-value.red     { color: #dc2626; }
    .card-value.amber   { color: #d97706; }
    .card-value.neutral { color: #475569; }

    /* Card sub — the small descriptive line below */
    .card-sub {
        font-size: 13px;
        font-weight: 500;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Card accent/danger variants */
    .metric-card.accent { border-color: #0284c7; background: #f0f9ff; }
    .metric-card.danger { border-color: #dc2626; background: #fff5f5; }
    .metric-card.warn   { border-color: #d97706; background: #fffbeb; }

    /* Signal / text cards (trend, RSI, recommendation) */
    .card-value.signal {
        font-size: 22px;
        font-weight: 700;
        line-height: 1.3;
    }

    /* Badge styles for BUY / SELL / HOLD */
    .badge {
        display: inline-block;
        border-radius: 8px;
        padding: 0.5rem 1.6rem;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-top: 6px;
    }
    .badge-buy  { background: #dcfce7; color: #15803d; border: 2px solid #16a34a; }
    .badge-sell { background: #fee2e2; color: #b91c1c; border: 2px solid #dc2626; }
    .badge-hold { background: #fef9c3; color: #92400e; border: 2px solid #d97706; }

    /* Style the tip/advice boxes */
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

    /* Insight box */
    .insight-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        font-size: 15px;
        color: #1e293b;
        line-height: 1.8;
    }

    /* Verdict inline */
    .verdict {
        display: inline-block;
        font-weight: 700;
        font-size: 17px;
        color: #0284c7;
        margin-left: 4px;
    }

    /* Styled table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 15px;
    }
    .styled-table th {
        background: #f1f5f9;
        color: #475569;
        font-weight: 700;
        padding: 10px 14px;
        text-align: left;
        font-size: 13px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .styled-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
    }
    .styled-table tr:last-child td { border-bottom: none; }

    /* Section tag */
    .section-tag {
        display: inline-block;
        background: #0284c7;
        color: white;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 3px 10px;
        border-radius: 4px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .section-title {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.25rem;
    }

    /* Divider */
    .styled-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }

    /* Make Streamlit's default metric look nicer */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }

    .section-intro {
        font-size: 16px;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt_pct(v):
    """Format as percentage string, e.g. 0.1234 → '+12.34%'"""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"

def fmt_money(v):
    """Format as dollar string."""
    return f"${v:,.2f}"

@st.cache_data(ttl=300)   # Cache 5 minutes so repeated runs don't re-hit Yahoo
def fetch_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.
    Returns a DataFrame with a DatetimeIndex.
    period examples: '6mo', '1y'
    """
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
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def trend_label(price, ma20, ma50):
    """Classify price trend based on MA crossover."""
    if price > ma20 > ma50:
        return "Strong Uptrend", "green"
    elif price < ma20 < ma50:
        return "Strong Downtrend", "red"
    else:
        return "Mixed", "amber"


def rsi_signal(rsi_val):
    """Map RSI value to trading signal."""
    if rsi_val > 70:
        return "Overbought — Possible Sell", "red"
    elif rsi_val < 30:
        return "Oversold — Possible Buy", "green"
    else:
        return "Neutral", "neutral"


def volatility_label(vol):
    """Classify annualized volatility."""
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
    Returns 'Buy', 'Sell', or 'Hold' plus an explanation string.
    """
    bullish = 0
    bearish = 0
    notes   = []

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
# TABS
# ============================================================
tab1, tab2 = st.tabs(["  PART 1 — Individual Stock  ", "  PART 2 — Portfolio Dashboard  "])


# ╔══════════════════════════════════════════════════════════╗
# ║  PART 1 — INDIVIDUAL STOCK ANALYSIS                     ║
# ╚══════════════════════════════════════════════════════════╝
with tab1:

    st.markdown('<div class="section-tag">Part 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Individual Stock Analysis</div>', unsafe_allow_html=True)

    # ── Inputs ──────────────────────────────────────────────
    col_in1, col_in2 = st.columns([2, 1])
    with col_in1:
        ticker_input = st.text_input(
            "Enter a stock ticker (e.g. AAPL, TSLA, NVDA)",
            value="AAPL",
            max_chars=10
        ).upper().strip()
    with col_in2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("Run Analysis", key="run1")

    if run_analysis or ticker_input:

        with st.spinner(f"Fetching 6 months of data for {ticker_input}…"):
            try:
                df = fetch_data(ticker_input, "6mo")
            except Exception as e:
                st.error(f"Could not download data for '{ticker_input}'. Check the ticker and try again.")
                st.stop()

        if df.empty or len(df) < 55:
            st.error("Not enough data returned. Try a different ticker or check your internet connection.")
            st.stop()

        close = df["Close"].squeeze()

        # ── STEP 1: Basic price info ─────────────────────────
        current_price = float(close.iloc[-1])
        start_price   = float(close.iloc[0])
        period_return = (current_price - start_price) / start_price

        # ── STEP 2: Moving averages ──────────────────────────
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        st.write(f"Price: {current_price}, MA20: {ma20}, MA50: {ma50}")
        st.write(f"price > ma20: {current_price > ma20}, ma20 > ma50: {ma20 > ma50}")
        trend_text, trend_col = trend_label(current_price, ma20, ma50)

        # ── STEP 3: RSI ──────────────────────────────────────
        rsi_series = compute_rsi(close, 14)
        rsi_val    = float(rsi_series.iloc[-1])
        rsi_text, rsi_col = rsi_signal(rsi_val)

        # ── STEP 4: Volatility ───────────────────────────────
        # Daily log returns, then annualize by multiplying std by √252
        daily_returns = np.log(close / close.shift(1)).dropna()
        rolling_20_vol = daily_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        vol_text, vol_col = volatility_label(rolling_20_vol)

        # ── STEP 5: Recommendation ───────────────────────────
        rec, explanation = final_recommendation(trend_col, rsi_col, vol_col)
        badge_cls = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[rec]

        # ── DISPLAY: Key metrics row ─────────────────────────
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown(f"### {ticker_input} — Snapshot")

        m1, m2, m3, m4, m5 = st.columns(5)
        price_col  = "green" if period_return >= 0 else "red"
        pct_sign   = "+" if period_return >= 0 else ""

        with m1:
            st.markdown(f"""
            <div class="metric-card {'accent' if period_return >= 0 else 'danger'}">
                <div class="card-label">Current Price</div>
                <div class="card-value cyan">{fmt_money(current_price)}</div>
                <div class="card-sub">{pct_sign}{period_return*100:.2f}% last 6 mo</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">20-Day MA</div>
                <div class="card-value">{fmt_money(ma20)}</div>
                <div class="card-sub">Short-term trend</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">50-Day MA</div>
                <div class="card-value">{fmt_money(ma50)}</div>
                <div class="card-sub">Medium-term trend</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            rsi_card_cls = {"green": "accent", "red": "danger", "neutral": ""}.get(rsi_col, "")
            st.markdown(f"""
            <div class="metric-card {rsi_card_cls}">
                <div class="card-label">14-Day RSI</div>
                <div class="card-value {rsi_col}">{rsi_val:.1f}</div>
                <div class="card-sub">{rsi_text.split('—')[0].strip()}</div>
            </div>""", unsafe_allow_html=True)

        with m5:
            vol_card_cls = {"green": "accent", "red": "danger", "amber": "warn"}.get(vol_col, "")
            st.markdown(f"""
            <div class="metric-card {vol_card_cls}">
                <div class="card-label">20-Day Vol (Ann.)</div>
                <div class="card-value {vol_col}">{rolling_20_vol*100:.1f}%</div>
                <div class="card-sub">{vol_text.split('(')[0].strip()}</div>
            </div>""", unsafe_allow_html=True)

        # ── DISPLAY: Trend + RSI + Rec row ───────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        sig1, sig2, sig3 = st.columns(3)

        with sig1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">Trend Signal</div>
                <div class="card-value signal {trend_col}" style="margin-bottom:0.4rem;">
                    {trend_text}
                </div>
                <div class="card-sub">
                    Price {fmt_money(current_price)} · 20MA {fmt_money(ma20)} · 50MA {fmt_money(ma50)}
                </div>
            </div>""", unsafe_allow_html=True)

        with sig2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">Momentum (RSI)</div>
                <div class="card-value signal {rsi_col}" style="margin-bottom:0.4rem;">
                    {rsi_text}
                </div>
                <div class="card-sub">RSI = {rsi_val:.1f} · threshold: 30 / 70</div>
            </div>""", unsafe_allow_html=True)

        with sig3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">Final Recommendation</div>
                <span class="badge {badge_cls}">
                    {rec}
                </span>
                <div class="card-sub" style="margin-top:0.6rem;">{explanation}</div>
            </div>""", unsafe_allow_html=True)

        # ── CHART 1: Price + Moving Averages ─────────────────
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("#### Price Chart with Moving Averages")

        fig1, ax1 = plt.subplots(figsize=(12, 4.5))
        dates = close.index

        ax1.fill_between(dates, close, alpha=0.08, color="#00e5c8")
        ax1.plot(dates, close,                    color="#00e5c8", lw=1.6, label="Close Price")
        ax1.plot(dates, close.rolling(20).mean(), color="#f59e0b", lw=1.3, linestyle="--", label="20-Day MA")
        ax1.plot(dates, close.rolling(50).mean(), color="#8b5cf6", lw=1.3, linestyle="--", label="50-Day MA")

        ax1.set_title(f"{ticker_input} — Daily Closing Price (6 Months)", fontsize=11, pad=12, color="#94a3b8")
        ax1.set_ylabel("Price (USD)", fontsize=9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.legend(framealpha=0.2, facecolor="#0f1923", edgecolor="#1e3a4a", fontsize=9)
        ax1.grid(True)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # ── CHART 2: RSI ─────────────────────────────────────
        st.markdown("#### RSI (14-Day Relative Strength Index)")

        fig2, ax2 = plt.subplots(figsize=(12, 2.8))
        rsi_dates = rsi_series.dropna().index

        ax2.plot(rsi_dates, rsi_series.dropna(), color="#00e5c8", lw=1.4, label="RSI")
        ax2.axhline(70, color="#ef4444", lw=1.0, linestyle="--", label="Overbought (70)")
        ax2.axhline(30, color="#22c55e", lw=1.0, linestyle="--", label="Oversold (30)")
        ax2.axhline(50, color="#4a7c8a", lw=0.7, linestyle=":")
        ax2.fill_between(rsi_dates, rsi_series.dropna(), 70,
                         where=rsi_series.dropna() > 70, alpha=0.15, color="#ef4444")
        ax2.fill_between(rsi_dates, rsi_series.dropna(), 30,
                         where=rsi_series.dropna() < 30, alpha=0.15, color="#22c55e")

        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", fontsize=9)
        ax2.set_title(f"{ticker_input} — RSI (14-Day)", fontsize=11, pad=10, color="#94a3b8")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.legend(framealpha=0.2, facecolor="#0f1923", edgecolor="#1e3a4a", fontsize=9)
        ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── WRITTEN INTERPRETATION ───────────────────────────
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("#### Written Interpretation")

        trend_prose = {
            "green": f"The price is above both the 20-day MA ({fmt_money(ma20)}) and 50-day MA ({fmt_money(ma50)}), "
                     "forming a classic <strong>strong uptrend</strong> structure. This suggests buyers are in control "
                     "and momentum is positive.",
            "red":   f"The price is below both the 20-day MA ({fmt_money(ma20)}) and 50-day MA ({fmt_money(ma50)}), "
                     "signalling a <strong>strong downtrend</strong>. Sellers dominate the tape and the path of least "
                     "resistance is lower.",
            "amber": f"The price ({fmt_money(current_price)}) sits between or around its moving averages, indicating "
                     "a <strong>mixed / sideways market</strong>. No clear directional bias is present; "
                     "wait for a decisive breakout."
        }[trend_col]

        rsi_prose = {
            "red":     f"With an RSI of <strong>{rsi_val:.1f}</strong>, the stock is in <strong>overbought</strong> "
                       "territory (above 70). This does not guarantee an immediate reversal but warns that the "
                       "rally may be extended and a pullback is possible.",
            "green":   f"With an RSI of <strong>{rsi_val:.1f}</strong>, the stock is in <strong>oversold</strong> "
                       "territory (below 30). Historically, oversold conditions at this level have preceded "
                       "rebounds, presenting a potential contrarian entry.",
            "neutral": f"RSI is at <strong>{rsi_val:.1f}</strong>, sitting comfortably in neutral territory "
                       "(30–70). No strong momentum extreme in either direction; the signal is inconclusive."
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
        <div class="insight-box">
            <strong>Trend Analysis</strong><br>{trend_prose}
            <br><br>
            <strong>RSI / Momentum</strong><br>{rsi_prose}
            <br><br>
            <strong>Volatility</strong><br>{vol_prose}
            <br><br>
            <strong>Final Verdict:</strong>
            <span class="verdict">{rec}</span> — {explanation}
        </div>
        """, unsafe_allow_html=True)

        # ── Raw data table (expandable) ───────────────────────
        with st.expander("📋  View Raw OHLCV Data"):
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

    st.markdown('<div class="section-tag">Part 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Portfolio Performance Dashboard</div>', unsafe_allow_html=True)

    # ── Portfolio Setup ──────────────────────────────────────
    st.markdown("#### Build Your Portfolio")
    st.caption("Enter 5 stock tickers and assign weights that sum to 1.00")

    DEFAULT_TICKERS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    DEFAULT_WEIGHTS  = [0.25,    0.25,   0.20,    0.15,   0.15]
    DEFAULT_BENCH    = "SPY"

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
        benchmark = st.text_input("Benchmark Ticker", value=DEFAULT_BENCH).upper().strip()
    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_portfolio = st.button("Analyze Portfolio", key="run2")

    # Weight validation
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.001:
        st.warning(f"Weights sum to {total_weight:.2f}. They should sum to 1.00. Please adjust.")

    if run_portfolio or True:   # auto-run on load with defaults

        with st.spinner("Downloading 1 year of price data…"):
            try:
                all_tickers = tickers + [benchmark]
                price_data  = {}
                for tk in all_tickers:
                    raw = fetch_data(tk, "1y")
                    if not raw.empty:
                        price_data[tk] = raw["Close"]
            except Exception as e:
                st.error(f"Data download failed: {e}")
                st.stop()

        # Build aligned price DataFrame
        prices_df = pd.DataFrame(price_data).dropna()

        if prices_df.empty or len(prices_df) < 50:
            st.error("Not enough data. Check tickers.")
            st.stop()

        # ── STEP 4: Returns ─────────────────────────────────
        returns_df = prices_df.pct_change().dropna()

        # Weighted portfolio daily returns
        port_rets    = (returns_df[tickers] * weights).sum(axis=1)
        bench_rets   = returns_df[benchmark]

        # Cumulative growth of $1
        port_cum  = (1 + port_rets).cumprod()
        bench_cum = (1 + bench_rets).cumprod()

        # ── STEP 5: Performance metrics ──────────────────────
        total_port_ret  = float(port_cum.iloc[-1]  - 1)
        total_bench_ret = float(bench_cum.iloc[-1] - 1)
        outperform      = total_port_ret - total_bench_ret

        port_vol   = float(port_rets.std()  * np.sqrt(252))
        bench_vol  = float(bench_rets.std() * np.sqrt(252))
        port_sharpe  = sharpe_ratio(port_rets)
        bench_sharpe = sharpe_ratio(bench_rets)

        # Max drawdown helper (kept for written interpretation)
        def max_drawdown(cum_ret_series):
            roll_max = cum_ret_series.cummax()
            drawdown = (cum_ret_series - roll_max) / roll_max
            return float(drawdown.min())

        port_dd  = max_drawdown(port_cum)
        bench_dd = max_drawdown(bench_cum)

        # ── DISPLAY: Summary metrics — Row 1 ─────────────────
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("#### Performance Summary")

        pm1, pm2, pm3, pm4 = st.columns(4)
        out_col = "green" if outperform >= 0 else "red"
        out_card = "accent" if outperform >= 0 else "danger"

        with pm1:
            st.markdown(f"""
            <div class="metric-card accent">
                <div class="card-label">Portfolio Return</div>
                <div class="card-value cyan">{fmt_pct(total_port_ret)}</div>
                <div class="card-sub">1-year total return</div>
            </div>""", unsafe_allow_html=True)

        with pm2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">{benchmark} Return</div>
                <div class="card-value">{fmt_pct(total_bench_ret)}</div>
                <div class="card-sub">Benchmark (1-year)</div>
            </div>""", unsafe_allow_html=True)

        with pm3:
            st.markdown(f"""
            <div class="metric-card {out_card}">
                <div class="card-label">vs Benchmark</div>
                <div class="card-value {out_col}">{fmt_pct(outperform)}</div>
                <div class="card-sub">{'Outperformed ✓' if outperform >= 0 else 'Underperformed ✗'}</div>
            </div>""", unsafe_allow_html=True)

        with pm4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">Portfolio Sharpe</div>
                <div class="card-value {'cyan' if port_sharpe >= 1 else 'amber'}">{port_sharpe:.2f}</div>
                <div class="card-sub">Risk-adjusted return</div>
            </div>""", unsafe_allow_html=True)

        # ── DISPLAY: Summary metrics — Row 2 ─────────────────
        pm5, pm6, pm7 = st.columns(3)

        with pm5:
            st.markdown(f"""
            <div class="metric-card {'danger' if port_vol > bench_vol else 'accent'}">
                <div class="card-label">Portfolio Volatility</div>
                <div class="card-value {'red' if port_vol > bench_vol else 'green'}">{port_vol*100:.1f}%</div>
                <div class="card-sub">Annualized (daily returns)</div>
            </div>""", unsafe_allow_html=True)

        with pm6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">{benchmark} Volatility</div>
                <div class="card-value">{bench_vol*100:.1f}%</div>
                <div class="card-sub">Annualized</div>
            </div>""", unsafe_allow_html=True)

        with pm7:
            sharpe_delta = port_sharpe - bench_sharpe
            sharpe_delta_col = "green" if sharpe_delta >= 0 else "red"
            sharpe_delta_card = "accent" if sharpe_delta >= 0 else "danger"
            sharpe_delta_sign = "+" if sharpe_delta >= 0 else ""
            st.markdown(f"""
            <div class="metric-card {sharpe_delta_card}">
                <div class="card-label">{benchmark} Sharpe Ratio</div>
                <div class="card-value">{bench_sharpe:.2f}</div>
                <div class="card-sub" style="color: {'#16a34a' if sharpe_delta >= 0 else '#dc2626'}; font-weight:600;">
                    Portfolio delta: {sharpe_delta_sign}{sharpe_delta:.2f}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── CHART: Cumulative Returns ─────────────────────────
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("#### Cumulative Growth — Portfolio vs Benchmark")
        st.caption("Starting value = $1.00")

        fig3, ax3 = plt.subplots(figsize=(12, 4.5))
        ax3.plot(port_cum.index,  port_cum,  color="#00e5c8", lw=2.0, label="Portfolio")
        ax3.plot(bench_cum.index, bench_cum, color="#8b5cf6", lw=1.5, linestyle="--", label=benchmark)
        ax3.fill_between(port_cum.index, port_cum, bench_cum,
                         where=port_cum >= bench_cum, alpha=0.1, color="#00e5c8",
                         label="Portfolio Outperforms")
        ax3.fill_between(port_cum.index, port_cum, bench_cum,
                         where=port_cum < bench_cum,  alpha=0.1, color="#ef4444",
                         label="Portfolio Underperforms")
        ax3.axhline(1.0, color="#1e3a4a", lw=0.8, linestyle=":")
        ax3.set_title("Portfolio vs Benchmark — Cumulative Return (1 Year)", fontsize=11, pad=12, color="#94a3b8")
        ax3.set_ylabel("Growth of $1", fontsize=9)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax3.legend(framealpha=0.2, facecolor="#0f1923", edgecolor="#1e3a4a", fontsize=9)
        ax3.grid(True)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ── CHART: Individual stock returns ───────────────────
        st.markdown("#### Individual Stock Returns (1 Year)")

        ind_returns = {t: float((1 + returns_df[t]).cumprod().iloc[-1] - 1) for t in tickers}
        colors_bar  = ["#00e5c8" if v >= 0 else "#ef4444" for v in ind_returns.values()]

        fig4, ax4 = plt.subplots(figsize=(9, 3.5))
        bars = ax4.bar(ind_returns.keys(), [v * 100 for v in ind_returns.values()],
                       color=colors_bar, width=0.5, edgecolor="#0a0d12", linewidth=1.5)
        for bar, val in zip(bars, ind_returns.values()):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1 if val >= 0 else -3),
                     f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9,
                     color="#00e5c8" if val >= 0 else "#ef4444")
        ax4.axhline(total_bench_ret * 100, color="#8b5cf6", lw=1.2, linestyle="--",
                    label=f"{benchmark} = {total_bench_ret*100:.1f}%")
        ax4.set_title("Individual Stock Total Returns vs Benchmark", fontsize=11, pad=10, color="#94a3b8")
        ax4.set_ylabel("Total Return (%)", fontsize=9)
        ax4.legend(framealpha=0.2, facecolor="#0f1923", edgecolor="#1e3a4a", fontsize=9)
        ax4.grid(True, axis="y")
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

        # ── Holdings table ─────────────────────────────────────
        st.markdown("#### Portfolio Holdings Summary")

        rows = []
        for t, w in zip(tickers, weights):
            if t not in returns_df.columns:
                continue
            ret  = float((1 + returns_df[t]).cumprod().iloc[-1] - 1)
            vol  = float(returns_df[t].std() * np.sqrt(252))
            shr  = sharpe_ratio(returns_df[t])
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
            ret_val = float(r["1-Yr Return"].replace("+","").replace("%","")) / 100
            ret_color = "#22c55e" if ret_val >= 0 else "#ef4444"
            html_rows += f"""
            <tr>
                <td style="color:#0284c7; font-weight:700;">{r['Ticker']}</td>
                <td>{r['Weight']}</td>
                <td style="color:{ret_color}; font-weight:600;">{r['1-Yr Return']}</td>
                <td>{r['Ann. Volatility']}</td>
                <td>{r['Sharpe Ratio']}</td>
                <td style="color:{ret_color}; font-weight:600;">{r['Weighted Contrib.']}</td>
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

        # ── STEP 6: Written Interpretation ───────────────────
        st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
        st.markdown("#### Written Interpretation")

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
            "is generating adequate return per unit of risk."
            if port_sharpe >= 1.0 else
            "A Sharpe ratio below 1.0 suggests the portfolio's return does not "
            "adequately compensate for its risk. Consider rebalancing toward higher "
            "quality or lower-volatility positions."
        )

        sharpe_compare_text = (
            f"The portfolio's Sharpe of <strong>{port_sharpe:.2f}</strong> exceeds the "
            f"benchmark's <strong>{bench_sharpe:.2f}</strong>, indicating superior risk-adjusted returns."
            if port_sharpe >= bench_sharpe else
            f"The benchmark's Sharpe of <strong>{bench_sharpe:.2f}</strong> exceeds the "
            f"portfolio's <strong>{port_sharpe:.2f}</strong>, suggesting the benchmark delivered "
            f"more return per unit of risk over this period."
        )

        st.markdown(f"""
        <div class="insight-box">
            <strong>Benchmark Comparison</strong><br>
            Over the past year, this portfolio returned <strong>{fmt_pct(total_port_ret)}</strong>
            vs the {benchmark} benchmark's <strong>{fmt_pct(total_bench_ret)}</strong>.
            The portfolio {out_text}.
            <br><br>
            <strong>Risk Assessment</strong><br>
            The portfolio was {risk_text}.
            {"Taking on additional risk was rewarded with higher returns." if port_vol > bench_vol and outperform > 0 else
             "Higher risk was not rewarded with proportionally higher returns." if port_vol > bench_vol and outperform <= 0 else
             "Lower risk with competitive returns is an attractive combination."}
            <br><br>
            <strong>Efficiency (Sharpe Ratio)</strong><br>
            {sharpe_compare_text} {sharpe_text}
            <br><br>
            <strong>Key Takeaway</strong><br>
            {"The portfolio demonstrates strong risk-adjusted performance. Continue monitoring individual position volatility and rebalance if any single weight drifts significantly." if port_sharpe >= bench_sharpe and outperform >= 0 else
             "Consider reviewing the largest detractors and evaluating whether their risk contribution is justified by their expected return."}
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("<hr class='styled-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
     color:#2d4a5a; text-align:center; letter-spacing:0.1em;">
    BUILT WITH PYTHON · STREAMLIT · YFINANCE · FOR EDUCATIONAL PURPOSES ONLY
    · NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)

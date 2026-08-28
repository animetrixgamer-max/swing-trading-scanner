# ============================================================
# NIFTY SWING TRADING SCANNER
# STREAMLIT APP
#
# Strategy:
#   NIFTY regime
#   Close > 20 EMA > 50 EMA > 200 SMA
#   RSI 55-72
#   Within 7% of 52W High
#   Close > Previous 20D High
#   Volume >= 1.5x Avg20 Volume
#   Entry <= Breakout * 1.03
#   Stop = Entry - 1.2 ATR(14)
#   Risk = user-defined % of capital
#   Target = 2R
#   Maximum hold = 10 sessions
#
# IMPORTANT:
#   Score >= 80 -> Candidate / Developing
#   Score >= 90 + ALL technical conditions -> BUY
#
# Failed conditions are explicitly marked 🔴.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import math
import time
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Swing Trading Scanner",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# DEFAULT STRATEGY SETTINGS
# ============================================================

DEFAULT_CAPITAL = 20_000
DEFAULT_RISK_PERCENT = 1.0

DEFAULT_CANDIDATE_THRESHOLD = 80
DEFAULT_BUY_THRESHOLD = 90

RSI_MIN = 55
RSI_MAX = 72

MAX_DISTANCE_52W = 0.07

MIN_VOLUME_RATIO = 1.5

ATR_MULTIPLIER = 1.2

MAX_ENTRY_EXTENSION = 0.03

MAX_HOLDING_SESSIONS = 10

DEFAULT_EARNINGS_WINDOW = 7

DEFAULT_HISTORY = "5y"


# ============================================================
# NIFTY 200 SOURCE
# ============================================================

NIFTY200_URL = (
    "https://www.niftyindices.com/"
    "IndexConstituent/ind_nifty200list.csv"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
    }

    .sub-title {
        font-size: 18px;
        color: #888888;
    }

    .buy-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #22aa55;
        background-color: rgba(0,180,80,0.08);
    }

    .developing-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d99a00;
        background-color: rgba(255,180,0,0.08);
    }

    .danger-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dd3333;
        background-color: rgba(255,0,0,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📈 Swing Trading Scanner</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Technical breakout scanner for short-term swing trades'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Scanner Parameters")


universe = st.sidebar.selectbox(
    "Universe",
    [
        "NIFTY 200",
        "NIFTY 50",
        "Custom Stocks"
    ],
    index=0
)


capital = st.sidebar.number_input(
    "Capital (₹)",
    min_value=1_000,
    value=DEFAULT_CAPITAL,
    step=1_000
)


risk_percent = st.sidebar.number_input(
    "Risk per trade (%)",
    min_value=0.1,
    max_value=5.0,
    value=DEFAULT_RISK_PERCENT,
    step=0.1
)


candidate_threshold = st.sidebar.slider(
    "Candidate score threshold",
    min_value=0,
    max_value=100,
    value=DEFAULT_CANDIDATE_THRESHOLD,
    step=5
)


buy_threshold = st.sidebar.slider(
    "Buy score threshold",
    min_value=0,
    max_value=100,
    value=DEFAULT_BUY_THRESHOLD,
    step=5
)


earnings_window = st.sidebar.number_input(
    "Earnings exclusion window (days)",
    min_value=0,
    max_value=30,
    value=DEFAULT_EARNINGS_WINDOW,
    step=1
)


history_period = st.sidebar.selectbox(
    "Historical data",
    [
        "2y",
        "3y",
        "5y"
    ],
    index=2
)


custom_symbols_text = ""

if universe == "Custom Stocks":

    custom_symbols_text = st.sidebar.text_area(
        "Enter NSE symbols",
        placeholder="RELIANCE\nTCS\nINFY\nHDFCBANK"
    )


st.sidebar.markdown("---")

st.sidebar.subheader("📋 Strategy")

st.sidebar.markdown(
    """
**Trend**

Close > 20 EMA > 50 EMA > 200 SMA

**RSI**

55 – 72

**52W High**

Within 7%

**Breakout**

Close > Previous 20D High

**Volume**

≥ 1.5 × Avg20 Volume

**Entry**

Maximum 3% above breakout

**Stop**

Entry − 1.2 × ATR(14)

**Risk**

1% default

**Target**

2R

**Maximum Hold**

10 sessions
"""
)


# ============================================================
# CALCULATIONS
# ============================================================

risk_amount = capital * risk_percent / 100


# ============================================================
# EMA
# ============================================================

def calculate_ema(series, period):

    return (
        series
        .ewm(
            span=period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )


# ============================================================
# SMA
# ============================================================

def calculate_sma(series, period):

    return (
        series
        .rolling(
            period,
            min_periods=period
        )
        .mean()
    )


# ============================================================
# RSI - WILDER
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


# ============================================================
# ATR - WILDER
# ============================================================

def calculate_atr(df, period=14):

    previous_close = (
        df["Close"].shift(1)
    )

    tr1 = (
        df["High"] -
        df["Low"]
    )

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    true_range = (
        pd.concat(
            [
                tr1,
                tr2,
                tr3
            ],
            axis=1
        )
        .max(axis=1)
    )

    return (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )


# ============================================================
# GET NIFTY 200
# ============================================================

@st.cache_data(ttl=86400)
def get_nifty200_symbols():

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36",

        "Referer":
            "https://www.niftyindices.com/"
    }

    response = requests.get(
        NIFTY200_URL,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text)
    )

    if "Symbol" not in df.columns:

        raise ValueError(
            "NIFTY 200 Symbol column not found."
        )

    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    symbols = list(
        dict.fromkeys(symbols)
    )

    if len(symbols) < 150:

        raise ValueError(
            f"Incomplete NIFTY 200 list: "
            f"{len(symbols)} stocks received."
        )

    return symbols


# ============================================================
# GET NIFTY 50
# ============================================================

@st.cache_data(ttl=86400)
def get_nifty50_symbols():

    # Current NIFTY 50 constituent list
    # from NSE indices page.

    url = (
        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty50list.csv"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36",

        "Referer":
            "https://www.niftyindices.com/"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.StringIO(response.text)
    )

    if "Symbol" not in df.columns:

        raise ValueError(
            "NIFTY 50 Symbol column not found."
        )

    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    symbols = list(
        dict.fromkeys(symbols)
    )

    if len(symbols) < 40:

        raise ValueError(
            "Incomplete NIFTY 50 universe."
        )

    return symbols


# ============================================================
# GET UNIVERSE
# ============================================================

def get_universe():

    if universe == "NIFTY 200":

        return get_nifty200_symbols()

    elif universe == "NIFTY 50":

        return get_nifty50_symbols()

    else:

        symbols = []

        for line in custom_symbols_text.splitlines():

            symbol = (
                line
                .strip()
                .upper()
            )

            if symbol:

                symbol = (
                    symbol
                    .replace(".NS", "")
                )

                symbols.append(symbol)

        return list(
            dict.fromkeys(symbols)
        )


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_stock(symbol, period):

    ticker = symbol + ".NS"

    try:

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:

            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            column in df.columns
            for column in required
        ):

            return None

        df = df[
            required
        ].copy()

        df.dropna(
            inplace=True
        )

        if len(df) < 260:

            return None

        return df

    except Exception:

        return None


# ============================================================
# ADD INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    df["EMA20"] = calculate_ema(
        df["Close"],
        20
    )

    df["EMA50"] = calculate_ema(
        df["Close"],
        50
    )

    df["SMA200"] = calculate_sma(
        df["Close"],
        200
    )

    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    df["ATR14"] = calculate_atr(
        df,
        14
    )

    df["AvgVolume20"] = (
        df["Volume"]
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    # Previous 20 completed sessions.
    #
    # Shift(1) prevents today's candle
    # from being included.

    df["Previous20DHigh"] = (
        df["High"]
        .shift(1)
        .rolling(
            20,
            min_periods=20
        )
        .max()
    )

    df["52WHigh"] = (
        df["High"]
        .rolling(
            252,
            min_periods=252
        )
        .max()
    )

    return df


# ============================================================
# SCAN ONE STOCK
# ============================================================

def scan_stock(symbol):

    df = download_stock(
        symbol,
        history_period
    )

    if df is None:

        return None

    df = add_indicators(df)

    if len(df) < 2:

        return None

    # --------------------------------------------------------
    # LAST COMPLETED CANDLE
    # --------------------------------------------------------

    row = df.iloc[-2]

    signal_date = df.index[-2]

    # --------------------------------------------------------
    # VALUES
    # --------------------------------------------------------

    try:

        close = float(row["Close"])

        ema20 = float(row["EMA20"])

        ema50 = float(row["EMA50"])

        sma200 = float(row["SMA200"])

        rsi14 = float(row["RSI14"])

        atr14 = float(row["ATR14"])

        volume = float(row["Volume"])

        avg_volume20 = float(
            row["AvgVolume20"]
        )

        previous_20d_high = float(
            row["Previous20DHigh"]
        )

        high_52w = float(
            row["52WHigh"]
        )

    except Exception:

        return None

    values = [
        close,
        ema20,
        ema50,
        sma200,
        rsi14,
        atr14,
        volume,
        avg_volume20,
        previous_20d_high,
        high_52w
    ]

    if any(
        pd.isna(value)
        for value in values
    ):

        return None

    # ========================================================
    # CONDITIONS
    # ========================================================

    trend_pass = (
        close > ema20
        and
        ema20 > ema50
        and
        ema50 > sma200
    )

    rsi_pass = (
        RSI_MIN <= rsi14 <= RSI_MAX
    )

    distance_52w = (
        high_52w - close
    ) / high_52w

    near_52w_pass = (
        distance_52w <=
        MAX_DISTANCE_52W
    )

    breakout_pass = (
        close >
        previous_20d_high
    )

    volume_ratio = (
        volume /
        avg_volume20
    )

    volume_pass = (
        volume_ratio >=
        MIN_VOLUME_RATIO
    )

    breakout_level = (
        previous_20d_high
    )

    max_entry_price = (
        breakout_level *
        (1 + MAX_ENTRY_EXTENSION)
    )

    extension_pass = (
        close <=
        max_entry_price
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = 0

    if trend_pass:
        score += 30

    if rsi_pass:
        score += 10

    if near_52w_pass:
        score += 10

    if breakout_pass:
        score += 20

    if volume_pass:
        score += 20

    if extension_pass:
        score += 10

    # ========================================================
    # ALL TECHNICAL CONDITIONS
    # ========================================================

    technical_pass = (
        trend_pass
        and
        rsi_pass
        and
        near_52w_pass
        and
        breakout_pass
        and
        volume_pass
        and
        extension_pass
    )

    # ========================================================
    # TRADE MANAGEMENT
    # ========================================================

    entry = close

    stop = (
        entry -
        ATR_MULTIPLIER *
        atr14
    )

    risk_per_share = (
        entry -
        stop
    )

    if risk_per_share <= 0:

        return None

    quantity = math.floor(
        risk_amount /
        risk_per_share
    )

    capital_required = (
        quantity *
        entry
    )

    actual_risk = (
        quantity *
        risk_per_share
    )

    target_2r = (
        entry +
        2 *
        risk_per_share
    )

    # ========================================================
    # CONDITION MARKERS
    # ========================================================

    trend_marker = (
        "🟢 PASS"
        if trend_pass
        else
        "🔴 FAIL"
    )

    rsi_marker = (
        "🟢 PASS"
        if rsi_pass
        else
        "🔴 FAIL"
    )

    near_52w_marker = (
        "🟢 PASS"
        if near_52w_pass
        else
        "🔴 FAIL"
    )

    breakout_marker = (
        "🟢 PASS"
        if breakout_pass
        else
        "🔴 FAIL"
    )

    volume_marker = (
        "🟢 PASS"
        if volume_pass
        else
        "🔴 FAIL"
    )

    extension_marker = (
        "🟢 PASS"
        if extension_pass
        else
        "🔴 FAIL"
    )

    # ========================================================
    # CANDIDATE CLASSIFICATION
    # ========================================================

    if score >= buy_threshold and technical_pass:

        technical_status = (
            "🟢 BUY CANDIDATE"
        )

    elif score >= candidate_threshold:

        technical_status = (
            "🟡 DEVELOPING"
        )

    else:

        technical_status = (
            "⚪ BELOW THRESHOLD"
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "Stock":
            symbol,

        "Signal Date":
            signal_date.strftime(
                "%Y-%m-%d"
            ),

        "Close":
            round(close, 2),

        "20 EMA":
            round(ema20, 2),

        "50 EMA":
            round(ema50, 2),

        "200 SMA":
            round(sma200, 2),

        "RSI(14)":
            round(rsi14, 2),

        "52W High":
            round(high_52w, 2),

        "Distance 52W High %":
            round(
                distance_52w * 100,
                2
            ),

        "Previous 20D High":
            round(
                previous_20d_high,
                2
            ),

        "Volume":
            int(volume),

        "Avg20 Volume":
            int(avg_volume20),

        "Volume Ratio":
            round(
                volume_ratio,
                2
            ),

        "ATR(14)":
            round(atr14, 2),

        "Entry":
            round(entry, 2),

        "Stop":
            round(stop, 2),

        "Risk/Share":
            round(
                risk_per_share,
                2
            ),

        "Quantity":
            quantity,

        "Capital Required":
            round(
                capital_required,
                2
            ),

        "Actual Risk":
            round(
                actual_risk,
                2
            ),

        "2R Target":
            round(
                target_2r,
                2
            ),

        "Max Hold":
            MAX_HOLDING_SESSIONS,

        # ====================================================
        # CONDITION COLUMNS
        # ====================================================

        "Trend":
            trend_marker,

        "RSI Condition":
            rsi_marker,

        "52W Condition":
            near_52w_marker,

        "Breakout Condition":
            breakout_marker,

        "Volume Condition":
            volume_marker,

        "Entry Extension":
            extension_marker,

        "Score":
            score,

        "Technical Status":
            technical_status
    }


# ============================================================
# EARNINGS CHECK
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def check_upcoming_earnings(
    symbol,
    days
):

    try:

        ticker = yf.Ticker(
            symbol + ".NS"
        )

        earnings = (
            ticker
            .get_earnings_dates(
                limit=12
            )
        )

        if (
            earnings is None
            or
            earnings.empty
        ):

            return {
                "Earnings Status":
                    "🟠 UNKNOWN",

                "Earnings Date":
                    "N/A",

                "Days Until Earnings":
                    "N/A"
            }

        today = (
            pd.Timestamp
            .now()
            .normalize()
        )

        future_dates = []

        for date in earnings.index:

            try:

                earnings_date = (
                    pd.Timestamp(date)
                )

                if earnings_date.tzinfo:

                    earnings_date = (
                        earnings_date
                        .tz_localize(None)
                    )

                earnings_date = (
                    earnings_date
                    .normalize()
                )

                if earnings_date >= today:

                    future_dates.append(
                        earnings_date
                    )

            except Exception:

                continue

        if not future_dates:

            return {
                "Earnings Status":
                    "🟠 UNKNOWN",

                "Earnings Date":
                    "N/A",

                "Days Until Earnings":
                    "N/A"
            }

        next_earnings = min(
            future_dates
        )

        days_until = (
            next_earnings -
            today
        ).days

        if days_until <= days:

            status = (
                "🔴 WITHIN WINDOW"
            )

        else:

            status = (
                "🟢 CLEAR"
            )

        return {

            "Earnings Status":
                status,

            "Earnings Date":
                next_earnings.strftime(
                    "%Y-%m-%d"
                ),

            "Days Until Earnings":
                days_until
        }

    except Exception:

        return {

            "Earnings Status":
                "🟠 UNKNOWN",

            "Earnings Date":
                "N/A",

            "Days Until Earnings":
                "N/A"
        }


# ============================================================
# NIFTY MARKET REGIME
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def calculate_nifty_regime():

    try:

        df = yf.download(
            "^NSEI",
            period=history_period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:

            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df["EMA20"] = calculate_ema(
            df["Close"],
            20
        )

        df["EMA50"] = calculate_ema(
            df["Close"],
            50
        )

        if len(df) < 2:

            return None

        row = df.iloc[-2]

        close = float(
            row["Close"]
        )

        ema20 = float(
            row["EMA20"]
        )

        ema50 = float(
            row["EMA50"]
        )

        bullish = (
            close > ema20
            and
            ema20 > ema50
        )

        return {

            "bullish":
                bullish,

            "close":
                close,

            "ema20":
                ema20,

            "ema50":
                ema50
        }

    except Exception:

        return None


# ============================================================
# RUN BUTTON
# ============================================================

run_scanner = st.button(
    "🚀 RUN SCANNER",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN SCANNER
# ============================================================

if run_scanner:

    start_time = time.time()

    # ========================================================
    # MARKET
    # ========================================================

    with st.spinner(
        "Checking NIFTY market regime..."
    ):

        market = (
            calculate_nifty_regime()
        )

    if market is None:

        st.error(
            "🔴 Unable to obtain NIFTY data."
        )

        st.stop()

    # ========================================================
    # MARKET DISPLAY
    # ========================================================

    st.subheader(
        "📊 Market Regime"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "NIFTY Close",
        f"₹{market['close']:,.2f}"
    )

    c2.metric(
        "20 EMA",
        f"₹{market['ema20']:,.2f}"
    )

    c3.metric(
        "50 EMA",
        f"₹{market['ema50']:,.2f}"
    )

    if market["bullish"]:

        c4.success(
            "🟢 BULLISH"
        )

    else:

        c4.error(
            "🔴 NOT BULLISH"
        )

    st.markdown("---")

    # ========================================================
    # UNIVERSE
    # ========================================================

    with st.spinner(
        "Loading stock universe..."
    ):

        try:

            symbols = get_universe()

        except Exception as e:

            st.error(
                f"🔴 Universe unavailable: {e}"
            )

            st.stop()

    if not symbols:

        st.warning(
            "No stocks selected."
        )

        st.stop()

    st.info(
        f"Universe: **{universe}** | "
        f"Stocks to scan: **{len(symbols)}**"
    )

    # ========================================================
    # SCAN
    # ========================================================

    results = []

    failed = []

    progress = st.progress(0)

    status_text = st.empty()

    for i, symbol in enumerate(
        symbols,
        1
    ):

        status_text.write(
            f"Scanning {i}/{len(symbols)}: **{symbol}**"
        )

        result = scan_stock(
            symbol
        )

        if result is None:

            failed.append(symbol)

        else:

            results.append(result)

        progress.progress(
            i / len(symbols)
        )

    progress.empty()

    status_text.empty()

    if not results:

        st.error(
            "🔴 No valid stock data received."
        )

        st.stop()

    all_results = pd.DataFrame(
        results
    )

    # ========================================================
    # CANDIDATES
    #
    # IMPORTANT:
    # Candidate = score >= 80
    #
    # It does NOT have to satisfy every condition.
    # This allows us to see developing setups.
    # ========================================================

    candidates = (
        all_results[
            all_results["Score"]
            >=
            candidate_threshold
        ]
        .sort_values(
            by=[
                "Score",
                "Volume Ratio"
            ],
            ascending=[
                False,
                False
            ]
        )
        .copy()
    )

    # ========================================================
    # EARNINGS CHECK
    #
    # ONLY candidates.
    # ========================================================

    final_rows = []

    earnings_progress = st.progress(0)

    earnings_status_text = st.empty()

    for i, (_, candidate) in enumerate(
        candidates.iterrows(),
        1
    ):

        symbol = candidate["Stock"]

        earnings_status_text.write(
            f"Checking earnings: **{symbol}**"
        )

        earnings = (
            check_upcoming_earnings(
                symbol,
                earnings_window
            )
        )

        row = candidate.to_dict()

        row.update(
            earnings
        )

        score = int(
            row["Score"]
        )

        technical_status = (
            row["Technical Status"]
        )

        earnings_status = (
            row["Earnings Status"]
        )

        # ====================================================
        # FINAL DECISION
        # ====================================================

        if (
            score >= buy_threshold
            and
            technical_status
            ==
            "🟢 BUY CANDIDATE"
            and
            earnings_status
            ==
            "🟢 CLEAR"
        ):

            final_decision = (
                "🟢 BUY CANDIDATE"
            )

        elif (
            earnings_status
            ==
            "🔴 WITHIN WINDOW"
        ):

            final_decision = (
                "🔴 SKIP - EARNINGS SOON"
            )

        elif (
            earnings_status
            ==
            "🟠 UNKNOWN"
        ):

            final_decision = (
                "🟠 VERIFY EARNINGS"
            )

        elif score >= candidate_threshold:

            final_decision = (
                "🟡 DEVELOPING"
            )

        else:

            final_decision = (
                "⚪ WATCH"
            )

        row[
            "FINAL DECISION"
        ] = final_decision

        final_rows.append(
            row
        )

        if len(candidates) > 0:

            earnings_progress.progress(
                i / len(candidates)
            )

    earnings_progress.empty()

    earnings_status_text.empty()

    if final_rows:

        final_candidates = (
            pd.DataFrame(
                final_rows
            )
        )

    else:

        final_candidates = (
            pd.DataFrame()
        )

    # ========================================================
    # FINAL SORT
    # ========================================================

    if not final_candidates.empty:

        decision_order = {
            "🟢 BUY CANDIDATE": 1,
            "🟡 DEVELOPING": 2,
            "🟠 VERIFY EARNINGS": 3,
            "🔴 SKIP - EARNINGS SOON": 4
        }

        final_candidates[
            "_sort"
        ] = (
            final_candidates[
                "FINAL DECISION"
            ]
            .map(decision_order)
            .fillna(99)
        )

        final_candidates = (
            final_candidates
            .sort_values(
                by=[
                    "_sort",
                    "Score",
                    "Volume Ratio"
                ],
                ascending=[
                    True,
                    False,
                    False
                ]
            )
            .drop(
                columns=["_sort"]
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎯 Scanner Summary"
    )

    buy_count = 0
    developing_count = 0
    earnings_blocked = 0
    verify_count = 0

    if not final_candidates.empty:

        buy_count = (
            final_candidates[
                final_candidates[
                    "FINAL DECISION"
                ]
                ==
                "🟢 BUY CANDIDATE"
            ]
            .shape[0]
        )

        developing_count = (
            final_candidates[
                final_candidates[
                    "FINAL DECISION"
                ]
                ==
                "🟡 DEVELOPING"
            ]
            .shape[0]
        )

        earnings_blocked = (
            final_candidates[
                final_candidates[
                    "FINAL DECISION"
                ]
                ==
                "🔴 SKIP - EARNINGS SOON"
            ]
            .shape[0]
        )

        verify_count = (
            final_candidates[
                final_candidates[
                    "FINAL DECISION"
                ]
                ==
                "🟠 VERIFY EARNINGS"
            ]
            .shape[0]
        )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "🟢 BUY",
        buy_count
    )

    s2.metric(
        "🟡 DEVELOPING",
        developing_count
    )

    s3.metric(
        "🔴 Earnings Blocked",
        earnings_blocked
    )

    s4.metric(
        "🟠 Verify Earnings",
        verify_count
    )

    # ========================================================
    # FINAL CANDIDATES
    # ========================================================

    st.markdown("---")

    st.subheader(
        f"🎯 Final Candidates "
        f"(Score ≥ {candidate_threshold})"
    )

    if final_candidates.empty:

        st.warning(
            "No stocks reached the candidate threshold."
        )

    else:

        display_columns = [

            "Stock",

            "Signal Date",

            "Close",

            "20 EMA",

            "50 EMA",

            "200 SMA",

            "RSI(14)",

            "52W High",

            "Distance 52W High %",

            "Previous 20D High",

            "Volume Ratio",

            "ATR(14)",

            "Entry",

            "Stop",

            "Risk/Share",

            "Quantity",

            "Capital Required",

            "Actual Risk",

            "2R Target",

            "Max Hold",

            "Trend",

            "RSI Condition",

            "52W Condition",

            "Breakout Condition",

            "Volume Condition",

            "Entry Extension",

            "Score",

            "Technical Status",

            "Earnings Status",

            "Earnings Date",

            "Days Until Earnings",

            "FINAL DECISION"
        ]

        display_columns = [
            column
            for column in display_columns
            if column
            in final_candidates.columns
        ]

        st.dataframe(
            final_candidates[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # BUY CANDIDATES ONLY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🟢 BUY CANDIDATES"
    )

    if not final_candidates.empty:

        buy_df = (
            final_candidates[
                final_candidates[
                    "FINAL DECISION"
                ]
                ==
                "🟢 BUY CANDIDATE"
            ]
        )

        if buy_df.empty:

            st.info(
                "No stock currently qualifies as a BUY."
            )

        else:

            st.dataframe(
                buy_df[
                    [
                        "Stock",
                        "Close",
                        "RSI(14)",
                        "Volume Ratio",
                        "ATR(14)",
                        "Entry",
                        "Stop",
                        "Quantity",
                        "Capital Required",
                        "Actual Risk",
                        "2R Target",
                        "Score",
                        "FINAL DECISION"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # DEVELOPING STOCKS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🟡 DEVELOPING WATCHLIST"
    )

    if not final_candidates.empty:

        developing_df = (
            final_candidates[
                final_candidates[
                    "FINAL DECISION"
                ]
                ==
                "🟡 DEVELOPING"
            ]
        )

        if developing_df.empty:

            st.info(
                "No developing setups today."
            )

        else:

            st.dataframe(
                developing_df[
                    [
                        "Stock",
                        "Close",
                        "RSI(14)",
                        "Volume Ratio",
                        "Trend",
                        "RSI Condition",
                        "52W Condition",
                        "Breakout Condition",
                        "Volume Condition",
                        "Entry Extension",
                        "Score",
                        "Technical Status",
                        "FINAL DECISION"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "🔴 FAIL means the stock is currently "
                "missing that strategy condition. "
                "These stocks are shown because their "
                f"score is ≥ {candidate_threshold}."
            )

    # ========================================================
    # COMPLETE SCAN
    # ========================================================

    st.markdown("---")

    with st.expander(
        "📊 Complete NIFTY Universe Scan"
    ):

        st.write(
            f"Valid stocks processed: "
            f"{len(all_results)}"
        )

        st.write(
            f"Data unavailable: "
            f"{len(failed)}"
        )

        st.dataframe(
            all_results,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # EXCEL DOWNLOAD
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📥 Download Results"
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    # --------------------------------------------------------
    # Final Candidates Excel
    # --------------------------------------------------------

    candidates_buffer = io.BytesIO()

    with pd.ExcelWriter(
        candidates_buffer,
        engine="openpyxl"
    ) as writer:

        final_candidates.to_excel(
            writer,
            sheet_name="Final Candidates",
            index=False
        )

        if not final_candidates.empty:

            buy_df = (
                final_candidates[
                    final_candidates[
                        "FINAL DECISION"
                    ]
                    ==
                    "🟢 BUY CANDIDATE"
                ]
            )

            developing_df = (
                final_candidates[
                    final_candidates[
                        "FINAL DECISION"
                    ]
                    ==
                    "🟡 DEVELOPING"
                ]
            )

            buy_df.to_excel(
                writer,
                sheet_name="Buy Candidates",
                index=False
            )

            developing_df.to_excel(
                writer,
                sheet_name="Developing Watchlist",
                index=False
            )

        all_results.to_excel(
            writer,
            sheet_name="Complete Scan",
            index=False
        )

    candidates_buffer.seek(0)

    st.download_button(
        label="📥 Download Final Excel",
        data=candidates_buffer,
        file_name=(
            f"Swing_Scanner_{timestamp}.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )

    # ========================================================
    # SCAN INFORMATION
    # ========================================================

    elapsed = (
        time.time() -
        start_time
    )

    st.markdown("---")

    st.caption(
        f"Universe: {universe} | "
        f"Capital: ₹{capital:,.0f} | "
        f"Risk: ₹{risk_amount:,.2f} | "
        f"Candidate threshold: {candidate_threshold} | "
        f"Buy threshold: {buy_threshold} | "
        f"History: {history_period} | "
        f"Scan time: {elapsed / 60:.1f} minutes"
    )

    st.caption(
        "Data source: Yahoo Finance. "
        "Technical indicators are calculated locally "
        "from downloaded historical OHLCV data. "
        "Last completed daily candle is used."
    )

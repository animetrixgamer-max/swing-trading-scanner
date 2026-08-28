import io
import json
import math
import os
import time
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Optional Google Drive support
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    DRIVE_LIBS_AVAILABLE = True
except Exception:
    DRIVE_LIBS_AVAILABLE = False


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DEFAULTS
# ============================================================
DEFAULT_CAPITAL = 20_000
DEFAULT_RISK_PERCENT = 1.0
DEFAULT_UNIVERSE = "NIFTY 200"
DEFAULT_HISTORY = "5y"
DEFAULT_EARNINGS_DAYS = 7
DEFAULT_CANDIDATE_SCORE = 80
DEFAULT_BUY_SCORE = 90

# Strategy constants
RSI_MIN = 55
RSI_MAX = 72
MAX_DISTANCE_52W = 0.07
MIN_VOLUME_RATIO = 1.5
ATR_MULTIPLIER = 1.2
MAX_ENTRY_EXTENSION = 0.03
MAX_HOLDING_SESSIONS = 10

NIFTY200_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"

# ============================================================
# UTILS
# ============================================================
def flatten_yf_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        # Keep the field names for single-ticker and multi-ticker downloads.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def clean_ohlcv(df):
    if df is None or df.empty:
        return None
    df = flatten_yf_columns(df.copy())
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) >= 260 else None


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_sma(series, period):
    return series.rolling(period, min_periods=period).mean()


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    previous_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df):
    df = df.copy()
    df["EMA20"] = calculate_ema(df["Close"], 20)
    df["EMA50"] = calculate_ema(df["Close"], 50)
    df["SMA200"] = calculate_sma(df["Close"], 200)
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = calculate_atr(df, 14)
    df["AvgVolume20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["Previous20DHigh"] = df["High"].shift(1).rolling(20, min_periods=20).max()
    df["52WHigh"] = df["High"].rolling(252, min_periods=252).max()
    return df


# ============================================================
# NIFTY UNIVERSES
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty200_symbols():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "Referer": "https://www.niftyindices.com/",
    }
    r = requests.get(NIFTY200_URL, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Symbol" not in df.columns:
        raise ValueError("Official NIFTY 200 list does not contain Symbol column.")
    symbols = df["Symbol"].dropna().astype(str).str.strip().tolist()
    symbols = list(dict.fromkeys(symbols))
    if len(symbols) < 150:
        raise ValueError(f"Only {len(symbols)} NIFTY 200 symbols received; refusing incomplete universe.")
    return symbols


@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty50_symbols():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
        "Referer": "https://www.niftyindices.com/",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Symbol" not in df.columns:
        raise ValueError("Official NIFTY 50 list does not contain Symbol column.")
    return list(dict.fromkeys(df["Symbol"].dropna().astype(str).str.strip().tolist()))


# ============================================================
# DATA DOWNLOAD
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def download_history_batch(symbols, period="5y"):
    tickers = [f"{s}.NS" for s in symbols]
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    return data


@st.cache_data(ttl=900, show_spinner=False)
def download_index_history(period="5y"):
    df = yf.download(
        "^NSEI",
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return clean_ohlcv(df)


def extract_ticker_df(batch, symbol):
    ticker = f"{symbol}.NS"
    if batch is None or batch.empty:
        return None

    # Typical yfinance multi-index shape: level 0=ticker, level 1=field.
    if isinstance(batch.columns, pd.MultiIndex):
        if ticker in batch.columns.get_level_values(0):
            df = batch[ticker].copy()
        elif symbol in batch.columns.get_level_values(0):
            df = batch[symbol].copy()
        else:
            return None
    else:
        df = batch.copy()
    return clean_ohlcv(df)


# ============================================================
# MARKET REGIME
# ============================================================
def calculate_market_regime():
    df = download_index_history(DEFAULT_HISTORY)
    if df is None or len(df) < 210:
        return None
    df["EMA20"] = calculate_ema(df["Close"], 20)
    df["EMA50"] = calculate_ema(df["Close"], 50)
    row = df.iloc[-2]  # last completed candle
    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    bullish = close > ema20 and ema20 > ema50
    return {
        "date": df.index[-2].strftime("%Y-%m-%d"),
        "close": close,
        "ema20": ema20,
        "ema50": ema50,
        "bullish": bullish,
    }


# ============================================================
# STOCK SCANNER
# ============================================================
def scan_stock(symbol, df):
    df = add_indicators(df)
    if len(df) < 260:
        return None

    row = df.iloc[-2]  # completed candle
    signal_date = df.index[-2]

    required = [
        "Close", "EMA20", "EMA50", "SMA200", "RSI14", "ATR14",
        "Volume", "AvgVolume20", "Previous20DHigh", "52WHigh"
    ]
    if any(pd.isna(row[x]) for x in required):
        return None

    close = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    sma200 = float(row["SMA200"])
    rsi = float(row["RSI14"])
    atr = float(row["ATR14"])
    volume = float(row["Volume"])
    avg20 = float(row["AvgVolume20"])
    prev20 = float(row["Previous20DHigh"])
    high52 = float(row["52WHigh"])

    close_above_ema20 = close > ema20
    ema20_above_ema50 = ema20 > ema50
    ema50_above_sma200 = ema50 > sma200
    trend_pass = close_above_ema20 and ema20_above_ema50 and ema50_above_sma200
    rsi_pass = RSI_MIN <= rsi <= RSI_MAX
    distance_52w = (high52 - close) / high52
    near_52w_pass = distance_52w <= MAX_DISTANCE_52W
    breakout_pass = close > prev20
    volume_ratio = volume / avg20 if avg20 else np.nan
    volume_pass = volume_ratio >= MIN_VOLUME_RATIO
    max_entry_price = prev20 * (1 + MAX_ENTRY_EXTENSION)
    extension_pass = close <= max_entry_price

    failed = []
    if not close_above_ema20:
        failed.append("🔴 Close < 20 EMA")
    if not ema20_above_ema50:
        failed.append("🔴 20 EMA < 50 EMA")
    if not ema50_above_sma200:
        failed.append("🔴 50 EMA < 200 SMA")
    if not rsi_pass:
        failed.append(f"🔴 RSI {'below' if rsi < RSI_MIN else 'above'} {RSI_MIN if rsi < RSI_MIN else RSI_MAX}")
    if not near_52w_pass:
        failed.append("🔴 More than 7% below 52W High")
    if not breakout_pass:
        failed.append("🔴 No 20D breakout")
    if not volume_pass:
        failed.append("🔴 Volume < 1.5× Avg20")
    if not extension_pass:
        failed.append("🔴 Price > Breakout + 3%")

    technical_pass = trend_pass and rsi_pass and near_52w_pass and breakout_pass and volume_pass and extension_pass

    # Score is based only on the technical rules. NIFTY regime is a separate gate.
    score = 0
    score += 30 if trend_pass else 0
    score += 10 if rsi_pass else 0
    score += 10 if near_52w_pass else 0
    score += 20 if breakout_pass else 0
    score += 20 if volume_pass else 0
    score += 10 if extension_pass else 0

    entry = close
    stop = entry - ATR_MULTIPLIER * atr
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None

    qty = math.floor((RISK_AMOUNT_GLOBAL) / risk_per_share)
    actual_risk = qty * risk_per_share
    capital_required = qty * entry
    target_2r = entry + 2 * risk_per_share

    return {
        "Stock": symbol,
        "Signal Date": signal_date.strftime("%Y-%m-%d"),
        "Close": round(close, 2),
        "20 EMA": round(ema20, 2),
        "50 EMA": round(ema50, 2),
        "200 SMA": round(sma200, 2),
        "RSI(14)": round(rsi, 2),
        "52W High": round(high52, 2),
        "Distance 52W High %": round(distance_52w * 100, 2),
        "Previous 20D High": round(prev20, 2),
        "Volume": int(volume),
        "Avg20 Volume": int(avg20),
        "Volume Ratio": round(volume_ratio, 2),
        "ATR(14)": round(atr, 2),
        "Entry": round(entry, 2),
        "Stop": round(stop, 2),
        "Risk/Share": round(risk_per_share, 2),
        "Quantity": qty,
        "Capital Required": round(capital_required, 2),
        "Actual Risk": round(actual_risk, 2),
        "2R Target": round(target_2r, 2),
        "Max Hold": MAX_HOLDING_SESSIONS,
        "Score": score,
        "Trend": "PASS" if trend_pass else "FAIL",
        "RSI": "PASS" if rsi_pass else "FAIL",
        "52W Proximity": "PASS" if near_52w_pass else "FAIL",
        "20D Breakout": "PASS" if breakout_pass else "FAIL",
        "Volume": "PASS" if volume_pass else "FAIL",
        "Entry Extension": "PASS" if extension_pass else "FAIL",
        "Technical Pass": technical_pass,
        "Failed Conditions": " | ".join(failed) if failed else "🟢 ALL CONDITIONS PASSED",
        "Technical Status": "🟢 BUY CANDIDATE" if technical_pass else ("🟡 DEVELOPING WATCH" if score >= DEVELOPING_SCORE_MIN_GLOBAL else "🔴 SKIP"),
    }


# ============================================================
# STAGE 2 - EARNINGS
# ============================================================
def check_upcoming_earnings(symbol, days=7):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        earnings = ticker.get_earnings_dates(limit=12)
        if earnings is None or earnings.empty:
            return {"Earnings Status": "🟠 UNKNOWN", "Earnings Date": "N/A", "Days Until Earnings": "N/A"}

        today = pd.Timestamp.now().normalize()
        future_dates = []
        for d in earnings.index:
            try:
                d = pd.Timestamp(d)
                if d.tzinfo is not None:
                    d = d.tz_localize(None)
                d = d.normalize()
                if d >= today:
                    future_dates.append(d)
            except Exception:
                pass

        if not future_dates:
            return {"Earnings Status": "🟠 UNKNOWN", "Earnings Date": "N/A", "Days Until Earnings": "N/A"}

        next_date = min(future_dates)
        days_until = (next_date - today).days
        return {
            "Earnings Status": "🔴 WITHIN 7 DAYS" if days_until <= days else "🟢 CLEAR",
            "Earnings Date": next_date.strftime("%Y-%m-%d"),
            "Days Until Earnings": days_until,
        }
    except Exception:
        return {"Earnings Status": "🟠 UNKNOWN", "Earnings Date": "N/A", "Days Until Earnings": "N/A"}


# ============================================================
# GOOGLE DRIVE UPLOAD (OPTIONAL)
# ============================================================
def drive_upload_file(local_path, drive_folder_id=None):
    if not DRIVE_LIBS_AVAILABLE or "gcp_service_account" not in st.secrets:
        return False, "Google Drive upload is not configured."

    try:
        info = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        metadata = {"name": os.path.basename(local_path)}
        if drive_folder_id:
            metadata["parents"] = [drive_folder_id]

        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        with open(local_path, "rb") as fh:
            media = MediaIoBaseUpload(fh, mimetype=mime, resumable=True)
            result = service.files().create(
                body=metadata,
                media_body=media,
                fields="id,name,webViewLink",
            ).execute()
        return True, f"Uploaded: {result.get('name')}"
    except Exception as exc:
        return False, f"Drive upload failed: {exc}"


# ============================================================
# EXCEL HELPERS
# ============================================================
def dataframe_to_excel_bytes(sheets):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    out.seek(0)
    return out


# ============================================================
# APP STATE / GLOBALS
# ============================================================
RISK_AMOUNT_GLOBAL = DEFAULT_CAPITAL * DEFAULT_RISK_PERCENT / 100
DEVELOPING_SCORE_MIN_GLOBAL = DEFAULT_CANDIDATE_SCORE


# ============================================================
# UI
# ============================================================
st.title("📈 Indian Swing Scanner")
st.caption("NIFTY 200 / NIFTY 50 / Custom • Stage 1 technical scan • Stage 2 earnings check • ₹20K default")

with st.sidebar:
    st.header("Scanner Parameters")

    universe = st.selectbox(
        "Universe",
        ["NIFTY 200", "NIFTY 50", "Custom Stocks"],
        index=0,
    )

    capital = st.number_input(
        "Capital (₹)",
        min_value=1000.0,
        value=float(DEFAULT_CAPITAL),
        step=1000.0,
    )

    risk_percent = st.number_input(
        "Risk per trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=DEFAULT_RISK_PERCENT,
        step=0.1,
    )

    candidate_score = st.slider(
        "Candidate score threshold",
        min_value=0,
        max_value=100,
        value=DEFAULT_CANDIDATE_SCORE,
        step=10,
    )

    buy_score = st.slider(
        "Buy score threshold",
        min_value=0,
        max_value=100,
        value=DEFAULT_BUY_SCORE,
        step=10,
    )

    earnings_days = st.number_input(
        "Earnings exclusion window (days)",
        min_value=1,
        max_value=30,
        value=DEFAULT_EARNINGS_DAYS,
        step=1,
    )

    history_period = st.selectbox(
        "Historical data",
        ["2y", "5y"],
        index=1,
    )

    st.divider()
    st.write("**Strategy (fixed)**")
    st.write("• Trend: Close > 20 EMA > 50 EMA > 200 SMA")
    st.write("• RSI: 55–72")
    st.write("• Within 7% of 52W high")
    st.write("• Breakout: previous 20D high")
    st.write("• Volume: ≥1.5× Avg20")
    st.write("• Max breakout extension: 3%")
    st.write("• Stop: 1.2× ATR(14)")
    st.write("• Target: 2R")
    st.write("• Max hold: 10 sessions")

    if universe == "Custom Stocks":
        custom_text = st.text_area(
            "Enter NSE symbols (one per line or comma-separated)",
            value="RELIANCE\nTCS\nINFY",
            height=140,
        )
    else:
        custom_text = ""

run = st.button("🚀 RUN SCANNER", use_container_width=True, type="primary")

if run:
    # Update globals used in scan_stock without changing the function contract.
    RISK_AMOUNT_GLOBAL = float(capital) * float(risk_percent) / 100
    DEVELOPING_SCORE_MIN_GLOBAL = int(candidate_score)
    globals()["RISK_AMOUNT_GLOBAL"] = RISK_AMOUNT_GLOBAL
    globals()["DEVELOPING_SCORE_MIN_GLOBAL"] = DEVELOPING_SCORE_MIN_GLOBAL

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------
    with st.status("Preparing universe...", expanded=False) as status:
        try:
            if universe == "NIFTY 200":
                symbols = get_nifty200_symbols()
            elif universe == "NIFTY 50":
                symbols = get_nifty50_symbols()
            else:
                raw = custom_text.replace("\n", ",")
                symbols = [x.strip().upper().replace(".NS", "") for x in raw.split(",") if x.strip()]
                symbols = list(dict.fromkeys(symbols))
                if not symbols:
                    st.error("No custom symbols supplied.")
                    st.stop()
            status.update(label=f"Universe ready: {len(symbols)} symbols", state="complete")
        except Exception as exc:
            status.update(label="Universe failed", state="error")
            st.error(f"Could not load universe: {exc}")
            st.stop()

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------
    with st.spinner("Checking NIFTY 50 market regime..."):
        market = calculate_market_regime()

    if market is None:
        st.error("NIFTY 50 market data unavailable. Scanner stopped.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NIFTY Close", f"₹{market['close']:,.2f}")
    c2.metric("20 EMA", f"₹{market['ema20']:,.2f}")
    c3.metric("50 EMA", f"₹{market['ema50']:,.2f}")
    c4.metric("Market", "🟢 BULLISH" if market["bullish"] else "🔴 NOT BULLISH")
    st.caption(f"Market candle used: {market['date']} (last completed daily session)")

    # --------------------------------------------------------
    # Batch download and Stage 1
    # --------------------------------------------------------
    progress = st.progress(0, text="Downloading historical data...")
    batch = download_history_batch(tuple(symbols), period=history_period)
    progress.progress(100, text="Historical data downloaded")

    results = []
    failed = []

    progress = st.progress(0, text="Running Stage 1 technical scan...")
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        df = extract_ticker_df(batch, symbol)
        if df is None:
            failed.append(symbol)
        else:
            try:
                result = scan_stock(symbol, df)
                if result is None:
                    failed.append(symbol)
                else:
                    results.append(result)
            except Exception:
                failed.append(symbol)
        progress.progress(i / total, text=f"Stage 1: {i}/{total} stocks")

    progress.empty()

    if not results:
        st.error("No valid stock data was processed.")
        st.stop()

    all_results = pd.DataFrame(results)

    # NIFTY regime is a gate for actual BUY status, but we keep developing setups visible.
    all_results["NIFTY Regime"] = "PASS" if market["bullish"] else "FAIL"

    # Candidate pool = score >= selected threshold.
    candidate_pool = (
        all_results[all_results["Score"] >= int(candidate_score)]
        .sort_values(["Score", "Volume Ratio"], ascending=[False, False])
        .copy()
    )

    # Actual Stage-1 BUY = complete technical pass + score >= buy threshold + bullish NIFTY.
    stage1_buy = (
        all_results[
            (all_results["Technical Pass"] == True)
            & (all_results["Score"] >= int(buy_score))
            & (market["bullish"])
        ]
        .sort_values(["Score", "Volume Ratio"], ascending=[False, False])
        .copy()
    )

    # Developing = candidates not currently meeting the final Stage-1 BUY gate.
    developing = candidate_pool[~candidate_pool["Stock"].isin(stage1_buy["Stock"])]

    if not developing.empty:
        developing["Final Status"] = "🟡 DEVELOPING WATCH"
    if not stage1_buy.empty:
        stage1_buy["Final Status"] = "🟢 TECHNICAL BUY CANDIDATE"

    # --------------------------------------------------------
    # Stage 2 only for actual buy candidates
    # --------------------------------------------------------
    stage2 = []
    if not stage1_buy.empty:
        st.subheader("Stage 2 — Earnings Check")
        for _, row in stage1_buy.iterrows():
            earnings = check_upcoming_earnings(row["Stock"], int(earnings_days))
            record = row.to_dict()
            record.update(earnings)
            if earnings["Earnings Status"] == "🔴 WITHIN 7 DAYS":
                decision = "🔴 SKIP - EARNINGS SOON"
            elif earnings["Earnings Status"] == "🟠 UNKNOWN":
                decision = "🟠 VERIFY EARNINGS"
            else:
                decision = "🟢 BUY CANDIDATE"
            record["FINAL DECISION"] = decision
            stage2.append(record)
        final_buy = pd.DataFrame(stage2)
    else:
        final_buy = pd.DataFrame()

    # --------------------------------------------------------
    # Final candidate sheet = Score >= threshold.
    # Developing candidates are included; actual BUYs carry Stage 2 details.
    # --------------------------------------------------------
    if not final_buy.empty:
        buy_map = final_buy.set_index("Stock")
        developing_copy = developing.copy()
        developing_copy["Earnings Status"] = "N/A"
        developing_copy["Earnings Date"] = "N/A"
        developing_copy["Days Until Earnings"] = "N/A"
        developing_copy["FINAL DECISION"] = "🟡 DEVELOPING WATCH"
        final_candidates = pd.concat([final_buy, developing_copy], ignore_index=True, sort=False)
    else:
        final_candidates = developing.copy()
        if not final_candidates.empty:
            final_candidates["Earnings Status"] = "N/A"
            final_candidates["Earnings Date"] = "N/A"
            final_candidates["Days Until Earnings"] = "N/A"
            final_candidates["FINAL DECISION"] = "🟡 DEVELOPING WATCH"

    if not final_candidates.empty:
        final_candidates = final_candidates.sort_values(
            ["Score", "Volume Ratio"], ascending=[False, False]
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------
    st.subheader("🎯 Final Candidates")
    st.caption(f"Candidate threshold: Score ≥ {candidate_score}. Buy threshold: Score ≥ {buy_score} plus all technical conditions and bullish NIFTY.")

    if final_candidates.empty:
        st.warning("NO SCORE-QUALIFYING CANDIDATES TODAY")
    else:
        display_cols = [
            "Stock", "Signal Date", "Close", "20 EMA", "50 EMA", "200 SMA",
            "RSI(14)", "52W High", "Distance 52W High %", "Previous 20D High",
            "Volume Ratio", "ATR(14)", "Entry", "Stop", "Risk/Share", "Quantity",
            "Capital Required", "Actual Risk", "2R Target", "Max Hold", "Score",
            "Technical Status", "Failed Conditions", "Earnings Status", "Earnings Date",
            "Days Until Earnings", "FINAL DECISION"
        ]
        display_cols = [c for c in display_cols if c in final_candidates.columns]
        st.dataframe(final_candidates[display_cols], use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Scanned", len(all_results))
    s2.metric("Data unavailable", len(failed))
    s3.metric("Score ≥ threshold", len(candidate_pool))
    s4.metric("Technical BUY", len(stage1_buy))
    s5.metric("Final BUY", int((final_buy["FINAL DECISION"] == "🟢 BUY CANDIDATE").sum()) if not final_buy.empty else 0)

    # --------------------------------------------------------
    # Excel files
    # --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    archive_dir = os.path.join("/tmp", "swing_scanner_archive")
    os.makedirs(archive_dir, exist_ok=True)

    all_path = os.path.join(archive_dir, f"All_NIFTY200_{timestamp}.xlsx")
    candidate_path = os.path.join(archive_dir, f"Candidates_{timestamp}.xlsx")

    # Complete scan keeps every successfully processed stock.
    all_results.to_excel(all_path, index=False)
    final_candidates.to_excel(candidate_path, index=False)

    # Latest copies in the running app environment.
    latest_all = os.path.join(archive_dir, "Latest_All.xlsx")
    latest_candidates = os.path.join(archive_dir, "Latest_Candidates.xlsx")
    all_results.to_excel(latest_all, index=False)
    final_candidates.to_excel(latest_candidates, index=False)

    # Downloads
    st.subheader("📥 Excel Files")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Complete NIFTY 200",
            data=open(all_path, "rb").read(),
            file_name=os.path.basename(all_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download Final Candidates",
            data=open(candidate_path, "rb").read(),
            file_name=os.path.basename(candidate_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Optional Drive upload if configured.
    st.subheader("☁️ Google Drive Archive")
    folder_id = st.text_input("Optional Google Drive folder ID", type="password")
    if st.button("Upload both Excel files to Drive"):
        if "gcp_service_account" not in st.secrets:
            st.warning("Drive is not configured yet. The app still lets you download the Excel files directly.")
        else:
            ok1, msg1 = drive_upload_file(all_path, folder_id.strip() or None)
            ok2, msg2 = drive_upload_file(candidate_path, folder_id.strip() or None)
            st.write(("✅ " if ok1 else "❌ ") + msg1)
            st.write(("✅ " if ok2 else "❌ ") + msg2)

    # --------------------------------------------------------
    # Parameter record for reproducibility.
    # --------------------------------------------------------
    st.subheader("⚙️ Scan Configuration")
    config = pd.DataFrame([{
        "Scan Time": timestamp,
        "Candle Used": all_results["Signal Date"].iloc[0],
        "Universe": universe,
        "Universe Count": len(symbols),
        "Capital": capital,
        "Risk %": risk_percent,
        "Risk Amount": RISK_AMOUNT_GLOBAL,
        "Candidate Score": candidate_score,
        "Buy Score": buy_score,
        "Earnings Window Days": earnings_days,
        "History": history_period,
        "Market Regime": "BULLISH" if market["bullish"] else "NOT BULLISH",
        "Data Source": "Yahoo Finance via yfinance",
    }])
    st.dataframe(config, use_container_width=True, hide_index=True)

    st.info(
        "Important: this is a screening system, not a guarantee of profit. "
        "Yahoo Finance is a third-party data source; verify final price/volume and material corporate events before placing a Zerodha order."
    )

else:
    st.info("Set your parameters in the sidebar and tap **RUN SCANNER**.")
    st.markdown(
        "### Default setup\n"
        "**Universe:** NIFTY 200  •  **Capital:** ₹20,000  •  **Risk:** 1% / ₹200  •  **History:** 5 years\n\n"
        "The app scans the last completed daily candle and calculates the strategy indicators itself from downloaded OHLCV data."
    )

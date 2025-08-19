# api/index.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from SmartApi import SmartConnect
import os, requests, json
from datetime import datetime, timedelta
import pyotp
import pandas as pd
import numpy as np
import logging, sys

# ----------------- Logging -----------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ----------------- Vercel writable dir -----------------
try:
    os.chdir("/tmp")
except Exception:
    pass

# ----------------- FastAPI app & CORS -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ----------------- Env vars (set in Vercel) -----------------
API_KEY           = os.environ.get("ANGEL_API_KEY")
ANGEL_USERNAME    = os.environ.get("ANGEL_USERNAME")
ANGEL_MPIN        = os.environ.get("ANGEL_MPIN")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET")

# ----------------- Constants -----------------
INSTRUMENT_LIST_PATH = "/tmp/instrument_list.json"
IST_OPEN  = "09:15"
IST_CLOSE = "15:30"
DEFAULT_DAYS = 730  # ~2 years (~500 sessions)

# ----------------- Utilities -----------------
def _require_creds():
    if not all([API_KEY, ANGEL_USERNAME, ANGEL_MPIN, ANGEL_TOTP_SECRET]):
        raise HTTPException(status_code=500, detail="Server configuration error: missing Angel Broking credentials.")

def _ist_str(dt: datetime, clock: str) -> str:
    return dt.strftime(f"%Y-%m-%d {clock}")

def get_instrument_list():
    if os.path.exists(INSTRUMENT_LIST_PATH):
        try:
            with open(INSTRUMENT_LIST_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        data = r.json()
        with open(INSTRUMENT_LIST_PATH, "w") as f:
            json.dump(data, f)
        return data
    except requests.RequestException as e:
        log.error(f"Instrument list download failed: {e}")
        raise HTTPException(status_code=503, detail=f"Could not download instrument list: {e}")

def get_token_from_symbol(symbol: str, exchange: str = "NSE") -> str:
    inst = get_instrument_list()
    s = symbol.upper()
    if exchange == "NSE":
        search = f"{s}-EQ"
        for it in inst:
            if it.get("symbol") == search and it.get("exch_seg") == "NSE":
                return it.get("token")
    elif exchange == "BSE":
        for it in inst:
            if it.get("symbol") == s and it.get("exch_seg") == "BSE":
                return it.get("token")
    log.error(f"Symbol '{symbol}' not found on {exchange}")
    raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'.")

def _aggregate_daily_to_weekly(candles):
    if not candles:
        return []
    # lazy import (daily path avoids this at cold start)
    import pandas as pd
    df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','volume'])
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    wk = pd.DataFrame({
        'open':   df['open'].resample('W-FRI').first(),
        'high':   df['high'].resample('W-FRI').max(),
        'low':    df['low'].resample('W-FRI').min(),
        'close':  df['close'].resample('W-FRI').last(),
        'volume': df['volume'].resample('W-FRI').sum(),
    }).dropna()
    wk.reset_index(inplace=True)
    wk['timestamp'] = wk['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S+05:30')
    return wk[['timestamp','open','high','low','close','volume']].values.tolist()

def _login():
    _require_creds()
    smart_api = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
    session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_MPIN, totp)
    if not session_data or session_data.get("status") is False:
        msg = (session_data or {}).get("message", "Unknown error during session generation.")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {msg}")
    return smart_api

def _fetch_candles(stock_symbol: str, exchange: str, days: int, interval: str):
    token = get_token_from_symbol(stock_symbol, exchange)
    to_dt = datetime.now()
    span = days if (days and days > 0) else DEFAULT_DAYS
    from_dt = to_dt - timedelta(days=span)

    # Always fetch daily; aggregate if weekly requested (SmartAPI weekly can be flaky)
    params = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": _ist_str(from_dt, IST_OPEN),
        "todate":   _ist_str(to_dt, IST_CLOSE),
    }
    log.info(f"getCandleData({stock_symbol},{exchange}) params: {params}")

    smart_api = _login()
    try:
        resp = smart_api.getCandleData(params)
    finally:
        try: smart_api.terminateSession(ANGEL_USERNAME)
        except Exception: pass

    if not resp or resp.get("status") is False:
        msg = (resp or {}).get("message", "unknown error")
        raise HTTPException(status_code=400, detail=f"Failed to fetch OHLC data: {msg}")

    data = resp.get("data") or []
    if interval == "ONE_WEEK":
        data = _aggregate_daily_to_weekly(data)
    return data

# ----------------- Indicators & TA helpers -----------------
def _to_df(data):
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df

def _sma(s, n): return s.rolling(n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(close, fast=12, slow=26, signal=9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _bb(close, n=20, k=2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = ma + k*sd
    lower = ma - k*sd
    width = (upper - lower) / ma
    return ma, upper, lower, width

def _stoch(df, k=14, d=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    k_line = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def _atr(df, n=14):
    hl = (df['high'] - df['low']).abs()
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _with_indicators(df):
    out = df.copy()
    out['sma20']  = _sma(out['close'], 20)
    out['sma50']  = _sma(out['close'], 50)
    out['sma200'] = _sma(out['close'], 200)
    out['rsi14']  = _rsi(out['close'], 14)
    macd, sig, hist = _macd(out['close'])
    out['macd'] = macd; out['macd_signal'] = sig; out['macd_hist'] = hist
    ma, up, lo, width = _bb(out['close'], 20, 2.0)
    out['bb_ma'] = ma; out['bb_up'] = up; out['bb_lo'] = lo; out['bb_width'] = width
    k, d = _stoch(out); out['stoch_k'] = k; out['stoch_d'] = d
    out['atr14'] = _atr(out, 14)
    out['vol_ma20'] = _sma(out['volume'], 20)
    return out

# ----------------- Strategy logic -----------------
def strat_golden_cross(df):
    if len(df) < 200: return {"signal":"not_enough_data"}
    df = _with_indicators(df)
    if pd.isna(df['sma50'].iloc[-1]) or pd.isna(df['sma200'].iloc[-1]):
        return {"signal":"not_enough_data"}
    prev50, prev200 = df['sma50'].iloc[-2], df['sma200'].iloc[-2]
    last50, last200 = df['sma50'].iloc[-1], df['sma200'].iloc[-1]
    sig = "buy" if (prev50 <= prev200 and last50 > last200) else ("sell" if (prev50 >= prev200 and last50 < last200) else "none")
    return {"signal": sig, "ma50": float(last50), "ma200": float(last200)}

def strat_rsi_reversal(df):
    df = _with_indicators(df)
    rsi = df['rsi14']
    # find last cross from <30 to >30
    crossed = (rsi.shift(1) < 30) & (rsi >= 30)
    if crossed.iloc[-50:].any():
        return {"signal":"buy", "rsi": float(rsi.iloc[-1])}
    # optional note if >70
    if rsi.iloc[-1] > 70:
        return {"signal":"sell_note", "rsi": float(rsi.iloc[-1])}
    return {"signal":"none", "rsi": float(rsi.iloc[-1])}

def strat_pullback_ma(df):
    df = _with_indicators(df)
    c = df['close'].iloc[-1]; sma50 = df['sma50'].iloc[-1]; sma200 = df['sma200'].iloc[-1]
    if pd.isna(sma50) or pd.isna(sma200): return {"signal":"not_enough_data"}
    cond_trend = c > sma200
    near50 = abs(c - sma50) / sma50 <= 0.05 if sma50 else False
    bounce = c > df['close'].iloc[-2]  # simple bounce proxy
    sig = "buy" if (cond_trend and near50 and bounce) else "none"
    return {"signal": sig, "close": float(c), "sma50": float(sma50), "sma200": float(sma200)}

def strat_macd_rsi_swing(df):
    df = _with_indicators(df)
    macd_val = float(df['macd'].iloc[-1])
    macd_sig = float(df['macd_signal'].iloc[-1])
    rsi_val  = float(df['rsi14'].iloc[-1])

    if macd_val > macd_sig and rsi_val > 50:
        sig = "buy"
    elif macd_val < macd_sig and rsi_val < 50:
        sig = "sell"
    else:
        sig = "none"

    return {
        "signal": sig,
        "metrics": {
            "macd": macd_val,
            "macd_signal": macd_sig,
            "rsi": rsi_val
        }
    }

def strat_bb_squeeze_breakout(df):
    df = _with_indicators(df)
    # squeeze = bb width in last 100 days is low (<= 20th percentile), breakout = close > upper band today
    recent = df.tail(100)
    if recent.empty: return {"signal":"not_enough_data"}
    thresh = np.nanpercentile(recent['bb_width'], 20)
    squeeze = df['bb_width'].iloc[-1] <= thresh
    breakout = df['close'].iloc[-1] > df['bb_up'].iloc[-1]
    sig = "buy" if (squeeze and breakout) else "none"
    return {"signal": sig, "bb_width": float(df['bb_width'].iloc[-1])}

def strat_pullback_vol_midcap(df):
    # NOTE: market cap filter (₹5K–₹30K Cr) NOT applied here (OHLC-only service). GPT can add that via web step.
    df = _with_indicators(df)
    c = df['close'].iloc[-1]; sma50 = df['sma50'].iloc[-1]; rsi = df['rsi14'].iloc[-1]
    near50 = pd.notna(sma50) and (abs(c - sma50)/sma50 <= 0.05)
    vol_ok = pd.notna(df['vol_ma20'].iloc[-1]) and (df['volume'].iloc[-1] > 1.5 * df['vol_ma20'].iloc[-1])
    rsi_band = pd.notna(rsi) and (40 <= rsi <= 60)
    sig = "buy" if (near50 and vol_ok and rsi_band) else "none"
    return {"signal": sig, "note": "market_cap_filter_not_applied"}

def strat_inside_bar_breakout(df):
    if len(df) < 3: return {"signal":"not_enough_data"}
    df = df.copy()
    # Inside bar on day -2 relative to -3
    ib = (df['high'].shift(1) < df['high'].shift(2)) & (df['low'].shift(1) > df['low'].shift(2))
    # Breakout on latest day above inside high with volume > 3-day avg
    vol3 = df['volume'].rolling(3).mean()
    breakout = (df['close'] > df['high'].shift(1)) & (df['volume'] > vol3)
    sig = "buy" if (ib.iloc[-1] and breakout.iloc[-1]) else "none"
    return {"signal": sig}

STRATEGY_FUNCS = {
    "golden_cross": strat_golden_cross,
    "rsi_reversal": strat_rsi_reversal,
    "pullback_ma": strat_pullback_ma,
    "macd_rsi_swing": strat_macd_rsi_swing,
    "bb_squeeze_breakout": strat_bb_squeeze_breakout,
    "pullback_vol_midcap": strat_pullback_vol_midcap,  # market-cap filter not applied
    "inside_bar_breakout": strat_inside_bar_breakout,
}

# ----------------- Endpoints -----------------
@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/get-ohlc")
def get_ohlc_endpoint(stock_symbol: str,
                      exchange: str = "NSE",
                      days: int = DEFAULT_DAYS,
                      interval: str = "ONE_DAY"):
    try:
        data = _fetch_candles(stock_symbol, exchange, days, interval)
        return {"status": "success", "symbol": stock_symbol.upper(), "interval": interval, "data": data}
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Unexpected error in get_ohlc_endpoint")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

@app.get("/api/strategyscan")
def strategyscan_endpoint(symbols: str,
                          strategy: str,
                          exchange: str = "NSE",
                          days: int = 300,
                          interval: str = "ONE_DAY"):
    """
    Example:
      /api/strategyscan?symbols=SBIN,RELIANCE&strategy=golden_cross
      NOTE: For golden_cross you should have >=250 days.
    """
    strategy = strategy.strip().lower()
    if strategy not in STRATEGY_FUNCS:
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy}' not supported.")
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="No symbols provided.")

    results = []
    for s in syms:
        try:
            data = _fetch_candles(s, exchange, days, interval)
            df = _to_df(data)
            out = STRATEGY_FUNCS[strategy](df)
            results.append({"symbol": s, "strategy": strategy, "result": out})
        except Exception as e:
            log.warning(f"strategyscan failed for {s}: {e}")
            results.append({"symbol": s, "strategy": strategy, "result": {"signal":"error","error":str(e)}})

    return {"status": "success", "results": results}

@app.get("/api/momentumscan")
def momentumscan_endpoint(symbols: str,
                          exchange: str = "NSE",
                          days: int = 300,
                          interval: str = "ONE_DAY"):
    """
    Simple momentum scan near 52w high + RSI>60 + vol strong
    """
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="No symbols provided.")

    results = []
    for s in syms:
        try:
            data = _fetch_candles(s, exchange, days, interval)
            df = _to_df(data)
            if len(df) < 250:
                results.append({"symbol": s, "signal":"not_enough_data"}); continue
            hi52 = df['high'].iloc[-250:].max()
            close = df['close'].iloc[-1]
            rsi = _rsi(df['close']).iloc[-1]
            vol_strong = df['volume'].iloc[-1] > df['volume'].iloc[-20:].mean() * 1.2
            near = close >= hi52 * 0.95
            sig = "buy" if (near and (rsi > 60) and vol_strong) else "none"
            results.append({"symbol": s, "signal": sig, "near_52w_high": bool(near), "rsi": float(rsi), "close": float(close)})
        except Exception as e:
            log.warning(f"momentumscan failed for {s}: {e}")
            results.append({"symbol": s, "signal":"error","error":str(e)})

    return {"status": "success", "scan_type": "momentum", "results": results}

@app.get("/api/swingsetup")
def swingsetup_endpoint(stock_symbol: str,
                        exchange: str = "NSE",
                        days: int = 400,
                        interval: str = "ONE_DAY"):
    """
    Compact summary for a single ticker (trend + momentum hints).
    Weekly needs a longer lookback (~1500 days) to warm up SMA200/W.
    """
    try:
        # 1) Ensure enough history for WEEKLY (SMA200 weekly needs ~200 weeks)
        if interval == "ONE_WEEK":
            min_days_for_weekly = 1500  # ~ 4 years calendar to be safe
            if not days or days < min_days_for_weekly:
                days = min_days_for_weekly

        # 2) Fetch and compute indicators
        data = _fetch_candles(stock_symbol, exchange, days, interval)
        df = _with_indicators(_to_df(data))

        # 3) Require the key fields to be non-null; otherwise report not_enough_data
        needed = ['sma20','sma50','sma200','rsi14','macd','macd_signal']
        dfx = df.dropna(subset=needed)
        if dfx.empty:
            return {
                "status": "success",
                "symbol": stock_symbol.upper(),
                "interval": interval,
                "trend": "insufficient",
                "momentum": "insufficient",
                "levels": {},
                "note": "not_enough_data: extend lookback (e.g., days=1500 for weekly) to warm up indicators"
            }

        last = dfx.iloc[-1]

        trend = (
            "uptrend" if last['sma20'] > last['sma50'] > last['sma200']
            else "downtrend" if last['sma20'] < last['sma50'] < last['sma200']
            else "mixed"
        )
        momentum = (
            "bullish" if (last['macd'] > last['macd_signal'] and last['rsi14'] > 50)
            else "bearish" if (last['macd'] < last['macd_signal'] and last['rsi14'] < 50)
            else "neutral"
        )

        # Optional squeeze note (guard with enough history)
        note = ""
        if len(df) >= 100 and df['bb_width'].notna().tail(100).any():
            try:
                recent = df['bb_width'].tail(100).dropna()
                if not recent.empty:
                    import numpy as np
                    if df['bb_width'].iloc[-1] <= np.nanpercentile(recent, 20):
                        note = "BB squeeze"
            except Exception:
                pass

        return {
            "status": "success",
            "symbol": stock_symbol.upper(),
            "interval": interval,
            "trend": trend,
            "momentum": momentum,
            "levels": {
                "sma20": float(last['sma20']),
                "sma50": float(last['sma50']),
                "sma200": float(last['sma200']),
                "rsi14": float(last['rsi14']),
                "macd": float(last['macd']),
                "macd_signal": float(last['macd_signal'])
            },
            "note": note
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Unexpected error in swingsetup")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

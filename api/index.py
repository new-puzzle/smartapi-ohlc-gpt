from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from SmartApi import SmartConnect
import os, requests, json
from datetime import datetime, timedelta
import pyotp
import pandas as pd
import traceback
import logging, sys

# -------- Logging --------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# -------- Vercel writable dir (SmartAPI likes to write logs) --------
os.chdir("/tmp")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------- Env vars expected on Vercel --------
API_KEY           = os.environ.get("ANGEL_API_KEY")
API_SECRET        = os.environ.get("ANGEL_API_SECRET")  # not used here, but OK to keep
ANGEL_USERNAME    = os.environ.get("ANGEL_USERNAME")
ANGEL_MPIN        = os.environ.get("ANGEL_MPIN")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET")

# -------- Constants --------
INSTRUMENT_LIST_PATH = "/tmp/instrument_list.json"
IST_OPEN  = "09:15"
IST_CLOSE = "15:30"
DEFAULT_DAYS = 730  # ~2 years (≈500 trading sessions)

# -------- Helpers --------
def get_instrument_list():
    if os.path.exists(INSTRUMENT_LIST_PATH):
        with open(INSTRUMENT_LIST_PATH, "r") as f:
            return json.load(f)

    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        with open(INSTRUMENT_LIST_PATH, "w") as f:
            json.dump(data, f)
        return data
    except requests.RequestException as e:
        log.error(f"Could not download instrument list: {e}")
        raise HTTPException(status_code=503, detail=f"Could not download instrument list: {e}")

def get_token_from_symbol(symbol: str, exchange: str = "NSE"):
    """
    NSE equities use 'SYMBOL-EQ' in the instrument list; BSE uses plain 'SYMBOL'.
    We must also match exch_seg.
    """
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

    log.error(f"Symbol '{symbol}' not found on exchange '{exchange}'.")
    raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'.")

def _ist_str(dt: datetime, clock: str) -> str:
    # SmartAPI accepts "YYYY-MM-DD HH:MM" as a string; we supply IST trading times explicitly.
    return dt.strftime(f"%Y-%m-%d {clock}")

def get_ohlc_data_internal(stock_symbol: str,
                           exchange: str = "NSE",
                           days: int = DEFAULT_DAYS,
                           interval: str = "ONE_DAY"):
    log.info(f"get_ohlc_data_internal: symbol={stock_symbol}, exch={exchange}, days={days}, interval={interval}")

    if not all([API_KEY, ANGEL_USERNAME, ANGEL_MPIN, ANGEL_TOTP_SECRET]):
        log.error("Missing Angel Broking API credentials.")
        raise HTTPException(status_code=500, detail="Server configuration error: Missing one or more Angel Broking API credentials.")

    try:
        # ---- Login ----
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        token = get_token_from_symbol(stock_symbol, exchange)

        smart_api = SmartConnect(api_key=API_KEY)
        session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_MPIN, totp)
        log.info(f"Session status: {session_data.get('status')}")

        if not session_data or session_data.get("status") is False:
            msg = session_data.get("message", "Unknown error during session generation.")
            raise HTTPException(status_code=401, detail=f"Authentication Failed: {msg}")

        # ---- Date window (default ~2 years) ----
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days if (days and days > 0) else DEFAULT_DAYS)

        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": interval,  # "ONE_DAY" or "ONE_WEEK"
            "fromdate": _ist_str(from_dt, IST_OPEN),
            "todate":   _ist_str(to_dt, IST_CLOSE),
        }
        log.info(f"getCandleData params: {params}")

        resp = smart_api.getCandleData(params)
        smart_api.terminateSession(ANGEL_USERNAME)

        if not resp or resp.get("status") is False:
            msg = (resp or {}).get("message", "unknown error")
            raise HTTPException(status_code=400, detail=f"Failed to fetch OHLC data: {msg}")

        return resp.get("data") or []

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unexpected error in get_ohlc_data_internal")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

# -------- Simple TA examples retained --------
def calculate_golden_cross(data):
    if not data or len(data) < 200:
        return {"signal": "not_enough_data", "detail": f"Need at least 200 days; got {len(data)}."}
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['close'] = pd.to_numeric(df['close'])
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    if pd.isna(df['ma50'].iloc[-1]) or pd.isna(df['ma200'].iloc[-1]):  # still warming up
        return {"signal": "not_enough_data", "detail": "MA warm-up not complete."}
    prev50, prev200 = df['ma50'].iloc[-2], df['ma200'].iloc[-2]
    last50, last200 = df['ma50'].iloc[-1], df['ma200'].iloc[-1]
    if prev50 <= prev200 and last50 > last200:
        return {"signal": "buy", "ma50": last50, "ma200": last200}
    if prev50 >= prev200 and last50 < last200:
        return {"signal": "sell", "ma50": last50, "ma200": last200}
    return {"signal": "none", "ma50": last50, "ma200": last200}

def calculate_momentum_scan(data_list):
    results = []
    for item in data_list:
        symbol = item["symbol"]; data = item["data"]
        if not data or len(data) < 250:
            results.append({"symbol": symbol, "signal": "not_enough_data"}); continue
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
        df['close'] = pd.to_numeric(df['close']); df['volume'] = pd.to_numeric(df['volume'])
        hi52 = df['high'].iloc[-250:].max(); close = df['close'].iloc[-1]
        near = (close >= hi52*0.95)
        delta = df['close'].diff()
        gain = (delta.where(delta>0,0)).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rsi = 100 - (100/(1 + (gain/(loss.replace(0, pd.NA)))))
        vstrong = df['volume'].iloc[-1] > df['volume'].iloc[-20:].mean()*1.2
        sig = "buy" if (near and (rsi.iloc[-1] > 60) and vstrong) else "none"
        results.append({"symbol": symbol, "signal": sig, "near_52w_high": near,
                        "current_rsi": float(rsi.iloc[-1]), "current_close": float(close)})
    return results

# -------- Endpoints --------
@app.get("/api/get-ohlc")
def get_ohlc_endpoint(stock_symbol: str,
                      exchange: str = "NSE",
                      days: int = DEFAULT_DAYS,
                      interval: str = "ONE_DAY"):
    try:
        data = get_ohlc_data_internal(stock_symbol, exchange, days, interval)
        return {"status": "success", "symbol": stock_symbol.upper(), "data": data}
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Unexpected error in get_ohlc_endpoint")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

@app.get("/api/strategyscan")
def run_strategy_scan(stock_symbol: str, strategy: str, exchange: str = "NSE"):
    if strategy.lower() != "golden_cross":
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy}' not supported. Only 'golden_cross'.")
    try:
        data = get_ohlc_data_internal(stock_symbol, exchange, days=250, interval="ONE_DAY")
        return {"status": "success", "symbol": stock_symbol.upper(), "strategy": strategy,
                "result": calculate_golden_cross(data)}
    except HTTPException as e:
        raise e
    except Exception as e:
        log.exception("Unexpected error in run_strategy_scan")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

@app.get("/api/momentumscan")
def run_momentum_scan(symbols: str, exchange: str = "NSE"):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="No symbols provided.")
    all_data = []
    for s in syms:
        try:
            d = get_ohlc_data_internal(s, exchange, days=250, interval="ONE_DAY")
            all_data.append({"symbol": s.upper(), "data": d})
        except Exception as e:
            log.warning(f"Fetch failed for {s}: {e}")
            all_data.append({"symbol": s.upper(), "data": []})
    return {"status": "success", "scan_type": "momentum", "results": calculate_momentum_scan(all_data)}

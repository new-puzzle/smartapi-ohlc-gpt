from fastapi import FastAPI, HTTPException
from SmartApi import SmartConnect
import os
import requests
import json
from datetime import datetime, timedelta
import pyotp
import pandas as pd
import traceback
import logging
import sys

# --- Setup logging ---
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
log = logging.getLogger(__name__)

# --- Directory Trick ---
os.chdir("/tmp")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from the ChatGPT UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables ---
API_KEY = os.environ.get("ANGEL_API_KEY")
API_SECRET = os.environ.get("ANGEL_API_SECRET")
ANGEL_USERNAME = os.environ.get("ANGEL_USERNAME")
ANGEL_MPIN = os.environ.get("ANGEL_MPIN")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET")

INSTRUMENT_LIST_PATH = "/tmp/instrument_list.json"

def get_instrument_list():
    if os.path.exists(INSTRUMENT_LIST_PATH):
        with open(INSTRUMENT_LIST_PATH, 'r') as f:
            return json.load(f)
        
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        instrument_list = response.json()
        with open(INSTRUMENT_LIST_PATH, 'w') as f:
            json.dump(instrument_list, f)
        return instrument_list
    except requests.RequestException as e:
        return {"status": "error", "error": f"Could not download instrument list: {e}"}

def get_token_from_symbol(symbol: str, exchange: str = "NSE"):
    instrument_list = get_instrument_list()
    search_symbol = f"{symbol.upper()}-EQ"
    for instrument in instrument_list:
        if instrument.get("symbol") == search_symbol and instrument.get("exch_seg") == exchange:
            return instrument.get("token")
    raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'. Please check the symbol and try again.")

def get_ohlc_data_internal(stock_symbol: str, exchange: str = "NSE", days: int = 30):
    if not all([API_KEY, API_SECRET, ANGEL_USERNAME, ANGEL_MPIN, ANGEL_TOTP_SECRET]):
        raise HTTPException(status_code=500, detail="Server configuration error: Missing one or more Angel Broking API credentials.")
    try:
        instrument_list = get_instrument_list()
        search_symbol = f"{symbol.upper()}-EQ"
        for instrument in instrument_list:
            if instrument.get("symbol") == search_symbol and instrument.get("exch_seg") == exchange:
                return instrument.get("token")
        log.error(f"Symbol '{symbol}' not found on {exchange}.")
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'")
    except Exception as e:
        log.exception("Error in get_token_from_symbol")
        raise

@app.get("/api/get-ohlc")
def get_ohlc_data(stock_symbol: str, exchange: str = "NSE", days: int = 30):
    log.info(f"get_ohlc_data called for symbol={stock_symbol}, exchange={exchange}, days={days}")
    
    if not all([API_KEY, API_SECRET, ANGEL_USERNAME, ANGEL_MPIN, ANGEL_TOTP_SECRET]):
        return {"status": "error", "error": "Missing Angel Broking API credentials in environment variables."}

    try:
        # Generate TOTP
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        symbol_token = get_token_from_symbol(stock_symbol, exchange)
        smart_api = SmartConnect(api_key=API_KEY)
        session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_MPIN, totp)
        
        if not session_data or session_data.get("status") is False:
            error_message = session_data.get("message", "Unknown error")
            raise HTTPException(status_code=401, detail=f"Authentication Failed: {error_message}")
        log.debug("TOTP generated.")

        # Symbol token
        symbol_token = get_token_from_symbol(stock_symbol, exchange)
        log.debug(f"Symbol token retrieved: {symbol_token}")

        # Smart API connect
        smart_api = SmartConnect(api_key=API_KEY)
        log.debug("SmartConnect initialized.")

        session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_MPIN, totp)
        log.debug(f"Session generation status: {session_data.get('status')}")

        if not session_data or session_data.get("status") is False:
            error_message = session_data.get("message", "Unknown error during session generation.")
            return {"status": "error", "error": f"Authentication Failed: {error_message}"}

        # Fetch OHLC
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        historic_params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        ohlc_data = smart_api.getCandleData(historic_params)
        smart_api.terminateSession(ANGEL_USERNAME)

        if ohlc_data.get("status") is False:
             raise HTTPException(status_code=400, detail=f"Failed to fetch OHLC data: {ohlc_data.get('message')}")
        log.debug(f"getCandleData status: {ohlc_data.get('status')}")
        smart_api.terminateSession(ANGEL_USERNAME)
        log.debug("Session terminated.")

        if ohlc_data.get("status") is False:
            return {"status": "error", "error": f"Failed to fetch OHLC: {ohlc_data.get('message')}"}

        return ohlc_data.get("data")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

def calculate_golden_cross(data):
    if not data or len(data) < 200:
        return {"signal": "not_enough_data", "detail": f"Need at least 200 days of data, but got {len(data)}."}

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = pd.to_numeric(df['close'])
    
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['ma200'] = df['close'].rolling(window=200).mean()
    
    last_ma50 = df['ma50'].iloc[-1]
    last_ma200 = df['ma200'].iloc[-1]
    prev_ma50 = df['ma50'].iloc[-2]
    prev_ma200 = df['ma200'].iloc[-2]

    if pd.isna(last_ma50) or pd.isna(last_ma200) or pd.isna(prev_ma50) or pd.isna(prev_ma200):
        return {"signal": "not_enough_data", "detail": "Could not calculate moving averages for the full period."}

    signal = "none"
    if prev_ma50 <= prev_ma200 and last_ma50 > last_ma200:
        signal = "buy"
    elif prev_ma50 >= prev_ma200 and last_ma50 < last_ma200:
        signal = "sell" # Death Cross

    return {"signal": signal, "ma50": last_ma50, "ma200": last_ma200}

def calculate_momentum_scan(data_list):
    results = []
    for stock_data in data_list:
        symbol = stock_data["symbol"]
        data = stock_data["data"]

        if not data or len(data) < 250: # Need enough data for 52W high and RSI
            results.append({"symbol": symbol, "signal": "not_enough_data", "detail": f"Not enough data for {symbol}."})
            continue

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])

        # Calculate 52-week high (approx 250 trading days)
        high_52_weeks = df['high'].iloc[-250:].max()
        current_close = df['close'].iloc[-1]
        
        # Check if trading near 52W high (e.g., within 5%)
        near_52w_high = (current_close >= high_52_weeks * 0.95)

        # Calculate RSI (14-period RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Check strong volume (e.g., above 20D average)
        avg_volume_20d = df['volume'].iloc[-20:].mean()
        current_volume = df['volume'].iloc[-1]
        strong_volume = (current_volume > avg_volume_20d * 1.2) # 20% above avg

        signal = "none"
        if near_52w_high and current_rsi > 60 and strong_volume:
            signal = "buy"
        
        results.append({"symbol": symbol, "signal": signal, "near_52w_high": near_52w_high, "current_rsi": current_rsi, "strong_volume": strong_volume, "52w_high_value": high_52_weeks, "current_close": current_close})
    return results

@app.get("/api/get-ohlc")
def get_ohlc_endpoint(stock_symbol: str, exchange: str = "NSE", days: int = 30):
    try:
        ohlc_data = get_ohlc_data_internal(stock_symbol, exchange, days)
        return {"status": "success", "symbol": stock_symbol, "data": ohlc_data}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@app.get("/api/strategyscan")
def run_strategy_scan(stock_symbol: str, strategy: str, exchange: str = "NSE"):
    if strategy.lower() != "golden_cross":
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy}' is not supported. Currently, only 'golden_cross' is available.")

    try:
        # Need at least 200 days for golden cross, plus some buffer
        ohlc_data = get_ohlc_data_internal(stock_symbol, exchange, days=250)
        
        result = calculate_golden_cross(ohlc_data)
        
        return {"status": "success", "symbol": stock_symbol, "strategy": strategy, "result": result}

    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[CRITICAL ERROR] An unexpected error occurred in strategyscan: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during strategy scan: {str(e)}.")

@app.get("/api/momentumscan")
def run_momentum_scan(symbols: str, exchange: str = "NSE"):
    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided for momentum scan.")

    all_ohlc_data = []
    for symbol in symbol_list:
        try:
            # Request enough days for 52W high and RSI calculation
            ohlc_data = get_ohlc_data_internal(symbol, exchange, days=250)
            all_ohlc_data.append({"symbol": symbol, "data": ohlc_data})
        except HTTPException as e:
            # Log the error but continue with other symbols
            print(f"[WARNING] Could not fetch data for {symbol}: {e.detail}")
            all_ohlc_data.append({"symbol": symbol, "data": [], "error": e.detail})
        except Exception as e:
            print(f"[WARNING] Unexpected error fetching data for {symbol}: {str(e)}")
            all_ohlc_data.append({"symbol": symbol, "data": [], "error": str(e)})

    scan_results = calculate_momentum_scan(all_ohlc_data)
    
    return {"status": "success", "scan_type": "momentum", "results": scan_results}


        return {"status": "error", "error": e.detail}
    except Exception as e:
        tb = traceback.format_exc()
        log.critical(f"Unexpected error: {str(e)}\n{tb}")
        return {"status": "error", "error": str(e), "traceback": tb}


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
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- THE DIRECTORY TRICK ---
# Change the current working directory to the only writable one
os.chdir("/tmp")
# --------------------------

app = FastAPI()

# --- CORS Middleware ---
from fastapi.middleware.cors import CORSMiddleware

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
        log.error(f"Could not download instrument list: {e}")
        raise HTTPException(status_code=503, detail=f"Could not download instrument list: {e}")

def get_token_from_symbol(symbol: str, exchange: str = "NSE"):
    instrument_list = get_instrument_list()
    search_symbol = f"{symbol.upper()}-EQ"
    for instrument in instrument_list:
        if instrument.get("symbol") == search_symbol and instrument.get("exch_seg") == exchange:
            return instrument.get("token")
    log.error(f"Symbol '{symbol}' not found on exchange '{exchange}'.")
    raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'. Please check the symbol and try again.")

def get_ohlc_data_internal(stock_symbol: str, exchange: str = "NSE", days: int = 30):
    log.info(f"get_ohlc_data_internal called for {stock_symbol}, requesting {days} days.")
    
    if not all([API_KEY, API_SECRET, ANGEL_USERNAME, ANGEL_MPIN, ANGEL_TOTP_SECRET]):
        log.error("Missing Angel Broking API credentials.")
        raise HTTPException(status_code=500, detail="Server configuration error: Missing one or more Angel Broking API credentials.")

    try:
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        log.debug("TOTP generated.")
            
        symbol_token = get_token_from_symbol(stock_symbol, exchange)
        log.debug(f"Symbol token retrieved: {symbol_token}")
            
        smart_api = SmartConnect(api_key=API_KEY)
        log.debug("SmartConnect initialized.")
            
        session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_MPIN, totp)
        log.debug(f"Session generation attempt. Status: {session_data.get("status")}")
            
        if not session_data or session_data.get("status") is False:
            error_message = session_data.get("message", "Unknown error during session generation.")
            log.error(f"Authentication Failed: {error_message}")
            raise HTTPException(status_code=401, detail=f"Authentication Failed: {error_message}")

        all_ohlc_data = []
        current_to_date = datetime.now()
        
        # Fetch data in chunks to overcome the observed ~115-day limit
        # We will fetch in chunks of 115 days until we get enough data or reach the requested 'days'
        
        CHUNK_SIZE = 115 # Observed max days returned per call
        days_to_fetch_total = days
        
        while days_to_fetch_total > 0:
            # Calculate from_date for the current chunk
            chunk_from_date = current_to_date - timedelta(days=min(days_to_fetch_total, CHUNK_SIZE))
            
            historic_params = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": "ONE_DAY",
                "fromdate": chunk_from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": current_to_date.strftime("%Y-%m-%d %H:%M")
            }
            log.info(f"Sending historic_params to getCandleData (chunk): {historic_params}")
                
            ohlc_data_chunk = smart_api.getCandleData(historic_params)
            log.info(f"Raw response from getCandleData (chunk): {ohlc_data_chunk}")

            if ohlc_data_chunk.get("status") is False:
                log.error(f"Failed to fetch OHLC data chunk: {ohlc_data_chunk.get('message')}")
                raise HTTPException(status_code=400, detail=f"Failed to fetch OHLC data chunk: {ohlc_data_chunk.get('message')}")
            
            chunk_data = ohlc_data_chunk.get("data")
            if not chunk_data: # No more data in this chunk, or reached end of available history
                log.info(f"No data returned for chunk ending {current_to_date.strftime('%Y-%m-%d')}. Breaking loop.")
                break

            # Prepend new data to maintain chronological order
            all_ohlc_data = chunk_data + all_ohlc_data
            
            # Update for next iteration
            days_fetched_in_chunk = len(chunk_data)
            days_to_fetch_total -= days_fetched_in_chunk
            current_to_date = chunk_from_date - timedelta(days=1) # Move to the day before the start of the last chunk
            
            # If we fetched less than the CHUNK_SIZE, it means we hit the end of available history
            if days_fetched_in_chunk < CHUNK_SIZE:
                log.info(f"Fetched {days_fetched_in_chunk} days in chunk, less than CHUNK_SIZE {CHUNK_SIZE}. Assuming end of history.")
                break

        smart_api.terminateSession(ANGEL_USERNAME)
        log.debug("Session terminated.")

        return all_ohlc_data

    except HTTPException as e:
        log.error(f"HTTPException in get_ohlc_data_internal: {e.detail}")
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        log.critical(f"""An unexpected error occurred in get_ohlc_data_internal: {str(e)}
{error_traceback}""")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}. Please check server logs for details.")

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
        log.error(f"HTTPException in get_ohlc_endpoint: {e.detail}")
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        log.critical(f"""An unexpected error occurred in get_ohlc_endpoint: {str(e)}
{error_traceback}""")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}. Please check server logs for details.")


@app.get("/api/strategyscan")
def run_strategy_scan(stock_symbol: str, strategy: str, exchange: str = "NSE"):
    log.info(f"run_strategy_scan called for symbol={stock_symbol}, strategy={strategy}")
    if strategy.lower() != "golden_cross":
        log.warning(f"Unsupported strategy: {strategy}")
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy}' is not supported. Currently, only 'golden_cross' is available.")

    try:
        # Need at least 200 days for golden cross, plus some buffer
        ohlc_data = get_ohlc_data_internal(stock_symbol, exchange, days=250)
        
        result = calculate_golden_cross(ohlc_data)
        
        return {"status": "success", "symbol": stock_symbol, "strategy": strategy, "result": result}

    except HTTPException as e:
        log.error(f"HTTPException in run_strategy_scan: {e.detail}")
        raise e
    except Exception as e:
        error_traceback = traceback.format_exc()
        log.critical(f"""An unexpected error occurred in run_strategy_scan: {str(e)}
{error_traceback}""")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during strategy scan: {str(e)}. Please check server logs for details.")

@app.get("/api/momentumscan")
def run_momentum_scan(symbols: str, exchange: str = "NSE"):
    log.info(f"run_momentum_scan called for symbols={symbols}")
    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
    if not symbol_list:
        log.warning("No symbols provided for momentum scan.")
        raise HTTPException(status_code=400, detail="No symbols provided for momentum scan.")

    all_ohlc_data = []
    for symbol in symbol_list:
        try:
            # Request enough days for 52W high and RSI calculation
            ohlc_data = get_ohlc_data_internal(symbol, exchange, days=250)
            all_ohlc_data.append({"symbol": symbol, "data": ohlc_data})
        except HTTPException as e:
            # Log the error but continue with other symbols
            log.warning(f"Could not fetch data for {symbol}: {e.detail}")
            all_ohlc_data.append({"symbol": symbol, "data": [], "error": e.detail})
        except Exception as e:
            error_traceback = traceback.format_exc()
            log.warning(f"""Unexpected error fetching data for {symbol}: {str(e)}
{error_traceback}""")
            all_ohlc_data.append({"symbol": symbol, "data": [], "error": str(e)})

    scan_results = calculate_momentum_scan(all_ohlc_data)
    
    return {"status": "success", "scan_type": "momentum", "results": scan_results}

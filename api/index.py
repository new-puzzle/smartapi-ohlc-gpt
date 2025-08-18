from fastapi import FastAPI, HTTPException
from SmartApi import SmartConnect
import os
import requests
import json
from datetime import datetime, timedelta
import pyotp
import traceback
import logging
import sys

# --- Setup logging ---
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
log = logging.getLogger(__name__)

# --- Directory Trick ---
os.chdir("/tmp")

app = FastAPI()

# --- Environment Variables ---
API_KEY = os.environ.get("ANGEL_API_KEY")
API_SECRET = os.environ.get("ANGEL_API_SECRET")
ANGEL_USERNAME = os.environ.get("ANGEL_USERNAME")
ANGEL_MPIN = os.environ.get("ANGEL_MPIN")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET")
CUSTOM_GPT_API_KEY = os.environ.get("CUSTOM_GPT_API_KEY")

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
        log.debug(f"getCandleData status: {ohlc_data.get('status')}")
        smart_api.terminateSession(ANGEL_USERNAME)
        log.debug("Session terminated.")

        if ohlc_data.get("status") is False:
            return {"status": "error", "error": f"Failed to fetch OHLC: {ohlc_data.get('message')}"}

        return {"status": "success", "symbol": stock_symbol, "data": ohlc_data.get("data")}

    except HTTPException as e:
        log.error(f"HTTPException: {e.detail}")
        return {"status": "error", "error": e.detail}
    except Exception as e:
        tb = traceback.format_exc()
        log.critical(f"Unexpected error: {str(e)}\n{tb}")
        return {"status": "error", "error": str(e), "traceback": tb}

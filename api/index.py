from fastapi import FastAPI, HTTPException
from smartapi import SmartConnect
import os
import requests
import json
from datetime import datetime, timedelta
import pyotp # <-- IMPORT THE NEW LIBRARY

app = FastAPI()

# --- Environment Variables (to be set in Vercel) ---
API_KEY = os.environ.get("ANGEL_API_KEY")
API_SECRET = os.environ.get("ANGEL_API_SECRET")
ANGEL_USERNAME = os.environ.get("ANGEL_USERNAME")
ANGEL_PASSWORD = os.environ.get("ANGEL_PASSWORD")
# --- THIS IS THE NEW, PERMANENT SECRET KEY ---
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
        raise HTTPException(status_code=503, detail=f"Could not download instrument list: {e}")

def get_token_from_symbol(symbol: str, exchange: str = "NSE"):
    instrument_list = get_instrument_list()
    search_symbol = f"{symbol.upper()}-EQ"
    for instrument in instrument_list:
        if instrument.get("symbol") == search_symbol and instrument.get("exch_seg") == exchange:
            return instrument.get("token")
    raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'.")

@app.get("/api/get-ohlc")
def get_ohlc_data(stock_symbol: str, exchange: str = "NSE", days: int = 30):
    if not all([API_KEY, API_SECRET, ANGEL_USERNAME, ANGEL_PASSWORD, ANGEL_TOTP_SECRET]):
        raise HTTPException(status_code=500, detail="Server configuration error: Missing one or more API credentials in Vercel settings.")

    try:
        # --- GENERATE TOTP AUTOMATICALLY ---
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET)
        current_totp = totp.now()
        # ------------------------------------

        symbol_token = get_token_from_symbol(stock_symbol, exchange)
        smart_api = SmartConnect(api_key=API_KEY)
            
        # --- Use the generated TOTP for login ---
        session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_PASSWORD, current_totp)
            
        if not session_data or session_data.get("status") is False:
            error_message = session_data.get("message", "Unknown error during session generation.")
            raise HTTPException(status_code=401, detail=f"Authentication Failed: {error_message}")

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

        return {"status": "success", "symbol": stock_symbol, "data": ohlc_data.get("data")}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

from fastapi import FastAPI, HTTPException, Header, Depends
from SmartApi import SmartConnect
import os
import requests
import json
from datetime import datetime, timedelta
import pyotp

# --- THE DIRECTORY TRICK ---
# Change the current working directory to the only writable one
os.chdir("/tmp")
# --------------------------

app = FastAPI()

# --- Environment Variables ---
API_KEY = os.environ.get("ANGEL_API_KEY")
API_SECRET = os.environ.get("ANGEL_API_SECRET")
ANGEL_USERNAME = os.environ.get("ANGEL_USERNAME")
ANGEL_MPIN = os.environ.get("ANGEL_MPIN")
ANGEL_TOTP_SECRET = os.environ.get("ANGEL_TOTP_SECRET")
CUSTOM_GPT_API_KEY = os.environ.get("CUSTOM_GPT_API_KEY")

# --- API Key Authentication Dependency ---
# async def verify_api_key(x_api_key: str = Header(...)):
#     if not CUSTOM_GPT_API_KEY:
#         raise HTTPException(status_code=500, detail="Server configuration error: CUSTOM_GPT_API_KEY not set.")
#     if x_api_key != CUSTOM_GPT_API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid API Key")
#     return x_api_key
# --- END API Key Authentication Dependency ---
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
        raise HTTPException(status_code=503, detail=f"Could not download instrument list: {e}")

def get_token_from_symbol(symbol: str, exchange: str = "NSE"):
    instrument_list = get_instrument_list()
    search_symbol = f"{symbol.upper()}-EQ"
    for instrument in instrument_list:
        if instrument.get("symbol") == search_symbol and instrument.get("exch_seg") == exchange:
            return instrument.get("token")
    print(f"[DEBUG] get_token_from_symbol called for symbol: {symbol}, exchange: {exchange}")
    instrument_list = get_instrument_list()
    print(f"[DEBUG] Instrument list loaded. Size: {len(instrument_list) if instrument_list else 0}")
    search_symbol = f"{symbol.upper()}-EQ"
    for instrument in instrument_list:
        if instrument.get("symbol") == search_symbol and instrument.get("exch_seg") == exchange:
            print(f"[DEBUG] Found token: {instrument.get("token")}")
            return instrument.get("token")
    print(f"[DEBUG] Symbol '{symbol}' not found after search.")
    raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found on exchange '{exchange}'. Please check the symbol and try again.")

@app.get("/api/get-ohlc")
def get_ohlc_data(stock_symbol: str, exchange: str = "NSE", days: int = 30):
    print(f"[DEBUG] get_ohlc_data called for {stock_symbol}")
    
    if not all([API_KEY, API_SECRET, ANGEL_USERNAME, ANGEL_MPIN, ANGEL_TOTP_SECRET]):
        raise HTTPException(status_code=500, detail="Server configuration error: Missing one or more Angel Broking API credentials in Vercel settings. Please ensure all are set.")

    try:
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        print(f"[DEBUG] TOTP generated.")
            
        symbol_token = get_token_from_symbol(stock_symbol, exchange)
        print(f"[DEBUG] Symbol token retrieved: {symbol_token}")
            
        # This will now create its 'logs' folder inside /tmp, which is allowed
        smart_api = SmartConnect(api_key=API_KEY)
        print(f"[DEBUG] SmartConnect initialized.")
            
        session_data = smart_api.generateSession(ANGEL_USERNAME, ANGEL_MPIN, totp)
        print(f"[DEBUG] Session generation attempt. Status: {session_data.get("status")}")
            
        if not session_data or session_data.get("status") is False:
            error_message = session_data.get("message", "Unknown error during session generation.")
            raise HTTPException(status_code=401, detail=f"Authentication Failed with Angel Broking: {error_message}. Please check your Angel Broking credentials.")

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        print(f"[DEBUG] Fetching data from {from_date.strftime("%Y-%m-%d %H:%M")} to {to_date.strftime("%Y-%m-%d %H:%M")}")
            
        historic_params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": "ONE_DAY",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }
            
        ohlc_data = smart_api.getCandleData(historic_params)
        print(f"[DEBUG] getCandleData attempt. Status: {ohlc_data.get("status")}")
        smart_api.terminateSession(ANGEL_USERNAME)
        print(f"[DEBUG] Session terminated.")

        if ohlc_data.get("status") is False:
             raise HTTPException(status_code=400, detail=f"Failed to fetch OHLC data from Angel Broking: {ohlc_data.get('message')}. Please check parameters or Angel Broking status.")

        return {"status": "success", "symbol": stock_symbol, "data": ohlc_data.get("data")}

    except HTTPException as e:
        # Re-raise HTTPException directly as they are expected errors
        print(f"[ERROR] HTTPException caught: {e.detail}")
        raise e
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"[CRITICAL ERROR] An unexpected error occurred: {str(e)}
{error_traceback}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}. Please check server logs for details.")


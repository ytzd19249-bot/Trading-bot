import os
import logging
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
import ccxt
import psycopg2
import requests
from datetime import datetime

# === CONFIGURACIÓN INICIAL ===
app = FastAPI()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("varestin_trading_bot")

# === VARIABLES DE ENTORNO ===
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
BROKER_ENV = os.getenv("BROKER_ENV", "real")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
PUBLIC_URL = os.getenv("PUBLIC_URL")
SCHEDULE_MINUTES = int(os.getenv("SCHEDULE_MINUTES", "15"))
TRADE_PERCENTAGE = float(os.getenv("TRADE_PERCENTAGE", "0.10"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.02"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.04"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT").split(",")
MIN_VOLUME_USDT = float(os.getenv("MIN_VOLUME_USDT", "50"))

# === CONEXIÓN BINANCE ===
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(DRY_RUN)

logger.info(f"🔗 Conectado a Binance ({'MODO SIMULACIÓN' if DRY_RUN else 'REAL'})")

# === CONEXIÓN A LA BASE DE DATOS ===
def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"❌ Error conectando a la base de datos: {e}")
        return None

# === ENVIAR MENSAJE TELEGRAM ===
def send_telegram_message(msg: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": ADMIN_CHAT_ID, "text": msg}
        requests.post(url, data=data)
    except Exception as e:
        logger.warning(f"No se pudo enviar mensaje a Telegram: {e}")

# === MODELO PARA ENDPOINT MANUAL ===
class TradeRequest(BaseModel):
    symbol: str
    side: str  # "buy" o "sell"
    amount: float

# === FUNCIÓN DE ANÁLISIS ===
async def analyze_and_trade():
    logger.info("🔍 Analizando mercado...")
    for symbol in SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=50)
            closes = [c[4] for c in ohlcv]
            last_close = closes[-1]
            avg = sum(closes) / len(closes)

            if last_close > avg * 1.01:
                signal = "BUY"
            elif last_close < avg * 0.99:
                signal = "SELL"
            else:
                signal = "HOLD"

            logger.info(f"{symbol} → Señal: {signal} | Precio: {last_close:.2f}")
            if not DRY_RUN and signal in ["BUY", "SELL"]:
                balance = exchange.fetch_balance()
                usdt_balance = balance['total'].get('USDT', 0)
                trade_amount = (usdt_balance * TRADE_PERCENTAGE) / last_close
                if trade_amount * last_close >= MIN_VOLUME_USDT:
                    order = exchange.create_market_order(symbol, signal.lower(), trade_amount)
                    logger.info(f"✅ Orden ejecutada: {order}")
                    send_telegram_message(f"💹 {signal} {symbol} @ {last_close:.2f}")
        except Exception as e:
            logger.error(f"Error analizando {symbol}: {e}")

# === ENDPOINTS FASTAPI ===
@app.get("/")
async def root():
    return {"status": "running", "mode": "simulation" if DRY_RUN else "real"}

@app.get("/health")
async def health():
    return {"ok": True, "env": BROKER_ENV, "symbols": SYMBOLS}

@app.post("/manual_trade")
async def manual_trade(req: TradeRequest):
    try:
        if DRY_RUN:
            return {"msg": f"(Simulado) {req.side.upper()} {req.amount} {req.symbol}"}
        order = exchange.create_market_order(req.symbol, req.side, req.amount)
        return {"msg": f"Orden ejecutada: {order}"}
    except Exception as e:
        return {"error": str(e)}

# === TAREA PROGRAMADA ===
async def schedule_loop():
    while True:
        await analyze_and_trade()
        await asyncio.sleep(SCHEDULE_MINUTES * 60)

# === MAIN RUN ===
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(schedule_loop())
    send_telegram_message("🚀 Bot de Trading Varestin iniciado correctamente")

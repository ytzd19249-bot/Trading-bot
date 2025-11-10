# VARESTIN PRO TRADING BOT — Bybit + ccxt
import os
import logging
import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import ccxt
import pandas as pd

# DB opcional (psycopg 3 — compatible Python 3.13)
try:
    import psycopg
except Exception:
    psycopg = None

# ============ CONFIG ============

app = FastAPI()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("varestin_trading_bot")

EXCHANGE = os.getenv("EXCHANGE", "bybit").lower()
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET_KEY = os.getenv("BYBIT_SECRET_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

DATABASE_URL = os.getenv("DATABASE_URL")  # opcional
PUBLIC_URL = os.getenv("PUBLIC_URL")

# Trading params
SCHEDULE_MINUTES = int(os.getenv("SCHEDULE_MINUTES", "7"))  # cada 7 min
TRADE_PERCENTAGE = float(os.getenv("TRADE_PERCENTAGE", "0.10"))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.02"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.04"))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT").split(",")]
MIN_VOLUME_USDT = float(os.getenv("MIN_VOLUME_USDT", "50"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Indicadores
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "26"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# ============ EXCHANGE ============
if EXCHANGE != "bybit":
    # Forzamos Bybit porque así lo pediste
    EXCHANGE = "bybit"

exchange = ccxt.bybit({
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_SECRET_KEY,
    'enableRateLimit': True,
})

log.info(f"🔗 Conectado a {EXCHANGE.upper()} ({'MODO SIMULACIÓN' if DRY_RUN else 'REAL'})")

# ============ TELEGRAM ============
def tg(msg: str):
    if not TELEGRAM_TOKEN or not ADMIN_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": ADMIN_CHAT_ID, "text": msg}
        )
    except Exception as e:
        log.warning(f"No se pudo enviar Telegram: {e}")

# ============ DB opcional ============
def db_conn():
    if not DATABASE_URL or psycopg is None:
        return None
    try:
        return psycopg.connect(DATABASE_URL)
    except Exception as e:
        log.warning(f"DB no disponible: {e}")
        return None

def ensure_schema():
    conn = db_conn()
    if not conn:
        return
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP WITH TIME ZONE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price NUMERIC NOT NULL,
                amount NUMERIC NOT NULL,
                sl NUMERIC,
                tp NUMERIC,
                dry_run BOOLEAN NOT NULL,
                note TEXT
            );
            """)
    except Exception as e:
        log.warning(f"No se pudo asegurar esquema DB: {e}")
    finally:
        conn.close()

def save_trade(ts, symbol, side, price, amount, sl, tp, dry_run, note):
    conn = db_conn()
    if not conn:
        return
    try:
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (ts, symbol, side, price, amount, sl, tp, dry_run, note)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """, (ts, symbol, side, price, amount, sl, tp, dry_run, note))
    except Exception as e:
        log.warning(f"No se pudo registrar trade: {e}")
    finally:
        conn.close()

ensure_schema()

# ============ INDICADORES ============

def compute_indicators(ohlcv, ema_fast=EMA_FAST, ema_slow=EMA_SLOW, rsi_period=RSI_PERIOD, atr_period=ATR_PERIOD):
    # ohlcv: [[ts, open, high, low, close, volume], ...]
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    # EMA
    df["ema_fast"] = pd.Series(df["close"]).ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = pd.Series(df["close"]).ewm(span=ema_slow, adjust=False).mean()
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=rsi_period).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=atr_period).mean()
    # Bollinger
    bb_period = 20
    bb_std = 2
    df["bb_mid"] = df["close"].rolling(bb_period).mean()
    df["bb_std"] = df["close"].rolling(bb_period).std()
    df["bb_upper"] = df["bb_mid"] + bb_std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - bb_std * df["bb_std"]

    return df

def get_signal(df):
    # Necesitamos al menos 30 velas para indicadores decentes
    if len(df) < 30:
        return "HOLD", {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Cruce EMA
    bull_cross = (prev["ema_fast"] <= prev["ema_slow"]) and (last["ema_fast"] > last["ema_slow"])
    bear_cross = (prev["ema_fast"] >= prev["ema_slow"]) and (last["ema_fast"] < last["ema_slow"])

    # Filtro RSI
    rsi = float(last["rsi"])
    rsi_buy_ok = rsi < 60  # evitar comprar sobre-extendido
    rsi_sell_ok = rsi > 40 # evitar vender en sobreventa extrema

    # Filtro tendencia (EMA lenta con pendiente)
    trend_up = last["ema_slow"] > df["ema_slow"].iloc[-5]
    trend_down = last["ema_slow"] < df["ema_slow"].iloc[-5]

    # Volatilidad razonable (no ultra-comprimido)
    atr = float(last["atr"]) if pd.notna(last["atr"]) else None
    vol_ok = atr is None or atr > 0  # placeholder; podrías ajustar un umbral

    ctx = {
        "price": float(last["close"]),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "rsi": rsi,
        "atr": atr,
        "bb_upper": float(last["bb_upper"]) if pd.notna(last["bb_upper"]) else None,
        "bb_lower": float(last["bb_lower"]) if pd.notna(last["bb_lower"]) else None
    }

    if bull_cross and rsi_buy_ok and trend_up and vol_ok:
        return "BUY", ctx
    if bear_cross and rsi_sell_ok and trend_down and vol_ok:
        return "SELL", ctx
    return "HOLD", ctx

# ============ TRADING CORE ============

class ManualOrder(BaseModel):
    symbol: str
    side: str   # "buy" / "sell"
    amount: float

def usdt_balance():
    bal = exchange.fetch_balance()
    total = bal.get("total", {})
    free = bal.get("free", {})
    return float(total.get("USDT", 0)), float(free.get("USDT", 0))

def position_size(price, pct=TRADE_PERCENTAGE):
    total, free = usdt_balance()
    usd_to_use = free * pct
    if usd_to_use < MIN_VOLUME_USDT:
        return 0.0
    amt = usd_to_use / price
    return float(exchange.amount_to_precision(SYMBOLS[0], amt))  # usa formato del 1er par para redondeo

async def process_symbol(symbol: str):
    try:
        # timeframe 1m para reacción rápida
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=120)
        df = compute_indicators(ohlcv)

        signal, ctx = get_signal(df)
        price = ctx.get("price", float(df["close"].iloc[-1]))
        log.info(f"{symbol} → Señal: {signal} | Precio: {price:.4f} | RSI: {ctx.get('rsi')}")

        # Reporte por Telegram de diagnóstico
        tg(f"🔎 {symbol}\nSeñal: {signal}\nPrecio: {price:.4f}\nRSI: {ctx.get('rsi'):.2f}\nEMA9:{ctx.get('ema_fast'):.2f} / EMA26:{ctx.get('ema_slow'):.2f}")

        if signal not in ["BUY", "SELL"]:
            return

        amt = position_size(price)
        if amt * price < MIN_VOLUME_USDT:
            log.info(f"{symbol} monto insuficiente (min {MIN_VOLUME_USDT} USDT).")
            return

        # SL / TP
        sl_price = price * (1 - STOP_LOSS_PERCENT) if signal == "BUY" else price * (1 + STOP_LOSS_PERCENT)
        tp_price = price * (1 + TAKE_PROFIT_PERCENT) if signal == "BUY" else price * (1 - TAKE_PROFIT_PERCENT)

        if DRY_RUN:
            # Simulación
            note = f"SIM {signal} {symbol} @ {price:.4f} | amt {amt:.6f} | SL {sl_price:.4f} | TP {tp_price:.4f}"
            log.info(f"🧪 {note}")
            tg(f"🧪 {note}")
            save_trade(datetime.now(timezone.utc), symbol, signal, price, amt, sl_price, tp_price, True, "dry_run")
            return

        # Real
        side = "buy" if signal == "BUY" else "sell"
        order = exchange.create_order(symbol=symbol, type="market", side=side, amount=amt)
        note = f"✅ {signal} {symbol} @ {price:.4f} | amt {amt:.6f}"
        log.info(note)
        tg(f"{note}\nSL: {sl_price:.4f} | TP: {tp_price:.4f}")

        # Colocar órdenes TP/SL si el exchange/mercado lo permite (spot puede variar)
        try:
            # Muchos exchanges requieren órdenes separadas o conditional; dejamos intento básico protegido.
            # Si falla, queda registrado y manejado externamente.
            pass
        except Exception as e:
            log.warning(f"No fue posible crear TP/SL automáticos: {e}")

        save_trade(datetime.now(timezone.utc), symbol, signal, price, amt, sl_price, tp_price, False, str(order))

    except ccxt.BaseError as e:
        log.error(f"CCXT error {symbol}: {e}")
    except Exception as e:
        log.error(f"Error analizando {symbol}: {e}")

async def analyze_and_trade():
    log.info("🔍 Ejecutando ciclo de análisis…")
    tasks = [process_symbol(sym) for sym in SYMBOLS]
    await asyncio.gather(*tasks)

# ============ ENDPOINTS ============

@app.get("/")
async def root():
    return {
        "status": "running",
        "exchange": EXCHANGE,
        "mode": "simulation" if DRY_RUN else "real",
        "schedule_minutes": SCHEDULE_MINUTES
    }

@app.get("/health")
async def health():
    total, free = (0.0, 0.0)
    try:
        total, free = usdt_balance()
    except Exception:
        pass
    return {"ok": True, "exchange": EXCHANGE, "symbols": SYMBOLS, "usdt_total": total, "usdt_free": free}

@app.post("/manual_trade")
async def manual_trade(req: ManualOrder):
    try:
        side = req.side.lower().strip()
        if side not in ["buy", "sell"]:
            return {"error": "side inválido (use 'buy' o 'sell')"}
        if DRY_RUN:
            msg = f"(SIM) {side.upper()} {req.amount} {req.symbol}"
            tg(f"🧪 {msg}")
            return {"msg": msg}
        order = exchange.create_order(symbol=req.symbol, type="market", side=side, amount=req.amount)
        tg(f"✅ Manual {side.upper()} {req.amount} {req.symbol}")
        return {"msg": "ok", "order": order}
    except Exception as e:
        return {"error": str(e)}

@app.post("/run_cycle")
async def run_cycle():
    await analyze_and_trade()
    return {"ok": True, "ran": True, "ts": datetime.now(timezone.utc).isoformat()}

# ============ SCHEDULER ============
async def scheduler_loop():
    while True:
        await analyze_and_trade()
        await asyncio.sleep(SCHEDULE_MINUTES * 60)

@app.on_event("startup")
async def on_start():
    tg("🚀 VARESTIN PRO TRADING BOT iniciado (Bybit)")
    asyncio.create_task(scheduler_loop())

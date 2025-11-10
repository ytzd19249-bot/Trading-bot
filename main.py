import os
import time
import math
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import requests
import pandas as pd

import ccxt
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# -----------------------------
# Configuración y logging
# -----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("trading-bot")

# -----------------------------
# Env vars (ajustá en Render)
# -----------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# 'real' (por defecto) o 'testnet'
BROKER_ENV = os.getenv("BROKER_ENV", "real").lower()  # 'real' | 'testnet'
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT").split(",")

DATABASE_URL = os.getenv("DATABASE_URL", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")

SCHEDULE_MINUTES = int(os.getenv("SCHEDULE_MINUTES", "15"))
REPORT_EVERY_HOURS = int(os.getenv("REPORT_EVERY_HOURS", "12"))

# Gestión de riesgo
TRADE_PERCENTAGE = float(os.getenv("TRADE_PERCENTAGE", "0.10"))  # 10% del saldo por trade
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.02"))  # 2%
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "0.04"))  # 4%

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"  # Si querés probar con llaves reales pero sin ejecutar órdenes

# -----------------------------
# App FastAPI
# -----------------------------
app = FastAPI(title="Bot de Trading Varestin", version="1.0.0")

# -----------------------------
# DB
# -----------------------------
def get_engine() -> Engine:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL no configurada")
    return create_engine(DATABASE_URL, pool_pre_ping=True)

engine = get_engine()

def ensure_schema() -> None:
    with engine.begin() as conn:
        # Tabla de trades
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS trades (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL, -- BUY / SELL
            amount NUMERIC,
            entry_price NUMERIC,
            exit_price NUMERIC,
            status TEXT NOT NULL, -- OPEN / CLOSED / CANCELLED
            sl NUMERIC, -- stop loss
            tp NUMERIC, -- take profit
            pnl NUMERIC, -- ganancia/pérdida al cerrar
            opened_at TIMESTAMP WITH TIME ZONE,
            closed_at TIMESTAMP WITH TIME ZONE,
            raw_order JSONB
        );
        """))
        # Índices útiles
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status);"))
        # Tabla de ejecuciones/heartbeat para reportes
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS bot_runs (
            id BIGSERIAL PRIMARY KEY,
            run_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            symbols TEXT,
            decisions JSONB
        );
        """))

ensure_schema()

# -----------------------------
# Telegram util
# -----------------------------
def send_telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not ADMIN_CHAT_ID:
        log.warning("TELEGRAM_TOKEN/ADMIN_CHAT_ID no configurados; omitiendo notificación.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": ADMIN_CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            log.warning("Fallo enviando Telegram: %s %s", r.status_code, r.text)
    except Exception as e:
        log.exception("Error Telegram: %s", e)

# -----------------------------
# Exchange (ccxt)
# -----------------------------
def get_exchange() -> ccxt.binance:
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET_KEY,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot"
        }
    })
    # Testnet si el broker env está en testnet
    if BROKER_ENV == "testnet":
        try:
            exchange.set_sandbox_mode(True)
            # ccxt maneja urls de testnet automáticamente en sandbox_mode
        except Exception as e:
            log.warning("No se pudo habilitar sandbox_mode: %s", e)
    return exchange

# -----------------------------
# Indicadores técnicos
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera df con columnas: ['timestamp','open','high','low','close','volume']
    Agrega: ema9, ema21, rsi14, atr14
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = up / (down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ATR(14)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()

    return df

# -----------------------------
# Estrategia / Señales
# -----------------------------
class Decision(BaseModel):
    symbol: str
    action: str  # "BUY" / "SELL" / "HOLD"
    price: float
    amount: float = 0.0
    reason: str = ""
    sl: Optional[float] = None
    tp: Optional[float] = None

def generate_signal(df: pd.DataFrame, symbol: str) -> Decision:
    """
    Reglas simples y robustas:
    - BUY: cruce EMA9 por encima de EMA21 + RSI < 65 y volumen razonable.
    - SELL: si hay posición abierta, cierra por TP/SL o cruce a la baja EMA9<EMA21 o RSI>75.
    - HOLD: lo demás.
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema9 = float(last["ema9"])
    ema21 = float(last["ema21"])
    ema9_prev = float(prev["ema9"])
    ema21_prev = float(prev["ema21"])
    rsi = float(last["rsi14"])
    price = float(last["close"])

    # ¿Hay posición abierta?
    open_pos = get_open_position(symbol)

    if open_pos:
        # Reglas de cierre
        sl = float(open_pos["sl"]) if open_pos["sl"] is not None else None
        tp = float(open_pos["tp"]) if open_pos["tp"] is not None else None
        entry = float(open_pos["entry_price"])

        # Stop loss / Take profit por precio
        if sl and price <= sl:
            return Decision(symbol=symbol, action="SELL", price=price, reason=f"StopLoss alcanzado ({sl})")
        if tp and price >= tp:
            return Decision(symbol=symbol, action="SELL", price=price, reason=f"TakeProfit alcanzado ({tp})")

        # Señal técnica de salida
        if ema9 < ema21 or rsi > 75:
            return Decision(symbol=symbol, action="SELL", price=price, reason="Señal técnica de salida (EMA9<EMA21 o RSI>75)")

        # Mantener
        return Decision(symbol=symbol, action="HOLD", price=price, reason="Mantener posición")

    else:
        # Posible compra
        crossed_up = (ema9_prev <= ema21_prev) and (ema9 > ema21)
        if crossed_up and rsi < 65:
            # Definir SL/TP dinámicos con ATR
            atr = float(last["atr14"]) if not math.isnan(float(last["atr14"])) else price * 0.01
            sl = price * (1 - STOP_LOSS_PERCENT) if STOP_LOSS_PERCENT > 0 else price - 1.5 * atr
            tp = price * (1 + TAKE_PROFIT_PERCENT) if TAKE_PROFIT_PERCENT > 0 else price + 2.0 * atr
            return Decision(symbol=symbol, action="BUY", price=price, reason="Cruce EMA y RSI favorable", sl=sl, tp=tp)

        return Decision(symbol=symbol, action="HOLD", price=price, reason="Sin señal de entrada")

# -----------------------------
# DB helpers de posiciones/trades
# -----------------------------
def get_open_position(symbol: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT id, symbol, side, amount, entry_price, sl, tp, opened_at
            FROM trades
            WHERE symbol=:s AND status='OPEN'
            ORDER BY opened_at DESC
            LIMIT 1;
        """), {"s": symbol}).mappings().first()
        return dict(row) if row else None

def open_trade(symbol: str, side: str, amount: float, entry_price: float, sl: Optional[float], tp: Optional[float], raw_order: Dict[str, Any]) -> int:
    with engine.begin() as conn:
        row = conn.execute(text("""
            INSERT INTO trades(symbol, side, amount, entry_price, status, sl, tp, opened_at, raw_order)
            VALUES(:symbol, :side, :amount, :entry_price, 'OPEN', :sl, :tp, NOW(), CAST(:raw AS JSONB))
            RETURNING id;
        """), {
            "symbol": symbol, "side": side, "amount": amount, "entry_price": entry_price,
            "sl": sl, "tp": tp, "raw": json.dumps(raw_order or {})
        }).first()
        return int(row[0])

def close_trade(trade_id: int, exit_price: float, raw_order: Dict[str, Any]) -> None:
    with engine.begin() as conn:
        # Calcular PnL
        row = conn.execute(text("SELECT side, amount, entry_price FROM trades WHERE id=:id;"), {"id": trade_id}).mappings().first()
        if not row:
            return
        side = row["side"]
        amount = float(row["amount"])
        entry = float(row["entry_price"])
        # Para BUY primero: PnL = (exit - entry) * amount ; Si fuera short, invertido
        pnl = (exit_price - entry) * amount if side.upper() == "BUY" else (entry - exit_price) * amount
        conn.execute(text("""
            UPDATE trades
            SET exit_price=:exit_price, status='CLOSED', pnl=:pnl, closed_at=NOW(), raw_order = CAST(:raw AS JSONB)
            WHERE id=:id;
        """), {"exit_price": exit_price, "pnl": pnl, "id": trade_id, "raw": json.dumps(raw_order or {})})

def record_run(symbols: List[str], decisions: List[Decision]) -> None:
    payload = [d.dict() for d in decisions]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_runs(symbols, decisions) VALUES (:symbols, CAST(:decisions AS JSONB));
        """), {"symbols": ",".join(symbols), "decisions": json.dumps(payload)})

# -----------------------------
# Helpers de trading
# -----------------------------
def fetch_ohlcv_df(exchange: ccxt.binance, symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return compute_indicators(df)

def get_free_balance(exchange: ccxt.binance, ticker_quote: str = "USDT") -> float:
    bal = exchange.fetch_balance()
    free = bal.get("free", {}) or bal.get("total", {})
    return float(free.get(ticker_quote, 0.0))

def amount_for_buy(exchange: ccxt.binance, symbol: str, price: float) -> float:
    # Invertimos % del saldo en la divisa quote (ej: USDT)
    quote_ccy = symbol.split("/")[1]
    free = get_free_balance(exchange, quote_ccy)
    budget = free * TRADE_PERCENTAGE
    if budget <= 0:
        return 0.0
    amt = budget / price
    # Ajuste de precisión según mercado
    market = exchange.market(symbol)
    amount_precision = market.get("precision", {}).get("amount", 6)
    step = 10 ** (-amount_precision)
    amt = math.floor(amt / step) * step
    return max(0.0, amt)

# -----------------------------
# Núcleo de ejecución por ciclo
# -----------------------------
async def run_cycle() -> Dict[str, Any]:
    exchange = get_exchange()
    decisions: List[Decision] = []
    executed_actions: List[Dict[str, Any]] = []

    for raw_symbol in SYMBOLS:
        symbol = raw_symbol.strip().upper()
        try:
            df = fetch_ohlcv_df(exchange, symbol, TIMEFRAME, limit=250)
            if len(df) < 50:
                log.info("Pocas velas para %s", symbol)
                continue

            decision = generate_signal(df, symbol)
            decisions.append(decision)

            # Ejecutar decisión
            if decision.action == "BUY":
                # Si ya hay posición, no duplicar
                if get_open_position(symbol):
                    executed_actions.append({"symbol": symbol, "skipped": "Ya existe posición OPEN"})
                    continue

                amt = amount_for_buy(exchange, symbol, decision.price)
                if amt <= 0:
                    executed_actions.append({"symbol": symbol, "skipped": "Monto insuficiente"})
                    continue

                order_resp = {}
                if not DRY_RUN:
                    order_resp = exchange.create_market_buy_order(symbol, amt)
                trade_id = open_trade(symbol, "BUY", amt, decision.price, decision.sl, decision.tp, order_resp)
                executed_actions.append({"symbol": symbol, "buy": amt, "price": decision.price, "trade_id": trade_id, "reason": decision.reason})

            elif decision.action == "SELL":
                open_pos = get_open_position(symbol)
                if not open_pos:
                    executed_actions.append({"symbol": symbol, "skipped": "No hay posición para cerrar"})
                    continue
                amt = float(open_pos["amount"])
                order_resp = {}
                if not DRY_RUN:
                    order_resp = exchange.create_market_sell_order(symbol, amt)
                close_trade(open_pos["id"], decision.price, order_resp)
                executed_actions.append({"symbol": symbol, "sell": amt, "price": decision.price, "trade_id": open_pos["id"], "reason": decision.reason})

            else:
                executed_actions.append({"symbol": symbol, "hold": True, "reason": decision.reason})

            # Respetar rate limit
            time.sleep(exchange.rateLimit / 1000.0)

        except Exception as e:
            log.exception("Error en símbolo %s: %s", symbol, e)
            executed_actions.append({"symbol": symbol, "error": str(e)})

    # Guardar corrida
    record_run(SYMBOLS, decisions)
    return {"decisions": [d.dict() for d in decisions], "executed": executed_actions}

# -----------------------------
# Scheduler simple con asyncio
# -----------------------------
scheduler_running = False

async def scheduler_loop():
    global scheduler_running
    if scheduler_running:
        return
    scheduler_running = True
    log.info("Scheduler iniciado: cada %s minutos", SCHEDULE_MINUTES)
    while True:
        try:
            result = await run_cycle()
            log.info("Ciclo ejecutado: %s", json.dumps(result)[:800])
        except Exception as e:
            log.exception("Fallo en ciclo: %s", e)
        # Reporte cada X horas (aprox por tiempo)
        now = datetime.now(timezone.utc)
        if now.hour % REPORT_EVERY_HOURS == 0 and now.minute < (SCHEDULE_MINUTES // 2):
            try:
                send_periodic_report()
            except Exception:
                log.exception("Fallo al enviar reporte")
        await asyncio.sleep(SCHEDULE_MINUTES * 60)

def send_periodic_report():
    with engine.begin() as conn:
        # Últimas 24h
        rows = conn.execute(text("""
            SELECT symbol, COUNT(*) AS n, COALESCE(SUM(pnl),0) AS pnl_sum
            FROM trades
            WHERE opened_at >= NOW() - INTERVAL '24 hours'
            GROUP BY symbol
            ORDER BY symbol;
        """)).mappings().all()

        closed = conn.execute(text("""
            SELECT COUNT(*) AS closed, COALESCE(SUM(pnl),0) AS pnl
            FROM trades
            WHERE closed_at >= NOW() - INTERVAL '24 hours';
        """)).mappings().first()

    lines = [f"🧾 <b>Reporte {TIMEFRAME} (últ. 24h)</b>",
             f"• Pares: {', '.join(SYMBOLS)}",
             f"• Ambiente: {'TESTNET' if BROKER_ENV=='testnet' else 'REAL'} {'(DRY_RUN)' if DRY_RUN else ''}",
             "— — —"]
    for r in rows:
        lines.append(f"• {r['symbol']}: {int(r['n'])} ops | PnL: {float(r['pnl_sum']):.4f}")
    if closed:
        lines.append(f"• Cerradas 24h: {int(closed['closed'])} | PnL total: {float(closed['pnl']):.4f}")
    send_telegram("\n".join(lines))

# -----------------------------
# Endpoints
# -----------------------------
class RunResponse(BaseModel):
    decisions: Any
    executed: Any

@app.on_event("startup")
async def on_startup():
    # Arranca scheduler
    asyncio.create_task(scheduler_loop())
    send_telegram("✅ Bot de Trading Varestin iniciado.")

@app.get("/health")
def health():
    return {"ok": True, "env": BROKER_ENV, "timeframe": TIMEFRAME, "symbols": SYMBOLS}

@app.get("/status")
def status():
    with engine.begin() as conn:
        last_runs = conn.execute(text("SELECT id, run_at, symbols FROM bot_runs ORDER BY id DESC LIMIT 5;")).mappings().all()
        open_positions = conn.execute(text("SELECT id, symbol, side, amount, entry_price, sl, tp, opened_at FROM trades WHERE status='OPEN' ORDER BY opened_at DESC;")).mappings().all()
        last_closed = conn.execute(text("""
            SELECT id, symbol, side, amount, entry_price, exit_price, pnl, closed_at
            FROM trades
            WHERE status='CLOSED'
            ORDER BY closed_at DESC LIMIT 10;
        """)).mappings().all()
    return {
        "runs": [dict(r) for r in last_runs],
        "open_positions": [dict(r) for r in open_positions],
        "last_closed": [dict(r) for r in last_closed],
    }

@app.post("/run_cycle", response_model=RunResponse)
async def trigger_cycle(background_tasks: BackgroundTasks):
    result = await run_cycle()
    return result

class TelegramPing(BaseModel):
    message: Optional[str] = "Ping del admin"

@app.post("/admin/ping")
def admin_ping(body: TelegramPing):
    send_telegram(f"🔔 {body.message}")
    return {"sent": True}

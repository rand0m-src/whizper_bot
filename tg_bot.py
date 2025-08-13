#tg_bot.py
import os
import logging
import asyncio
import math
from typing import List, Tuple

import aiohttp
from aiohttp import ClientConnectorError
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Token only; API_BASE is no longer used
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Binance klines (free, no key)
BINANCE_BASES = [
    "https://api.binance.com/api/v3/klines",   # try global first
    "https://api.binance.us/api/v3/klines",    # fallback for U.S.
]

SYMBOLS: List[Tuple[str, str]] = [
    ("BTC", "BTCUSDT"),  # (label, api_symbol)
    ("ETH", "ETHUSDT"),
    ("SOL", "SOLUSDT")
    ]   
ALLOWED_INTERVALS: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
MAX_TELEGRAM_MSG = 4096

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("chart-whisperer-tg")

# -------------------- data + indicators (ported from main.py, minimal edits) --------------------
async def fetch_klines_async(sess: aiohttp.ClientSession, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err = None
    for base in BINANCE_BASES:
        try:
            async with sess.get(base, params=params) as r:
                data = await r.json()
                if r.status != 200:
                    # if 451 (geo-block), try next base immediately
                    if r.status == 451:
                        last_err = f"{base} -> 451 (geo-block)"
                        continue
                    raise RuntimeError(f"{base} {r.status}: {str(data)[:200]}")
            # success -> build dataframe
            cols = [
                "open_time","open","high","low","close","volume","close_time","qav",
                "num_trades","taker_buy_base","taker_buy_quote","ignore"
            ]
            df = pd.DataFrame(data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            return df[["open_time","open","high","low","close","volume","close_time"]]
        except Exception as e:
            last_err = e
            # try next base
            continue
    raise RuntimeError(f"All Binance bases failed: {last_err}")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr14"] = atr(df, 14)
    window = 120
    df["swing_high"] = df["high"].rolling(window).max()
    df["swing_low"] = df["low"].rolling(window).min()
    return df

def fib_levels(row):
    hi = row["swing_high"]
    lo = row["swing_low"]
    if pd.isna(hi) or pd.isna(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {
        "0.236": hi - 0.236 * diff,
        "0.382": hi - 0.382 * diff,
        "0.5": hi - 0.5 * diff,
        "0.618": hi - 0.618 * diff,
        "0.786": hi - 0.786 * diff,
    }

def trend_signal(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    if last["ema20"] > last["ema50"] and last["macd_hist"] > 0:
        return "bullish"
    if last["ema20"] < last["ema50"] and last["macd_hist"] < 0:
        return "bearish"
    return "neutral"

def rsi_state(val: float) -> str:
    if val >= 70:
        return "overbought"
    if val <= 30:
        return "oversold"
    return "neutral"

def generate_analysis(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    price = float(last["close"])
    tr = trend_signal(df)
    rsi_s = rsi_state(float(last["rsi14"]))
    fibs = fib_levels(last)
    atr_val = float(last["atr14"]) if not math.isnan(last["atr14"]) else None
    notes = []
    if tr == "bullish":
        notes.append("EMA20 > EMA50 and MACD momentum positive")
    elif tr == "bearish":
        notes.append("EMA20 < EMA50 and MACD momentum negative")
    else:
        notes.append("Mixed signals; momentum unclear")
    if rsi_s == "overbought":
        notes.append("RSI > 70; risk of pullback")
    elif rsi_s == "oversold":
        notes.append("RSI < 30; bounce potential")
    if atr_val:
        notes.append(f"ATR14 ≈ {atr_val:.2f}; expect ±{atr_val:.2f} range")
    return {
        "price": price,
        "trend": tr,
        "rsi_state": rsi_s,
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "macd_hist": float(last["macd_hist"]),
        "atr14": atr_val,
        "fibs": fibs,
        "notes": notes,
        "time": str(last["close_time"]),
    }

def build_report(symbol: str, interval: str, result: dict) -> str:
    lines = []
    lines.append(f"Whizper_BOT — {symbol} on {interval}")
    lines.append("─" * 48)
    lines.append(f"Time: {result['time']}")
    lines.append(f"Price: {result['price']:.2f}")
    lines.append("")
    lines.append(f"Trend: {result['trend'].upper()}  |  RSI14: {result['rsi_state'].upper()}")
    lines.append(f"EMA20: {result['ema20']:.2f}   EMA50: {result['ema50']:.2f}")
    lines.append(f"MACD: {result['macd']:.5f}  Signal: {result['macd_signal']:.5f}  Hist: {result['macd_hist']:.5f}")
    if result.get('atr14') is not None:
        lines.append(f"ATR14: {result['atr14']:.2f} (≈ expected intrvl range)")
    fibs = result.get('fibs') or {}
    if fibs:
        levels = ", ".join([f"{k}:{v:.2f}" for k, v in fibs.items()])
        lines.append(f"Fibs(120 bars): {levels}")
    if result.get('notes'):
        lines.append("")
        lines.append("Notes:")
        for n in result['notes']:
            lines.append(f" • {n}")
    lines.append("")
    lines.append("Trade Ideas (not advice):")
    if result['trend'] == 'bullish':
        lines.append(" • Consider pullback buys near EMA20/EMA50 confluence")
        lines.append(" • Watch RSI cooling from overbought for safer entries")
    elif result['trend'] == 'bearish':
        lines.append(" • Rallies into EMA20/EMA50 can be fade zones")
        lines.append(" • Confirm momentum with MACD histogram turning down")
    else:
        lines.append(" • Range tactics; wait for EMA cross or MACD shift")
    return "\n".join(lines)

# Build report end-to-end with retry (replaces old _fetch_text_report)
async def _fetch_text_report(context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str) -> str:
    sess = await _ensure_session(context)
    last_err = None
    for attempt in range(3):
        try:
            df = await fetch_klines_async(sess, symbol=symbol, interval=interval, limit=500)
            df = add_indicators(df)
            result = generate_analysis(df)
            return build_report(symbol, interval, result)
        except (ClientConnectorError, asyncio.TimeoutError) as e:
            last_err = e
            await asyncio.sleep(0.7 * (attempt + 1))
        except Exception as e:
            # non-network error (parse/indicator)
            raise RuntimeError(f"Report build failed: {e}") from e
    raise RuntimeError(f"Data source unreachable after retries: {last_err}")

# -------------------- http session --------------------
async def _ensure_session(context: ContextTypes.DEFAULT_TYPE) -> aiohttp.ClientSession:
    sess: aiohttp.ClientSession | None = context.application.bot_data.get("session")
    if sess is None or sess.closed:
        timeout = aiohttp.ClientTimeout(total=20)
        sess = aiohttp.ClientSession(timeout=timeout)
        context.application.bot_data["session"] = sess
    return sess

def _chunk(text: str, n: int = MAX_TELEGRAM_MSG) -> List[str]:
    return [text[i:i+n] for i in range(0, len(text), n)]

# -------------------- keyboards + copy --------------------
def _symbol_keyboard() -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(lbl, callback_data=f"sym|{sym}") for (lbl, sym) in SYMBOLS]
    return InlineKeyboardMarkup([buttons])

def _interval_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("1m", callback_data="int|1m"),
         InlineKeyboardButton("5m", callback_data="int|5m"),
         InlineKeyboardButton("15m", callback_data="int|15m")],
        [InlineKeyboardButton("1h", callback_data="int|1h"),
         InlineKeyboardButton("4h", callback_data="int|4h"),
         InlineKeyboardButton("1d", callback_data="int|1d")],
        [InlineKeyboardButton("« Change Coin", callback_data="nav|symbol")]
    ]
    return InlineKeyboardMarkup(rows)

def _action_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Refresh", callback_data="act|refresh")],
        [InlineKeyboardButton("🪙 Change Coin", callback_data="nav|symbol"),
         InlineKeyboardButton("⏱️ Change Interval", callback_data="nav|interval")]
    ])

WELCOME = (
    "🐸 Whizper Bot — So you need help? Fine. Here's the deal..\n\n"
    "Enter /start to wake me up:\n"
    "Tap the shiny buttons to choose your coin & timeframe.\n"
    "I’ll cough up my “analysis” — trend, RSI, EMAs, MACD, ATR, fib levels… and my unsolicited opinion.\n\n"
    "⏳ Supported intervals:\n"
    f"{', '.join(ALLOWED_INTERVALS)}\n\n"
    "💡 I’m sarcastic, slightly rude, but sometimes useful. Think of me as your caffeinated, chart-obsessed, crypto-degen friend."
)

# -------------------- commands --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    await update.message.reply_text("Wake up call received. Pick a coin:", reply_markup=_symbol_keyboard())

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)

# -------------------- callback flow --------------------
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    await q.answer()
    data = q.data or ""

    if data == "nav|symbol":
        context.user_data.pop("interval", None)
        await q.edit_message_text("Pick a coin:", reply_markup=_symbol_keyboard())
        return
    if data == "nav|interval":
        await q.edit_message_text("Pick a timeframe:", reply_markup=_interval_keyboard())
        return

    if data.startswith("sym|"):
        symbol = data.split("|", 1)[1]
        context.user_data["symbol"] = symbol
        await q.edit_message_text(f"Coin locked: {symbol}\nNow pick a timeframe:", reply_markup=_interval_keyboard())
        return

    if data.startswith("int|"):
        interval = data.split("|", 1)[1]
        if interval not in ALLOWED_INTERVALS:
            await q.edit_message_text("Bad interval. Try again:", reply_markup=_interval_keyboard())
            return
        context.user_data["interval"] = interval
        symbol = context.user_data.get("symbol", SYMBOLS[0][1])
        try:
            text = await _fetch_text_report(context, symbol, interval)
        except Exception as e:
            await q.edit_message_text(f"Couldn’t fetch report: {e}")
            return
        header = f"📈 Whispering {symbol} on {interval}:\n\n"
        chunks = _chunk(header + text)
        await q.edit_message_text(chunks[0], reply_markup=_action_keyboard())
        for more in chunks[1:]:
            await q.message.reply_text(more)
        return

    if data == "act|refresh":
        symbol = context.user_data.get("symbol", SYMBOLS[0][1])
        interval = context.user_data.get("interval", "1h")
        try:
            text = await _fetch_text_report(context, symbol, interval)
        except Exception as e:
            await q.edit_message_text(f"Couldn’t fetch report: {e}", reply_markup=_action_keyboard())
            return
        header = f"📈 Whispering {symbol} on {interval}:\n\n"
        chunks = _chunk(header + text)
        await q.edit_message_text(chunks[0], reply_markup=_action_keyboard())
        for more in chunks[1:]:
            await q.message.reply_text(more)
        return

# -------------------- lifecycle --------------------
async def on_shutdown(app: Application) -> None:
    sess: aiohttp.ClientSession | None = app.bot_data.get("session")
    if sess and not sess.closed:
        await sess.close()

def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.post_shutdown = on_shutdown
    log.info("Bot up. Buttons enabled. Commands: /start, /help")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
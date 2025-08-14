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
from persona import WhizperPersonality
from telegram.ext import MessageHandler, filters
import re

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
PERSONA = WhizperPersonality(api_base=None, anthropic_key=None)
ALLOWED_INTERVALS: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
MAX_TELEGRAM_MSG = 4096
WAITING_FOR_CA = {}

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if WAITING_FOR_CA.get(user_id):
        await update.message.reply_text(
            "What are you waiting for? 🙄 Send me the CA and I'll see what I can do..."
        )
        WAITING_FOR_CA[user_id] = False  # reset after prompt
    else:
        # normal fallback if not expecting CA
        await update.message.reply_text("Use the menu buttons to get started.")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "contract_mode":
        WAITING_FOR_CA[query.from_user.id] = True
        await query.message.reply_text(
            "Contract analysis mode activated. Paste the CA below."
        )
    else:
        # handle button callbacks here
        pass

def _extract_contract(s: str) -> str:
    s = s.strip()
    # Pull 0x... if it's embedded in a URL or sentence
    m = re.search(r'0x[a-fA-F0-9]{40}', s)
    if m:
        return m.group(0)
    # Try a loose Solana Base58 (32–44 chars, no 0/O/I/l)
    m = re.search(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b', s)
    if m:
        return m.group(0)
    return s

def _friendly_contract_error(err: Exception) -> str:
    msg = str(err)
    if "No pools found" in msg:
        return "I couldn’t find a liquid pool for that contract. Make sure it’s the **token** address (not LP), on a supported chain (ETH, BSC, Base, Solana)."
    if "suitable pool" in msg or "missing network/address" in msg:
        return "That pool looks weird or illiquid. Try the token’s main CA, or another chain/pair."
    if "ohlcv" in msg or "search pools" in msg:
        return "Upstream API coughed. Try again in a bit."
    return "Couldn’t fetch contract data. Double‑check the CA and try again."

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("chart-whisperer-tg")

# ---- GeckoTerminal (free, no key) ----
GT_API = "https://api.geckoterminal.com/api/v2"

def _looks_like_evm(addr: str) -> bool:
    s = (addr or "").strip()
    return s.startswith("0x") and len(s) == 42

def _looks_like_solana(addr: str) -> bool:
    s = (addr or "").strip()
    # loose check; Solana base58 is usually 32–44 chars, commonly 43–44
    return 32 <= len(s) <= 44 and s.isalnum() and not s.startswith("0x")

def _interval_to_gt(interval: str) -> tuple[str, int]:
    # map our intervals -> (timeframe, aggregate)
    m = {
        "1m": ("minute", 1),
        "5m": ("minute", 5),
        "15m": ("minute", 15),
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day", 1),
    }
    return m.get(interval, ("hour", 1))

async def _gt_find_top_pool(sess: aiohttp.ClientSession, addr: str) -> tuple[str, str, str, str]:
    """
    Return (network_id, pool_address, base_symbol, quote_symbol) for the most liquid pool
    that contains the token 'addr'. We use the search endpoint so user doesn't need to pick a network.
    """
    url = f"{GT_API}/search/pools?query={addr}"
    async with sess.get(url, headers={"accept": "application/json"}) as r:
        j = await r.json()
        if r.status != 200:
            raise RuntimeError(f"search pools {r.status}: {str(j)[:200]}")

    pools = j.get("data", []) or []
    if not pools:
        raise RuntimeError("No pools found for that address.")

    # Try to pick by highest liquidity (reserve_in_usd)
    best = None
    best_liq = -1.0
    for item in pools:
        attrs = item.get("attributes", {})
        liq = float(attrs.get("reserve_in_usd") or 0.0)
        if liq > best_liq:
            best = item
            best_liq = liq

    if not best:
        raise RuntimeError("No suitable pool found.")

    attrs = best.get("attributes", {})
    relationships = best.get("relationships", {}) or {}
    base = (relationships.get("base_token", {}).get("data", {}) or {}).get("id", "")
    quote = (relationships.get("quote_token", {}).get("data", {}) or {}).get("id", "")
    base_symbol = (attrs.get("base_token_symbol") or "TOKEN").upper()
    quote_symbol = (attrs.get("quote_token_symbol") or "USD").upper()

    network_id = attrs.get("network_id") or ""
    pool_address = attrs.get("address") or ""
    if not network_id or not pool_address:
        raise RuntimeError("Pool record missing network/address.")

    return network_id, pool_address, base_symbol, quote_symbol

async def _gt_fetch_ohlcv_df(
    sess: aiohttp.ClientSession, network: str, pool: str, interval: str, limit: int = 500
) -> pd.DataFrame:
    timeframe, aggregate = _interval_to_gt(interval)
    url = f"{GT_API}/networks/{network}/pools/{pool}/ohlcv/{timeframe}"
    params = {"aggregate": aggregate, "limit": min(limit, 500), "currency": "usd"}
    async with sess.get(url, params=params, headers={"accept": "application/json"}) as r:
        j = await r.json()
        if r.status != 200:
            raise RuntimeError(f"ohlcv {r.status}: {str(j)[:200]}")

    data = j.get("data", []) or []
    if not data:
        raise RuntimeError("No OHLCV returned.")

    # GeckoTerminal returns seconds timestamps; build the same schema as Binance klines
    rows = []
    for cndl in data:
        ts = int(cndl.get("timestamp"))
        open_ = float(cndl.get("open"))
        high = float(cndl.get("high"))
        low = float(cndl.get("low"))
        close = float(cndl.get("close"))
        vol = float(cndl.get("volume") or 0.0)
        open_time = pd.to_datetime(ts, unit="s", utc=True)
        rows.append([open_time, open_, high, low, close, vol, open_time])

    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","close_time"])
    return df

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
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
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

    notes: List[str] = []
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

    if atr_val is not None:
        notes.append(f"ATR14 ≈ {atr_val:.2f}; expect ±{atr_val:.2f} range")

    # --- VWAP ---
    vwap_val = float(last["vwap"]) if not math.isnan(last["vwap"]) else None
    vwap_delta = None
    if vwap_val is not None:
        vwap_delta = (price - vwap_val) / vwap_val * 100.0
        if abs(vwap_delta) < 0.15:
            notes.append(f"Price at VWAP ({vwap_delta:+.2f}%)")
        elif vwap_delta > 0:
            notes.append(f"Price above VWAP ({vwap_delta:+.2f}%)")
        else:
            notes.append(f"Price below VWAP ({vwap_delta:+.2f}%)")

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
        "vwap": vwap_val,
        "vwap_delta": vwap_delta,
    }

def risk_badge(atr: float | None, price: float | None) -> str:
    if not atr or not price or price <= 0:
        return "N/A"
    pct = atr / price * 100.0
    if pct < 0.5:
        label = "🔹 Low"
    elif pct < 1.0:
        label = "🔷 Med"
    else:
        label = "🔷🔷 High"
    return f"{label} ({pct:.2f}%)"

def build_summary(symbol: str, interval: str, r: dict) -> str:
    if r["ema20"] > r["ema50"]:
        ema_rel = "EMA20>EMA50"
    elif r["ema20"] < r["ema50"]:
        ema_rel = "EMA20<EMA50"
    else:
        ema_rel = "EMA20=EMA50"

    macd_sig = "MACD+" if r["macd_hist"] > 0 else ("MACD-" if r["macd_hist"] < 0 else "MACD~")
    risk = risk_badge(r.get("atr14"), r.get("price"))

    vwap_str = ""
    if r.get("vwap"):
        vdelta = (r["price"] - r["vwap"]) / r["vwap"] * 100.0
        vwap_str = f" | VWAP {vdelta:+.2f}%"

    line1 = (
        f"{symbol} {interval} — {r['trend'].title()} | RSI {r['rsi_state'].title()} | "
        f"{ema_rel} | {macd_sig}{vwap_str} | Risk {risk} | ${r['price']:.2f}"
    )

    fibs = r.get("fibs") or {}
    line2 = f"Fibs: .382 {fibs['0.382']:.2f} / .618 {fibs['0.618']:.2f}" if "0.382" in fibs and "0.618" in fibs else ""
    return line1 if not line2 else f"{line1}\n{line2}"
    
def build_report(symbol: str, interval: str, result: dict) -> str:
    lines = []
    lines.append(f"Whizper_BOT — {symbol} on {interval}")
    lines.append("─" * 48)
    lines.append(f"Time: {result['time']}")
    lines.append(f"Price: {result['price']:.2f}")
    lines.append("")
    lines.append(f"Trend: {result['trend'].upper()}  |  RSI14: {result['rsi_state'].upper()}")
    if result.get("vwap") is not None:
        if result.get("vwap_delta") is not None:
            lines.append(f"VWAP: {result['vwap']:.2f}  Δ: {result['vwap_delta']:+.2f}%")
        else:
            lines.append(f"VWAP: {result['vwap']:.2f}")
    lines.append(f"EMA20: {result['ema20']:.2f}   EMA50: {result['ema50']:.2f}")
    lines.append(f"MACD: {result['macd']:.5f}  Signal: {result['macd_signal']:.5f}  Hist: {result['macd_hist']:.5f}")
    if result.get('atr14') is not None:
        lines.append(f"ATR14: {result['atr14']:.2f} | Risk: {risk_badge(result['atr14'], result['price'])}")
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
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("₿ BTC", callback_data="sym|BTCUSDT"),
            InlineKeyboardButton("Ξ ETH", callback_data="sym|ETHUSDT"),
            InlineKeyboardButton("◎ SOL", callback_data="sym|SOLUSDT"),
        ],
        [
            InlineKeyboardButton("📜 Analyze Contract", callback_data="nav|contract")
        ]
    ])

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
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("awaiting_contract"):
        address = update.message.text.strip()
        context.user_data.pop("awaiting_contract", None)
        # Fetch contract report
        try:
            sess = await _ensure_session(context)
            url = f"http://127.0.0.1:8000/contract_report?address={address}&interval=1h"
            async with sess.get(url) as r:
                text = await r.text()
            await update.message.reply_text(f"📈 Whispering contract {address} on 1h:\n\n{text}")
        except Exception as e:
            await update.message.reply_text(f"Couldn't fetch contract report: {e}")

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
    
    if data == "nav|contract":
        context.user_data.clear()
        context.user_data["awaiting_contract"] = True
        await q.edit_message_text("What are you waiting for? 🙄 Send me the CA and I'll see what I can do...")
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
            text = PERSONA.decorate_report(text, symbol, interval)
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
            text = PERSONA.decorate_report(text, symbol, interval)
        except Exception as e:
            await q.edit_message_text(f"Couldn’t fetch report: {e}", reply_markup=_action_keyboard())
            return
        header = f"📈 Whispering {symbol} on {interval}:\n\n"
        chunks = _chunk(header + text)
        await q.edit_message_text(chunks[0], reply_markup=_action_keyboard())
        for more in chunks[1:]:
            await q.message.reply_text(more)
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    # Only react when user explicitly tapped "Analyze Contract"
    if not context.user_data.get("awaiting_contract"):
        # Optional: if user pasted an obvious CA without pressing the button, we could ignore.
        return

    context.user_data.pop("awaiting_contract", None)
    address = _extract_contract(text)
    interval = "1h"  # keep it simple for now
    sess = await _ensure_session(context)

    try:
        net, pool, base_sym, quote_sym = await _gt_find_top_pool(sess, address)
        df = await _gt_fetch_ohlcv_df(sess, net, pool, interval, limit=200)
        df = add_indicators(df)
        r = generate_analysis(df)

        pair = f"{base_sym}{'/' + quote_sym if quote_sym else ''}"
        # If you want just name/price/mcap later, we can swap to a slim summary here.
        body = build_summary(pair, interval, r)
        body = PERSONA.decorate_report(body, pair.upper(), interval)

        header = f"📈 Whispering {pair.upper()} on {interval}:\n\n"
        for chunk in _chunk(header + body):
            await update.message.reply_text(chunk, reply_markup=_action_keyboard())

    except Exception as e:
        await update.message.reply_text(_friendly_contract_error(e))

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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.post_shutdown = on_shutdown
    log.info("Bot up. Buttons enabled. Commands: /start, /help")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()

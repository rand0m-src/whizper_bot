# tg_bot.py
import os
import logging
import asyncio
import math
from typing import List, Tuple
import urllib.parse
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
    ("BTC", "BTCUSDT"),
    ("ETH", "ETHUSDT"),
    ("SOL", "SOLUSDT"),
]
PERSONA = WhizperPersonality(api_base=None, anthropic_key=None)

# Trimmed intervals + Summary
ALLOWED_INTERVALS: List[str] = ["1h", "4h", "1d", "summary"]
MAX_TELEGRAM_MSG = 4096

# New: compact mode output (snappy, alpha-leaning)
COMPACT_MODE = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("chart-whisperer-tg")

# ---- GeckoTerminal (free, no key) ----
GT_API = "https://api.geckoterminal.com/api/v2"

def _looks_like_evm(addr: str) -> bool:
    s = (addr or "").strip()
    return s.startswith("0x") and len(s) == 42

def _looks_like_solana(addr: str) -> bool:
    s = (addr or "").strip()
    return 32 <= len(s) <= 44 and s.isalnum() and not s.startswith("0x")

def _interval_to_gt(interval: str) -> tuple[str, int]:
    m = {
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day", 1),
    }
    return m.get(interval, ("hour", 1))

async def _gt_find_top_pool(sess: aiohttp.ClientSession, addr: str) -> tuple[str, str, str, str]:
    url = f"{GT_API}/search/pools?query={addr}"
    async with sess.get(url, headers={"accept": "application/json"}) as r:
        j = await r.json()
        if r.status != 200:
            raise RuntimeError(f"search pools {r.status}: {str(j)[:200]}")

    pools = j.get("data", []) or []
    if not pools:
        raise RuntimeError("No pools found for that address.")

    best = None
    best_liq = -1.0
    for item in pools:
        attrs = item.get("attributes", {}) or {}
        liq = float(attrs.get("reserve_in_usd") or 0.0)
        if liq > best_liq:
            best = item
            best_liq = liq

    if not best:
        raise RuntimeError("No suitable pool found.")

    attrs = best.get("attributes", {}) or {}
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

# -------------------- data + indicators --------------------
async def fetch_klines_async(sess: aiohttp.ClientSession, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err = None
    for base in BINANCE_BASES:
        try:
            async with sess.get(base, params=params) as r:
                data = await r.json()
                if r.status != 200:
                    if r.status == 451:
                        last_err = f"{base} -> 451 (geo-block)"
                        continue
                    raise RuntimeError(f"{base} {r.status}: {str(data)[:200]}")
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

    vwap_val = float(last["vwap"]) if not math.isnan(last["vwap"]) else None
    vwap_delta = None
    if vwap_val is not None:
        vwap_delta = (price - vwap_val) / vwap_val * 100.0

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

# ---------- Compact/alpha output helpers ----------
def _recent_move(df: pd.DataFrame, interval: str) -> tuple[float, float, float]:
    """
    Returns (start, end, pct_change) over a sensible multi-day window per interval.
    1h -> last 72h; 4h -> last 14 candles (~2–3 days); 1d -> last 7 days.
    """
    if df.empty:
        return (0.0, 0.0, 0.0)
    if interval == "1h":
        window = min(len(df), 72)
    elif interval == "4h":
        window = min(len(df), 14)
    else:
        window = min(len(df), 7)
    w = df.tail(window)
    start = float(w.iloc[0]["close"])
    end = float(w.iloc[-1]["close"])
    pct = ((end - start) / start * 100.0) if start else 0.0
    return (start, end, pct)

def _alpha_hint(r: dict) -> str:
    tr = r["trend"]
    rsi_s = r["rsi_state"]
    vwap_delta = r.get("vwap_delta")
    notes = []
    if tr == "bullish":
        notes.append("Buy dips toward EMA20/50; look for MACD hold.")
        if vwap_delta is not None and vwap_delta < 0:
            notes.append("Below VWAP — better RR on reclaim.")
    elif tr == "bearish":
        notes.append("Fade rallies into EMAs; wait for MACD roll-up to flip.")
        if rsi_s == "oversold":
            notes.append("Oversold — bounce risk; size down.")
    else:
        notes.append("Range tactics; wait for EMA cross or VWAP reclaim.")
        if rsi_s == "overbought":
            notes.append("Overbought in range — mean reversion likely.")
    return " ".join(notes[:1])

def build_concise_report(symbol: str, interval: str, df: pd.DataFrame, r: dict) -> str:
    trend = r["trend"].upper()
    risk = risk_badge(r.get("atr14"), r.get("price"))
    start, end, pct = _recent_move(df, interval)
    header = f"{symbol} • {interval} • {trend} • Risk {risk}"
    move = f"Move (recent): {pct:+.2f}%  ({start:,.2f} → {end:,.2f})"
    alpha = f"Alpha: {_alpha_hint(r)}"
    ctx_bits = []
    if r.get("vwap_delta") is not None:
        ctx_bits.append(f"VWAP {r['vwap_delta']:+.2f}%")
    ctx_bits.append(f"Price ${r['price']:.2f}")
    ctx = " | ".join(ctx_bits)
    return "\n".join([header, move, alpha, ctx])

# ---------- Legacy verbose builders (kept for fallback/toggling) ----------
def build_summary(pair_label: str, interval: str, r: dict) -> str:
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
        f"{pair_label} {interval} — {r['trend'].title()} | RSI {r['rsi_state'].title()} | "
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

# --- Symbol 24h summary (for the Summary button) ---
async def _fetch_symbol_24h_summary(context: ContextTypes.DEFAULT_TYPE, symbol: str) -> str:
    sess = await _ensure_session(context)
    df = await fetch_klines_async(sess, symbol=symbol, interval="1h", limit=30)
    if len(df) < 2:
        return "24h summary: not enough data."

    window = df.tail(24) if len(df) >= 24 else df
    start = float(window.iloc[0]["close"])
    end = float(window.iloc[-1]["close"])
    change = ((end - start) / start) * 100.0 if start else 0.0

    if change > 2:
        tone = "bullish momentum; buy dips only"
    elif change < -2:
        tone = "heavy sellers; fade bounces"
    else:
        tone = "range-bound; wait for break or VWAP reclaim"

    return f"24h: {change:+.2f}%  ({start:,.2f} → {end:,.2f}) — {tone}."

# Build report end-to-end with retry
async def _fetch_text_report(context: ContextTypes.DEFAULT_TYPE, symbol: str, interval: str) -> str:
    sess = await _ensure_session(context)
    last_err = None
    for attempt in range(3):
        try:
            df = await fetch_klines_async(sess, symbol=symbol, interval=interval, limit=500)
            df = add_indicators(df)
            result = generate_analysis(df)
            if COMPACT_MODE:
                return build_concise_report(symbol, interval, df, result)
            else:
                return build_report(symbol, interval, result)
        except (ClientConnectorError, asyncio.TimeoutError) as e:
            last_err = e
            await asyncio.sleep(0.7 * (attempt + 1))
        except Exception as e:
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

def interval_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("1h", callback_data="int|1h"),
        InlineKeyboardButton("4h", callback_data="int|4h"),
        InlineKeyboardButton("1d", callback_data="int|1d"),
        InlineKeyboardButton("Summary", callback_data="int|summary"),
    ]])

def _action_keyboard() -> InlineKeyboardMarkup:
    # Labels nudged toward the snappier vibe; functionality unchanged
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔁 Re-run", callback_data="act|refresh")],
        [
            InlineKeyboardButton("🪙 Change Coin", callback_data="nav|symbol"),
            InlineKeyboardButton("⏱️ Timeframe", callback_data="nav|interval")
        ]
    ])

def _contract_action_keyboard(dex_url: str, gt_url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔗 Dexscreener", url=dex_url),
            InlineKeyboardButton("🦎 GeckoTerminal", url=gt_url),
        ],
        [
            InlineKeyboardButton("🪙 Change Coin", callback_data="nav|symbol"),
            InlineKeyboardButton("⏱️ Timeframe", callback_data="nav|interval"),
        ]
    ])

WELCOME = (
    "🐸 Whizper Bot — I read charts so you don’t have to.\n\n"
    "Tap /start, pick a coin and timeframe (1h/4h/1d), or hit Summary.\n"
    "I’ll give you a tight take: **Trend**, **Risk**, **Recent Move**, and an **Alpha hint**.\n\n"
    "⏳ Supported intervals: 1h, 4h, 1d, Summary\n"
    "💡 I’m sarcastic, a little rude, occasionally useful."
)

# -------------------- commands --------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    await update.message.reply_text("Wake up call received. Pick a coin:", reply_markup=_symbol_keyboard())

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)

# -------------------- callback + text flow --------------------
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
        await q.edit_message_text("Pick a timeframe:", reply_markup=interval_keyboard())
        return

    if data.startswith("sym|"):
        symbol = data.split("|", 1)[1]
        context.user_data["symbol"] = symbol
        await q.edit_message_text(f"Coin locked: {symbol}\nNow pick a timeframe:", reply_markup=interval_keyboard())
        return

    if data.startswith("int|"):
        interval = data.split("|", 1)[1]
        if interval not in ALLOWED_INTERVALS:
            await q.edit_message_text("Bad interval. Try again:", reply_markup=interval_keyboard())
            return
        context.user_data["interval"] = interval
        symbol = context.user_data.get("symbol", SYMBOLS[0][1])
        try:
            if interval == "summary":
                text = await _fetch_symbol_24h_summary(context, symbol)
                text = PERSONA.decorate_report(text, symbol, interval)
                header = f"📈 Whispering {symbol} — 24h Summary:\n\n"
                chunks = _chunk(header + text)
                await q.edit_message_text(chunks[0], reply_markup=_action_keyboard())
                for more in chunks[1:]:
                    await q.message.reply_text(more)
                return
            else:
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
            if interval == "summary":
                text = await _fetch_symbol_24h_summary(context, symbol)
                text = PERSONA.decorate_report(text, symbol, interval)
                header = f"📈 Whispering {symbol} — 24h Summary:\n\n"
            else:
                text = await _fetch_text_report(context, symbol, interval)
                text = PERSONA.decorate_report(text, symbol, interval)
                header = f"📈 Whispering {symbol} on {interval}:\n\n"
        except Exception as e:
            await q.edit_message_text(f"Couldn’t fetch report: {e}", reply_markup=_action_keyboard())
            return
        chunks = _chunk(header + text)
        await q.edit_message_text(chunks[0], reply_markup=_action_keyboard())
        for more in chunks[1:]:
            await q.message.reply_text(more)
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    # Only handle CAs when we're in contract mode
    if not context.user_data.get("awaiting_contract"):
        return

    address = _extract_contract(text)
    interval = "1h"
    sess = await _ensure_session(context)

    try:
        # Try to build contract report
        net, pool, base_sym, quote_sym = await _gt_find_top_pool(sess, address)
        df = await _gt_fetch_ohlcv_df(sess, net, pool, interval, limit=200)
        df = add_indicators(df)
        r = generate_analysis(df)
        pair = f"{base_sym}{'/' + quote_sym if quote_sym else ''}"
        body = build_summary(pair, interval, r)
        body = PERSONA.decorate_report(body, pair.upper(), interval)

        # ✅ success: clear the mode flag
        context.user_data.pop("awaiting_contract", None)

        # 🔗 add links
        dex_url = dexscreener_url(address, network_id=net)
        gt_url = geckoterminal_pool_url(net, pool)
        kb = _contract_action_keyboard(dex_url, gt_url)

        header = f"📈 Whispering {pair.upper()} on {interval}:\n\n"
        for i, chunk in enumerate(_chunk(header + body)):
            if i == 0:
                await update.message.reply_text(chunk, reply_markup=kb)
            else:
                await update.message.reply_text(chunk)

    except Exception as e:
        # ❌ Error: keep the mode ON and re-prompt
        err = _friendly_contract_error(e)
        context.user_data["awaiting_contract"] = True
        await update.message.reply_text(
            f"{err}\n\nPaste another contract address (ETH/BSC/Base/SOL supported), "
            f"or /start to exit contract mode."
        )

# -------------------- helpers for contract errors + CA extraction --------------------
def _extract_contract(s: str) -> str:
    s = s.strip()
    m = re.search(r'0x[a-fA-F0-9]{40}', s)
    if m:
        return m.group(0)
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
    return "Couldn’t fetch contract data. Double-check the CA and try again."

# -------------------- lifecycle --------------------
async def on_shutdown(app: Application) -> None:
    sess: aiohttp.ClientSession | None = app.bot_data.get("session")
    if sess and not sess.closed:
        await sess.close()

# --- External links helpers ---
def _dexscreener_chain_slug(network_id: str | None) -> str | None:
    if not network_id:
        return None
    nid = network_id.lower()
    mapping = {
        # EVM majors
        "eth": "ethereum", "ethereum": "ethereum",
        "bsc": "bsc", "bnb": "bsc",
        "base": "base",
        "arb": "arbitrum", "arbitrum": "arbitrum",
        "op": "optimism", "optimism": "optimism",
        "polygon": "polygon", "matic": "polygon",
        "avax": "avalanche", "avalanche": "avalanche",
        "ftm": "fantom", "fantom": "fantom",
        # Solana (+ sonic if you use it)
        "sol": "solana", "solana": "solana",
        "s": "sonic", "sonic": "sonic",
    }
    return mapping.get(nid)

def dexscreener_url(address: str, network_id: str | None = None) -> str:
    chain = _dexscreener_chain_slug(network_id)
    if chain:
        return f"https://dexscreener.com/{chain}/{address}"
    # robust fallback: URL-encode the query
    q = urllib.parse.quote_plus(address)
    return f"https://dexscreener.com/search?q={q}"

def geckoterminal_pool_url(network_id: str, pool_address: str) -> str:
    return f"https://www.geckoterminal.com/{network_id}/pools/{pool_address}"

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

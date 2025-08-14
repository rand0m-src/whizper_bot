#main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
import uvicorn
import requests
import pandas as pd
import numpy as np
import math

app = FastAPI(title="Whizper_BOT — The Chart Whisperer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# BINANCE (free, no key)
# BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_BASES = [
    "https://api.binance.com/api/v3/klines",
    "https://api.binance.us/api/v3/klines",
]

# GeckoTerminal (free, no key)
GT_API = "https://api.geckoterminal.com/api/v2"

def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    last_err = None
    for url in BINANCE_BASES:
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 451:
                last_err = "451 geo-block"
                continue
            r.raise_for_status()
            data = r.json()
            cols = ["open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_buy_base","taker_buy_quote","ignore"]
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

def _gt_interval_map(interval: str) -> tuple[str, int]:
    # map our intervals to GeckoTerminal timeframe/aggregate
    m = {
        "1m": ("minute", 1),
        "5m": ("minute", 5),
        "15m": ("minute", 15),
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day", 1),
    }
    return m.get(interval, ("hour", 1))

def _gt_find_top_pool(address: str) -> tuple[str, str, str, str]:
    """
    Returns (network_id, pool_address, base_symbol, quote_symbol) for the most liquid pool
    that contains the token at 'address'. We search across networks so user needn't specify chain.
    """
    url = f"{GT_API}/search/pools"
    r = requests.get(url, params={"query": address}, headers={"accept": "application/json"}, timeout=20)
    r.raise_for_status()
    j = r.json()
    pools = j.get("data") or []
    if not pools:
        raise RuntimeError("No pools found for that address.")

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
    rel = best.get("relationships", {}) or {}
    base_sym = (attrs.get("base_token_symbol") or "TOKEN").upper()
    quote_sym = (attrs.get("quote_token_symbol") or "USD").upper()
    network_id = attrs.get("network_id") or ""
    pool_addr = attrs.get("address") or ""
    if not network_id or not pool_addr:
        raise RuntimeError("Pool record missing network/address.")
    return network_id, pool_addr, base_sym, quote_sym

def _gt_fetch_ohlcv_df(network: str, pool: str, interval: str, limit: int = 500) -> pd.DataFrame:
    timeframe, aggregate = _gt_interval_map(interval)
    url = f"{GT_API}/networks/{network}/pools/{pool}/ohlcv/{timeframe}"
    params = {"aggregate": aggregate, "limit": min(limit, 500), "currency": "usd"}
    r = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=20)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") or []
    if not data:
        raise RuntimeError("No OHLCV returned.")
    # handle both flat and attributes-wrapped shapes
    rows = []
    for item in data:
        attrs = item.get("attributes", item)
        ts = int(attrs.get("timestamp"))
        o = float(attrs.get("open"))
        h = float(attrs.get("high"))
        l = float(attrs.get("low"))
        c = float(attrs.get("close"))
        v = float(attrs.get("volume") or 0.0)
        t = pd.to_datetime(ts, unit="s", utc=True)
        rows.append([t, o, h, l, c, v, t])
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","close_time"])
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")

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

    # --- VWAP (cumulative over the fetched window) ---
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

def build_report(symbol: str, interval: str, result: dict) -> str:
    lines = []
    lines.append(f"Whizper_BOT — {symbol} on {interval}")
    lines.append("─" * 48)
    lines.append(f"Time: {result['time']}")
    lines.append(f"Price: {result['price']:.2f}")
    lines.append("")
    lines.append(f"Trend: {result['trend'].upper()}  |  RSI14: {result['rsi_state'].upper()}")

    # --- VWAP display ---
    if result.get("vwap") is not None:
        if result.get("vwap_delta") is not None:
            lines.append(f"VWAP: {result['vwap']:.2f}  Δ: {result['vwap_delta']:+.2f}%")
        else:
            lines.append(f"VWAP: {result['vwap']:.2f}")

    lines.append(f"EMA20: {result['ema20']:.2f}   EMA50: {result['ema50']:.2f}")
    lines.append(f"MACD: {result['macd']:.5f}  Signal: {result['macd_signal']:.5f}  Hist: {result['macd_hist']:.5f}")

    # --- Risk badge on ATR line ---
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

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/analyze")
def analyze(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500):
    df = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    df = add_indicators(df)
    result = generate_analysis(df)
    return {"symbol": symbol, "interval": interval, "result": result}

@app.get("/contract_analyze")
def contract_analyze(address: str, interval: str = "1h", limit: int = 500):
    net, pool, base_sym, quote_sym = _gt_find_top_pool(address)
    df = _gt_fetch_ohlcv_df(net, pool, interval, limit)
    df = add_indicators(df)
    result = generate_analysis(df)
    symbol = f"{base_sym}/{quote_sym}" if quote_sym else base_sym
    return {"symbol": symbol, "interval": interval, "result": result, "network": net, "pool": pool}

@app.get("/contract_report", response_class=PlainTextResponse)
def contract_report(address: str, interval: str = "1h", limit: int = 500):
    net, pool, base_sym, quote_sym = _gt_find_top_pool(address)
    df = _gt_fetch_ohlcv_df(net, pool, interval, limit)
    df = add_indicators(df)
    result = generate_analysis(df)
    symbol = f"{base_sym}/{quote_sym}" if quote_sym else base_sym
    text = build_report(symbol, interval, result)
    return text

@app.get("/report", response_class=PlainTextResponse)
def report(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500):
    df = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    df = add_indicators(df)
    result = generate_analysis(df)
    text = build_report(symbol, interval, result)
    return text

@app.get("/summary")
def summary(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500):
    df = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    df = add_indicators(df)
    result = generate_analysis(df)
    notes_str = " | ".join(result["notes"])
    summary = (
        f"[{result['time']}] {symbol} ({interval}) — Price: ${result['price']:.2f}, "
        f"Trend: {result['trend'].upper()}, RSI: {result['rsi_state']}, "
        f"EMA20: {result['ema20']:.2f}, EMA50: {result['ema50']:.2f}. "
        f"Notes: {notes_str}"
    )
    return {"summary": summary, "raw": result}

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!doctype html>
    <html>
    <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>Whizper_BOT — The Chart Whisperer</title>
    </head>
    <body style="font-family: ui-sans-serif, system-ui; padding: 24px; max-width: 800px; margin: auto;">
        <h1>Whizper_BOT — The Chart Whisperer</h1>
        <p>Try the JSON endpoint: <code>/analyze?symbol=BTCUSDT&interval=1h</code></p>
        <p>Or the human-readable report: <code>/report?symbol=BTCUSDT&interval=1h</code></p>
        <button id="run">Run Report</button>
        <pre id="out" style="white-space: pre-wrap; background:#111; color:#0f0; padding:16px; border-radius:8px; min-height:200px;"></pre>
        <script>
        document.getElementById('run').onclick = async () => {
            const r = await fetch('/report?symbol=BTCUSDT&interval=1h');
            const t = await r.text();
            document.getElementById('out').textContent = t;
        }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

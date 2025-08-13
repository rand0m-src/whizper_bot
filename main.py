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

BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    cols = [
        "open_time","open","high","low","close","volume","close_time","qav","num_trades","taker_buy_base","taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["open_time","open","high","low","close","volume","close_time"]]
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

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/analyze")
def analyze(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500):
    df = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    df = add_indicators(df)
    result = generate_analysis(df)
    return {"symbol": symbol, "interval": interval, "result": result}

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
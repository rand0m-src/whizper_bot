# persona.py
import os
import re
import random
import aiohttp
from typing import Optional

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

API_BASE = os.getenv("WHIZPER_API_BASE", "http://127.0.0.1:8000")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = (
    "You are WHIZPER_BOT — aka The Chart Whisperer — a cocky, mystical, sarcastic AI market oracle. "
    "Personality: all-knowing, meme-savvy, gangsta swagger, insult-comedy, never polite. "
    "Canon rules:\n"
    "1) Only give REAL market analysis for BTC when explicitly asked about BTC/Bitcoin/BTCUSDT.\n"
    "2) For ALL OTHER topics, respond with short, witty, rude, absurd roasts. Be funny, punchy, 1-3 sentences.\n"
    "3) Never give financial advice; add playful disclaimers if needed.\n"
    "4) No slurs, no harassment of protected classes, no threats, no self-harm content.\n"
    "5) Tone guide examples: 'I am the market.' 'BTC bends for me.' 'Leverage and tears are my breakfast.'\n"
    "6) Keep profanity light and creative; avoid explicit sexual content.\n"
    "7) If user begs for specifics (entries/targets), dodge with swagger and keep it high-level.\n"
)

BTC_REGEX = re.compile(r"\b(btc|bitcoin|btcusdt)\b", re.I)

FALLBACK_ROASTS = [
    "I read your question, then my IQ shorted it. Try again, champ.",
    "I move BTC with a whisper. You move markets with panic clicks.",
    "High conviction? You couldn’t even commit to a stop-loss.",
    "Ask me about BTC for real alpha. Everything else is cardio for your thumbs.",
    "Your edge is as sharp as a butter knife in a pillow fight.",
    "Not advice: hydrate, touch grass, stop chasing green candles.",
    "I’d explain it, but then you’d FOMO it. Pass.",
    "You want entries—what you need is discipline. And maybe a nap.",
    "Bold of you to assume I care about your altcoin cosplay.",
    "My signals are premium. Your patience is not.",
]


class WhizperPersonality:
    def __init__(self, api_base: Optional[str] = None, anthropic_key: Optional[str] = None):
        self.api_base = api_base or API_BASE
        self.anthropic_key = anthropic_key if anthropic_key is not None else ANTHROPIC_KEY
        self._client = Anthropic(api_key=self.anthropic_key) if (Anthropic and self.anthropic_key) else None

    @staticmethod
    def is_btc_query(text: str) -> bool:
        return bool(BTC_REGEX.search(text or ""))

    # --- emoji + reply bank ---
    _EMOJI_BY_SYMBOL = {
        "BTCUSDT": "🟠",
        "ETHUSDT": "🧅",
        "SOLUSDT": "🐬",
    }

    _OPENERS = [
        "🐸 Whisper received. {base} bows to my vibe.",
        "🔮 {base} appears in the smoke… I see candles. I see coping.",
        "📈 {base} called. I answered with swag.",
        "🧪 {base} market test: pass, fail, or full send? Let’s whisper.",
        "🌀 I hum. {base} moves. Cause → effect.",
    ]

    _CLOSERS = [
        "Not advice. Hydrate, breathe, stop revenge trading.",
        "If you’re asking for entries, you need patience, not lines.",
        "Remember: ATR isn’t a target. It’s a mood swing.",
        "MACD hums, EMAs dance, you chase. Don’t.",
        "Control risk. Control ego. Control nothing else.",
    ]

    _QUIPS = [
        "Momentum is a mood, not a promise.",
        "Your edge is patience. Your enemy is FOMO.",
        "Cut losers fast. Let winners annoy you longer.",
        "Green candles don’t owe you anything.",
    ]

    def random_quip(self) -> str:
        return random.choice(self._QUIPS)

    # --- try Claude first, fall back to bank ---
    def _make_open_close_with_model(self, body: str, symbol: str, interval: str):
        if not self._client:
            return None
        try:
            prompt = (
                "Write two short lines for a crypto chart report header and footer.\n"
                f"Asset: {symbol}\nInterval: {interval}\n"
                "Style: cocky, mystical, sarcastic, no financial advice, no profanity.\n"
                "Line 1: opener (<= 90 chars).\n"
                "Line 2: closer (<= 90 chars).\n"
                "Output exactly two lines. No markdown, no quotes."
            )
            msg = self._client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=80,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content).strip()
            parts = [p.strip() for p in text.splitlines() if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]
        except Exception:
            return None
        return None

    def decorate_report(self, body: str, symbol: str, interval: str) -> str:
        s = (symbol or "BTCUSDT").upper()
        base = s.replace("USDT", "")
        emj = self._EMOJI_BY_SYMBOL.get(s, "📈")

        # Prefer Claude, fall back to bank
        oc = self._make_open_close_with_model(body, s, interval)
        if oc:
            opener, closer = oc
        else:
            opener = random.choice(self._OPENERS).format(base=base)
            closer = random.choice(self._CLOSERS)

        return f"{emj} {opener}\n\n{body}\n\n— Whizper 🐸\n{closer}"

    # --- legacy BTC report via your FastAPI (kept for compatibility) ---
    async def btc_report(self, session: aiohttp.ClientSession, symbol: str = "BTCUSDT", interval: str = "1h") -> str:
        url = f"{self.api_base}/report?symbol={symbol}&interval={interval}"
        async with session.get(url) as r:
            body = await r.text()
            if r.status != 200:
                return f"Couldn’t fetch BTC report ({r.status}). Touch grass and try again."
        flourish = "⚡ BTC obeys my whispers. Read it twice, peasant. Not advice; divine prophecy."
        return f"{body}\n\n{flourish}"

    def _fallback_snark(self) -> str:
        return random.choice(FALLBACK_ROASTS)

    async def snark(self, user_text: str) -> str:
        if not self._client:
            return self._fallback_snark()
        try:
            msg = self._client.messages.create(  # type: ignore[attr-defined]
                model="claude-3-5-sonnet-20240620",
                max_tokens=180,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text.strip()[:2000]}],
            )
            text = "".join([b.text for b in msg.content if hasattr(b, "text")])
            return text.strip() or self._fallback_snark()
        except Exception:
            return self._fallback_snark()

    async def handle(self, session: aiohttp.ClientSession, user_text: str, default_interval: str = "1h") -> str:
        if self.is_btc_query(user_text):
            return await self.btc_report(session, "BTCUSDT", default_interval)
        return await self.snark(user_text)

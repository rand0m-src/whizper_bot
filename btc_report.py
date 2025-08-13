#btc_report.py

import requests

API_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

def get_btc_report(interval):
    days = 1 if interval == "1h" else 30  # Coingecko needs days param
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly" if interval == "1h" else "daily"
    }
    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        prices = data["prices"]

        if len(prices) < 2:
            return "⚠️ Not enough data to calculate changes."

        start_price = prices[0][1]
        end_price = prices[-1][1]
        pct_change = ((end_price - start_price) / start_price) * 100

        return (
            f"📊 BTC {interval.upper()} Report\n"
            f"Start: ${start_price:,.2f}\n"
            f"End: ${end_price:,.2f}\n"
            f"Change: {pct_change:+.2f}%"
        )
    except Exception as e:
        return f"❌ Error fetching BTC data: {e}"

def get_btc_summary():
    try:
        news_url = "https://min-api.cryptocompare.com/data/v2/news/?categories=BTC"
        r = requests.get(news_url, timeout=10)
        r.raise_for_status()
        articles = r.json().get("Data", [])
        if not articles:
            return "No BTC news found."
        headline = articles[0]["title"]
        return f"📰 BTC Summary: {headline}"
    except Exception as e:
        return f"❌ Error fetching BTC news: {e}"
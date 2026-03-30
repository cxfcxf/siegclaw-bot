import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

log = logging.getLogger("siegclaw.finance")

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_http = httpx.Client(timeout=10, headers={"User-Agent": "SiegClaw/1.0"})

# Common aliases -> CoinGecko IDs
CRYPTO_ALIASES = {
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "ethereum": "ethereum",
    "sol": "solana", "solana": "solana",
    "xrp": "ripple", "ripple": "ripple",
    "ada": "cardano", "cardano": "cardano",
    "doge": "dogecoin", "dogecoin": "dogecoin",
    "dot": "polkadot", "polkadot": "polkadot",
    "matic": "matic-network", "polygon": "matic-network",
    "avax": "avalanche-2", "avalanche": "avalanche-2",
    "link": "chainlink", "chainlink": "chainlink",
    "bnb": "binancecoin", "binance": "binancecoin",
    "ltc": "litecoin", "litecoin": "litecoin",
    "usdt": "tether", "tether": "tether",
    "usdc": "usd-coin",
}

# Common commodity/forex tickers for Yahoo Finance
COMMODITY_ALIASES = {
    "gold": "GC=F", "silver": "SI=F", "platinum": "PL=F",
    "oil": "CL=F", "crude": "CL=F", "brent": "BZ=F",
    "natural gas": "NG=F", "copper": "HG=F",
    "wheat": "ZW=F", "corn": "ZC=F", "soybean": "ZS=F",
}

CURRENCY_ALIASES = {
    "usd/jpy": "USDJPY=X", "eur/usd": "EURUSD=X", "gbp/usd": "GBPUSD=X",
    "usd/cny": "USDCNY=X", "usd/myr": "USDMYR=X", "usd/sgd": "USDSGD=X",
    "usd/thb": "USDTHB=X", "usd/idr": "USDIDR=X", "usd/php": "USDPHP=X",
}

_STOP_WORDS = frozenset(("I", "A", "THE", "AND", "OR", "IS", "IT", "AT", "TO", "IN", "ON"))


def _fetch_crypto_prices(ids: list[str]) -> dict | None:
    try:
        resp = _http.get(
            f"{COINGECKO_BASE}/simple/price",
            params={
                "ids": ",".join(ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            },
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.error("CoinGecko API failed: %s", e)
        return None


def _fetch_yahoo_quote(symbol: str) -> dict | None:
    try:
        resp = _http.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol,
            params={"interval": "1d", "range": "1d"},
        )
        resp.raise_for_status()
        data = resp.json()
        meta = data["chart"]["result"][0]["meta"]
        return {
            "symbol": symbol,
            "price": meta.get("regularMarketPrice"),
            "previous_close": meta.get("previousClose"),
            "currency": meta.get("currency", "USD"),
        }
    except Exception as e:
        log.error("Yahoo Finance API failed for %s: %s", symbol, e)
        return None


def _format_change(change_pct: float) -> str:
    direction = "+" if change_pct >= 0 else ""
    return f"({direction}{change_pct:.2f}%)"


def _detect_assets(query: str) -> tuple[list[str], list[tuple[str, str]]]:
    words = set(query.lower().split())
    # Also check multi-word aliases against the full lowercase query
    q = query.lower()
    crypto_ids = []
    yahoo_assets = []

    for alias, cg_id in CRYPTO_ALIASES.items():
        if alias in words and cg_id not in crypto_ids:
            crypto_ids.append(cg_id)

    for alias, symbol in {**COMMODITY_ALIASES, **CURRENCY_ALIASES}.items():
        # Multi-word aliases (e.g. "natural gas") use substring, single-word use token match
        if " " in alias:
            if alias in q:
                yahoo_assets.append((alias, symbol))
        elif alias in words:
            yahoo_assets.append((alias, symbol))

    for word in query.split():
        clean = word.strip("?!.,")
        if (
            clean.isupper()
            and 1 <= len(clean) <= 5
            and clean.isalpha()
            and clean.lower() not in CRYPTO_ALIASES
            and clean not in _STOP_WORDS
        ):
            yahoo_assets.append((clean, clean))

    return crypto_ids, yahoo_assets


def get_financial_data(query: str) -> str | None:
    crypto_ids, yahoo_assets = _detect_assets(query)

    if not crypto_ids and not yahoo_assets:
        return None

    parts = []

    if crypto_ids:
        prices = _fetch_crypto_prices(crypto_ids)
        if prices:
            for cg_id, data in prices.items():
                price = data.get("usd")
                change = data.get("usd_24h_change")
                mcap = data.get("usd_market_cap")
                line = f"**{cg_id.title()}**: ${price:,.2f}"
                if change is not None:
                    line += f" {_format_change(change)} 24h"
                if mcap:
                    line += f" | MCap: ${mcap:,.0f}"
                parts.append(line)

    # Fetch Yahoo quotes in parallel
    if yahoo_assets:
        symbols = [sym for _, sym in yahoo_assets]
        names = [name for name, _ in yahoo_assets]
        with ThreadPoolExecutor(max_workers=5) as pool:
            quotes = list(pool.map(_fetch_yahoo_quote, symbols))

        for name, symbol, quote in zip(names, symbols, quotes):
            if quote and quote["price"]:
                price = quote["price"]
                prev = quote.get("previous_close")
                currency = quote["currency"]
                line = f"**{name.title()}** ({symbol}): {price:,.2f} {currency}"
                if prev and prev > 0:
                    change_pct = ((price - prev) / prev) * 100
                    line += f" {_format_change(change_pct)}"
                parts.append(line)

    if not parts:
        return None

    log.info("Finance data: %d assets found", len(parts))
    return "\n".join(parts)

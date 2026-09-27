from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CRYPTO_LIST_PATH = ROOT_DIR / "crypto_list.txt"
CACHE_DIR = ROOT_DIR / "data" / "temp"

LOOKBACK_DAYS = 365
CACHE_TTL_SECONDS = 15 * 60
KLINE_INTERVAL = "1d"
KLINE_LIMIT = 365

# data-api.binance.vision works from geo-restricted hosts (e.g. GitHub Actions US).
# api.binance.com is the fallback when vision fails DNS/TLS on some local networks.
BINANCE_BASE_URLS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)

_QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "FDUSD")


def load_crypto_pairs(path: Path | None = None) -> list[str]:
    list_path = path or CRYPTO_LIST_PATH
    if not list_path.exists():
        raise FileNotFoundError(f"Crypto list not found: {list_path}")
    pairs: list[str] = []
    seen: set[str] = set()
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().upper()
        if not line or line.startswith("#") or line in seen:
            continue
        seen.add(line)
        pairs.append(line)
    if not pairs:
        raise ValueError(f"No trading pairs found in {list_path}")
    return pairs


def to_binance_symbol(symbol: str) -> str:
    text = symbol.strip().upper().replace("-", "/").replace("_", "/")
    if "/" in text:
        base, quote = text.split("/", 1)
        return f"{base}{quote}"
    for quote in _QUOTE_ASSETS:
        if text.endswith(quote) and len(text) > len(quote):
            return text
    return f"{text}USDT"


def to_display_pair(symbol: str) -> str:
    binance_symbol = to_binance_symbol(symbol)
    for quote in _QUOTE_ASSETS:
        if binance_symbol.endswith(quote) and len(binance_symbol) > len(quote):
            return f"{binance_symbol[:-len(quote)]}/{quote}"
    return binance_symbol


def base_asset(symbol: str) -> str:
    pair = to_display_pair(symbol)
    return pair.split("/", 1)[0]


def asset_name(symbol: str) -> str:
    """Display name for a symbol — base asset from crypto_list.txt pairs."""
    return base_asset(symbol)


def format_usd(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if number != number:
        return ""
    sign = "-" if number < 0 else ""
    abs_value = abs(number)
    if abs_value >= 1:
        body = f"{abs_value:,.2f}"
    elif abs_value >= 0.01:
        body = f"{abs_value:,.4f}"
    else:
        body = f"{abs_value:,.6f}"
    return f"{sign}${body}"


def format_compact_usd(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    abs_value = abs(number)
    if abs_value >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${number / 1_000:.2f}K"
    return format_usd(number)

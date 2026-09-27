from __future__ import annotations

import logging
from typing import Any

import requests

from config import BINANCE_BASE_URLS

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


class BinanceAPIError(RuntimeError):
    """Raised when every Binance endpoint fails or returns an error payload."""


def binance_get(path: str, params: dict[str, Any] | None = None) -> Any:
    last_error: Exception | None = None
    for index, base_url in enumerate(BINANCE_BASE_URLS):
        url = f"{base_url}{path}"
        try:
            response = requests.get(
                url,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=30,
            )
            payload = _parse_payload(response)
            response.raise_for_status()
            if index > 0:
                logger.info("Binance request succeeded via fallback %s", base_url)
            return payload
        except (requests.RequestException, ValueError, BinanceAPIError) as exc:
            last_error = exc
            remaining = len(BINANCE_BASE_URLS) - index - 1
            if remaining:
                logger.warning(
                    "Binance request failed (%s), trying next endpoint: %s",
                    url,
                    exc,
                )
            else:
                logger.warning("Binance request failed (%s): %s", url, exc)
    raise BinanceAPIError(f"All Binance endpoints failed for {path}: {last_error}")


def _parse_payload(response: requests.Response) -> Any:
    try:
        payload = response.json()
    except ValueError as exc:
        raise BinanceAPIError("Binance returned invalid JSON") from exc
    if isinstance(payload, dict) and "code" in payload and "msg" in payload:
        raise BinanceAPIError(f"Binance error {payload['code']}: {payload['msg']}")
    return payload

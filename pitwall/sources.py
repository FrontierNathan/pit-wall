"""Where the public data lives, and a tiny downloader.

TracingInsights publishes FastF1-derived timing and telemetry on GitHub, one
repository per season (https://github.com/TracingInsights). Files are laid out
as <year>/<event>/<session>/<file>.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/TracingInsights/{year}/main/{path}"


def race_url(year: int, event: str, session: str, filename: str) -> str:
    path = "/".join(urllib.parse.quote(part) for part in (event, session, filename))
    return RAW_BASE.format(year=year, path=path)


def telemetry_url(year: int, event: str, session: str, driver: str, lap: int) -> str:
    return race_url(year, event, session, f"{driver}/{lap}_tel.json")


def download(url: str, retries: int = 3, timeout: float = 30.0) -> bytes:
    """GET a URL, retrying a couple of times on network hiccups."""
    request = urllib.request.Request(url, headers={"User-Agent": "pit-wall/0.1"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise  # 404 etc.: retrying won't help
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")

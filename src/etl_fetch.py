"""Fetch daily BTC-USD OHLCV data and store it in raw storage (CSV).

Sources, tried in order:
  1. yfinance (Yahoo Finance)
  2. CoinGecko public API (close + volume; open/high/low approximated)
  3. Synthetic data -- ONLY if --allow-synthetic / ALLOW_SYNTHETIC=1 is set.
     Used for offline CI tests; never used silently.
"""

import argparse
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("etl_fetch")
RAW_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def fetch_yfinance(period: str = "5y") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download("BTC-USD", period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns=str.lower)
    return df[RAW_COLUMNS]


def fetch_coingecko(days: int = 365) -> pd.DataFrame:
    import requests

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    resp = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    prices = pd.DataFrame(payload["prices"], columns=["ts", "close"])
    vols = pd.DataFrame(payload["total_volumes"], columns=["ts", "volume"])
    df = prices.merge(vols, on="ts")
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return df[RAW_COLUMNS]


def synthetic_prices(days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Geometric random walk resembling BTC. For offline testing only."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(datetime.now(timezone.utc).date())
    dates = pd.date_range(end=end, periods=days, freq="D")
    rets = rng.normal(0.0008, 0.03, size=days)
    close = 30000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.01, size=days))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": rng.uniform(2e10, 6e10, size=days),
        }
    )


def fetch(allow_synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    for name, fn in (("yfinance", fetch_yfinance), ("coingecko", fetch_coingecko)):
        try:
            df = fn()
            log.info("Fetched %d rows from %s", len(df), name)
            return df, name
        except Exception as exc:  # noqa: BLE001 - try next source
            log.warning("Source %s failed: %s", name, exc)
    if allow_synthetic:
        log.warning("All live sources failed -- using SYNTHETIC data (offline mode)")
        return synthetic_prices(), "synthetic"
    raise RuntimeError("Could not fetch BTC data from any source")


def run(out_path: str, allow_synthetic: bool = False) -> str:
    df, source = fetch(allow_synthetic)
    df = df.dropna(subset=["close"])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    with open(out_path + ".source", "w") as fh:
        fh.write(source)
    log.info("Saved %d rows (%s) to %s", len(df), source, out_path)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.getenv("RAW_DATA_PATH", "data/raw/btc_usd.csv"))
    parser.add_argument("--allow-synthetic", action="store_true", default=os.getenv("ALLOW_SYNTHETIC", "0") == "1")
    args = parser.parse_args()
    run(args.out, args.allow_synthetic)


if __name__ == "__main__":
    main()

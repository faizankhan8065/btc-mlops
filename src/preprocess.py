"""Clean raw BTC data and compute technical indicators (MA7, MA30, volatility)."""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from src.features import clean_prices

log = logging.getLogger("preprocess")


def run(raw_path: str, out_path: str) -> str:
    df = clean_prices(pd.read_csv(raw_path))
    df["ma_7"] = df["close"].rolling(7).mean()
    df["ma_30"] = df["close"].rolling(30).mean()
    df["log_return"] = np.log(df["close"]).diff()
    df["volatility_7"] = df["log_return"].rolling(7).std()
    df["volatility_30"] = df["log_return"].rolling(30).std()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Processed %d rows -> %s", len(df), out_path)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default=os.getenv("RAW_DATA_PATH", "data/raw/btc_usd.csv"))
    p.add_argument("--out", default=os.getenv("PROCESSED_DATA_PATH", "data/processed/btc_features.csv"))
    a = p.parse_args()
    run(a.raw, a.out)


if __name__ == "__main__":
    main()

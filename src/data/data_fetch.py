# src/src_data/data_fetch.py
"""
Download historical OHLCV for TSLA, BND, SPY using yfinance
Saves CSVs to data/data-raw/
"""

import os
from datetime import datetime
import pandas as pd
import yfinance as yf

TICKERS = ["TSLA", "BND", "SPY"]
START = "2015-07-01"
END = "2025-07-31"   # inclusive desired end date

OUT_DIR = os.path.join("data", "data-raw")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_and_save(ticker: str, start: str, end: str, out_dir: str):
    print(f"Downloading {ticker} from {start} to {end} ...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=True, interval="1d", auto_adjust=False)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return

    if df is None or df.empty:
        print(f"No data for {ticker}.")
        return

    # Ensure Date is a column and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    csv_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(csv_path, index=True)
    print(f"Saved {ticker} -> {csv_path} (rows: {len(df)})")


def main():
    for t in TICKERS:
        fetch_and_save(t, START, END, OUT_DIR)
    print("All done.")


if __name__ == "__main__":
    main()

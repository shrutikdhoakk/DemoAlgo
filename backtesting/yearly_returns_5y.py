# backtesting/yearly_returns_5y.py
# Trailing 5-year yearly returns for each NIFTY500 stock (+ 5Y Total & CAGR)
# Run from repo root:
#   python -m backtesting.yearly_returns_5y
#
# Env knobs (optional):
#   NIFTY500_PATH = "nifty500.csv"
#   UNIVERSE_LIMIT = "500"          # cap tickers for faster tests (e.g., "50")
#   APPEND_NS = "1"                 # append ".NS" to symbols if not present
#   INCLUDE_YTD = "1"               # include current year as YTD column

from __future__ import annotations
import os, time, math, datetime as dt
import numpy as np
import pandas as pd

try:
    import yfinance as yf  # pip install yfinance
except Exception as e:
    raise SystemExit("Please install yfinance:  pip install yfinance") from e

# -------- settings --------
NIFTY500_PATH = os.getenv("NIFTY500_PATH", "nifty500.csv")
UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "500"))
APPEND_NS = os.getenv("APPEND_NS", "1") == "1"
INCLUDE_YTD = os.getenv("INCLUDE_YTD", "1") == "1"

TODAY = dt.date.today()
CURR_YEAR = TODAY.year
YEARS = 5

# Choose the 5 calendar years to output:
# If INCLUDE_YTD: last 5 calendar years including current year (current year is YTD)
# Else: last 5 *completed* years (excludes the current year)
if INCLUDE_YTD:
    YEARS_LIST = list(range(CURR_YEAR - YEARS + 1, CURR_YEAR + 1))  # e.g., 2021..2025
else:
    YEARS_LIST = list(range(CURR_YEAR - YEARS, CURR_YEAR))          # e.g., 2020..2024

START_DATE = dt.date(min(YEARS_LIST), 1, 1)

# -------- helpers --------
def _read_symbols(path: str, limit: int) -> list[str]:
    df = pd.read_csv(path)
    # Try common column names
    for c in ["SYMBOL", "Symbol", "Ticker", "ticker", "symbol"]:
        if c in df.columns:
            syms = df[c].astype(str).str.strip().tolist()
            break
    else:
        # if the file only has a single column
        if df.shape[1] == 1:
            syms = df.iloc[:, 0].astype(str).str.strip().tolist()
        else:
            raise ValueError(f"Could not find a symbol column in {path}")
    # sanitize
    syms = [s.upper().replace(".NS", "").strip() for s in syms if isinstance(s, str) and s.strip()]
    if APPEND_NS:
        syms = [f"{s}.NS" for s in syms]
    if limit > 0:
        syms = syms[:limit]
    return syms

def _download_multi(tickers: list[str], start: dt.date) -> pd.DataFrame:
    """
    Download OHLCV for many tickers.
    Returns a tidy DataFrame with columns: ['Ticker','Date','Open','High','Low','Close','Volume']
    """
    if not tickers:
        return pd.DataFrame()

    # yfinance can fetch multiple tickers; to avoid URL length issues, chunk them
    chunks = []
    BATCH = 50
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        df = yf.download(batch, start=start.isoformat(), progress=False, group_by="ticker", auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            # reshape to tidy
            df = df.stack(0).reset_index().rename(columns={"level_1":"Ticker", "Date":"Date"})
        else:
            # single ticker fallback
            df = df.reset_index()
            df["Ticker"] = batch[0]
        # normalize column names
        rename = {c: c.title() for c in df.columns}
        df = df.rename(columns=rename)
        need = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        avail = [c for c in need if c in df.columns]
        df = df[avail].dropna()
        chunks.append(df)
        time.sleep(0.6)  # be polite to Yahoo
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    # ensure Date is date (not datetime) for grouping
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"]).dt.date
    return out

def _yearly_returns_for_symbol(df_sym: pd.DataFrame) -> dict[int, float]:
    """
    For a single symbol's tidy frame with columns Date, Close, compute calendar-year returns.
    Return dict {year: return_pct} where return_pct is in percent (e.g., 12.34 for +12.34%)
    For current year: YTD from first trading day of year to latest available.
    """
    if df_sym.empty:
        return {}

    df = df_sym.sort_values("Date").copy()
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    result = {}
    for yr in YEARS_LIST:
        sub = df[df["Year"] == yr]
        if sub.empty:
            continue
        first_close = float(sub["Close"].iloc[0])
        last_close = float(sub["Close"].iloc[-1])
        if first_close > 0 and math.isfinite(first_close) and math.isfinite(last_close):
            ret = (last_close / first_close - 1.0) * 100.0
            result[yr] = ret
    return result

def _five_year_total_and_cagr(df_sym: pd.DataFrame) -> tuple[float, float]:
    """
    Compute 5Y total return and 5Y CAGR over the entire window covering YEARS_LIST.
    Uses first available close on/after START_DATE and last available close <= today.
    Returns (total_pct, cagr_pct); if not enough data, returns (nan, nan).
    """
    if df_sym.empty:
        return (np.nan, np.nan)

    df = df_sym.sort_values("Date").copy()
    df = df[df["Date"] >= START_DATE]
    if df.empty:
        return (np.nan, np.nan)

    start_close = float(df["Close"].iloc[0])
    end_close   = float(df["Close"].iloc[-1])
    if not (math.isfinite(start_close) and math.isfinite(end_close)) or start_close <= 0:
        return (np.nan, np.nan)

    total = (end_close / start_close - 1.0) * 100.0
    # years elapsed (approx): exact calendar span between first & last dates
    years_elapsed = max(1e-9, (pd.to_datetime(df["Date"].iloc[-1]) - pd.to_datetime(df["Date"].iloc[0])).days / 365.25)
    cagr = ( (end_close / start_close) ** (1.0 / years_elapsed) - 1.0 ) * 100.0
    return (total, cagr)

def compute_yearly_returns():
    tickers = _read_symbols(NIFTY500_PATH, UNIVERSE_LIMIT)
    print(f"[yr] Loaded {len(tickers)} tickers from {NIFTY500_PATH} (limit={UNIVERSE_LIMIT})")
    prices = _download_multi(tickers, START_DATE)
    if prices.empty:
        print("[yr] No data downloaded.")
        return

    # keep needed cols
    need = ["Ticker", "Date", "Close"]
    for c in need:
        if c not in prices.columns:
            raise RuntimeError(f"Missing column {c} in downloaded data")
    # compute per-symbol
    rows_long = []
    rows_wide = []
    for tkr, df_sym in prices.groupby("Ticker"):
        yr_map = _yearly_returns_for_symbol(df_sym[["Date","Close"]])
        total5, cagr5 = _five_year_total_and_cagr(df_sym[["Date","Close"]])
        # long rows
        for yr in YEARS_LIST:
            val = yr_map.get(yr, np.nan)
            rows_long.append({"Symbol": tkr, "Year": yr, "ReturnPct": round(val, 2) if pd.notna(val) else np.nan})
        # wide row
        wide_entry = {"Symbol": tkr, "Total5Y%": round(total5, 2) if pd.notna(total5) else np.nan,
                      "CAGR5Y%": round(cagr5, 2) if pd.notna(cagr5) else np.nan}
        for yr in YEARS_LIST:
            wide_entry[str(yr)+"%"] = round(yr_map.get(yr, np.nan), 2) if yr in yr_map else np.nan
        rows_wide.append(wide_entry)

    df_long = pd.DataFrame(rows_long).sort_values(["Symbol","Year"])
    df_wide = pd.DataFrame(rows_wide).sort_values("Symbol")

    out1 = "nifty500_returns_5y_long.csv"
    out2 = "nifty500_returns_5y_matrix.csv"
    df_long.to_csv(out1, index=False)
    df_wide.to_csv(out2, index=False)

    print(f"[yr] Saved: {out1}  (columns: Symbol, Year, ReturnPct)")
    print(f"[yr] Saved: {out2}  (columns: Symbol, {', '.join([str(y)+'%' for y in YEARS_LIST])}, Total5Y%, CAGR5Y%)")

if __name__ == "__main__":
    compute_yearly_returns()

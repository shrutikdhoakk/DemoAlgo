# backtesting/portfolio_backtest_5y.py
# 5Y portfolio backtest over NIFTY500 with risk-based sizing & fixed stop %
# Run:
#   cd <repo root>
#   python -m backtesting.portfolio_backtest_5y
#
# Env knobs (optional):
#   CAPITAL=1000000           # starting cash (₹)
#   STOP_PCT=0.055            # 5.5% hard stop
#   RISK_PCT=0.05             # 5% of current equity risked per trade
#   MAX_POS=10                # max open positions
#   NIFTY500_PATH=nifty500.csv
#   UNIVERSE_LIMIT=500
#   APPEND_NS=1               # append ".NS" if missing
#   START_DATE=2019-01-01     # pull slightly >5y to warm indicators
#   END_DATE=                  # default = today
#
# Outputs in repo root:
#   portfolio_trades.csv
#   portfolio_equity.csv
#   portfolio_yearly_returns.csv

from __future__ import annotations
import os, math, time, datetime as dt
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please install yfinance first: pip install yfinance") from e

# ------------------- config -------------------
CAPITAL = float(os.getenv("CAPITAL", "1000000"))
STOP_PCT = float(os.getenv("STOP_PCT", "0.055"))     # 5.5% hard stop
RISK_PCT = float(os.getenv("RISK_PCT", "0.05"))      # risk per trade = 5% of current equity
MAX_POS  = int(os.getenv("MAX_POS", "10"))

NIFTY500_PATH  = os.getenv("NIFTY500_PATH", "nifty500.csv")
UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "500"))
APPEND_NS      = os.getenv("APPEND_NS", "1") == "1"

START_DATE = os.getenv("START_DATE", "2019-01-01")
END_DATE   = os.getenv("END_DATE", "")

ENTRY_LOOKBACK = 20    # breakout window
EXIT_LOOKBACK  = 20
RSI_N  = 14
ADX_N  = 14

# ------------------- helpers -------------------
def ensure_date_col(df: pd.DataFrame, name: str = "Date") -> pd.DataFrame:
    """
    Guarantee a tz-naive datetime column named `Date`, even if the date is in the index
    or named differently (e.g., 'Datetime', 'level_0', 'index').
    """
    if name in df.columns:
        df[name] = pd.to_datetime(df[name], errors="coerce").dt.tz_localize(None)
        return df
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": name})
        df[name] = pd.to_datetime(df[name], errors="coerce").dt.tz_localize(None)
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis(name).reset_index()
        df[name] = pd.to_datetime(df[name], errors="coerce").dt.tz_localize(None)
        return df
    for alt in ("level_0", "index"):
        if alt in df.columns:
            df = df.rename(columns={alt: name})
            df[name] = pd.to_datetime(df[name], errors="coerce").dt.tz_localize(None)
            return df
    return df

def read_symbols(path: str, limit: int) -> list[str]:
    df = pd.read_csv(path)
    for c in ["SYMBOL","Symbol","Ticker","ticker","symbol"]:
        if c in df.columns:
            syms = df[c].astype(str).str.strip().tolist()
            break
    else:
        if df.shape[1] == 1:
            syms = df.iloc[:,0].astype(str).str.strip().tolist()
        else:
            raise ValueError(f"No symbol column in {path}")
    syms = [s.upper().replace(".NS","").strip() for s in syms if isinstance(s,str) and s.strip()]
    if APPEND_NS:
        syms = [f"{s}.NS" for s in syms]
    if limit>0:
        syms = syms[:limit]
    return syms

def download_prices(tickers: list[str], start: str, end: str|None) -> pd.DataFrame:
    """
    Download multi-ticker OHLCV and normalize to columns:
    ['Ticker','Date','Open','High','Low','Close','Volume']
    Works for both single- and multi-ticker yfinance outputs. Pandas ≥2.1 safe.
    """
    if not tickers:
        return pd.DataFrame()
    chunks, B = [], 30  # smaller batch reduces Yahoo timeouts
    for i in range(0, len(tickers), B):
        batch = tickers[i:i+B]
        df = yf.download(
            batch, start=start, end=(end or None),
            progress=False, group_by="ticker", auto_adjust=False
        )
        # Normalize shape
        if isinstance(df.columns, pd.MultiIndex):
            # NEW: future_stack=True to silence deprecation
            df = df.stack(0, future_stack=True).reset_index()
            # Ticker usually in level_1 after stacking:
            if "level_1" in df.columns and "Ticker" not in df.columns:
                df = df.rename(columns={"level_1": "Ticker"})
        else:
            df = df.reset_index()
            # Single-ticker case: add explicit Ticker column
            if "Ticker" not in df.columns:
                df["Ticker"] = batch[0]

        # Title-case for uniform downstream access, then ensure Date column exists & is tz-naive
        df.columns = [str(c).title() for c in df.columns]
        df = ensure_date_col(df, "Date")

        # Keep needed cols
        need = ["Ticker","Date","Open","High","Low","Close","Volume"]
        df = df[need].dropna()
        chunks.append(df)
        time.sleep(0.4)

    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return out

def rsi(close: pd.Series, n: int=14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").astype(float)
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100/(1+rs))
    return out.fillna(50.0)

def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int=14) -> pd.Series:
    h = pd.to_numeric(h, errors="coerce").astype(float)
    l = pd.to_numeric(l, errors="coerce").astype(float)
    c = pd.to_numeric(c, errors="coerce").astype(float)
    up = h.diff()
    dn = -l.diff()
    plusDM  = ((up > dn) & (up > 0)).astype(float) * up.clip(lower=0)
    minusDM = ((dn > up) & (dn > 0)).astype(float) * dn.clip(lower=0)
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pDI = 100 * (plusDM.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0,np.nan))
    mDI = 100 * (minusDM.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0,np.nan))
    dx = 100 * ( (pDI - mDI).abs() / (pDI + mDI).replace(0,np.nan) )
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().fillna(0)

def yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    df = equity.copy()
    df["Year"] = df.index.year
    rows=[]
    for y, sub in df.groupby("Year"):
        if sub.empty: continue
        ret = (sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1)*100
        rows.append({"Year": int(y), "PortfolioReturn%": round(ret,2)})
    return pd.DataFrame(rows).sort_values("Year")

# ------------------- backtest -------------------
def run_backtest():
    tickers = read_symbols(NIFTY500_PATH, UNIVERSE_LIMIT)
    print(f"[bt] Universe size: {len(tickers)}")
    prices = download_prices(tickers, START_DATE, END_DATE or None)
    if prices.empty:
        print("[bt] No data. Aborting.")
        return

    # build per-symbol frames with indicators & breakouts
    frames = {}
    for tkr, df in prices.groupby("Ticker"):
        df = df.sort_values("Date").set_index("Date")
        df["RSI"] = rsi(df["Close"], RSI_N)
        df["ADX"] = adx(df["High"], df["Low"], df["Close"], ADX_N)
        df["HH"]  = df["High"].rolling(ENTRY_LOOKBACK).max().shift(1)  # yesterday's 20D high
        # (the first assignment to LL was a max by mistake; keep the min one)
        df["LL"]  = df["Low"].rolling(EXIT_LOOKBACK).min().shift(1)    # yesterday's 20D low
        frames[tkr] = df

    # master calendar
    all_days = pd.Index(sorted(prices["Date"].unique()))
    cash = CAPITAL
    equity = CAPITAL
    positions = {}  # tkr -> dict(entry, shares, stop)
    trades = []

    def size_position(px: float, eq: float) -> int:
        risk_budget = eq * RISK_PCT
        stop_dist   = max(1e-9, px * STOP_PCT)
        qty = int(risk_budget // stop_dist)
        return max(0, qty)

    for day in all_days:
        # mark-to-market
        mtm = 0.0
        for tkr, pos in list(positions.items()):
            px = frames[tkr].loc[day]["Close"] if day in frames[tkr].index else np.nan
            if not math.isfinite(float(px)):
                continue
            mtm += pos["qty"] * float(px)
        equity = cash + mtm

        # ---- exits first
        for tkr, pos in list(positions.items()):
            df = frames[tkr]
            if day not in df.index:  # no bar today
                continue
            row = df.loc[day]
            px_close = float(row["Close"])
            px_low   = float(row["Low"])
            stop_px  = float(pos["stop"])

            exit_reason = None
            exit_px = None
            # hard stop (intraday low breach)
            if px_low <= stop_px:
                exit_reason = "STOP"
                exit_px = stop_px
            # breakdown / weakness at close
            elif px_close < float(row["LL"]) or float(row["RSI"]) < 45:
                exit_reason = "RULE"
                exit_px = px_close

            if exit_reason:
                cash += pos["qty"] * exit_px
                trades.append({
                    "date": day, "symbol": tkr, "side": "SELL",
                    "price": round(exit_px,2), "qty": pos["qty"], "reason": exit_reason
                })
                del positions[tkr]

        # recompute equity after exits
        mtm = 0.0
        for tkr, pos in positions.items():
            px = frames[tkr].loc[day]["Close"] if day in frames[tkr].index else np.nan
            if math.isfinite(float(px)):
                mtm += pos["qty"] * float(px)
        equity = cash + mtm

        # ---- entries
        # build candidate list with signals today
        cands = []
        for tkr, df in frames.items():
            if tkr in positions or day not in df.index:
                continue
            row = df.loc[day]
            # signal today? close > yesterday's 20D high, RSI>50, ADX>15
            try:
                cond = (
                    float(row["Close"]) > float(row["HH"]) and
                    float(row["RSI"])   > 50.0 and
                    float(row["ADX"])   > 15.0
                )
            except Exception:
                cond = False
            if cond:
                cands.append((tkr, float(row["ADX"]), float(row["Close"]), float(row["High"]), float(row["Low"])))

        # rank by ADX desc (stronger trend first)
        cands.sort(key=lambda x: x[1], reverse=True)

        # fill up to MAX_POS
        for tkr, adx_val, px_close, px_high, px_low in cands:
            if len(positions) >= MAX_POS:
                break
            qty = size_position(px_close, equity)
            if qty <= 0:  # not enough risk budget/cash
                continue
            notional = qty * px_close
            if notional > cash:
                # try smaller qty fitting cash
                qty = int(cash // px_close)
            if qty <= 0:
                continue
            stop_px = px_close * (1.0 - STOP_PCT)

            # place trade at close
            cash -= qty * px_close
            positions[tkr] = {"entry": px_close, "qty": qty, "stop": stop_px}
            trades.append({
                "date": day, "symbol": tkr, "side": "BUY",
                "price": round(px_close,2), "qty": qty, "reason": f"ADX={adx_val:.1f}"
            })

        # record equity line
        # (we’ll append later outside loop to keep simple)

    # finalize MTM on last day
    equity_curve = []
    # rebuild daily equity cleanly
    cash2 = CAPITAL
    positions2 = {}
    for day in all_days:
        # replay trades that occurred this day
        for tr in [t for t in trades if t["date"] == day]:
            if tr["side"] == "BUY":
                cash2 -= tr["qty"] * tr["price"]
                positions2[tr["symbol"]] = tr["qty"]
            else:
                cash2 += tr["qty"] * tr["price"]
                positions2.pop(tr["symbol"], None)
        # MTM
        mtm2 = 0.0
        for tkr, qty in positions2.items():
            df = frames[tkr]
            if day in df.index and math.isfinite(float(df.loc[day,"Close"])):
                mtm2 += qty * float(df.loc[day,"Close"])
        equity_curve.append((pd.to_datetime(day), cash2 + mtm2, cash2, mtm2, len(positions2)))

    ec = pd.DataFrame(equity_curve, columns=["date","equity","cash","mtm","positions"]).set_index("date")
    ec.to_csv("portfolio_equity.csv", index=True)

    # trades
    td = pd.DataFrame(trades)
    if not td.empty:
        td = td.sort_values(["date","symbol","side"])
    td.to_csv("portfolio_trades.csv", index=False)

    # yearly summary
    yr = yearly_returns(ec)
    yr.to_csv("portfolio_yearly_returns.csv", index=False)

    # print tail preview
    print(ec.tail(8))
    print("\n[bt] Saved: portfolio_trades.csv, portfolio_equity.csv, portfolio_yearly_returns.csv")

if __name__ == "__main__":
    run_backtest()

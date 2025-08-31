# backtesting/swing_backtest_5y.py
# 5Y backtest that mirrors swing_strategy.py logic:
# - Volume spike (>= 1.5x 20D avg)
# - Breakout above max(20D/50D highs) with optional tolerance (EASY_SIM)
# - RSI(7/14/21) > 40, ADX14 > 20
# - Regime filter: Nifty EMA50 > EMA200; purge when risk-off
# - Trailing stop: stop = max(stop, Close - 2*ATR14)
# - Exits: stop OR Close < SMA20 OR RSI14 < 30
# - Sizing: min( cash-slot size , MAX_RISK_PER_TRADE / ATR14 )
# - Portfolio caps: MAX_POS and MAX_INVESTED
#
# Run:
#   python -m backtesting.swing_backtest_5y

from __future__ import annotations
import os, math, time, datetime as dt
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("Please: pip install yfinance pandas numpy") from e

# ---------- env / knobs ----------
CAPITAL = float(os.getenv("CAPITAL", "1000000"))
MAX_POS = int(os.getenv("MAX_POS", "5"))
MAX_INVESTED = float(os.getenv("MAX_INVESTED", "600000"))      # total notional cap
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "50000"))
NIFTY500_PATH = os.getenv("NIFTY500_PATH", "nifty500.csv")
UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "500"))
APPEND_NS = os.getenv("APPEND_NS", "1") == "1"
EASY_SIM = os.getenv("EASY_SIM", "0") == "1"                    # tolerance for breakout
START_DATE = os.getenv("START_DATE", "2019-01-01")
END_DATE = os.getenv("END_DATE", "")

ENTRY_LOOKBACK_20 = 20
ENTRY_LOOKBACK_50 = 50
RSI_N = (7, 14, 21)
ADX_N = 14
ATR_N = 14

# ---------- helpers ----------
def read_symbols(path: str, limit: int) -> list[str]:
    df = pd.read_csv(path)
    for c in ["SYMBOL", "Symbol", "Ticker", "ticker", "symbol"]:
        if c in df.columns:
            syms = df[c].astype(str).str.strip().tolist()
            break
    else:
        syms = df.iloc[:, 0].astype(str).str.strip().tolist()
    syms = [s.upper().replace(".NS", "").strip() for s in syms if s.strip()]
    if APPEND_NS:
        syms = [f"{s}.NS" for s in syms]
    return syms[:limit] if limit > 0 else syms

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to columns: Date, Ticker, Open, High, Low, Close, Volume
    Works for both single- and multi-ticker downloads. Pandas ≥2.1 safe.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # CHANGED: silence FutureWarning and get consistent shape
        df = df.stack(0, future_stack=True).reset_index()
        # After stacking level 0, the former top-level (ticker) becomes 'level_1'
        if "level_1" in df.columns and "Ticker" not in df.columns:
            df = df.rename(columns={"level_1": "Ticker"})
    else:
        df = df.reset_index()

    # Normalize datetime column naming
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "level_0" in df.columns:
            df = df.rename(columns={"level_0": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})

    # Title-case columns for uniform downstream access
    df.columns = [str(c).title() for c in df.columns]
    return df

def download_multi(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    chunks, B = [], 50
    for i in range(0, len(tickers), B):
        batch = tickers[i : i + B]
        df = yf.download(batch, start=start, end=(end or None), progress=False, group_by="ticker", auto_adjust=False)
        df = _flatten(df)
        if "Ticker" not in df.columns:
            # single-ticker case: yfinance returns plain OHLCV columns
            df["Ticker"] = batch[0]
        # keep needed
        need = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[need].dropna()
        chunks.append(df)
        time.sleep(0.4)  # be polite
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out

def rsi(series: pd.Series, n: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = pd.to_numeric(df["High"], errors="coerce").astype(float)
    l = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    c = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = pd.to_numeric(df["High"], errors="coerce").astype(float)
    l = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    c = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    up = h.diff()
    dn = -l.diff()
    plusDM = ((up > dn) and (up > 0)).astype(float) * up.clip(lower=0) if False else ((up > dn) & (up > 0)).astype(float) * up.clip(lower=0)
    minusDM = ((dn > up) and (dn > 0)).astype(float) * dn.clip(lower=0) if False else ((dn > up) & (dn > 0)).astype(float) * dn.clip(lower=0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.rolling(n, min_periods=n).mean().replace(0, np.nan)
    pDI = 100 * (plusDM.rolling(n, min_periods=n).mean() / atr_)
    mDI = 100 * (minusDM.rolling(n, min_periods=n).mean() / atr_)
    dx = 100 * ((pDI - mDI).abs() / (pDI + mDI).replace(0, np.nan))
    return dx.rolling(n, min_periods=n).mean().fillna(0)

def ema(s: pd.Series, span: int, minp: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=minp).mean()

def yearly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    df = equity.copy()
    df["Year"] = df.index.year
    rows = []
    for y, sub in df.groupby("Year"):
        if sub.empty: continue
        ret = (sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1) * 100
        rows.append({"Year": int(y), "PortfolioReturn%": round(ret, 2)})
    return pd.DataFrame(rows).sort_values("Year")

# ---------- build per-symbol frames with indicators ----------
def build_symbol_frames(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    for tkr, df in prices.groupby("Ticker"):
        df = df.sort_values("Date").set_index("Date")
        # indicators
        df["RSI7"] = rsi(df["Close"], 7)
        df["RSI14"] = rsi(df["Close"], 14)
        df["RSI21"] = rsi(df["Close"], 21)
        df["ADX"] = adx(df, ADX_N)
        df["ATR"] = atr(df, ATR_N)
        df["High20"] = pd.Series(df["High"]).rolling(ENTRY_LOOKBACK_20, min_periods=ENTRY_LOOKBACK_20).max()
        df["High50"] = pd.Series(df["High"]).rolling(ENTRY_LOOKBACK_50, min_periods=ENTRY_LOOKBACK_50).max()
        df["SMA20"] = pd.Series(df["Close"]).rolling(20, min_periods=20).mean()
        df["Vol20"] = pd.Series(df["Volume"]).rolling(20, min_periods=20).mean()
        frames[tkr] = df
    return frames

# ---------- regime data ----------
def fetch_nifty(start: str, end: str | None) -> pd.DataFrame:
    idx = yf.download("^NSEI", start=start, end=(end or None), progress=False, auto_adjust=False)
    if idx.empty:
        raise RuntimeError("Index (^NSEI) download failed")

    # CHANGED: robustly create a Date column from the index
    idx = idx.rename_axis("Date").reset_index()
    if "Datetime" in idx.columns and "Date" not in idx.columns:
        idx = idx.rename(columns={"Datetime": "Date"})

    idx.columns = [str(c).title() for c in idx.columns]
    idx["Date"] = pd.to_datetime(idx["Date"], errors="coerce").dt.tz_localize(None)
    idx = idx[["Date", "Close"]].dropna()
    idx = idx.sort_values("Date").set_index("Date")
    idx["EMA50"] = ema(idx["Close"], 50, 50)
    idx["EMA200"] = ema(idx["Close"], 200, 200)
    return idx

# ---------- scoring & signal check (mirrors swing_strategy.py) ----------
def passes_entry(row) -> bool:
    # volume spike
    try:
        v_ok = float(row["Volume"]) >= 1.5 * float(row["Vol20"])
    except Exception:
        v_ok = False
    if not v_ok:
        return False

    c = float(row["Close"])
    rh20 = float(row["High20"])
    rh50 = float(row["High50"])
    if any(np.isnan([c, rh20, rh50])):
        return False

    tol = 0.995 if EASY_SIM else 1.0
    cond_breakout = (c > tol * max(rh20, rh50))
    cond_rsi = (float(row["Rsi14"]) > 40.0) and (float(row["Rsi7"]) > 40.0) and (float(row["Rsi21"]) > 40.0)
    cond_adx = float(row["Adx"]) > 20.0
    return bool(cond_breakout and cond_rsi and cond_adx)

def score_row(row) -> float:
    c = float(row["Close"]); rh20 = float(row["High20"]); rh50 = float(row["High50"])
    adxv = float(row["Adx"]); r14 = float(row["Rsi14"])
    eps = 1e-9
    breakout_ratio = max(c / max(rh20, eps), c / max(rh50, eps))
    return 0.6 * breakout_ratio + 0.3 * (adxv / 25.0) + 0.1 * (r14 / 50.0)

# ---------- backtest ----------
def main():
    tickers = read_symbols(NIFTY500_PATH, UNIVERSE_LIMIT)
    print(f"[bt] Universe: {len(tickers)} symbols")

    prices = download_multi(tickers, START_DATE, END_DATE or None)
    if prices.empty:
        print("[bt] No symbol data; abort.")
        return

    frames = build_symbol_frames(prices)
    all_days = pd.Index(sorted(prices["Date"].unique()))

    # Nifty regime
    idx = fetch_nifty(START_DATE, END_DATE or None)

    cash = CAPITAL
    positions: dict[str, dict] = {}   # symbol -> {qty, stop, last_close}
    trades = []
    equity_curve = []

    for day in all_days:
        day = pd.to_datetime(day)
        # regime check
        regime_ok = True
        if day in idx.index:
            ema50 = float(idx.loc[day, "Ema50"]) if "Ema50" in idx.columns else float(idx.loc[day, "EMA50"])
            ema200 = float(idx.loc[day, "Ema200"]) if "Ema200" in idx.columns else float(idx.loc[day, "EMA200"])
            if not (np.isnan(ema50) or np.isnan(ema200)):
                regime_ok = ema50 > ema200

        # mark-to-market
        mtm = 0.0
        for sym, pos in positions.items():
            df = frames[sym]
            if day in df.index and np.isfinite(df.loc[day, "Close"]):
                mtm += pos["qty"] * float(df.loc[day, "Close"])
        equity = cash + mtm

        # ---- exits first (and trailing stop update)
        for sym, pos in list(positions.items()):
            df = frames[sym]
            if day not in df.index:
                continue
            row = df.loc[day]

            # trailing stop: max(prev_stop, Close - 2*ATR)
            atrv = float(row["Atr"]) if "Atr" in df.columns else float(row["ATR"])
            cl = float(row["Close"])
            new_stop = cl - 2.0 * atrv if np.isfinite(atrv) and np.isfinite(cl) else pos["stop"]
            if new_stop > pos["stop"]:
                pos["stop"] = new_stop

            # exit rules: stop OR Close < SMA20 OR RSI14 < 30 OR regime off
            sma20 = float(row["Sma20"]) if "Sma20" in df.columns else float(row["SMA20"])
            r14 = float(row["Rsi14"]) if "Rsi14" in df.columns else float(row["RSI14"])

            exit_now = False
            exit_reason = ""
            if cl <= pos["stop"]:
                exit_now = True; exit_reason = "STOP"
            elif np.isfinite(sma20) and cl < sma20:
                exit_now = True; exit_reason = "SMA20"
            elif np.isfinite(r14) and r14 < 30.0:
                exit_now = True; exit_reason = "RSI<30"
            elif not regime_ok:
                exit_now = True; exit_reason = "RISK_OFF"

            if exit_now:
                cash += pos["qty"] * cl
                trades.append({"date": day, "symbol": sym, "side": "SELL", "price": round(cl,2),
                               "qty": pos["qty"], "reason": exit_reason})
                del positions[sym]

        # recompute equity after exits
        mtm = 0.0
        for sym, pos in positions.items():
            df = frames[sym]
            if day in df.index and np.isfinite(df.loc[day, "Close"]):
                mtm += pos["qty"] * float(df.loc[day, "Close"])
        equity = cash + mtm

        # ---- entries (only if risk-on)
        if regime_ok and len(positions) < MAX_POS and MAX_INVESTED > 0:
            invested_now = 0.0
            for s, p in positions.items():
                df = frames[s]
                if day in df.index and np.isfinite(df.loc[day, "Close"]):
                    invested_now += p["qty"] * float(df.loc[day, "Close"])
            remaining_budget = max(0.0, MAX_INVESTED - invested_now)
            if remaining_budget > 0:
                per_slot = MAX_INVESTED / float(MAX_POS)

                # build candidate list
                cands = []
                for sym, df in frames.items():
                    if sym in positions or day not in df.index:
                        continue
                    row = df.loc[day]
                    # ensure required columns exist for that day
                    needed = ["Close","High20","High50","Rsi7","Rsi14","Rsi21","Adx","Atr","Volume","Vol20"]
                    if any(k not in df.columns for k in needed):
                        continue
                    if any(pd.isna(row[k]) for k in needed):
                        continue
                    if not passes_entry(row):
                        continue
                    sc = score_row(row)
                    cands.append((sym, sc, float(row["Close"]), float(row["Atr"])))

                # rank by score desc
                cands.sort(key=lambda x: x[1], reverse=True)

                for sym, sc, px, atrv in cands:
                    if len(positions) >= MAX_POS or remaining_budget <= 0:
                        break
                    # cash slot sizing
                    budget = min(per_slot, remaining_budget, cash)
                    qty_cash = int(budget // max(px, 1e-9))
                    # ATR risk sizing
                    qty_risk = int(max(1.0, MAX_RISK_PER_TRADE / max(atrv, 1e-9)))
                    qty = max(1, min(qty_cash, qty_risk))
                    notional = qty * px
                    if qty < 1 or notional <= 0:
                        continue
                    if notional > cash:  # fit to cash
                        qty = int(cash // max(px, 1e-9))
                        notional = qty * px
                        if qty < 1:
                            continue

                    # open position
                    cash -= notional
                    stop0 = px - 2.0 * atrv
                    positions[sym] = {"qty": qty, "stop": stop0, "entry": px}
                    trades.append({"date": day, "symbol": sym, "side": "BUY", "price": round(px,2),
                                   "qty": qty, "reason": f"score={sc:.3f}"})
                    remaining_budget -= notional

        # record equity
        equity_curve.append((day, equity, cash, len(positions)))

    # finalize outputs
    ec = pd.DataFrame(equity_curve, columns=["date","equity","cash","positions"]).set_index("date")
    ec.to_csv("swing_equity.csv", index=True)

    td = pd.DataFrame(trades)
    if not td.empty:
        td = td.sort_values(["date","symbol","side"])
    td.to_csv("swing_trades.csv", index=False)

    yr = yearly_returns(ec)
    yr.to_csv("swing_yearly_returns.csv", index=False)

    print(ec.tail(8))
    print("\n[bt] Saved: swing_trades.csv, swing_equity.csv, swing_yearly_returns.csv")

if __name__ == "__main__":
    main()

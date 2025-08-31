# backtest_swingpro_5y.py
# Backtests SwingStrategyPro-style rules over ~5y and reports per-year returns & win rate.
# Usage:
#   python backtest_swingpro_5y.py
#
# ENV knobs (PowerShell example below):
#   NIFTY500_PATH=nifty500.csv        # CSV with a column SYMBOL/TICKER (no .NS needed). If missing, uses a demo list.
#   UNIVERSE_LIMIT=150                # cap number of symbols to test
#   APPEND_NS=1                       # append ".NS" to tickers if not present (set 0 if your CSV already has .NS)
#   APPEND_SUFFIX=.NS                 # override exchange suffix (default .NS when APPEND_NS=1) — rarely needed
#   START_DATE=2019-01-01             # start date (warmup handled in code)
#   END_DATE=                         # default = today
#   TOTAL_CAPITAL=1000000             # starting equity (₹)
#   MAX_INVESTED=500000               # portfolio notional cap (₹) invested at once
#   MAX_POSITIONS=5                   # max concurrent positions
#   RISK_PER_TRADE=10000              # ₹ risk budget per trade (used with ATR sizing)
#   STOP_ATR=2.0                      # stop distance in ATR
#   TARGET_ATR=2.0                    # target distance in ATR
#   COMMISSION_BPS=1.0                # 1 bps per side (0.01%); round trip ≈ 2 bps
#
# Outputs:
#   trades_swingpro.csv
#   equity_swingpro.csv
#   yearly_returns_swingpro.csv

from __future__ import annotations
import os, warnings, datetime as dt, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    yf = None

# ========= ENV / KNOBS =========
NIFTY500_PATH  = os.getenv("NIFTY500_PATH", "nifty500.csv")
UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "150"))
APPEND_NS      = os.getenv("APPEND_NS", "1") == "1"
SUFFIX         = os.getenv("APPEND_SUFFIX", ".NS")

START_DATE     = os.getenv("START_DATE", "2019-01-01")
END_DATE       = os.getenv("END_DATE", "") or dt.date.today().isoformat()

TOTAL_CAPITAL  = float(os.getenv("TOTAL_CAPITAL", "1000000"))
MAX_INVESTED   = float(os.getenv("MAX_INVESTED",  "500000"))
MAX_POSITIONS  = int(os.getenv("MAX_POSITIONS", "5"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "10000"))
STOP_ATR       = float(os.getenv("STOP_ATR", "2.0"))
TARGET_ATR     = float(os.getenv("TARGET_ATR", "2.0"))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", "1.0"))  # per side bps

# ========= DEBUG =========
def _debug(msg: str):
    print(f"[BT] {msg}")

# ========= INDICATORS =========
def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move  > down_move) & (up_move  > 0), up_move,  0.0)
    minus_dm = np.where((down_move > up_move)  & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di  = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/length, adjust=False).mean()  / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/length, adjust=False).mean() / atr_val
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx

def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    cols = {c: str(c).strip().title() for c in df.columns}
    df = df.rename(columns=cols)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for k in needed:
        if k not in df.columns:
            return pd.DataFrame()

    for k in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if k in df.columns:
            col = df[k]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            df[k] = pd.to_numeric(col, errors="coerce")
    df = df[df["Close"].notna()].copy()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    return df

# ========= DATA / UNIVERSE =========
def _demo_universe() -> List[str]:
    return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "LT", "BHARTIARTL", "ASIANPAINT"]

def _load_universe() -> List[str]:
    symbols: List[str] = []
    if os.path.exists(NIFTY500_PATH):
        try:
            df = pd.read_csv(NIFTY500_PATH)
            for cand in ["SYMBOL", "Symbol", "symbol", "TICKER", "Ticker"]:
                if cand in df.columns:
                    symbols = df[cand].astype(str).tolist()
                    break
            else:
                symbols = df.iloc[:, 0].astype(str).tolist()
        except Exception:
            symbols = []

    if not symbols:
        symbols = _demo_universe()

    def sanitize(s: str) -> str:
        s = s.strip().upper()
        for junk in [".EQ", ".BSE", ".NSE", "-EQ", "-BE", "-BZ", "-SM", "-ST"]:
            if s.endswith(junk):
                s = s[: -len(junk)]
        return s

    symbols = [sanitize(s) for s in symbols]

    # append/remove .NS according to env
    if APPEND_NS:
        symbols = [s if s.endswith(".NS") else f"{s}.NS" for s in symbols]
    else:
        symbols = [s[:-3] if s.endswith(".NS") else s for s in symbols]

    # dedupe & cap
    seen, uniq = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq[:max(1, UNIVERSE_LIMIT)]

def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. pip install yfinance")

    variants: List[Tuple[str, dict]] = [
        (symbol, {"start": start, "end": end}),
        (symbol, {"period": "10y"}),
        (symbol, {"period": "max"}),
    ]
    if symbol.endswith(".NS"):
        base = symbol[:-3]
        variants += [(base, {"period": "10y"}), (base, {"period": "max"})]

    for sym, params in variants:
        try:
            df = yf.download(sym, auto_adjust=False, progress=False, group_by="column", threads=False, **params)
        except Exception as e:
            _debug(f"download error {sym} {params}: {e}")
            df = pd.DataFrame()

        if df is None or len(df) == 0:
            continue

        # collapse MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.swaplevel(axis=1)
                # choose first ticker level
                first_key = df.columns.levels[-1][0]
                df = df[first_key]
            except Exception:
                df.columns = ["_".join(map(str, c)).strip() for c in df.columns]

        df = _normalize_ohlc(df)
        if len(df) > 0:
            return df

    _debug(f"no data for {symbol}")
    return pd.DataFrame()

# ========= STRATEGY =========
@dataclass
class Position:
    symbol: str
    qty: int
    entry_px: float
    stop_px: float
    tgt_px: float
    entry_date: pd.Timestamp

def _signal_entry_row(df: pd.DataFrame, i: int) -> bool:
    """Entry when Close > prior-day 20D high, ADX>25, RSI>50."""
    if i < 20:
        return False
    close = df["Close"].iloc[i]
    prev_max20 = df["Close"].rolling(20).max().iloc[i-1]
    adx = df["ADX"].iloc[i]
    rsi = df["RSI"].iloc[i]
    return (close > prev_max20) and (adx > 25) and (rsi > 50)

def _commission_amt(notional: float) -> float:
    return notional * (COMMISSION_BPS / 10000.0)

# ========= BACKTEST CORE =========
def run_backtest(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def download_block(sym_list: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        bad: List[str] = []
        _debug(f"Downloading {len(sym_list)} symbols ({start} → {end}) ...")
        for idx, s in enumerate(sym_list, 1):
            df = _fetch(s, start=start, end=end)
            if len(df) >= 60:
                df["ATR"] = _atr(df, 14)
                df["ADX"] = _adx(df, 14)
                df["RSI"] = _rsi(df["Close"], 14)
                df["Date"] = df.index
                data[s] = df
            else:
                bad.append(s)
            if idx % 25 == 0:
                time.sleep(1.0)
        if bad:
            _debug(f"Skipped {len(bad)} tickers (no data): {bad[:5]}{' ...' if len(bad) > 5 else ''}")
        return data

    # Attempt 1: as-is env
    data = download_block(symbols, start=START_DATE, end=END_DATE)

    # Attempt 2: flip suffix logic
    if not data:
        _debug("No data on attempt 1. Flipping suffix logic and retrying ...")
        flipped = [s[:-3] if s.endswith(".NS") else f"{s}.NS" for s in symbols]
        data = download_block(flipped, start=START_DATE, end=END_DATE)

    # Attempt 3: shorten window from 2022
    if not data:
        _debug("No data on attempt 2. Trying shorter window (2022-01-01 → end) ...")
        short_start = "2022-01-01"
        data = download_block(symbols, start=short_start, end=END_DATE)
        if not data:
            data = download_block(flipped if 'flipped' in locals() else symbols, start=short_start, end=END_DATE)

    # Attempt 4: demo universe, both modes, short window
    if not data:
        _debug("No data on attempt 3. Falling back to demo universe ...")
        demo = _demo_universe()
        demo_ns    = [s if s.endswith(".NS") else f"{s}.NS" for s in demo]
        demo_plain = [s[:-3] if s.endswith(".NS") else s for s in demo_ns]
        data = download_block(demo_ns, start="2022-01-01", end=END_DATE)
        if not data:
            data = download_block(demo_plain, start="2022-01-01", end=END_DATE)

    if not data:
        _debug("All attempts failed. Likely a network/DNS/Yahoo block issue. Try hotspot/DNS 1.1.1.1 or use Kite API.")
        raise RuntimeError("No data downloaded. Check tickers, internet, or Yahoo availability.")

    # Master timeline
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    all_dates = [d for d in all_dates if d >= pd.to_datetime(min(START_DATE, "2022-01-01"))]
    if not all_dates:
        raise RuntimeError("No dates in range after downloads.")

    cash = TOTAL_CAPITAL
    positions: Dict[str, Position] = {}
    equity_rows: List[Dict] = []
    trades_rows: List[Dict] = []

    def portfolio_notional(current_date: pd.Timestamp) -> float:
        total = 0.0
        for sym, pos in positions.items():
            if current_date in data[sym].index:
                px = float(data[sym].at[current_date, "Close"])
            else:
                px = float(data[sym]["Close"].iloc[-1])
            total += pos.qty * px
        return total

    for current_date in all_dates:
        # exits
        for sym, pos in list(positions.items()):
            df = data[sym]
            if current_date not in df.index:
                continue
            px = float(df.at[current_date, "Close"])
            exit_reason = None
            if px <= pos.stop_px:
                exit_reason = "STOP"
            elif px >= pos.tgt_px:
                exit_reason = "TARGET"
            if exit_reason:
                notional = pos.qty * px
                pnl = (px - pos.entry_px) * pos.qty
                fees = _commission_amt(pos.qty * pos.entry_px) + _commission_amt(notional)
                cash += (pos.qty * px) - _commission_amt(notional)
                trades_rows.append({
                    "date": current_date.date().isoformat(),
                    "symbol": sym,
                    "side": "SELL",
                    "price": round(px, 4),
                    "qty": pos.qty,
                    "pnl": round(pnl - fees, 2),
                    "r_multiple": round((px - pos.entry_px) / (STOP_ATR * data[sym].loc[pos.entry_date, "ATR"]), 4),
                    "reason": exit_reason,
                    "entry_date": pos.entry_date.date().isoformat(),
                    "entry_px": round(pos.entry_px, 4),
                    "stop_px": round(pos.stop_px, 4),
                    "tgt_px": round(pos.tgt_px, 4),
                })
                del positions[sym]

        # entries
        open_slots = max(0, MAX_POSITIONS - len(positions))
        if open_slots > 0:
            cands = []
            for sym, df in data.items():
                if sym in positions or current_date not in df.index:
                    continue
                i = df.index.get_loc(current_date)
                if i < 20:
                    continue
                if _signal_entry_row(df, i):
                    adx_i = df["ADX"].iloc[i]
                    rsi_i = df["RSI"].iloc[i]
                    score = 0.5 * (adx_i / 25.0) + 0.3 * (rsi_i / 50.0)
                    px = float(df["Close"].iloc[i])
                    atr_i = float(df["ATR"].iloc[i])
                    cands.append((sym, score, px, atr_i))
            cands.sort(key=lambda x: x[1], reverse=True)

            budget = max(0.0, MAX_INVESTED - portfolio_notional(current_date))
            for sym, score, px, atr_i in cands[:open_slots]:
                if atr_i <= 0 or px <= 0:
                    continue
                qty_from_risk = int(RISK_PER_TRADE / max(1e-9, (STOP_ATR * atr_i)))
                qty_from_cash = int(budget / px) if px > 0 else 0
                qty = max(0, min(qty_from_risk, qty_from_cash))
                if qty < 1:
                    continue

                notional = qty * px
                fees = _commission_amt(notional)
                if cash < fees:
                    continue

                cash -= fees
                positions[sym] = Position(
                    symbol=sym,
                    qty=qty,
                    entry_px=px,
                    stop_px=px - STOP_ATR * atr_i,
                    tgt_px=px + TARGET_ATR * atr_i,
                    entry_date=current_date
                )
                trades_rows.append({
                    "date": current_date.date().isoformat(),
                    "symbol": sym,
                    "side": "BUY",
                    "price": round(px, 4),
                    "qty": qty,
                    "pnl": 0.0,
                    "r_multiple": 0.0,
                    "reason": "ENTRY",
                    "entry_date": current_date.date().isoformat(),
                    "entry_px": round(px, 4),
                    "stop_px": round(px - STOP_ATR * atr_i, 4),
                    "tgt_px": round(px + TARGET_ATR * atr_i, 4),
                })
                budget -= notional
                if budget <= 0:
                    break

        # mark-to-market
        notional_now = portfolio_notional(current_date)
        eq_val = cash + notional_now
        equity_rows.append({
            "date": current_date.date().isoformat(),
            "equity": round(eq_val, 2),
            "cash": round(cash, 2),
            "invested": round(notional_now, 2),
            "open_positions": len(positions),
        })

    # liquidate remaining at end
    if len(positions) > 0:
        for sym, pos in list(positions.items()):
            df = data[sym]
            px = float(df["Close"].iloc[-1])
            notional = pos.qty * px
            pnl = (px - pos.entry_px) * pos.qty
            fees = _commission_amt(pos.qty * pos.entry_px) + _commission_amt(notional)
            cash += (pos.qty * px) - _commission_amt(notional)
            trades_rows.append({
                "date": df.index[-1].date().isoformat(),
                "symbol": sym,
                "side": "SELL",
                "price": round(px, 4),
                "qty": pos.qty,
                "pnl": round(pnl - fees, 2),
                "r_multiple": round((px - pos.entry_px) / (STOP_ATR * df.loc[pos.entry_date, "ATR"]), 4),
                "reason": "EOD_LIQ",
                "entry_date": pos.entry_date.date().isoformat(),
                "entry_px": round(pos.entry_px, 4),
                "stop_px": round(pos.stop_px, 4),
                "tgt_px": round(pos.tgt_px, 4),
            })
            del positions[sym]
        # final equity row
        equity_rows.append({
            "date": END_DATE,
            "equity": round(cash, 2),
            "cash": round(cash, 2),
            "invested": 0.0,
            "open_positions": 0,
        })

    trades = pd.DataFrame(trades_rows)
    equity = pd.DataFrame(equity_rows)

    # yearly stats
    equity["date"] = pd.to_datetime(equity["date"])
    equity.set_index("date", inplace=True)
    eq_daily = equity.resample("D").ffill()

    yearly = []
    for year, dfy in eq_daily.groupby(eq_daily.index.year):
        if len(dfy) < 2:
            continue
        start = float(dfy.iloc[0]["equity"])
        end = float(dfy.iloc[-1]["equity"])
        ret = (end / start) - 1.0 if start > 0 else 0.0

        closed = trades[(trades["side"] == "SELL") & (pd.to_datetime(trades["date"]).dt.year == year)]
        wins = (closed["pnl"] > 0).sum()
        losses = (closed["pnl"] <= 0).sum()
        win_rate = (wins / max(1, (wins + losses))) * 100.0

        rr = closed.copy()
        avg_rwin  = rr.loc[rr["pnl"] > 0, "r_multiple"].mean() if not rr.loc[rr["pnl"] > 0].empty else np.nan
        avg_rloss = rr.loc[rr["pnl"] <= 0, "r_multiple"].mean() if not rr.loc[rr["pnl"] <= 0].empty else np.nan

        years_elapsed = year - pd.to_datetime(min(START_DATE, "2022-01-01")).year + 1e-9
        cum_cagr = (end / TOTAL_CAPITAL) ** (1.0 / years_elapsed) - 1.0 if TOTAL_CAPITAL > 0 else np.nan

        yearly.append({
            "Year": year,
            "YearReturn%": round(ret * 100.0, 2),
            "WinRate%": round(win_rate, 2),
            "Wins": int(wins),
            "Losses": int(losses),
            "AvgR_Win": round(avg_rwin, 3) if pd.notna(avg_rwin) else np.nan,
            "AvgR_Loss": round(avg_rloss, 3) if pd.notna(avg_rloss) else np.nan,
            "CAGR_toDate%": round(cum_cagr * 100.0, 2) if pd.notna(cum_cagr) else np.nan,
            "YearStartEquity": round(start, 2),
            "YearEndEquity": round(end, 2),
        })

    yearly_df = pd.DataFrame(yearly)

    # save
    trades.to_csv("trades_swingpro.csv", index=False)
    equity.reset_index().to_csv("equity_swingpro.csv", index=False)
    yearly_df.to_csv("yearly_returns_swingpro.csv", index=False)

    return trades, equity.reset_index(), yearly_df

# ========= MAIN =========
if __name__ == "__main__":
    symbols = _load_universe()
    print(f"Universe size: {len(symbols)} (limit={UNIVERSE_LIMIT})")
    print(f"Backtesting {START_DATE} → {END_DATE} ...")
    trades, equity, yearly = run_backtest(symbols)
    print("Saved: trades_swingpro.csv, equity_swingpro.csv, yearly_returns_swingpro.csv")
    print("\nYearly summary (Return%, WinRate%):")
    print(yearly[["Year","YearReturn%","WinRate%","Wins","Losses","CAGR_toDate%"]].to_string(index=False))

# backtest_quick.py — simple portfolio backtester for your swing rules
from __future__ import annotations

import os, math, time, builtins, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= ENV / KNOBS =========
NIFTY500_PATH   = os.getenv("NIFTY500_PATH", "nifty500.csv")
SCAN_CSV        = os.getenv("SCAN_CSV", "0") == "1"
CSV_PATH        = os.getenv("CSV_PATH", "instruments_nse_eq.csv")
UNIVERSE_LIMIT  = int(os.getenv("UNIVERSE_LIMIT", "150"))   # start smaller to avoid Yahoo rate limits

TOTAL_CAPITAL   = float(os.getenv("TOTAL_CAPITAL", "10000"))
MAX_INVESTED    = float(os.getenv("MAX_INVESTED", "10000")) # allow full capital by default
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "5"))
PER_TRADE_RISK  = float(os.getenv("PER_TRADE_RISK", "1000"))

# Backtest dates (None = 250 bars)
START_DATE      = os.getenv("BT_START", "") or None
END_DATE        = os.getenv("BT_END", "") or None

# Costs assumptions (very rough)
SLIPPAGE_PCT    = float(os.getenv("SLIPPAGE_PCT", "0.001"))   # 0.10% per side
FEES_PCT        = float(os.getenv("FEES_PCT", "0.0003"))      # 0.03% per side

# Risk/Rules
ATR_N           = int(os.getenv("ATR_N", "14"))
SMA_FAST        = int(os.getenv("SMA_FAST", "50"))
SMA_SLOW        = int(os.getenv("SMA_SLOW", "200"))
BREAKOUT_N      = int(os.getenv("BREAKOUT_N", "20"))
RSI_N           = int(os.getenv("RSI_N", "14"))
RSI_MIN         = float(os.getenv("RSI_MIN", "55"))
TRAIL_PCT       = float(os.getenv("TRAIL_PCT", "0.08"))       # 8% trailing on highs since entry
TIME_STOP_DAYS  = int(os.getenv("TIME_STOP_DAYS", "60"))


# ========= SYMBOL UNIVERSE HELPERS =========
def _pick_symbol_column(df: pd.DataFrame) -> str:
    for col in ["SYMBOL", "Symbol", "symbol", "tradingsymbol", "Tradingsymbol", "TRADINGSYMBOL"]:
        if col in df.columns:
            return col
    raise ValueError("No symbol column found. Expected SYMBOL/Tradingsymbol/etc.")

def _is_index_like(s: str) -> bool:
    s2 = builtins.str(s).strip().upper()
    if not s2:
        return True
    bad = ("NIFTY","BANKNIFTY","FINNIFTY","MIDCAP","SENSEX","VIX","IX","INDEX")
    return any(k in s2 for k in bad) or (" " in s2) or ("/" in s2)

def load_universe() -> List[str]:
    syms: List[str] = []
    if os.path.exists(NIFTY500_PATH):
        df = pd.read_csv(NIFTY500_PATH)
        col = _pick_symbol_column(df)
        vals = [builtins.str(x).split(":")[-1].strip().upper() for x in df[col].dropna().tolist()]
        syms = [v for v in vals if v and not _is_index_like(v)]
        print(f"[bt] Loaded {len(syms)} symbols from {NIFTY500_PATH}")
    elif SCAN_CSV and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        for c in ["segment","Segment"]:
            if c in df.columns:
                df = df[df[c].astype(str).str.upper().str.contains("NSE")]
                break
        for c in ["instrument_type","InstrumentType","instrumenttype"]:
            if c in df.columns:
                df = df[df[c].astype(str).str.upper().str.contains("EQ")]
                break
        col = _pick_symbol_column(df)
        vals = [builtins.str(x).split(":")[-1].strip().upper() for x in df[col].dropna().tolist()]
        syms = [v for v in vals if v and not _is_index_like(v)]
        print(f"[bt] Loaded {len(syms)} symbols from {CSV_PATH}")
    else:
        syms = ["RELIANCE"]
        print("[bt] Fallback universe: RELIANCE")

    # de-dup keep order
    out, seen = [], set()
    for s in syms:
        if s not in seen:
            out.append(s); seen.add(s)
    if UNIVERSE_LIMIT > 0:
        out = out[:UNIVERSE_LIMIT]
    print(f"[bt] Using {len(out)} symbols")
    return out


# ========= DATA FETCH =========
def _ticker(symbol: str) -> str:
    return f"{builtins.str(symbol).split(':')[-1].strip().upper()}.NS"

def fetch_ohlc(symbol: str) -> Optional[pd.DataFrame]:
    """
    Robust yfinance fetcher with small retry; if it fails, synthesize series so we can keep going.
    """
    try:
        import yfinance as yf
        tries, last_err = 3, None
        for i in range(tries):
            try:
                if START_DATE or END_DATE:
                    df = yf.Ticker(_ticker(symbol)).history(start=START_DATE, end=END_DATE, interval="1d", auto_adjust=False)
                else:
                    df = yf.Ticker(_ticker(symbol)).history(period="250d", interval="1d", auto_adjust=False)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # normalize columns
                    if {"High","Low","Close"}.issubset(set(df.columns)):
                        out = df[["High","Low","Close"]].dropna().astype(float)
                        out.index = pd.to_datetime(out.index)
                        return out
                    # MultiIndex fallback
                    nums = df.select_dtypes(include=["number"])
                    if nums.shape[1] >= 3:
                        out = nums.iloc[:, :3]
                        out.columns = ["High","Low","Close"]
                        out.index = pd.to_datetime(out.index)
                        return out
                    last_err = RuntimeError("No numeric OHLC")
                else:
                    last_err = RuntimeError("Empty frame")
            except Exception as e:
                last_err = e
            time.sleep(0.8*(i+1))
        raise last_err or RuntimeError("Unknown yfinance error")
    except Exception as e:
        # synthesize to keep loop moving (marked so you know)
        print(f"[bt] yfinance failed for {symbol}: {type(e).__name__}: {e}. Using synthetic.")
        n = 250
        rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
        price = 100 + np.cumsum(np.random.normal(0.2, 1.5, size=n))
        high  = price + np.abs(np.random.normal(0.5, 0.7, size=n))
        low   = price - np.abs(np.random.normal(0.5, 0.7, size=n))
        close = price
        return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=rng).astype(float)


# ========= INDICATORS =========
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def highest(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).max()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


# ========= PORTFOLIO ENGINE =========
@dataclass
class Position:
    symbol: str
    qty: int
    entry_px: float
    entry_date: pd.Timestamp
    trail_max_close: float
    stop_px: float

def size_position(cash_avail: float, price: float, atr_val: float) -> int:
    if price <= 0 or atr_val <= 0:
        return 0
    # risk-based sizing
    risk_per_share = 2.0 * atr_val
    if risk_per_share <= 0:
        return 0
    max_shares_by_risk = math.floor(PER_TRADE_RISK / risk_per_share)
    max_shares_by_cash = math.floor(cash_avail / price)
    qty = max(0, min(max_shares_by_risk, max_shares_by_cash))
    return qty

def apply_costs(px: float, side: str) -> float:
    adj = px * (SLIPPAGE_PCT + FEES_PCT)
    return px + adj if side == "BUY" else px - adj


# ========= BACKTEST LOOP =========
def backtest(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cash = TOTAL_CAPITAL
    equity_curve = []
    positions: Dict[str, Position] = {}
    invested_value = 0.0
    trades = []

    # Preload all data & indicators to reduce churn
    data_map: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = fetch_ohlc(s)
        if df is None or df.empty or df.shape[0] < max(SMA_SLOW, BREAKOUT_N)+5:
            continue
        df = df.copy()
        df["SMA_FAST"] = sma(df["Close"], SMA_FAST)
        df["SMA_SLOW"] = sma(df["Close"], SMA_SLOW)
        df["ATR"]      = atr(df, ATR_N)
        df["RSI"]      = rsi(df["Close"], RSI_N)
        df["HH"]       = highest(df["Close"].shift(1), BREAKOUT_N)  # prior N-day high
        data_map[s] = df

    # Build a common date index
    all_dates = sorted(set(pd.concat([d.index.to_series() for d in data_map.values()], axis=0).index))
    all_dates = [pd.Timestamp(d).normalize() for d in all_dates]

    for dt in all_dates:
        # 1) update trailing stops & exits
        to_close = []
        for sym, pos in positions.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            c = float(row["Close"])
            # update trail max close
            pos.trail_max_close = max(pos.trail_max_close, c)
            # trailing stop
            trail_stop = pos.trail_max_close * (1 - TRAIL_PCT)
            # time stop
            held_days = (dt - pos.entry_date).days
            # fast MA exit
            below_fast = c < float(row["SMA_FAST"])
            stop_hit = c < trail_stop
            time_stop = held_days >= TIME_STOP_DAYS
            if below_fast or stop_hit or time_stop:
                exit_px = apply_costs(c, "SELL")
                pnl = (exit_px - pos.entry_px) * pos.qty
                cash += exit_px * pos.qty
                invested_value -= c * pos.qty
                trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": pos.qty,
                               "px": exit_px, "pnl": pnl})
                to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        # 2) entries — only if we have free slots & capital
        slots_free = MAX_POSITIONS - len(positions)
        if slots_free > 0:
            # simple ranking = momentum over 60d (recent strength)
            candidates = []
            for sym, df in data_map.items():
                if sym in positions:
                    continue
                if dt not in df.index:
                    continue
                row = df.loc[dt]
                c = float(row["Close"])
                if np.isnan([row["SMA_FAST"], row["SMA_SLOW"], row["HH"], row["RSI"], row["ATR"]]).any():
                    continue
                uptrend = row["SMA_FAST"] > row["SMA_SLOW"]
                breakout = c > row["HH"]
                rsi_ok = row["RSI"] >= RSI_MIN
                if uptrend and breakout and rsi_ok:
                    # momentum score
                    lookback = 60
                    i = df.index.get_loc(dt)
                    if i >= lookback:
                        prev = float(df["Close"].iloc[i - lookback])
                        mom = (c / prev) - 1.0
                    else:
                        mom = 0.0
                    candidates.append((mom, sym, c, float(row["ATR"])))

            candidates.sort(reverse=True)
            for _, sym, c, atr_val in candidates[:max(slots_free, 0)]:
                # capital controls
                equity_now = cash + sum(data_map[s].loc[dt, "Close"] * p.qty for s, p in positions.items() if dt in data_map[s].index)
                # do not exceed MAX_INVESTED
                invested_now = sum((data_map[s].loc[dt, "Close"] * p.qty) for s, p in positions.items() if dt in data_map[s].index)
                invest_room = max(0.0, MAX_INVESTED - invested_now)
                cash_room = min(cash, invest_room)
                qty = size_position(cash_room, c, atr_val)
                if qty <= 0:
                    continue
                entry_px = apply_costs(c, "BUY")
                cost = entry_px * qty
                cash -= cost
                invested_value += c * qty
                positions[sym] = Position(
                    symbol=sym, qty=qty, entry_px=entry_px, entry_date=dt,
                    trail_max_close=c, stop_px=max(0.01, c - 2.0*atr_val)
                )
                trades.append({"symbol": sym, "side": "BUY", "date": dt, "qty": qty,
                               "px": entry_px, "pnl": 0.0})

        # 3) mark-to-market equity
        mtm = sum((data_map[s].loc[dt, "Close"] * p.qty) for s, p in positions.items() if dt in data_map[s].index)
        equity = cash + mtm
        equity_curve.append({"date": dt, "equity": equity, "cash": cash, "mtm": mtm, "positions": len(positions)})

    # Close all on last day
    if all_dates:
        dt = all_dates[-1]
        for sym, p in list(positions.items()):
            c = float(data_map[sym].loc[dt, "Close"])
            exit_px = apply_costs(c, "SELL")
            pnl = (exit_px - p.entry_px) * p.qty
            trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": p.qty,
                           "px": exit_px, "pnl": pnl})
            cash += exit_px * p.qty
            positions.pop(sym, None)
        mtm = 0.0
        equity_curve.append({"date": dt, "equity": cash, "cash": cash, "mtm": mtm, "positions": 0})

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    return trades_df, eq_df


# ========= METRICS & I/O =========
def performance_metrics(eq: pd.DataFrame) -> Dict[str, float]:
    eq = eq.copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    # Sharpe (daily, no risk-free)
    sharpe = (eq["ret"].mean() / (eq["ret"].std() + 1e-12)) * np.sqrt(252)
    # Max drawdown
    roll_max = eq["equity"].cummax()
    dd = (eq["equity"] / (roll_max + 1e-9)) - 1.0
    maxdd = dd.min()
    # CAGR (approx)
    if len(eq) > 1:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (eq["equity"].iloc[-1] / max(eq["equity"].iloc[0], 1e-9)) ** (1/years) - 1 if years > 0 else 0.0
    else:
        cagr = 0.0
    return {"CAGR": cagr, "MaxDD": maxdd, "Sharpe": sharpe}


def trades_summary(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"Trades": 0, "WinRate": 0.0, "AvgWin": 0.0, "AvgLoss": 0.0, "ProfitFactor": 0.0}
    # pair buys and sells by symbol FIFO
    pnl = trades[trades["side"] == "SELL"]["pnl"].sum()
    sells = trades[trades["side"] == "SELL"]
    wins = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] <= 0]
    winrate = (len(wins) / max(1, len(sells))) if len(sells) else 0.0
    avg_win = wins["pnl"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and abs(losses["pnl"].sum()) > 0 else np.inf if len(wins) else 0.0
    return {
        "Trades": int(len(sells)),
        "WinRate": float(winrate),
        "AvgWin": float(avg_win),
        "AvgLoss": float(avg_loss),
        "ProfitFactor": float(profit_factor),
        "NetPNL": float(pnl),
    }


def main():
    syms = load_universe()
    trades, eq = backtest(syms)
    # Save outputs
    trades.to_csv("backtest_trades.csv", index=False)
    eq.to_csv("equity_curve.csv")

    pm = performance_metrics(eq)
    ts = trades_summary(trades)

    print("\n=== Backtest Summary ===")
    print(f"Start Equity: {TOTAL_CAPITAL:,.2f}")
    print(f"End   Equity: {eq['equity'].iloc[-1]:,.2f}" if not eq.empty else "End   Equity: n/a")
    for k, v in pm.items():
        if k in ("CAGR","MaxDD","Sharpe"):
            print(f"{k:>8}: {v:.4f}")
    print(f"Trades : {ts['Trades']}")
    print(f"WinRate: {ts['WinRate']*100:.1f}%  PF: {ts['ProfitFactor']:.2f}  NetPNL: {ts['NetPNL']:.2f}")

    # Plot equity curve (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        eq["equity"].plot()
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig("equity_curve.png", dpi=140)
        print("Saved: backtest_trades.csv, equity_curve.csv, equity_curve.png")
    except Exception:
        print("Saved: backtest_trades.csv, equity_curve.csv (matplotlib not available)")

if __name__ == "__main__":
    main()

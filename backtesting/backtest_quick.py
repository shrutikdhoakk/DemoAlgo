from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
# ensure the repo root (parent of /backtesting) is importable
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
print("[dbg] swing_strategy import OK")

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
print("[dbg] repo root added to sys.path")



import numpy as np
import pandas as pd

from swing_strategy import (
    _normalize_ohlc,
    _compute_rsi,
    _compute_atr,
    _compute_adx,
    _last_scalar,
)



# ----------------- data helpers -----------------

def fetch_ohlc(symbol: str) -> pd.DataFrame:
    """Fetch OHLCV data for backtest. Falls back to synthetic series."""
    try:
        import yfinance as yf
        df_raw = yf.download(
            f"{symbol}.NS",
            period="500d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if isinstance(df_raw, pd.DataFrame) and len(df_raw):
            return _normalize_ohlc(df_raw)
        raise RuntimeError("empty")
    except Exception:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=500, freq="B")
        base = np.linspace(100.0, 120.0, len(idx)) + np.random.normal(0, 0.8, len(idx))
        high = base * (1.004 + np.random.normal(0, 0.0008, len(idx)))
        low = base * (0.996 + np.random.normal(0, 0.0008, len(idx)))
        close = base
        volume = np.random.normal(1e6, 2e5, len(idx)).clip(min=0)
        return pd.DataFrame(
            {"High": high, "Low": low, "Close": close, "Volume": volume},
            index=idx,
        ).astype(float)


def fetch_index() -> pd.DataFrame:
    """Fetch index data for risk-on regime filter."""
    try:
        import yfinance as yf
        df_raw = yf.download(
            "^NSEI",
            period="500d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if isinstance(df_raw, pd.DataFrame) and len(df_raw):
            return _normalize_ohlc(df_raw)
        raise RuntimeError("empty")
    except Exception:
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=500, freq="B")
        trend = np.linspace(18000.0, 22000.0, len(idx)) + np.random.normal(0, 15.0, len(idx))
        high = trend * (1.003 + np.random.normal(0, 0.0005, len(idx)))
        low = trend * (0.997 + np.random.normal(0, 0.0005, len(idx)))
        close = trend
        return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx).astype(float)


# ----------------- backtest engine -----------------

@dataclass
class Position:
    qty: int
    stop_price: float
    entry_px: float


def backtest(
    symbols: List[str],
    max_positions: int = 5,
    total_capital: float = 10000.0,
    max_invested: float = 5000.0,
    max_risk_per_trade: float = 1000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run vectorised backtest of SwingStrategy rules."""

    data: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = fetch_ohlc(s)
        if df is None or df.empty or len(df) < 60:
            continue
        close = df["Close"]
        df["RSI7"] = _compute_rsi(close, 7)
        df["RSI14"] = _compute_rsi(close, 14)
        df["RSI21"] = _compute_rsi(close, 21)
        df["ATR"] = _compute_atr(df, 14)
        df["ADX"] = _compute_adx(df, 14)
        df["RH20"] = close.rolling(20, min_periods=20).max()
        df["RH50"] = close.rolling(50, min_periods=50).max()
        df["AVG_VOL20"] = df["Volume"].rolling(20, min_periods=20).mean()
        data[s] = df

    index_df = fetch_index()
    index_df["EMA50"] = index_df["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    index_df["EMA200"] = index_df["Close"].ewm(span=200, adjust=False, min_periods=200).mean()

    all_dates = sorted(set(pd.concat([df.index.to_series() for df in data.values()], axis=0)))

    cash = total_capital
    positions: Dict[str, Position] = {}
    trades = []
    equity_curve = []

    for dt in all_dates:
        # update exits
        to_close = []
        for sym, pos in positions.items():
            df = data.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            c = float(row["Close"])
            atr_val = float(row["ATR"])
            pos.stop_price = max(pos.stop_price, c - 2.0 * atr_val)
            if c <= pos.stop_price or c < float(row.get("Close", c)):
                pnl = (c - pos.entry_px) * pos.qty
                cash += c * pos.qty
                trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": pos.qty, "px": c, "pnl": pnl})
                to_close.append(sym)
        for sym in to_close:
            positions.pop(sym, None)

        # risk-on check
        risk_on = True
        if dt in index_df.index:
            row = index_df.loc[dt]
            a = float(row["EMA50"]); b = float(row["EMA200"])
            if not np.isnan(a) and not np.isnan(b):
                risk_on = a > b
        if not risk_on:
            for sym, pos in list(positions.items()):
                df = data.get(sym)
                if df is None or dt not in df.index:
                    continue
                c = float(df.loc[dt, "Close"])
                pnl = (c - pos.entry_px) * pos.qty
                cash += c * pos.qty
                trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": pos.qty, "px": c, "pnl": pnl})
                positions.pop(sym)
            equity_curve.append({"date": dt, "equity": cash, "cash": cash, "mtm": 0.0, "positions": 0})
            continue

        # entries
        slots = max_positions - len(positions)
        if slots > 0:
            invested = sum(data[sym].loc[dt, "Close"] * pos.qty for sym, pos in positions.items() if dt in data[sym].index)
            remaining_budget = max(0.0, max_invested - invested)
            per_slot_budget = max_invested / float(max_positions)
            candidates = []
            for sym, df in data.items():
                if sym in positions or dt not in df.index:
                    continue
                row = df.loc[dt]
                c = float(row["Close"])
                atr_val = float(row["ATR"])
                adx = float(row["ADX"])
                r14 = float(row["RSI14"])
                r7 = float(row["RSI7"])
                r21 = float(row["RSI21"])
                rh20 = float(row["RH20"])
                rh50 = float(row["RH50"])
                v = float(row["Volume"])
                vavg = float(row["AVG_VOL20"])
                if any(np.isnan(x) for x in (c, atr_val, adx, r14, r7, r21, rh20, rh50, v, vavg)):
                    continue
                if v <= 1.5 * vavg:
                    continue
                cond_breakout = (c > rh20) or (c > rh50)
                cond_rsi = (r14 > 40.0) and (r7 > 40.0) and (r21 > 40.0)
                cond_adx = adx > 20.0
                if not (cond_breakout and cond_rsi and cond_adx):
                    continue
                breakout_ratio = max(c / max(rh20, 1e-9), c / max(rh50, 1e-9))
                score = 0.6 * breakout_ratio + 0.3 * (adx / 25.0) + 0.1 * (r14 / 50.0)
                candidates.append((score, sym, c, atr_val))
            candidates.sort(reverse=True)
            for score, sym, price, atr_val in candidates:
                if slots <= 0 or remaining_budget <= 0:
                    break
                budget_for_this = min(per_slot_budget, remaining_budget)
                qty_cash = int(budget_for_this // max(price, 1e-9))
                qty_risk = int(max(1.0, max_risk_per_trade / max(atr_val, 1e-9)))
                qty = max(1, min(qty_cash, qty_risk))
                notional = qty * price
                if notional > remaining_budget + 1e-6:
                    qty = int(remaining_budget // max(price, 1e-9))
                    notional = qty * price
                if qty < 1:
                    continue
                cash -= notional
                positions[sym] = Position(qty=qty, stop_price=price - 2.0 * atr_val, entry_px=price)
                trades.append({"symbol": sym, "side": "BUY", "date": dt, "qty": qty, "px": price, "pnl": 0.0})
                remaining_budget -= notional
                slots -= 1

        # mark-to-market
        mtm = sum(data[sym].loc[dt, "Close"] * pos.qty for sym, pos in positions.items() if dt in data[sym].index)
        equity = cash + mtm
        equity_curve.append({"date": dt, "equity": equity, "cash": cash, "mtm": mtm, "positions": len(positions)})

    # Close remaining positions
    if all_dates:
        dt = all_dates[-1]
        for sym, pos in list(positions.items()):
            df = data.get(sym)
            if df is None or dt not in df.index:
                continue
            c = float(df.loc[dt, "Close"])
            pnl = (c - pos.entry_px) * pos.qty
            cash += c * pos.qty
            trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": pos.qty, "px": c, "pnl": pnl})
            positions.pop(sym)
        equity_curve.append({"date": dt, "equity": cash, "cash": cash, "mtm": 0.0, "positions": 0})

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    return trades_df, eq_df


if __name__ == "__main__":
    symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "TITAN"]
    trades, eq = backtest(symbols)
    trades.to_csv("backtest_trades.csv", index=False)
    eq.to_csv("equity_curve.csv")
    print(eq.tail())
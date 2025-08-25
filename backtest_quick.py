# backtest_quick.py — portfolio backtester aligned with SwingStrategy rules
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
MAX_INVESTED    = float(os.getenv("MAX_INVESTED", "5000"))
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "5"))
PER_TRADE_RISK  = float(os.getenv("PER_TRADE_RISK", "1000"))

EVAL_STEP_DAYS  = int(os.getenv("EVAL_STEP_DAYS", "1"))     # evaluate signals daily

# Backtest dates (None = last 250 bars)
START_DATE      = os.getenv("BT_START", "") or None
END_DATE        = os.getenv("BT_END", "") or None

# Costs
SLIPPAGE_PCT    = float(os.getenv("SLIPPAGE_PCT", "0.001"))   # 0.10% per side
FEES_PCT        = float(os.getenv("FEES_PCT", "0.0003"))      # 0.03% per side

# Indicator params (must mirror strategy)
ATR_N           = int(os.getenv("ATR_N", "14"))
ADX_N           = int(os.getenv("ADX_N", "14"))
BREAKOUT_1      = int(os.getenv("BREAKOUT_1", "20"))
BREAKOUT_2      = int(os.getenv("BREAKOUT_2", "50"))
RSI_A           = int(os.getenv("RSI_A", "7"))
RSI_B           = int(os.getenv("RSI_B", "14"))
RSI_C           = int(os.getenv("RSI_C", "21"))
ADX_MIN         = float(os.getenv("ADX_MIN", "20"))
RSI_MIN_ALL     = float(os.getenv("RSI_MIN_ALL", "40"))
VOL_MULT        = float(os.getenv("VOL_MULT", "1.5"))         # current vol >= VOL_MULT * avg20

# Exit helpers
RSI_EXIT_BELOW  = float(os.getenv("RSI_EXIT_BELOW", "30"))
SMA_EXIT_N      = int(os.getenv("SMA_EXIT_N", "20"))
ATR_STOP_MULT   = float(os.getenv("ATR_STOP_MULT", "2.0"))
ATR_TGT_MULT    = float(os.getenv("ATR_TGT_MULT", "2.0"))

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
    yfinance with retry + synthetic fallback (High/Low/Close/Volume).
    """
    try:
        import yfinance as yf
        tries, last_err = 3, None
        for i in range(tries):
            try:
                if START_DATE or END_DATE:
                    df = yf.Ticker(_ticker(symbol)).history(
                        start=START_DATE, end=END_DATE, interval="1d", auto_adjust=False
                    )
                else:
                    df = yf.Ticker(_ticker(symbol)).history(period="250d", interval="1d", auto_adjust=False)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out = pd.DataFrame(index=pd.to_datetime(df.index))
                    # Pull columns robustly
                    for name in ["High","Low","Close","Volume"]:
                        if name in df.columns:
                            out[name] = df[name].astype(float)
                        else:
                            # multiindex or weird columns
                            num = df.select_dtypes(include=["number"])
                            if num.shape[1]:
                                out[name] = num.iloc[:, 0].astype(float)
                            else:
                                raise RuntimeError("No numeric columns")
                    return out.dropna()
                last_err = RuntimeError("Empty frame")
            except Exception as e:
                last_err = e
            time.sleep(0.8*(i+1))
        raise last_err or RuntimeError("Unknown yfinance error")
    except Exception as e:
        # synthesize to keep loop moving
        print(f"[bt] yfinance failed for {symbol}: {type(e).__name__}: {e}. Using synthetic.")
        n = 250
        rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
        price = 100 + np.cumsum(np.random.normal(0.2, 1.5, size=n))
        high  = price + np.abs(np.random.normal(0.5, 0.7, size=n))
        low   = price - np.abs(np.random.normal(0.5, 0.7, size=n))
        close = price
        vol   = np.random.uniform(1e5, 1e6, size=n)
        return pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": vol}, index=rng).astype(float)

# ========= INDICATORS (match SwingStrategy) =========
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").astype(float)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(100.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    df = df.copy()
    h, l, c = df["High"], df["Low"], df["Close"]

    plus_dm  = h.diff().clip(lower=0.0)
    minus_dm = (-l.diff()).clip(lower=0.0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0.0
    minus_dm[~mask] = 0.0

    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atrv = tr.rolling(n, min_periods=n).mean().replace(0.0, np.nan)

    plus_di  = 100.0 * (plus_dm.rolling(n, min_periods=n).sum() / atrv)
    minus_di = 100.0 * (minus_dm.rolling(n, min_periods=n).sum() / atrv)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di) ) * 100.0
    return dx.rolling(n, min_periods=n).mean()

def highest(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).max()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def apply_costs(px: float, side: str) -> float:
    adj = px * (SLIPPAGE_PCT + FEES_PCT)
    return px + adj if side == "BUY" else px - adj

# ========= PORTFOLIO ENGINE =========
@dataclass
class Position:
    symbol: str
    qty: int
    entry_px: float
    stop_px: float
    target_px: float
    last_close: float

def size_position(cash_avail: float, price: float, atr_val: float) -> int:
    if price <= 0 or atr_val <= 0 or cash_avail <= 0:
        return 0
    risk_per_share = ATR_STOP_MULT * atr_val
    if risk_per_share <= 0:
        return 0
    by_risk = int(max(1.0, PER_TRADE_RISK / max(risk_per_share, 1e-9)))
    by_cash = int(cash_avail // max(price, 1e-9))
    return max(0, min(by_risk, by_cash))

# ========= BACKTEST LOOP =========
def backtest(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cash = TOTAL_CAPITAL
    positions: Dict[str, Position] = {}
    trades = []
    equity_curve = []

    # Preload data & indicators
    data_map: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = fetch_ohlc(s)
        if df is None or df.empty or df.shape[0] < max(ATR_N, BREAKOUT_2, ADX_N, RSI_B)+5:
            continue
        df = df.copy()
        df["ATR"] = atr(df, ATR_N)
        df["ADX"] = adx(df, ADX_N)
        df["RSI7"]  = rsi(df["Close"], RSI_A)
        df["RSI14"] = rsi(df["Close"], RSI_B)
        df["RSI21"] = rsi(df["Close"], RSI_C)
        df["HH20"]  = highest(df["Close"].shift(1), BREAKOUT_1)  # prior high
        df["HH50"]  = highest(df["Close"].shift(1), BREAKOUT_2)
        df["VOLAVG20"] = df["Volume"].rolling(20, min_periods=20).mean()
        df["SMA20"] = sma(df["Close"], SMA_EXIT_N)
        data_map[s] = df

    if not data_map:
        return pd.DataFrame(), pd.DataFrame()

    # unified calendar
    all_dates = sorted(set(pd.concat([d.index.to_series() for d in data_map.values()], axis=0).index))
    all_dates = [pd.Timestamp(d).normalize() for d in all_dates]

    for di, dt in enumerate(all_dates):
        # evaluate only at step (daily by default)
        if di % max(1, EVAL_STEP_DAYS) != 0:
            # MTM only
            mtm = sum((data_map[s].loc[dt, "Close"] * p.qty) for s, p in positions.items() if dt in data_map[s].index)
            equity = cash + mtm
            equity_curve.append({"date": dt, "equity": equity, "cash": cash, "mtm": mtm, "positions": len(positions)})
            continue

        # 1) exits & trailing updates
        to_close = []
        for sym, pos in positions.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            c = float(row["Close"])
            a14 = float(row["ATR"])
            # update last_close
            pos.last_close = c
            # trail both stop & target with ATR
            new_stop   = c - ATR_STOP_MULT * a14
            new_target = c + ATR_TGT_MULT  * a14
            if new_stop > pos.stop_px:
                pos.stop_px = new_stop
            if new_target > pos.target_px:
                pos.target_px = new_target

            # exit conditions
            rsi14 = float(row["RSI14"])
            sma20 = float(row["SMA20"])
            stop_hit   = c <= pos.stop_px
            target_hit = c >= pos.target_px
            sma_break  = c < sma20 if not np.isnan(sma20) else False
            rsi_break  = rsi14 < RSI_EXIT_BELOW if not np.isnan(rsi14) else False

            if stop_hit or target_hit or sma_break or rsi_break:
                exit_px = apply_costs(c, "SELL")
                pnl = (exit_px - pos.entry_px) * pos.qty
                cash += exit_px * pos.qty
                trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": pos.qty, "px": exit_px, "pnl": pnl})
                to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        # 2) entries (respect caps & notional)
        open_slots = max(0, MAX_POSITIONS - len(positions))
        if open_slots > 0:
            invested_now = sum((data_map[s].loc[dt, "Close"] * p.qty) for s, p in positions.items() if dt in data_map[s].index)
            remaining_budget = max(0.0, MAX_INVESTED - invested_now)
            if remaining_budget > 0 and cash > 0:
                # build candidates
                cands = []
                for sym, df in data_map.items():
                    if sym in positions or dt not in df.index:
                        continue
                    row = df.loc[dt]
                    c   = float(row["Close"])
                    a14 = float(row["ATR"])
                    adxv= float(row["ADX"])
                    r7, r14, r21 = float(row["RSI7"]), float(row["RSI14"]), float(row["RSI21"])
                    hh20, hh50   = float(row["HH20"]), float(row["HH50"])
                    v, vavg      = float(row["Volume"]), float(row["VOLAVG20"])

                    if np.isnan([c,a14,adxv,r7,r14,r21,hh20,hh50,v,vavg]).any():
                        continue
                    if v < VOL_MULT * vavg:
                        continue
                    # breakout conditions
                    tol = 1.0  # align with live: set 0.995 if EASY_SIM=1
                    breakout = (c > tol * hh20) or (c > tol * hh50)
                    rsi_ok   = (r7 > RSI_MIN_ALL) and (r14 > RSI_MIN_ALL) and (r21 > RSI_MIN_ALL)
                    adx_ok   = adxv > ADX_MIN
                    if not (breakout and rsi_ok and adx_ok):
                        continue

                    # score
                    eps = 1e-9
                    bratio = max(c / max(hh20, eps), c / max(hh50, eps))
                    score = 0.6 * bratio + 0.3 * (adxv / 25.0) + 0.1 * (r14 / 50.0)

                    cands.append((score, sym, c, a14))

                cands.sort(reverse=True)
                selected = cands[:open_slots]
                total_score = sum(max(0.0, s[0]) for s in selected) or 1.0

                for score, sym, price, a14 in selected:
                    if remaining_budget <= 0 or cash <= 0:
                        break
                    # budget proportional to score
                    budget_for_this = min(remaining_budget * (score / total_score), remaining_budget, cash)
                    qty_cash = int(budget_for_this // max(price, 1e-9))
                    qty_risk = int(max(1.0, PER_TRADE_RISK / max(ATR_STOP_MULT * a14, 1e-9)))
                    qty = max(1, min(qty_cash, qty_risk))
                    notional = qty * price
                    if qty < 1 or notional <= 0:
                        continue
                    if notional > remaining_budget + 1e-6:
                        qty = int(remaining_budget // max(price, 1e-9))
                        if qty < 1:
                            continue
                        notional = qty * price

                    entry_px = apply_costs(price, "BUY")
                    cost = entry_px * qty
                    if cost > cash + 1e-6:
                        continue

                    positions[sym] = Position(
                        symbol=sym,
                        qty=qty,
                        entry_px=entry_px,
                        stop_px=price - ATR_STOP_MULT * a14,
                        target_px=price + ATR_TGT_MULT  * a14,
                        last_close=price,
                    )
                    cash -= cost
                    trades.append({"symbol": sym, "side": "BUY", "date": dt, "qty": qty, "px": entry_px, "pnl": 0.0})
                    remaining_budget -= notional

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
            trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": p.qty, "px": exit_px, "pnl": pnl})
            cash += exit_px * p.qty
            positions.pop(sym, None)
        equity_curve.append({"date": dt, "equity": cash, "cash": cash, "mtm": 0.0, "positions": 0})

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    return trades_df, eq_df

# ========= METRICS & I/O =========
def performance_metrics(eq: pd.DataFrame) -> Dict[str, float]:
    if eq.empty:
        return {"CAGR": 0.0, "MaxDD": 0.0, "Sharpe": 0.0}
    eq = eq.copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    sharpe = (eq["ret"].mean() / (eq["ret"].std() + 1e-12)) * np.sqrt(252)
    roll_max = eq["equity"].cummax()
    dd = (eq["equity"] / (roll_max + 1e-9)) - 1.0
    maxdd = float(dd.min())
    if len(eq) > 1:
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = (eq["equity"].iloc[-1] / max(eq["equity"].iloc[0], 1e-9)) ** (1/years) - 1 if years > 0 else 0.0
    else:
        cagr = 0.0
    return {"CAGR": float(cagr), "MaxDD": maxdd, "Sharpe": float(sharpe)}

def trades_summary(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"Trades": 0, "WinRate": 0.0, "AvgWin": 0.0, "AvgLoss": 0.0, "ProfitFactor": 0.0, "NetPNL": 0.0}
    sells = trades[trades["side"] == "SELL"]
    pnl_sum = float(sells["pnl"].sum())
    wins = sells[sells["pnl"] > 0]
    losses = sells[sells["pnl"] <= 0]
    winrate = (len(wins) / max(1, len(sells))) if len(sells) else 0.0
    avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
    profit_factor = float(wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and abs(losses["pnl"].sum()) > 0 else (float("inf") if len(wins) else 0.0)
    return {
        "Trades": int(len(sells)),
        "WinRate": float(winrate),
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "ProfitFactor": profit_factor,
        "NetPNL": pnl_sum,
    }

def main():
    syms = load_universe()
    trades, eq = backtest(syms)
    trades.to_csv("backtest_trades.csv", index=False)
    eq.to_csv("equity_curve.csv")

    pm = performance_metrics(eq)
    ts = trades_summary(trades)

    print("\n=== Backtest Summary ===")
    print(f"Start Equity: {TOTAL_CAPITAL:,.2f}")
    print(f"End   Equity: {eq['equity'].iloc[-1]:,.2f}" if not eq.empty else "End   Equity: n/a")
    print(f"   CAGR: {pm['CAGR']:.4f}")
    print(f"  MaxDD: {pm['MaxDD']:.4f}")
    print(f" Sharpe: {pm['Sharpe']:.4f}")
    print(f"Trades : {ts['Trades']}")
    print(f"WinRate: {ts['WinRate']*100:.1f}%  PF: {ts['ProfitFactor']:.2f}  NetPNL: {ts['NetPNL']:.2f}")

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

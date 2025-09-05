# backtesting/backtest_quick.py
# Portfolio backtester aligned with SwingStrategy rules — type-checker safe.
from __future__ import annotations

import os, math, time, builtins, warnings, importlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ========= ENV / KNOBS =========
NIFTY500_PATH   = os.getenv("NIFTY500_PATH", "nifty500.csv")
SCAN_CSV        = os.getenv("SCAN_CSV", "0") == "1"
CSV_PATH        = os.getenv("CSV_PATH", "instruments_nse_eq.csv")
UNIVERSE_LIMIT  = int(os.getenv("UNIVERSE_LIMIT", "150"))
APPEND_NS       = os.getenv("APPEND_NS", "1") == "1"  # append ".NS" to tickers if missing

TOTAL_CAPITAL   = float(os.getenv("TOTAL_CAPITAL", "10000"))
MAX_INVESTED    = float(os.getenv("MAX_INVESTED", "5000"))
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "5"))
PER_TRADE_RISK  = float(os.getenv("PER_TRADE_RISK", "1000"))

EVAL_STEP_DAYS  = int(os.getenv("EVAL_STEP_DAYS", "1"))

# Backtest dates (None = last ~250 bars)
START_DATE      = os.getenv("BT_START", "") or None
END_DATE        = os.getenv("BT_END", "") or None

# Costs per side
SLIPPAGE_PCT    = float(os.getenv("SLIPPAGE_PCT", "0.001"))   # 0.10%
FEES_PCT        = float(os.getenv("FEES_PCT", "0.0003"))      # 0.03%

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
VOL_MULT        = float(os.getenv("VOL_MULT", "1.5"))  # current vol >= VOL_MULT * avg20

# Exits
RSI_EXIT_BELOW  = float(os.getenv("RSI_EXIT_BELOW", "30"))
SMA_EXIT_N      = int(os.getenv("SMA_EXIT_N", "20"))
ATR_STOP_MULT   = float(os.getenv("ATR_STOP_MULT", "2.0"))
ATR_TGT_MULT    = float(os.getenv("ATR_TGT_MULT", "2.0"))

# Pattern filters (optional; if module is absent, it stays off automatically)
ENABLE_PATTERN_FILTERS = os.getenv("ENABLE_PATTERN_FILTERS", "0") == "1"  # default OFF to keep things silent
PAT_MIN_BONUS   = int(os.getenv("PAT_MIN_BONUS", "2"))
PAT_SQUEEZE_LB  = int(os.getenv("PAT_SQUEEZE_LB", "20"))
PAT_SQUEEZE_PCTL= float(os.getenv("PAT_SQUEEZE_PCTL", "5"))
PAT_TRI_WINDOW  = int(os.getenv("PAT_TRI_WINDOW", "30"))
PAT_TRI_SHRINK  = float(os.getenv("PAT_TRI_SHRINK", "25"))

# ========= tiny, safe helpers =========
def ffloat(x: Any) -> float:
    """Coerce pandas/NumPy scalars or length-1 series to a Python float; NaN if not numeric."""
    try:
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return float("nan")
            return float(x.iloc[-1])
        if isinstance(x, (np.generic,)):
            return float(x)  # type: ignore[arg-type]
        if isinstance(x, (pd.Timestamp,)):
            return float("nan")
        if x is None:
            return float("nan")
        return float(x)  # may raise
    except Exception:
        return float("nan")

def safe_get(df: pd.DataFrame, idx: Any, col: str) -> float:
    """Safely pull a scalar cell, else NaN."""
    try:
        if idx in df.index and col in df.columns:
            # use .at if exact, else .loc then coerce
            try:
                val = df.at[idx, col]
            except Exception:
                val = df.loc[idx, col]
            return ffloat(val)
        return float("nan")
    except Exception:
        return float("nan")

def list_sum(vals: List[float]) -> float:
    total = 0.0
    for v in vals:
        total += float(v)
    return total

# ========= SYMBOL UNIVERSE =========
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
    sym = builtins.str(symbol).split(':')[-1].strip().upper()
    if APPEND_NS and not sym.endswith(".NS"):
        sym = f"{sym}.NS"
    return sym

def fetch_ohlc(symbol: str) -> Optional[pd.DataFrame]:
    """yfinance with retry + synthetic fallback (High/Low/Close/Volume)."""
    try:
        import yfinance as yf
        tries, last_err = 3, None
        for i in range(tries):
            try:
                hist_kw = dict(interval="1d", auto_adjust=False)
                if START_DATE or END_DATE:
                    df = yf.Ticker(_ticker(symbol)).history(start=START_DATE, end=END_DATE, **hist_kw)
                else:
                    df = yf.Ticker(_ticker(symbol)).history(period="250d", **hist_kw)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out = pd.DataFrame(index=pd.to_datetime(df.index))
                    for name in ["High","Low","Close","Volume"]:
                        if name in df.columns:
                            out[name] = pd.to_numeric(df[name], errors="coerce").astype(float)
                        else:
                            num = df.select_dtypes(include=["number"])
                            if num.shape[1]:
                                out[name] = pd.to_numeric(num.iloc[:, 0], errors="coerce").astype(float)
                            else:
                                raise RuntimeError("No numeric columns")
                    out = out.dropna()
                    return out if not out.empty else None
                last_err = RuntimeError("Empty frame")
            except Exception as e:
                last_err = e
            time.sleep(0.8*(i+1))
        raise last_err or RuntimeError("Unknown yfinance error")
    except Exception as e:
        print(f"[bt] yfinance failed for {symbol}: {type(e).__name__}: {e}. Using synthetic.")
        n = 250
        rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
        price = 100 + np.cumsum(np.random.normal(0.2, 1.5, size=n))
        high  = price + np.abs(np.random.normal(0.5, 0.7, size=n))
        low   = price - np.abs(np.random.normal(0.5, 0.7, size=n))
        close = price
        vol   = np.random.uniform(1e5, 1e6, size=n)
        return pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": vol}, index=rng).astype(float)

# ========= INDICATORS (pure pandas, outputs Series[float]) =========
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    delta = s.diff()
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
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100.0
    return dx.rolling(n, min_periods=n).mean()

def highest(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).max()

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def apply_costs(px: float, side: str) -> float:
    adj = px * (SLIPPAGE_PCT + FEES_PCT)
    return px + adj if side == "BUY" else px - adj

# ========= PATTERN GATE (optional, importlib so no unresolved-import warnings) =========
def run_pattern_gate(df_tail: pd.DataFrame) -> Tuple[bool, int]:
    """Return (ok, bonus). If module missing or disabled, always (True, 0)."""
    if not ENABLE_PATTERN_FILTERS:
        return True, 0
    try:
        # Try both names, but only via strings (no static import)
        mod = None
        for name in ("backtesting.pattern_filters", ".pattern_filters"):
            try:
                mod = importlib.import_module(name)  # type: ignore[no-redef]
                break
            except Exception:
                continue
        if mod is None:
            # one more try relative to this file's package
            pkg = __package__ or "backtesting"
            try:
                mod = importlib.import_module(f"{pkg}.pattern_filters")
            except Exception:
                return True, 0  # fail-open
        # set defaults so the module can read env
        os.environ.setdefault("PAT_MIN_BONUS", str(PAT_MIN_BONUS))
        os.environ.setdefault("PAT_SQUEEZE_LB", str(PAT_SQUEEZE_LB))
        os.environ.setdefault("PAT_SQUEEZE_PCTL", str(PAT_SQUEEZE_PCTL))
        os.environ.setdefault("PAT_TRI_WINDOW", str(PAT_TRI_WINDOW))
        os.environ.setdefault("PAT_TRI_SHRINK", str(PAT_TRI_SHRINK))
        os.environ.setdefault("ENABLE_PATTERN_FILTERS", "1")
        gate_fn = getattr(mod, "pattern_gate", None)
        if gate_fn is None:
            return True, 0
        out = gate_fn(df_tail)
        ok = bool(out.get("ok", True))
        bonus = int(out.get("bonus", 0))
        return ok, bonus
    except Exception:
        return True, 0  # fail-open if anything goes wrong

# ========= PORTFOLIO ENGINE =========
@dataclass
class Position:
    symbol: str
    qty: int
    entry_px: float
    stop_px: float
    target_px: float
    last_close: float

# ========= CORE BACKTEST =========
def backtest(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cash: float = float(TOTAL_CAPITAL)
    positions: Dict[str, Position] = {}
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Dict[str, Any]] = []

    # preload data & indicators
    data_map: Dict[str, pd.DataFrame] = {}
    min_rows = max(ATR_N, BREAKOUT_2, ADX_N, RSI_B) + 5
    for s in symbols:
        df = fetch_ohlc(s)
        if df is None or df.empty or df.shape[0] < min_rows:
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
        # evaluate only at step
        if di % max(1, EVAL_STEP_DAYS) != 0:
            # MTM only
            mtm_vals: List[float] = []
            for sym, pos in positions.items():
                df = data_map.get(sym)
                if df is None or dt not in df.index:
                    continue
                px = safe_get(df, dt, "Close")
                mtm_vals.append(px * float(pos.qty))
            mtm = list_sum(mtm_vals)
            equity = cash + mtm
            equity_curve.append({"date": dt, "equity": float(equity), "cash": float(cash), "mtm": float(mtm), "positions": int(len(positions))})
            continue

        # 1) exits & trailing updates
        to_close: List[str] = []
        for sym, pos in list(positions.items()):
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            c   = safe_get(df, dt, "Close")
            a14 = safe_get(df, dt, "ATR")
            if math.isnan(c) or math.isnan(a14):
                continue
            # update last_close
            pos.last_close = c
            # trail both stop & target with ATR
            new_stop   = c - ATR_STOP_MULT * a14
            new_target = c + ATR_TGT_MULT  * a14
            if new_stop > pos.stop_px:
                pos.stop_px = float(new_stop)
            if new_target > pos.target_px:
                pos.target_px = float(new_target)

            # exit conditions
            rsi14 = safe_get(df, dt, "RSI14")
            sma20 = safe_get(df, dt, "SMA20")
            stop_hit   = c <= pos.stop_px
            target_hit = c >= pos.target_px
            sma_break  = (not math.isnan(sma20)) and (c < sma20)
            rsi_break  = (not math.isnan(rsi14)) and (rsi14 < RSI_EXIT_BELOW)

            if stop_hit or target_hit or sma_break or rsi_break:
                exit_px = apply_costs(c, "SELL")
                pnl = (exit_px - pos.entry_px) * float(pos.qty)
                cash += exit_px * float(pos.qty)
                trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": int(pos.qty), "px": float(exit_px), "pnl": float(pnl)})
                to_close.append(sym)

        for sym in to_close:
            positions.pop(sym, None)

        # 2) entries (respect caps & notional)
        open_slots = max(0, MAX_POSITIONS - len(positions))
        if open_slots > 0:
            # invested now
            invested_vals: List[float] = []
            for sym, pos in positions.items():
                df = data_map.get(sym)
                if df is None or dt not in df.index: 
                    continue
                px = safe_get(df, dt, "Close")
                invested_vals.append(px * float(pos.qty))
            invested_now = list_sum(invested_vals)
            remaining_budget = max(0.0, MAX_INVESTED - invested_now)

            if remaining_budget > 0 and cash > 0:
                cands: List[Tuple[float, str, float, float]] = []
                for sym, df in data_map.items():
                    if sym in positions or dt not in df.index:
                        continue
                    c    = safe_get(df, dt, "Close")
                    a14  = safe_get(df, dt, "ATR")
                    adxv = safe_get(df, dt, "ADX")
                    r7   = safe_get(df, dt, "RSI7")
                    r14  = safe_get(df, dt, "RSI14")
                    r21  = safe_get(df, dt, "RSI21")
                    hh20 = safe_get(df, dt, "HH20")
                    hh50 = safe_get(df, dt, "HH50")
                    v    = safe_get(df, dt, "Volume")
                    vavg = safe_get(df, dt, "VOLAVG20")

                    if any(math.isnan(x) for x in [c,a14,adxv,r7,r14,r21,hh20,hh50,v,vavg]):
                        continue
                    if v < VOL_MULT * vavg:
                        continue

                    # breakout conditions
                    tol = 1.0  # set 0.995 if you want slight tolerance
                    breakout = (c > tol * hh20) or (c > tol * hh50)
                    rsi_ok   = (r7 > RSI_MIN_ALL) and (r14 > RSI_MIN_ALL) and (r21 > RSI_MIN_ALL)
                    adx_ok   = adxv > ADX_MIN
                    if not (breakout and rsi_ok and adx_ok):
                        continue

                    # optional: pattern gate
                    bonus = 0
                    if ENABLE_PATTERN_FILTERS:
                        ctx = df.loc[:dt].tail(max(60, PAT_TRI_WINDOW + 5)).copy()
                        ok, bonus = run_pattern_gate(ctx)
                        if not ok:
                            continue

                    # base score + bonus
                    eps = 1e-9
                    bratio = max(c / max(hh20, eps), c / max(hh50, eps))
                    score = 0.6 * bratio + 0.3 * (adxv / 25.0) + 0.1 * (r14 / 50.0) + 0.5 * float(bonus)

                    cands.append((float(score), builtins.str(sym), float(c), float(a14)))

                cands.sort(reverse=True)
                selected = cands[:open_slots]
                # total score as float
                total_score = list_sum([max(0.0, s[0]) for s in selected]) or 1.0

                for score, sym, price, a14 in selected:
                    if remaining_budget <= 0 or cash <= 0:
                        break
                    # budget proportional to score
                    budget_for_this = min(remaining_budget * (score / total_score), remaining_budget, cash)
                    qty_cash = int(budget_for_this // max(price, 1e-9))
                    qty_risk = int(max(1.0, PER_TRADE_RISK / max(ATR_STOP_MULT * a14, 1e-9)))
                    qty = max(1, min(qty_cash, qty_risk))
                    notional = float(qty) * float(price)
                    if qty < 1 or notional <= 0:
                        continue
                    if notional > remaining_budget + 1e-6:
                        qty = int(remaining_budget // max(price, 1e-9))
                        if qty < 1:
                            continue
                        notional = float(qty) * float(price)

                    entry_px = apply_costs(float(price), "BUY")
                    cost = entry_px * float(qty)
                    if cost > cash + 1e-6:
                        continue

                    positions[sym] = Position(
                        symbol=sym,
                        qty=int(qty),
                        entry_px=float(entry_px),
                        stop_px=float(price - ATR_STOP_MULT * a14),
                        target_px=float(price + ATR_TGT_MULT  * a14),
                        last_close=float(price),
                    )
                    cash -= cost
                    trades.append({"symbol": sym, "side": "BUY", "date": dt, "qty": int(qty), "px": float(entry_px), "pnl": 0.0})
                    remaining_budget -= notional

        # 3) mark-to-market equity
        mtm_vals2: List[float] = []
        for sym, pos in positions.items():
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            px = safe_get(df, dt, "Close")
            mtm_vals2.append(px * float(pos.qty))
        mtm = list_sum(mtm_vals2)
        equity = cash + mtm
        equity_curve.append({"date": dt, "equity": float(equity), "cash": float(cash), "mtm": float(mtm), "positions": int(len(positions))})

    # Close all on last day
    if all_dates:
        dt = all_dates[-1]
        for sym, p in list(positions.items()):
            df = data_map.get(sym)
            if df is None or dt not in df.index:
                continue
            c = safe_get(df, dt, "Close")
            exit_px = apply_costs(c, "SELL")
            pnl = (exit_px - p.entry_px) * float(p.qty)
            trades.append({"symbol": sym, "side": "SELL", "date": dt, "qty": int(p.qty), "px": float(exit_px), "pnl": float(pnl)})
            cash += exit_px * float(p.qty)
            positions.pop(sym, None)
        equity_curve.append({"date": dt, "equity": float(cash), "cash": float(cash), "mtm": 0.0, "positions": 0})

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    return trades_df, eq_df

# ========= METRICS & I/O =========
def performance_metrics(eq: pd.DataFrame) -> Dict[str, float]:
    if eq.empty:
        return {"CAGR": 0.0, "MaxDD": 0.0, "Sharpe": 0.0}
    eq = eq.copy()
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    sharpe = float((eq["ret"].mean() / (eq["ret"].std() + 1e-12)) * np.sqrt(252))
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

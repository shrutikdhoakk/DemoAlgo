# backtest_quick.py — portfolio backtester aligned with HighProbabilitySwing strategy
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
UNIVERSE_LIMIT  = int(os.getenv("UNIVERSE_LIMIT", "150"))   # smaller to avoid Yahoo rate limits

ACCOUNT_SIZE    = float(os.getenv("ACCOUNT_SIZE", os.getenv("TOTAL_CAPITAL","100000")))
RISK_PERCENT    = float(os.getenv("RISK_PERCENT", "0.02"))  # 2% per trade by default
MAX_POSITIONS   = int(os.getenv("MAX_POSITIONS", "5"))
MAX_INVESTED    = float(os.getenv("MAX_INVESTED", str(ACCOUNT_SIZE*0.5)))

EVAL_STEP_DAYS  = int(os.getenv("EVAL_STEP_DAYS", "1"))     # evaluate daily
LONG_ONLY       = os.getenv("LONG_ONLY","1") == "1"         # Indian CNC realism

# Backtest dates (None = last 500 bars)
START_DATE      = os.getenv("BT_START", "") or None
END_DATE        = os.getenv("BT_END", "") or None

# Costs
SLIPPAGE_PCT    = float(os.getenv("SLIPPAGE_PCT", "0.001"))   # 0.10% per side
FEES_PCT        = float(os.getenv("FEES_PCT", "0.0003"))      # 0.03% per side

# Indicator params (MUST mirror strategy)
ATR_N           = int(os.getenv("ATR_N", "14"))
RSI_N           = int(os.getenv("RSI_N", "14"))
BB_N            = int(os.getenv("BB_N", "20"))
BB_K            = float(os.getenv("BB_K", "2.0"))
EMA_FAST        = int(os.getenv("EMA_FAST", "20"))
SMA_SLOW        = int(os.getenv("SMA_SLOW", "50"))
VOL_MULT        = float(os.getenv("VOL_MULT", "1.2"))         # vol > 1.2x avg20

# Exits / partials
INIT_ATR_MULT   = float(os.getenv("INIT_ATR_MULT", "2.0"))
TRAIL_ATR_MULT  = float(os.getenv("TRAIL_ATR_MULT", "1.0"))
PART_1_R        = float(os.getenv("PART_1_R", "1.5"))
PART_2_R        = float(os.getenv("PART_2_R", "2.5"))
PART_1_PCT      = float(os.getenv("PART_1_PCT", "0.25"))
PART_2_PCT      = float(os.getenv("PART_2_PCT", "0.50"))

# ========= UTILS =========
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

# ========= DATA =========
def _ticker(symbol: str) -> str:
    return f"{builtins.str(symbol).split(':')[-1].strip().upper()}.NS"

def fetch_ohlc(symbol: str) -> Optional[pd.DataFrame]:
    """
    yfinance with retry (Open/High/Low/Close/Volume).
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
                    df = yf.Ticker(_ticker(symbol)).history(period="500d", interval="1d", auto_adjust=False)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    out = pd.DataFrame(index=pd.to_datetime(df.index))
                    def pick(name: str) -> pd.Series:
                        if name in df.columns:
                            return pd.to_numeric(df[name], errors="coerce").astype(float)
                        num = df.select_dtypes(include=["number"])
                        if num.shape[1]:
                            return pd.to_numeric(num.iloc[:,0], errors="coerce").astype(float)
                        raise RuntimeError("No numeric columns")
                    for name in ["Open","High","Low","Close","Volume"]:
                        out[name] = pick(name)
                    return out.dropna()
                last_err = RuntimeError("Empty frame")
            except Exception as e:
                last_err = e
            time.sleep(0.8*(i+1))
        raise last_err or RuntimeError("Unknown yfinance error")
    except Exception as e:
        # synthesize to keep loop moving
        print(f"[bt] yfinance failed for {symbol}: {type(e).__name__}: {e}. Using synthetic.")
        n = 500
        rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
        price = 100 + np.cumsum(np.random.normal(0.2, 1.5, size=n))
        high  = price + np.abs(np.random.normal(0.5, 0.7, size=n))
        low   = price - np.abs(np.random.normal(0.5, 0.7, size=n))
        openp = price + np.random.normal(0.0, 0.3, size=n)
        close = price
        vol   = np.random.uniform(1e5, 1e6, size=n)
        return pd.DataFrame({"Open":openp,"High": high, "Low": low, "Close": close, "Volume": vol}, index=rng).astype(float)

# ========= INDICATORS =========
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").astype(float)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(n, min_periods=n).mean()
    avg_loss = loss.rolling(n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).ewm(span=n, adjust=False).mean()

def sma(series: pd.Series, n: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).rolling(n, min_periods=n).mean()

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = sma(series, n)
    sd = series.rolling(n, min_periods=n).std()
    upper = m + k*sd
    lower = m - k*sd
    return lower, m, upper

# ========= PATTERNS =========
def _body(o: float, c: float) -> float:
    return abs(c-o)

def morning_star(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    cond1 = c.shift(2) < o.shift(2)  # bearish
    small = (_body(o.shift(1), c.shift(1)) <= (_body(o.shift(2), c.shift(2)) * 0.5))
    cond3 = c > o  # bullish large
    pierce = c >= (o.shift(2) + (c.shift(2)-o.shift(2)) * 0.5)
    return (cond1 & small & cond3 & pierce).fillna(False)

def evening_star(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    cond1 = c.shift(2) > o.shift(2)  # bullish
    small = (_body(o.shift(1), c.shift(1)) <= (_body(o.shift(2), c.shift(2)) * 0.5))
    cond3 = c < o  # bearish large
    pierce = c <= (o.shift(2) + (c.shift(2)-o.shift(2)) * 0.5)
    return (cond1 & small & cond3 & pierce).fillna(False)

def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    o, c = df["Open"], df["Close"]
    b1 = c > o
    b2 = c.shift(1) > o.shift(1)
    b3 = c.shift(2) > o.shift(2)
    higher = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    return (b1 & b2 & b3 & higher).fillna(False)

def three_black_crows(df: pd.DataFrame) -> pd.Series:
    o, c = df["Open"], df["Close"]
    b1 = c < o
    b2 = c.shift(1) < o.shift(1)
    b3 = c.shift(2) < o.shift(2)
    lower = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    return (b1 & b2 & b3 & lower).fillna(False)

def hammer(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs()
    lower_shadow = (o.combine(c, max) - l)
    upper_shadow = (h - o.combine(c, min))
    return ((lower_shadow >= body*2.5) & (upper_shadow <= body*0.5)).fillna(False)

# ========= PORTFOLIO =========
@dataclass
class Position:
    symbol: str
    qty: int
    entry_px: float
    stop_px: float
    trail_px: float
    risk_per_share: float
    r_multiple: float = 0.0
    part1_done: bool = False
    part2_done: bool = False

def apply_costs(px: float, side: str) -> float:
    adj = px * (SLIPPAGE_PCT + FEES_PCT)
    return px + adj if side == "BUY" else px - adj

def weekly_bias(df: pd.DataFrame) -> pd.Series:
    wk = df.resample("W-FRI").last()
    wk["ema20"] = ema(wk["Close"], EMA_FAST)
    wk["sma50"] = sma(wk["Close"], SMA_SLOW)
    bias = (wk["Close"] > wk["ema20"]) & (wk["Close"] > wk["sma50"])
    # forward fill back to daily index
    return bias.reindex(df.index, method="ffill").fillna(False)

def long_entry_signal(df: pd.DataFrame) -> pd.Series:
    # Trend alignment (daily)
    ema20 = ema(df["Close"], EMA_FAST)
    sma50 = sma(df["Close"], SMA_SLOW)
    bias_d = (df["Close"] > ema20) & (df["Close"] > sma50)
    # Weekly confluence
    bias_w = weekly_bias(df)
    # Momentum (RSI rising from sub-40 toward 60)
    r = rsi(df["Close"], RSI_N)
    mom = (r > 40) & (r < 65) & (r > r.shift(1)) & (r.rolling(5).min() < 45)
    # Bollinger context / support
    lower, mid, upper = bollinger(df["Close"], BB_N, BB_K)
    near_support = (df["Close"] <= mid)
    # Patterns
    patt = morning_star(df) | three_white_soldiers(df) | (hammer(df) & near_support)
    # Volume expansion
    vexp = df["Volume"] > (df["Volume"].rolling(20, min_periods=20).mean() * VOL_MULT)
    return (bias_d & bias_w & mom & patt & vexp).fillna(False)

def short_entry_signal(df: pd.DataFrame) -> pd.Series:
    ema20 = ema(df["Close"], EMA_FAST)
    sma50 = sma(df["Close"], SMA_SLOW)
    bias_d = (df["Close"] < ema20) & (df["Close"] < sma50)
    wk = df.resample("W-FRI").last()
    wk["ema20"] = ema(wk["Close"], EMA_FAST)
    wk["sma50"] = sma(wk["Close"], SMA_SLOW)
    bias_w = ((wk["Close"] < wk["ema20"]) & (wk["Close"] < wk["sma50"])).reindex(df.index, method="ffill").fillna(False)
    r = rsi(df["Close"], RSI_N)
    mom = (r < 60) & (r > 35) & (r < r.shift(1)) & (r.rolling(5).max() > 58)
    lower, mid, upper = bollinger(df["Close"], BB_N, BB_K)
    near_res = (df["Close"] >= mid)
    patt = evening_star(df) | three_black_crows(df)
    vexp = df["Volume"] > (df["Volume"].rolling(20, min_periods=20).mean() * VOL_MULT)
    return (bias_d & bias_w & mom & (patt | near_res) & vexp).fillna(False)

# ========= BACKTEST =========
def run_backtest(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades = []
    equity = []

    cash = ACCOUNT_SIZE
    positions: Dict[str, Position] = {}
    ts_equity = 0.0

    # prefetch data
    data = {s: fetch_ohlc(s) for s in symbols}
    data = {s: df for s,df in data.items() if isinstance(df, pd.DataFrame) and df.shape[0] > 120}

    # iterate by time index (use union of all dates)
    all_idx = pd.Index(sorted(set().union(*[df.index for df in data.values()])))
    last_eval = None

    for t in all_idx:
        # 1) Update positions and equity
        for sym, pos in list(positions.items()):
            df = data[sym]
            if t not in df.index:
                continue
            row = df.loc[t]
            c = float(row["Close"])
            # update r-multiple
            pos.r_multiple = (c - pos.entry_px) / max(1e-8, pos.risk_per_share)
            # trail after first partial
            atrv = float(atr(df.loc[:t].tail(ATR_N*3), ATR_N).iloc[-1])
            ema20 = float(ema(df.loc[:t]["Close"], EMA_FAST).iloc[-1])
            if pos.part1_done or pos.part2_done:
                pos.trail_px = max(pos.trail_px, c - TRAIL_ATR_MULT*atrv)

            # partial exits
            to_sell = 0
            if (not pos.part1_done) and (pos.r_multiple >= PART_1_R):
                sell_qty = max(1, int(pos.qty * PART_1_PCT))
                to_sell += sell_qty
                pos.part1_done = True
            if (not pos.part2_done) and (pos.r_multiple >= PART_2_R):
                sell_qty = max(1, int(pos.qty * PART_2_PCT))
                to_sell += sell_qty
                pos.part2_done = True

            # hard exits
            exit_reason = None
            if c <= pos.trail_px and (pos.part1_done or pos.part2_done):
                exit_reason = "TRAIL_ATR"
                to_sell = pos.qty
            elif c < ema20:
                exit_reason = "EMA20_BREACH"
                to_sell = pos.qty

            if to_sell > 0:
                px = apply_costs(c, "SELL")
                qty = min(to_sell, pos.qty)
                cash += qty * px
                pos.qty -= qty
                trades.append(dict(time=t, symbol=sym, side="SELL", qty=qty, price=px, reason=exit_reason or "PARTIAL"))
                if pos.qty <= 0:
                    del positions[sym]

        # mark-to-market
        mark_value = 0.0
        for sym, pos in positions.items():
            df = data[sym]
            if t in df.index:
                c = float(df.loc[t, "Close"])
                mark_value += pos.qty * c
        ts_equity = cash + mark_value
        equity.append(dict(time=t, equity=ts_equity, cash=cash, invested=mark_value))

        # 2) Evaluate entries at steps & capacity
        if last_eval is not None and (t - last_eval).days < EVAL_STEP_DAYS:
            continue
        last_eval = t

        seats_left = MAX_POSITIONS - len(positions)
        budget_left = MAX_INVESTED - sum(pos.qty * data[s].loc[t, "Close"] for s,pos in positions.items() if t in data[s].index)
        if seats_left <= 0 or budget_left <= 0:
            continue

        for sym, df in data.items():
            if sym in positions or t not in df.index:
                continue
            sub = df.loc[:t].copy()
            if sub.shape[0] < max(EMA_FAST, SMA_SLOW, ATR_N, RSI_N, BB_N) + 5:
                continue

            long_sig = bool(long_entry_signal(sub).iloc[-1])
            short_sig = (not LONG_ONLY) and bool(short_entry_signal(sub).iloc[-1])
            side = "BUY" if long_sig else ("SELL" if short_sig else None)
            if side is None:
                continue

            c = float(sub["Close"].iloc[-1])
            a = float(atr(sub, ATR_N).iloc[-1])
            risk_per_share = INIT_ATR_MULT * a if side=="BUY" else 1.5*a
            if risk_per_share <= 0 or math.isnan(risk_per_share):
                continue

            # base qty from risk
            account_risk = ACCOUNT_SIZE * RISK_PERCENT
            qty = int(max(1, math.floor(account_risk / risk_per_share)))

            # volatility adjustment
            atr_pct = a / max(1e-6, c)
            if atr_pct > 0.02:
                qty = int(max(1, math.floor(qty * 0.75)))
            elif atr_pct < 0.01:
                qty = int(max(1, math.floor(qty * 1.15)))

            # respect budget
            notional = qty * c
            if notional > budget_left:
                qty = int(max(1, math.floor(budget_left / max(1e-6, c))))
            if qty < 1:
                continue

            px = apply_costs(c, side)
            if side == "BUY":
                stop = c - INIT_ATR_MULT * a
                trail = stop
                positions[sym] = Position(
                    symbol=sym, qty=qty, entry_px=px, stop_px=stop,
                    trail_px=trail, risk_per_share=(px - stop)
                )
            else:
                stop = c + 1.5 * a
                trail = stop
                positions[sym] = Position(
                    symbol=sym, qty=qty, entry_px=px, stop_px=stop,
                    trail_px=trail, risk_per_share=(stop - px)
                )

            cash -= qty * px
            trades.append(dict(time=t, symbol=sym, side=side, qty=qty, price=px, reason="ENTRY"))
            seats_left -= 1
            budget_left -= qty * c
            if seats_left <= 0 or budget_left <= 0:
                break

    # Liquidate residuals at final bar
    if positions:
        last_t = all_idx[-1]
        for sym, pos in list(positions.items()):
            df = data[sym]
            if last_t in df.index:
                c = float(df.loc[last_t, "Close"])
                px = apply_costs(c, "SELL")
                trades.append(dict(time=last_t, symbol=sym, side="SELL", qty=pos.qty, price=px, reason="EOD_LIQ"))
                cash += pos.qty * px
        positions.clear()
        equity.append(dict(time=last_t, equity=cash, cash=cash, invested=0.0))

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity).drop_duplicates(subset=["time"]).sort_values("time")
    return trades_df, eq_df

# ========= MAIN =========
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
        years = (pd.to_datetime(eq["time"].iloc[-1]) - pd.to_datetime(eq["time"].iloc[0])).days / 365.25
        cagr = (eq["equity"].iloc[-1] / max(eq["equity"].iloc[0], 1e-9)) ** (1/years) - 1 if years > 0 else 0.0
    else:
        cagr = 0.0
    return {"CAGR": float(cagr), "MaxDD": maxdd, "Sharpe": float(sharpe)}

def trades_summary(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"Trades": 0, "WinRate": 0.0, "AvgWin": 0.0, "AvgLoss": 0.0, "ProfitFactor": 0.0, "NetPNL": 0.0}
    sells = trades[trades["side"] == "SELL"]
    pnl_sum = float((sells.get("pnl") if "pnl" in sells.columns else (sells["price"]*sells["qty"])).sum())  # fallback
    wins = sells[sells.get("pnl", pd.Series([0]*len(sells))) > 0]
    losses = sells[sells.get("pnl", pd.Series([0]*len(sells))) <= 0]
    winrate = (len(wins) / max(1, len(sells))) if len(sells) else 0.0
    avg_win = float(wins.get("pnl", pd.Series(dtype=float)).mean()) if len(wins) else 0.0
    avg_loss = float(losses.get("pnl", pd.Series(dtype=float)).mean()) if len(losses) else 0.0
    profit_factor = float(wins.get("pnl", pd.Series(dtype=float)).sum() / abs(losses.get("pnl", pd.Series(dtype=float)).sum())) if len(losses) and abs(losses.get("pnl", pd.Series(dtype=float)).sum()) > 0 else (float("inf") if len(wins) else 0.0)
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
    trades, eq = run_backtest(syms)
    trades.to_csv("backtest_trades.csv", index=False)
    eq.to_csv("equity_curve.csv", index=False)

    pm = performance_metrics(eq)
    ts = trades_summary(trades)

    print("\n=== Backtest Summary ===")
    print(f"Start Equity: {ACCOUNT_SIZE:,.2f}")
    if not eq.empty:
        print(f"End   Equity: {eq['equity'].iloc[-1]:,.2f}")
    print(f"   CAGR: {pm['CAGR']:.4f}")
    print(f"  MaxDD: {pm['MaxDD']:.4f}")
    print(f" Sharpe: {pm['Sharpe']:.4f}")
    print(f"Trades : {ts['Trades']}")
    print(f"WinRate: {ts['WinRate']*100:.1f}%  PF: {ts['ProfitFactor']:.2f}  NetPNL: {ts['NetPNL']:.2f}")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(pd.to_datetime(eq["time"]), eq["equity"])
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig("equity_curve.png", dpi=140)
        print("Saved: backtest_trades.csv, equity_curve.csv, equity_curve.png")
    except Exception:
        print("Saved: backtest_trades.csv, equity_curve.csv (matplotlib not available)")

if __name__ == "__main__":
    main()
 
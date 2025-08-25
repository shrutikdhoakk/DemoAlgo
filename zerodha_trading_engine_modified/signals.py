# signals.py — NIFTY-500 / CSV aware, fast-SIM friendly
from __future__ import annotations
import os, time
from typing import Iterator, List, Optional, Callable, Any
import pandas as pd
from trade_signal import TradeSignal

# ---------- helpers ----------
def _normalize_symbols(symbols: List[str]) -> List[str]:
    return [str(s).split(":")[-1].strip().upper() for s in symbols]

def _is_index_like(s: str) -> bool:
    s2 = str(s).strip().upper()
    if not s2:
        return True
    bad = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCAP", "SENSEX", "VIX", "IX", "INDEX")
    return any(k in s2 for k in bad) or (" " in s2) or ("/" in s2)

def _dedupe_keep_order(vals: List[str]) -> List[str]:
    seen, out = set(), []
    for v in vals:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

def _pick_symbol_column(df: pd.DataFrame) -> str:
    for col in ["SYMBOL", "Symbol", "symbol", "tradingsymbol", "Tradingsymbol", "TRADINGSYMBOL"]:
        if col in df.columns:
            return col
    raise ValueError("No symbol column found. Expected one of SYMBOL/Tradingsymbol/tradingsymbol/symbol.")

def _load_nifty500_symbols(csv_path: str, limit: int = 0) -> List[str]:
    df = pd.read_csv(csv_path)
    col = _pick_symbol_column(df)
    vals = [str(x).split(":")[-1].strip().upper() for x in df[col].dropna().tolist()]
    vals = [v for v in vals if v and not _is_index_like(v)]
    vals = _dedupe_keep_order(vals)
    if limit > 0:
        vals = vals[:limit]
    return vals

def _load_instruments_symbols(csv_path: str, limit: int = 0) -> List[str]:
    df = pd.read_csv(csv_path)
    for c in ["segment", "Segment"]:
        if c in df.columns:
            df = df[df[c].astype(str).str.upper().str.contains("NSE")]
            break
    for c in ["instrument_type", "InstrumentType", "instrumenttype"]:
        if c in df.columns:
            df = df[df[c].astype(str).str.upper().str.contains("EQ")]
            break
    col = _pick_symbol_column(df)
    vals = [str(x).split(":")[-1].strip().upper() for x in df[col].dropna().tolist()]
    vals = [v for v in vals if v and not _is_index_like(v)]
    vals = _dedupe_keep_order(vals)
    if limit > 0:
        vals = vals[:limit]
    return vals

def _make_offline_fetcher() -> Callable[[str], pd.DataFrame]:
    import numpy as np
    def _fetch(symbol: str) -> pd.DataFrame:
        n = 250
        rng = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
        price = 100 + np.cumsum(np.random.normal(0.2, 1.5, size=n))
        high = price + np.abs(np.random.normal(0.5, 0.7, size=n))
        low = price - np.abs(np.random.normal(0.5, 0.7, size=n))
        close = price
        return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=rng).astype(float)
    return _fetch

def _make_yfinance_fetcher() -> Callable[[str], pd.DataFrame]:
    import yfinance as yf
    def pick_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns
        def pick(name):
            if name in cols:
                return df[name]
            if hasattr(cols, "get_level_values"):  # MultiIndex
                for c in cols:
                    parts = c if isinstance(c, tuple) else (c,)
                    if any(str(p).lower() == name.lower() for p in parts):
                        s = df[c]
                        if hasattr(s, "columns"):
                            s = s.iloc[:, 0]
                        return s
            num = df.select_dtypes(include=["number"])
            if num.shape[1] == 0:
                raise RuntimeError("No numeric columns for OHLC.")
            return num.iloc[:, 0]
        out = pd.concat(
            [pick("High").rename("High"), pick("Low").rename("Low"), pick("Close").rename("Close")],
            axis=1
        ).dropna()
        out["High"]  = out["High"].astype(float)
        out["Low"]   = out["Low"].astype(float)
        out["Close"] = out["Close"].astype(float)
        return out

    def _fetch(symbol: str) -> pd.DataFrame:
        ticker = f"{str(symbol).split(':')[-1].strip().upper()}.NS"
        df = yf.download(ticker, period="250d", interval="1d", auto_adjust=False, progress=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Empty data for {symbol}")
        return pick_ohlc(df)
    return _fetch

# ---------- main generator ----------
def generate_signals(symbols_or_cfg: Optional[Any] = None) -> Iterator[TradeSignal]:
    """
    Accepts either:
      - a list of symbols, OR
      - an AppConfig-like object, OR
      - nothing (uses env/CSV).
    Env knobs (preferred):
      NIFTY500_PATH=nifty500.csv
      SCAN_CSV=1  + CSV_PATH=instruments_nse_eq.csv
      UNIVERSE_LIMIT=0 (0 = all)
      EVAL_SECONDS=5
      REGIME_FILTER=0/1
      ALWAYS_TRADE=0/1
      OFFLINE_MODE=0/1
      MAX_POSITIONS, TOTAL_CAPITAL, MAX_INVESTED, PER_TRADE_RISK
    """
    from swing_strategy import SwingStrategy

    # ---- env knobs ----
    NIFTY500_PATH   = os.getenv("NIFTY500_PATH", "").strip()
    SCAN_CSV        = os.getenv("SCAN_CSV", "0") == "1"
    CSV_PATH        = os.getenv("CSV_PATH", "instruments_nse_eq.csv")
    UNIVERSE_LIMIT  = int(os.getenv("UNIVERSE_LIMIT", "0"))
    EVAL_SECONDS    = int(os.getenv("EVAL_SECONDS", "5"))
    REGIME_FILTER   = os.getenv("REGIME_FILTER", "0") == "1"
    ALWAYS_TRADE    = os.getenv("ALWAYS_TRADE", "0") == "1"
    OFFLINE_MODE    = os.getenv("OFFLINE_MODE", "0") == "1"

    # Capital & positions
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))
    TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", "10000"))
    MAX_INVESTED  = float(os.getenv("MAX_INVESTED", "5000"))
    PER_TRADE_RISK = float(os.getenv("PER_TRADE_RISK", "1000"))

    # ---- interpret input arg (list or cfg) ----
    cfg = None
    input_symbols: List[str] = []
    if isinstance(symbols_or_cfg, (list, tuple)):
        input_symbols = _normalize_symbols(list(symbols_or_cfg))
    elif symbols_or_cfg is not None:
        cfg = symbols_or_cfg  # AppConfig-like
        # If cfg has marketdata.symbols, prefer them
        try:
            md = getattr(cfg, "marketdata")
            maybe_syms = getattr(md, "symbols", None)
            if isinstance(maybe_syms, (list, tuple)):
                input_symbols = _normalize_symbols(list(maybe_syms))
        except Exception:
            pass

    # ---- choose universe (prefer explicit NIFTY500 file) ----
    syms: List[str] = []
    if input_symbols:
        syms = _dedupe_keep_order([s for s in input_symbols if s and not _is_index_like(s)])
        print(f"[signals] Using provided symbols: {syms[:5]}{'...' if len(syms)>5 else ''}", flush=True)
    elif NIFTY500_PATH:
        try:
            syms = _load_nifty500_symbols(NIFTY500_PATH, limit=UNIVERSE_LIMIT)
            print(f"[signals] Loaded {len(syms)} symbols from {NIFTY500_PATH}", flush=True)
        except Exception as e:
            print(f"[signals] NIFTY500_PATH failed: {e}. Falling back.", flush=True)
    elif SCAN_CSV:
        try:
            syms = _load_instruments_symbols(CSV_PATH, limit=UNIVERSE_LIMIT)
            print(f"[signals] Loaded {len(syms)} symbols from {CSV_PATH}", flush=True)
        except Exception as e:
            print(f"[signals] instruments CSV failed: {e}. Falling back.", flush=True)

    if not syms:
        syms = ["RELIANCE"]
        print(f"[signals] Using default symbols: {syms}", flush=True)

    # ---- immediate demo trades (prove path) ----
    if ALWAYS_TRADE and syms:
        yield TradeSignal(symbol=syms[0], side="BUY",  quantity=1)
        time.sleep(5)
        yield TradeSignal(symbol=syms[0], side="SELL", quantity=1)

    # ---- data fetcher selection ----
    if OFFLINE_MODE:
        fetcher = _make_offline_fetcher()
        print("[signals] OFFLINE_MODE=1 (synthetic OHLC)", flush=True)
    else:
        try:
            fetcher = _make_yfinance_fetcher()
            print("[signals] Using yfinance fetcher", flush=True)
        except Exception:
            fetcher = _make_offline_fetcher()
            print("[signals] yfinance not available; using synthetic OHLC", flush=True)

    # ---- build & run strategy ----
    strat = SwingStrategy(
        symbols=syms,
        data_fetcher=fetcher,
        evaluation_interval=EVAL_SECONDS,
        regime_filter=REGIME_FILTER,
        max_risk_per_trade=PER_TRADE_RISK,   # ATR-based sizing (optional)
        max_positions=MAX_POSITIONS,         # cap concurrent holdings to 5
        total_capital=TOTAL_CAPITAL,         # total capital = 10k
        max_invested=MAX_INVESTED,           # at most 5k invested at once
        timezone="Asia/Kolkata",
    )

    yield from strat.generate_signals()

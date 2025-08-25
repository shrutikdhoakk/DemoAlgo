"""
swing_strategy.py — top-5 selection + portfolio cap
- Scans a universe, ranks candidates by breakout/ADX/RSI score
- Limits open positions to max_positions
- Enforces portfolio notional cap (max_invested) from total capital
- Sizes per position by BOTH cash budget and ATR risk
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterator, List, Optional

import numpy as _np
import pandas as _pd

try:
    import yfinance as _yf
except ImportError:
    _yf = None

from trade_signal import TradeSignal


# -------------- helpers --------------

def _to_float_series(s: _pd.Series) -> _pd.Series:
    return _pd.to_numeric(s, errors="coerce").astype(float)

def _last_scalar(x) -> float:
    if isinstance(x, _pd.Series):
        return float(x.iloc[-1])
    if isinstance(x, _pd.DataFrame):
        return float(x.iloc[-1, 0]) if x.shape[1] else _np.nan
    try:
        return float(x)
    except Exception:
        return _np.nan

def _first_numeric_column(df: _pd.DataFrame) -> _pd.Series:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        num = df.apply(_pd.to_numeric, errors="coerce")
    return num.iloc[:, 0]

def _pick_col(df: _pd.DataFrame, name: str) -> _pd.Series:
    lname = name.lower()
    if not isinstance(df.columns, _pd.MultiIndex):
        if name in df.columns:
            return _to_float_series(df[name]).dropna()
        lowmap = {str(c).lower(): c for c in df.columns}
        if lname in lowmap:
            return _to_float_series(df[lowmap[lname]]).dropna()
        return _to_float_series(_first_numeric_column(df)).dropna()

    # MultiIndex
    if lname in [str(x).lower() for x in df.columns.get_level_values(0)]:
        sub = df.xs(key=name, axis=1, level=0, drop_level=False)
        s = sub.iloc[:, 0]
        if isinstance(s, _pd.DataFrame):
            s = s.iloc[:, 0]
        return _to_float_series(s).dropna()

    for col in df.columns:
        parts = col if isinstance(col, tuple) else (col,)
        if any(str(p).lower() == lname for p in parts):
            s = df[col]
            if isinstance(s, _pd.DataFrame):
                s = s.iloc[:, 0]
            return _to_float_series(s).dropna()

    return _to_float_series(_first_numeric_column(df)).dropna()

def _normalize_ohlc(df: _pd.DataFrame) -> _pd.DataFrame:
    high = _pick_col(df, "High").rename("High")
    low = _pick_col(df, "Low").rename("Low")
    close = _pick_col(df, "Close").rename("Close")
    out = _pd.concat([high, low, close], axis=1).dropna()
    out["High"] = out["High"].astype(float)
    out["Low"] = out["Low"].astype(float)
    out["Close"] = out["Close"].astype(float)
    return out

# -------------- indicators --------------

def _compute_rsi(series: _pd.Series, window: int = 14) -> _pd.Series:
    series = _to_float_series(series)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, _np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(100.0)

def _compute_atr(df: _pd.DataFrame, window: int = 14) -> _pd.Series:
    df = _normalize_ohlc(df)
    tr = _pd.concat(
        [
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def _compute_adx(df: _pd.DataFrame, window: int = 14) -> _pd.Series:
    df = _normalize_ohlc(df)
    high, low = df["High"], df["Low"]
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0; minus_dm[~mask] = 0
    tr = (high - low).abs()
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    atr = tr.rolling(window, min_periods=window).mean().replace(0.0, _np.nan)
    plus_di = 100 * (plus_dm.rolling(window, min_periods=window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window, min_periods=window).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window, min_periods=window).mean()

# -------------- strategy --------------

class SwingStrategy:
    def __init__(
        self,
        symbols: List[str],
        data_fetcher: Optional[Callable[[str], _pd.DataFrame]] = None,
        max_risk_per_trade: float = 1000.0,       # risk×ATR sizing cap
        regime_filter: bool = True,
        evaluation_interval: int = 86400,
        timezone: str = "Asia/Kolkata",
        max_positions: int = 5,                   # <= 5 concurrent names
        total_capital: float = 10000.0,           # informational
        max_invested: float = 5000.0,             # hard notional cap in market
    ) -> None:
        self.symbols = symbols
        self.data_fetcher = data_fetcher or self._default_fetcher
        self.max_risk_per_trade = max_risk_per_trade
        self.regime_filter = regime_filter
        self.evaluation_interval = evaluation_interval
        self.timezone = timezone

        self.max_positions = int(max_positions)
        self.total_capital = float(total_capital)
        self.max_invested = float(max_invested)

        self._open_positions: Dict[str, Dict[str, float]] = {}

    # -------- data --------

    def _default_fetcher(self, symbol: str) -> _pd.DataFrame:
        try:
            if _yf is None:
                raise RuntimeError("yfinance missing")
            df_raw = _yf.download(
                f"{symbol}.NS", period="250d", interval="1d",
                auto_adjust=False, progress=False
            )
            if isinstance(df_raw, _pd.DataFrame) and len(df_raw) > 0:
                df = _normalize_ohlc(df_raw)
                if len(df) > 0:
                    return df
            raise RuntimeError("empty/malformed")
        except Exception:
            idx = _pd.date_range(end=_pd.Timestamp.now(tz=self.timezone).normalize(), periods=250, freq="B")
            base = _np.linspace(100.0, 120.0, len(idx)) + _np.random.normal(0, 0.6, len(idx))
            high = base * (1.004 + _np.random.normal(0, 0.0008, len(idx)))
            low  = base * (0.996 + _np.random.normal(0, 0.0008, len(idx)))
            close = base
            return _pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx).astype(float)

    def _fetch_index_data(self) -> _pd.DataFrame:
        try:
            if _yf is None:
                raise RuntimeError("yfinance missing")
            df_raw = _yf.download("^NSEI", period="250d", interval="1d", auto_adjust=False, progress=False)
            if isinstance(df_raw, _pd.DataFrame) and len(df_raw) > 0:
                df = _normalize_ohlc(df_raw)
                if len(df) > 0:
                    return df
            raise RuntimeError("empty/malformed")
        except Exception:
            idx = _pd.date_range(end=_pd.Timestamp.now(tz=self.timezone).normalize(), periods=250, freq="B")
            trend = _np.linspace(18000.0, 22000.0, len(idx)) + _np.random.normal(0, 15.0, len(idx))
            high = trend * (1.003 + _np.random.normal(0, 0.0005, len(idx)))
            low  = trend * (0.997 + _np.random.normal(0, 0.0005, len(idx)))
            close = trend
            return _pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx).astype(float)

    # -------- regime --------

    def _is_risk_on(self) -> bool:
        if not self.regime_filter:
            return True
        df = self._fetch_index_data()
        if df is None or len(df) < 200:
            return True
        close = df["Close"]
        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
        a, b = _last_scalar(ema50), _last_scalar(ema200)
        if _np.isnan(a) or _np.isnan(b):
            return True
        return a > b

    # -------- portfolio helpers --------

    def _portfolio_notional(self) -> float:
        """Estimate current invested notional using last_close stored on positions."""
        total = 0.0
        for pos in self._open_positions.values():
            qty = int(pos.get("qty", 0))
            last_c = float(pos.get("last_close", 0.0))
            total += qty * last_c
        return float(total)

    # -------- entry/exit logic --------

    def _build_candidates(self) -> List[Dict[str, float]]:
        """
        Scan universe and return list of candidate dicts:
        {symbol, price, atr, stop, score}
        """
        import os as _os
        tol = 0.995 if _os.getenv("EASY_SIM", "0") == "1" else 1.0

        out: List[Dict[str, float]] = []
        for sym in self.symbols:
            if sym in self._open_positions:
                continue
            try:
                df = self.data_fetcher(sym)
                df = _normalize_ohlc(df)
            except Exception:
                continue
            if df is None or len(df) < 50:
                continue

            close = df["Close"]
            c = _last_scalar(close)
            if _np.isnan(c) or c <= 0:
                continue

            try:
                rsi14 = _compute_rsi(close, 14)
                rsi7  = _compute_rsi(close, 7)
                rsi21 = _compute_rsi(close, 21)
                atr   = _compute_atr(df, 14)
                adx   = _compute_adx(df, 14)
                rh20  = close.rolling(20, min_periods=20).max()
                rh50  = close.rolling(50, min_periods=50).max()
            except Exception:
                continue

            a14 = _last_scalar(atr); adx14 = _last_scalar(adx)
            r14 = _last_scalar(rsi14); r7 = _last_scalar(rsi7); r21 = _last_scalar(rsi21)
            rh20v = _last_scalar(rh20); rh50v = _last_scalar(rh50)
            if any(_np.isnan(x) for x in (a14, adx14, r14, r7, r21, rh20v, rh50v)):
                continue
            if a14 <= 0:
                continue

            cond_breakout = (c > tol * rh20v) or (c > tol * rh50v)
            cond_rsi = (r14 > 40.0) and (r7 > 40.0) and (r21 > 40.0)
            cond_adx = adx14 > 20.0
            if not (cond_breakout and cond_rsi and cond_adx):
                continue

            # score: breakout strength + trend strength + momentum
            eps = 1e-9
            breakout_ratio = max(c / max(rh20v, eps), c / max(rh50v, eps))
            score = 0.6 * breakout_ratio + 0.3 * (adx14 / 25.0) + 0.1 * (r14 / 50.0)

            out.append({
                "symbol": sym,
                "price": float(c),
                "atr": float(a14),
                "stop": float(c - 2.0 * a14),
                "score": float(score),
            })
        # strongest first
        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    def _evaluate_exit(self, df: _pd.DataFrame, pos: Dict[str, float]) -> bool:
        df = _normalize_ohlc(df)
        close = df["Close"]
        rsi14 = _compute_rsi(close, 14)
        sma20 = close.rolling(window=20, min_periods=20).mean()
        c = _last_scalar(close); r14 = _last_scalar(rsi14); s20 = _last_scalar(sma20)
        stop = float(pos.get("stop_price", -_np.inf))
        cond_stop = (not _np.isnan(c)) and (c <= stop)
        cond_sma  = (not _np.isnan(c) and not _np.isnan(s20)) and (c < s20)
        cond_rsi  = (not _np.isnan(r14)) and (r14 < 30.0)
        return bool(cond_stop or cond_sma or cond_rsi)

    # -------- main generator --------

    def generate_signals(self) -> Iterator[TradeSignal]:
        next_eval_time = 0.0
        interval = float(self.evaluation_interval)

        while True:
            now = time.time()
            if now >= next_eval_time:

                # 1) Exits & trailing updates first (free budget)
                for sym, pos in list(self._open_positions.items()):
                    try:
                        df = self.data_fetcher(sym)
                    except Exception:
                        continue
                    df_n = _normalize_ohlc(df)
                    c = _last_scalar(df_n["Close"])
                    a14 = _last_scalar(_compute_atr(df_n, 14))
                    if not _np.isnan(c):
                        pos["last_close"] = float(c)
                    if not _np.isnan(c) and not _np.isnan(a14):
                        new_stop = float(c - 2.0 * a14)
                        if new_stop > pos.get("stop_price", -_np.inf):
                            pos["stop_price"] = new_stop
                    if self._evaluate_exit(df_n, pos):
                        qty = int(pos["qty"])
                        yield TradeSignal(symbol=sym, side="SELL", quantity=qty)
                        self._open_positions.pop(sym, None)

                # 2) Risk-off purge
                if not self._is_risk_on():
                    for sym, pos in list(self._open_positions.items()):
                        qty = int(pos["qty"])
                        yield TradeSignal(symbol=sym, side="SELL", quantity=qty)
                        self._open_positions.pop(sym, None)
                else:
                    # 3) Entries (respect caps)
                    open_count = len(self._open_positions)
                    slots = max(0, self.max_positions - open_count)
                    if slots > 0:
                        invested = self._portfolio_notional()
                        remaining_budget = max(0.0, self.max_invested - invested)
                        if remaining_budget > 0.0:
                            per_slot_budget = self.max_invested / float(self.max_positions)

                            candidates = self._build_candidates()
                            for cand in candidates:
                                if slots == 0:
                                    break
                                if remaining_budget <= 0:
                                    break

                                sym = cand["symbol"]; price = cand["price"]; atr = cand["atr"]; stop = cand["stop"]

                                # cash sizing
                                budget_for_this = min(per_slot_budget, remaining_budget)
                                qty_cash = int(budget_for_this // max(price, 1e-9))

                                # risk sizing (ATR stop)
                                qty_risk = int(max(1.0, self.max_risk_per_trade / max(atr, 1e-9)))

                                qty = max(1, min(qty_cash, qty_risk))
                                notional = qty * price

                                if qty < 1 or notional <= 0:
                                    continue
                                if notional > remaining_budget + 1e-6:
                                    qty = int(remaining_budget // max(price, 1e-9))
                                    if qty < 1:
                                        continue
                                    notional = qty * price

                                # open
                                self._open_positions[sym] = {
                                    "qty": int(qty),
                                    "stop_price": float(stop),
                                    "last_close": float(price),
                                }
                                yield TradeSignal(symbol=sym, side="BUY", quantity=int(qty))

                                remaining_budget -= notional
                                slots -= 1

                next_eval_time = now + interval

            time.sleep(1.0)

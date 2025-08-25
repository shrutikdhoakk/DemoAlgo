"""
swing_strategy.py — top-5 selection + portfolio cap
- Scans a universe, ranks candidates by breakout/ADX/RSI score
- Limits open positions to max_positions
- Enforces portfolio notional cap (max_invested) from total capital
- Sizes per position by BOTH cash budget and ATR risk
- Confirms breakouts with volume and uses ATR-based profit targets
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Iterator, List, Optional

import numpy as _np
import pandas as _pd

try:
    import yfinance as _yf
except ImportError:
    _yf = None

from trade_signal import TradeSignal
from performance import PerformanceTracker  # NEW


# -------------- helpers --------------

def _to_float_series(s: _pd.Series) -> _pd.Series:
    return _pd.to_numeric(s, errors="coerce").astype(float)


def _last_scalar(x) -> float:
    if isinstance(x, _pd.Series):
        return float(x.iloc[-1]) if len(x) else _np.nan
    try:
        return float(x)
    except Exception:
        return _np.nan


def _first_numeric_column(df: _pd.DataFrame) -> _pd.Series:
    """Return first numeric-like column as Series (robust to MultiIndex)."""
    if df is None or df.empty:
        return _pd.Series(dtype=float)

    # If MultiIndex columns, flatten candidates
    cols = list(df.columns)
    for col in cols:
        s = df[col]
        if isinstance(s, _pd.DataFrame):
            # take first subcolumn if needed
            s = s.iloc[:, 0]
        s_num = _pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            return s_num
    # Fallback: take first column directly coerced
    first = df.iloc[:, 0]
    if isinstance(first, _pd.DataFrame):
        first = first.iloc[:, 0]
    return _pd.to_numeric(first, errors="coerce")


def _pick_col(df: _pd.DataFrame, name: str) -> _pd.Series:
    """Best-effort extraction of OHLCV columns from varied DataFrame shapes."""
    if df is None or df.empty:
        return _pd.Series(dtype=float)

    lname = str(name).lower()

    # Simple columns
    if lname in [str(c).lower() for c in df.columns if not isinstance(c, tuple)] and not isinstance(df.columns, _pd.MultiIndex):
        return _to_float_series(df[name]).dropna()

    # Try a few common aliases
    lowmap = {
        "open": ["Open", "open", "OPEN", "o"],
        "high": ["High", "high", "HIGH", "h"],
        "low": ["Low", "low", "LOW", "l"],
        "close": ["Close", "close", "CLOSE", "Adj Close", "adj close", "c"],
        "volume": ["Volume", "volume", "VOLUME", "vol", "Vol"],
    }
    if lname in lowmap:
        for cand in lowmap[lname]:
            if cand in df.columns:
                return _to_float_series(df[cand]).dropna()

    # MultiIndex top-level direct match
    if isinstance(df.columns, _pd.MultiIndex):
        top = [str(x).lower() for x in df.columns.get_level_values(0)]
        if lname in top:
            sub = df.xs(key=name, axis=1, level=0, drop_level=False)
            s = sub.iloc[:, 0] if isinstance(sub, _pd.DataFrame) else sub
            if isinstance(s, _pd.DataFrame):
                s = s.iloc[:, 0]
            return _to_float_series(s).dropna()

        # Search any level
        for col in df.columns:
            parts = col if isinstance(col, tuple) else (col, )
            if any(str(p).lower() == lname for p in parts):
                s = df[col]
                if isinstance(s, _pd.DataFrame):
                    s = s.iloc[:, 0]
                return _to_float_series(s).dropna()

    # Last resort: first numeric column
    return _to_float_series(_first_numeric_column(df)).dropna()


def _normalize_ohlc(df: _pd.DataFrame) -> _pd.DataFrame:
    """Return DataFrame with float columns: High, Low, Close, and (if present) Volume."""
    high = _pick_col(df, "High").rename("High")
    low = _pick_col(df, "Low").rename("Low")
    close = _pick_col(df, "Close").rename("Close")

    cols = [high, low, close]

    # try volume if available
    vol = None
    try:
        vol = _pick_col(df, "Volume").rename("Volume")
        if len(vol):
            cols.append(vol)
    except Exception:
        pass

    out = _pd.concat(cols, axis=1).dropna()
    if out.empty:
        # Ensure columns exist to avoid key errors downstream
        out = _pd.DataFrame({"High": [], "Low": [], "Close": []})
        if vol is not None:
            out["Volume"] = []

    out["High"] = _pd.to_numeric(out["High"], errors="coerce").astype(float)
    out["Low"] = _pd.to_numeric(out["Low"], errors="coerce").astype(float)
    out["Close"] = _pd.to_numeric(out["Close"], errors="coerce").astype(float)
    if "Volume" in out.columns:
        out["Volume"] = _pd.to_numeric(out["Volume"], errors="coerce").astype(float)
    return out


# -------------- indicators --------------

def _compute_rsi(series: _pd.Series, window: int = 14) -> _pd.Series:
    series = _to_float_series(series)
    if series.empty:
        return _pd.Series(dtype=float)
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
    if df.empty:
        return _pd.Series(dtype=float)
    tr = _pd.concat(
        [
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return atr


def _compute_adx(df: _pd.DataFrame, window: int = 14) -> _pd.Series:
    """Classic Wilder ADX implementation using rolling means."""
    df = _normalize_ohlc(df)
    if df.empty:
        return _pd.Series(dtype=float)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0.0
    minus_dm[minus_dm < 0] = 0.0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0.0
    minus_dm[~mask] = 0.0

    tr = _pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing via rolling mean (approximation)
    atr = tr.rolling(window, min_periods=window).mean().replace(0.0, _np.nan)
    plus_di = 100.0 * (plus_dm.rolling(window, min_periods=window).sum() / atr)
    minus_di = 100.0 * (minus_dm.rolling(window, min_periods=window).sum() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100.0
    adx = dx.rolling(window, min_periods=window).mean()
    return adx


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
        regime_rsi_min: float = 50.0,             # index RSI threshold
        performance_tracker: Optional[PerformanceTracker] = None,  # NEW
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
        self.regime_rsi_min = float(regime_rsi_min)

        self.tracker = performance_tracker or PerformanceTracker()  # NEW

        # position state: sym -> {"qty", "stop_price", "target_price", "last_close"}
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
            # synthetic fallback for offline/SIM
            idx = _pd.date_range(
                end=_pd.Timestamp.now(tz=self.timezone).normalize(),
                periods=250, freq="B"
            )
            base = _np.linspace(100.0, 120.0, len(idx)) + _np.random.normal(0, 0.6, len(idx))
            high = base * (1.004 + _np.random.normal(0, 0.0008, len(idx)))
            low = base * (0.996 + _np.random.normal(0, 0.0008, len(idx)))
            close = base
            vol = _np.random.uniform(1e5, 1e6, len(idx))
            return _pd.DataFrame(
                {"High": high, "Low": low, "Close": close, "Volume": vol},
                index=idx
            ).astype(float)

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
            idx = _pd.date_range(
                end=_pd.Timestamp.now(tz=self.timezone).normalize(),
                periods=250, freq="B"
            )
            trend = _np.linspace(18000.0, 22000.0, len(idx)) + _np.random.normal(0, 15.0, len(idx))
            high = trend * (1.003 + _np.random.normal(0, 0.0005, len(idx)))
            low = trend * (0.997 + _np.random.normal(0, 0.0005, len(idx)))
            close = trend
            vol = _np.random.uniform(1e6, 5e6, len(idx))
            return _pd.DataFrame(
                {"High": high, "Low": low, "Close": close, "Volume": vol},
                index=idx
            ).astype(float)

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
        rsi_idx = _compute_rsi(close, 14)
        a, b = _last_scalar(ema50), _last_scalar(ema200)
        r = _last_scalar(rsi_idx)
        if _np.isnan(a) or _np.isnan(b) or _np.isnan(r):
            return True
        return (a > b) and (r > self.regime_rsi_min)

    # -------- portfolio helpers --------

    def _portfolio_notional(self) -> float:
        """Estimate current invested notional using last_close stored on positions."""
        total = 0.0
        for pos in self._open_positions.values():
            qty = int(pos.get("qty", 0))
            last_c = float(pos.get("last_close", 0.0))
            total += qty * last_c
        return float(total)

    # -------- tracker helper --------

    def _record_trade(self, symbol: str, side: str, qty: int, price: float) -> None:
        """Send trade to tracker safely."""
        try:
            self.tracker.record(TradeSignal(symbol=symbol, side=side, quantity=int(qty)), float(price))
        except Exception:
            pass

    # -------- entry/exit logic --------

    def _build_candidates(self) -> List[Dict[str, float]]:
        """
        Scan universe and return list of candidate dicts:
        {symbol, price, atr, stop, target, score}
        """
        tol = 0.995 if os.getenv("EASY_SIM", "0") == "1" else 1.0

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

            vol = df.get("Volume")
            if vol is None or len(vol) < 20:
                continue
            v_curr = _last_scalar(vol)
            v_avg = _last_scalar(vol.rolling(20, min_periods=20).mean())
            if _np.isnan(v_curr) or _np.isnan(v_avg) or v_curr <= 1.5 * v_avg:
                continue

            try:
                rsi14 = _compute_rsi(close, 14)
                rsi7 = _compute_rsi(close, 7)
                rsi21 = _compute_rsi(close, 21)
                atr = _compute_atr(df, 14)
                adx = _compute_adx(df, 14)
                rh20 = close.rolling(20, min_periods=20).max()
                rh50 = close.rolling(50, min_periods=50).max()
            except Exception:
                continue

            a14 = _last_scalar(atr)
            adx14 = _last_scalar(adx)
            r14 = _last_scalar(rsi14)
            r7 = _last_scalar(rsi7)
            r21 = _last_scalar(rsi21)
            rh20v = _last_scalar(rh20)
            rh50v = _last_scalar(rh50)
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

            out.append(
                {
                    "symbol": sym,
                    "price": float(c),
                    "atr": float(a14),
                    "stop": float(c - 2.0 * a14),
                    "target": float(c + 2.0 * a14),
                    "score": float(score),
                }
            )
        # strongest first
        out.sort(key=lambda d: d["score"], reverse=True)
        return out

    def _evaluate_exit(self, df: _pd.DataFrame, pos: Dict[str, float]) -> bool:
        df = _normalize_ohlc(df)
        if df.empty:
            return False
        close = df["Close"]
        rsi14 = _compute_rsi(close, 14)
        sma20 = close.rolling(window=20, min_periods=20).mean()
        c = _last_scalar(close)
        r14 = _last_scalar(rsi14)
        s20 = _last_scalar(sma20)
        stop = float(pos.get("stop_price", -_np.inf))
        target = float(pos.get("target_price", _np.inf))

        cond_stop = (not _np.isnan(c)) and (c <= stop)
        cond_target = (not _np.isnan(c)) and (c >= target)
        cond_sma = (not _np.isnan(c) and not _np.isnan(s20)) and (c < s20)
        cond_rsi = (not _np.isnan(r14)) and (r14 < 30.0)
        return bool(cond_stop or cond_target or cond_sma or cond_rsi)

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
                        df_n = _normalize_ohlc(df)
                    except Exception:
                        continue

                    if df_n.empty:
                        continue

                    c = _last_scalar(df_n["Close"])
                    a14 = _last_scalar(_compute_atr(df_n, 14))
                    if not _np.isnan(c):
                        pos["last_close"] = float(c)
                    if not _np.isnan(c) and not _np.isnan(a14):
                        new_stop = float(c - 2.0 * a14)
                        if new_stop > pos.get("stop_price", -_np.inf):
                            pos["stop_price"] = new_stop
                        new_target = float(c + 2.0 * a14)
                        if new_target > pos.get("target_price", -_np.inf):
                            pos["target_price"] = new_target

                    if self._evaluate_exit(df_n, pos):
                        qty = int(pos["qty"])
                        # record SELL with current close (best available)
                        sell_price = float(pos.get("last_close", c if not _np.isnan(c) else 0.0))
                        self._record_trade(sym, "SELL", qty, sell_price)  # NEW
                        yield TradeSignal(symbol=sym, side="SELL", quantity=qty)
                        self._open_positions.pop(sym, None)

                # 2) Risk-off purge
                if not self._is_risk_on():
                    for sym, pos in list(self._open_positions.items()):
                        qty = int(pos["qty"])
                        price = float(pos.get("last_close", 0.0))
                        self._record_trade(sym, "SELL", qty, price)  # NEW
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
                            candidates = self._build_candidates()
                            selected = candidates[:slots]
                            total_score = sum(c.get("score", 0.0) for c in selected) or 1.0
                            budget_pool = remaining_budget

                            for cand in selected:
                                if slots == 0 or remaining_budget <= 0:
                                    break

                                sym = cand["symbol"]
                                price = cand["price"]
                                atr = cand["atr"]
                                stop = cand["stop"]
                                target = cand["target"]
                                score = cand.get("score", 0.0)

                                # cash sizing (proportional to score)
                                budget_for_this = min(budget_pool * (score / total_score), remaining_budget)
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
                                    "target_price": float(target),
                                    "last_close": float(price),
                                }
                                # record BUY at entry price
                                self._record_trade(sym, "BUY", int(qty), float(price))  # NEW
                                yield TradeSignal(symbol=sym, side="BUY", quantity=int(qty))

                                remaining_budget -= notional
                                slots -= 1

                next_eval_time = now + interval

            time.sleep(1.0)

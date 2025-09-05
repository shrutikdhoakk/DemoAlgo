# backtesting/patterns.py
from __future__ import annotations
import numpy as np
import pandas as pd

# --- 1) Volatility squeeze via Bollinger Bandwidth percentile ---
def bb_squeeze(df: pd.DataFrame, lookback: int = 20, pct: float = 0.10) -> pd.Series:
    """
    Returns a boolean Series: True when current BB bandwidth is in the bottom `pct` percentile
    of the last `lookback` observations (tight consolidation).
    Requires columns: 'close'
    """
    c = df["close"].astype(float)
    ma = c.rolling(lookback).mean()
    sd = c.rolling(lookback).std(ddof=0)
    upper = ma + 2*sd
    lower = ma - 2*sd
    bbw = (upper - lower) / ma.replace(0, np.nan)
    # rolling percentile threshold
    thr = bbw.rolling(lookback).apply(lambda x: np.nanpercentile(x, pct*100), raw=True)
    return (bbw <= thr).fillna(False)

# --- 2) Simple flag/triangle proxy using range compression + HH/HL structure ---
def consolidation_breakout_signal(df: pd.DataFrame,
                                  base_lb: int = 15,
                                  compress_pct: float = 0.35,
                                  breakout_lookback: int = 20) -> pd.Series:
    """
    Heuristic:
      - Recent N bars show compressed range vs prior N (flags/triangles often compress).
      - Current close breaks above recent highs => breakout signal.
    Returns boolean Series True on bars that satisfy the breakout condition.
    Requires columns: 'high','low','close'
    """
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    rng = (h - l)
    recent = rng.rolling(base_lb).mean()
    prior = rng.shift(base_lb).rolling(base_lb).mean()
    compression = (recent / prior.replace(0, np.nan)) <= compress_pct

    hh = h.rolling(breakout_lookback).max().shift(1)  # yesterday's 20-bar high
    breakout = c > hh

    return (compression & breakout).fillna(False)

# --- 3) Bullish candlestick confirmation (basic set) ---
def bullish_candle(df: pd.DataFrame) -> pd.Series:
    """
    Flags a basic bullish body (close > open).
    Requires 'open','close'
    """
    o, c = df["open"].astype(float), df["close"].astype(float)
    return (c > o)

def hammer(df: pd.DataFrame, body_pct_max: float = 0.35, lower_shadow_min: float = 2.0) -> pd.Series:
    """
    Hammer: small body near top, long lower shadow.
    """
    o, h, l, c = [df[x].astype(float) for x in ("open","high","low","close")]
    body = abs(c - o)
    range_ = (h - l).replace(0, np.nan)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(c, o) - l
    body_small = (body / range_) <= body_pct_max
    long_lower = (lower_shadow / (body.replace(0, np.nan))) >= lower_shadow_min
    body_top = upper_shadow <= (0.35 * range_)
    return (body_small & long_lower & body_top).fillna(False)

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Engulfing: prior red, current green, current real body engulfs prior body.
    """
    o, c = df["open"].astype(float), df["close"].astype(float)
    o1, c1 = o.shift(1), c.shift(1)
    prior_red = c1 < o1
    curr_green = c > o
    engulfs = (c >= o1) & (o <= c1)
    return (prior_red & curr_green & engulfs).fillna(False)

def morning_star(df: pd.DataFrame) -> pd.Series:
    """
    Very simple 3-bar Morning Star heuristic:
      red candle -> small body gap/down -> strong green close near/above mid of first body
    """
    o, c = df["open"].astype(float), df["close"].astype(float)
    o1, c1 = o.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)

    red1 = c2 < o2
    small2 = (abs(c1 - o1) <= 0.4 * abs(c2 - o2))
    gap_down = o1 < c2  # crude
    strong3 = c > (o2 + c2) / 2
    green3 = c > o
    return (red1 & small2 & gap_down & strong3 & green3).fillna(False)

def bullish_confirmation(df: pd.DataFrame) -> pd.Series:
    """
    Combine any bullish candle signal.
    """
    return bullish_candle(df) | hammer(df) | bullish_engulfing(df) | morning_star(df)

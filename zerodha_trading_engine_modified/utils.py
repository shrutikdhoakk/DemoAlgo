"""
utils.py
---------

Helper functions for calculations, logging setup, rate limiting and ATR.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from typing import Deque, Iterable, Tuple


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger that writes to both stdout and a file.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str
        Path to the log file.
    level : int
        Logging level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create logs directory if necessary
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # avoid duplicate logs
    logger.propagate = False
    return logger


class RateLimiter:
    """Simple token bucket rate limiter for API calls."""

    def __init__(self, max_calls: int, period: float) -> None:
        self.max_calls = max_calls
        self.period = period
        self.calls: Deque[float] = deque()

    def acquire(self) -> None:
        """Block until a new call can be made."""
        while True:
            now = time.monotonic()
            # remove expired timestamps
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            # sleep until earliest call expires
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)


def calculate_atr(highs: Iterable[float], lows: Iterable[float], closes: Iterable[float], period: int = 14) -> float:
    """
    Compute the Average True Range (ATR) using a simple formula.
    For demonstration we compute ATR over the last `period` bars.

    Parameters
    ----------
    highs, lows, closes : Iterable[float]
        The high, low and close prices.  Must be at least length `period`.
    period : int
        Lookback period for ATR.

    Returns
    -------
    float
        The computed ATR.
    """
    highs = list(highs)[-period:]
    lows = list(lows)[-period:]
    closes = list(closes)[-period:]
    trs = []
    for i in range(len(highs)):
        if i == 0:
            prev_close = closes[0]
        else:
            prev_close = closes[i-1]
        tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    return sum(trs) / len(trs)
"""
risk_engine.py
----------------

Implements pre-trade gates, exposure calculations and risk limits
enforcement. This module encapsulates all checks that must pass
before an order is allowed to hit the broker. It also tracks
portfolio exposures in memory for real-time supervision.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

from config import AppConfig
from utils import RateLimiter


class RiskEngine:
    """
    Pre-trade risk engine enforcing SEBI-aligned controls.

    The engine maintains two exposure maps:
    - `exposures_notional` stores the rupee notional exposure per symbol.
    - `positions_qty` stores the net quantity position per symbol.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        # rupee exposures keyed by symbol
        self.exposures_notional: Dict[str, float] = {}
        # net quantity positions keyed by symbol
        self.positions_qty: Dict[str, int] = {}
        # order-to-trade ratio tracking
        self.order_count: int = 0
        self.trade_count: int = 0
        # simple rate limiter based on API limits
        self.rate_limiter = RateLimiter(
            max_calls=self.cfg.broker.rate_limit_per_sec,
            period=1.0
        )

    # ------------------------------------------------------------------ #
    # Position/exposure bookkeeping
    # ------------------------------------------------------------------ #
    def update_position(self, symbol: str, notional: float, qty: int) -> None:
        """
        Update internal exposures after a fill.

        notional: signed rupee notional (qty × price). + for buy, - for sell.
        qty: signed shares. + for buy, - for sell.
        """
        try:
            n = float(notional)
        except Exception:
            n = 0.0
        try:
            q = int(qty)
        except Exception:
            q = 0

        self.exposures_notional[symbol] = float(self.exposures_notional.get(symbol, 0.0) + n)
        self.positions_qty[symbol] = int(self.positions_qty.get(symbol, 0) + q)

    def get_exposure(self, symbol: str) -> float:
        return float(self.exposures_notional.get(symbol, 0.0))

    def get_position_qty(self, symbol: str) -> int:
        return int(self.positions_qty.get(symbol, 0))

    # ------------------------------------------------------------------ #
    # Checks
    # ------------------------------------------------------------------ #
    def _check_exposure_limits(self, symbol: str, notional_f: float) -> Tuple[bool, str]:
        """Check per-symbol and portfolio exposure limits."""
        per_symbol_cap = float(getattr(self.cfg.risk, "per_symbol_cap", 0.0))
        max_portfolio_value = float(getattr(self.cfg.risk, "max_portfolio_value", 0.0))

        existing = abs(float(self.exposures_notional.get(symbol, 0.0)))
        if per_symbol_cap > 0.0 and existing + abs(notional_f) > per_symbol_cap:
            return False, (
                f"per_symbol_cap exceeded for {symbol}: "
                f"{existing + abs(notional_f)} > {per_symbol_cap}"
            )
        total = sum(abs(float(v)) for v in self.exposures_notional.values())
        if max_portfolio_value > 0.0 and total + abs(notional_f) > max_portfolio_value:
            return False, (
                f"max_portfolio_value exceeded: "
                f"{total + abs(notional_f)} > {max_portfolio_value}"
            )
        return True, ""

    def _check_price_band(self, ltp_f: float, limit_f: float) -> Tuple[bool, str]:
        """
        Check that the limit price is within the configured price band buffer
        around LTP (in basis points).
        """
        if not (math.isfinite(ltp_f) and math.isfinite(limit_f)) or ltp_f <= 0.0 or limit_f <= 0.0:
            return False, f"invalid prices (ltp={ltp_f}, limit={limit_f})"

        buffer_bps = float(getattr(self.cfg.risk, "price_band_buffer_bps", 0.0))
        upper = ltp_f * (1.0 + buffer_bps / 10000.0)
        lower = ltp_f * (1.0 - buffer_bps / 10000.0)

        # Use scalar comparisons only
        if limit_f > upper or limit_f < lower:
            return False, (
                f"limit price {limit_f} outside [{lower}, {upper}] "
                f"(ltp={ltp_f}, buffer={buffer_bps}bps)"
            )
        return True, ""

    def _check_symbol_whitelist(self, symbol: str) -> Tuple[bool, str]:
        """Ensure the symbol is in the configured symbols list."""
        whitelist = [s.split(":")[-1] for s in self.cfg.marketdata.symbols]
        if symbol not in whitelist:
            return False, f"symbol {symbol} not in whitelist {whitelist}"
        return True, ""

    def _check_rate_limit(self) -> Tuple[bool, str]:
        """Throttle API calls according to broker limits."""
        try:
            self.rate_limiter.acquire()
            return True, ""
        except Exception as e:
            return False, f"rate limit error: {e}"

    # ------------------------------------------------------------------ #
    # Public gate
    # ------------------------------------------------------------------ #
    def pre_trade_checks(self, symbol: str, notional, ltp, limit_price) -> Tuple[bool, str]:
        """
        Run all pre-trade checks; return (allowed, reason).
        All numeric inputs are coerced to floats to avoid pandas Series ambiguity.
        """
        # Coerce to scalars (avoid pandas Series / numpy arrays)
        try:
            notional_f = float(notional if notional is not None else 0.0)
        except Exception:
            notional_f = 0.0
        try:
            ltp_f = float(ltp if ltp is not None else 0.0)
        except Exception:
            ltp_f = 0.0
        try:
            limit_f = float(limit_price if limit_price is not None else 0.0)
        except Exception:
            limit_f = 0.0

        if ltp_f <= 0.0:
            return False, "Invalid LTP (<= 0)."
        if limit_f <= 0.0:
            return False, "Invalid limit price (<= 0)."
        if notional_f <= 0.0:
            return False, "Invalid notional (<= 0)."

        # symbol whitelist
        ok, reason = self._check_symbol_whitelist(symbol)
        if not ok:
            return False, reason

        # exposure limits
        ok, reason = self._check_exposure_limits(symbol, notional_f)
        if not ok:
            return False, reason

        # price band validation
        ok, reason = self._check_price_band(ltp_f, limit_f)
        if not ok:
            return False, reason

        # rate limit
        ok, reason = self._check_rate_limit()
        if not ok:
            return False, reason

        return True, ""

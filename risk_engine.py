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
from typing import Dict, Tuple, Optional

from config import AppConfig
from utils import RateLimiter


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


class RiskEngine:
    """
    Pre-trade risk engine enforcing SEBI-aligned controls.

    Maps maintained in-memory:
    - `exposures_notional`: rupee notional per symbol (signed)
    - `positions_qty`: net quantity per symbol (signed)
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self.exposures_notional: Dict[str, float] = {}
        self.positions_qty: Dict[str, int] = {}
        self.order_count: int = 0
        self.trade_count: int = 0
        self.rate_limiter = RateLimiter(
            max_calls=self.cfg.broker.rate_limit_per_sec,
            period=1.0
        )

    # ------------------------------------------------------------------ #
    # Position/exposure bookkeeping
    # ------------------------------------------------------------------ #
    def update_position(self, symbol: str, notional: float, qty: int) -> None:
        """
        Update internal exposures after a *fill*.

        notional: signed rupee amount (qty × price). + for buy, - for sell.
        qty: signed shares. + for buy, - for sell.
        """
        n = _to_float(notional)
        q = _to_int(qty)
        self.exposures_notional[symbol] = float(self.exposures_notional.get(symbol, 0.0) + n)
        self.positions_qty[symbol] = int(self.positions_qty.get(symbol, 0) + q)

    def get_exposure(self, symbol: str) -> float:
        return float(self.exposures_notional.get(symbol, 0.0))

    def get_position_qty(self, symbol: str) -> int:
        return int(self.positions_qty.get(symbol, 0))

    # ------------------------------------------------------------------ #
    # Convenience helpers for order flow
    # ------------------------------------------------------------------ #
    def calc_signed_notional(self, side: str, price, qty) -> float:
        """Return signed notional given side/price/qty."""
        p = _to_float(price)
        q = _to_int(qty)
        if p <= 0 or q <= 0:
            return 0.0
        sign = 1.0 if str(side).upper() == "BUY" else -1.0
        return sign * p * q

    def seats_available(self, max_positions: int) -> bool:
        """
        Returns True if opening a new symbol would not exceed `max_positions`.
        A symbol with zero qty is not counted as open.
        """
        open_syms = sum(1 for q in self.positions_qty.values() if int(q) != 0)
        return open_syms < int(max_positions)

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
        Ensure limit price is within a band around LTP (bps buffer).
        """
        if not (math.isfinite(ltp_f) and math.isfinite(limit_f)) or ltp_f <= 0.0 or limit_f <= 0.0:
            return False, f"invalid prices (ltp={ltp_f}, limit={limit_f})"

        buffer_bps = float(getattr(self.cfg.risk, "price_band_buffer_bps", 0.0))
        upper = ltp_f * (1.0 + buffer_bps / 10000.0)
        lower = ltp_f * (1.0 - buffer_bps / 10000.0)

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
    # Public gates
    # ------------------------------------------------------------------ #
    def pre_trade_checks(
        self,
        symbol: str,
        notional,
        ltp,
        limit_price,
    ) -> Tuple[bool, str]:
        """
        Run all pre-trade checks; return (allowed, reason).
        All numeric inputs are coerced to floats to avoid pandas/NumPy ambiguity.
        """
        notional_f = _to_float(notional)
        ltp_f = _to_float(ltp)
        limit_f = _to_float(limit_price)

        if ltp_f <= 0.0:
            return False, "Invalid LTP (<= 0)."
        if limit_f <= 0.0:
            return False, "Invalid limit price (<= 0)."
        if notional_f == 0.0:
            return False, "Invalid notional (== 0)."

        ok, reason = self._check_symbol_whitelist(symbol)
        if not ok:
            return False, reason

        ok, reason = self._check_exposure_limits(symbol, notional_f)
        if not ok:
            return False, reason

        ok, reason = self._check_price_band(ltp_f, limit_f)
        if not ok:
            return False, reason

        ok, reason = self._check_rate_limit()
        if not ok:
            return False, reason

        return True, ""

    def allow_order(
        self,
        symbol: str,
        side: str,
        qty,
        ltp,
        limit_price,
        *,
        check_positions_cap: Optional[int] = None,
    ) -> Tuple[bool, str, float]:
        """
        Convenience wrapper for order managers.

        Returns (allowed, reason, signed_notional).

        - Computes signed notional from side/ltp/qty.
        - Applies pre-trade checks.
        - Optionally enforces a max open positions cap (`check_positions_cap`).
        """
        qty_i = _to_int(qty)
        if qty_i <= 0:
            return False, "qty must be > 0", 0.0

        ltp_f = _to_float(ltp)
        limit_f = _to_float(limit_price)
        signed_notional = self.calc_signed_notional(side, ltp_f, qty_i)

        # optional cap on number of concurrent symbols
        if check_positions_cap is not None and not self.seats_available(check_positions_cap):
            return False, f"max open positions reached ({check_positions_cap})", signed_notional

        ok, reason = self.pre_trade_checks(symbol, signed_notional, ltp_f, limit_f)
        return ok, reason, signed_notional

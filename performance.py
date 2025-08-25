from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from trade_signal import TradeSignal


@dataclass
class TradeRecord:
    symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float


class PerformanceTracker:
    """Track trade entries/exits and basic performance metrics."""

    def __init__(self) -> None:
        # Open positions keyed by symbol
        self._open: Dict[str, Dict[str, float]] = {}
        # Closed trades history
        self._closed: List[TradeRecord] = []

    def record(self, signal: TradeSignal, price: float) -> None:
        """Record a TradeSignal with its execution price."""
        side = signal.side.upper()
        sym = signal.symbol
        qty = int(signal.quantity)
        px = float(price)

        if side == "BUY":
            self._open[sym] = {"entry_price": px, "quantity": qty}
        elif side == "SELL":
            entry = self._open.pop(sym, None)
            if entry:
                entry_price = float(entry["entry_price"])
                pnl = (px - entry_price) * qty
                rec = TradeRecord(
                    symbol=sym,
                    quantity=qty,
                    entry_price=entry_price,
                    exit_price=px,
                    pnl=pnl,
                )
                self._closed.append(rec)
                self._print_latest(rec)

    def success_rate(self) -> float:
        """Return percentage of profitable closed trades."""
        total = len(self._closed)
        if total == 0:
            return 0.0
        wins = sum(1 for t in self._closed if t.pnl > 0)
        return 100.0 * wins / total

    def _print_latest(self, trade: TradeRecord) -> None:
        rate = self.success_rate()
        print(
            f"[performance] {trade.symbol} qty={trade.quantity} pnl={trade.pnl:.2f} success_rate={rate:.2f}%",
            flush=True,
        )

    def trades(self) -> List[TradeRecord]:
        return list(self._closed)

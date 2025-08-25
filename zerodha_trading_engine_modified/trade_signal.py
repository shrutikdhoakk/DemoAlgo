# trade_signal.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeSignal:
    symbol: str
    side: str            # "BUY" or "SELL"
    quantity: int
    time_in_force: str = "DAY"
    limit_price: Optional[float] = None
    algo_id: Optional[str] = None

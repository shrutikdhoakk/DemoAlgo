# backtesting/backtest5y_05092025.py
import os
import pathlib
import sys
from datetime import date

# Ensure repo root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# ---- Fixed 5-year window ending on 2025-09-05 ----
BT_END = "2025-09-05"
BT_START = "2020-09-01"  # small buffer before exact 5y to warm indicators

# Environment knobs (tweak as needed)
DEFAULTS = {
    # Date range
    "BT_START": BT_START,
    "BT_END": BT_END,

    # Universe sizing (keep reasonable to avoid Yahoo rate limits)
    "UNIVERSE_LIMIT": "150",       # try 150–300

    # Risk/Reward via ATR
    "ATR_STOP_MULT": "1.2",        # ~1.2 ATR stop
    "ATR_TGT_MULT": "3.0",         # ~3.0 ATR target

    # Capital & risk (only if your backtester reads these)
    "TOTAL_CAPITAL": "1000000",    # ₹10L starting capital
    "MAX_INVESTED": "800000",      # cap invested notional (₹8L)
    "MAX_POSITIONS": "10",         # max concurrent positions
    "PER_TRADE_RISK": "10000",     # risk per trade (₹10k)

    # Symbols handling (common for NSE lists)
    "APPEND_NS": "1",              # append ".NS" if missing

    # Optional: if your backtester supports a run label/prefix for output files
    "RUN_LABEL": "backtest5y_05092025",
}

for k, v in DEFAULTS.items():
    os.environ.setdefault(k, v)

# Import the backtest entrypoint
from backtesting import backtest_quick02 as bt  # type: ignore

def _echo_config(keys):
    print("\n=== Backtest Config (5Y) ===")
    for k in keys:
        print(f"{k:>16}: {os.getenv(k)}")
    print("============================\n")

if __name__ == "__main__":
    _echo_config([
        "BT_START", "BT_END",
        "UNIVERSE_LIMIT",
        "ATR_STOP_MULT", "ATR_TGT_MULT",
        "TOTAL_CAPITAL", "MAX_INVESTED", "MAX_POSITIONS", "PER_TRADE_RISK",
        "APPEND_NS", "RUN_LABEL"
    ])
    bt.main()

# File: backtesting/pattern_filters.py
from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from .backtesting import patterns

__all__ = ["pattern_gate"]


def pattern_gate(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate optional technical patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Price history with columns for open, high, low and close. Column
        names are case-insensitive.

    Returns
    -------
    Dict[str, Any]
        ``{"ok": bool, "bonus": int, "flags": dict, "why": str}``
    """
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]

    min_bonus = int(os.getenv("PAT_MIN_BONUS", "2"))
    squeeze_lb = int(os.getenv("PAT_SQUEEZE_LB", "20"))
    squeeze_pct = float(os.getenv("PAT_SQUEEZE_PCTL", "5")) / 100.0
    tri_window = int(os.getenv("PAT_TRI_WINDOW", "30"))
    tri_shrink = float(os.getenv("PAT_TRI_SHRINK", "25")) / 100.0

    flags: Dict[str, bool] = {}
    notes = []

    try:
        squeeze = patterns.bb_squeeze(data, lookback=squeeze_lb, pct=squeeze_pct)
        flags["squeeze"] = bool(squeeze.iloc[-1]) if len(squeeze) else False
    except Exception as exc:  # pragma: no cover - defensive
        flags["squeeze"] = False
        notes.append(f"squeeze error: {exc}")

    try:
        tri = patterns.consolidation_breakout_signal(
            data, base_lb=tri_window, compress_pct=tri_shrink
        )
        flags["triangle"] = bool(tri.iloc[-1]) if len(tri) else False
    except Exception as exc:  # pragma: no cover - defensive
        flags["triangle"] = False
        notes.append(f"triangle error: {exc}")

    try:
        bull = patterns.bullish_confirmation(data)
        flags["bullish"] = bool(bull.iloc[-1]) if len(bull) else False
    except Exception as exc:  # pragma: no cover - defensive
        flags["bullish"] = False
        notes.append(f"bullish error: {exc}")

    bonus = sum(1 for v in flags.values() if v)
    ok = bonus >= min_bonus
    why = "pass" if ok else f"bonus {bonus} < required {min_bonus}"
    if notes:
        why += "; " + "; ".join(notes)

    return {"ok": ok, "bonus": bonus, "flags": flags, "why": why}

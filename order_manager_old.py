# order_manager_old.py
"""
OrderManager (updated for swing/risk/perf)

- Integrates tightly with RiskEngine (prefer allow_order(); fallback to pre_trade_checks()).
- Records fills in PerformanceTracker.
- Accepts dict or TradeSignal dataclass/object.
- SIM mode: paper fills with realistic prices (tick -> Yahoo -> fallback) and updates exposures.
- LIVE mode: optional KiteTicker and real orders (unchanged semantics).
- Optional market-hours gate via MARKET_HOURS_ONLY=1 (IST 09:15–15:30).
- Respects DISABLE_TICKER=1 so Twisted signal handlers are never touched.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, is_dataclass, asdict
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect, KiteTicker

from config import AppConfig
from risk_engine import RiskEngine
from performance import PerformanceTracker
from trade_signal import TradeSignal
from utils import setup_logger


@dataclass
class OrderSlice:
    symbol: str
    side: str            # 'BUY' or 'SELL'
    quantity: int
    limit_price: float
    order_id: Optional[str] = None
    filled_qty: int = 0
    status: str = "PENDING"


class OrderManager:
    def __init__(
        self,
        cfg: AppConfig,
        risk_engine: RiskEngine,
        kite: KiteConnect,
        tracker: Optional[PerformanceTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.risk = risk_engine
        self.kite = kite
        self.tracker = tracker or PerformanceTracker()

        self.logger = setup_logger("order_manager", "logs/order_manager.log")
        self._quotes: Dict[str, Dict] = {}
        self.ticker: Optional[KiteTicker] = None

        # cache for SIM prices so we don't hit the network repeatedly
        self._sim_price_cache: Dict[str, float] = {}

        # Flags from env / cfg
        self._ticker_disabled = (
            os.getenv("DISABLE_TICKER", "0") == "1"
            or str(getattr(getattr(cfg, "marketdata", object()), "source", "")).lower() in ("none", "")
        )
        self._market_hours_only = os.getenv("MARKET_HOURS_ONLY", "0") == "1"

        # Start market data (if applicable)
        self._start_market_data()  # no-op in SIM or when disabled

    # ---------------- Public API ----------------

    def handle_signal(self, sig_in: Any) -> None:
        """
        Accepts dict or TradeSignal-like object and routes to SIM or LIVE path.
        Applies risk checks using RiskEngine (allow_order -> pre_trade_checks -> fallback).
        """
        sig = self._normalize_sig(sig_in)
        symbol, side, qty, otype, price = (
            sig["symbol"], sig["side"], int(sig["qty"]), sig["type"], sig.get("price")
        )

        # Fetch LTP for checks/fills when price not provided
        ltp = self._last_price(symbol) or None
        if ltp is None or not self._is_pos_float(ltp):
            # pull a SIM price (tick->Yahoo->fallback), but do not mutate cache unless used
            ltp = self._get_sim_price(symbol)

        allowed, reason, signed_notional = self._risk_gate(sig, ltp)
        if not allowed:
            self.logger.warning(f"Signal rejected by risk gate: {reason} | {sig}")
            return

        env = str(self.cfg.env).upper()
        if env == "SIM":
            self._paper_trade(sig, fill_price=price if self._is_pos_float(price) else ltp)
        else:
            if self._market_hours_only and not self._is_market_open_ist():
                self.logger.warning(f"Market closed; dropping LIVE signal: {sig}")
                return
            self._live_trade(sig)

    def stop(self) -> None:
        """Graceful shutdown for websocket (if any)."""
        try:
            if self.ticker is not None:
                try:
                    self.ticker.close()
                except Exception:
                    pass
        except Exception:
            self.logger.exception("Error stopping ticker")

    # ---------------- Risk plumbing ----------------

    def _risk_gate(self, sig: Dict[str, Any], ltp: float) -> Tuple[bool, str, float]:
        """
        Prefer RiskEngine.allow_order(); fallback to pre_trade_checks(); else built-in checks.
        Returns (allowed, reason, signed_notional).
        """
        symbol = sig["symbol"]
        side = sig["side"]
        qty = int(sig["qty"])
        limit_price = float(sig.get("price") or ltp or 0.0)

        # Preferred: allow_order
        if hasattr(self.risk, "allow_order"):
            try:
                # Pass check_positions_cap if available on cfg
                cap = getattr(self.cfg, "max_positions", None)
                ok, reason, signed_notional = self.risk.allow_order(
                    symbol=symbol, side=side, qty=qty, ltp=ltp, limit_price=limit_price,
                    check_positions_cap=cap
                )
                return bool(ok), str(reason or ""), float(signed_notional)
            except Exception as e:
                self.logger.warning(f"risk.allow_order raised {e}; falling back to pre_trade_checks().")

        # Next: pre_trade_checks
        if hasattr(self.risk, "pre_trade_checks"):
            try:
                signed_notional = self._signed_notional(side, ltp, qty)
                ok, reason = self.risk.pre_trade_checks(symbol, signed_notional, ltp, limit_price)
                return bool(ok), str(reason or ""), float(signed_notional)
            except Exception as e:
                self.logger.warning(f"risk.pre_trade_checks raised {e}; falling back to built-in checks.")

        # Built-in minimal checks
        ok = self._builtin_checks(sig)
        return (ok, "" if ok else "basic validation failed", self._signed_notional(side, ltp, qty))

    # ---------------- Internals ----------------

    def _normalize_sig(self, sig_in: Any) -> Dict[str, Any]:
        """
        Normalize incoming signal into:
            {"symbol": str, "side": "BUY"/"SELL", "qty": int, "type": "MARKET"/"LIMIT", "price": float|None}
        """
        if isinstance(sig_in, dict):
            d = dict(sig_in)
        elif is_dataclass(sig_in):
            d = asdict(sig_in)
        else:
            # generic object with attributes
            d = {}
            for k in ("symbol", "side", "quantity", "qty", "limit_price", "price", "type", "order_type"):
                if hasattr(sig_in, k):
                    d[k] = getattr(sig_in, k)

        symbol = self._to_nse_symbol(d.get("symbol"))
        side = str(d.get("side", "BUY")).upper()
        qty = int(d.get("qty") or d.get("quantity") or 0)
        price = d.get("price", d.get("limit_price"))
        otype = (d.get("type") or d.get("order_type") or ("LIMIT" if price is not None else "MARKET"))
        otype = "LIMIT" if str(otype).upper() == "LIMIT" else "MARKET"

        return {"symbol": symbol, "side": side, "qty": qty, "type": otype, "price": price}

    def _builtin_checks(self, sig: Dict[str, Any]) -> bool:
        symbol = sig.get("symbol")
        side = sig.get("side")
        qty = sig.get("qty")
        otype = sig.get("type", "MARKET")
        price = sig.get("price", None)

        if not symbol or not isinstance(symbol, str):
            return False
        if side not in ("BUY", "SELL"):
            return False
        try:
            qty = int(qty)
        except Exception:
            return False
        if qty <= 0:
            return False
        if otype == "LIMIT":
            try:
                p = float(price)
            except Exception:
                return False
            if p <= 0:
                return False
        return True

    @staticmethod
    def _to_nse_symbol(symbol: Any) -> str:
        """Normalize 'NSE:RELIANCE'/'reliance' -> 'RELIANCE'."""
        if symbol is None:
            return ""
        return str(symbol).split(":")[-1].strip().upper()

    @staticmethod
    def _signed_notional(side: str, px: float, qty: int) -> float:
        sign = 1.0 if str(side).upper() == "BUY" else -1.0
        return sign * float(px or 0.0) * int(qty or 0)

    @staticmethod
    def _is_pos_float(x: Any) -> bool:
        try:
            return float(x) > 0.0
        except Exception:
            return False

    # ---------- SIM pricing helpers ----------

    def _yahoo_symbol(self, symbol: str) -> str:
        """Map 'RELIANCE' or 'NSE:RELIANCE' -> 'RELIANCE.NS' for Yahoo."""
        sym = str(symbol).split(":")[-1].strip().upper()
        return f"{sym}.NS"

    def _get_sim_price(self, symbol: str) -> float:
        """
        SIM fill price: try last tick -> Yahoo last close (cached) -> fallback.
        Never raises; always returns a positive float.
        """
        # 0) cache first
        if symbol in self._sim_price_cache:
            return float(self._sim_price_cache[symbol])

        # 1) last tick (if any)
        lp = self._last_price(symbol)
        if self._is_pos_float(lp):
            px = float(lp)
            self._sim_price_cache[symbol] = px
            return px

        # 2) Yahoo last close (best-effort, cache the result)
        try:
            import yfinance as yf  # optional dependency
            import pandas as pd
            ticker = self._yahoo_symbol(symbol)
            df = yf.download(ticker, period="5d", interval="1d", auto_adjust=False, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                # robust extraction for plain or MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        sub = df.xs(key="Close", axis=1, level=0, drop_level=False)
                    except Exception:
                        sub = df
                    if isinstance(sub, pd.DataFrame):
                        sub = sub.select_dtypes(include=["number"])
                        if sub.shape[1] == 0:
                            raise ValueError("No numeric columns in MultiIndex frame")
                        s = sub.iloc[:, 0]
                    else:
                        s = pd.to_numeric(sub, errors="coerce")
                else:
                    if "Close" in df.columns:
                        s = df["Close"]
                    else:
                        nums = df.select_dtypes(include=["number"])
                        if nums.shape[1] == 0:
                            raise ValueError("No numeric columns in frame")
                        s = nums.iloc[:, 0]
                    s = pd.to_numeric(s, errors="coerce")

                last_val = s.iloc[-1]
                if hasattr(last_val, "iloc"):  # guard if it's a Series
                    last_val = last_val.iloc[0]
                px = float(last_val)

                if px > 0:
                    self._sim_price_cache[symbol] = px
                    return px
        except Exception:
            pass

        # 3) safe fallback
        px = 100.0
        self._sim_price_cache[symbol] = px
        return px

    # ---------- SIM & LIVE execution ----------

    def _paper_trade(self, sig: Dict, fill_price: float) -> None:
        """
        SIM execution: assume full fill immediately at `fill_price`.
        - Records trade in PerformanceTracker
        - Updates RiskEngine exposures
        """
        symbol = sig["symbol"]
        side = sig["side"].upper()
        qty = int(sig["qty"])
        px = float(fill_price)

        # Tracker record
        try:
            self.tracker.record(TradeSignal(symbol=symbol, side=side, quantity=qty), px)
        except Exception:
            pass

        # Exposure update (signed notional, signed qty)
        try:
            signed_notional = self._signed_notional(side, px, qty)
            signed_qty = qty if side == "BUY" else -qty
            self.risk.update_position(symbol, signed_notional, signed_qty)
        except Exception:
            pass

        self.logger.info(f"[SIM] {side} {qty} {symbol} @ {px:.2f}")

    def _live_trade(self, sig: Dict) -> None:
        side = sig["side"].upper()
        symbol = sig["symbol"]
        qty = int(sig["qty"])
        otype = sig.get("type", "MARKET").upper()
        price = sig.get("price")

        params = dict(
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type=side,
            quantity=qty,
            order_type=otype,
            product="CNC",   # adjust if needed (CNC/NRML/MIS)
        )
        if otype == "LIMIT":
            if price is None:
                self.logger.error("LIMIT order requires 'price'; dropping.")
                return
            params["price"] = float(price)

        self.logger.info(f"[LIVE] Placing order: {params}")
        try:
            resp = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, **params)
            self.logger.info(f"Order placed: {resp}")
        except Exception as e:
            self.logger.exception(f"Order placement failed: {e}")

    # ---------- Market data helpers ----------

    def _last_price(self, symbol: str) -> Optional[float]:
        q = self._quotes.get(symbol)
        if not q:
            return None
        return q.get("last_price") or q.get("ltp")

    # ---------------- Market-hours logic ----------------

    def _is_market_open_ist(self) -> bool:
        """
        NSE regular session: Mon–Fri, 09:15–15:30 IST (no holiday calendar here).
        """
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if now.weekday() > 4:  # 0=Mon ... 6=Sun
            return False
        start = dtime(9, 15)
        end = dtime(15, 30)
        t = now.time()
        return start <= t <= end

    # ---------------- Market Data (LIVE only, optional) ----------------

    def _start_market_data(self) -> None:
        """Start KiteTicker only in LIVE mode and only if not disabled by env/cfg."""
        if str(self.cfg.env).upper() != "LIVE":
            self.logger.info("SIM mode: skipping KiteTicker startup.")
            return

        if self._ticker_disabled:
            self.logger.info("KiteTicker disabled by env/cfg; not starting.")
            return

        api_key = os.getenv("KITE_API_KEY") or getattr(self.cfg.broker, "api_key", None)
        access_token = os.getenv("KITE_ACCESS_TOKEN")  # set by your OAuth helper

        if not api_key or not access_token:
            self.logger.error(
                "LIVE requires KITE_API_KEY and KITE_ACCESS_TOKEN in environment. "
                "Run your OAuth app to obtain the token."
            )
            return

        try:
            self.ticker = KiteTicker(api_key, access_token)
        except Exception as e:
            self.logger.exception(f"Failed to create KiteTicker: {e}")
            return

        def on_ticks(ws, ticks):
            try:
                for t in ticks or []:
                    tsym = t.get("tradingsymbol")
                    if tsym:
                        self._quotes[str(tsym)] = t
            except Exception:
                pass

        def on_connect(ws, response):
            tokens = getattr(self.cfg, "instrument_tokens", []) or []
            if tokens:
                try:
                    ws.subscribe(tokens)
                    ws.set_mode(ws.MODE_QUOTE, tokens)
                    self.logger.info(f"Subscribed to {len(tokens)} tokens.")
                except Exception as e:
                    self.logger.exception(f"Subscribe/set_mode failed: {e}")
            else:
                self.logger.warning("No instrument tokens supplied; ticker connected but not subscribed.")

        def on_close(ws, code, reason):
            self.logger.warning(f"KiteTicker closed: code={code}, reason={reason}")

        def on_error(ws, code, reason):
            self.logger.error(f"KiteTicker error: code={code}, reason={reason}")

        self.ticker.on_ticks = on_ticks
        self.ticker.on_connect = on_connect
        self.ticker.on_close = on_close
        self.ticker.on_error = on_error

        # Important: call connect(threaded=True) from the MAIN thread.
        try:
            self.ticker.connect(threaded=True, disable_ssl_verification=False)
            self.logger.info("KiteTicker started (threaded=True).")
        except Exception as e:
            self.logger.exception(f"KiteTicker.connect failed: {e}")

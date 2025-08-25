# order_manager_old.py
"""
OrderManager (old) — patched

- Do NOT access kite._access_token (private/absent).
- In SIM mode: skip KiteTicker; do paper fills.
- In LIVE mode: start KiteTicker with env tokens and place real orders.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from kiteconnect import KiteConnect, KiteTicker

from config import AppConfig
from risk_engine import RiskEngine
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
    def __init__(self, cfg: AppConfig, risk_engine: RiskEngine, kite: KiteConnect) -> None:
        self.cfg = cfg
        self.risk = risk_engine
        self.kite = kite
        self.logger = setup_logger("order_manager", "logs/order_manager.log")

        # quotes cache from websocket ticks
        self._quotes: Dict[str, Dict] = {}

        # websocket/thread controls
        self._ticker: Optional[KiteTicker] = None
        self._md_thread: Optional[threading.Thread] = None
        self._md_stop = threading.Event()

        # Start market data only in LIVE
        self._start_market_data_thread()

    # ---------------- Public API ----------------

    def handle_signal(self, sig: Dict) -> None:
        """
        sig example:
          {
            "symbol": "INFY",
            "side": "BUY",
            "qty": 50,
            "type": "LIMIT" | "MARKET",
            "price": 1545.0   # required for LIMIT
          }
        """
        if not self.risk.check(sig):
            self.logger.warning(f"Signal rejected by risk engine: {sig}")
            return

        env = str(self.cfg.env).upper()
        if env == "SIM":
            self._paper_trade(sig)
        else:
            self._live_trade(sig)

    def stop(self) -> None:
        """Graceful shutdown for websocket thread."""
        try:
            self._md_stop.set()
        except Exception:
            pass

    # ---------------- Internals ----------------

    def _paper_trade(self, sig: Dict) -> None:
        # Simple SIM execution: fill at provided price or last price if available
        symbol = sig["symbol"]
        side = sig["side"].upper()
        qty = int(sig["qty"])
        price = sig.get("price") or self._last_price(symbol) or 0.0
        self.logger.info(f"[SIM] {side} {qty} {symbol} @ {price}")
        # TODO: extend with SIM positions/PnL if you need

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
            product="MIS",   # adjust if you use CNC/NRML
        )
        if otype == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires 'price'")
            params["price"] = float(price)

        self.logger.info(f"[LIVE] Placing order: {params}")
        try:
            resp = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, **params)
            self.logger.info(f"Order placed: {resp}")
        except Exception as e:
            self.logger.exception(f"Order placement failed: {e}")

    def _last_price(self, symbol: str) -> Optional[float]:
        q = self._quotes.get(symbol)
        if not q:
            return None
        # Kite ticks usually carry 'last_price' (LTP)
        return q.get("last_price") or q.get("ltp")

    # ---------------- Market Data (LIVE only) ----------------

    def _start_market_data_thread(self) -> None:
        """Start KiteTicker only in LIVE mode with valid env vars."""
        if str(self.cfg.env).upper() != "LIVE":
            self.logger.info("SIM mode: skipping KiteTicker startup.")
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
            self._ticker = KiteTicker(api_key, access_token)
        except Exception as e:
            self.logger.exception(f"Failed to create KiteTicker: {e}")
            return

        def on_ticks(ws, ticks):
            # ticks: list[dict]; map to cache by tradingsymbol when present
            for t in ticks:
                tsym = t.get("tradingsymbol")
                if tsym:
                    self._quotes[str(tsym)] = t

        def on_connect(ws, response):
            # Subscribe to instrument tokens from config if provided
            # Expecting cfg.instrument_tokens as List[int]
            tokens = getattr(self.cfg, "instrument_tokens", []) or []
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_QUOTE, tokens)
                self.logger.info(f"Subscribed to {len(tokens)} tokens.")
            else:
                self.logger.warning("No instrument tokens supplied; ticker connected but not subscribed.")

        def on_close(ws, code, reason):
            self.logger.warning(f"KiteTicker closed: code={code}, reason={reason}")

        def on_error(ws, code, reason):
            self.logger.error(f"KiteTicker error: code={code}, reason={reason}")

        # bind handlers
        self._ticker.on_ticks = on_ticks
        self._ticker.on_connect = on_connect
        self._ticker.on_close = on_close
        self._ticker.on_error = on_error

        # run loop with reconnect/backoff
        def _run():
            backoff = 2
            while not self._md_stop.is_set():
                try:
                    # threaded=False to keep control; loop exits on close
                    self._ticker.connect(threaded=False, disable_ssl_verification=False)
                    backoff = 2  # reset after a successful session
                except Exception as e:
                    self.logger.exception(f"KiteTicker connect loop exception: {e}")
                    time.sleep(min(backoff, 30))
                    backoff = min(backoff * 2, 30)

        self._md_thread = threading.Thread(target=_run, name="kiteticker", daemon=True)
        self._md_thread.start()
        self.logger.info("KiteTicker thread started.")

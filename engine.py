from __future__ import annotations

import os
import sys
import signal
import time
import traceback
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Generator, Dict, Any, Optional, Callable

from dotenv import load_dotenv
from kiteconnect import KiteConnect

from config import load_config
from risk_engine import RiskEngine
from order_manager_old import OrderManager  # <-- old manager
from utils import setup_logger

# Try to import signals module but don't hard-fail if it's missing
try:
    import signals as signal_module
except Exception:  # noqa: BLE001
    signal_module = None  # we will fallback


# ---------------- Utilities ----------------

def _ensure_dirs():
    Path("logs").mkdir(parents=True, exist_ok=True)


def _mode(env_value: str | None) -> str:
    if not env_value:
        return "SIM"
    env = str(env_value).upper()
    return "LIVE" if env == "LIVE" else "SIM"


def _crash(msg: str, logger=None):
    # Print a readable crash message and exit(1)
    banner = "=" * 72
    tb = traceback.format_exc()
    text = f"\n{banner}\nFATAL: {msg}\n{banner}\n{tb}\n"
    try:
        # Windows console-friendly
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass
    if logger:
        try:
            logger.error(msg)
            logger.error(tb)
        except Exception:
            pass
    sys.exit(1)


def _normalise_signal(sig: Any) -> Dict[str, Any]:
    """
    Convert various signal shapes into:
      {"symbol": str, "side": "BUY"/"SELL", "qty": int, "type": "MARKET"/"LIMIT", "price": float|None}
    """
    if isinstance(sig, dict):
        out = {}
        out["symbol"] = sig.get("symbol") or sig.get("tradingsymbol") or sig.get("s")
        out["side"] = (sig.get("side") or sig.get("transaction_type") or "BUY").upper()
        out["qty"] = int(sig.get("qty") or sig.get("quantity") or 0)
        otype = (sig.get("type") or sig.get("order_type") or ("LIMIT" if sig.get("limit_price") else "MARKET")).upper()
        out["type"] = "LIMIT" if otype == "LIMIT" else "MARKET"
        out["price"] = sig.get("price") or sig.get("limit_price")
        if not out["symbol"] or out["qty"] <= 0:
            raise ValueError(f"Incomplete signal dict: {sig!r}")
        return out

    # object with attributes
    for attr in ("symbol", "side", "quantity", "qty", "limit_price", "price"):
        if hasattr(sig, attr):
            symbol = getattr(sig, "symbol", None)
            side = getattr(sig, "side", "BUY")
            qty = getattr(sig, "quantity", None)
            if qty is None:
                qty = getattr(sig, "qty", 0)
            price = getattr(sig, "limit_price", None)
            if price is None:
                price = getattr(sig, "price", None)
            otype = getattr(sig, "type", None) or getattr(sig, "order_type", None)
            otype = (otype or ("LIMIT" if price is not None else "MARKET")).upper()
            if not symbol or int(qty) <= 0:
                raise ValueError(f"Incomplete signal object: {sig!r}")
            return {
                "symbol": symbol,
                "side": str(side).upper(),
                "qty": int(qty),
                "type": "LIMIT" if otype == "LIMIT" else "MARKET",
                "price": price,
            }

    # tuple/list: (symbol, side, qty, price=None, type=None)
    if isinstance(sig, (tuple, list)) and len(sig) >= 3:
        symbol = sig[0]
        side = str(sig[1]).upper()
        qty = int(sig[2])
        price = sig[3] if len(sig) >= 4 else None
        otype = (str(sig[4]).upper() if len(sig) >= 5 and sig[4] else
                 ("LIMIT" if price is not None else "MARKET"))
        if not symbol or qty <= 0:
            raise ValueError(f"Incomplete signal tuple/list: {sig!r}")
        return {"symbol": symbol, "side": side, "qty": qty, "type": "LIMIT" if otype == "LIMIT" else "MARKET", "price": price}

    raise ValueError(f"Unsupported signal format: {sig!r}")


def _fallback_signal_generator() -> Generator[Dict[str, Any], None, None]:
    """
    Minimal generator so SIM mode can run even if signals.py is missing/broken.
    Emits a single MARKET buy for DEMO:INFY with qty 1.
    """
    yield {"symbol": "INFY", "side": "BUY", "qty": 1, "type": "MARKET", "price": None}


def _make_signal_generator(cfg, logger):
    """
    Try to obtain a real generator from signals.py, otherwise use the fallback.
    Priority:
      1) generate_signals(symbols) if cfg.marketdata.symbols exists
      2) generate_signals(cfg)
      3) generate_signals() no args
      4) fallback generator
    """
    if signal_module and hasattr(signal_module, "generate_signals"):
        # 1) symbols list
        try:
            symbols = []
            try:
                symbols = [s.split(":")[-1] for s in getattr(cfg, "marketdata").symbols]  # "NSE:INFY" -> "INFY"
            except Exception:
                symbols = []
            if symbols:
                gen = signal_module.generate_signals(symbols)
                logger.info(f"Signal generator started with {len(symbols)} symbols from config.")
                return gen
        except Exception as e:
            logger.warning(f"generate_signals(symbols) failed: {e}")

        # 2) cfg
        try:
            gen = signal_module.generate_signals(cfg)
            logger.info("Signal generator started with cfg.")
            return gen
        except Exception as e:
            logger.warning(f"generate_signals(cfg) failed: {e}")

        # 3) no args
        try:
            gen = signal_module.generate_signals()
            logger.info("Signal generator started with default settings.")
            return gen
        except Exception as e:
            logger.warning(f"generate_signals() failed: {e}")

    logger.warning("Using fallback signal generator (single demo trade).")
    return _fallback_signal_generator()


# ---------------- Non-blocking signal pump ----------------

class SignalPump:
    """
    Runs the (potentially blocking) signal generator in a background thread
    and feeds a Queue. Engine main loop consumes with timeout and stays responsive.
    """
    def __init__(self, factory: Callable, cfg, logger) -> None:
        self._factory = factory
        self._cfg = cfg
        self._logger = logger
        self.queue: "Queue[Any]" = Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="signal-pump", daemon=True)
        self._thread.start()

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                gen = self._factory(self._cfg, self._logger)
                # reset backoff after a successful (re)build
                backoff = 1.0
                for item in gen:
                    if self._stop.is_set():
                        break
                    try:
                        self.queue.put(item, timeout=0.5)
                    except Exception:
                        # queue full: drop oldest by getting one, then put
                        try:
                            _ = self.queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.queue.put(item, timeout=0.5)
                        except Exception:
                            pass
                # generator exhausted: loop to rebuild immediately
                self._logger.info("Signal generator exhausted. Reinitialising...")
            except Exception as e:
                self._logger.exception(f"Signal generator error in pump thread: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    def get(self, timeout: float = 0.5) -> Optional[Any]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def shutdown(self):
        self._stop.set()
        # Let the generator finish its current sleep/yield naturally
        self._thread.join(timeout=3.0)


# ---------------- Market-hours helper ----------------

import datetime as _dt
try:
    from zoneinfo import ZoneInfo as _ZoneInfo  # py3.9+
    _IST = _ZoneInfo("Asia/Kolkata")
except Exception:  # fallback if tzdata missing
    _ZoneInfo = None
    _IST = None

def _is_market_open() -> bool:
    """
    Simple NSE equity hours check: Mon–Fri, 09:15–15:30 IST.
    Used to drop LIVE signals after hours when MARKET_HOURS_ONLY=1.
    """
    now = _dt.datetime.now(tz=_IST) if _IST else _dt.datetime.now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


# ---------------- Main ----------------

def main():
    # Load .env first
    if Path(".env").exists():
        try:
            load_dotenv(".env")
        except Exception:
            pass

    # Config & logger
    try:
        cfg = load_config("config.yaml")
    except Exception:
        _crash("Failed to load config.yaml", None)

    _ensure_dirs()
    logger = setup_logger("engine", "logs/engine.log")
    mode = _mode(getattr(cfg, "env", None))
    logger.info(f"Starting engine in {mode} mode...")

    # --- Force-disable ticker if requested BEFORE creating OrderManager ---
    # This prevents downstream modules (e.g., OrderManager) from starting KiteTicker,
    # which otherwise can crash with "signal only works in main thread".
    if os.getenv("DISABLE_TICKER", "0") == "1":
        try:
            if hasattr(cfg, "marketdata"):
                setattr(cfg.marketdata, "source", "none")
            else:
                class _MD: pass
                cfg.marketdata = _MD()
                cfg.marketdata.source = "none"
            logger.info("Ticker disabled via env; forcing marketdata.source='none'.")
        except Exception:
            logger.exception("Failed to force-disable ticker via cfg override.")

    # Broker credentials
    api_key = os.getenv("KITE_API_KEY") or getattr(getattr(cfg, "broker", object()), "api_key", None)
    if not api_key:
        _crash("KITE_API_KEY missing. Set it in .env or config.yaml -> broker.api_key", logger)

    kite = KiteConnect(api_key=api_key)

    # LIVE needs access token
    access_token = os.getenv("KITE_ACCESS_TOKEN")
    if mode == "LIVE":
        if not access_token:
            _crash("KITE_ACCESS_TOKEN missing for LIVE. Run your OAuth app to generate it.", logger)
        try:
            kite.set_access_token(access_token)
        except Exception:
            _crash("Failed to set Kite access token in LIVE.", logger)
    else:
        # SIM: attempt to set if present; ignore failure
        if access_token:
            try:
                kite.set_access_token(access_token)
            except Exception:
                pass

    # Core components
    try:
        risk_engine = RiskEngine(cfg)
    except Exception:
        _crash("Failed to initialise RiskEngine(cfg). Check your config fields.", logger)

    try:
        order_manager = OrderManager(cfg, risk_engine, kite)
    except Exception:
        _crash("Failed to initialise OrderManager(cfg, risk_engine, kite).", logger)

    # Graceful shutdown
    shutting_down = {"flag": False}
    pump_holder: Dict[str, Optional[SignalPump]] = {"pump": None}

    def _shutdown(signame: str):
        if shutting_down["flag"]:
            return
        shutting_down["flag"] = True
        logger.info(f"Shutdown requested ({signame}). Stopping services...")
        try:
            pump = pump_holder.get("pump")
            if pump:
                pump.shutdown()
        except Exception:
            logger.exception("Error during signal pump shutdown")
        try:
            if hasattr(order_manager, "stop"):
                order_manager.stop()
        except Exception:
            logger.exception("Error during order_manager.stop()")
        logger.info("Shutdown complete.")
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGINT, lambda *_: _shutdown("SIGINT"))
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, lambda *_: _shutdown("SIGTERM"))
    except Exception:
        pass

    # Signal pump (non-blocking wrapper over the generator)
    pump = SignalPump(_make_signal_generator, cfg, logger)
    pump_holder["pump"] = pump
    logger.info("Waiting for signals...")

    # Main loop (non-blocking, responsive to shutdown)
    try:
        while not shutting_down["flag"]:
            raw = pump.get(timeout=0.5)  # None if no signal yet
            if raw is None:
                continue

            try:
                sig = _normalise_signal(raw)

                # --- Optional market-hours gate for LIVE ---
                if (mode == "LIVE"
                    and os.getenv("MARKET_HOURS_ONLY", "1") == "1"
                    and not _is_market_open()):
                    logger.warning("Market closed; dropping LIVE signal: %s", sig)
                    # Skip placing orders after hours; avoid Zerodha errors
                    continue

                logger.info(f"Received signal: {sig}")
                order_manager.handle_signal(sig)
            except Exception as e:
                logger.exception(f"Error processing signal {raw}: {e}")
                print("\n[ENGINE] Error processing signal:", raw, "->", e, file=sys.stderr)
                traceback.print_exc()

            time.sleep(0.1)
    except KeyboardInterrupt:
        _shutdown("KeyboardInterrupt")
    except Exception:
        _crash("Fatal error in engine main loop.", logger)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # allow clean exits
    except Exception:
        # Last-resort crash report if something blew up before logger existed
        print("\n[ENGINE] Unhandled exception at top level:", file=sys.stderr)
        traceback.print_exc()
        _crash("Unhandled exception at top level.", None)

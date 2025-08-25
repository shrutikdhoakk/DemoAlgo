"""
config.py
-----------

Utility functions to load the YAML configuration file.

The configuration defines high level settings (environment, risk
limits, symbols, etc.) and broker credentials.  We read this
configuration once at startup and make it available to the rest
of the engine.  Secrets (API key/secret) should also be defined
in the `.env` file – values in `config.yml` are loaded as
defaults and may be overridden by environment variables.
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# --------- dataclasses ---------

@dataclass
class BrokerConfig:
    name: str = ""
    api_key: str = ""
    api_secret: str = ""
    rate_limit_per_sec: int = 3

@dataclass
class MarketDataConfig:
    source: str = ""               # e.g. "broker_ws" or "yfinance"
    symbols: List[str] = field(default_factory=list)

@dataclass
class RiskConfig:
    max_portfolio_value: float = 0.0
    per_symbol_cap: float = 0.0
    sector_cap_pct: float = 0.0
    min_adv_pct: float = 0.0
    price_band_buffer_bps: float = 0.0

@dataclass
class StopsConfig:
    atr_multiple_init: float = 2.0
    atr_multiple_trail: float = 1.5

@dataclass
class MonitoringConfig:
    alert_email: str = ""

@dataclass
class StrategyConfig:
    max_positions: int = 5
    total_capital: float = 10000.0
    max_invested: float = 5000.0
    per_trade_risk: float = 1000.0

@dataclass
class AppConfig:
    env: str = "SIM"
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    marketdata: MarketDataConfig = field(default_factory=MarketDataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    stops: StopsConfig = field(default_factory=StopsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # Optional LIVE-only convenience (used by OrderManager ticker)
    instrument_tokens: List[int] = field(default_factory=list)

    # Raw YAML for reference
    raw: Dict[str, Any] = field(default_factory=dict)

    # Convenience mirrors (so callers can do cfg.max_positions, etc.)
    max_positions: int = 5
    total_capital: float = 10000.0
    max_invested: float = 5000.0
    per_trade_risk: float = 1000.0

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from a YAML file (yml/yaml)."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Sections with safe defaults
        broker_data = data.get("broker", {}) or {}
        marketdata_data = data.get("marketdata", {}) or {}
        risk_data = data.get("risk", {}) or {}
        stops_data = data.get("stops", {}) or {}
        mon_data = data.get("monitoring", {}) or {}
        strat_data = data.get("strategy", {}) or {}

        broker = BrokerConfig(
            name=str(broker_data.get("name", "")),
            api_key=str(broker_data.get("api_key", "")),
            api_secret=str(broker_data.get("api_secret", "")),
            rate_limit_per_sec=int(broker_data.get("rate_limit_per_sec", 3)),
        )
        marketdata = MarketDataConfig(
            source=str(marketdata_data.get("source", "")),
            symbols=list(marketdata_data.get("symbols", []) or []),
        )
        risk = RiskConfig(
            max_portfolio_value=float(risk_data.get("max_portfolio_value", 0.0)),
            per_symbol_cap=float(risk_data.get("per_symbol_cap", 0.0)),
            sector_cap_pct=float(risk_data.get("sector_cap_pct", 0.0)),
            min_adv_pct=float(risk_data.get("min_adv_pct", 0.0)),
            price_band_buffer_bps=float(risk_data.get("price_band_buffer_bps", 0.0)),
        )
        stops = StopsConfig(
            atr_multiple_init=float(stops_data.get("atr_multiple_init", 2.0)),
            atr_multiple_trail=float(stops_data.get("atr_multiple_trail", 1.5)),
        )
        monitoring = MonitoringConfig(
            alert_email=str(mon_data.get("alert_email", "")),
        )
        strategy = StrategyConfig(
            max_positions=int(strat_data.get("max_positions", 5)),
            total_capital=float(strat_data.get("total_capital", 10000.0)),
            max_invested=float(strat_data.get("max_invested", 5000.0)),
            per_trade_risk=float(strat_data.get("per_trade_risk", 1000.0)),
        )

        env = str(data.get("env", "SIM")).upper()

        cfg = cls(
            env=env,
            broker=broker,
            marketdata=marketdata,
            risk=risk,
            stops=stops,
            monitoring=monitoring,
            strategy=strategy,
            instrument_tokens=list(data.get("instrument_tokens", []) or []),
            raw=data,
        )

        # Mirror strategy fields to top-level convenience attributes
        cfg.max_positions = strategy.max_positions
        cfg.total_capital = strategy.total_capital
        cfg.max_invested = strategy.max_invested
        cfg.per_trade_risk = strategy.per_trade_risk
        return cfg

    def override_from_env(self) -> None:
        """
        Override sensitive fields (API key/secret) and env with values from
        environment variables if present. Keep secrets outside version control.
        """
        self.broker.api_key = os.getenv("KITE_API_KEY", self.broker.api_key)
        self.broker.api_secret = os.getenv("KITE_API_SECRET", self.broker.api_secret)

        # Allow switching SIM/LIVE via ENV
        env_override = os.getenv("ENV")
        if env_override:
            self.env = str(env_override).upper()


# --------- public API ---------

def _default_config_path(explicit_path: Optional[str]) -> str:
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path
    # Try common filenames
    for candidate in ("config.yml", "config.yaml"):
        if os.path.exists(candidate):
            return candidate
    # Fall back to provided (even if missing, will raise cleanly)
    return explicit_path or "config.yml"


def load_config(config_path: str | None = None) -> AppConfig:
    """Load configuration and apply env overrides."""
    path = _default_config_path(config_path)
    cfg = AppConfig.from_yaml(path)
    cfg.override_from_env()
    return cfg

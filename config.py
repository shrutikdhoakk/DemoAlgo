"""
config.py
-----------

Utility functions to load the YAML configuration file.

The configuration defines high level settings (environment, risk
limits, symbols, etc.) and broker credentials.  We read this
configuration once at startup and make it available to the rest
of the engine.  Secrets (API key/secret) should also be defined
in the `.env` file – values in `config.yaml` are loaded as
defaults and may be overridden by environment variables.
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BrokerConfig:
    name: str
    api_key: str
    api_secret: str
    rate_limit_per_sec: int

@dataclass
class MarketDataConfig:
    source: str
    symbols: List[str]

@dataclass
class RiskConfig:
    max_portfolio_value: float
    per_symbol_cap: float
    sector_cap_pct: float
    min_adv_pct: float
    price_band_buffer_bps: float

@dataclass
class StopsConfig:
    atr_multiple_init: float
    atr_multiple_trail: float

@dataclass
class MonitoringConfig:
    alert_email: str

@dataclass
class AppConfig:
    env: str
    broker: BrokerConfig
    marketdata: MarketDataConfig
    risk: RiskConfig
    stops: StopsConfig
    monitoring: MonitoringConfig

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> 'AppConfig':
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        # flatten keys and instantiate dataclasses
        broker_data = data.get('broker', {})
        broker = BrokerConfig(
            name=broker_data.get('name', ''),
            api_key=broker_data.get('api_key', ''),
            api_secret=broker_data.get('api_secret', ''),
            rate_limit_per_sec=int(broker_data.get('rate_limit_per_sec', 3)),
        )
        md_data = data.get('marketdata', {})
        marketdata = MarketDataConfig(
            source=md_data.get('source', ''),
            symbols=md_data.get('symbols', []),
        )
        risk_data = data.get('risk', {})
        risk = RiskConfig(
            max_portfolio_value=float(risk_data.get('max_portfolio_value', 0)),
            per_symbol_cap=float(risk_data.get('per_symbol_cap', 0)),
            sector_cap_pct=float(risk_data.get('sector_cap_pct', 0)),
            min_adv_pct=float(risk_data.get('min_adv_pct', 0)),
            price_band_buffer_bps=float(risk_data.get('price_band_buffer_bps', 0)),
        )
        stops_data = data.get('stops', {})
        stops = StopsConfig(
            atr_multiple_init=float(stops_data.get('atr_multiple_init', 2.0)),
            atr_multiple_trail=float(stops_data.get('atr_multiple_trail', 1.5)),
        )
        mon_data = data.get('monitoring', {})
        monitoring = MonitoringConfig(
            alert_email=mon_data.get('alert_email', ''),
        )
        env = data.get('env', 'SIM').upper()
        return cls(env=env, broker=broker, marketdata=marketdata,
                   risk=risk, stops=stops, monitoring=monitoring, raw=data)

    def override_from_env(self):
        """
        Override sensitive fields (API key/secret) with values from
        environment variables if present.  This allows you to keep
        secrets outside version control.
        """
        self.broker.api_key = os.getenv('KITE_API_KEY', self.broker.api_key)
        self.broker.api_secret = os.getenv('KITE_API_SECRET', self.broker.api_secret)
        # environment variable to switch env
        env_override = os.getenv('ENV')
        if env_override:
            self.env = env_override.upper()


def load_config(config_path: str = 'config.yaml') -> AppConfig:
    """Public helper to load configuration and apply overrides."""
    cfg = AppConfig.from_yaml(config_path)
    cfg.override_from_env()
    return cfg
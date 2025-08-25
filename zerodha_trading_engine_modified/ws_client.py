"""
ws_client.py
-------------

Standalone script to subscribe to live market data via Zerodha’s
WebSocket and print incoming ticks.  This utility is optional – the
execution engine (`engine.py`) already starts its own WebSocket
listener – but having a separate ticker can be useful for debugging
market data connectivity.

Usage:

```
python ws_client.py
```

It reads `KITE_API_KEY` and `KITE_ACCESS_TOKEN` from `.env` and the
symbols list from `config.yaml`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect, KiteTicker

from config import load_config


def main():
    # load env and config
    if Path('.env').exists():
        load_dotenv('.env')
    cfg = load_config('config.yaml')
    api_key = os.getenv('KITE_API_KEY') or cfg.broker.api_key
    access_token = os.getenv('KITE_ACCESS_TOKEN')
    if not api_key or not access_token:
        print('KITE_API_KEY or KITE_ACCESS_TOKEN missing. Please set them in .env.')
        return
    # map symbols to tokens
    tokens = []
    token_to_symbol = {}
    try:
        df = pd.read_csv('instruments_nse_eq.csv')
        for sym_full in cfg.marketdata.symbols:
            sym = sym_full.split(':')[-1]
            row = df[(df['exchange'] == 'NSE') & (df['tradingsymbol'] == sym)]
            if not row.empty:
                token = int(row.iloc[0]['instrument_token'])
                tokens.append(token)
                token_to_symbol[token] = sym
    except Exception as e:
        print(f'Error reading instruments.csv: {e}')
        return
    ticker = KiteTicker(api_key, access_token)

    def on_ticks(ws, ticks):
        for tick in ticks:
            token = tick.get('instrument_token')
            ltp = tick.get('last_price')
            sym = token_to_symbol.get(token, token)
            print(f'Tick: {sym} -> {ltp}')

    def on_connect(ws, response):
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_QUOTE, tokens)
    def on_error(ws, code, reason):
        print(f'WS error {code}: {reason}')
    def on_close(ws, code, reason):
        print(f'WS closed: {code} {reason}')

    ticker.on_ticks = on_ticks
    ticker.on_connect = on_connect
    ticker.on_error = on_error
    ticker.on_close = on_close
    ticker.connect(threaded=False)


if __name__ == '__main__':
    main()
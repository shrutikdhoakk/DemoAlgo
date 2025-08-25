"""
app.py
-------

Tiny Flask application to facilitate Zerodha Kite OAuth login.
Run this script once per trading day to obtain a fresh access token.
It writes the token to the `.env` file so that the engine can pick it up.

Usage:
```
python app.py
# then open http://127.0.0.1:5000/kite/login in your browser
```
"""
from __future__ import annotations

import os
from pathlib import Path
from config import AppConfig
from dotenv import load_dotenv
from flask import Flask, redirect, request
from kiteconnect import KiteConnect


def create_app() -> Flask:
    app = Flask(__name__)
    # load env variables
    if Path('.env').exists():
        load_dotenv('.env')
    api_key = os.getenv('KITE_API_KEY')
    api_secret = os.getenv('KITE_API_SECRET')
    kite = KiteConnect(api_key=api_key)

    @app.route('/')
    def index():
        return 'Kite OAuth app running. Visit /kite/login to login.'

    @app.route('/kite/login')
    def kite_login():
        return redirect(kite.login_url())

    @app.route('/kite/callback')
    def kite_callback():
        request_token = request.args.get('request_token')
        if not request_token:
            return 'Missing request_token', 400
        try:
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data['access_token']
        except Exception as e:
            return f'Error generating session: {e}', 500
        # update .env file with access token
        env_path = Path('.env')
        lines = []
        if env_path.exists():
            with env_path.open('r') as f:
                lines = f.readlines()
        # remove old token line
        lines = [ln for ln in lines if not ln.startswith('KITE_ACCESS_TOKEN=')]
        # append new token
        lines.append(f'KITE_ACCESS_TOKEN={access_token}\n')
        with env_path.open('w') as f:
            f.writelines(lines)
        return 'Login OK. Access token saved in .env. You can close this window.', 200

    return app


if __name__ == '__main__':
    # create and run flask app
    app = create_app()
    # run on localhost
    app.run(host='127.0.0.1', port=5000, debug=False)
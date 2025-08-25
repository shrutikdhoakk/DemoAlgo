import os, requests
from dotenv import load_dotenv

load_dotenv()
h = {
  "X-Kite-Version": "3",
  "Authorization": f"token {os.getenv('KITE_API_KEY')}:{os.getenv('KITE_ACCESS_TOKEN')}"
}
r = requests.get("https://api.kite.trade/quote/ltp?i=NSE:INFY", headers=h)
print("HTTP:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print(r.text[:300])

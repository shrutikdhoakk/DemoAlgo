import requests

# Replace with your API credentials
api_key = "5ffdmumv2e5a3nsr"
access_token = "a4yWkkbw6HkeTUeh2t3RthiyyFFwoYBq"

# NSE instruments only
url = "https://api.kite.trade/instruments/NSE"
headers = {
    "X-Kite-Version": "3",
    "Authorization": f"token {api_key}:{access_token}"
}

# Make request
response = requests.get(url, headers=headers)

if response.status_code == 200:
    with open("instruments_nse_eq.csv", "wb") as f:
        f.write(response.content)
    print("✅ instruments_nse_eq.csv downloaded successfully!")
else:
    print("❌ Failed to download. Status:", response.status_code)
    print(response.text)

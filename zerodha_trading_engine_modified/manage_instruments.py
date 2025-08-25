# manage_instruments.py
import os, csv, gzip, shutil, argparse, sys
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent

def download():
    load_dotenv()
    api_key = os.getenv("KITE_API_KEY")
    access  = os.getenv("KITE_ACCESS_TOKEN")
    if not api_key or not access:
        print("❌ Missing KITE_API_KEY or KITE_ACCESS_TOKEN in .env")
        sys.exit(1)

    url = "https://api.kite.trade/instruments"
    headers = {"X-Kite-Version":"3", "Authorization": f"token {api_key}:{access}"}

    gz = ROOT / "instruments.csv.gz"
    csv_path = ROOT / "instruments.csv"

    print("⬇️  Downloading instruments.csv.gz ...")
    r = requests.get(url, headers=headers, stream=True)
    if r.status_code != 200:
        print(f"❌ HTTP {r.status_code} – {r.text[:200]}")
        sys.exit(1)

    with open(gz, "wb") as f:
        for chunk in r.iter_content(1024*64):
            if chunk:
                f.write(chunk)

    print("🗜️  Unzipping to instruments.csv ...")
    with gzip.open(gz, "rb") as src, open(csv_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    # sanity check
    with open(csv_path, newline="", encoding="utf-8") as f:
        first = f.readline().strip()
        if "instrument_token" not in first:
            print("⚠️  instruments.csv header not found. Download may be invalid.")
            sys.exit(1)

    print("✅ Downloaded and ready:", csv_path.resolve())

def _open_dict_reader(path: Path):
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)
    f = open(path, newline="", encoding="utf-8")
    r = csv.DictReader(f)
    if not r.fieldnames:
        print(f"❌ No header in {path}. Is the file empty/corrupt?")
        f.close()
        sys.exit(1)
    return f, r

def filter_nse_eq():
    src = ROOT / "instruments.csv"
    dst = ROOT / "instruments_nse_eq.csv"
    f, r = _open_dict_reader(src)
    with f, open(dst, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=r.fieldnames)
        w.writeheader()
        n = 0
        for row in r:
            if row.get("exchange") == "NSE" and row.get("instrument_type") == "EQ":
                w.writerow(row); n += 1
    print(f"✅ Saved {dst.name} with {n} rows")

def lookup(symbol: str):
    # prefer filtered file; fall back to full
    src = ROOT / "instruments_nse_eq.csv"
    if not src.exists():
        src = ROOT / "instruments.csv"
        print("ℹ️  Using instruments.csv (filtered file not found).")

    f, r = _open_dict_reader(src)
    with f:
        for row in r:
            if row.get("tradingsymbol") == symbol:
                print(f"{symbol} -> instrument_token={row.get('instrument_token')}, "
                      f"tick={row.get('tick_size')}, lot_size={row.get('lot_size')}, "
                      f"segment={row.get('segment')}, exchange={row.get('exchange')}")
                return
    print("❌ Symbol not found.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kite instruments helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download")
    sub.add_parser("filter")
    q = sub.add_parser("lookup")
    q.add_argument("symbol", help="e.g., INFY")

    args = p.parse_args()
    if args.cmd == "download":
        download()
    elif args.cmd == "filter":
        filter_nse_eq()
    elif args.cmd == "lookup":
        lookup(args.symbol)

import csv, sys
symbol = sys.argv[1]  # e.g. INFY
with open("instruments_nse_eq.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["tradingsymbol"] == symbol:
            print(f'{symbol} -> instrument_token={row["instrument_token"]}, tick={row["tick_size"]}')
            break
    else:
        print("Symbol not found in NSE EQ list.")

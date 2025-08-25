import csv

with open("instruments.csv", newline="", encoding="utf-8") as f, \
     open("instruments_nse_eq.csv", "w", newline="", encoding="utf-8") as out:
    r = csv.DictReader(f)
    if not r.fieldnames:
        raise ValueError("⚠ instruments.csv has no header, check if download worked")

    w = csv.DictWriter(out, fieldnames=r.fieldnames)
    w.writeheader()

    count = 0
    for row in r:
        if row["exchange"] == "NSE" and row["instrument_type"] == "EQ":
            w.writerow(row)
            count += 1

print(f"✅ Saved instruments_nse_eq.csv with {count} NSE EQ rows")

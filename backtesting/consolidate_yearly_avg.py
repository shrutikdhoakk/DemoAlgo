import pandas as pd

# Load your yearly returns file
df = pd.read_csv("nifty500_returns_5y_long.csv")

# Drop missing values
df = df.dropna(subset=["ReturnPct"])

# Group by year and compute stats
summary = df.groupby("Year")["ReturnPct"].agg(
    AvgReturn="mean",
    MedianReturn="median",
    PositivePct=lambda x: (x > 0).mean() * 100,
    Count="count"
).reset_index()

# Round nicely
summary = summary.round(2)

print(summary)

# Save to CSV for later use
summary.to_csv("nifty500_yearly_consolidated.csv", index=False)
print("\nSaved consolidated results to nifty500_yearly_consolidated.csv")

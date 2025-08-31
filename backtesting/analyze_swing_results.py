import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(".")
TRADES = ROOT / "swing_trades.csv"
EQUITY = ROOT / "swing_equity.csv"
YEARLY = ROOT / "swing_yearly_returns.csv"

def max_drawdown(eq: pd.Series):
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.min(), dd

def pair_trades(trades: pd.DataFrame):
    """Assumes one entry then one exit per symbol cycle; skip any open positions."""
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades.sort_values(["symbol","date","side"])
    pairs = []
    open_pos = {}  # symbol -> {'date','price','qty'}
    for _, r in trades.iterrows():
        sym = r["symbol"]; side = r["side"]; dt = r["date"]; px = float(r["price"]); qty = int(r["qty"])
        if side == "BUY":
            open_pos[sym] = {"date": dt, "price": px, "qty": qty}
        elif side == "SELL" and sym in open_pos:
            e = open_pos.pop(sym)
            if e["qty"] <= 0: 
                continue
            ret_pct = (px / e["price"] - 1.0) * 100.0
            pnl = (px - e["price"]) * e["qty"]
            hold = (dt - e["date"]).days
            pairs.append({
                "symbol": sym,
                "entry_date": e["date"].date(),
                "exit_date": dt.date(),
                "entry_px": round(e["price"],2),
                "exit_px": round(px,2),
                "qty": e["qty"],
                "ret_pct": round(ret_pct,2),
                "pnl": round(pnl,2),
                "hold_days": hold,
                "exit_reason": r.get("reason","")
            })
    return pd.DataFrame(pairs)

def main():
    if not EQUITY.exists():
        raise SystemExit("Run the backtest first (swing_equity.csv not found).")

    eq = pd.read_csv(EQUITY, parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(TRADES, parse_dates=["date"]) if TRADES.exists() else pd.DataFrame()
    yearly = pd.read_csv(YEARLY) if YEARLY.exists() else pd.DataFrame()

    # Equity stats
    start_eq = float(eq["equity"].iloc[0])
    end_eq   = float(eq["equity"].iloc[-1])
    days = (eq.index[-1] - eq.index[0]).days
    years = max(1e-9, days / 365.25)
    total_return = (end_eq / start_eq - 1.0) * 100.0
    cagr = ((end_eq / start_eq) ** (1.0 / years) - 1.0) * 100.0
    mdd_val, dd_series = max_drawdown(eq["equity"])

    # Trade stats
    pair_df = pair_trades(trades) if not trades.empty else pd.DataFrame()
    winrate = (pair_df["ret_pct"] > 0).mean() * 100.0 if not pair_df.empty else np.nan
    avg_gain = pair_df.loc[pair_df["ret_pct"] > 0, "ret_pct"].mean() if not pair_df.empty else np.nan
    avg_loss = pair_df.loc[pair_df["ret_pct"] <= 0, "ret_pct"].mean() if not pair_df.empty else np.nan
    avg_hold = pair_df["hold_days"].mean() if not pair_df.empty else np.nan
    exit_counts = pair_df["exit_reason"].value_counts().rename_axis("exit_reason").reset_index(name="count") if "exit_reason" in pair_df else pd.DataFrame()

    # Save summary
    summary_lines = []
    summary_lines.append(f"Start Equity: {start_eq:,.2f}")
    summary_lines.append(f"End Equity:   {end_eq:,.2f}")
    summary_lines.append(f"Total Return: {total_return:.2f}%")
    summary_lines.append(f"CAGR:         {cagr:.2f}%")
    summary_lines.append(f"Max Drawdown: {mdd_val*100:.2f}%")
    if not pair_df.empty:
        summary_lines.append(f"Trades:       {len(pair_df)}  (Win rate: {winrate:.2f}%)")
        summary_lines.append(f"Avg Gain%:    {0 if np.isnan(avg_gain) else round(avg_gain,2)}   Avg Loss%: {0 if np.isnan(avg_loss) else round(avg_loss,2)}")
        summary_lines.append(f"Avg Hold (d): {0 if np.isnan(avg_hold) else round(avg_hold,1)}")
    if not yearly.empty:
        summary_lines.append("\nYearly portfolio returns:")
        for _, r in yearly.sort_values("Year").iterrows():
            summary_lines.append(f"  {int(r['Year'])}: {r['PortfolioReturn%']}%")
    if not exit_counts.empty:
        summary_lines.append("\nExit reason counts:")
        for _, r in exit_counts.iterrows():
            summary_lines.append(f"  {r['exit_reason']}: {r['count']}")

    (ROOT / "swing_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # Top / bottom trades csv
    if not pair_df.empty:
        top = pair_df.sort_values("ret_pct", ascending=False).head(20)
        bot = pair_df.sort_values("ret_pct").head(20)
        tb = pd.concat([top.assign(bucket="TOP"), bot.assign(bucket="BOTTOM")], ignore_index=True)
        tb.to_csv("swing_top_bottom_trades.csv", index=False)

    # Plot equity curve
    try:
        plt.figure(figsize=(10,4))
        eq["equity"].plot()
        plt.title("SwingStrategy Equity Curve")
        plt.xlabel("Date"); plt.ylabel("Equity (₹)")
        plt.tight_layout()
        plt.savefig("swing_equity.png", dpi=150)
    except Exception:
        pass

    print("\n".join(summary_lines))
    print("\nSaved: swing_summary.txt, swing_top_bottom_trades.csv (top/bottom), swing_equity.png")

if __name__ == "__main__":
    main()

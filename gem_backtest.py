"""Backtest of Gary Antonacci's Global Equities Momentum (GEM) dual momentum strategy.

Rules (evaluated monthly, using trailing 12-month total returns):
  1. If US equities' 12-month return > cash's 12-month return (absolute momentum),
     hold whichever of US or ex-US equities had the higher 12-month return
     (relative momentum).
  2. Otherwise, hold bonds.

Usage:
    python gem_backtest.py [--start 2007-01-01] [--end 2024-12-31] [--out gem_backtest.png]
"""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

US_EQUITY = "SPY"      # US equities
INTL_EQUITY = "VEU"    # ex-US (all-world ex-US) equities
BONDS = "AGG"          # US aggregate bonds
CASH = "BIL"           # 1-3 month T-bill proxy
LOOKBACK_MONTHS = 12


def download_monthly_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    monthly = data.resample("ME").last().dropna(how="all")

    missing = [t for t in tickers if t not in monthly.columns or monthly[t].dropna().empty]
    if missing:
        raise RuntimeError(
            f"No price data returned for {missing}. Check network access and ticker "
            "symbols, or widen the --start/--end date range."
        )
    monthly = monthly.dropna()
    if len(monthly) <= LOOKBACK_MONTHS:
        raise RuntimeError(
            f"Only {len(monthly)} months of overlapping data available; need more than "
            f"{LOOKBACK_MONTHS} to compute momentum. Widen the --start/--end date range."
        )
    return monthly


def momentum(prices, months=LOOKBACK_MONTHS):
    return prices / prices.shift(months) - 1.0


def run_gem(prices, months=LOOKBACK_MONTHS):
    mom = momentum(prices, months)
    monthly_returns = prices.pct_change()

    choice = []
    dates = prices.index

    for i in range(len(dates)):
        if i < months:
            choice.append(np.nan)
            continue

        us_mom = mom[US_EQUITY].iloc[i]
        intl_mom = mom[INTL_EQUITY].iloc[i]
        cash_mom = mom[CASH].iloc[i]

        if us_mom > cash_mom:
            asset = US_EQUITY if us_mom >= intl_mom else INTL_EQUITY
        else:
            asset = BONDS

        choice.append(asset)

    signal = pd.Series(choice, index=dates)
    # Signal decided at month t is applied to the return of month t+1.
    applied_signal = signal.shift(1)

    realized = []
    for i, dt in enumerate(dates):
        asset = applied_signal.iloc[i]
        if pd.isna(asset):
            realized.append(np.nan)
        else:
            realized.append(monthly_returns[asset].iloc[i])

    strat = pd.Series(realized, index=dates, name="GEM")
    return strat, signal


def performance_stats(returns, periods_per_year=12):
    returns = returns.dropna()
    growth = (1 + returns).cumprod()
    n_years = len(returns) / periods_per_year
    cagr = growth.iloc[-1] ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else np.nan
    running_max = growth.cummax()
    drawdown = growth / running_max - 1
    max_dd = drawdown.min()
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Final Growth of $1": growth.iloc[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="GEM dual momentum backtest")
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="gem_backtest.png")
    args = parser.parse_args()

    tickers = [US_EQUITY, INTL_EQUITY, BONDS, CASH]
    prices = download_monthly_prices(tickers, args.start, args.end)

    strat_returns, signal = run_gem(prices)
    benchmark_returns = prices[US_EQUITY].pct_change()

    combined = pd.DataFrame({"GEM": strat_returns, "SPY (buy & hold)": benchmark_returns}).dropna()
    growth = (1 + combined).cumprod()

    print("\n=== GEM Strategy Performance ===")
    for name, series in combined.items():
        stats = performance_stats(series)
        print(f"\n{name}:")
        for k, v in stats.items():
            if k == "Final Growth of $1":
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v:.2%}")

    print("\nCurrent allocation signal (most recent month):")
    print(signal.dropna().tail(1).to_string())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    growth.plot(ax=ax)
    ax.set_title("GEM Dual Momentum vs. Buy & Hold SPY")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved equity curve chart to {args.out}")


if __name__ == "__main__":
    main()

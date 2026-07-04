"""Backtest of Gary Antonacci's Global Equities Momentum (GEM) dual momentum strategy.

Rules (evaluated monthly, using trailing N-month total returns, N=12 by default):
  1. If US equities' trailing return > the cash/T-bill yardstick's trailing
     return (absolute momentum), hold whichever of US or ex-US equities had
     the higher trailing return (relative momentum).
  2. Otherwise, hold bonds.

Long-history mutual funds are used instead of ETFs so the test can reach
back into the 1990s:
    VFINX  - Vanguard 500 Index (US equities), inception 1976
    VGTSX  - Vanguard Total International Stock Index, inception 1996
    VBMFX  - Vanguard Total Bond Market Index, inception 1986
    ^IRX   - 13-week T-bill annualized discount yield, used only as the
             absolute-momentum cash yardstick (never actually held)

Every position is decided using only the prior month's trailing window (no
look-ahead), and switches between holdings are charged a per-switch cost.

Usage:
    python gem_backtest.py [--start 1970-01-01] [--end 2024-12-31]
                           [--lookback 12] [--cost 0.001] [--out gem_backtest.png]
"""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

US_TICKER = "VFINX"
INTL_TICKER = "VGTSX"
BOND_TICKER = "VBMFX"
CASH_TICKER = "^IRX"

DEFAULT_LOOKBACK = 12
LOOKBACK_GRID = [6, 9, 10, 12]
DEFAULT_COST = 0.0010
SPLIT_DATE = "2013-01-01"

SERIES_LABELS = {
    "US": US_TICKER,
    "INTL": INTL_TICKER,
    "BOND": BOND_TICKER,
    "CASH": f"{CASH_TICKER} (cash yardstick)",
}


def _download_monthly_close(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    monthly = raw.resample("ME").last().dropna()
    monthly.name = ticker
    return monthly


def build_price_panel(start, end):
    us = _download_monthly_close(US_TICKER, start, end)
    intl = _download_monthly_close(INTL_TICKER, start, end)
    bond = _download_monthly_close(BOND_TICKER, start, end)
    irx = _download_monthly_close(CASH_TICKER, start, end)

    # ^IRX quotes the annualized 13-week T-bill discount yield in percent.
    # Compound it into a synthetic NAV so it can be measured for trailing
    # momentum on the same footing as the fund NAVs (it is never held).
    cash_monthly_return = (1 + irx / 100) ** (1 / 12) - 1
    cash_nav = (1 + cash_monthly_return).cumprod()

    series = {"US": us, "INTL": intl, "BOND": bond, "CASH": cash_nav}
    first_valid = {name: s.index.min() for name, s in series.items()}
    bottleneck = max(first_valid, key=first_valid.get)

    print("Individual series start dates:")
    for name, dt in first_valid.items():
        print(f"  {SERIES_LABELS[name]}: {dt.date()}")

    prices = pd.concat(series, axis=1).dropna()
    print(f"\nTrue usable start date: {prices.index.min().date()} "
          f"(limited by {SERIES_LABELS[bottleneck]})")

    if len(prices) <= max(LOOKBACK_GRID):
        raise RuntimeError(
            f"Only {len(prices)} months of overlapping data available; need more than "
            f"{max(LOOKBACK_GRID)} to compute momentum. Widen --start/--end."
        )
    return prices


def compute_signal(prices, lookback):
    mom = prices / prices.shift(lookback) - 1.0

    def choose(row):
        if pd.isna(row["US"]) or pd.isna(row["CASH"]):
            return np.nan
        if row["US"] > row["CASH"]:
            return "US" if row["US"] >= row["INTL"] else "INTL"
        return "BOND"

    return mom.apply(choose, axis=1)


def run_gem(prices, lookback=DEFAULT_LOOKBACK, cost=DEFAULT_COST):
    signal = compute_signal(prices, lookback)
    # No look-ahead: the position held during month t is the signal computed
    # from the trailing window ending at month t-1, i.e. decided using only
    # data available before month t begins.
    held = signal.shift(1)

    asset_returns = prices.pct_change()
    gross = pd.Series(np.nan, index=prices.index)
    for dt, asset in held.items():
        if isinstance(asset, str):
            gross.loc[dt] = asset_returns.loc[dt, asset]

    switched = held.ne(held.shift(1)) & held.notna()
    net = gross - switched.astype(float) * cost
    net.name = f"GEM ({lookback}mo, net)"
    return net, held


def _longest_drawdown_months(growth):
    running_max = growth.cummax()
    at_peak = np.isclose(growth.values, running_max.values)
    peak_positions = np.flatnonzero(at_peak)
    if len(peak_positions) == 0:
        return len(growth)
    gaps = np.diff(peak_positions)
    longest = gaps.max() if len(gaps) else 0
    trailing = (len(growth) - 1) - peak_positions[-1]
    return int(max(longest, trailing))


def performance_stats(returns, periods_per_year=12):
    returns = returns.dropna()
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "MaxDrawdown",
            "LongestDrawdownMonths", "FinalGrowth"]
    if len(returns) < 2:
        return {k: np.nan for k in keys}

    growth = (1 + returns).cumprod()
    n_years = len(returns) / periods_per_year
    cagr = growth.iloc[-1] ** (1 / n_years) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol > 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = (returns.mean() * periods_per_year) / downside_vol if downside_vol else np.nan

    running_max = growth.cummax()
    max_dd = (growth / running_max - 1).min()
    longest_dd = _longest_drawdown_months(growth)

    return {
        "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDrawdown": max_dd, "LongestDrawdownMonths": longest_dd,
        "FinalGrowth": growth.iloc[-1],
    }


def sixty_forty(prices):
    # Simplifying assumption: rebalanced back to the 60/40 mix every month.
    blend = 0.6 * prices["US"].pct_change() + 0.4 * prices["BOND"].pct_change()
    blend.name = "60/40 (VFINX/VBMFX)"
    return blend


def _fmt(x, pct=True):
    if pd.isna(x):
        return "n/a"
    return f"{x:.2%}" if pct else f"{x:.2f}"


def report_window(label, window_returns):
    print(f"\n--- {label} ---")
    stats = {}
    for name, s in window_returns.items():
        stats[name] = performance_stats(s)
        st = stats[name]
        print(f"{name:24s} CAGR {_fmt(st['CAGR'])}  Vol {_fmt(st['Volatility'])}  "
              f"Sharpe {_fmt(st['Sharpe'], False)}  Sortino {_fmt(st['Sortino'], False)}  "
              f"MaxDD {_fmt(st['MaxDrawdown'])}  LongestDD {st['LongestDrawdownMonths']}mo  "
              f"Growth of $1 {_fmt(st['FinalGrowth'], False)}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="GEM dual momentum backtest")
    parser.add_argument("--start", default="1970-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--cost", type=float, default=DEFAULT_COST)
    parser.add_argument("--split", default=SPLIT_DATE)
    parser.add_argument("--out", default="gem_backtest.png")
    args = parser.parse_args()

    prices = build_price_panel(args.start, args.end)

    gem_net, held = run_gem(prices, lookback=args.lookback, cost=args.cost)
    buy_hold = prices["US"].pct_change()
    buy_hold.name = f"Buy & Hold ({US_TICKER})"
    blend = sixty_forty(prices)

    returns = {gem_net.name: gem_net, buy_hold.name: buy_hold, blend.name: blend}

    split = pd.Timestamp(args.split)
    in_sample = {name: s[s.index < split] for name, s in returns.items()}
    out_sample = {name: s[s.index >= split] for name, s in returns.items()}

    print(f"\n=== Split at {split.date()} ===")
    in_stats = report_window(f"In-Sample (before {split.date()})", in_sample)
    out_stats = report_window(f"Out-of-Sample ({split.date()} to present)", out_sample)

    print(f"\n=== Lookback Sensitivity (overfit check, full period, net of {args.cost:.2%} per switch) ===")
    lb_stats = {}
    for lb in LOOKBACK_GRID:
        net, _ = run_gem(prices, lookback=lb, cost=args.cost)
        lb_stats[lb] = performance_stats(net)
        st = lb_stats[lb]
        print(f"{lb:>2}mo: CAGR {_fmt(st['CAGR'])}  Sharpe {_fmt(st['Sharpe'], False)}  "
              f"MaxDD {_fmt(st['MaxDrawdown'])}")
    cagr_vals = [s["CAGR"] for s in lb_stats.values() if not pd.isna(s["CAGR"])]
    sharpe_vals = [s["Sharpe"] for s in lb_stats.values() if not pd.isna(s["Sharpe"])]
    if cagr_vals and sharpe_vals:
        print(f"Spread across lookbacks -> CAGR: {max(cagr_vals) - min(cagr_vals):.2%}, "
              f"Sharpe: {max(sharpe_vals) - min(sharpe_vals):.2f}")

    print("\nCurrent allocation signal (most recent month):")
    print(held.dropna().tail(1).to_string())

    gem_key, bh_key = gem_net.name, buy_hold.name
    in_beats = in_stats[gem_key]["Sharpe"] > in_stats[bh_key]["Sharpe"]
    out_beats = out_stats[gem_key]["Sharpe"] > out_stats[bh_key]["Sharpe"]
    verdict = "YES" if (in_beats and out_beats) else "NO"
    print(f"\nVERDICT: GEM beat buy-and-hold on a risk-adjusted (Sharpe) basis, net of costs, "
          f"in BOTH windows -> {verdict} "
          f"(in-sample: {'beat' if in_beats else 'lagged'}, "
          f"out-of-sample: {'beat' if out_beats else 'lagged'})")

    import matplotlib.pyplot as plt

    combined = pd.DataFrame(returns).dropna()
    growth = (1 + combined).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    growth.plot(ax=ax, logy=True)
    ax.axvline(split, color="gray", linestyle="--", linewidth=1, label="in/out-of-sample split")
    ax.set_title(f"GEM ({args.lookback}mo, net) vs. Buy & Hold and 60/40")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved equity curve chart to {args.out}")


if __name__ == "__main__":
    main()

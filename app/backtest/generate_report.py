"""
Generate a comprehensive HTML performance report from a backtest_results*
output directory (equity curves, drawdown, win/loss distribution, R-multiple
distribution, monthly returns, session/entry-type breakdown).

Usage:
    python generate_report.py --results-dir backtest_results_realistic \
        --out /path/to/report.html --title "Realistic Backtest (3m intrabar)"
"""
import argparse
import base64
import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Palette (dataviz skill reference instance) ───────────────────────────────
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4e0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_results(results_dir: str):
    with open(os.path.join(results_dir, "backtest_results.json")) as f:
        master = json.load(f)
    metrics = master["metrics"]

    equity, trades = {}, {}
    for sym in metrics:
        eq_path = os.path.join(results_dir, f"equity_{sym}.json")
        if os.path.exists(eq_path):
            with open(eq_path) as f:
                equity[sym] = json.load(f)
        tr_path = os.path.join(results_dir, f"trades_{sym}.csv")
        if os.path.exists(tr_path):
            df = pd.read_csv(tr_path)
            if not df.empty:
                df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
                trades[sym] = df
    return master, metrics, equity, trades


def _combined_equity(equity: dict, initial_per_symbol: float) -> pd.Series:
    """Sum per-symbol equity curves (each starts at initial_balance) into one
    portfolio curve: portfolio_pnl(t) = sum of each symbol's pnl-so-far(t),
    aligned by trade index (curves are event-indexed, not time-indexed, so
    this indexes by cumulative trade count per symbol — a reasonable proxy
    for "all symbols running in parallel with independent capital")."""
    max_len = max((len(v) for v in equity.values()), default=0)
    total = np.zeros(max_len)
    for sym, curve in equity.items():
        arr = np.array(curve, dtype=float)
        pnl = arr - arr[0]
        if len(pnl) < max_len:
            pnl = np.concatenate([pnl, np.full(max_len - len(pnl), pnl[-1])])
        total += pnl
    return pd.Series(total + initial_per_symbol * len(equity))


def chart_equity_curve(equity: dict, initial_balance: float) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    combined = _combined_equity(equity, initial_balance)
    ax.plot(combined.index, combined.values, color=CATEGORICAL[0], linewidth=1.8)
    ax.fill_between(combined.index, combined.values, combined.values.min(),
                    color=CATEGORICAL[0], alpha=0.08)
    ax.axhline(initial_balance * len(equity), color=TEXT_SECONDARY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Combined Portfolio Equity (all symbols, by trade sequence)", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (USD)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_drawdown(equity: dict, initial_balance: float) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    combined = _combined_equity(equity, initial_balance)
    running_peak = combined.cummax()
    dd = (combined - running_peak) / running_peak * 100
    ax.fill_between(dd.index, dd.values, 0, color=CRITICAL, alpha=0.25)
    ax.plot(dd.index, dd.values, color=CRITICAL, linewidth=1.2)
    ax.set_title("Portfolio Drawdown (%)", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown %")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig), float(dd.min())


def chart_per_symbol_equity(equity: dict) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (sym, curve) in enumerate(equity.items()):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        arr = np.array(curve, dtype=float)
        idx = np.linspace(0, 1, len(arr))
        ax.plot(idx, arr, color=color, linewidth=1.6, label=sym)
    ax.set_title("Per-Symbol Equity Curves (normalized by trade progress)", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("Trade progress (normalized)")
    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="upper left", frameon=False, ncol=4, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_win_loss_dist(all_trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    wins = all_trades[all_trades["result"] == "WIN"]["pnl"]
    losses = all_trades[all_trades["result"] == "LOSS"]["pnl"]
    bins = np.linspace(min(all_trades["pnl"].min(), -1), max(all_trades["pnl"].max(), 1), 40)
    ax.hist(wins, bins=bins, color=GOOD, alpha=0.75, label=f"Wins ({len(wins)})")
    ax.hist(losses, bins=bins, color=CRITICAL, alpha=0.75, label=f"Losses ({len(losses)})")
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("PnL Distribution per Trade", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("PnL (USD)")
    ax.set_ylabel("Trade count")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_r_multiple_dist(all_trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    r = all_trades["realized_r"].dropna()
    r = r[(r > -5) & (r < 5)]
    ax.hist(r, bins=40, color=CATEGORICAL[0], alpha=0.85)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.axvline(r.mean(), color=CRITICAL, linewidth=1.2, linestyle="--", label=f"mean={r.mean():.2f}R")
    ax.set_title("Realized R-Multiple Distribution", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("R multiple")
    ax.set_ylabel("Trade count")
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_monthly_pnl(all_trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    df = all_trades.dropna(subset=["exit_time"]).copy()
    df["month"] = df["exit_time"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["pnl"].sum()
    colors = [GOOD if v >= 0 else CRITICAL for v in monthly.values]
    ax.bar(monthly.index, monthly.values, color=colors, width=0.6)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("Net PnL by Month (all symbols combined)", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_ylabel("Net PnL (USD)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_by_market_state(all_trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    g = all_trades.groupby("market_state").agg(
        trades=("pnl", "count"), pnl=("pnl", "sum"),
        win_rate=("result", lambda s: (s == "WIN").mean() * 100),
    ).sort_values("pnl", ascending=False)
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(g))]
    ax2 = ax.twinx()
    ax.bar(g.index, g["pnl"], color=colors, alpha=0.85, width=0.6)
    ax2.plot(g.index, g["win_rate"], color=TEXT_PRIMARY, marker="o", markersize=5, linewidth=1.2)
    ax.set_title("PnL & Win-Rate by Market State", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_ylabel("Net PnL (USD)")
    ax2.set_ylabel("Win rate %")
    ax2.set_ylim(0, 100)
    ax.axhline(0, color=GRID, linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_by_entry_type(all_trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    g = all_trades.groupby("entry_type").agg(
        trades=("pnl", "count"), pnl=("pnl", "sum"),
    ).sort_values("pnl", ascending=False)
    colors = [GOOD if v >= 0 else CRITICAL for v in g["pnl"].values]
    ax.barh(g.index, g["pnl"], color=colors)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("PnL by Entry Type", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("Net PnL (USD)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def chart_hourly_heat(all_trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 2.6))
    g = all_trades.groupby("hour_utc")["pnl"].sum().reindex(range(24), fill_value=0)
    colors = [GOOD if v >= 0 else CRITICAL for v in g.values]
    ax.bar(g.index, g.values, color=colors, width=0.7)
    ax.set_title("Net PnL by Entry Hour (UTC) — Session Behavior", loc="left", fontsize=12, color=TEXT_PRIMARY)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Net PnL")
    ax.set_xticks(range(0, 24, 2))
    ax.axhline(0, color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return _fig_to_data_uri(fig)


def build_report(results_dir: str, out_path: str, title: str, initial_balance: float):
    master, metrics, equity, trades = _load_results(results_dir)
    all_trades = pd.concat(trades.values(), ignore_index=True) if trades else pd.DataFrame()

    charts = {}
    if equity:
        charts["equity"] = chart_equity_curve(equity, initial_balance)
        charts["drawdown"], portfolio_maxdd = chart_drawdown(equity, initial_balance)
        charts["per_symbol_equity"] = chart_per_symbol_equity(equity)
    else:
        portfolio_maxdd = 0.0
    if not all_trades.empty:
        charts["win_loss"] = chart_win_loss_dist(all_trades)
        charts["r_multiple"] = chart_r_multiple_dist(all_trades)
        charts["monthly"] = chart_monthly_pnl(all_trades)
        charts["by_state"] = chart_by_market_state(all_trades)
        charts["by_entry_type"] = chart_by_entry_type(all_trades)
        charts["hourly"] = chart_hourly_heat(all_trades)

    total_trades = sum(m.get("total_trades", 0) for m in metrics.values())
    total_pnl = sum(m.get("net_pnl", 0) for m in metrics.values())
    total_wins = sum(round(m.get("total_trades", 0) * m.get("win_rate", 0)) for m in metrics.values())
    overall_wr = (total_wins / total_trades * 100) if total_trades else 0
    total_commission = sum(m.get("total_commission", 0) for m in metrics.values())

    rows = ""
    for sym, m in sorted(metrics.items(), key=lambda kv: -kv[1].get("net_pnl", 0) if "net_pnl" in kv[1] else 0):
        if m.get("total_trades", 0) == 0:
            rows += f'<tr><td>{sym}</td><td colspan="7" class="muted">no trades / error</td></tr>'
            continue
        pnl_class = "good" if m["net_pnl"] >= 0 else "critical"
        rows += f"""<tr>
            <td><strong>{sym}</strong></td>
            <td>{m['total_trades']}</td>
            <td>{m['win_rate']*100:.1f}%</td>
            <td class="{pnl_class}">${m['net_pnl']:,.2f}</td>
            <td>{m['profit_factor']:.2f}</td>
            <td>{m['max_drawdown_pct']:.1f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td>{m.get('avg_bars_held', 0):.1f}</td>
        </tr>"""

    def img(key, alt=""):
        return f'<img src="{charts[key]}" alt="{alt}" class="chart-img"/>' if key in charts else ""

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{
    --surface: {SURFACE}; --text-primary: {TEXT_PRIMARY}; --text-secondary: {TEXT_SECONDARY};
    --grid: {GRID}; --good: {GOOD}; --critical: {CRITICAL};
  }}
  /* Deliberately no dark-mode override: the chart panels below are
     matplotlib-rendered SVGs baked with a fixed light surface — a
     light/dark split page shell would look broken against them. One
     consistent light theme throughout instead. */
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 64px; background: var(--surface); color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 1180px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--text-secondary); margin-bottom: 28px; font-size: 0.92rem; }}
  .stat-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 32px; }}
  .stat-tile {{ background: color-mix(in srgb, var(--text-primary) 4%, var(--surface)); border: 1px solid var(--grid); border-radius: 10px; padding: 14px 16px; }}
  .stat-tile .label {{ color: var(--text-secondary); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat-tile .value {{ font-size: 1.5rem; font-weight: 650; margin-top: 4px; }}
  section {{ margin-bottom: 40px; }}
  section h2 {{ font-size: 1.05rem; border-bottom: 1px solid var(--grid); padding-bottom: 8px; margin-bottom: 16px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .chart-grid.full {{ grid-template-columns: 1fr; }}
  .chart-img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--grid); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: var(--text-secondary); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  .good {{ color: var(--good); font-weight: 600; }}
  .critical {{ color: var(--critical); font-weight: 600; }}
  .muted {{ color: var(--text-secondary); text-align: left; }}
  .overflow-x {{ overflow-x: auto; }}
  footer {{ color: var(--text-secondary); font-size: 0.8rem; margin-top: 40px; border-top: 1px solid var(--grid); padding-top: 16px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · {len(metrics)} symbols · {total_trades} total trades</div>

  <div class="stat-row">
    <div class="stat-tile"><div class="label">Total Trades</div><div class="value">{total_trades:,}</div></div>
    <div class="stat-tile"><div class="label">Overall Win Rate</div><div class="value">{overall_wr:.1f}%</div></div>
    <div class="stat-tile"><div class="label">Total Net PnL</div><div class="value {'good' if total_pnl>=0 else 'critical'}">${total_pnl:,.2f}</div></div>
    <div class="stat-tile"><div class="label">Portfolio MaxDD</div><div class="value critical">{portfolio_maxdd:.1f}%</div></div>
    <div class="stat-tile"><div class="label">Total Commission</div><div class="value">${total_commission:,.2f}</div></div>
  </div>

  <section>
    <h2>Per-Symbol Summary</h2>
    <div class="overflow-x">
    <table>
      <thead><tr><th>Symbol</th><th>Trades</th><th>Win Rate</th><th>Net PnL</th><th>PF</th><th>MaxDD</th><th>Sharpe</th><th>Avg Bars Held</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
  </section>

  <section>
    <h2>Equity &amp; Drawdown</h2>
    <div class="chart-grid full">
      {img('equity', 'Combined portfolio equity curve')}
      {img('drawdown', 'Portfolio drawdown')}
      {img('per_symbol_equity', 'Per-symbol equity curves')}
    </div>
  </section>

  <section>
    <h2>Trade Distribution</h2>
    <div class="chart-grid">
      {img('win_loss', 'Win/loss PnL distribution')}
      {img('r_multiple', 'R-multiple distribution')}
    </div>
  </section>

  <section>
    <h2>Time-Based Breakdown</h2>
    <div class="chart-grid full">
      {img('monthly', 'Monthly PnL')}
      {img('hourly', 'PnL by entry hour (session behavior)')}
    </div>
  </section>

  <section>
    <h2>Regime &amp; Strategy Breakdown</h2>
    <div class="chart-grid">
      {img('by_state', 'PnL and win-rate by market state')}
      {img('by_entry_type', 'PnL by entry type')}
    </div>
  </section>

  <footer>
    Realistic backtest: 15m bar-close signal generation + 3m intrabar polling
    (matches the live runner's check_price_protection cadence). Commission
    {master['config'].get('commission_pct', 0)*100:.3f}% · Slippage
    {master['config'].get('slippage_pct', 0)*100:.3f}% per side.
  </footer>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Adaptive Bot — Realistic Backtest Report")
    parser.add_argument("--balance", type=float, default=10_000.0)
    args = parser.parse_args()
    build_report(args.results_dir, args.out, args.title, args.balance)

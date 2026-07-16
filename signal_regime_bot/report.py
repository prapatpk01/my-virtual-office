"""Performance reporting — stats + a self-contained dark-themed HTML report."""
from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def compute_stats(trades: list, initial_balance: float) -> dict:
    if not trades:
        return {"total_trades": 0}

    pnl = [t.pnl_usd for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]

    running = initial_balance
    eq = [initial_balance]
    for t in sorted(trades, key=lambda x: x.exit_time or pd.Timestamp.min.tz_localize("UTC")):
        running += t.pnl_usd
        eq.append(running)
    peak = initial_balance
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100 if peak > 0 else 0)

    total_pnl = sum(pnl)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    longs = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]

    n_months = len(set((t.exit_time.year, t.exit_time.month) for t in trades if t.exit_time)) or 1

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "net_pnl": total_pnl,
        "total_return_pct": total_pnl / initial_balance * 100,
        "max_drawdown_pct": max_dd,
        "avg_r": np.mean([t.r_multiple for t in trades]),
        "monthly_trades": len(trades) / n_months,
        "long_win_rate": (sum(1 for t in longs if t.pnl_usd > 0) / len(longs) * 100) if longs else 0.0,
        "short_win_rate": (sum(1 for t in shorts if t.pnl_usd > 0) / len(shorts) * 100) if shorts else 0.0,
        "long_count": len(longs),
        "short_count": len(shorts),
        "tp1_hit_count": sum(1 for t in trades if t.tp1_hit),
        "sl_count": sum(1 for t in trades if t.exit_reason == "SL"),
        "be_count": sum(1 for t in trades if t.exit_reason == "BE"),
        "tp2_count": sum(1 for t in trades if t.exit_reason == "TP2"),
        "early_exit_count": sum(1 for t in trades if t.exit_reason in
                                ("EMA_CROSS_REVERSAL", "PRICE_OPEN_BEYOND_EMA")),
    }


def per_symbol_stats(trades: list) -> pd.DataFrame:
    rows = []
    for sym in sorted(set(t.symbol for t in trades)):
        st = [t for t in trades if t.symbol == sym]
        wins = [t for t in st if t.pnl_usd > 0]
        rows.append({
            "Symbol": sym, "Trades": len(st),
            "Win Rate %": round(len(wins) / len(st) * 100, 1) if st else 0,
            "Net PnL $": round(sum(t.pnl_usd for t in st), 2),
            "Avg R": round(np.mean([t.r_multiple for t in st]), 2) if st else 0,
        })
    return pd.DataFrame(rows)


def regime_stats(trades: list) -> pd.DataFrame:
    rows = []
    for regime in sorted(set(t.regime_at_entry for t in trades)):
        st = [t for t in trades if t.regime_at_entry == regime]
        wins = [t for t in st if t.pnl_usd > 0]
        rows.append({
            "Regime": regime, "Trades": len(st),
            "Win Rate %": round(len(wins) / len(st) * 100, 1) if st else 0,
            "Net PnL $": round(sum(t.pnl_usd for t in st), 2),
        })
    return pd.DataFrame(rows)


def bias_stats(trades: list) -> pd.DataFrame:
    rows = []
    for bias in sorted(set(t.bias_at_entry for t in trades)):
        st = [t for t in trades if t.bias_at_entry == bias]
        wins = [t for t in st if t.pnl_usd > 0]
        rows.append({
            "Bias": bias, "Trades": len(st),
            "Win Rate %": round(len(wins) / len(st) * 100, 1) if st else 0,
            "Net PnL $": round(sum(t.pnl_usd for t in st), 2),
        })
    return pd.DataFrame(rows)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    plt.close(fig)
    return b64


def _dark_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")


def chart_equity(trades: list, initial_balance: float) -> str:
    if not HAS_MPL or not trades:
        return ""
    running = initial_balance
    eq = [initial_balance]
    for t in sorted(trades, key=lambda x: x.exit_time or pd.Timestamp.min.tz_localize("UTC")):
        running += t.pnl_usd
        eq.append(running)
    peak = np.maximum.accumulate(eq)
    dd = (peak - np.array(eq)) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax1); _dark_ax(ax2)
    ax1.plot(eq, color="#58a6ff", lw=1.6)
    ax1.axhline(initial_balance, color="#484f58", lw=0.8, ls="--")
    ax1.set_title("Equity Curve", color="#e6edf3")
    ax2.fill_between(range(len(dd)), 0, -dd, color="#f85149", alpha=0.6)
    ax2.set_title("Drawdown %", color="#8b949e")
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_breakdown(df: pd.DataFrame, label_col: str, title: str) -> str:
    if not HAS_MPL or df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax)
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in df["Net PnL $"]]
    ax.bar(df[label_col], df["Net PnL $"], color=colors)
    ax.set_title(title, color="#e6edf3")
    ax.axhline(0, color="#484f58", lw=0.8)
    plt.tight_layout()
    return _fig_to_b64(fig)


def build_html_report(trades: list, stats: dict, sym_df: pd.DataFrame, regime_df: pd.DataFrame,
                      bias_df: pd.DataFrame, initial_balance: float) -> str:
    def img(b64, title=""):
        if not b64:
            return "<p style='color:#8b949e'>chart unavailable</p>"
        return f'<figure><figcaption>{title}</figcaption><img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px"/></figure>'

    def table(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No data.</p>"
        head = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
        body = ""
        for _, row in df.iterrows():
            body += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        return f'<table class="data-table"><thead>{head}</thead><tbody>{body}</tbody></table>'

    cards = "".join(
        f'<div class="card"><div class="card-label">{k}</div><div class="card-value">{v}</div></div>'
        for k, v in [
            ("Total Trades", stats.get("total_trades", 0)),
            ("Win Rate", f"{stats.get('win_rate', 0):.1f}%"),
            ("Profit Factor", f"{stats.get('profit_factor', 0):.2f}"),
            ("Net PnL", f"${stats.get('net_pnl', 0):+,.0f}"),
            ("Total Return", f"{stats.get('total_return_pct', 0):+.1f}%"),
            ("Max Drawdown", f"{stats.get('max_drawdown_pct', 0):.1f}%"),
            ("Avg R", f"{stats.get('avg_r', 0):.2f}"),
            ("Monthly Trades", f"{stats.get('monthly_trades', 0):.1f}"),
            ("Long WR", f"{stats.get('long_win_rate', 0):.1f}% ({stats.get('long_count',0)})"),
            ("Short WR", f"{stats.get('short_win_rate', 0):.1f}% ({stats.get('short_count',0)})"),
        ]
    )

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background:#0d1117; color:#e6edf3; font-family:-apple-system,sans-serif; padding:24px; }
    h1 { color:#58a6ff; } h2 { color:#8b949e; margin:24px 0 12px; border-bottom:1px solid #30363d; padding-bottom:6px; }
    .cards { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }
    .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 18px; min-width:130px; flex:1; }
    .card-label { font-size:.72rem; color:#8b949e; text-transform:uppercase; }
    .card-value { font-size:1.3rem; font-weight:700; }
    .data-table { width:100%; border-collapse:collapse; font-size:.85rem; }
    .data-table th,.data-table td { padding:6px 10px; border:1px solid #21262d; text-align:right; }
    .data-table th { background:#161b22; color:#8b949e; text-align:center; }
    figure { margin:0 0 20px; } figcaption { color:#8b949e; font-size:.8rem; margin-bottom:6px; }
    """
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Signal Regime Bias Backtest</title><style>{css}</style></head><body>
<h1>Signal Regime Bias Strategy — Backtest Report</h1>
<div class="cards">{cards}</div>
{img(chart_equity(trades, initial_balance), "Equity Curve")}
<h2>Per-Symbol</h2>{table(sym_df)}
<h2>Regime Performance</h2>{table(regime_df)}
{img(chart_breakdown(regime_df, "Regime", "PnL by Regime"))}
<h2>Bias Performance</h2>{table(bias_df)}
{img(chart_breakdown(bias_df, "Bias", "PnL by Bias"))}
</body></html>"""

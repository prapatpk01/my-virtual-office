#!/usr/bin/env bash
# Quantum Capital Fund — startup script
# Usage: ./start-fund.sh [paper|live]

set -e

ENV="${1:-paper}"
CONFIG="$(dirname "$0")/portfolio-config.json"

echo "╔══════════════════════════════════════╗"
echo "║     QUANTUM CAPITAL FUND             ║"
echo "║     Environment: ${ENV^^}                 ║"
echo "╚══════════════════════════════════════╝"

# Verify required env vars
check_env() {
  if [ -z "${!1}" ]; then
    echo "❌  Missing required env var: $1"
    echo "    Set it with: export $1=<value>"
    exit 1
  else
    echo "✅  $1 is set"
  fi
}

echo ""
echo "── Checking API credentials ──"
check_env ALPACA_API_KEY
check_env ALPACA_API_SECRET
check_env FINNHUB_API_KEY

echo ""
echo "── Trading Universe ──"
python3 - <<'PY'
import json, os
with open(os.path.join(os.path.dirname(os.path.abspath("$0")), "portfolio-config.json")) as f:
    cfg = json.load(f)
universe = cfg["trading"]["universe"]
print(f"  {len(universe)} symbols: {', '.join(universe)}")
print(f"  Base notional/trade: ${cfg['trading']['baseNotionalPerTrade']:,}")
print(f"  Max position size:   {cfg['trading']['maxPositionPct']*100:.0f}% of portfolio")
print(f"  Max drawdown limit:  {cfg['trading']['maxDrawdownPct']*100:.0f}%")
PY

echo ""
echo "── Teams Online ──"
echo "  👑 Fund Manager:  Victoria Chen"
echo "  📊 Research:      Dr. Emily Zhao, Marcus Webb, Nina Patel"
echo "  ⚙️  Quant:         Kenji Tanaka, Aisha Okonkwo"
echo "  ⚠️  Risk:          Sam Rivera, Chris Morgan"
echo "  ⚡ Execution:     Jordan Lee, Taylor Kim"

echo ""
echo "── Daily Cycle Tools ──"
echo "  hf_research_brief   → market sentiment & news"
echo "  hf_generate_signals → RSI + MACD + sentiment signals"
echo "  hf_risk_review      → position sizing & risk gates"
echo "  hf_execute_trades   → Alpaca order submission"
echo "  hf_run_daily_cycle  → full automated pipeline"

echo ""
echo "── Quick Start ──"
echo "  Dry run (safe):  hf_run_daily_cycle mode=full_dry_run"
echo "  Live trading:    hf_run_daily_cycle mode=full confirm=true"

echo ""
echo "🟢  Fund office ready. Open virtual office to see agents at their desks."

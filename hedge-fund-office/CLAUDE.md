# Quantum Capital Fund — Agent Guide

You are an agent in the **Quantum Capital Fund** virtual office. Use this guide to run fund operations.

## Team & Roles

| Agent | Name | Role | Desk |
|-------|------|------|------|
| `hf-manager` | Victoria Chen | Fund Manager | Trading Command Center (center) |
| `hf-research-1` | Dr. Emily Zhao | Research Analyst | Research Wing (left) |
| `hf-research-2` | Marcus Webb | Research Analyst | Research Wing (left) |
| `hf-research-3` | Nina Patel | Senior Analyst | Research Wing (left) |
| `hf-quant-1` | Kenji Tanaka | Quant Analyst | Trading Command Center |
| `hf-quant-2` | Aisha Okonkwo | Quant Analyst | Trading Command Center |
| `hf-risk-1` | Sam Rivera | Risk Manager | Risk & Execution Wing (right) |
| `hf-risk-2` | Chris Morgan | Risk Analyst | Risk & Execution Wing (right) |
| `hf-execution-1` | Jordan Lee | Execution Trader | Risk & Execution Wing (right) |
| `hf-execution-2` | Taylor Kim | Execution Trader | Risk & Execution Wing (right) |

## Fund Configuration

See `portfolio-config.json` for full settings. Key parameters:

- **Universe**: 17 symbols — AAPL MSFT GOOGL AMZN NVDA META TSLA JPM V UNH SPY QQQ GLD TLT IWM BTCUSD ETHUSD
- **Base notional/trade**: $2,000 (scaled by confidence score)
- **Max position size**: 12% of portfolio
- **Drawdown halt**: 8% max drawdown

## Daily Cycle — Quick Reference

### 1. Full dry run (safe — no orders placed)
```
hf_run_daily_cycle
  mode: "full_dry_run"
```

### 2. Research only
```
hf_run_daily_cycle
  mode: "research_only"
  symbols: ["AAPL", "NVDA", "BTCUSD"]   ← optional override
```

### 3. Signals only (Research + Quant)
```
hf_run_daily_cycle
  mode: "signals_only"
```

### 4. Live trading (requires confirm)
```
hf_run_daily_cycle
  mode: "full"
  confirm: true
```

## Individual Tools

| Tool | Team | What it does |
|------|------|-------------|
| `hf_research_brief` | Research | News, sentiment, snapshots for each symbol |
| `hf_generate_signals` | Quant | RSI-14 + MACD + sentiment → BUY/SELL/HOLD |
| `hf_risk_review` | Risk | Position sizing, drawdown check, approval |
| `hf_execute_trades` | Execution | Submit market orders via Alpaca |
| `hf_get_fund_status` | All | Portfolio value, positions, P&L |
| `hf_set_fund_config` | Manager | Update universe, risk params, notional |

## Adjust Fund Parameters

```
hf_set_fund_config
  universe: ["AAPL", "NVDA", "GLD", "BTCUSD"]
  max_position_pct: 0.10
  base_notional_per_trade: 1500
```

## Required Environment Variables

```bash
export ALPACA_API_KEY=<your-key>
export ALPACA_API_SECRET=<your-secret>
export FINNHUB_API_KEY=<your-key>
```

Start with `alpacaEnv: "paper"` (default) until validated on paper trading.

## Safety Rules

1. Always run `full_dry_run` before `full` on a new day
2. Review `hf_risk_review` output — check `tradingHalted` and `riskFlags`
3. Never skip `confirm: true` gate — it is there for a reason
4. SELL signals only execute if a long position exists (no accidental short selling)
5. Max drawdown > 8% automatically halts all trading

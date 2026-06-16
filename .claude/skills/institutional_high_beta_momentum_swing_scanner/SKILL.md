---
name: Institutional High-Beta Momentum Swing Scanner
description: >
  Scans the US equity market to identify the top 5 highest-conviction swing
  trading opportunities for a 7–15 trading day holding period. Uses web search
  to verify real-time prices, technicals, and catalysts. Applies a 7-phase
  institutional momentum framework covering market regime, momentum alpha,
  volume accumulation, trend structure, high-beta expansion, sector strength,
  and catalyst drift. Returns trade entries, stops, targets, and R:R ratios
  with full swing thesis for each pick. Suitable for momentum hedge fund,
  prop trading, or active swing trader use cases.
version: "6.1"
author: Quantitative Strategy Desk
tags:
  - momentum
  - swing trading
  - technical analysis
  - institutional
  - stocks
  - equity
  - high-beta
  - AI
  - semiconductors
---

# INSTITUTIONAL HIGH-BETA MOMENTUM SWING SCANNER (v6.1 — AI WEB-SEARCH OPTIMIZED)

## ROLE

Act as a Tier-1 Hedge Fund Quantitative Analyst, Momentum Portfolio Manager,
Institutional Flow Analyst, Technical Strategist, and Risk Manager.

Your objective is to scan the entire US equity market and identify EXACTLY 5
highest-conviction swing trading opportunities for a 7–15 trading day holding period.

Use today's current date as the reference trading day.

Use the most recent verifiable market data available via web search.

The goal is NOT to find value stocks.

The goal is to identify stocks with the highest probability of producing a
10%–30% move within the next 1–3 weeks.

Focus exclusively on institutional-quality momentum setups.

---

## MANDATORY PRE-EXECUTION STEP — WEB SEARCH PROTOCOL

Before running any phase, execute these web searches in order:

1. Search: `SPY QQQ VIX price today [current date]` → establish market regime
2. Search: `top momentum stocks breakout high volume [current date]` → initial candidate list
3. Search: `[TICKER] stock price RSI volume [current month year]` → per-stock verification
4. Search: `[TICKER] recent earnings catalyst news [current date]` → catalyst check

**Acceptable Data Sources (in priority order):**

1. Exchange real-time quote (Nasdaq, NYSE)
2. Broker real-time feed (Robinhood, Schwab live quote pages)
3. Financial data sites with date-stamped quotes:
   - Investing.com (with timestamp)
   - Barchart.com (with session date)
   - CNN Markets (with session date)
   - StockAnalysis.com
   - Finviz.com
4. TradingView posts referencing today's price
5. Yahoo Finance / Robinhood (with visible date)

**Data Freshness Standard:**

- ACCEPT: Price confirmed from current trading session or most recent session close
- ACCEPT: Pre-market or after-hours quote with visible timestamp from current date
- ACCEPT: Web search result that explicitly states today's date AND a price
- REJECT: Price with no date reference whatsoever
- REJECT: Price from more than 2 trading sessions ago
- REJECT: Analyst price targets used as current price
- REJECT: Estimated or inferred prices with no source

**State clearly for each stock:**
> Source: [site name] | Date confirmed: [date] | Price: $[X.XX]

---

## PHASE 1 — DATA QUALITY CONTROL

After web search, verify for each candidate:

| Data Point | Requirement | If Missing |
|---|---|---|
| Current Price | From acceptable source above | Exclude stock |
| Date Confirmation | Current or prior session | Exclude stock |
| Daily Volume | Current or recent session | Note as PARTIAL DATA |
| 20D Avg Volume | From any technical site | Note as PARTIAL DATA |
| Market Cap | Any source | Note as PARTIAL DATA |
| RSI(14) | From Barchart / Investing.com / Altindex | Note as PARTIAL DATA |
| MACD | From any technical analysis site | Note as PARTIAL DATA |
| Beta | From any financial data source | Note as PARTIAL DATA |
| ATR% | Calculate or source from Barchart | Note as PARTIAL DATA |

**PARTIAL DATA stocks may still qualify** if price is confirmed and
at least 4 of the remaining 7 data points are available.

Mark each missing field clearly as: `DATA UNAVAILABLE`

Do not estimate or fabricate any field marked unavailable.

---

## PHASE 2 — MARKET REGIME FILTER

Search for and evaluate:

- SPY price vs 20 EMA → search: "SPY 20 EMA today"
- QQQ price vs 20 EMA → search: "QQQ 20 EMA today"
- VIX level → search: "VIX index today"
- RSP vs SPY → search: "RSP vs SPY breadth today"
- Sector leadership → search: "sector rotation today best performing"

**Assign Market Regime Score (0–100):**

| Score | Classification |
|---|---|
| 80–100 | Strong Risk-On |
| 60–79 | Risk-On |
| 40–59 | Neutral |
| 20–39 | Risk-Off |
| 0–19 | Defensive |

**Regime Rules:**

- IF SPY > 20 EMA AND QQQ > 20 EMA AND VIX < 20 → accept all qualifying setups
- IF SPY < 20 EMA OR QQQ < 20 EMA OR VIX > 20 → accept only sector leaders and
  relative strength outliers; reject average setups
- IF VIX > 30 → pause scanner; market in dislocation; reduce position sizing 50%

---

## PHASE 3 — MOMENTUM ALPHA MODEL

Score each candidate across 6 dimensions:

### A. MOMENTUM & RELATIVE STRENGTH — 35%

**Requirements (all must be met):**
- RSI(14) between 55 and 78
- MACD above zero line
- MACD histogram expanding (not rolling over)
- 20D return outperforms SPY
- 20D return outperforms QQQ
- Price making higher highs and higher lows

**Bonus:** ATR expansion; strong weekly trend above 10 EMA

**Auto-Reject:**
- RSI > 80 (parabolic extension risk)
- Bearish MACD divergence
- MACD histogram rolling over 3+ bars
- Momentum decelerating vs 5D ago

---

### B. VOLUME ACCUMULATION — 25%

**Requirements:**
- Current session volume > 1.5× 20D average volume
- 5D average volume > 20D average volume
- Multiple prior accumulation days visible
- OBV trend rising
- A/D line rising

**Bonus:** Volume > 2× average = maximum score

**Auto-Reject:**
- Isolated single-day spike with no follow-through
- Distribution patterns (price down on heavy volume)
- OBV declining while price rises

---

### C. STRUCTURE & TREND QUALITY — 15%

**Price must be above:** 10 EMA, 20 EMA, and 50 SMA

**Alignment required:** 10 EMA > 20 EMA > 50 SMA

**Preferred patterns:**
1. Volatility Contraction Pattern (VCP)
2. High Tight Flag
3. Bull Flag
4. Flat Base
5. Ascending Base
6. Cup & Handle

**Auto-Reject:** Broken bases, failed breakouts, late-stage bases (3rd or 4th)

---

### D. HIGH-BETA EXPANSION FILTER — 10%

**Requirements:**
- Beta > 1.3
- ATR% > 3% (daily ATR as % of price)
- Average daily dollar volume > $50M

If Beta is DATA UNAVAILABLE but stock has moved >50% in 12 months, qualifies
conditionally — note in output.

---

### E. SECTOR STRENGTH — 5%

**Preferred themes (priority order):**
1. AI Infrastructure & Custom Silicon
2. Semiconductors & Chip Equipment
3. Defense Technology & Autonomy
4. Quantum Computing
5. Cybersecurity
6. Cloud Infrastructure & Data Centers
7. Nuclear & Alternative Energy
8. Robotics & Autonomous Systems
9. AI Software & Platforms
10. Aerospace & Space

Only consider sector leaders — top 1–3 stocks in their category.

---

### F. CATALYST DRIFT — 10%

**Identify at least one of:**
- Post-Earnings Announcement Drift (PEAD) — beat within prior 10 sessions
- Positive EPS / revenue estimate revisions (3+ analysts)
- Major product launch or technology demonstration
- Large contract win or strategic partnership
- AI adoption or hyperscaler partnership announcement
- Regulatory approval or government designation
- Capacity expansion or production ramp

**Catalyst Horizon:** Must have at least 2–4 weeks of runway remaining.

---

## PHASE 4 — EXTENSION FILTER

**Auto-reject if ANY of the following:**
- Price > 10% above 10 EMA
- Price gained > 20% in last 5 sessions without consolidation
- Parabolic vertical move (gap-up > 15% with continued straight-line advance)

**Mark as:** `EXTENDED — WAIT FOR PULLBACK TO 10 EMA`

*Note: A gap-up of 10–15% on earnings followed by consolidation near 10 EMA
is NOT extended — it is a valid PEAD setup.*

---

## PHASE 5 — EARNINGS RISK FILTER

Search: "[TICKER] earnings date [current month/year]"

- Reject stocks reporting earnings within next 5 trading days
- Exception: If earnings already occurred within last 10 sessions (PEAD setup)
- If earnings date DATA UNAVAILABLE → proceed with caution; flag in output

---

## PHASE 6 — TRADE CONSTRUCTION

**Entry Zone** — must be one of:
- Within 3% of 10 EMA (pullback entry)
- Within 3% of breakout pivot level
- First consolidation bar after earnings gap (PEAD entry)

**Stop Loss** — place below:
- 20 EMA (primary)
- Structural support / base low (secondary)
- Breakout failure level (if applicable)

**Price Target** — calculate using:
- Fibonacci Extension 1.618 from swing low to entry
- Measured move projection from pattern depth
- Prior momentum expansion analog

**Required:** Minimum 10% upside from entry | Minimum R:R 1:3 | Preferred 1:4+

Reject any trade where R:R < 1:3 after stops and targets are defined.

---

## PHASE 7 — MOMENTUM ALPHA RANKING ENGINE

| Dimension | Weight |
|---|---|
| Momentum & Relative Strength | 35% |
| Volume Accumulation | 25% |
| Structure & Trend Quality | 15% |
| High-Beta Expansion | 10% |
| Catalyst Drift | 10% |
| Sector Strength | 5% |

Rank all qualifying candidates. Return ONLY the TOP 5.
If fewer than 5 qualify, return only qualified candidates — do not force 5.

---

## OUTPUT FORMAT

```
════════════════════════════════════════════════
MARKET REGIME — [DATE]
════════════════════════════════════════════════
Market Score:          [0–100]
SPY Trend:             [Above/Below 20 EMA + price]
QQQ Trend:             [Above/Below 20 EMA + price]
VIX:                   [Level + trend]
RSP vs SPY:            [Outperform / Underperform / Neutral]
Market Classification: [Strong Risk-On / Risk-On / Neutral / Risk-Off / Defensive]
Institutional Rotation:[Leading sectors]

════════════════════════════════════════════════
#[N] — [TICKER] ([COMPANY NAME])
════════════════════════════════════════════════
Setup Type:            [Pattern name]
Momentum Alpha Score:  [0–100]
Expected Return:       [X%–Y%]
Win Probability:       [X%]

Timeframe  | Risk:Reward | Entry      | Target     | Stop
-----------|-------------|------------|------------|----------
7–15 days  | 1:[X]       | $[X]–$[Y]  | $[X]–$[Y]  | $[X]

─── PRICE DATA ──────────────────────────────────
Current Price:     $[X.XX]
Source:            [Site name]
Date Confirmed:    [Session date]
Market Session:    [Regular / Pre-Market / After-Hours]

─── TECHNICAL PROFILE ───────────────────────────
RSI(14):           [Value or DATA UNAVAILABLE]
MACD Status:       [Bullish/Bearish + detail]
ATR%:              [Value or DATA UNAVAILABLE]
Beta:              [Value or DATA UNAVAILABLE]
Volume vs 20D Avg: [Xx average or DATA UNAVAILABLE]
OBV Trend:         [Rising / Falling / Flat]
A/D Trend:         [Accumulation / Distribution / Neutral]

─── RELATIVE STRENGTH ───────────────────────────
20D Return:        [+X%]
vs SPY (20D):      [+X% / -X%]
vs QQQ (20D):      [+X% / -X%]

─── VOLUME PROFILE ──────────────────────────────
5D vs 20D Volume:  [Xx or DATA UNAVAILABLE]
Accumulation:      [Confirmed / Partial / Unconfirmed]

─── PRICE STRUCTURE ─────────────────────────────
Pattern:           [Name]
10 EMA:            $[X] (Price [above/below])
20 EMA:            $[X] (Price [above/below])
50 SMA:            $[X] (Price [above/below])

─── CATALYST ────────────────────────────────────
Primary Catalyst:  [Description]
Catalyst Horizon:  [X weeks remaining]
Sector Theme:      [AI / Semis / Defense / etc.]

─── SWING THESIS ────────────────────────────────
• Why momentum is sustainable
• Why institutions are likely involved
• Why the setup has superior expectancy
• Trade management logic
• Conditions that invalidate the thesis

════════════════════════════════════════════════
[Repeat for all 5 picks]
════════════════════════════════════════════════

FINAL SUMMARY
─────────────────────────────────────────────
Top Conviction Pick:       [TICKER] — [reason]
Top Risk-Adjusted Pick:    [TICKER] — [reason]
Top High-Beta Aggressive:  [TICKER] — [reason]
Top PEAD Pick:             [TICKER] — [reason]
Top Sector Leader:         [TICKER] — [reason]

FINAL FUND MANAGER DECISION
─────────────────────────────────────────────
1. Largest allocation:   [TICKER]
2. Allocation %:         [X%]
3. Conviction rationale: [2–3 sentences]
4. Invalidation event:   [Specific trigger]
```

---

## ANTI-HALLUCINATION RULES (STRICT)

**Never fabricate:**
- Price (must be web search confirmed with date)
- Volume / RSI / MACD / ATR / Beta (use DATA UNAVAILABLE if not found)
- Earnings date (search to confirm; do not guess)
- Catalyst details (must be sourced)

**When data is unavailable:**
- Write `DATA UNAVAILABLE` in that field
- Do not estimate, interpolate, or infer the value
- Stock may still qualify if price is confirmed + 4 other fields available

**Accuracy > Completeness.**
Return fewer than 5 picks if fewer than 5 qualify.

---

## VERSION NOTES (v6.0 → v6.1)

| Change | Reason |
|---|---|
| YAML frontmatter added | Required for Claude Skills upload |
| Phase 0 replaced with Web Search Protocol | Claude uses web search, not live exchange feeds |
| Acceptable sources list added | Removes ambiguity about usable data |
| RSI threshold widened to 55–78 (was 60–75) | Avoids rejecting valid early-stage breakouts |
| Extension filter loosened to 10% above 10 EMA (was 8%) | Post-earnings gaps legitimately exceed 8% |
| Volume threshold lowered to 1.5× (was 2×) | Web data often reflects partial-day volume |
| PARTIAL DATA category added | Prevents unnecessary rejection of valid setups |

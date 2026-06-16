---
name: data-engineering-v2
description: >
  Structured data log template, feed quality rating A/B/C by latency, data lineage tracker, lag detection alerts, and cross-source conflict resolution for Sentinel Global Fund. Use when logging any data point, rating source quality, detecting stale data, and resolving conflicts between sources. Owned by Nina Okonkwo, Data and Source Engineer.
---

# DATA ENGINEERING v2.0
**Owner: Nina Okonkwo — Data & Source Engineer**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
ข้อมูลที่ผิดแต่ดูน่าเชื่อถือ อันตรายกว่า
ข้อมูลที่ผิดแต่รู้ว่าผิด
ทุก data point ที่เข้าสู่ทีมต้องผ่าน Nina ก่อน
ไม่มีตัวเลขใดที่ "ชัดเจนในตัวเอง" โดยไม่มีแหล่งและวันที่

---

## MODULE 1 — STRUCTURED LOG TEMPLATE

### 1A. Standard Data Entry
```
DATA LOG ENTRY — Nina Okonkwo | [timestamp]
══════════════════════════════════════════════════
Indicator:      [VIX / Price / RSI / CPI / etc]
Value:          [X.XX]
Source:         [TradingView / Bloomberg / SEC / etc]
Source URL:     [URL ถ้ามี]
Reported Date:  [YYYY-MM-DD HH:MM TZ]
Received Date:  [YYYY-MM-DD HH:MM TZ]
Lag:            [X hours/days]
Feed Quality:   [A/B/C] (see Module 2)
Flag:           [V/E/U]
Confidence:     [High/Medium/Low]
Conflict:       [none / conflicts with SOURCE at VALUE]
══════════════════════════════════════════════════
```

### 1B. Batch Log (Pre-Meeting Pack)
```
PRE-MEETING DATA PACK — [วันที่] | Nina Okonkwo

MARKET DATA (feed สด priority):
  VIX:     [X] [A/B/C] [timestamp] [V/E]
  SPY:     $[X] [A/B/C] [timestamp] [V/E]
  QQQ:     $[X] [A/B/C] [timestamp] [V/E]
  10Y:     [X]% [A/B/C] [timestamp] [V/E]

MACRO DATA:
  CPI:     [X]% [A/B/C] [date] [V/E]
  FOMC:    Next [วันที่] | Rate [X]% [V]
  NFP:     Last [X]K on [วันที่] [V]

PORTFOLIO DATA:
  NAV:     $[X] [user-provided E]
  Holdings: [last update วันที่]

WATCHLIST (from Maya/Aisha):
  [Ticker]: $[X] [A/B/C] [timestamp] [V/E/U]

SESSION DQS: [V X% / E X% / U X%]
CONFLICTS: [รายการถ้ามี]
STALE DATA: [รายการที่ lag > 24 ชม.]
```

---

## MODULE 2 — FEED QUALITY RATING

### 2A. Quality Tiers
```
TIER A — Real-time / Same session
  Latency: < 15 minutes
  Sources: User TradingView screenshot (timestamped),
           Broker feed, Bloomberg terminal
  Use for: All trading decisions, regime scoring
  Flag: [V] always

TIER B — Same day
  Latency: 15 min - 8 hours
  Sources: Yahoo Finance close, ETF provider pages,
           Official earnings releases, SEC filings
  Use for: Fundamental analysis, daily reporting
  Flag: [V] if < 4 hours, [E] if 4-8 hours

TIER C — Delayed / Cached
  Latency: > 8 hours or unknown
  Sources: Search engine snippets, news aggregators,
           Social media, forums
  Use for: Background research only, NOT for scoring
  Flag: [E] with lag disclosure, or [U] if unreliable

STALE FLAG:
  Any data > 24 hours old = append [STALE + age]
  Example: "VIX 22.22 [STALE - 2 days]"
```

### 2B. Source Ranking by Asset Class
```
EQUITIES / ETF PRICES:
  A: User TradingView > Broker app > Bloomberg
  B: Yahoo Finance close > ETF provider NAV page
  C: Search snippets (Tier C — do not use for trading)

MACRO INDICATORS (CPI/NFP/FOMC):
  A: Fed.gov / BLS.gov official release [V]
  B: Bloomberg / Reuters with date
  C: News headline without date [E/U]

TECHNICAL INDICATORS (RSI/MACD/ADX):
  A: User chart screenshot (timestamped) [V]
  B: Stockanalysis / Financhill with date [E]
  C: Search snippet [U — do not use for Phase 3]

EARNINGS DATA:
  A: SEC EDGAR filing [V]
  B: Company IR press release [V]
  C: Analyst estimate aggregators [E]
```

---

## MODULE 3 — DATA LINEAGE TRACKER

### 3A. Lineage Chain
```
DATA LINEAGE — [Indicator] | [วันที่]
────────────────────────────────────────────────
Origin:    [original source + date]
Received:  Nina Okonkwo [timestamp]
Verified:  [ใช่/ไม่ + method]
Processed: [any transformation — e.g., % change calc]
Distributed to:
  → Daniel Cho (regime scoring) [timestamp]
  → Maya Chen (Phase 3 scoring) [timestamp]
  → Aisha Fontaine (catalyst) [timestamp]
  → Kai Tanaka (risk) [timestamp]
Used in:   [decision/report name]
────────────────────────────────────────────────
CHAIN INTEGRITY: ✅ no transformation errors
```

---

## MODULE 4 — LAG DETECTION & ALERTS

### 4A. Lag Thresholds
```
INDICATOR           MAX ACCEPTABLE LAG    ACTION IF EXCEEDED
────────────────────────────────────────────────────────────
VIX / Index prices  15 minutes            Request user feed
Individual stocks   15 minutes            Request user feed
10Y Treasury yield  1 hour                Use Fed/CNBC direct
CPI / NFP           Same day of release   BLS.gov only
FOMC decision       Same day              Fed.gov only
Earnings data       24 hours post-release  SEC EDGAR
Analyst ratings     48 hours              Mark [E]
Short interest      2 weeks (report lag)   Always [E] + date
────────────────────────────────────────────────────────────
```

### 4B. Stale Data Alert Format
```
⚠️ STALE DATA ALERT — Nina Okonkwo
Indicator: [X]
Last value: [X] from [source]
Last updated: [วันที่] ([X] hours/days ago)
Exceeds threshold: [X] hours
Action required: [Request user screenshot / check source]
Impact: [Which team members are using this data]
```

---

## MODULE 5 — CONFLICT RESOLUTION PROTOCOL

### 5A. When Two Sources Disagree
```
CONFLICT DETECTED — [Indicator] | [วันที่]
────────────────────────────────────────────
Source A: [X.XX] from [source] at [timestamp]
Source B: [X.XX] from [source] at [timestamp]
Difference: [X]% or [X] pts

RESOLUTION HIERARCHY:
  1. User's own feed (TradingView/broker) WINS always
  2. More recent data wins (if same quality tier)
  3. Higher tier wins (A > B > C)
  4. Official source wins (SEC > news > aggregator)
  5. If unresolved: flag [CONFLICT] + use conservative value
     + notify team before using in scoring

RESOLUTION: Using [Source] value [X.XX]
Reason: [tier/recency/official]
Other value: [X.XX] — discarded
```

### 5B. Real Case Example (GEV RSI)
```
CONFLICT EXAMPLE — GEV RSI (12 มิ.ย. 2026)
Source A: Financhill RSI = 66.48
Source B: Tradingkey RSI = 29.35
Difference: 37.13 pts (MAJOR conflict)
Resolution: BOTH flagged [U] — DATA UNAVAILABLE
  Neither source has clear timestamp
  Score = 0 pts per Governance Rule #5
  Action: request user TradingView chart screenshot
```

---

## HARD RULES
```
❌ ห้ามส่ง data ไม่มี timestamp ให้ทีม
❌ Search snippet > 24 ชม. = [STALE] label บังคับ
❌ Conflict ที่แก้ไม่ได้ = [U] flag + score 0
❌ ห้ามใช้ Tier C sources สำหรับ Phase 3 scoring
❌ User feed สด override ทุก source เสมอ
❌ Pre-meeting pack ต้องส่งก่อนประชุม ≥ 15 นาที
```

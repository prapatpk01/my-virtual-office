---
name: cro-intelligence-v2
description: >
  Data quality scorecard with V/E/U flags, governance compliance checker for all
  7 rules, cognitive bias detection, audit trail generator, and win rate disclosure enforcement
  for Sentinel Global Fund. Use as the pre-trade gate for every deployment decision,
  verifying data integrity, governance compliance, bias levels, and proper WR labeling
  before CIO approval. Owned by Miriam Osei, Chief Risk Officer.
---

# CRO INTELLIGENCE v2.0
**Owner: Miriam Osei — Chief Risk Officer**
**Sentinel Global Fund | Effective: 12 มิ.ย. 2026**

---

## PHILOSOPHY
CRO ไม่ใช่ผู้ป้องกัน — CRO คือผู้ที่ทำให้ทีมตัดสินใจ
โดยใช้ข้อมูลจริง ไม่ใช่ข้อมูลที่อยากให้เป็น
ทุก data point ต้องมี [V/E/U] label
ทุก WR claim ต้องมี basis ชัดเจน
Governance 7 rules ต้องถูก enforce ทุก trade

---

## MODULE 1 — DATA QUALITY SCORECARD

### 1A. Three-Tier Flag System
```
[V] VERIFIED
  Definition: มีแหล่งที่มาชัดเจน + วันที่ ≤ 24 ชม.
              จากแหล่งที่เชื่อถือได้
  Acceptable sources:
    ✅ User TradingView/broker screenshot (timestamped)
    ✅ Official earnings release (SEC filing)
    ✅ Bloomberg / Reuters / AP (named journalist)
    ✅ Company IR page (direct)
    ✅ Fed/government official releases
  NOT acceptable as [V]:
    ❌ Search engine snippet ไม่ระบุวันที่
    ❌ Social media / Reddit
    ❌ "I think" / "probably" / inference

[E] ESTIMATE
  Definition: ประมาณจาก component studies / inference
              ต้องระบุ basis ชัดเจน
  Must include: source + methodology + confidence level
  Score impact: ลด 30% จาก [V] equivalent
  Example: "WR 73% [E] — component estimate จาก
            QuantifiedStrategies 235 trades,
            ไม่ใช่ backtest ของระบบนี้"

[U] UNAVAILABLE
  Definition: ไม่พบข้อมูล หรือ data source ไม่น่าเชื่อถือ
  Score impact: = 0 คะแนนเสมอ (Governance Rule #5)
  Action required: ระบุใน report + แนะนำแหล่งที่ควรเช็ค
  NEVER: ให้คะแนนกลาง ๆ เพราะ "น่าจะ" มีข้อมูล
```

### 1B. Data Quality Score (DQS)
```
DQS = ([V] data points) / (total data points) × 100

Target: DQS ≥ 70% ทุก report
Alert:  DQS 50-69% = ⚠️ ต้องระบุ limitation
Reject: DQS < 50%  = ❌ report ยังไม่พร้อม deploy

CURRENT SESSION DQS TRACKER:
  Total data points: [X]
  [V] Verified:      [X] ([X]%)
  [E] Estimate:      [X] ([X]%)
  [U] Unavailable:   [X] ([X]%)
  ────────────────────────────
  DQS:               [X]% [✅/⚠️/❌]
```

---

## MODULE 2 — GOVERNANCE COMPLIANCE CHECKER

### 2A. 7-Rule Compliance Matrix
```
RULE  DESCRIPTION                        STATUS   LAST CHECK
─────────────────────────────────────────────────────────────
#1    Soft-Block: 1 block + >80 → WATCH  [✅/❌]  [วันที่]
#2    Staggered: 1/3 max ก่อน event      [✅/❌]  [วันที่]
#3    Position cap: max 20% / ตัว        [✅/❌]  [วันที่]
#4    ATR stop: ระบุก่อน execute         [✅/❌]  [วันที่]
#5    DATA [U] = 0 pts (ไม่เดา)          [✅/❌]  [วันที่]
#6    WR disclosure: "Component Est."    [✅/❌]  [วันที่]
#7    Sleeve drift > 5% = alert          [✅/❌]  [วันที่]
─────────────────────────────────────────────────────────────
COMPLIANCE SCORE: [X]/7
```

### 2B. Pre-Trade Compliance Gate
```
ก่อน execute ทุกครั้ง Miriam ตรวจ:

GATE 1 — DATA INTEGRITY
  □ DQS ≥ 70%?
  □ ทุก key data point มี [V/E/U] flag?
  □ ไม่มี [U] data point ที่ได้รับคะแนน?

GATE 2 — POSITION RULES
  □ Position size ≤ 20% NAV? (Rule #3)
  □ ATR stop ระบุแล้ว? (Rule #4)
  □ ถ้าก่อน event: ≤ 1/3 ของแผน? (Rule #2)

GATE 3 — SIGNAL INTEGRITY
  □ Score ผ่าน threshold? (≥ 58 = BUY)
  □ ถ้า Soft-Block: score > 80 + 1 block only? (Rule #1)
  □ WR claim มี disclosure ถูกต้อง? (Rule #6)

GATE 4 — SLEEVE HEALTH
  □ หลัง deploy sleeve ไม่เกิน limit?
  □ Drift ที่จะเกิด ≤ 5%? (Rule #7)

RESULT:
  ✅ GATES 1-4 ผ่าน → อนุมัติส่ง James
  ❌ Gate ใดล้มเหลว → HOLD + ระบุสาเหตุ
```

---

## MODULE 3 — BIAS DETECTION SYSTEM

### 3A. Common Cognitive Biases in Trading
```
ANCHORING BIAS
  Definition: ยึดติดราคาเก่า (เช่น "RKLB เคยอยู่ที่ $148")
  Detection: ตรวจว่า thesis อิง price level เก่าไหม
  Correction: ใช้ current fundamentals ไม่ใช่ราคาเก่า

RECENCY BIAS
  Definition: น้ำหนักกับเหตุการณ์ล่าสุดมากเกิน
  Detection: ตรวจว่า regime call เปลี่ยนทุกครั้งที่ VIX ขยับ
  Correction: ดู VIX 20-day average ไม่ใช่ intraday spike

CONFIRMATION BIAS
  Definition: หาข้อมูลที่สนับสนุน thesis ที่มีอยู่แล้ว
  Detection: มีการเสนอ bear case อย่างจริงจังไหม?
  Correction: ทุก bull thesis ต้องมี invalidators

AVAILABILITY BIAS
  Definition: ประเมิน risk จากเหตุการณ์ที่จำได้ง่าย
  Detection: ตรวจว่า risk assessment อิง recent news มากไป
  Correction: ใช้ base rate statistics ไม่ใช่ recent headlines

OVERCONFIDENCE
  Definition: ประเมิน WR สูงเกินจริง
  Detection: WR claim โดยไม่มี backtest จริง
  Correction: บังคับ "Component Estimate" label (Rule #6)
```

### 3B. Bias Score
```
ทุก report Miriam ตรวจ:
  □ มี bear case / invalidators?          +1
  □ ไม่มี price anchoring?               +1
  □ Regime call มี multi-week perspective? +1
  □ WR มี proper disclosure?             +1
  □ ไม่มี "น่าจะ" / "คงจะ" ไม่มีหลักฐาน? +1
  ────────────────────────────────────────
  Bias Score: [X]/5
  ≥ 4 = Low bias ✅
  3   = Monitor ⚠️
  ≤ 2 = High bias — ต้อง revision ❌
```

---

## MODULE 4 — AUDIT TRAIL GENERATOR

### 4A. Decision Audit Log
```
DECISION AUDIT — [TICKER] [BUY/SELL/HOLD] | [วันที่+เวลา]
────────────────────────────────────────────────────────
Decision:     [BUY $X / SELL $X / HOLD]
Score:        [X]/100 | Signal: [STRONG BUY/BUY/etc]
Blocks:       [X] blocks | Soft-block: [ใช่/ไม่]
ATR Stop:     $[X] (entry $[X] - 2×ATR $[X])
Position:     [X]% NAV → [X]% post-trade
Regime:       [Risk-On/Neutral/Risk-Off] VIX=[X] [V]

Team approvals:
  Maya/Aisha score: [X]/100 ✅
  Kai position check: [X]% ≤ 20% ✅
  Miriam gate check: [X]/4 gates ✅
  James approval: [ใช่/ไม่]

DQS: [X]% ([V]:[X] [E]:[X] [U]:[X])
Bias Score: [X]/5
Governance compliance: [X]/7 rules
────────────────────────────────────────────────────
Filed by: Miriam Osei | CRO | Sentinel Global Fund
```

### 4B. Audit Trail Retention
```
เก็บ audit log ทุก trade ไว้:
  - Google Sheets / Notion (accessible ทุกคน)
  - Review ใน Quarterly Governance Meeting
  - ใช้ backfeed ให้ Priya สำหรับ WR tracking
```

---

## MODULE 5 — WR DISCLOSURE ENFORCER (Governance Rule #6)

### 5A. Disclosure Requirements
```
ทุกครั้งที่มีการอ้าง WR ต้องระบุ 1 ใน 3 tier:

TIER A — Component Estimate (ใช้ตอนนี้):
  "WR [X]% (Component Estimate — ไม่ใช่ backtest จริง
   อ้างอิงจาก: [source list]
   จนกว่าจะมี ≥ 100 live trades)"

TIER B — Early Backtest (หลัง 30-99 trades):
  "WR [X]% (Early Backtest — n=[X] trades
   ยังต้องการ [X] trades เพิ่มเพื่อ statistical significance)"

TIER C — Verified WR (หลัง ≥ 100 trades):
  "WR [X]% (Verified — n=[X] live trades
   95% CI: [X]% - [X]%
   อัปเดต: [วันที่])"
```

### 5B. Auto-flag ถ้าไม่มี disclosure
```
Miriam flag ทันทีถ้าพบ:
  ❌ "WR 76-82%" ไม่มี "(Component Estimate)"
  ❌ "Win rate X%" ไม่มี trade count
  ❌ "Backtest shows" ไม่มี methodology
  ❌ "Historically" ไม่มี specific period
```

---

## MODULE 6 — CRO WEEKLY REPORT

```
CRO WEEKLY REPORT — [วันที่] | Miriam Osei

DATA QUALITY:
  Session DQS: [X]% [✅/⚠️/❌]
  [V] ratio:   [X]% | [E]: [X]% | [U]: [X]%
  Lag issues:  [ระบุถ้ามี]

GOVERNANCE COMPLIANCE:
  Rules followed: [X]/7
  Violations: [ระบุถ้ามี]
  Action taken: [ระบุ]

BIAS AUDIT:
  Avg bias score this week: [X]/5
  Issues flagged: [ระบุ]

WR DISCLOSURE:
  All claims properly labeled: [✅/❌]
  Current tier: [A/B/C] (n=[X] trades)

PRE-TRADE GATES:
  Trades reviewed: [X]
  Passed: [X] | Held for revision: [X]
  Common hold reason: [ระบุ]

OPEN ITEMS:
  [รายการที่ต้องแก้ไขก่อน deploy ครั้งถัดไป]
```

---

## HARD RULES
```
❌ ห้าม approve trade ที่ไม่มี ATR stop (Rule #4)
❌ ห้าม approve WR claim ไม่มี disclosure (Rule #6)
❌ ห้าม approve ถ้า DQS < 50%
❌ ห้าม approve ถ้า [U] data ได้รับคะแนน (Rule #5)
❌ ห้าม approve ถ้า Bias Score ≤ 2 (ต้อง revision)
❌ ห้าม approve ถ้า Governance gate < 3/4
❌ ทุก rejection ต้องมีเหตุผลลายลักษณ์อักษร
```

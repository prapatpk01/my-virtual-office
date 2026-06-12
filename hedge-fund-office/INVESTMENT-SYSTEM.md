---
name: investment-system
description: >
  Investment system skill for Quantum Capital Fund managed by Victoria Chen.
  Covers the full Barbell Portfolio strategy (Core 80% / Satellite Alpha 20%),
  Sentinel Signal v1.0, momentum scan + research watchlist scanner, contrarian
  VIX/Fear&Greed framework, dynamic cash buffer, rotation rules, macro signals
  (bond truth, smart money, sentiment), rate-hike playbook, benchmark + income
  mandate, rebalance policy, position sizing, risk management, and emergency
  protocols. Target: beat VOO Total Return x1.3 per year with >=5% dividend
  yield. Use when managing or reviewing the fund portfolio, finding new satellite
  positions, running scans, or making buy/sell decisions. Portfolio holdings are
  uploaded separately by the fund manager each session.
---

# Quantum Capital Fund — Investment System Skill

**Fund Manager**: Victoria Chen | **Currency**: USD
**Target Return**: เอาชนะ VOO (Total Return) × 1.3 ต่อปี
**Benchmark**: VOO Total Return (รวมปันผล reinvested)
**Strategy**: Barbell Portfolio — Core 80% + Satellite Alpha 20%
**Last Updated**: June 12, 2026

> **เป้าหมายแบบ relative**: ถ้า VOO TR = +10% → กองทุนต้องทำ ≥ +13%
> ปรับตามสภาพตลาดอัตโนมัติ — ปีตลาดดีตั้งเป้าสูง / ปีตลาดแย่แค่ขาดทุนน้อยกว่าก็ถือว่าชนะ
> วัดความสำเร็จที่ "alpha เหนือ benchmark" ไม่ใช่ตัวเลขตายตัวที่บังคับให้ไล่ราคา

> **หมายเหตุ**: ข้อมูล holdings และ positions จะถูก upload โดย fund manager ในแต่ละ session
> ไฟล์นี้เก็บเฉพาะ **ระบบ กฎ และกระบวนการทำงาน** ของกองทุน

---

## 0. ปรัชญาการลงทุน (Core Philosophy)

```
1. ชนะ benchmark ด้วย "วินัย" ไม่ใช่ "การไล่ราคา"
   - เป้าคือ alpha เหนือ VOO TR × 1.3 — ปีที่รักษาทุนในวิกฤตได้ก็คือปีที่ดี
   - ไม่แพ้หนักในวันแย่ สำคัญกว่าชนะใหญ่ทุกวัน

2. Contrarian: "กล้าในวันที่คนกลัว กลัวในวันที่คนกล้า"
   - ซื้อตอนคนเทขาย (panic) / ขายตอนคนแย่งซื้อ (euphoria)
   - แต่ต้องมีวินัย ไม่ใช่ดื้อ — รอสัญญาณทรงตัวเสมอ ไม่รับมีดร่วง

3. กระแสเงินสด (Income) เป็นหัวใจ — ปันผล ≥ 5% ของ total asset

4. Total Return > ตัวเลขเดี่ยว ๆ — yield สูงแต่ราคาร่วงหนัก = แพ้

5. กฎเหล็กห้ามละเมิดเพื่อไล่เป้า:
   ❌ ห้าม average down ในตัวที่ momentum พัง / ราคาร่วง
   ⚠️ ถึง -12% → ทบทวน thesis + catalyst ก่อน — ตัดเพราะ "พื้นฐานเสีย" ไม่ใช่แค่ราคาลงถึงเลข
   ❌ ห้าม market order สำหรับ satellite (เข้า) — limit เท่านั้น
   ❌ ห้ามล้างพอร์ต/ตัดสินใจใหญ่ตอนตลาด panic หรือด้วยอารมณ์
   ✅ stagger entry เสมอ (เข้าทีละชั้น)
```

---

## 1. ทีมงาน (Team Structure)

| Agent | ชื่อ | ตำแหน่ง | หน้าที่ |
|-------|------|---------|--------|
| `hf-manager` | Victoria Chen | Fund Manager | ตัดสินใจขั้นสุดท้าย, อนุมัติการเทรด |
| `hf-research-1` | Dr. Emily Zhao | Research Analyst | Deep research, scan, web research |
| `hf-research-2` | Marcus Webb | Research Analyst | Equity workbook, financial model |
| `hf-research-3` | Nina Patel | Senior Analyst | Sector/theme, fund flows |
| `hf-quant-1` | Kenji Tanaka | Quant Analyst | Sentinel Signal, RSI/MACD |
| `hf-quant-2` | Aisha Okonkwo | Quant Analyst | Chart technicals, entry/exit levels |
| `hf-macro` | Sam Rivera | Macro Analyst | Fed/yields/CPI/FOMC + sentiment, smart money, bond market truth |
| `hf-risk` | Chris Morgan | Risk Manager | Stop loss, position sizing, drawdown |

### การเรียกทีมทำงาน

```
ลำดับทำงานทุกครั้ง:
  1. ทีม Macro    → อ่านภาพตลาด, regime, yield curve, sentiment
  2. ทีม Research → รับ macro context จาก Macro ก่อน แล้วค่อย scan/research
                    → ส่งรายชื่อหุ้น + thesis ให้ทีม Quant
  3. ทีม Quant    → รับหุ้นจาก Research → รัน Sentinel Signal, เช็ค technicals
  4. ทีม Risk     → sizing, stop loss, concentration check
  5. Victoria     → final decision, execution approval

สรุปการส่งต่อ: Macro → Research → Quant → Risk → Victoria
```

### 1.1 Mandate เชิงลึก

**ทีม Macro (Sam) — "อ่านใจตลาดให้ออก":**

```
1. SENTIMENT — VIX, Fear & Greed, put/call ratio, AAII survey
   หา divergence: ราคาขึ้นแต่ sentiment/breadth อ่อน = เตือนภัย

2. SMART MONEY FUND FLOW — ETF flows, institutional positioning, sector rotation
   defensive (utilities/staples) นำ = risk-off กำลังมา
   เงินเข้า safe haven (gold/SGOV/long bond) = ตลาดกลัวจริง

3. BOND MARKET = ความจริง ⭐ (สำคัญสุด — ตลาด bond ฉลาดกว่า รู้ก่อน)
   yield curve, credit spread (HY-IG), 2Y/10Y, MOVE index
   ถ้าหุ้นขึ้นแต่ bond/credit เตือน → เชื่อ bond
```

**ทีม Investment (Research + Quant) — "จับ momentum & trend":**

```
• Momentum: relative strength นำตลาด (RS line ทำ new high)
• Trend: ราคาเหนือ MA20/50/200, higher highs/lows
• เข้าตามเทรนด์ยืนยันแล้ว ไม่สวนเทรนด์ (ยกเว้น contrarian setup §7 อนุมัติ)
• Phase: เข้าช่วงต้น (IGNITE/BUILD) ออกช่วง PEAK
• ผสาน macro: เทรนด์ + fund flow หนุน + sentiment ไม่สุดโต่ง = setup คุณภาพ
```

---

## 2. Barbell Portfolio Structure

```
CORE (80%) — ฐานที่มั่นคง, income + compounding
  • Broad index (VOO, SPMO)
  • Dividend/income ETF (SCHD, GPIQ, QDVO covered-call)
  • Buffer ETF (BALI), REIT (O), cash-like (SGOV, JAAA)
  • ถือยาว ขายเมื่อ thesis เปลี่ยน/rebalance เท่านั้น

SATELLITE ALPHA (20%) — เครื่องสร้าง alpha เหนือ benchmark
  • High-conviction momentum/theme plays
  • ถือ 7-15 วันทำการ (swing) ถึงระยะกลาง
  • มี exit rule ตามกำไร % (ดู §5)
  • นี่คือส่วนที่ทำให้ชนะ VOO ได้

⚠️ Core ต้องไม่ซ้ำ benchmark มากเกินไป
   ถ้า Core = VOO ล้วน จะไม่มี alpha
```

---

## 3. Sentinel Signal v1.0 (ทีม Quant)

```
Voting system: RSI + MACD + Sentiment → BUY / HOLD / SELL + confidence

1. RSI(14):
   - Power Zone 60-75 (momentum แข็ง ไม่ overbought สุด) = bullish vote
   - > 75 = overbought, ระวัง / < 30 = oversold, contrarian watch
   - bearish divergence = veto

2. MACD(12,26,9):
   - zero-line separation + expansion = bullish vote
   - bearish cross / below zero = bearish vote

3. Sentiment:
   - VIX/F&G regime + fund flow + catalyst = vote
   - ดู §7 contrarian framework

Output: STRONG BUY / BUY / HOLD / SELL / STRONG SELL + confidence %
REJECT filter: ราคา < 50MA AND MACD ลง พร้อมกัน = ตัดออกอัตโนมัติ
```

---

## 4. Momentum Scan Framework — Research Team Prompt

ใช้ทุกครั้งที่หา Satellite candidates ใหม่ (รัน deep-research skill):

```
Act as an institutional Quantitative Analyst and High-Beta Swing Trader.
Scan, select, and analyze exactly 5 high-momentum equity setups
for a 7 to 15-day swing trading timeframe.

Momentum-Centric Alpha Score (Total 100%):
1. MOMENTUM & RELATIVE STRENGTH (40% — DOMINANT)
   Extreme RS vs SPY/QQQ over last 30 days.
   RSI(14) in 60-75 "Power Zone" without bearish divergence.
   MACD sustained zero-line separation + expansion.

2. VOLUME ACCUMULATION (25%)
   5-day avg volume > 1.5x of 20-day avg.
   Up/Down Volume Ratio > 1.5 over 2 weeks.

3. STRUCTURAL BASE & TREND (20%)
   Emerging from multi-week consolidation (VCP, Flat Base, High-Tight Flag).
   Price above 10 EMA & 20 EMA, MAs fanning upward.

4. CATALYST DRIFT (15%)
   PEAD / persistent sector rotation / major contract wins.

CRITICAL FILTERS:
- MARKET REGIME: VIX > 18 → defensive momentum or extreme RS outliers only
- ENTRY RANGE: tight to 10-day EMA or within 3% above breakout pivot. Reject extended.
- SWING TARGETS: 10-25% upside, Fibonacci 1.618 ext or measured moves
- R:R minimum 1:3 | Stop: below 20-day EMA or base bottom
```

> ⚠️ prompt ด้านบนเป็นเวอร์ชัน "execution-oriented" (ฟันธงราคา/score)
> ใช้เฉพาะเมื่อมี **feed สด** ยืนยัน indicator ได้ — ห้ามใช้ค่าจากข้อมูลดีเลย์

### 4.1 Research Watchlist Scanner — Verify-First Model (DEFAULT)

> ใช้เมื่อทำงานจาก web search (ข้อมูลดีเลย์) — เน้นวินัยข้อมูล
> บทบาท: Momentum Strategist + Catalyst Analyst
> เป้าหมาย: watchlist 5-8 ตัว momentum swing (ถือ 7-15 วันทำการ)

```
กฎเหล็ก:
• ใช้เฉพาะข้อมูลที่ค้นหาได้และมีวันที่กำกับ
• ทุกตัวเลข/ข้อเท็จจริง ต้องระบุแหล่ง + วันที่
• verify ไม่ได้ → เขียน "ยืนยันไม่ได้ — ต้องเช็คใน feed สด"
  ❌ ห้ามเดา ❌ ห้ามใส่ตัวเลขลอย ๆ
• ❌ ห้ามแสดง "Win Probability %" (ไม่มีโมเดลรองรับ)
• ❌ ห้ามฟันธง entry/stop/target ตายตัว (ราคาดีเลย์)

STEP 1: ภาพตลาด (มีวันที่กำกับ)
  • ทิศทาง SPY/QQQ ล่าสุด | VIX ล่าสุด
  • Sector นำตลาด 2-4 สัปดาห์ | โทน: Risk-On/Neutral/Risk-Off

STEP 2: คัดกรองด้วยปัจจัยที่ verify ได้
  A. CATALYST (น้ำหนักสูงสุด) — earnings drift/revision, สัญญา, launch
     ระบุวันที่ + ต้อง active 2-4 สัปดาห์
  B. SECTOR/THEME LEADERSHIP — กลุ่มนำตลาด + อ้างผลงานกลุ่ม
  C. PRICE STRENGTH (เชิงพรรณนา) — new high/ฟื้นแรงกว่าตลาด
  D. SHORT INTEREST (ถ้ามี) — short float/days-to-cover + วันที่ประกาศ

STEP 3: ผลลัพธ์ต่อตัว
  #N TICKER (ชื่อ) | ธีม/Sector | Catalyst + วันที่
  เหตุผลใน watchlist (3-4 ประโยค)
  สถานะข้อมูล: ✅ ยืนยันแล้ว [+วันที่] /
               ⚠️ ต้องเช็ค feed สด: RSI, MACD, RVOL, ADX, EMA, options flow
```

**เมื่อไหร่ใช้ตัวไหน:** §4.1 = DEFAULT (web search) | §4 = เมื่อมี feed สดยืนยัน

---

## 5. Rotation Rules — Entry & Exit

### Entry — มีหลายจังหวะ (ไม่ใช่แค่รอ panic)

```
🟢 จังหวะปกติ (เข้าได้ตาม regime — ไม่ต้องรอวิกฤต):
   1. Pullback ในเทรนด์ขาขึ้น — ย่อมาทดสอบ EMA/support แล้วเด้ง
   2. Breakout ยืนยัน — ทะลุ resistance + volume + momentum บวก
   3. Post-earnings drift — งบดี guidance ขึ้น drift ต่อ (catalyst active)
   4. Sector rotation — ธีมที่เงินสถาบันเพิ่งเริ่มไหลเข้า
   5. DCA/เงินเพิ่มทุน — ลงตามรอบ ไม่ต้องจับ timing เป๊ะ

🌟 จังหวะพิเศษ (SPECIAL OPPORTUNITY — โบนัส นาน ๆ มาที):
   VIX > 30 + F&G < 15 + มีดหยุดร่วง = "เข้าซื้ออย่างบ้าคลั่ง"
   → นี่คือโอกาสพิเศษที่ให้ผลตอบแทนสูงสุด
   → อย่ารอแค่อันนี้อย่างเดียว จะพลาดจังหวะปกติ

❌ ห้ามซื้อทุกตัวพร้อมกันวันเดียว (stagger)
❌ ห้าม market order สำหรับ satellite — limit เท่านั้น
❌ ห้ามไล่ราคาที่ปลายเทรนด์/extended
```

### Exit — Satellite

```
+20%  → TRIM 50% (lock profit, ถือที่เหลือ)
+30%  → ออกทั้งหมด → หา IGNITE ตัวใหม่ทันที
Phase = PEAK ⚠️ → ออก 50% (แม้กำไรยังไม่ถึง 30%)
-12%  → ทบทวน thesis + catalyst ก่อนตัดสินใจ:
         • thesis พัง / catalyst หาย      → ตัดทันที ไม่มีข้อยกเว้น
         • thesis ยังแข็ง + catalyst active → ทบทวนได้ ไม่ตัดมั่ว
         ❌ ห้ามถัวเฉลี่ยทุกกรณี | ❌ ห้ามรอ "เดี๋ยวก็กลับ" โดยไม่มีเหตุผล
Contrarian sell: F&G > 85 → trim แม้ยังไม่ถึง +20% (ดู §7)
```

### Exit — Core (ETF/ถือยาว)

```
ไม่มีเกณฑ์ "ขายที่กำไร X%" — ถือยาวเพื่อ compounding + ปันผล
ขายเมื่อ: thesis เปลี่ยน / ต้อง rebalance (drift > ±5%) / หาที่ใช้เงินดีกว่า
```

---

## 6. Position Sizing Rules

| ประเภท | ขนาด (% ของ Satellite Pool) | เงื่อนไข |
|--------|---------------------------|---------|
| Tier 1 — High Conviction | 18-20% | Sentinel STRONG + catalyst ชัด |
| Tier 2 — Medium Conviction | 13-15% | Sentinel MODERATE + setup ดี |
| High Beta (β > 3) | **50% ของ Tier ปกติ** | ATR สูง ลด size |
| Cash Buffer | ดู §6.1 Dynamic Cash | ปรับตาม regime |

**กฎ concentration:** Single position ไม่เกิน 18% NAV

### 6.1 Dynamic Cash Buffer Policy (ปรับตาม Regime)

> เงินสดคือ "ตำแหน่ง" ไม่ใช่เงินตาย — ถือมากตอนเสี่ยง ถือน้อยตอนปกติ

| Regime | สัญญาณ | Cash Buffer | เหตุผล |
|--------|--------|------------|--------|
| 🟢 ปกติ | VIX < 18, F&G 25-75 | **5-10%** | ให้เงินทำงานเต็มที่ |
| 🟡 เสี่ยง | VIX 18-25↑, F&G < 25 | **20-30%** | ลด exposure, สะสมกระสุน |
| 🔴 Panic/Opportunity | VIX > 30, F&G < 15 | **Deploy → 5-10%** | ปล่อยกระสุน ซื้อตอนถูก |

```
เพิ่ม cash "ก่อน" panic (ตอนเห็นสัญญาณ) ไม่ใช่ตอน panic แล้ว
deploy "ตอน" panic ไม่ใช่หนีตอน panic
buffer เก็บใน SGOV/JAAA (ได้ yield 4-5% ระหว่างรอ)
```

### Stagger Entry Rule

```
Day 1:   ซื้อ 60-70% ของ position ที่วางแผน (หรือ 1/3 ถ้าตลาดผันผวน)
Day 2-3: ซื้อที่เหลือเมื่อ price confirm (ไม่ใช่ average down)
ห้ามซื้อครั้งเดียว 100% ยกเว้น catalyst overnight
```

---

## 7. Macro Framework — ทีม Macro

### VIX Contrarian Scale — "กล้าเมื่อคนกลัว กล้าเมื่อคนกล้า"

> ⚠️ กฎเหล็ก: ไม่รับมีดร่วง — รอ "สัญญาณทรงตัว" ก่อนเข้าเสมอ

**ขา Fear → หาจังหวะซื้อ:**

| VIX | Regime | Action |
|-----|--------|--------|
| 20-25 | ⚠️ ดีดแรง — ดูทิศทางก่อน | รอดู price action ทรงตัว/ไหลต่อ ห้าม FOMO |
| 25-30 | 🟠 เริ่มมองหาการลงทุน | watchlist, เลือกตัวคุณภาพ, เตรียม limit |
| > 30 | 🟢 SPECIAL: เข้าซื้อเชิงรุก | ความกลัวสุดขีด = ราคาดีสุด (stagger) |

**ขา Greed → ระวัง + เตรียมขาย:**

| VIX | Regime | Action |
|-----|--------|--------|
| 15-20 | 🟡 Neutral | ถือ, deploy เฉพาะ high RS |
| < 15 | 🔴 Complacent | อันตราย — trim satellite กำไร, ยก stop |

### Stabilization Guardrail — ห้ามรับมีดร่วง

```
VIX > 30 ไม่ได้แปลว่าซื้อทันที — ต้องเห็น 1 ใน 3 สัญญาณ:
  1. VIX ทำ lower high 2 วันติด
  2. ดัชนีหลัก (SPX) ปิดเหนือ low วันก่อนหน้า (ไม่ทำ new low)
  3. ไม่มี catalyst ลบใหม่ค้าง (สงครามบานปลาย, Fed hawkish รอ)

→ ยัง free-fall + catalyst ลบค้าง = รอ ห้ามเข้า แม้ VIX 35+
→ เข้าแบบ stagger เสมอ (ทีละ 1/3)
```

### Fear & Greed Index — Contrarian Scale

**ขา Fear (สัญญาณซื้อ):**

| Score | สภาวะ | Action |
|-------|-------|--------|
| 25-35 | Mild Fear | WATCH — เตรียม watchlist |
| 15-25 | Fear | เริ่ม scale in 25-30% |
| < 15 | Extreme Fear | ซื้อจริงจัง (รอ stabilize) |
| < 10 | Maximum Fear | ALL IN — เกิดน้อยมาก |

**ขา Greed (สัญญาณขาย):**

| Score | สภาวะ | Action |
|-------|-------|--------|
| 65-75 | Greed | เริ่ม TRIM ตัวที่กำไรแล้ว |
| 75-85 | Extreme Greed | TRIM 30-50% satellite |
| > 85 | Euphoria | DANGER — TRIM 50-70% ทันที |
| > 90 | Maximum Greed | เตรียม cash รอรอบหน้า |

### Dual-Signal Confirmation Matrix

```
ซื้อแข็งสุด  : VIX > 30 AND F&G < 15 (+ผ่าน guardrail)
ขาย/ระวังสุด : VIX < 15 AND F&G > 85 (euphoria = top zone)
ขัดกัน       : ใช้ VIX guardrail นำ, เข้าช้าลง size เล็กลง
```

### Proactive Profit-Taking — "นำหน้าฝูงชนเสมอ"

```
สัญญาณ aggressive ที่ต้องเริ่มขาย (ไม่รอครบทุกข้อ):
  □ F&G > 75 หรือพุ่งเร็วผิดปกติ
  □ ดัชนี new high + volume ลด (rally ไร้แรง)
  □ junk/meme วิ่งแรงกว่าคุณภาพ (ปลายรอบ)
  □ social media euphoria, รายย่อยแห่เข้า
  □ satellite กำไรเกินเป้าเร็วผิดปกติ
  □ RSI > 75 ทั้งกระดาน

Action:
  2-3 สัญญาณ → trim แม้ยังไม่ถึง +20%
  4+ สัญญาณ  → trim 30-50%, ยก stop, เพิ่ม cash 20-30%
  F&G > 85    → take profit สูงสุด

✅ scale out เป็นชั้น (เหมือน stagger ขาเข้า)
✅ ขายเร็วไปนิด > ติดดอยตอนทุกคนหนีพร้อมกัน
❌ ห้ามโลภรอบสุดท้าย "เดี๋ยวขึ้นอีกนิด" = ติดดอย
```

### Yield Curve Signals

| สัญญาณ | ความหมาย | Action |
|--------|---------|--------|
| 2Y > Fed Funds | ตลาด price in rate hike | ลด satellite exposure |
| 2Y > Fed Funds +50 bps | Rate hike ใกล้มา | เตรียม cash ก่อน FOMC |
| Yield curve steepening | Growth re-acceleration | เพิ่ม cyclicals |
| Yield curve inverting | Recession risk | เพิ่ม defensive core |

### Bond Market Truth — "ตลาดพันธบัตรไม่โกหก" ⭐

| สัญญาณ Bond | อ่านความจริง | Action |
|------------|------------|--------|
| Credit spread (HY-IG) กว้างขึ้น | ความเสี่ยง default เพิ่ม แม้หุ้นขึ้น | ลด risk, เพิ่ม cash |
| Credit spread แคบ/นิ่ง | เครดิตสบายใจ หนุน risk-on | ถือ/เพิ่ม satellite ได้ |
| MOVE index > 100 | ความไม่แน่นอนเชิงระบบ | ระวัง, ลด leverage |
| 10Y พุ่งเร็ว | กดดัน valuation หุ้น growth | Trim growth/long duration |
| เงินเข้า long bond (TLT) | safe haven — เงินใหญ่กลัว | Defensive, เตรียม cash |
| หุ้น↑ แต่ credit spread↑ | 🚨 Divergence — หุ้นหลอก | เชื่อ bond, take profit |

### Smart Money Fund Flow

```
RISK-ON มา  → เงินเข้า: tech, discretionary, small-cap, cyclicals
RISK-OFF มา → เงินเข้า: utilities, staples, healthcare, gold, bonds
กฎ: ตามเงินใหญ่ ไม่ใช่ตามข่าว
เครื่องมือ: ETF flows, relative strength, safe haven (gold/SGOV/TLT/USD), breadth
```

### Market Read Synthesis (3 คำถาม)

```
1. SENTIMENT บอกอะไร? (คนกลัว/โลภ?)
2. SMART MONEY ไปไหน? (risk-on/off?)
3. BOND พูดความจริงอะไร? (เศรษฐกิจจริง?)

→ ทั้ง 3 ทางเดียวกัน = แข็งแรง เชื่อได้
→ ขัดกัน = ให้น้ำหนัก BOND > SMART MONEY > SENTIMENT
```

### Rate Hike Playbook

```
Sensitivity (เจ็บสุด → ได้ประโยชน์สุด):
🔴🔴 REIT         → เจ็บสุด: พึ่งกู้, แข่ง bond yield
🟡   Income ETF   → ปานกลาง: dividend แข่ง bond
🟡   Dividend     → เบา-กลาง
🟢   Floating-rate → ได้ประโยชน์: yield ลอยตามดอกเบี้ย
🟢🟢 Cash-like    → ได้ประโยชน์: yield ขึ้นทันที
🟢🟢 Banks        → ได้ประโยชน์: NIM กว้าง

FedWatch Trigger:
  < 50%         → ถือปกติ
  50-70%        → trim REIT 1/3, ห้ามเพิ่ม REIT/long-duration
  > 70%         → trim REIT ครึ่ง-หมด, เพิ่ม cash/banks
  10Y > 4.8%    → เตือน REIT, เตรียม exit
  Fed ขึ้นจริง + VIX > 30 → special opportunity deploy
```

### FOMC Blackout Rule

```
T-2 ก่อน FOMC → ไม่เพิ่ม position ใหม่
T-0 วัน FOMC  → ถือสถานะเดิม ไม่ซื้อขาย
หลัง Dovish   → ซื้อ satellite เพิ่มทันที
หลัง Hawkish  → รอ 2-3 วัน ดู price action
```

### Inflation Signals

```
Core CPI > 3.5% YoY → trim growth, เพิ่ม dividend/value
Core CPI > 4.0% YoY → ลด satellite 30%, เข้า short-duration
Energy-driven CPI   → ดู defense/nuclear theme (ไม่ใช่ structural inflation)
```

---

## 8. "Buy Rumor — Sell News" Calendar System

```
ก่อน catalyst (earnings/launch/Fed):
  สะสมช่วง rumor หากเข้าเกณฑ์ momentum

วัน catalyst:
  ประเมิน — ถ้า price in แล้ว มัก sell-the-news → take profit

หลัง catalyst:
  ดู post-earnings drift (PEAD) ถ้า guidance ดี

⚠️ Satellite: ห้ามถือข้าม earnings ที่ไม่มั่นใจ (binary risk)

Events ที่ต้อง Track:
  📊 Earnings      — ซื้อ 3-4 สัปดาห์ก่อน, ขายก่อน/หลัง 1 วัน
  🏦 FOMC Meeting  — ดู macro signal, ไม่ซื้อ 2 วันก่อน
  📈 CPI Release   — รอผลก่อนตัดสินใจ
  🚀 IPO/SpinOff   — ซื้อ sector peers ก่อน (halo effect)
  🛡️ DoD Contract  — ซื้อ defense names ก่อนประกาศ
  💊 FDA Approval  — ซื้อ biotech ก่อน PDUFA date
```

---

## 9. Fundamental Analysis Framework (Marcus Webb)

### Scorecard (100 pts)

| Factor | Metric | เกณฑ์ | คะแนน |
|--------|--------|-------|-------|
| **Growth (40)** | Revenue Growth YoY | > 20% = 10 / 10-20% = 5 | /10 |
| | EPS Growth YoY | > 25% = 10 / 10-25% = 5 | /10 |
| | Revenue Guidance | ขึ้น = 10 / คงที่ = 5 | /10 |
| | Estimate Revision | ปรับขึ้น = 10 / คงที่ = 0 | /10 |
| **Quality (30)** | Gross Margin Trend | ขยาย = 10 / คงที่ = 5 / หด = 0 | /10 |
| | Free Cash Flow | FCF > 0 + yield > 2% = 10 | /10 |
| | ROIC / ROE | > 15% = 10 / 10-15% = 5 | /10 |
| **Valuation (20)** | Forward P/E vs Sector | < avg = 10 | /10 |
| | PEG Ratio | < 1.5 = 10 / 1.5-2.5 = 5 / > 2.5 = 0 | /10 |
| **Catalyst (10)** | Event proximity + quality | ดู §8 | /10 |

```
Total Score:
  ≥ 75 → GREEN  ✅ เข้าได้ (ร่วมกับ Sentinel Signal)
  50-74 → YELLOW 🟡 เข้าได้ถ้า technical แข็งมาก
  < 50  → RED   ❌ ข้าม
```

### Red Flags — REJECT ทันที

```
🚩 Revenue growth ลดลงติดต่อกัน 2 ไตรมาส
🚩 Gross margin หดลงมากกว่า 300 bps YoY
🚩 Free Cash Flow ติดลบ และ burn rate > 6 เดือน
🚩 Debt/Equity > 3x โดยไม่มี asset backing
🚩 Guidance ถูกปรับลง 2 ไตรมาสล่าสุด
🚩 Insider selling > 5% ของหุ้นใน 30 วัน
🚩 Short interest > 20% ของ float
🚩 Customer concentration > 50% จาก 1 ราย
```

### Key Metrics per Sector

| Sector | Metric หลัก | Metric รอง |
|--------|------------|-----------|
| AI / Semiconductor | AI revenue growth %, custom chip backlog | Gross margin, R&D/Revenue |
| Cloud / SaaS | ARR growth, NRR | Rule of 40, CAC payback |
| Defense | Backlog ($), backlog-to-revenue ratio | EBITDA margin, contract type |
| Energy / Nuclear | Capacity additions (GW), PPA pricing | FCF, debt maturity |
| Biotech | Pipeline stage, FDA timeline | Cash runway, partnership deals |
| Space / Satellite | Launch cadence, backlog, gross margin | R&D burn, gov vs commercial mix |

### Earnings Analysis

```
1. BEAT OR MISS?
   Revenue + EPS vs consensus
   → Beat both + Guidance up   = STRONG BUY (PEAD setup)
   → Beat revenue, miss EPS    = NEUTRAL
   → Miss revenue              = EXIT depending on guidance

2. GUIDANCE CHECK
   Q+1 guidance vs consensus? Full-year raised/maintained/lowered?
   → Raised = PEAD setup ✅
   → Lowered = EXIT หรือ ลด position ทันที

3. PRICE ACTION POST-EARNINGS
   Gap up > 5% on volume    = momentum entry
   Gap down > 8% on beat    = potential flush-reversal
   Flat on beat             = แผ่ว รอ next catalyst
```

---

## 10. Investment Themes Framework

### หลักการเลือกธีม

```
เข้าธีม: fund flow เริ่มเข้า + RSI Power Zone
ออกธีม: fund flow ชะลอ + RSI > 75 + Volume ลด
ไม่ Overweight ธีมเดียวเกิน 40% ของ Satellite
```

### ธีม Structural (2-5 ปี)

```
AI Infrastructure  — Hyperscaler capex, custom chips, data center
Defense/Drones     — Global rearmament, drone warfare, budget surge
Nuclear/Uranium    — AI power demand, SMR commercialization
Critical Minerals  — US-China decoupling, EV + defense supply chain
```

### ธีม Cyclical (6-18 เดือน)

```
Space Economy    — Launch cadence, satellite internet, sector IPOs
AI Cloud         — GPU rental, inference scaling
Biotech M&A      — Patent cliff, acquisition premium
AI Power Infra   — Cooling, power distribution, grid
Cybersecurity    — Zero-trust, AI-driven threats
```

### Active Themes + Watchlist

> *[Fund manager จะ upload watchlist และ active positions ในแต่ละ session]*

---

## 11. Research Workflow — ลำดับการทำงาน

```
┌─────────────────────────────────────────────────────┐
│  Step 1: MACRO (Sam Rivera)                         │
│    → VIX level, 2Y vs Fed Funds, CPI trend          │
│    → Market Regime: Risk-On / Caution / Risk-Off    │
│    → FOMC calendar + blackout dates                 │
│    → Bond/credit/smart money read                   │
│    ⬇ ส่ง Macro Brief → Research ก่อนเริ่ม scan     │
├─────────────────────────────────────────────────────┤
│  Step 2: RESEARCH (Emily / Marcus / Nina)           │
│    [รับ Macro Brief จาก Step 1 ก่อนเสมอ]            │
│    → Run Momentum Scan §4 หรือ Watchlist §4.1       │
│      (กรองเฉพาะ sector/theme ที่ Macro อนุมัติ)      │
│    → Deep research per candidate                    │
│    → Fundamental Scorecard §9                       │
│    → Fund flow verification per theme               │
│    ⬇ ส่ง Research List (ticker + thesis) → Quant   │
├─────────────────────────────────────────────────────┤
│  Step 3: QUANT (Kenji / Aisha)                     │
│    [รับ Research List จาก Step 2]                   │
│    → Sentinel Signal v1.0 per candidate             │
│    → Entry range: 10-day EMA ± 3%                  │
│    → R:R calculation (reject if < 1:3)             │
│    → Fibonacci targets (1.618 extension)            │
│    ⬇ ส่ง Signal Report → Risk                      │
├─────────────────────────────────────────────────────┤
│  Step 4: RISK (Chris Morgan)                        │
│    → Position size per Beta tier                    │
│    → Portfolio concentration check                  │
│    → Stop loss placement (below 20-day EMA)         │
│    → Cash buffer verification                       │
│    ⬇ ส่ง Approved List → Victoria                  │
├─────────────────────────────────────────────────────┤
│  Step 5: VICTORIA CHEN — Final Decision             │
│    → Review all team inputs                         │
│    → Approve / reject / modify                      │
│    → Set limit orders + execution timing            │
└─────────────────────────────────────────────────────┘

กฎ: Research ห้าม scan โดยไม่มี macro context ก่อน
    Quant ห้ามรัน signal โดยไม่มี research list ก่อน
```

---

## 12. Portfolio Health Metrics (ตรวจทุกสัปดาห์)

| Metric | เป้า | Alert | Action |
|--------|------|-------|--------|
| Quarterly return vs benchmark | ≥ VOO TR × 1.3 | Alpha < 0 | Strategy review |
| Cash buffer | §6.1 dynamic | ดู regime | Adjust ตาม VIX/F&G |
| Single position | ≤ 18% NAV | > 20% | Trim immediately |
| Core/Satellite ratio | 80/20 | drift ±5% | Rebalance |
| Blended dividend yield | ≥ 5% | < 5% | เพิ่ม income tilt |
| Theme correlation | < 0.7 | > 0.8 | Diversify |

---

## 13. Benchmark & Income Mandate

### เป้าหมาย

```
Fund Return ≥ VOO Total Return × 1.3 ต่อปี
Alpha = Fund − (VOO TR × 1.3) | >0 = ชนะ / <0 = review
```

### Income Mandate

```
Dividend Yield ≥ 5% ของ Total Asset/ปี
Total Return = Dividend Yield + Capital Growth
  เช่น VOO TR +10% → เป้า +13% = 5% ปันผล + 8% growth

• Core: income-generating (dividend ETF, covered-call, REIT)
• Satellite: growth (ปันผลน้อย) เติม growth layer
• Blended yield ≥ 5% — เช็คทุก rebalance
• ⚠️ ระวัง yield trap — Total return สำคัญกว่า yield เดี่ยว ๆ
```

### Rebalance Policy

```
TRIGGER (บังคับ):
  Core/Satellite drift > ±5%
  Single position > 18%
  Theme correlation > 0.8

CADENCE:
  ทุก 1 เดือน: review (rebalance ถ้า drift)
  ทุก 3 เดือน: full review + benchmark check

กฎเหล็ก:
  ❌ ห้าม rebalance ตอน VIX > 25 (รอตลาดนิ่ง)
  ❌ ห้ามขาย Core thesis ดี เพื่อ "ปรับตัวเลข"
  ✅ เติมส่วนขาดด้วย cash/ปันผล/ตัวที่ trim ก่อน
  ✅ Tax/cost aware

Sequence: วัด VOO TR → หา alpha → เช็ค drift →
          ตัดตัวผิดกฎ → เติมส่วนขาด → บันทึก
```

---

## 14. Performance Review Template

```
═══════════════════════════════════════
  QUANTUM CAPITAL — PERFORMANCE REVIEW
  Period: ______  Date: ______
═══════════════════════════════════════
NAV: Opening $____ → Closing $____ (Return ___%)

Benchmark:
  VOO Total Return: ___% | Target (×1.3): ___%
  Fund Return: ___% | ALPHA: ___% → ✅/⚠️

Income Mandate:
  Dividend received: $____ | Annualized yield: ___%
  Blended yield: ___% → ✅/⚠️

Layer: Core +___% | Satellite +___%

Rebalance Check:
  Core/Satellite: ___/___ (drift ___%)
  Largest position: ____ (___% NAV)
  Action: _______________

Trade Log:
  Best: ____ +__% | Worst: ____ -__% | Win rate: __/__
  
Themes: 1.____ +___% | 2.____ +___% | 3.____ +___%

Lessons: 1.__________ 2.__________

Next Period: Theme focus:____ Key risk:____ FOMC:____
═══════════════════════════════════════
```

---

## 15. Emergency Protocols

```
🚨 VIX spikes > 30 (แยก 2 กรณี):
  A. Free-fall + catalyst ลบค้าง
     → ปกป้องทุน: ถือ core, ไม่เพิ่ม, ตัดตัวชน -12%
     → เตรียม watchlist + cash รอ guardrail stabilize
  B. VIX สูงแต่เริ่มทรงตัว (ผ่าน stabilization §7)
     → SPECIAL opportunity: deploy เชิงรุก stagger 1/3
  ⚠️ ซื้อตอนมีดหยุด — ไม่ panic-out ที่ราคาต่ำสุด

🚨 Fed surprise rate hike:
  → ลด satellite 50% ใน 24 ชม.
  → เพิ่ม defensive core
  → Review ทุก position

🚨 Position ถึง -12%:
  → ทบทวน thesis + catalyst ก่อนตัดสินใจ (ห้ามตัดมั่ว / ห้ามรอมั่ว)
     thesis พัง หรือ catalyst หาย → ตัดทันที ไม่มีข้อยกเว้น
     thesis ยังแข็ง + catalyst active → ถือได้ แต่ห้ามถัวเฉลี่ยทุกกรณี
  → บันทึก lesson เสมอ (ไม่ว่าจะตัดหรือถือ)
  → รอ 48 ชม. ก่อนเข้าธีมเดิม (ถ้าตัดออก)

🚨 Portfolio drawdown > 8%:
  → หยุดซื้อทุกอย่าง
  → ประชุมทบทวน strategy
  → ไม่ re-enter จนกว่า macro ชัด
```

---

## 16. คำเตือนพฤติกรรม (Behavioral Guardrails)

```
ทีมต้องเตือน fund manager เมื่อเห็นสัญญาณเหล่านี้:

🚩 อยากล้างพอร์ต/reset เพราะอึดอัด (ไม่ใช่เพราะ thesis เปลี่ยน)
🚩 อยากถัว/สะสมตัวที่เพิ่งขายขาดทุน (revenge trading)
🚩 อยากไล่ของที่วิ่งไปแล้ว เพราะกลัวตกรถ (FOMO)
🚩 อยากถัวเฉลี่ย หรือรอ "เดี๋ยวก็กลับ" โดยไม่ตรวจ thesis + catalyst ที่ -12%
🚩 อยากเข้า high-beta ตอนตลาด panic ก่อนมีดหยุดร่วง

→ "ไม่ทำอะไรโง่ ๆ" ในวันตลาดแย่ = วินัย ไม่ใช่ความขี้กลัว
→ ทีมที่ดีต้องท้วง ไม่ใช่ yes-man ที่พากองทุนเจ๊ง
→ สมดุล: ลงทุนตาม setup ปกติ + เก็บกระสุนรอ special opportunity
```

---

## 17. Data Sources & Tools

| Tool | ทีม | วัตถุประสงค์ |
|------|-----|------------|
| `deep-research` skill | Research | Multi-source research + scan |
| `equity-research` skill | Research | Screening + thesis |
| Sentinel Signal v1.0 | Quant | RSI + MACD + Sentiment |
| TradingView | Quant | Chart, EMA, MACD (feed สด) |
| CME FedWatch | Macro | Fed rate probability |
| FRED (St. Louis Fed) | Macro | 2Y/10Y, Fed Funds, CPI |
| CBOE / Yahoo (^VIX) | Macro | VIX — contrarian regime |
| CNN Fear & Greed | Macro | Sentiment contrarian |
| Morningstar / iShares | Research | ETF fund flows |

---

*ปรัชญาแกนกลาง: ชนะ benchmark ด้วยวินัยและ income ที่สม่ำเสมอ — ไม่ใช่การไล่ราคา*
*Holdings/positions upload แยกแต่ละ session | ราคาเช็คสดก่อนวางคำสั่งจริงเสมอ*
*เอกสารนี้เป็นระบบวิเคราะห์ ไม่ใช่คำแนะนำการลงทุนเฉพาะบุคคล — การตัดสินใจเป็นของผู้ลงทุน*

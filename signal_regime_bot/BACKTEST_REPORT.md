# DUALCORE V1.9 — รายงาน Backtest และการปรับจูนขั้นสุดท้าย

## ข้อมูลที่ใช้

- ใช้ข้อมูลจาก `Crypto(1).zip` ที่ผู้ใช้อัปโหลด.
- ทดสอบ Production Logic บนแท่ง **5M Native** ของ BTC และ SOL ช่วง **1 กุมภาพันธ์–1 มิถุนายน 2026** แบบต่อเนื่อง.
- ETH, HYPE และ XRP ไม่มีไฟล์ 5M Native จึงไม่ใช้เพื่อเลือกค่า Default.
- มกราคมไม่สามารถเป็น Holdout ที่เชื่อถือได้ เพราะไม่มีข้อมูลก่อนเดือนมกราคมเพียงพอสำหรับ Warm-up 4H 200+ แท่ง.

## สมมติฐานการจำลอง

- ใช้เฉพาะแท่งปิดแล้ว; Fill ที่ Open ของแท่ง 5M ถัดไป.
- Slippage ฝั่งเสียเปรียบ 0.05%.
- Fee 0.10% ต่อ Fill รวม Partial TP.
- หาก TP และ SL อยู่ในแท่งเดียวกัน จะถือว่า SL เกิดก่อน.
- จำลอง TP1, Runner, Fee-aware BE, Structure Stop และ Same-direction Re-entry Lock.
- ผลลัพธ์เป็น Net R หลังต้นทุนจำลอง.

## ผลเต็มช่วง

| Symbol | Trades | W/L/BE | WR ไม่รวม BE | PF | Net R | Avg R | Max DD | Trades/เดือน |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC | 5 | 4/0/1 | 100.0% | ∞ | +5.24 | +1.05 | 0.00R | 2.11 |
| SOL | 10 | 6/2/2 | 75.0% | 3.11 | +5.39 | +0.54 | -1.31R | 2.98 |
| **รวม** | **15** | **10/2/3** | **83.3%** | **5.16** | **+10.63** | **+0.71** | **-1.31R** | — |

## ก่อนและหลัง Final Trigger Tuning

| Version | Trades | W/L/BE | PF | Net R | Avg R | Max DD |
|---|---:|---:|---:|---:|---:|---:|
| ก่อน Final Trigger Gate | 19 | 10/8/1 | 1.26 | +2.74 | +0.14 | -4.15R |
| **Final Tuned** | **15** | **10/2/3** | **5.16** | **+10.63** | **+0.71** | **-1.31R** |

## การปรับจูนที่นำมาใช้

1. `EMA_RECLAIM_CONFIRM` ใน EARLY regime ต้องมี 5M directional edge ≥ 85.
2. `EMA_RECLAIM_CONFIRM` ใน STRONG regime ต้องมี directional edge ≥ 55.
3. EARLY breakout-retest ต้องมี Compression หรือ Volume Ratio ≥ 1.10.
4. Direct breakout ต้องมี Compression/Volume, EMA extension ≤ 0.85 ATR, level extension ≤ 0.45 ATR และ fee drag ≤ 0.28R.
5. Retest จำกัด EMA extension ≤ 1.35 ATR; ไม่อนุญาตให้ Retest ข้าม Anti-chase Gate.
6. Pullback extension: Precision assets 0.65 ATR; High-beta assets 0.70 ATR.

## ข้อสรุปเชิงความถี่

ผลเต็มช่วงให้ความถี่ประมาณ **2.1 เทรด/เดือนสำหรับ BTC** และ **3.0 เทรด/เดือนสำหรับ SOL** ซึ่งยังต่ำกว่าเป้าหมาย 10–15 เทรด/เดือน/Symbol และไม่พิสูจน์เป้าหมาย 3–4 เทรด/วันรวม 7 Symbol. การผ่อน Filter เพื่อไล่จำนวนเทรดทำให้ PF และ Drawdown แย่ลงในรอบก่อนหน้า จึงไม่ได้ตั้งค่าแบบนั้นเป็น Default.

## ข้อจำกัดและความเสี่ยง

- 15 เทรดยังเป็น Sample เล็ก และข้อมูลที่ใช้ปรับจูนไม่ใช่ Out-of-sample อิสระ.
- ยังไม่ได้ยืนยัน Exact 5M บน ETH/XRP/HYPE/XAU/XAG.
- ผล Backtest ไม่รวมเหตุการณ์ API, Gap, Funding, Orderbook impact และ Exchange outage ทั้งหมด.
- Risk 5% ต่อไม้สูงมาก แม้ Logic ดีขึ้น; แพ้ 2 ไม้ติดอาจลด Equity ใกล้ 10%.
- ควร Demo/Paper อย่างน้อย 50–100 เทรดก่อนใช้เงินจริงเต็ม Risk.

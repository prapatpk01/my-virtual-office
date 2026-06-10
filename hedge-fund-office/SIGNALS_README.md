# Quant Team — Technical Analysis Scripts

**Decision (10 Jun 2026): ใช้ Sentinel Signal v1 เป็น script มาตรฐานของทีม (Option A)**

## Official: Sentinel_Signal_v1.pine

All-in-one indicator (Pine v6, overlay) — แทนการใช้ KMTF + MCDX คู่กัน

### Setup บน TradingView
1. เปิด Pine Editor → วางโค้ดทั้งไฟล์ → Add to chart
2. ค่า default ใช้ได้เลย — Auto MTF จะปรับ timeframe ladder ตาม chart TF อัตโนมัติ
   - Chart ≥ 1W → M / W / D
   - Chart ≥ 1D → W / D / 4H
   - Chart ≤ 1H → 1H / 30m / 15m (+5m micro)
3. ตั้ง Alert: เลือก "Any alert() function call" — มี 14 alerts ในตัว (Long/Short, early_bull/early_bear, divergence ฯลฯ)

### อ่านสัญญาณ
- **LONG/SHORT label** = สัญญาณ entry ผ่าน gate ครบ (momentum + MTF + HMA + confidence)
- **Momentum phase**: IGNITE 🔥 → BUILD ⚡ → ACCEL 🚀 → PEAK ⚠️ → FADING ↘ → REVERSAL ↩
  - เข้าได้ช่วง IGNITE–ACCEL / ห้ามเข้าใหม่ช่วง PEAK–FADING
- **Forecast Cone + p_up%**: ความน่าจะเป็นทิศทาง (>65% = bias ชัด)
- **Fib targets**: TP ปรับตาม Elliott Wave (W3 → 1.618 ext, W5 → ระวัง reversal)
- **DWCS** (0–100): >60 = สะสม (bull), <40 = แจกของ (bear)

## Reference (เก็บไว้เปรียบเทียบ ไม่ใช่ standard)
- `KMTF_Technical_v6_4.pine` — overlay เดิม (มี bug ที่แก้แล้วใน Sentinel)
- `MCDX_Plus_v2_0.pine` — oscillator เดิม (DWCS v5.1, weight ไม่เสถียร)

"""
Compare: v5 Tight (optimized) vs v5 Wide (expanded RSI + lower vol threshold)
vs WaveTrend & UT Bot — same data, same execution.

Wide = expand RSI zones ±6 pts + vol_mult 1.4→1.2 to increase trade frequency.
"""

import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from trading.connectors.base import OHLCV
from trading.strategies.base import BaseStrategy
from trading.strategies.wt_adx_strategy import WTADXStrategy
from trading.strategies.ut_bot_strategy import UTBotStrategy

SYMBOL    = "BTC/USDT"
MONTHS    = 5
CAPITAL   = 260.0
TRADE_USD = 100.0
FEE_RATE  = 0.001
WARMUP    = 50
COOLDOWN_BARS = 3

# ── v5 Tight (current best) ──────────────────────────────────────────────────
TIGHT = {
    "SR_Tight":  dict(sl_atr=2.5, tp_atr=1.5, max_hold_days=3,
                      rsi_lo=40.0, rsi_hi=56.0, vol_mult=1.4,
                      regime="soft_bull", entry_type="momentum"),
    "CPK_Tight": dict(sl_atr=2.5, tp_atr=1.5, max_hold_days=4,
                      rsi_lo=46.0, rsi_hi=58.0, vol_mult=1.4,
                      regime="soft_bull", entry_type="momentum"),
    "Hyb_Tight": dict(sl_atr=2.5, tp_atr=2.0, max_hold_days=2,
                      rsi_lo=44.0, rsi_hi=58.0, vol_mult=1.4,
                      regime="soft_bull", entry_type="momentum"),
}

# ── v5 Wide (higher frequency) ───────────────────────────────────────────────
WIDE = {
    "SR_Wide":   dict(sl_atr=2.5, tp_atr=1.5, max_hold_days=3,
                      rsi_lo=34.0, rsi_hi=62.0, vol_mult=1.2,
                      regime="soft_bull", entry_type="momentum"),
    "CPK_Wide":  dict(sl_atr=2.5, tp_atr=1.5, max_hold_days=4,
                      rsi_lo=40.0, rsi_hi=64.0, vol_mult=1.2,
                      regime="soft_bull", entry_type="momentum"),
    "Hyb_Wide":  dict(sl_atr=2.5, tp_atr=2.0, max_hold_days=2,
                      rsi_lo=38.0, rsi_hi=64.0, vol_mult=1.2,
                      regime="soft_bull", entry_type="momentum"),
}

LEGACY_CONFIGS = {
    "WaveTrend": dict(sl_atr=2.5, tp_atr=3.0, max_hold_days=5),
    "UT_Bot":    dict(sl_atr=2.5, tp_atr=3.0, max_hold_days=5),
}


# ─── Synthetic data ─────────────────────────────────────────────────────────────

def fetch_candles():
    since_ms = int(time.time() * 1000) - MONTHS * 31 * 24 * 3600 * 1000
    sym_b = SYMBOL.replace("/", "")
    import urllib.request, json as _j
    for source, url_fn in [
        ("Binance", lambda f: f"https://api.binance.com/api/v3/klines?symbol={sym_b}&interval=1h&startTime={f}&limit=1000"),
        ("OKX",    lambda f: f"https://www.okx.com/api/v5/market/history-candles?instId={SYMBOL.replace('//','-')}&bar=1H&limit=300&after={f}"),
    ]:
        try:
            req = urllib.request.Request(url_fn(since_ms), headers={"User-Agent": "bt/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _j.loads(r.read())
            raw = data if isinstance(data, list) else data.get("data", [])
            if raw and len(raw) > 50:
                raw.sort(key=lambda r: int(r[0]))
                candles = [OHLCV(int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])) for r in raw]
                print(f"  {source}: {len(candles)} candles"); return candles
        except Exception: pass

    print("  ⚠ Synthetic data (seed=42)")
    rng = np.random.default_rng(42)
    n   = MONTHS * 31 * 24 + 50
    regimes = [
        (0.00,0.15,+0.00012,0.0020,0.93,8),(0.15,0.30,+0.00022,0.0030,0.91,7),
        (0.30,0.40,+0.00030,0.0040,0.89,6),(0.40,0.48,+0.00003,0.0050,0.86,6),
        (0.48,0.58,-0.00015,0.0060,0.88,6),(0.58,0.66,+0.00002,0.0035,0.90,7),
        (0.66,0.76,+0.00008,0.0022,0.93,8),(0.76,0.90,+0.00025,0.0032,0.90,7),
        (0.90,1.00,+0.00010,0.0045,0.87,6),
    ]
    closes=[32000.0]; vol=0.0022
    for i in range(1,n):
        frac=i/n; dr,bv,persist,df=0.0,0.003,0.90,7
        for rs,re,d,b,p,tdf in regimes:
            if rs<=frac<re: dr,bv,persist,df=d,b,p,tdf; break
        shock=float(rng.standard_t(df))*vol
        vol=float(np.sqrt(persist*vol**2+(1-persist)*bv**2+0.02*shock**2))
        vol=max(0.001,min(vol,0.015))
        closes.append(max(closes[-1]*(1+dr+shock),1000.0))
    candles=[]; ts0=since_ms
    for i,c in enumerate(closes):
        o=closes[i-1] if i>0 else c; an=abs(float(rng.normal(0,c*0.0018))); vm=float(rng.lognormal(6.5,0.6))
        candles.append(OHLCV(ts0+i*3600000,round(o,2),round(max(o,c)+an,2),round(min(o,c)-an,2),round(c,2),round(vm,0)))
    peak=max(closes); trough=min(closes[int(n*0.38):int(n*0.72)])
    print(f"  {len(candles)} bars  ${closes[0]:,.0f}→${closes[-1]:,.0f}  peak=${peak:,.0f}  dd={(peak-trough)/peak*100:.0f}%")
    return candles


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _nan(*vs): return any(math.isnan(float(v)) for v in vs)

def candle_bullish(closes,highs,lows,i):
    r=float(highs[i])-float(lows[i])
    return r>1e-6 and (float(closes[i])-float(lows[i]))/r>=0.60

def macd_cross_up(hist,i,lb=3):
    for j in range(i,max(i-lb,0),-1):
        if j<1 or _nan(hist[j],hist[j-1]): continue
        if float(hist[j-1])<0 and float(hist[j])>0: return True
    return False

def check_entry(i,closes,highs,lows,vols,rsi,hist,vol_ma,e80,e200,p):
    if i<8 or _nan(e80[i],e200[i]): return False
    if float(e80[i])<float(e200[i])*0.98: return False
    if not candle_bullish(closes,highs,lows,i): return False
    rv=float(rsi[i]) if not _nan(rsi[i]) else 50.0
    cv=float(vols[i]); mv=float(vol_ma[i]) if not _nan(vol_ma[i]) else 1.0
    return macd_cross_up(hist,i) and p["rsi_lo"]<=rv<=p["rsi_hi"] and cv>=mv*p["vol_mult"]

def run_swing(closes,highs,lows,vols,rsi,hist,vol_ma,e80,e200,atr,candles,p):
    mh=int(p["max_hold_days"]*24); trades=[]; cap=CAPITAL
    in_pos=False; ep=sl=tp=0.0; eb=cd=0; n=len(closes)
    for i in range(WARMUP,n-1):
        cc,ch,cl=float(closes[i]),float(highs[i]),float(lows[i])
        if in_pos:
            bh=i-eb; ex=et=None
            if cl<=sl:   ex,et=sl,"sl"
            elif ch>=tp: ex,et=tp,"tp"
            elif bh>=mh: ex,et=cc,"to"
            if ex:
                qty=TRADE_USD/ep; pnl=(ex-ep)*qty-ex*qty*FEE_RATE; cap+=pnl
                trades.append({"bars":bh,"pnl":round(pnl,4),"type":et,"cap":round(cap,2)})
                in_pos=False
                if et=="sl": cd=COOLDOWN_BARS
        if cd>0: cd-=1; continue
        if not in_pos and check_entry(i,closes,highs,lows,vols,rsi,hist,vol_ma,e80,e200,p):
            av=float(atr[i]) if not _nan(atr[i]) else cc*0.015
            ep=cc; sl=ep-p["sl_atr"]*av; tp=ep+p["tp_atr"]*av
            cap-=(TRADE_USD/ep)*ep*FEE_RATE; eb=i; in_pos=True
    if in_pos:
        c=float(closes[-1]); qty=TRADE_USD/ep; pnl=(c-ep)*qty-c*qty*FEE_RATE; cap+=pnl
        trades.append({"bars":n-1-eb,"pnl":round(pnl,4),"type":"end","cap":round(cap,2)})
    return trades

def run_legacy(buy_sig,closes,highs,lows,atr,cfg):
    mh=int(cfg["max_hold_days"]*24); trades=[]; cap=CAPITAL
    in_pos=False; ep=sl=tp=0.0; eb=0; n=len(closes)
    for i in range(WARMUP,n-1):
        cc,ch,cl=float(closes[i]),float(highs[i]),float(lows[i])
        if in_pos:
            bh=i-eb; ex=et=None
            if cl<=sl:   ex,et=sl,"sl"
            elif ch>=tp: ex,et=tp,"tp"
            elif bh>=mh: ex,et=cc,"to"
            if ex:
                qty=TRADE_USD/ep; pnl=(ex-ep)*qty-ex*qty*FEE_RATE; cap+=pnl
                trades.append({"bars":bh,"pnl":round(pnl,4),"type":et,"cap":round(cap,2)})
                in_pos=False
        if not in_pos and i<len(buy_sig) and bool(buy_sig[i]):
            av=float(atr[i]) if not math.isnan(float(atr[i])) else cc*0.015
            ep=cc; sl=ep-cfg["sl_atr"]*av; tp=ep+cfg["tp_atr"]*av
            cap-=(TRADE_USD/ep)*ep*FEE_RATE; eb=i; in_pos=True
    if in_pos:
        c=float(closes[-1]); qty=TRADE_USD/ep; pnl=(c-ep)*qty-c*qty*FEE_RATE; cap+=pnl
        trades.append({"bars":n-1-eb,"pnl":round(pnl,4),"type":"end","cap":round(cap,2)})
    return trades

def stats(trades):
    if not trades:
        return dict(n=0,w=0,l=0,wr=0.0,net=0.0,pf=0.0,pm=0.0,aw=0.0,al=0.0,bars=0.0)
    wins=[t for t in trades if t["pnl"]>0]; losses=[t for t in trades if t["pnl"]<=0]
    net=sum(t["pnl"] for t in trades)
    gw=abs(sum(t["pnl"] for t in wins)); gl=abs(sum(t["pnl"] for t in losses))
    aw=gw/max(len(wins),1); al=gl/max(len(losses),1)
    return dict(n=len(trades),w=len(wins),l=len(losses),
                wr=len(wins)/len(trades)*100,net=round(net,4),
                pf=round(gw/max(gl,1e-9),2),
                pm=round(net/MONTHS,4),
                aw=round(aw,4),al=round(al,4),
                bars=sum(t["bars"] for t in trades)/len(trades))


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    W=100
    print(f"\n{'═'*W}")
    print(f"  Tight vs Wide vs Legacy  |  {SYMBOL} 1H  |  {MONTHS} months")
    print(f"{'═'*W}")

    candles=fetch_candles()
    n=len(candles)
    closes =np.array([c.close  for c in candles])
    highs  =np.array([c.high   for c in candles])
    lows   =np.array([c.low    for c in candles])
    volumes=np.array([c.volume for c in candles])

    print(f"  Computing indicators...")
    rsi14      =BaseStrategy.rsi(closes.tolist(),14)
    _,_,mhist  =BaseStrategy.macd(closes.tolist(),12,26,9)
    vol_ma     =BaseStrategy.sma(volumes.tolist(),20)
    atr14      =BaseStrategy.atr(candles,14)
    ema80      =BaseStrategy.ema(closes.tolist(),80)
    ema200     =BaseStrategy.ema(closes.tolist(),200)

    wt=WTADXStrategy(SYMBOL); _,_,wt_buy,_,wt_atr=wt._build_signals(candles)
    ut=UTBotStrategy(SYMBOL); ut_buy,_,_,ut_atr=ut._build_signals(candles)

    print(f"\n  {'Strategy':<14} {'Trades':>7} {'WR':>7} {'PF':>6} {'Net/mo':>8} {'Net 5mo':>9} {'AvgHold':>8}")
    print(f"  {'─'*14} {'─'*7} {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*8}")

    all_results = {}

    def show(label, trades, note=""):
        s=stats(trades)
        flag=" ✓" if s["wr"]>=62 else "  "
        pm_str=f"${s['pm']:+.2f}/mo"
        print(f"  {label:<14} {s['n']:>7} {s['wr']:>6.1f}%{flag} {s['pf']:>6.2f} "
              f"{pm_str:>8} ${s['net']:>+8.2f} {s['bars']:>6.0f}h")
        all_results[label]=s

    # Legacy
    print(f"\n  ── Legacy (existing) ──────────────────────────────────────────────────────")
    for nm,sig,atr,cfg in [("WaveTrend",wt_buy,wt_atr,LEGACY_CONFIGS["WaveTrend"]),
                             ("UT_Bot",  ut_buy,ut_atr,LEGACY_CONFIGS["UT_Bot"])]:
        show(nm, run_legacy(sig,closes,highs,lows,atr,cfg))

    # Tight
    print(f"\n  ── v5 Tight (RSI tight, vol×1.4) ─────────────────────────────────────────")
    for nm,p in TIGHT.items():
        show(nm, run_swing(closes,highs,lows,volumes,rsi14,mhist,vol_ma,ema80,ema200,atr14,candles,p))

    # Wide
    print(f"\n  ── v5 Wide (RSI wider, vol×1.2) ──────────────────────────────────────────")
    for nm,p in WIDE.items():
        show(nm, run_swing(closes,highs,lows,volumes,rsi14,mhist,vol_ma,ema80,ema200,atr14,candles,p))

    # Summary table
    t_trades = sum(all_results[k]["n"] for k in ["SR_Tight","CPK_Tight","Hyb_Tight"])
    w_trades = sum(all_results[k]["n"] for k in ["SR_Wide","CPK_Wide","Hyb_Wide"])

    print(f"\n{'═'*W}")
    print(f"  SUMMARY — 3 strategies combined per month")
    print(f"{'─'*W}")
    print(f"  v5 Tight combined : {t_trades} trades / 5mo  = ~{t_trades/MONTHS:.1f} trades/mo  "
          f"net/mo = ${sum(all_results[k]['pm'] for k in ['SR_Tight','CPK_Tight','Hyb_Tight']):+.2f}")
    print(f"  v5 Wide  combined : {w_trades} trades / 5mo  = ~{w_trades/MONTHS:.1f} trades/mo  "
          f"net/mo = ${sum(all_results[k]['pm'] for k in ['SR_Wide','CPK_Wide','Hyb_Wide']):+.2f}")

    print(f"\n{'═'*W}")
    print(f"  COPYABLE CONFIG — v5 Tight (best WR)")
    print(f"{'─'*W}")
    print("""  SwingReversal : sl_atr=2.5  tp_atr=1.5  rsi_lo=40  rsi_hi=56  vol_mult=1.4  max_hold_days=3
  CPKRegime     : sl_atr=2.5  tp_atr=1.5  rsi_lo=46  rsi_hi=58  vol_mult=1.4  max_hold_days=4
  HybridSwing   : sl_atr=2.5  tp_atr=2.0  rsi_lo=44  rsi_hi=58  vol_mult=1.4  max_hold_days=2
  Regime (all)  : EMA80 > EMA200 × 0.98  (soft_bull)
  Entry  (all)  : MACD cross(neg→pos, 3-bar)  +  RSI in zone  +  Vol ≥ vol_mult × SMA20
  Candle (all)  : close > 60% of bar range
  Cooldown      : skip 3 bars after SL hit""")

    print(f"\n{'─'*W}")
    print(f"  COPYABLE CONFIG — v5 Wide (more trades)")
    print(f"{'─'*W}")
    print("""  SwingReversal : sl_atr=2.5  tp_atr=1.5  rsi_lo=34  rsi_hi=62  vol_mult=1.2  max_hold_days=3
  CPKRegime     : sl_atr=2.5  tp_atr=1.5  rsi_lo=40  rsi_hi=64  vol_mult=1.2  max_hold_days=4
  HybridSwing   : sl_atr=2.5  tp_atr=2.0  rsi_lo=38  rsi_hi=64  vol_mult=1.2  max_hold_days=2
  Regime (all)  : EMA80 > EMA200 × 0.98  (soft_bull)
  Entry  (all)  : MACD cross(neg→pos, 3-bar)  +  RSI in zone  +  Vol ≥ vol_mult × SMA20
  Candle (all)  : close > 60% of bar range
  Cooldown      : skip 3 bars after SL hit""")
    print(f"{'═'*W}")


if __name__ == "__main__":
    main()

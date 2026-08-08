"""Production patch that activates Trend Confirm V5 without disturbing UTBot routing."""
from __future__ import annotations
import logging, os
import run_bot
import run_enhanced_dual_bot as enhanced
from trading.bot import TradingBot
from trading.telegram_notifier import TelegramNotifier
logger=logging.getLogger("trend_confirm_v5_patch")
_ORIGINAL_FACTORY=enhanced._make_merged_trend_confirm
_ORIGINAL_LOG_SCAN=TradingBot._log_scan
_ORIGINAL_BUILD_CAPTION=TelegramNotifier.build_order_caption

def _env_bool(name:str,default:bool)->bool:
    raw=os.getenv(name); return default if raw is None else str(raw).strip().lower() in {"1","true","yes","on","enabled"}

def _factory(symbols:list,config:dict):
    if not _env_bool("USE_LAYER1_4H",True): return _ORIGINAL_FACTORY(symbols,config)
    from trading.strategies.trend_confirm_v5_strategy import TrendConfirmV5Strategy
    def ef(n,d):
        try:return float(os.getenv(n,str(d)))
        except:return float(d)
    def ei(n,d):
        try:return int(os.getenv(n,str(d)))
        except:return int(d)
    return [TrendConfirmV5Strategy(symbol=s,wt_oversold=-42.0,wt_overbought=45.0,structure_swing_span=ei("STRUCTURE_SWING_SPAN",3),structure_retest_min_bars=ei("STRUCTURE_RETEST_MIN_BARS",1),structure_retest_max_bars=ei("STRUCTURE_RETEST_MAX_BARS",3),structure_bos_buffer_atr=ef("STRUCTURE_BOS_BUFFER_ATR",.05),structure_touch_tolerance_atr=ef("STRUCTURE_TOUCH_TOLERANCE_ATR",.15),structure_invalidation_tolerance_atr=ef("STRUCTURE_INVALIDATION_TOLERANCE_ATR",.25),structure_max_close_distance_atr=ef("STRUCTURE_MAX_CLOSE_DISTANCE_ATR",.50),structure_max_fill_slippage_atr=ef("STRUCTURE_MAX_FILL_SLIPPAGE_ATR",.35)) for s in symbols]

def _log_scan(self,symbol,strategy_name,price,signal):
    if str(strategy_name).startswith("TrendConfirm("):
        meta=getattr(signal,"metadata",None) or {}; macro=meta.get("macro_4h") if isinstance(meta.get("macro_4h"),dict) else {}; ctx=meta.get("context_1h") if isinstance(meta.get("context_1h"),dict) else {}
        if str(meta.get("trend_confirm_version","")).startswith("5") or macro.get("layer_role")=="DIRECTION_ONLY":
            c=ctx.get("components") if isinstance(ctx.get("components"),dict) else {}; sig_type=getattr(getattr(signal,"type",None),"value","hold").upper()
            logger.info("[SCAN] %-28s %-22s px=%-11.4f sig=%-5s | L1 4H=%s score=%s (B=%s/S=%s) | L2 1H=%s Q=%s/100 [ADX %.1f=%s/25 | CHOP %.1f=%s/20 | STRUCT %s=%s/20 | MOM %s=%s/15 | ROOM %sR=%s/20] hard=%s | 15M=%s | %s",strategy_name,symbol,price,sig_type,macro.get("state","?"),macro.get("score","?"),macro.get("bull_score","?"),macro.get("bear_score","?"),ctx.get("label","?"),ctx.get("score","?"),float(ctx.get("adx",0) or 0),c.get("adx","?"),float(ctx.get("chop",0) or 0),c.get("chop","?"),ctx.get("structure","?"),c.get("structure","?"),"ALIGNED" if ctx.get("momentum_aligned") else "OPPOSED",c.get("momentum","?"),ctx.get("room_r","?"),c.get("room","?"),ctx.get("hard_block","?"),meta.get("entry_trigger_owner") or meta.get("entry_trigger") or meta.get("direction_15m","WAIT"),getattr(signal,"reason","")); return
    return _ORIGINAL_LOG_SCAN(self,symbol,strategy_name,price,signal)

def _caption_v5(self,*args,**kwargs):
    text=_ORIGINAL_BUILD_CAPTION(self,*args,**kwargs)
    if "Trend Confirm" not in str(text):return text
    lines=str(text).splitlines();out=[];t1=False;runner=False
    for line in lines:
        if line.startswith("🎯 T1") or (line.startswith("🎯 Target") and not t1):out.append("🎯 T1 : `+1.0R` → take profit `40%` → runner SL `BE`");t1=True;continue
        if line.startswith("🔒 Runner"):out.append("🔒 Runner `60%` : TP2 `+2.0R` | or entry-owner signal exit");runner=True;continue
        out.append(line)
    if t1 and not runner:
        idx=next((i for i,x in enumerate(out) if x.startswith("🎯 T1")),len(out)-1);out.insert(idx+1,"🔒 Runner `60%` : TP2 `+2.0R` | or entry-owner signal exit")
    return "\n".join(out)

def install()->None:
    enhanced._make_merged_trend_confirm=_factory;run_bot._make_strategies=_factory
    if not getattr(TradingBot,"_trend_confirm_v5_log_installed",False):TradingBot._log_scan=_log_scan;TradingBot._trend_confirm_v5_log_installed=True
    if not getattr(TelegramNotifier,"_trend_confirm_v5_caption_installed",False):TelegramNotifier.build_order_caption=_caption_v5;TelegramNotifier._trend_confirm_v5_caption_installed=True
    logger.warning("[TREND CONFIRM V5.1] installed: component log + ADX peak 25-46, decay 46-65, >65=0 + RR 1:2")
install()

from __future__ import annotations
import os, sys, math, shutil, tempfile, argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from config import Config
from pipeline import Pipeline

DATA_ROOT = Path(os.getenv('BACKTEST_DATA_ROOT', './historical_data')).resolve()

TF_MINUTES = {'5m':5,'15m':15,'1h':60,'4h':240}

def _read_month_file(p: Path) -> pd.DataFrame:
    with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
        first=fh.readline().strip()
    cols=['open_time','open','high','low','close','volume','close_time','quote_volume','count','taker_buy_volume','taker_buy_quote_volume','ignore']
    if first.startswith('open_time'):
        df=pd.read_csv(p)
    else:
        df=pd.read_csv(p, header=None, names=cols)
    for col in ['open','high','low','close','volume','close_time']:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    mx=float(df['close_time'].dropna().iloc[0])
    unit='us' if mx>1e14 else 'ms'
    idx=pd.to_datetime(df['close_time'], unit=unit, utc=True).dt.tz_convert(None)
    out=df[['open','high','low','close','volume']].copy()
    out.index=pd.DatetimeIndex(idx)
    return out.dropna()

def _resample_ohlcv(base: pd.DataFrame, rule: str) -> pd.DataFrame:
    out=base.resample(rule, label='right', closed='right').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna()
    out.index=out.index-pd.Timedelta(microseconds=1)
    return out

def load_tf(symbol: str, tf: str, months=range(1,7)) -> pd.DataFrame:
    root=DATA_ROOT/symbol
    frames=[]
    for m in months:
        exact=list(root.rglob(f'{symbol}USDT-{tf}-2026-{m:02d}.csv'))
        if exact:
            p=sorted(exact, key=lambda x: len(str(x)))[0]
            frames.append(_read_month_file(p))
            continue
        # Fill missing higher-timeframe months from the finest reliable source.
        if tf=='5m':
            base3=list(root.rglob(f'{symbol}USDT-3m-2026-{m:02d}.csv'))
            if base3:
                frames.append(_resample_ohlcv(_read_month_file(sorted(base3,key=lambda x:len(str(x)))[0]), '5min'))
        elif tf=='15m':
            base5=list(root.rglob(f'{symbol}USDT-5m-2026-{m:02d}.csv'))
            if base5:
                frames.append(_resample_ohlcv(_read_month_file(sorted(base5,key=lambda x:len(str(x)))[0]), '15min'))
            else:
                base3=list(root.rglob(f'{symbol}USDT-3m-2026-{m:02d}.csv'))
                if base3:
                    frames.append(_resample_ohlcv(_read_month_file(sorted(base3,key=lambda x:len(str(x)))[0]), '15min'))
        elif tf in ('1h','4h'):
            base15=list(root.rglob(f'{symbol}USDT-15m-2026-{m:02d}.csv'))
            if base15:
                base=_read_month_file(sorted(base15,key=lambda x:len(str(x)))[0])
            else:
                base5=list(root.rglob(f'{symbol}USDT-5m-2026-{m:02d}.csv'))
                if not base5:
                    continue
                base=_read_month_file(sorted(base5,key=lambda x:len(str(x)))[0])
            frames.append(_resample_ohlcv(base, '1h' if tf=='1h' else '4h'))
    if not frames:
        raise FileNotFoundError(f'No {symbol} {tf}')
    out=pd.concat(frames).sort_index()
    out=out[~out.index.duplicated(keep='last')].dropna()
    return out

def adverse_fill(open_price: float, side: str, slip: float) -> float:
    return open_price*(1+slip) if side=='LONG' else open_price*(1-slip)

def adverse_exit(price: float, side: str, slip: float) -> float:
    return price*(1-slip) if side=='LONG' else price*(1+slip)

def fee_adjusted_runner_stop(entry, one_r, full_qty, remaining_qty, realized_net, entry_fee, tp1_fill, side, cfg):
    qty=max(remaining_qty,1e-12); fee=cfg.fee_rate
    market_buffer=max(one_r*cfg.be_market_buffer_r, entry*0.0001)
    # Fixed breakeven+N·R lock — mirror position_manager._fee_adjusted_runner_stop.
    lock=getattr(cfg,'runner_lock_r',None)
    if lock is not None:
        if side=='LONG':
            return min(entry+lock*one_r, tp1_fill-market_buffer)
        return max(entry-lock*one_r, tp1_fill+market_buffer)
    target_cash=full_qty*one_r*cfg.be_trade_lock_r
    remaining_entry_fee=entry_fee*(remaining_qty/max(full_qty,1e-12))
    cash_needed=target_cash-realized_net+remaining_entry_fee
    per_unit_needed=cash_needed/qty
    if side=='LONG':
        required=(entry+per_unit_needed)/max(1-fee,1e-12)
        max_valid=tp1_fill-market_buffer
        stop=min(max_valid,required) if required>max_valid else required
    else:
        required=(entry-per_unit_needed)/(1+fee)
        min_valid=tp1_fill+market_buffer
        stop=max(min_valid,required) if required<min_valid else required
    return stop

def trade_leg_net(entry, exit_price, qty, side, entry_fee_alloc, fee_rate):
    gross=(exit_price-entry)*qty if side=='LONG' else (entry-exit_price)*qty
    exit_fee=exit_price*qty*fee_rate
    return gross-entry_fee_alloc-exit_fee

def slice_asof(df, t, limit=320):
    # close-time index <= current signal bar close
    j=df.index.searchsorted(t, side='right')
    if j<=0: return df.iloc[:0]
    return df.iloc[max(0,j-limit):j]

def run_symbol(symbol: str, cfg_overrides: dict, start='2026-02-01', end='2026-06-01', verbose=False):
    d5=load_tf(symbol,'5m')
    d15=load_tf(symbol,'15m')
    d1=load_tf(symbol,'1h')
    d4=load_tf(symbol,'4h')
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    # temp state, no disk churn
    # Backtest-only memoization for repeated indicator requests on the same
    # closed-candle window. Production logic remains unchanged.
    import indicators as _ind_global
    def _frame_key(obj):
        try:
            if obj is None or len(obj)==0: return (None,0)
            idx=str(obj.index[-1]); n=len(obj)
            if hasattr(obj, 'columns') and 'close' in obj.columns:
                return (idx,n,round(float(obj['close'].iloc[0]),8),round(float(obj['close'].iloc[-1]),8))
            return (idx,n,round(float(obj.iloc[0]),8),round(float(obj.iloc[-1]),8))
        except Exception:
            return (id(obj),)
    def _memoize_indicator(name, maxsize=2048):
        original=getattr(_ind_global,name); cache={}
        def wrapped(*args,**kwargs):
            key=(name,)+tuple(_frame_key(x) if hasattr(x,'index') else x for x in args)+tuple(sorted(kwargs.items()))
            if key not in cache:
                cache[key]=original(*args,**kwargs)
                if len(cache)>maxsize: cache.pop(next(iter(cache)))
            return cache[key]
        setattr(_ind_global,name,wrapped)
    for _name in ('true_range','atr','adx','choppiness_index','market_structure','structure_flags','recent_swing_levels','latest_bos'):
        _memoize_indicator(_name)
    cfg=Config()
    cfg.state_dir=tempfile.mkdtemp(prefix=f'bt_{symbol}_')
    cfg.symbols=[f'{symbol}/USDT:USDT']
    for k,v in cfg_overrides.items(): setattr(cfg,k,v)
    pipe=Pipeline(cfg)
    pipe.entry_engine._persist_state=lambda: None
    pipe.entry_engine._state.clear()
    # Backtest-only memoization. Production code is unchanged.
    def memoize_df_method(obj, name, maxsize=64):
        original=getattr(obj,name); cache={}
        def wrapped(df,*args,**kwargs):
            key=(str(df.index[-1]) if df is not None and len(df) else None, len(df) if df is not None else 0, args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key]=original(df,*args,**kwargs)
                if len(cache)>maxsize: cache.pop(next(iter(cache)))
            return cache[key]
        setattr(obj,name,wrapped)
    memoize_df_method(pipe.regime_engine,'_tf_regime',16)
    memoize_df_method(pipe.bias_engine,'_tf_score',32)
    # Cache zone detection by closed HTF bar.
    _orig_zones=pipe.entry_engine._detect_zones; _zone_cache={}
    def cached_zones(df,timeframe,lookback):
        key=(timeframe,str(df.index[-1]) if df is not None and len(df) else None,len(df) if df is not None else 0,lookback)
        if key not in _zone_cache:
            _zone_cache[key]=_orig_zones(df,timeframe,lookback)
            if len(_zone_cache)>64: _zone_cache.pop(next(iter(_zone_cache)))
        return _zone_cache[key]
    pipe.entry_engine._detect_zones=cached_zones
    _orig_snap=pipe.entry_engine._snapshot; _snap_cache={}
    def cached_snapshot(df,direction):
        key=(str(df.index[-1]) if df is not None and len(df) else None,len(df) if df is not None else 0,direction)
        if key not in _snap_cache:
            _snap_cache[key]=_orig_snap(df,direction)
            if len(_snap_cache)>256: _snap_cache.pop(next(iter(_snap_cache)))
        return _snap_cache[key]
    pipe.entry_engine._snapshot=cached_snapshot
    _orig_nearest=pipe.entry_engine._nearest_structure; _nearest_cache={}
    def cached_nearest(price,frames):
        key=(round(float(price),8),tuple((str(f.index[-1]),len(f)) if f is not None and len(f) else (None,0) for f in frames))
        if key not in _nearest_cache:
            _nearest_cache[key]=_orig_nearest(price,frames)
            if len(_nearest_cache)>128: _nearest_cache.pop(next(iter(_nearest_cache)))
        return _nearest_cache[key]
    pipe.entry_engine._nearest_structure=cached_nearest
    sym=f'{symbol}/USDT:USDT'
    slip=cfg.expected_slippage_pct
    trades=[]; pos=None; cooldown_until=None
    signal_counts={'bias':0,'entry':0,'allowed':0}
    # enough data before start due min bars; loop only relevant indices
    idxs=np.where((d5.index>=start)&(d5.index<end))[0]
    for ii in idxs:
        t=d5.index[ii]
        row=d5.iloc[ii]
        # manage existing position using current bar OHLC, conservative ordering
        if pos is not None:
            side=pos['side']; long=side=='LONG'
            o,h,l,c=map(float,[row.open,row.high,row.low,row.close])
            # gap/open fills first
            exit_reason=None; exit_price=None
            if long:
                if o<=pos['stop']: exit_reason='BE_HIT' if pos['tp1_hit'] else 'SL_HIT'; exit_price=adverse_exit(o,side,slip)
                elif (not pos['tp1_hit']) and o>=pos['tp1']: pass
                elif o>=pos['tp2']: exit_reason='TP2_HIT'; exit_price=pos['tp2']
            else:
                if o>=pos['stop']: exit_reason='BE_HIT' if pos['tp1_hit'] else 'SL_HIT'; exit_price=adverse_exit(o,side,slip)
                elif (not pos['tp1_hit']) and o<=pos['tp1']: pass
                elif o<=pos['tp2']: exit_reason='TP2_HIT'; exit_price=pos['tp2']
            if exit_reason is None:
                # conservative: stop before profit target when both in same candle
                stop_hit = l<=pos['stop'] if long else h>=pos['stop']
                if stop_hit:
                    exit_reason='BE_HIT' if pos['tp1_hit'] else 'SL_HIT'; exit_price=adverse_exit(pos['stop'],side,slip)
                else:
                    if not pos['tp1_hit']:
                        tp1_hit = h>=pos['tp1'] if long else l<=pos['tp1']
                        if tp1_hit:
                            q=pos['full_qty']*cfg.tp1_fraction
                            fill=adverse_exit(pos['tp1'],side,slip)
                            entry_fee_alloc=pos['entry_fee']*(q/pos['full_qty'])
                            net=trade_leg_net(pos['entry'],fill,q,side,entry_fee_alloc,cfg.fee_rate)
                            pos['realized_net']+=net; pos['qty']-=q; pos['tp1_hit']=True
                            pos['stop']=fee_adjusted_runner_stop(pos['entry'],pos['one_r'],pos['full_qty'],pos['qty'],pos['realized_net'],pos['entry_fee'],fill,side,cfg)
                    if pos is not None and pos['tp1_hit']:
                        tp2_hit = h>=pos['tp2'] if long else l<=pos['tp2']
                        if tp2_hit:
                            exit_reason='TP2_HIT'; exit_price=adverse_exit(pos['tp2'],side,slip)
            # Production also evaluates a closed-5M signal exit after the hard
            # intrabar SL/TP checks. Include it here so backtests and live logic
            # use the same position-management path.
            if exit_reason is None and (not cfg.signal_exit_requires_tp1 or pos['tp1_hit']):
                history=d5.iloc[max(0,ii-139):ii+1]
                bars_since=max(0, int((history.index > pos['entry_time']).sum()))
                signal_exit=pipe.entry_engine.check_exit(history, side, bars_since)
                if signal_exit.should_exit:
                    exit_reason=signal_exit.reason
                    exit_price=adverse_exit(c,side,slip)
            if exit_reason is not None:
                q=pos['qty']; entry_fee_alloc=pos['entry_fee']*(q/pos['full_qty'])
                net_leg=trade_leg_net(pos['entry'],exit_price,q,side,entry_fee_alloc,cfg.fee_rate)
                total=pos['realized_net']+net_leg
                result_r=total/(pos['full_qty']*pos['one_r'])
                trades.append({**pos,'exit_time':t,'exit_price':exit_price,'exit_reason':exit_reason,'net_r':result_r,'net_pnl_unit':total})
                pipe.entry_engine.on_position_closed(sym,side,exit_reason,total,closed_at=t)
                cd=cfg.symbol_sl_cooldown_min if exit_reason=='SL_HIT' else cfg.symbol_be_cooldown_min if exit_reason=='BE_HIT' else cfg.symbol_cooldown_min
                cooldown_until=t+pd.Timedelta(minutes=cd)
                pos=None
            continue
        if cooldown_until is not None and t<cooldown_until:
            continue
        if ii+1>=len(d5): break
        # history windows
        w5=d5.iloc[max(0,ii-139):ii+1]
        w15=slice_asof(d15,t,220); w1=slice_asof(d1,t,220); w4=slice_asof(d4,t,220)
        if len(w5)<120 or min(len(w15),len(w1),len(w4))<200: continue
        try:
            res=pipe.evaluate(w1,w4,w15,w5,None,sym)
        except Exception as e:
            if verbose: print('ERR',symbol,t,type(e).__name__,e)
            continue
        if res.blocked_layer=='BIAS': signal_counts['bias']+=1
        elif res.blocked_layer=='ENTRY': signal_counts['entry']+=1
        if res.direction not in ('LONG','SHORT') or not res.entry or not res.entry.allow_entry:
            continue
        signal_counts['allowed']+=1
        side=res.direction; ent=res.entry
        nextrow=d5.iloc[ii+1]
        fill=adverse_fill(float(nextrow.open),side,slip)
        stop=float(ent.planned_stop); target=float(ent.planned_target)
        risk=(fill-stop) if side=='LONG' else (stop-fill)
        if risk<=0: continue
        rr=((target-fill)/risk) if side=='LONG' else ((fill-target)/risk)
        if rr<cfg.minimum_actual_rr: continue
        # unit qty, entry fee full
        qty=1.0; entry_fee=fill*qty*cfg.fee_rate
        tp1=fill+cfg.tp1_r*risk if side=='LONG' else fill-cfg.tp1_r*risk
        pos={'symbol':symbol,'side':side,'entry_time':d5.index[ii+1],'signal_time':t,'entry':fill,'stop':stop,'initial_stop':stop,
             'tp1':tp1,'tp2':target,'one_r':risk,'qty':qty,'full_qty':qty,'entry_fee':entry_fee,'realized_net':0.0,'tp1_hit':False,
             'setup_type':ent.setup_type,'trigger':ent.trigger,'score':ent.entry_score,'threshold':ent.score_threshold,'planned_rr':rr,
             'room_r':ent.structure_room_r,'regime':res.regime.label,'bias':res.bias.direction if res.bias else '', 'components': ent.score_components}
        pipe.entry_engine.confirm_entry(sym,ent.cross_id)
    shutil.rmtree(cfg.state_dir,ignore_errors=True)
    return trades, signal_counts

def metrics(trades):
    if not trades: return {'trades':0,'wr':0,'pf':0,'net_r':0,'avg_r':0,'mdd_r':0,'tpm':0}
    r=np.array([x['net_r'] for x in trades],float)
    wins=r[r>0]; losses=r[r<0]
    eq=np.cumsum(r); peak=np.maximum.accumulate(np.r_[0,eq])[:-1]; dd=eq-peak
    months=max(1, (max(x['exit_time'] for x in trades)-min(x['entry_time'] for x in trades)).days/30.44)
    return {'trades':len(r),'wr':100*len(wins)/len(r),'pf':wins.sum()/abs(losses.sum()) if len(losses) else float('inf'),
            'net_r':r.sum(),'avg_r':r.mean(),'mdd_r':dd.min() if len(dd) else 0,'tpm':len(r)/months,
            'tp2':sum(x['exit_reason']=='TP2_HIT' for x in trades),'sl':sum(x['exit_reason']=='SL_HIT' for x in trades),'be':sum(x['exit_reason']=='BE_HIT' for x in trades),
            'ema_cross_15m':sum(x['setup_type']=='EMA_CROSS_15M' for x in trades),
            'ema_cross_5m':sum(x['setup_type']=='EMA_CROSS_5M' for x in trades),
            'structure_pullback':sum(x['setup_type']=='STRUCTURE_PULLBACK' for x in trades),
            'smc':sum(x['setup_type']=='SMC_ZONE_REJECTION' for x in trades),
            'breakout':sum(x['setup_type'] in ('DIRECT_BREAKOUT','BREAKOUT_RETEST') for x in trades),
            'sweep':sum(x['setup_type']=='LIQUIDITY_SWEEP' for x in trades),
            'continuation':sum(x['setup_type']=='MOMENTUM_CONTINUATION' for x in trades),
            'range':sum(x['setup_type']=='RANGE_REVERSAL' for x in trades)}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTC,SOL')
    ap.add_argument('--start', default='2026-02-01')
    ap.add_argument('--end', default='2026-06-01')
    ap.add_argument('--overrides', default='')
    ap.add_argument('--data-root', default=str(DATA_ROOT), help='Directory containing recursively extracted symbol CSV files')
    a=ap.parse_args()
    DATA_ROOT = Path(a.data_root).resolve()
    ov={}
    if a.overrides:
        import json; ov=json.loads(a.overrides)
    alltr=[]
    for s in a.symbols.split(','):
        tr,sc=run_symbol(s.strip(),ov,a.start,a.end)
        print(s,metrics(tr),sc)
        alltr+=tr
    print('ALL',metrics(alltr))

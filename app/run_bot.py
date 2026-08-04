"""Adaptive Bot v13.2 runner with paper/live execution, Telegram charts and stats."""
from __future__ import annotations
import asyncio,json,logging,os,signal,sys,time,urllib.parse,urllib.request,uuid
from datetime import datetime,timezone
from typing import Any,Dict,List
import ccxt
from trading.connectors.binance_conn import BinanceConnector
from trading.adaptive_trading_bot import TradingBot
from trading.indicator_engine import compute,ema

BUILD_ID="adaptive-v13.2-price-action-2026-08-04"
logging.basicConfig(level=getattr(logging,os.getenv("LOG_LEVEL","INFO").upper(),logging.INFO),format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",stream=sys.stdout,force=True)
logger=logging.getLogger("adaptive_v13_2")

def env_bool(key,default=False):
    value=os.getenv(key); return default if value is None else value.lower() in ("1","true","yes","on")
def first_env(*keys):
    for key in keys:
        value=os.getenv(key,"").strip()
        if value:return value
    return ""
def fx_open(now):
    try:
        from zoneinfo import ZoneInfo
        ny=now.astimezone(ZoneInfo("America/New_York"))
    except Exception: ny=now
    if ny.weekday()<4:return True
    if ny.weekday()==4:return ny.hour<17
    if ny.weekday()==5:return False
    return ny.hour>=13
def load_json(path,default):
    try:return json.load(open(path,encoding="utf-8"))
    except Exception:return default
def save_json(path,value):
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True); tmp=path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False)
    os.replace(tmp,path)
def field(c,name,index):
    value=getattr(c,name,None)
    if value is None and isinstance(c,dict):value=c.get(name)
    if value is None and isinstance(c,(list,tuple)) and len(c)>index:value=c[index]
    return float(value or 0.0)
def timestamp(c):
    value=getattr(c,"timestamp",None)
    if value is None and isinstance(c,dict):value=c.get("timestamp")
    if value is None and isinstance(c,(list,tuple)) and c:value=c[0]
    return value
def duration(seconds):
    seconds=max(0,int(seconds)); days,seconds=divmod(seconds,86400); hours,seconds=divmod(seconds,3600); minutes=seconds//60
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m" if hours else f"{minutes}m"
def trade_text(order_type,p,paper):
    mode="PAPER" if paper else "LIVE"; direction=str(p.get("direction","")); symbol=str(p.get("symbol",""))
    if order_type.startswith("OPEN_"):
        return (f"{'🟢' if direction=='LONG' else '🔴'} [{mode}] OPEN {direction} {symbol}\n"
                f"Strategy: {p.get('strategy','unknown')}\nTrigger: {p.get('trigger','unknown')}\n"
                f"Entry: {float(p.get('entry',0)):,.6f}\nSL: {float(p.get('sl',0)):,.6f}\nTP: {float(p.get('tp',0)):,.6f} (2.00R)\n"
                f"Room: {float(p.get('room_r',0)):.2f}R\nSize: {float(p.get('size',0)):.6f}")
    pnl=float(p.get("pnl",0)); return (f"{'✅' if pnl>=0 else '❌'} [{mode}] CLOSE {direction} {symbol}\nPrice: {float(p.get('price',0)):,.6f}\nReason: {p.get('reason','unknown')}\nPnL: ${pnl:+.2f} ({float(p.get('r_multiple',0)):+.2f}R)")
def stats_text(trades,bots,prices,paper,margin):
    closed=[t for t in trades if t.get("event")=="CLOSE"]; wins=[t for t in closed if float(t.get("pnl",0))>0]; net=sum(float(t.get("pnl",0)) for t in closed)
    lines=["📊 Adaptive Bot v13.2 Stats","",f"Mode: {'PAPER' if paper else 'LIVE'}","",f"OPEN POSITIONS ({sum(int(b.position_open) for b in bots.values())})","――――――――――――――――"]
    open_count=0; floating=0.0
    for symbol,bot in bots.items():
        p=bot.position
        if not p:continue
        open_count+=1; current=float(prices.get(symbol,p.entry)); pnl=(current-p.entry)*p.size if p.direction=="LONG" else (p.entry-current)*p.size
        risk=abs(p.entry-p.initial_sl)*p.size; r=pnl/risk if risk>0 else 0; floating+=pnl
        lines += [f"{'🟢' if p.direction=='LONG' else '🔴'} {symbol.split('/')[0]} {p.direction}",f"Entry : {p.entry:,.6f}",f"Now   : {current:,.6f}",f"PnL   : ${pnl:+.2f} ({r:+.2f}R)",f"SL    : {p.sl:,.6f}{' (BE)' if p.be_moved else ''}",f"TP    : {p.tp:,.6f}",f"Strategy: {p.strategy}",f"Trigger : {p.trigger}",f"Held  : {duration(time.time()-p.opened_at)}",""]
    if not open_count:lines += ["No open positions",""]
    wr=100*len(wins)/len(closed) if closed else 0
    lines += ["EXPOSURE","――――――――――――――――",f"Margin Used  : ${open_count*margin:.2f}",f"Floating PnL : ${floating:+.2f}","","OVERALL","――――――――――――――――",f"Trades   : {len(closed)}  ({len(wins)}W / {len(closed)-len(wins)}L)",f"Win rate : {wr:.0f}%",f"TP hit   : {sum(t.get('reason')=='TP' for t in closed)}/{len(closed)}",f"SL hit   : {sum(t.get('reason')=='SL' for t in closed)}/{len(closed)}",f"BE exit  : {sum(t.get('reason')=='BE' for t in closed)}/{len(closed)}",f"EMA13 exit: {sum(t.get('reason')=='EMA13_TRAIL' for t in closed)}/{len(closed)}",f"Net PnL  : ${net:+.2f}"]
    if closed:
        lines += ["","LAST 5 TRADES","――――――――――――――――"]
        for i,t in enumerate(reversed(closed[-5:]),1):
            pnl=float(t.get("pnl",0)); lines.append(f"{i}. {'✅' if pnl>0 else '❌'} {str(t.get('symbol','')).split('/')[0]} {t.get('direction','')} ${pnl:+.2f} — {t.get('reason','')}")
    return "\n".join(lines)
def chart(candles,payload,path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as error:logger.error("Chart unavailable: %s",error);return False
    rows=list(candles[-80:])
    if len(rows)<30:return False
    opens=[field(x,"open",1) for x in rows]; highs=[field(x,"high",2) for x in rows]; lows=[field(x,"low",3) for x in rows]; closes=[field(x,"close",4) for x in rows]; volumes=[field(x,"volume",5) for x in rows]
    e8,e13,e20=ema(closes,8),ema(closes,13),ema(closes,20)
    fig,(ax,vax)=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={"height_ratios":[4,1]})
    for i,(o,h,l,c,v) in enumerate(zip(opens,highs,lows,closes,volumes)):
        color="#26a69a" if c>=o else "#ef5350"; ax.vlines(i,l,h,color=color,linewidth=1); ax.add_patch(Rectangle((i-.3,min(o,c)),.6,max(abs(c-o),max(c,1)*1e-6),facecolor=color,edgecolor=color)); vax.bar(i,v,width=.7,color=color)
    ax.plot(e8,label="EMA8",linewidth=1.0); ax.plot(e13,label="EMA13",linewidth=1.0); ax.plot(e20,label="EMA20",linewidth=1.2)
    for key,label in (("entry","ENTRY"),("sl","SL"),("tp","TP 2R")):
        value=float(payload.get(key,0)); ax.axhline(value,linestyle="--",linewidth=1.3,label=f"{label} {value:,.4f}")
    entry=float(payload.get("entry",0)); ax.scatter(len(rows)-1,entry,marker="^" if payload.get("direction")=="LONG" else "v",s=100,zorder=6)
    swing=float(payload.get("structure_level",payload.get("sl",0))); ax.scatter(len(rows)-3,swing,marker="o",s=45,zorder=5,label="Structure")
    ax.legend(loc="best",fontsize=8); ax.grid(alpha=.2); vax.grid(alpha=.15); ax.set_title(f"{payload.get('symbol')} {payload.get('direction')} | 15M | {payload.get('strategy')} | {payload.get('trigger')}"); ax.set_ylabel("Price"); vax.set_ylabel("Volume"); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
    return os.path.exists(path) and os.path.getsize(path)>0

async def main():
    paper=env_bool("PAPER_TRADING",True) or os.getenv("TRADING_MODE","").lower()=="paper"
    symbols=[s.strip() for s in os.getenv("SYMBOLS","BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT").split(",") if s.strip()]
    leverage=int(os.getenv("LEVERAGE","20")); margin=float(os.getenv("ADAPTIVE_MARGIN_USDT","20")); interval=int(os.getenv("INTERVAL_SECONDS","60")); max_positions=int(os.getenv("MAX_POSITIONS","2"))
    token=first_env("TELEGRAM_BOT_TOKEN","TELEGRAM_TOKEN","TG_BOT_TOKEN"); chat_id=first_env("TELEGRAM_CHAT_ID","TG_CHAT_ID","CHAT_ID"); tg=bool(token and chat_id); queue=asyncio.Queue(); offset=0
    state_dir=os.getenv("BOT_STATE_DIR","/tmp/adaptive_v13_2"); ledger=os.getenv("TRADE_LEDGER_FILE",os.path.join(state_dir,"trade_ledger_v13_2.json")); trades=load_json(ledger,[]); latest={}; prices={}
    def tg_api(method,fields,photo=""):
        url=f"https://api.telegram.org/bot{token}/{method}"
        if not photo:req=urllib.request.Request(url,data=urllib.parse.urlencode(fields).encode(),method="POST")
        else:
            boundary="----Adaptive"+uuid.uuid4().hex; parts=[]
            for key,value in fields.items():parts += [f"--{boundary}\r\n".encode(),f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),str(value).encode(),b"\r\n"]
            image=open(photo,"rb").read(); parts += [f"--{boundary}\r\n".encode(),b'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n',b"Content-Type: image/png\r\n\r\n",image,b"\r\n",f"--{boundary}--\r\n".encode()]
            req=urllib.request.Request(url,data=b"".join(parts),method="POST",headers={"Content-Type":f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req,timeout=20) as response:result=json.loads(response.read().decode())
        if not result.get("ok"):raise RuntimeError(result.get("description","Telegram failed"))
        return result
    async def worker():
        while True:
            item=await queue.get()
            try:
                if item["kind"]=="photo":await asyncio.to_thread(tg_api,"sendPhoto",{"chat_id":chat_id,"caption":item["caption"]},item["path"]); os.remove(item["path"])
                else:await asyncio.to_thread(tg_api,"sendMessage",{"chat_id":chat_id,"text":item["text"],"disable_web_page_preview":"true"})
            except asyncio.CancelledError:raise
            except Exception as error:logger.warning("Telegram failed: %s",error)
            finally:queue.task_done()
    connector=BinanceConnector(api_key="" if paper else os.getenv("EXCHANGE_API_KEY",""),api_secret="" if paper else os.getenv("EXCHANGE_API_SECRET",""),paper=True,exchange_id=os.getenv("EXCHANGE","okx"),passphrase="" if paper else os.getenv("EXCHANGE_PASSPHRASE",""),leverage=leverage)
    live=None
    if not paper:
        from trading.connectors.okx_adapter import OKXAdapter
        live=OKXAdapter(api_key=os.getenv("EXCHANGE_API_KEY",""),api_secret=os.getenv("EXCHANGE_API_SECRET",""),api_passphrase=os.getenv("EXCHANGE_PASSPHRASE",""),paper=False,leverage=leverage)
    def execute(order_type,payload):
        if order_type.startswith("OPEN_"):
            payload["structure_level"]=payload["sl"]
            if tg:
                path=f"/tmp/v132_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                if not chart(latest.get(str(payload.get("symbol")),[]),payload,path):raise RuntimeError("Mandatory Telegram chart failed")
                queue.put_nowait({"kind":"photo","path":path,"caption":trade_text(order_type,payload,paper)})
        elif tg:queue.put_nowait({"kind":"text","text":trade_text(order_type,payload,paper)})
        if paper:logger.info("[PAPER] %s %s",order_type,payload);return {"paper":True}
        if live is None:raise RuntimeError("Live adapter unavailable")
        return live.execute(order_type,payload)
    bots={s:TradingBot(s,margin,leverage,paper,os.path.join(state_dir,s.replace("/","_").replace(":","_")+".json"),execute) for s in symbols}
    async def commands():
        nonlocal offset
        if not tg:return
        try:
            result=await asyncio.to_thread(tg_api,"getUpdates",{"timeout":0,"offset":offset})
            for update in result.get("result",[]):
                offset=max(offset,int(update.get("update_id",0))+1); message=update.get("message") or {}
                if str((message.get("chat") or {}).get("id",""))!=str(chat_id):continue
                text=str(message.get("text","")).lower()
                if text.startswith("/stats") or text.startswith("/restats"):queue.put_nowait({"kind":"text","text":stats_text(trades,bots,prices,paper,margin)})
        except Exception as error:logger.warning("Telegram polling: %s",error)
    stop=asyncio.Event(); loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):
        try:loop.add_signal_handler(sig,stop.set)
        except NotImplementedError:pass
    task=asyncio.create_task(worker()) if tg else None; last={}; disabled=set()
    logger.info("Adaptive Bot v13.2 | build=%s | mode=%s | telegram=%s chart=MANDATORY",BUILD_ID,"PAPER" if paper else "LIVE","CONNECTED" if tg else "DISABLED")
    if tg:queue.put_nowait({"kind":"text","text":f"🤖 Adaptive Bot v13.2 started\nMode: {'PAPER' if paper else 'LIVE'}\nLogic: 4H Trend → 1H Quality → 15M Location + Price Action\nTriggers: EMA8/13, Engulfing, Hammer, Inside Break, Continuation\nSL: Swing+ATR | TP: 2R | BE: 1R\nCommand: /stats"})
    try:
        while not stop.is_set():
            await commands(); entries=fx_open(datetime.now(timezone.utc))
            for symbol in symbols:
                if symbol in disabled:continue
                try:
                    r15=await connector.fetch_ohlcv(symbol,"15m",300); r1=await connector.fetch_ohlcv(symbol,"1h",200); r4=await connector.fetch_ohlcv(symbol,"4h",200)
                    if len(r15)<82 or len(r1)<82 or len(r4)<82:continue
                    c15,c1,c4=list(r15[:-1]),list(r1[:-1]),list(r4[:-1]); latest[symbol]=c15; ts=timestamp(c15[-1])
                    if ts==last.get(symbol):continue
                    last[symbol]=ts; i15,i1,i4=compute(c15),compute(c1),compute(c4)
                    if not i15 or not i1 or not i4:continue
                    prices[symbol]=float(i15["close"]); bot=bots[symbol]
                    if not bot.position_open:
                        if not entries:logger.info("[%s] SLEEP_MODE",symbol);continue
                        if sum(int(b.position_open) for b in bots.values())>=max_positions:logger.info("[%s] WAIT max positions",symbol);continue
                    event=bot.on_bar(i15,i1,i4,prices[symbol])
                    if event:trades.append({**event,"timestamp":time.time(),"version":"v13.2"});save_json(ledger,trades[-2000:])
                    logger.info("[%s] %s",symbol,event if event else bot.last_signal)
                except ccxt.BadSymbol as error:disabled.add(symbol);logger.error("[%s] unsupported: %s",symbol,error)
                except (ccxt.NetworkError,asyncio.TimeoutError) as error:logger.warning("[%s] network: %s",symbol,error)
                except Exception as error:
                    logger.exception("[%s] tick failed",symbol)
                    if tg:queue.put_nowait({"kind":"text","text":f"❌ Adaptive v13.2 error\nSymbol: {symbol}\n{type(error).__name__}: {error}"})
            try:await asyncio.wait_for(stop.wait(),timeout=interval)
            except asyncio.TimeoutError:pass
    finally:
        if tg:queue.put_nowait({"kind":"text","text":"⏹ Adaptive Bot v13.2 stopped"})
        if task:task.cancel()

if __name__=="__main__":asyncio.run(main())

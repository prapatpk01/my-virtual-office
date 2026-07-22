from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_exact_5m as bt

def task(args):
    symbol,start,end,ov=args
    bt.DATA_ROOT=Path('/mnt/data/historical_data')
    tr,sc=bt.run_symbol(symbol,ov,start,end)
    return symbol,start,tr,sc

def run(symbols, months, ov, workers=8):
    tasks=[]
    for s in symbols:
      for m in months:
        start=f'2026-{m:02d}-01'
        end=str((pd.Timestamp(start)+pd.offsets.MonthBegin(1)).date())
        tasks.append((s,start,end,ov))
    alltr=[]; rows=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
      futs=[ex.submit(task,x) for x in tasks]
      for f in as_completed(futs):
        s,st,tr,sc=f.result(); alltr+=tr; rows.append((s,st,bt.metrics(tr),sc)); print(rows[-1],flush=True)
    for s in symbols:
      tr=[x for x in alltr if x['symbol']==s]
      print('SYMBOL',s,bt.metrics(tr),flush=True)
    print('ALL',bt.metrics(alltr),flush=True)
    return alltr, rows

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('--overrides',default='{}');ap.add_argument('--months',default='2,3,4,5');ap.add_argument('--symbols',default='BTC,SOL');ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();run(a.symbols.split(','),[int(x) for x in a.months.split(',')],json.loads(a.overrides),a.workers)

import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import backtest_exact_5m as bt
bt.DATA_ROOT=Path('/mnt/data/historical_data')
sym,start,end=sys.argv[1:4]
ov=json.loads(sys.argv[4]) if len(sys.argv)>4 else {}
tr,sc=bt.run_symbol(sym,ov,start,end)
print(bt.metrics(tr),sc)
for x in tr:
 print(x['entry_time'],x['side'],x['setup_type'],x['trigger'],round(x['score'],1),round(x['net_r'],3),x['exit_reason'],round(x['planned_rr'],2),round(x['room_r'],2),x.get('components'))

#!/bin/zsh
# Wait for the scorer's result json, then read the paginated board.
cd ~/Projects/prc-challenge || exit 1
until PYTHONDONTWRITEBYTECODE=1 python3.11 -B -c "
import sys
from prc.s3 import S3, load_credentials
sys.exit(0 if 'merry-quicksand_v5.parquet_result.json' in [o['key'] for o in S3(load_credentials()).list_objects('prc-2026-merry-quicksand')] else 1)"; do sleep 20; done
sleep 25
cd /tmp && python3.11 - <<'PY'
import json,urllib.request,datetime
base="https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard?limit=200"
rows=[];cur=None
while True:
    d=json.load(urllib.request.urlopen(base+(f"&cursor={cur}" if cur else ""),timeout=60)); rows+=d["items"]; cur=d.get("nextCursor")
    if not cur: break
ours={r["filename"]:r["score"] for r in rows if r["teamName"]=="merry-quicksand"}
print("ours:", {k:round(v,4) for k,v in sorted(ours.items()) if v is not None})
best={}
for r in rows:
    t,s=r["teamName"],r["score"]
    if s is not None and (t not in best or s<best[t]): best[t]=s
v=sorted(best.values()); m=best["merry-quicksand"]
print(datetime.datetime.now().strftime("%H:%M"), f"best {m:.4f} -> rank {v.index(m)+1} of {len(v)} | 10th {v[9]:.2f} | leader {v[0]:.2f}")
s5,s4=ours.get("merry-quicksand_v5.parquet"),ours.get("merry-quicksand_v4.parquet")
if s5 is not None and s4 is not None:
    print(f"v5 vs v4: {s5-s4:+.2f} s = {s5**2-s4**2:+,.0f} MSE (matched lane: seeds + queue block)")
PY

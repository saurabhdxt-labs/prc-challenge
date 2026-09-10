"""Apply prereg Amendment 20.3 to a lgbm_fold queue_allrows JSON. Usage: result11_clauses.py <json>"""
import json, sys
d = json.load(open(sys.argv[1]))
tag = "SMOKE — NOT A RESULT" if d.get("smoke") else "fold A"
print(f"[{tag}] mode {d['mode']}  git {d['git_sha'][:7]}{' dirty' if d['git_dirty'] else ''}  wall {d['wall_s']/3600:.2f} h  peak {d['peak_rss_gb']} GB")
print(f"seed sd {d['seed_sd']['sd']:.3f} s ({d['seed_sd']['source']}); U arms best_iter/n_ref: " +
      ", ".join(f"{k}: {v['best_iter']}/{v['n_ref']}" for k, v in d['u_arms'].items()))
arms = [a for a in d["arms"] if a != "pipeline"]
sd = d["seed_sd"]["sd"]
def show(subset, title):
    s = d["subsets"][subset]; n = s["n_rows"]; r = s["rmse"]
    print(f"\n== {title}: n={n:,}  pipeline RMSE {r['pipeline']:.3f}")
    first = True
    for a in arms:
        p = s["pairs"].get(f"{a}_vs_pipeline")
        if p is None: continue
        if first: print("   pair keys:", list(p.keys())); first = False
        lo, hi = p["ci95"]; g = p["gain_s"]
        excl_fav = lo > 0
        imp = p.get("airports_improving", p.get("n_airports_improving"))
        print(f"   {a:11s} RMSE {r[a]:9.3f}  gain {g:+9.3f}  95% [{lo:+.3f}, {hi:+.3f}]  excludes 0 in arm's favour: {excl_fav}  gain/sd {g/sd:+.1f}  airports {imp}")
    return s
m = show("matched", "MATCHED rows (20.3 i)")
print("   20.3(i) verdicts:")
for w in ("w1", "w10"):
    u, b = f"U_{w}", f"Ublend_{w}"
    pu, pb = m["pairs"].get(f"{u}_vs_pipeline"), m["pairs"].get(f"{b}_vs_pipeline")
    if pu: print(f"     {u}: ADOPTED for matched rows = {pu['ci95'][0] > 0}")
    if pb: print(f"     {b}: adopted = interval excludes 0 ({pb['ci95'][0] > 0}) AND beats U alone ({m['rmse'][b] < m['rmse'][u]}) -> {pb['ci95'][0] > 0 and m['rmse'][b] < m['rmse'][u]}")
ex = show("unmatched_exmonster", "UNMATCHED ex-monster (decisional, 20.3 ii)")
po = show("unmatched", "UNMATCHED pooled (reported; not worse by > 20 s)")
print("   20.3(ii) verdicts:")
for a in arms:
    pe, pp = ex["pairs"].get(f"{a}_vs_pipeline"), po["pairs"].get(f"{a}_vs_pipeline")
    if pe and pp: print(f"     {a}: ex-monster excludes 0 in arm's favour {pe['ci95'][0] > 0}; pooled gain {pp['gain_s']:+.1f} s (not worse by > 20 s: {pp['gain_s'] >= -20}) -> ADOPTED for unmatched = {pe['ci95'][0] > 0 and pp['gain_s'] >= -20}")
t = d["total_2026"]
print(f"\n== TOTAL at 2026 weights (scale only, never a board prediction): pipeline {t['rmse']['pipeline']:.3f}; " + "; ".join(f"{a} {t['rmse'][a]:.3f} ({t['pairs'][a+'_vs_pipeline']['gain_s']:+.2f} s [{t['pairs'][a+'_vs_pipeline']['ci95'][0]:+.2f}, {t['pairs'][a+'_vs_pipeline']['ci95'][1]:+.2f}])" for a in arms if a+"_vs_pipeline" in t["pairs"]))
print("\n== |delta| bands (matched), gain vs pipeline with 95% CI:")
for b, v in d["bands"].items():
    row = f"   {b:6s} n={v['n_rows']:7,} SSE {v['share_of_pipeline_sse']*100:5.1f}%  pipeline {v['rmse']['pipeline']:8.2f}"
    for a in arms:
        p = v["pairs"].get(f"{a}_vs_pipeline")
        if p: row += f"  {a} {p['gain_s']:+7.2f} [{p['ci95'][0]:+.1f},{p['ci95'][1]:+.1f}]"
    print(row)
pa = m["per_airport"]
print("\n== per-airport matched RMSE, pipeline vs arms:")
for ap in m["airports"]:
    print(f"   {ap}: " + "  ".join(f"{a} {pa[a][ap]:7.2f}" for a in ["pipeline"] + arms if a in pa))

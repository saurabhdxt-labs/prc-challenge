"""Apply a block arm's REGISTERED clauses to its fold record. Usage: block_arm_verdict.py <json>...

Arm D  (19.3): interval excludes zero, gain >= 1.0 s, gain > 2 x seed sd, >= 6 of 10 airports,
               AND the |delta| > 10 min bands improve by >= 5 s.
Arm F  (22.3): interval excludes zero, gain >= 1.0 s, gain > 2 x seed sd, AND the over10 band's
               own interval excludes zero.
Arm W  (24.4): the same four clauses as arm F, plus January and July reported separately.
Arm FW (plans/PREREG_fw_combined_2026_09_10.md H-FW1): arm F's four clauses, against the queue
               baseline - reported on F's footing and NOT decisional; H-FW2 (FW against F) is
               tools/pair_arms.py.
The script states each clause and its measured value; it never invents a threshold.
"""
import json, sys

CLAUSES = {
    "19.1": dict(name="ARM D", bar=1.0, min_airports=6, band_rule="gain >= +5 s on the over-10-min bands",
                 band_min=5.0, band_needs_interval=False),
    "22":   dict(name="ARM F", bar=1.0, min_airports=None, band_rule="the over10 band's own interval excludes zero",
                 band_min=None, band_needs_interval=True),
    "24":   dict(name="ARM W", bar=1.0, min_airports=None, band_rule="the over10 band's own interval excludes zero",
                 band_min=None, band_needs_interval=True),
    "FW":   dict(name="ARM FW (H-FW1, not decisional)", bar=1.0, min_airports=None,
                 band_rule="the over10 band's own interval excludes zero", band_min=None, band_needs_interval=True),
}

def verdict(path):
    d = json.load(open(path))
    c = CLAUSES.get(d.get("amendment"))
    tag = "SMOKE - NOT A RESULT" if d.get("smoke") else "fold A"
    print(f"\n=== {path}")
    print(f"[{tag}] mode {d['mode']}  amendment {d.get('amendment')}  git {d['git_sha'][:7]}"
          f"{' dirty' if d.get('git_dirty') else ''}  wall {d['wall_s']/3600:.2f} h  peak {d['peak_rss_gb']} GB")
    if c is None:
        print(f"  no registered clause set for amendment {d.get('amendment')!r}"); return
    a = d["arms"]; p = d["pairs"]["treatment_vs_baseline"]; sd = d["seed_sd"]["sd"]
    print(f"  baseline {a['baseline']['rmse']:.3f}   treatment {a['treatment']['rmse']:.3f}   "
          f"n_features {d['config']['n_features']}   treatment n_ref {d['treatment']['n_ref']:,}")
    lo, hi = p["ci95"]; g = p["gain_s"]
    band = (d.get("bands", {}).get("over10", {}).get("pairs") or {}).get("treatment_vs_baseline")
    checks = [
        ("(i)   the paired interval excludes zero in the block's favour", lo > 0, f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]"),
        (f"(ii)  point gain >= +{c['bar']} s", g >= c["bar"], f"{g:+.3f} s"),
        ("(iii) gain > 2 x seed sd", (sd is not None and g > 2 * sd), f"{g:+.3f} vs 2 x {sd:.4f}" if sd else "no seed sd"),
    ]
    if c["min_airports"]:
        checks.append((f"(iv)  >= {c['min_airports']} of 10 airports improve",
                       p["airports_improving"] >= c["min_airports"], f"{p['airports_improving']} / {p['n_airports']}"))
    if band is None:
        checks.append((f"({'v' if c['min_airports'] else 'iv'})   {c['band_rule']}", False, "no over10 interval"))
    elif c["band_needs_interval"]:
        checks.append((f"(iv)  {c['band_rule']}", band["ci95"][0] > 0,
                       f"{band['gain_s']:+.3f} [{band['ci95'][0]:+.3f}, {band['ci95'][1]:+.3f}]"))
    else:
        checks.append((f"(v)   {c['band_rule']}", band["gain_s"] >= c["band_min"],
                       f"{band['gain_s']:+.3f} [{band['ci95'][0]:+.3f}, {band['ci95'][1]:+.3f}]"))
    for text, ok, measured in checks:
        print(f"    {text:62s} {'PASS' if ok else 'fail'}   {measured}")
    excl = checks[0][1]
    allc = all(ok for _, ok, _ in checks)
    print(f"  VERDICT: {'ESTABLISHED' if allc else ('NOT WORKING' if not excl else 'INCONCLUSIVE - the interval holds, a later clause does not; the ship call is the owners (RESULT 9 precedent)')}")
    pa = d["arms"]["treatment"].get("per_airport") or d["subsets"]["matched"]["per_airport"]["treatment"] if "subsets" in d else None
    if isinstance(pa, dict):
        base = d["arms"]["baseline"].get("per_airport", {})
        line = "  per airport (treatment - baseline): " + "  ".join(
            f"{k} {base.get(k, float('nan')) - v:+.2f}" for k, v in sorted(pa.items()))
        print(line)
    for name, rec in (d.get("bands") or {}).items():
        pr = (rec.get("pairs") or {}).get("treatment_vs_baseline")
        if pr:
            print(f"    band {name:7s} n {rec['n_rows']:7,}  SSE {100*rec.get('share_of_baseline_sse', 0):5.1f}%  "
                  f"gain {pr['gain_s']:+7.3f} [{pr['ci95'][0]:+7.3f}, {pr['ci95'][1]:+7.3f}]{'*' if pr['excludes_zero'] else ''}")

for f in sys.argv[1:]:
    verdict(f)

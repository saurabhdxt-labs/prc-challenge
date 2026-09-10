"""tools/pair_arms.py: pair the `treatment` columns of TWO fold records on identical rows.

The decisional instrument of plans/PREREG_fw_combined_2026_09_10.md H-FW2 (arm FW against arm F).
Every guard here exists because a BC-2 instance (reports/bug_classes.md) got past inspection and
was caught only by a control: the row-identity check, the reproduce-from-own-JSON check (which is
also the scale-convention check - a taxi-time column recovered as a delta does not reproduce), and
the sign-swap negative control that must centre on zero by construction.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import pair_arms as pa  # noqa: E402

APTS = ("AAAA", "BBBB", "CCCC")


def _records(tmp_path, *, body_gain=0.0, tail_loss=0.0, n=6000, seed=0, new_equals_ref=False, body_hi=600.0):
    """Two synthetic records (ref = "F", new = "FW") sharing rows, labels and the baseline column.
    delta is heavy-tailed so the over10 band exists; the ref treatment carries noise sd 60 s on
    every row; the new treatment carries sd (60 - body_gain) on rows with |delta| < body_hi,
    sd 60 on the rest of the body, and sd (60 + tail_loss) on over10 rows. Both parquets store
    the DELTA convention."""
    rng = np.random.default_rng(seed)
    delta = rng.standard_t(3, n) * 250.0
    proxy = np.abs(delta) + 900.0 + rng.uniform(0, 300, n)
    y = proxy - delta
    base_rows = dict(row=np.arange(n), month=np.where(np.arange(n) % 2 == 0, 1, 7),
                     ap=np.array(APTS)[np.arange(n) % 3], y=y, proxy=proxy, delta=delta,
                     sp=y + rng.normal(0, 50, n), baseline=delta + rng.normal(0, 80, n))
    a = np.abs(delta)
    ref_t = delta + rng.normal(0, 60, n)
    sd_new = np.where(a < body_hi, 60 - body_gain, np.where(a < 600.0, 60.0, 60 + tail_loss))
    new_t = ref_t.copy() if new_equals_ref else delta + rng.normal(0, 1, n) * sd_new
    out = {}
    for name, t, tsd in (("ref", ref_t, 0.12), ("new", new_t, 0.10)):
        d = pd.DataFrame({**base_rows, "treatment": t})
        pq = tmp_path / f"{name}.parquet"
        d.to_parquet(pq)
        rm = {c: float(np.sqrt(((np.maximum(d.proxy - d[c], 1.0) - d.y) ** 2).mean())) for c in ("baseline", "treatment")}
        js = {"mode": f"synthetic_{name}", "arms": {c: {"rmse": rm[c]} for c in rm},
              "seed_sd": {"sd": 0.14538, "treatment_sd": tsd}}
        jp = tmp_path / f"{name}.json"
        jp.write_text(json.dumps(js))
        out[name] = (jp, pq)
    return out


def test_compare_detects_a_planted_body_gain_that_keeps_the_tail(tmp_path):
    """New improves the body by construction (noise 60 -> 45 s) and leaves the tail alone: all four
    H-FW2 clauses hold and the verdict is COMBINES. Per-airport and per-month cuts are present and
    equal a direct recomputation; the negative control's p99 is far below the real gain.

    Fails when the gain's sign is flipped, or when the airport cut is not the airport's rows.
    Rehearsed 2026-09-10, each RED: `"gain_s": -p["gain_s"]` in _pair; the per-airport mask
    replaced by the full-row mask.
    """
    r = _records(tmp_path, body_gain=15.0)
    res = pa.compare(pa.load_record(*r["new"]), pa.load_record(*r["ref"]), n_boot=300, n_control=100, sd_floor=0.291)
    c = res["clauses"]
    assert res["pooled"]["gain_s"] > 2.0 and res["pooled"]["ci95"][0] > 0
    assert c["i_interval_excludes_zero"] and c["ii_gain_over_2x_sd"] and c["iii_tail_kept"] and c["iv_body_bands_gain"]
    assert res["verdict"] == "COMBINES"
    assert res["bands"]["lt2"]["gain_s"] > 0 and res["bands"]["2to10"]["gain_s"] > 0
    assert set(res["per_airport"]) == set(APTS) and set(res["per_month"]) == {"1", "7"}
    new, ref = pa.load_record(*r["new"]), pa.load_record(*r["ref"])
    m = (new["preds"].ap == "BBBB").to_numpy()
    e_new = (pa.recover(new["preds"], "treatment") - new["preds"].y.to_numpy()) ** 2
    e_ref = (pa.recover(ref["preds"], "treatment") - ref["preds"].y.to_numpy()) ** 2
    direct = float(np.sqrt(e_ref[m].mean()) - np.sqrt(e_new[m].mean()))
    assert res["per_airport"]["BBBB"]["gain_s"] == pytest.approx(direct, abs=1e-9)
    assert res["control"]["p99_abs_gain"] < 0.25 * res["pooled"]["gain_s"]
    assert res["control"]["real_over_p99"] > 4.0


def test_giving_back_the_tail_is_inconclusive_not_combines(tmp_path):
    """New gains the body but pays heavily in the tail (noise 60 -> 160 s on over10 rows): the tail
    clause (iii) fails, so the verdict is INCONCLUSIVE (or NOT WORKING if the pooled interval also
    fails) - never COMBINES. Fails when clause (iii) tests the upper bound in the wrong direction.
    Rehearsed 2026-09-10 RED: `hi >= 0` replaced by `lo <= 0` in the tail clause."""
    r = _records(tmp_path, body_gain=15.0, tail_loss=100.0)
    res = pa.compare(pa.load_record(*r["new"]), pa.load_record(*r["ref"]), n_boot=300, n_control=50, sd_floor=0.291)
    assert res["bands"]["over10"]["ci95"][1] < 0            # the tail interval excludes zero in ref's favour
    assert res["clauses"]["iii_tail_kept"] is False
    assert res["verdict"] != "COMBINES"


def test_a_gain_on_one_body_band_only_fails_the_mechanism_clause(tmp_path):
    """New gains on the < 2 min band only; the 2-10 min band is untouched (both arms noise 60 s,
    independent draws). Clause (iv) needs BOTH body bands' intervals to exclude zero, so it fails
    and the verdict is not COMBINES even though the pooled interval may hold.
    Rehearsed 2026-09-10 RED: clause (iv) reduced to the lt2 band."""
    r = _records(tmp_path, body_gain=20.0, body_hi=120.0)
    res = pa.compare(pa.load_record(*r["new"]), pa.load_record(*r["ref"]), n_boot=300, n_control=50, sd_floor=0.291)
    assert res["bands"]["lt2"]["ci95"][0] > 0
    assert not res["bands"]["2to10"]["ci95"][0] > 0
    assert res["clauses"]["iv_body_bands_gain"] is False and res["verdict"] != "COMBINES"


def test_identical_arms_are_not_working_and_the_control_is_exactly_zero(tmp_path):
    """new == ref row for row: the gain, its interval and every control draw are exactly 0, so
    clause (i) fails and the verdict is NOT WORKING; the control ratio is None rather than a
    division by zero."""
    r = _records(tmp_path, new_equals_ref=True)
    res = pa.compare(pa.load_record(*r["new"]), pa.load_record(*r["ref"]), n_boot=100, n_control=20, sd_floor=0.291)
    assert res["pooled"]["gain_s"] == 0.0 and res["pooled"]["ci95"] == [0.0, 0.0]
    assert res["control"]["p99_abs_gain"] == 0.0 and res["control"]["real_over_p99"] is None
    assert res["verdict"] == "NOT WORKING"


@pytest.mark.parametrize("col", ["row", "month", "ap", "y", "proxy", "baseline"])
def test_a_row_identity_mismatch_voids_the_pairing_and_names_the_column(tmp_path, col):
    """One changed value in any identity column refuses the comparison (H-FW2 is VOID, not re-run
    with a looser check), and the error names the column. main() exits 3.
    Rehearsed 2026-09-10 RED: `baseline` dropped from IDENTITY_COLS (the baseline case passes)."""
    r = _records(tmp_path, body_gain=10.0)
    d = pd.read_parquet(r["new"][1])
    d.loc[5, col] = "ZZZZ" if col == "ap" else d.loc[5, col] + 1
    d.to_parquet(r["new"][1])
    with pytest.raises(pa.IdentityError, match=col):
        pa.compare(pa.load_record(*r["new"]), pa.load_record(*r["ref"]), n_boot=50, n_control=10)
    assert pa.main([str(r["new"][0]), str(r["new"][1]), str(r["ref"][0]), str(r["ref"][1]),
                    "--n-boot", "50", "--n-control", "10", "--out", str(tmp_path / "o.json")]) == 3


def test_a_taxi_time_column_read_as_a_delta_does_not_reproduce(tmp_path):
    """BC-2 instance 2: the new record stores its treatment as TAXI TIMES while its JSON RMSE was
    computed on them directly. Recovered under the delta convention it misses its own JSON by far
    more than the tolerance, and the check refuses, naming the arm. main() exits 2.
    Rehearsed 2026-09-10 RED: the reproduce check skipped for the treatment arm."""
    r = _records(tmp_path, body_gain=10.0)
    d = pd.read_parquet(r["new"][1])
    d["treatment"] = np.maximum(d.proxy - d.treatment, 1.0)                   # now a taxi time
    d.to_parquet(r["new"][1])
    js = json.loads(r["new"][0].read_text())
    js["arms"]["treatment"]["rmse"] = float(np.sqrt(((d.treatment - d.y) ** 2).mean()))
    r["new"][0].write_text(json.dumps(js))
    with pytest.raises(pa.ReproduceError, match="treatment"):
        pa.check_reproduces(pa.load_record(*r["new"]))
    assert pa.main([str(r["new"][0]), str(r["new"][1]), str(r["ref"][0]), str(r["ref"][1]),
                    "--n-boot", "50", "--n-control", "10", "--out", str(tmp_path / "o.json")]) == 2


def test_rule_sd_is_the_largest_recorded_value_and_the_registered_floor():
    """The prereg's correction: sd = max(both arms' seed_sd.sd, both treatment sds, the floor)."""
    a = {"seed_sd": {"sd": 0.14538, "treatment_sd": 0.12769}}
    b = {"seed_sd": {"sd": 0.14538, "treatment_sd": 0.31}}
    assert pa.rule_sd(a, b, floor=0.291) == 0.31
    assert pa.rule_sd(a, a, floor=0.291) == 0.291
    assert pa.rule_sd(a, a, floor=None) == 0.14538
    assert pa.rule_sd({"seed_sd": {"sd": None, "treatment_sd": None}}, a, floor=None) == 0.14538
    with pytest.raises(ValueError, match="no seed sd"):
        pa.rule_sd({"seed_sd": {}}, {"seed_sd": {}}, floor=None)


def test_sign_swap_control_is_centred_on_zero_for_a_real_difference():
    """Swapping which arm is "new" on a random half of the rows destroys any real difference: the
    control's mean gain is ~0 even when the real gain is large. Tolerance: 0.05 x the real gain -
    with 200 draws on 20,000 rows the control's sd is a few percent of it; a sign error in the swap
    reproduces the full gain. Rehearsed 2026-09-10 RED: the swap mask applied to one arm only."""
    rng = np.random.default_rng(1)
    e_ref = rng.normal(0, 60, 20_000) ** 2
    e_new = rng.normal(0, 50, 20_000) ** 2
    real = float(np.sqrt(e_ref.mean()) - np.sqrt(e_new.mean()))
    ctl = pa.sign_swap_control(e_new, e_ref, n_draws=200, seed=0)
    assert real > 5.0
    assert abs(ctl["mean_gain"]) < 0.05 * real
    assert ctl["p99_abs_gain"] < 0.25 * real and ctl["n_draws"] == 200

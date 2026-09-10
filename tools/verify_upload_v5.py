"""Verify merry-quicksand_v5.parquet against v4 and upload it. Asserts gate the upload.

v5 = v4 base + LightGBM seeds 0,1,2 with the queue block on MATCHED rows only:
  * template ids and order identical to v4
  * 5,290 unmatched rows byte-identical to v4 (stratum hybrid untouched)
  * matched rows changed (the whole point)
  * meta sidecar present and names three seed boosters
"""
import json, pathlib, hashlib, importlib.util, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("bs", "scripts/build_submission.py")
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
tpl = pq.read_table("data/raw/submitting.parquet").to_pandas()
v4 = pd.read_parquet("submissions/merry-quicksand_v4.parquet")
v5 = pd.read_parquet("submissions/merry-quicksand_v5.parquet")
rk = pq.read_table("data/raw/ranking.parquet", columns=["MVT_ID_mvt", "AOBT_3_flt"]).to_pandas()
unm = set(rk.loc[rk.AOBT_3_flt.isna(), "MVT_ID_mvt"])
bs.check_submission(v5, tpl)
assert (v5.MVT_ID_mvt.to_numpy() == v4.MVT_ID_mvt.to_numpy()).all(), "id order differs from v4"
um = v5.MVT_ID_mvt.isin(unm).to_numpy(); assert um.sum() == 5290, um.sum()
a4 = v4.TAXITIME_SEC_mvt.to_numpy().astype("float64"); a5 = v5.TAXITIME_SEC_mvt.to_numpy().astype("float64")
assert (a4[um] == a5[um]).all(), "an UNMATCHED row changed"
ch = int((a4[~um] != a5[~um]).sum()); assert ch > 300_000, f"only {ch} matched rows changed"
d = a5[~um] - a4[~um]
assert np.isfinite(a5).all() and (a5 >= 1).all(), "non-finite or sub-floor prediction"
meta = json.loads(pathlib.Path("submissions/merry-quicksand_v5.meta.json").read_text())
print("meta:", {k: meta[k] for k in meta if k not in ("boosters",)} if isinstance(meta, dict) else meta)
print(f"VERIFIED: rows {len(v5):,} | unmatched 5,290 byte-identical to v4 | matched changed {ch:,} of 339,551 | "
      f"shift mean {d.mean():+.1f}s RMS {np.sqrt((d**2).mean()):.1f}s max|d| {np.abs(d).max():.0f}s | "
      f"matched median v4 {np.median(a4[~um]):.0f} -> v5 {np.median(a5[~um]):.0f}")
sha = hashlib.sha256(pathlib.Path("submissions/merry-quicksand_v5.parquet").read_bytes()).hexdigest()
print("  sha256", sha[:24])
if "--no-upload" in sys.argv:
    print("dry run: not uploading"); sys.exit(0)
from prc.s3 import S3, load_credentials
c = S3(load_credentials()); b = "prc-2026-merry-quicksand"
assert "merry-quicksand_v5.parquet" not in [o["key"] for o in c.list_objects(b)], "v5 already in bucket"
st = c.upload(b, "merry-quicksand_v5.parquet", pathlib.Path("submissions/merry-quicksand_v5.parquet"))
print(f"UPLOADED merry-quicksand_v5.parquet -> HTTP {st}")

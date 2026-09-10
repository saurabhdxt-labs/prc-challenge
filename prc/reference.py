"""A reusable baseline reference: predictions plus everything needed to prove they still apply.

WHY THIS EXISTS

`plans/TOP10_NEXT_RUNS_2026_09_10.md` step 1: fit one corrected baseline, save it, and have every
later experiment reuse it when its configuration matches. On this machine a baseline arm is ~30 min
of refit plus ~30 min of prediction per seed, and `scripts/cond_experts.py` currently pays that
again on every run purely to have something to pair against.

WHY IT IS DANGEROUS, AND WHAT GUARDS IT

Reusing a stored prediction vector is exactly the shape of error this repository has already paid
for twice in twenty-four hours: a floored arm compared against an unfloored one, and a delta-scale
record read as taxi times. A silently mismatched reference would look like a model difference and
would be believed.

So a reference is only reused when EVERY one of these matches, and each is stored beside the
predictions rather than assumed:

  * the ordered feature list (order matters -- the design matrix is positional)
  * the LightGBM params, the seeds, and n_ref
  * the months loaded and the name of the split function
  * the encoder variant that produced the encoding columns
  * the SCALE the predictions are on ("taxi_time" or "delta"), named explicitly, because the two
    fold records in this repo disagree and reading one as the other produced a -109 s result
  * a checksum of the holdout rows' MVT_ID, in order -- so a reference cannot be applied to a
    different or reordered row set even when every other field agrees

Any mismatch raises with the offending field named. There is no "close enough" path and no
force-reuse flag: the whole point is that a wrong reuse is silent, so it must be impossible rather
than discouraged.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

SCALES = ("taxi_time", "delta")


def row_checksum(mvt_ids) -> str:
    """Order-sensitive fingerprint of the holdout rows a prediction vector belongs to."""
    a = np.asarray(mvt_ids, dtype="float64")
    if not np.isfinite(a).all():
        raise ValueError("non-finite MVT_ID in the holdout row set")
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).hexdigest()


def signature(meta: dict) -> str:
    """Stable hash of everything that must match for a reference to be reusable."""
    required = ("feats", "params", "seeds", "n_ref", "months", "split", "variant", "scale",
                "rows")
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"reference metadata is missing {missing}")
    if meta["scale"] not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {meta['scale']!r}")
    payload = json.dumps({k: meta[k] for k in required}, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def save(path, preds, meta: dict) -> pathlib.Path:
    """Write predictions + metadata. Returns the path actually written."""
    path = pathlib.Path(path)
    preds = np.asarray(preds, dtype="float64")
    if preds.ndim != 1:
        raise ValueError(f"predictions must be 1-D, got shape {preds.shape}")
    if not np.isfinite(preds).all():
        raise ValueError(f"{int((~np.isfinite(preds)).sum()):,} non-finite predictions")
    if len(preds) != int(meta.get("n_rows", len(preds))):
        raise ValueError("n_rows disagrees with the prediction length")
    meta = dict(meta, n_rows=len(preds), signature=signature(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, preds=preds, meta=json.dumps(meta, default=str))
    return path


def load(path, meta: dict, strict: bool = True):
    """Return (preds, stored_meta) iff the stored reference matches `meta` exactly.

    Raises with the FIRST offending field named. `strict=False` returns None instead of raising
    when the file is simply absent -- it never relaxes a mismatch.
    """
    path = pathlib.Path(path)
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"no baseline reference at {path}")
        return None
    with np.load(path, allow_pickle=False) as z:
        preds = np.asarray(z["preds"], dtype="float64")
        stored = json.loads(str(z["meta"]))
    for k in ("scale", "variant", "split", "months", "seeds", "n_ref", "feats", "params",
              "rows"):
        if stored.get(k) != meta.get(k):
            raise ValueError(
                f"stored baseline reference does not apply: {k!r} differs\n"
                f"  stored:   {str(stored.get(k))[:200]}\n"
                f"  required: {str(meta.get(k))[:200]}")
    want = meta.get("n_rows")
    if want is not None and len(preds) != int(want):
        raise ValueError(f"stored reference has {len(preds):,} predictions but the caller "
                         f"expects {int(want):,} rows")
    return preds, stored

# PRC Data Challenge 2026 — taxi-out prediction

Predicting taxi-out duration for 344,841 departures at ten European airports
(January and July 2026), scored by RMSE.

Team **merry-quicksand**.

## The method in one paragraph

The target is `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt − BLOCK_TIME_UTC_mvt`. Take-off
(`MVT_TIME_UTC_mvt`) is supplied exactly on every scored row; off-block
(`BLOCK_TIME_UTC_mvt`) is withheld. Predicting taxi-out is therefore identical to
predicting off-block. For 98.47% of rows the Network Manager supplies an independent
second measurement of off-block, `AOBT_3_flt`, so we do not model the duration — we model
the **disagreement between the two clocks**:

```
y = proxy − delta        proxy = MVT_TIME − AOBT_3   (known exactly)
                         delta = BLOCK_TIME − AOBT_3 (predicted)
```

A gradient-boosted tree cannot represent `y = proxy − f(x)`: it must approximate the
identity in a wide-range variable by binning it. It can represent `f(x)` directly. On an
identical fold, features and seed, this reparameterisation moved matched-row RMSE from
**285.24 to 252.66**. It is the single largest modelling decision in this repository.

The remaining 1.53% of rows have no Network Manager record at all — every `*_flt` column
is null together — and are handled by a separate estimator keyed on airport and the
schedule offset, because the reparameterisation above has no second clock to lean on.

## Reproducing

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # Python 3.11.15
python scripts/fetch_data.py             # writes data/raw/ + FETCH_MANIFEST.json
python -m pytest tests -q                # unit tests
python scripts/build_submission.py       # writes the submission parquet
```

`data/` is git-ignored: the challenge data is not ours to redistribute.
`scripts/fetch_data.py` verifies every object against its bucket ETag and deletes
anything that fails verification.

## Validation protocol

* Whole **months** are held out, never random rows. Primary fold: January + July 2025
  held out, trained on the other ten months; replication fold: February + August.
* Every target encoding is fitted **inside the training fold only**.
* Every reported RMSE carries a 95% bootstrap interval, and every A-vs-B claim uses a
  **paired** bootstrap of the difference. A gain whose interval spans zero is reported as
  not established.
* The harness was validated against the live leaderboard without spending a submission,
  by comparing naive predictors reproducible by any team: a constant scores 686.07 on our
  fold against 689.69 observed; airport-mean 659.9 vs 660.5–663.0; airport×runway 653.1
  vs 650.6; +stand 629.0 vs 625.3–627.6. Agreement is within ±5 seconds.
* No threshold, hyper-parameter, feature or fold design is ever selected by comparing
  leaderboard scores. See `plans/PREREG_taxiout_2026_09_08.md`, Amendment 3.

## What is in `plans/` and `reports/`

The pre-registration (frozen before the data was read, with four dated amendments) and
the measurement reports behind every claim: the column inventory, the airport clock
artifacts, the unmatched-stratum analysis, the harness calibration, the matched-row model,
an adversarial red-team review, and the Family-B investigation.

Negative results are recorded as prominently as positive ones — weather (+0.49 s),
queue and surface-state features (+2.72 s), out-of-fold target encoding (−0.96 s),
winsorised training targets (−30.3 s), absolute-error loss (−14.9 s) and a single
undivided model (−144 s) are all documented with their intervals.

## External data

None is used in this baseline. Should ground ADS-B be incorporated, the archives are
published by [ADSB.lol](https://www.adsb.lol/docs/open-data/historical/) under
**ODbL 1.0**; every release tag consumed and the full extraction procedure would be
documented here with the required attribution. ADSB.lol is not affiliated with the
competition, which is supported by the OpenSky Network.

## Licence

GNU General Public License v3.0 — see `LICENSE`.

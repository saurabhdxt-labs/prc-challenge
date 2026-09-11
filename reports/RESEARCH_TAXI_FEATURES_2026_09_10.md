# Taxi-out features: the literature, the operations, and what they say about our model (2026-09-10)

**Question.** Before we build any more features: what does the published literature and the operational
documentation (A-CDM, EUROCONTROL Network Manager, airport AIPs) say drives taxi-out time? Which
aircraft-data sources can we legally use? What does each of the ten airports do? And which features
should we build next, ranked, with a leakage verdict on each?

**Answer in one paragraph.** Most of the literature predicts taxi-out time from the **queue the aircraft
meets after pushback** (Idris et al. 2002: the take-off queue during taxi explains R² 0.59 at Boston,
aircraft type 0.01, airline/terminal 0.02). After the queue come **stand-to-runway geometry** and
**runway configuration**. Aircraft size and weather are consistently secondary. Our model already has
all of those families, and descriptive cuts of arm F's residuals find almost nothing left in them: a
level-fix ceiling of ≤ 100 MSE each for aircraft size, runway-queue state and busy periods (§g).

The finding that changes how we should read the model comes from the operational documents, and I
checked it against our data (§1). **At four of our ten airports, `AOBT_3` is not a measured off-block
time.** It is Network Manager's M3 model time. NM does not use the airport's off-block field: the
A-DPI's aobt field is "currently not used". NM derives off-block times as "TOT –
last-received-taxi-time", and the taxi time is a per-stand, per-runway planning parameter. In our data, MVT − AOBT_3 sits on its stand×runway
modal minute on **41% of rows, against 13% for the true taxi time**. **Within a stand×runway pair it
carries essentially no information about the true taxi time at LEBL, LEMD, EDDM and LTFM**
(correlations −0.07, −0.04, 0.09, 0.12). It carries real information at EHAM, LSZH, LFPG, EDDF and EGLL
(0.48–0.63). At four airports our "clock gap" features are therefore the airport's planned taxi time,
and the physical taxi must come from congestion features. That is exactly what the literature is
about.

The ranked list (§e) leads with a cheap "NM taxi-parameter" block that makes this regime explicit.
Its value is **unknown until A/B-tested**. The aircraft source I recommend is the **FAA Aircraft
Characteristics Database**: a US-government work, so public domain, and it covers 98.5% of our
holdout rows. ICAO Doc 8643 data is **not usable**. One tempting public dataset, PRU's 2026 monthly
taxi-out CSV, **is refused as leakage**.

Compute used: web reading (five research sub-agents plus direct reading), and nine read-only
descriptive passes over the 2025 caches, the 2026 ranking cache and arm F's fold-A predictions, all
under `OMP_NUM_THREADS=1 nice -n 19`. The largest (`resid_cuts.py`) peaked at 0.58 GB RSS. Each
finished in about a minute or less. No model was
fitted and no repository file other than this one was written. Scripts and outputs are in
`…/scratchpad/research_lit/` (§h).

**Tags used throughout.** **[V]** means I read it in the primary source at the URL given. **[U]** means
it comes from an abstract, a search snippet or a secondary source only. **[M]** means we measured it
here (descriptive only, scripts in §h). **[I]** means it is our inference.

---

## 1. The finding that reorganises the feature question: what `AOBT_3` is

### 1a. What the documents say

- **The challenge's own definition.** `AOBT_3_flt` is "Actual Off-Block Time (AOBT) for flown (M3)
  trajectory" [V, https://prc-data-challenge-2026.netlify.app/data.html]. M3 is NM's ACTual / CTFM
  profile: "EST … FTFM", "CAL … RTFM", "ACT … CTFM" [V, NM Flight Progress Messages 3.100 §2.2.2.3.1.4.23,
  https://www.eurocontrol.int/archive_download/all/node/11547]. The aobt field exists only "when modeltyp
  is ACT" [V, ibid. §…4.24].
- **NM ignores the airport's off-block value.** "The aobt- and aobd-fields are currently not used. They
  will be used for information sharing purposes." [V, DPI Implementation Guide 2.700 §4.7.3.3,
  https://www.eurocontrol.int/sites/default/files/2025-06/eurocontrol-dpi-impl-guide-2-700.pdf.pdf].
  What NM does take from the A-DPI:
  - the **event**: "The first A-DPI shall be sent at the off-block event/push-back clearance delivery"
    [V, §9.3];
  - a **TTOT** equal to "AOBT+EXOT" [V, §4.7.26];
  - a **taxi time** [V, §4.7.21].
- **NM back-computes off-block times from take-off times.** "(Derived OBT in DPI msg is TOT –
  last-received-taxi-time)" [V, §4.7.26 and §9.3]. This is stated for the ETFMS_OBT of the *filed*
  model; that the M3 off-block is handled the same way is our inference, tested in §1b.
- **The taxi time links NM's off-block and take-off.** TAXITIME is "The average taxiing time for the
  runway in use which was considered by ETFMS to derive the take-off times from the off-block times
  when calculating the last flight profile" [V, ATFCM Users Manual MAINT-2 glossary,
  https://www.eurocontrol.int/sites/default/files/2024-05/eurocontrol-atfcm-users-manual-maint-2-wef-20240618.pdf].
- **The taxi time is a table value, not a measurement.**
  - "The taxitime shall at least depend on gate/stand/parking position and departure runway … shall also
    include any time it takes to de-ice" [V, DPI Guide §4.7.21.1].
  - It is updated only "when the taxitime-field changes by 5min or more" [V, §4.7.21.2].
  - At non-CDM airports "the default taxi time parameter is specified for each runway at an aerodrome in
    the CACD … overridden by the taxi time provided by the AO in the flight plan" [V, ATFCM UM §8.1.10.1].
- **Actual take-off reaches NM by a separate message:** "the FSA message gives the Actual Take-Off Time at
  the moment of take-off" [V, Flight Progress Messages §2.2.2.2.3.2].
- **EUROCONTROL's own performance unit does not treat NM's off-block as a source.** Its taxi-out
  indicator takes AOBT from airports' APDF only [V, ATXOT methodology, Table 3,
  https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf; APDF spec ODI-11 checks
  airport AOBT against airline data within ±3 min for ≥ 95% of flights, and not against NM,
  https://www.eurocontrol.int/sites/default/files/2019-10/ao-data-flow-specs.pdf].

- **Independent corroboration.** EUROCONTROL's own ML study of NM take-off times finds that "high taxi
  times are typically over-estimated. In other words, when the planned taxi time is high, ETFMS tends
  to report a later take-off time than the one that will be executed in reality". In other words, NM
  carries a *planned* taxi time that can be wrong by minutes [V, Dalmau et al., SESAR Innovation Days
  2019, https://www.sesarju.eu/sites/default/files/documents/sid/2019/papers/SIDs_2019_paper_36.pdf].

**Inference [I].** For at least part of the traffic, NM's M3 off-block is *take-off minus a planned
taxi time*. When the FSA/departure message shifts the M3 profile to the actual take-off, the off-block
moves with it and the taxi time stays fixed. That is the only mechanism I found that makes MVT − AOBT_3
a per-stand×runway constant. No document states it in one sentence. §1b tests it.

### 1b. What our data says [M]

**Test 1 — within a (stand, runway) pair, is MVT − AOBT_3 related to the true taxi time?**

- Rows: Jan+Jul 2025 matched departures, pairs with ≥ 30 rows, both quantities < 1 h.
- Both variables are demeaned within the pair (`within.py`).

| airport | corr(proxy, y) within pair | slope of y on proxy | sd proxy (s) | sd y (s) | reading |
|---|---:|---:|---:|---:|---|
| LEBL | **−0.065** | −0.16 | 110 | 263 | proxy is a constant; no information |
| LEMD | **−0.044** | −0.09 | 111 | 224 | same |
| EDDM | **0.094** | 0.15 | 193 | 305 | same |
| LTFM | **0.117** | 0.26 | 146 | 319 | nearly none (LTFM sends no DPI, §d) |
| LIRF | 0.228 | 0.59 | 167 | 433 | weak |
| EGLL | 0.479 | 0.43 | 400 | 360 | informative |
| EDDF | 0.530 | 0.98 | 141 | 262 | informative |
| LFPG | 0.540 | 0.62 | 301 | 343 | informative |
| EHAM | 0.614 | 0.46 | 335 | 253 | informative |
| LSZH | 0.625 | 1.15 | 141 | 259 | informative |

If `AOBT_3` were a noisy stamp of the same off-block event, the correlation would be high everywhere,
as it is at EHAM and LSZH. At LEBL, LEMD, EDDM and LTFM it is zero.

**Test 2 — does the proxy pile up on one minute per pair?**

- The modal minute of round(proxy/60) is taken per (airport, stand, runway) on the ten training months.
- The share of holdout rows sitting on it is compared with the same construction on y, as a control
  (`synth.py`).

| | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| proxy at modal minute | .33 | **.76** | .09 | .21 | .52 | .54 | .39 | .35 | .47 | .56 | **.41** |
| y at modal minute (control) | .15 | .14 | .08 | .16 | .15 | .14 | .13 | .09 | .15 | .11 | .13 |
| 2026 scored rows at the 2025 mode | .30 | .77 | .10 | .19 | .52 | .54 | .43 | .33 | .45 | .56 | — |

Three things follow:
- The proxy is 3.2× more concentrated on one minute than the real taxi time overall (EDDM 5.4×). If
  the proxy were a noisy measurement of the real taxi, it would be *less* concentrated, not more.
- The signature is **stable into 2026** (`synth2026.py`). The modal minute agrees on 74% of rows and
  moves 0.59 min on average, which is consistent with VTT tables being refreshed.
- Arm F is not worse on these rows: MSE 41.7k on modal rows against 53.7k on the rest, and modal rows
  hold 33.6% of holdout SSE.

**Test 3 — consistency with an experiment already run.** In arm Y (RESULT 19,
`reports/lgbm_fold_queue_ytarget.json`), a model with target y and a model with target delta score
almost identically exactly where the proxy is uninformative:

| airport | y target | delta target |
|---|---:|---:|
| EDDM | 166.5 | 166.5 |
| LEMD | 174.9 | 174.7 |
| LEBL | 219.7 | 220.3 |
| LTFM | 242.9 | 243.6 |
| EHAM | 168.3 | 163.1 |

At EHAM, where the proxy carries information, the y target is clearly worse.

**Verdict on H-AOBT3 ("MVT − AOBT_3 is a per-stand×runway planned taxi time, not a measurement"):**
- **WORKING at LEBL, LEMD, EDDM and LTFM.** The evidence is the documents (§1a), the zero within-pair
  correlation, the 3.5–5.4× modal concentration, and arm Y's equivalence.
- **PARTIAL at LIRF, EDDF, LFPG and LSZH.** Modal share is 33–47% but the within-pair correlation is
  0.23–0.63, so this is a mixture of synthetic and event-like rows.
- **NOT WORKING at EHAM and EGLL.** There the proxy behaves like an event stamp.
- **Not covered:** the NM-internal rule that decides which regime a row falls in. It is not public
  (NM: "the actual computation remains our internal process"
  [V, https://www.eurocontrol.int/sites/default/files/2022-06/nm-b2b-webinar-202206-qanda.pdf]).

**Consequences for features:**

1. At the synthetic airports delta = BLOCK − (MVT − EXOT) ≈ EXOT − y, so **the model is predicting the
   physical taxi time there**, and the literature's congestion drivers apply in full.
2. Features anchored at AOBT_3 as "the pushback" are anchored at MVT − EXOT at those airports. That
   covers `q_apt_at_push`, `q_rwy_at_push`, `q_pushed_after_me`, `q_rwy_tko_in_taxi` (Idris's Q), the
   weather as-of anchor, and `n_push`.
3. The **deviation of the proxy from its stand×runway mode** is how NM *signals* an extended taxi. The
   DPI guide defines EXOT_gen_ext for general de-icing, EXOT_DI_rem for remote de-icing, and A-DPI
   updates for runway closures [V, DPI Guide §11.4]. That deviation is not currently a feature.

---

## (a) Literature table

**Read this first.** The studies do not measure the same interval, so their errors cannot be compared
directly.
- **Off-block → take-off (our definition):** Idris, Clewlow, Simaiakis, Diana, PRU, Wu.
- **Stop at line-up and pool arrivals with departures:** Wang et al.
- **Gate exit → runway entry:** Vakaet (EHAM).
- **To a zone near the runway entry:** Lordan (LEBL).
- **Random train/test splits, which flatter the result:** Wang, Wu, Çömez & İnan.
- **Know the take-off time, as we do:** none of them. Wang's "known only after take-off" feature group is
  the closest measure of that advantage: +0.09–0.18 R² and −0.4 to −1.0 min RMSE.
- Verification status: [V] = full text or slides read; [A] = abstract only; [U] = second-hand.

| paper (venue) | year | airports | winning model | top features, with magnitudes | error reached | URL | status |
|---|---|---|---|---|---|---|---|
| Idris, Clarke, Bhuva, Kang (ATC Quarterly 10(1)) | 2002 | BOS, 26,302 flights | queue model T(Q), one per runway config × airline; beats a 14-day running mean | **take-off queue Q during own taxi R² 0.59**; N at pushback R² 0.19; scheduled demand < 0.03; airline/terminal 0.02; **aircraft type 0.01**; arrivals < 0.02; config shifts the mean 15–23 min; ground stop 46.4 vs 19.3 min | MAE 4.56 vs 5.69 min (mean taxi 19.2); 65.6% vs 53.7% within ±5 min | https://pdfs.semanticscholar.org/cdcb/8f9380999147e162fbb4a67b9cfc889e8bc4.pdf | [V] preprint |
| Clewlow, Simaiakis, Balakrishnan (AIAA GNC) | 2010 | JFK, BOS | linear regression | **take-offs during own taxi R² 0.76 / 0.64**; **landings during own taxi R² 0.71 / 0.58**; T = 8.02 + 0.77·D + 0.52·A, R² 0.86 (≈ 219k flights); shared arr/dep runway +0.85 min | residual SE ≈ 7.6 min | https://www.mit.edu/~hamsa/pubs/ClewlowSimaiakisBalakrishnanGNC2010.pdf | [V] |
| Simaiakis & Balakrishnan (Transportation Science 50(1)) | 2016 | EWR | D(t)/E_k(t)/1 runway queue + unimpeded time per airline × visibility × config | "adjusted traffic" preferred to Q (Q has a selection bias); convex growth; saturation at 16 ≤ N ≤ 31; congested mean 52 min vs unimpeded 14 | per-flight RMSE 8.81 (2011), 9.88 (2010), 16.09 (2007) min: **degrades across years** | https://www.mit.edu/~hamsa/pubs/SimaiakisBalakrishnan_TS2014.pdf | [V] |
| Simaiakis et al. (ATM Seminar) | 2011 | BOS | pushback-rate control, not a predictor | take-off rate saturates past N* | 12–15 t fuel saved over 8 four-hour tests; 247 held flights waited 4.3 min | https://www.mit.edu/~hamsa/pubs/SimaiakisATMseminar2011.pdf | [A] |
| Balakrishna, Ganesan, Sherry (ATM Seminar) | 2009 | DTW, TPA, JFK | reinforcement learning | runway queue, departures and arrivals taxiing, **mean taxi of the previous two quarter-hours**, time of day | 15-min means: 89.9–97.1% within ±3 min (DTW/TPA); JFK 49–81% within ±5 | https://catsr.vse.gmu.edu/pubs/ATM2009_TaxiOut.pdf | [V] |
| Balakrishna et al. (TRR 2052; TR-C 18) | 2008/10 | DTW, DCA, TPA | RL / approximate DP | counts, recent average taxi, time of day | RMSE ≈ 2.9 min on ≈ 80% of flights, 15 min ahead | https://doi.org/10.3141/2052-07 | [A]/[U] |
| Lee, Malik, Jung et al. (NASA; AIAA 2015-2272 + 2016 workshop) | 2015/16 | CLT (2016: 246k real departures) | RF / kNN (2015); linear regression ≈ RF (2016) | runway (distance) and spot dominate; heavy longer, small shorter; **no seasonal effect**; rain flag no help; unimpeded = P10 per gate–spot–runway | 2015 RMSE: RF 1.15–1.21 min vs unimpeded baseline 6.3–7.1; 2016 live LR 89.8% within ±5 min | https://ntrs.nasa.gov/api/citations/20160004946/downloads/20160004946.pdf ; https://ntrs.nasa.gov/api/citations/20170000660/downloads/20170000660.pdf | [V] slides |
| Herrema, Curran, Visser, Huet, Lacote (JAIS 15(3)) | 2018 | **LFPG** (≈ 250k flights, 42 features) | regression tree (beats NN, RL, MLP) | top 10 of 42 include unimpeded time, congestion level, departures in the last 20 min; top-10 subset loses almost nothing | "average error of 1.6 min" RMSE per day | https://research.tudelft.nl/en/publications/taxi-out-time-prediction-model-at-charles-de-gaulle-airport/ | [A] (features [U]) |
| Ravizza, Atkin, Maathuis, Burke (JORS); Ravizza, Chen et al. (Applied Soft Computing 14) | 2013/14 | ARN, **LSZH** | TSK fuzzy rules beat MLR, SVR, M5 | distance, total turning angle, arr/dep flag, traffic counts | cited as 85.3% (ARN) / 86.1% (ZRH) within ±2 min | https://link.springer.com/article/10.1057/jors.2012.123 | [U] |
| Atkin, Burke, Ravizza (MAPSP) | 2011 | **EGLL** | regression + ground-movement model | departure-queue delays "relatively unpredictable for individual aircraft" | — | https://people.cs.nott.ac.uk/pszja/papers/MAPSP2011.pdf | [V] 3-page abstract |
| Lordan, Sallan, Valenzuela-Arroyo (JATM 56) | 2016 | **LEBL**, Jun–Aug 2013 | linear regression per stand × runway × hour | **stand × runway ("area") alone 82.9% of explained variance**; airline, aircraft, terminal, config each < 4%, confounded with stand or hour; weekday < 0.7% | taxi-out adj. R² 44.9%, MAE 2.16 min | https://upcommons.upc.edu/server/api/core/bitstreams/371d032d-5e63-49bc-b5c7-1366b36f7d52/content | [V] |
| Wang, Brownlee, Woodward, Weiszer, Mahfouf, Chen (TR-C 124) | 2021 | MAN, **LSZH**, HKG | random forest > GBM > linear > MLP | RF importance: dep/arr flag, **distance**, **departures taxiing at pushback**, turn angle, long straights, **mean speed of the last 5/10 departures**. Airport-specific: weight (MAN), temperature (MAN, ZRH), runway (ZRH). Budget airline ≈ 0.002; arrival counts < 0.01 | test RMSE 3.31 / 3.10 / 3.10 min; 89.9–92.5% within 5 min | https://eprints.whiterose.ac.uk/169934/1/Taxi_Time_Prediction%20as%20finally%20submitted.pdf | [V] |
| Vakaet (TU Delft MSc with Schiphol KDC) | 2021 | **EHAM**, 2018–19 | Google AutoML > hist. GBM > Schiphol's operational lookup | **operational tool = mean per gate × runway (36L split by taxiway W5/Y/Z) × wake category**; days < 3 °C removed (de-icing); recent-error correction did not help (183.6 → 184.8 s) | 30-min horizon RMSE: AutoML 152.9 s; GBM 158.5 s; lookup 161.8 s; mean 262.2 s | https://repository.tudelft.nl/file/File_a472e35f-4ef0-470c-bca8-645d786f3bcc | [V] |
| Lim, Tan, Lilith, Alam (SESAR Innovation Days) | 2021 | ATL (surface surveillance) | GNN ≈ GBM | planned route, **queue position**, local taxiway flow, carrier, runway | RMSE GNN 76.0 s, GBM 78.4 s, FAA method 123.1, EUROCONTROL method 199.2, naive 216.4 | https://www.sesarju.eu/sites/default/files/documents/sid/2021/papers/SIDs_2021_paper_78.pdf | [V] |
| Lim et al. (J. Air Transportation) | 2025 | n/a | GNN + transformer beats GBM by ≈ 8 s (≈ 10%) | taxi split into taxiing + queue time | — | https://doi.org/10.2514/1.D0369 | [A] |
| Diana (FAA; TR-E 119) | 2018 | SEA, hourly means | OLS/Ridge (2015), GBM (2016); "no algorithm fits better in all cases" | departures, demand, capacity used, runway configuration; IMC not significant | aggregate only | https://api.mdsoar.org/server/api/core/bitstreams/df12efc7-b5fd-4a95-9e8d-e441d76bcf03/content | [V] |
| **Dalmau, Ballerini et al. (EUROCONTROL; SESAR Innovation Days)** | 2019 | ECAC, NM data | LightGBM ≈ NN (take-off time 1 h before EOBT) | available turnaround time is the top feature; **ETFMS reports late take-offs when its planned taxi time is high** | MAE 10m10s → 7m08s (−30%) | https://www.sesarju.eu/sites/default/files/documents/sid/2019/papers/SIDs_2019_paper_36.pdf | [V] |
| EUROCONTROL PRU, additional taxi-out indicator, Ed. 01.00 | 2023 | EU | reference = P10 per stand × runway, rolling 12 months | aircraft class *not* used ("not having major impact"); de-iced-after-AOBT flights and > 120 min excluded | n/a | https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf | [V] |
| Wu et al. (Scientific Reports) | 2026 | SZX, 5,952 departures | stacked ensemble | SHAP: take-offs during taxi ≈ 18%, landings during taxi ≈ 18%, airline group 9%, 30-min mean taxi 8%, distance 6.2%, **aircraft type 2%** | RMSE 140.5 s (random CV) | https://pmc.ncbi.nlm.nih.gov/articles/PMC13022428/ | [V] via fetch summary |
| Holländer & Kallén (Chalmers MSc with Saab) | 2026 | one anonymised major airport (A-SMGCS) | RF > linear regression | SHAP (min): one runway 0.96, runway queue 0.70, taxiing departures 0.41; hour, wind direction, MTOW 0.18 each | test RMSE 3.1–3.3 min, R² 0.3–0.4 | https://odr.chalmers.se/server/api/core/bitstreams/eba50fa0-ffea-47e0-b36a-516bb7d2c46b/content | [V] |
| Çömez & İnan (IMIENS 3(3)) | 2024 | **LTFM** | RF (R² 0.868 without PCA) | stand, runway, distance (490–11,514 m), weather | normalised units, pooled arr/dep, random split: not comparable | https://imiens.org/index.php/imiens/article/download/58/37/468 | [V], low quality |
| Informer-RFR (Transportmetrica A); MT-DSTGAN (Aerospace 11); XGBoost additional taxi-out (Appl. Sci. 14) | 2022/24 | PEK, PEK, PVG | transformer+RF; graph attention; XGBoost | gate-cluster time series; spatio-temporal flow; stand-group × runway corridor flows | PEK: 96.6% within ±5 min | https://doi.org/10.1080/23249935.2022.2071353 ; https://doi.org/10.3390/aerospace11050371 ; https://doi.org/10.3390/app14219968 | [A] |
| Durubi (Aalto BSc) | 2025 | ADD (A-CDM VTT) | CatBoost ≈ HGB ≈ RF | static fields only (stand, aircraft type, hour): R² < 0.1 | RMSE 9–15 min | (thesis PDF, scratch `aalto.txt`) | [V], context only |

Second-hand only [U], from Wang et al.'s Table 1:
- Jordan 2010, DFW: regression, 98.3% within 3 min, good-weather days only.
- Srivastava 2011, JFK (MITRE): queue position, distance, previous-quarter mean, severe-weather flag.
- Lian 2018, PEK: SVR, ≈ 95% within 5 min.
- Yin 2018, PVG.

**What the literature agrees on** (numbers from the table above):

1. **The queue met after pushback dominates** when it is counted over the aircraft's own taxi window:
   Idris 0.59 vs 0.19; Clewlow 0.76 vs 0.52; Wang lifts ZRH R² from 0.67 to 0.85.
   - **Caution [I, and the literature agent's own note]:** a count over one's own
     off-block→take-off window grows with the window. Part of that R² is built in by construction, and
     it needs the true off-block, which is our unknown.
   - Our `q_rwy_tko_in_taxi` uses NM's off-block, so it does not leak. At the four synthetic airports,
     though, its window is a constant EXOT.
2. **Stand × runway is the largest static effect:** LEBL 82.9% of explained variance; PRU's own
   reference; Schiphol's operational lookup is within 9 s RMSE of gradient boosting.
3. **Aircraft type and airline are weak once stand is in:** Idris 0.01 / 0.02; Wu 2%; PRU drops
   aircraft class; Pushback 2023 2nd place ranks engine class near the bottom (§h, competitions).
4. **Weather acts through capacity loss and de-icing, not directly.** IMC is weak or insignificant
   (Idris, Diana). Freezing temperature matters at ZRH and MAN. Schiphol and PRU *remove* de-iced
   flights, so no study quantifies the de-icing tail that our RMSE has to absorb.
5. **Trees win or tie almost everywhere.** Graph and transformer models win by seconds, and only with
   route and surveillance data we do not have.
6. **Errors degrade across years** (Simaiakis: RMSE 8.8 → 16.1 min when the 2011 model is applied to
   2007). Validate across time, not on random splits. This is consistent with our fold being optimistic
   by 4.26 s (TRAIN_SERVE_AUDIT §4).
7. **Nobody in the literature has our information set.** Every study predicts taxi time without knowing
   the take-off, or at best knows it but not a planned off-block. Our problem is different: reconcile
   NM's M3 off-block with the airport's block clock, given the take-off. §1 shows that for four airports
   this becomes a physical taxi prediction again, and for five it is mostly clock reconciliation.

**Prior competitions** [V unless marked]:
- **PRC 2024** (take-off weight):
  - Winner: LightGBM with 50,000 trees, seed-averaged (1 seed 1,611.7 → 20 seeds 1,561.5); target
    rescaled to (TOW − EOW)/(MTOW − EOW) with weights to keep RMSE on kg
    (https://github.com/PRC-Data-Challenge-2024/team_likable_jelly).
  - 2nd (ITA): CatBoost dominated its ensemble (weight 0.865; ensemble gain over CatBoost 0.2%).
    Ablation: aircraft characteristics +2.6%, weather +1.8%
    (https://journals.open.tudelft.nl/joas/article/view/7963).
  - A separate final set (52,190 new rows) was added about a week before the deadline
    (https://prc-data-challenge.netlify.app/).
- **PRC 2025** (fuel burn):
  - Temporal split. The winner was "strongly informed by domain knowledge"
    (https://prc-data-challenge-2025.netlify.app/outcome.html).
  - The editorial notes teams that "ranked well on one month's data dropped or rose on the other"
    (https://journals.open.tudelft.nl/joas/article/view/8750).
- **NASA/DrivenData "Pushback to the Future"** (2023, MAE on minutes to pushback, 10 US airports):
  - All five finalists used GBDTs per airport. Winner MAE 10.67 min against a > 19 min benchmark.
  - The winner trained on the **residual to the operational ETD**, the same move as our delta target.
  - 2nd place SHAP: ETD feature ≫ flight number > carrier > **aircraft type (4th)**; engine class near
    the bottom. Queue-ahead and rate features were tried and rejected
    (https://drivendata.co/blog/airport-pushback-finalists ;
    https://github.com/drivendataorg/nasa-airport-pushback).
- **No earlier public airport taxi-out competition** was found.

---

## (b) Mechanisms of taxi-out, and what we can measure for each

The decomposition every queuing paper uses is taxi-out = unimpeded taxi + runway-queue delay +
ramp/taxiway congestion (Simaiakis & Balakrishnan, [V] `simaiakis_ts.txt`). In our target, a fourth
term has to be added: **the difference between the airport's off-block clock and NM's** (§1).

| # | mechanism | what the sources say | measurable proxy from our columns | already have? |
|---|---|---|---|---|
| 1 | **Which event "off-block" stamps** | A-CDM Milestone 15: "commenced push-back or taxi … The start-up request from the Electronic Flight Strip may be used as the off-block event" [V, A-CDM Spec SPEC-198, https://www.eurocontrol.int/sites/default/files/2025-01/eurocontrol-specification-for-acdm.pdf]. Airlines use parking-brake release, airports first movement under push, handlers "cleared for beginning of pushback"; "might in some instances represent minute(s)" [V, EUROCONTROL Sustainable Taxi Ops ConOps 2024 §4.1]. FRA uses push-back clearance as AOBT [V, FRA crew briefing 5.2]. APDF AOBT = "vacated the parking position", equivalent to ACARS OUT [V, APDF ODI-11]. | per-airport offset in delta (LTFM median +296 s; CLOCK_ARTIFACTS §4c); the BLOCK seconds fingerprint | yes: ADEP encodings, `aobt_sec`, `mvt_sec` |
| 2 | **NM's M3 off-block = take-off − planned taxi time** | §1a | stand×runway modal proxy; proxy − mode; at-mode flag; airport-hour share of extended proxies | **no**, feature #1 in §e |
| 3 | **Pushback and engine start** | For a twin-engine medium: ~4 min push + start, ~1 min to unhook the tug, 2 min start + 3 min thermal stabilisation; "Taxi out time cannot be less than the needed warm-up time of the engines" [V, STO ConOps]. A local "ASAT + 10" is allowed "for special cases such as four-engine aircraft on remote stands" [V, A-CDM Spec]. Open stands may taxi on own power [V, STO ConOps]. | stand type/prefix; engine count (FAA ACD) | stand encodings and `stand_pref`: yes. Engine count: no, level ceiling 28 MSE (§g) |
| 4 | **Runway departure queue** | Take-offs during taxi (Q) R² 0.59 vs N at pushback R² 0.19 at BOS [V, Idris et al. 2002]. D(t)/E_k(t)/1 runway server; EWR saturates at 16 ≤ N ≤ 31 [V, Simaiakis & Balakrishnan TS]. | Q = take-offs on own runway during taxi; gap to previous take-off; busy-period length | Q: yes (`q_rwy_tko_in_taxi`), **but anchored at AOBT_3**, which is synthetic at 4 airports. Gap and busy period: no, ceiling ≤ 97 MSE |
| 5 | **Surface congestion** | Departures on the way to the runway at pushback (NDepDep) ranks third after dep/arr and distance; arrival counts < 0.01 importance [V, Wang et al. 2021]. Arrival demand R² < 0.02 [V, Idris]. | pushed-not-airborne counts; arrivals taxiing | yes: `q_apt_at_push`, `q_arr_taxiing_at_push`, `arr_b30`, … |
| 6 | **Geometry / unimpeded time** | PRU reference = P10 per stand×runway, rolling 12 months, ≥ 10 flights ≤ P10, aircraft class deliberately *not* used ("not having major impact") [V, ATXOT doc §3.2–3.4]. NASA CLT: P10 per gate–spot–runway [V, Lee et al.]. Distance and turning angles [U, Ravizza et al. 2013/2014]. | P10 of y per stand×runway, hierarchical back-off; stand support count | partial: `te_ars`/`de_ars` (means, smoothed toward the *global* prior) |
| 7 | **Runway configuration** | Idris built 56 models = 7 configs × 8 airlines [V]. At EHAM the reference taxi is 13.4–14.2 min to 36L (Polderbaan) vs 7.1–8.0 to 24 [V, PRU airport dashboard, https://aiu-airport-dashboard.netlify.app/EHAM.html]. | RUNWAY, stand×runway, active-runway set | yes. The runway-config *regime* block tested negative (PIPELINE §3 row 4), consistent with EXOT already encoding it |
| 8 | **De-icing** | Remote de-icing "will be seen as part of the time between off-block and take-off"; on-stand de-icing happens before off-block [V, DPI Guide §11.4.1]. EXOT_DI_rem = taxi to pad + queue + de-icing + taxi to runway [V, §11.4]. PRU excludes "flights with de-icing after the off-block time" [V, ATXOT §3.4]. Per-airport practice in §d. | freezing METAR × remote-pad airport; proxy − mode (an extended EXOT); airport-hour extension share | weather: arm W, small. Proxy deviation: no |
| 9 | **ATFM regulation and remote hold** | Slot window −5/+10 min around CTOT [V, DPI §4.7.26]. Push & hold for ATFM delay > 30–45 min, A-DPI postponed [V, §11.3, §9.3(6)]. LTFM Remote Holding Area for "CTOT waiting time of between 20 and 60 minutes" [V, DHMİ AIC A07/24, https://dhmi.gov.tr/AIPDocuments/LT_Circ_2024_A_07_en.pdf]. FRA remote holding needs TOBT–TSAT ≥ 15 min [V, FRA briefing 5.2]. | AOBT_3 − EOBT_1, LOBT/IOBT deltas; stand re-occupation before AOBT_3 | yes (`aobt_eobt`, `prev_arr_gap`; `n_sch_aobt` tested, +350 closed). §g: level already right |
| 10 | **Pre-departure sequencing** | TSAT = TTOT − EXOT: most waiting is absorbed at the stand, before AOBT [V, A-CDM Spec §3.5]. PRU additional taxi-out at our airports is only 2.5–9 min/dep [V, §d]. | none needed | n/a. It explains why queue features have less headroom than in US studies |
| 11 | **Taxi speed by size** | Speed falls with curve radius: 16 km/h at 15 m, 32 km/h at 60 m, 48 km/h at 135 m (Table 1-3). In the curved-taxiway separation example the allowable velocity is 25.4 km/h for code letter E and 27.7 km/h for F (Table 1-6, §1.2.24). Code letters are set by wingspan, e.g. F = 65 m up to 80 m [V, ICAO Doc 9157 Part 2, 5th ed. 2020, BAZL copy https://www.bazl.admin.ch/dam/de/sd-web/h7SYVJZ5kRTq/9157_p2_cons_en.pdf]. | code letter from wingspan | no; ceiling 32 MSE |
| 12 | **Minute resolution of AOBT_3** | AOBT_3 is on the 60 s grid on 97.7% of rows [M, DATA_DICTIONARY §2.29] | none. Irreducible ≈ U(0,60) ⇒ ~300 s² ≈ **~300 MSE floor** on event-stamped rows [I] | n/a |

---

## (c) Aircraft data sources and licence verdicts

**The licence bar.**
- The prize requires the code under GPLv3 (DC2026 eligibility page).
- "All used external datasets are openly accessible/usable and documented" and "All additional datasets
  used are openly available under an open source license" [V, https://prc-data-challenge-2026.netlify.app/eligibility.html].
- No OpenSky data (owner's rule).
- A non-commercial clause fails the bar: the FSF says such licences "do not qualify as free".

| # | source | licence (quoted) | fields / coverage | verdict |
|---|---|---|---|---|
| 1 | **FAA Aircraft Characteristics Database**, Oct 2024 edition (https://www.faa.gov/airports/engineering/aircraft_char_database; the xlsx is in scratch) | US-government work: "Copyright protection under this title is not available for any work of the United States Government" [V, 17 U.S.C. §105]. The page states no other terms [V] | **388 types keyed by ICAO designator** [V]: engine count and class, approach category (AAC), ADG, TDG, wingspan (± winglets), length, tail height, wheelbase, cockpit-to-main-gear, main-gear width, MTOW, MALW, ICAO WTC, US RECAT/CWT, parking area. **Coverage of our fold-A holdout: 98.54% of rows [M]**. Missing: CRJX (4,867 rows), then GA7C, FA6X, M600, DA62, … (≤ 25 rows each) | **USABLE**. Recommended. Credit FAA as good practice. Public-domain status is guaranteed by US law. Outside the US it is normally treated as free; not legally verified here |
| 2 | FAA Order JO 7360.1K, Aircraft Type Designators (June 2025; https://www.faa.gov/documentLibrary/media/Order/FAA_Order_JO_7360.1K_Aircraft_Type_Designators.pdf) | US-government work (same statute) | ≈ 2,650 types: class, engine count and type ("2J/H"), ICAO WTC, CWT | **USABLE**. A public-domain stand-in for Doc 8643; use it to fill engine fields for the ACD's gaps (e.g. CRJX) |
| 3 | OpenAP (TU Delft; https://github.com/TUDelft-CNS-ATM/openap) | repository LGPL-3.0; `openap/data/LICENSE` = **GNU GPL v3** [V per agent]; "Aircraft data: Collected from open literature" | 37 jet types: MTOW, MLW, OEW, length, span, wing area, engines | **USABLE** (keep the notice). Use only `aircraft/*.yml`. Avoid the WRAP kinematic data, which may be OpenSky-derived (not confirmed) |
| 4 | Wikipedia | CC BY-SA 4.0, "one-way compatible with the GNU GPL version 3" (FSF) | infobox scraping, error-prone | USABLE WITH ATTRIBUTION; low value |
| 5 | Wikidata | CC0 | **no ICAO type-designator property exists** [V per agent, SPARQL] | licence fine; **not practical** |
| 6 | EASA type-certificate data sheets | "Reproduction is authorised, provided the source is acknowledged" | one PDF per certificate | USABLE WITH ATTRIBUTION; impractical |
| 7 | **ICAO Doc 8643** (website and API) | site terms: "personal, non-commercial use, without any right to resell or redistribute them or to compile or create derivative works therefrom" [V, https://www.icao.int/terms-and-conditions]; API terms: "solely for the purpose of non-commercial use" | designator, engines, WTC; no dimensions | **NOT USABLE** |
| 8 | EUROCONTROL Aircraft Performance Database | "All data presented is only indicative … Copyright permission must be sought from EUROCONTROL" | 398 types incl. RECAT-EU; known errors (A220 listed Lower Medium) | **NOT USABLE** without permission |
| 9 | BADA | the licensee must not "copy, reproduce or disclose BADA … nor display BADA on any public … web site" | performance models | **NOT USABLE** |
| 10 | RECAT-EU 2.0 and the pair-wise separation documents (EUROCONTROL) | "provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes" | category rules and a 103-type list | **do not copy the list**. The *rules* are facts: recompute categories from ACD values (the agent reproduced 96/96 of EUROCONTROL's listed types) |
| 11 | tar1090-db, Mictronics readsb db, doc8643.com | no licence, or derived from ICAO | designator tables | **NOT USABLE / UNCLEAR, so avoid** |
| 12 | Airbus / Boeing airport-planning manuals | "must not be reproduced … without permission" | planning data | **NOT USABLE** |
| 13 | EUROCONTROL CODA taxi-time planning values by wake category (https://ansperformance.eu/reference/dataset/planning-taxi-times/) | EUROCONTROL portal non-commercial notice; not verified for the file | airport × WTC taxi-out statistics | do not commit; label-derived for the seasons covered. Local sanity check on 2025 seasons only |

**Recommendation.**
- Build one committed table `aircraft_types.csv` from the **FAA ACD**, with **FAA 7360.1K** filling
  engine fields and **OpenAP** filling dimensions for jets the ACD lacks.
- Compute three columns rather than copying them:
  - **ICAO code letter** from wingspan: A < 15 m, B 15–<24, C 24–<36, D 36–<52, E 52–<65, F 65–<80
    [V, ICAO Doc 9157 Pt 2 Table 1-1 extract];
  - **approach category** from approach speed (A < 91 kt … E ≥ 166, 14 CFR 97.3);
  - **RECAT-EU category** from MTOW and span (RECAT-EU 2.0 Figure 6).

**How size plausibly acts on taxi-out, and how much is left [M].**
- **Raw gaps are real.** CODA Heavy − Medium mean taxi-out, Summer 2025 [V per agent]: LIRF +9.5 min,
  EGLL +8.1, EDDF +4.7, LSZH +4.1, LFPG +4.2, EHAM +3.8, EDDM +3.0, LEMD +2.6, LTFM +1.8. In our holdout,
  LIRF H vs M mean y is 1,551 vs 1,144 s.
- **They flow through stand and runway.** Widebody stands and long-haul runways; the PRU says class has
  no "major impact" once stand × runway is fixed [V].
- **Arm F already absorbs them.** Its residual bias at LIRF is H −12 s vs M +4 s. The level-fix ceiling
  of airport × wake beyond airport bias is **13 MSE**. Code letter, engines and MTOW each have
  ≤ 32 MSE (§g).
- **Mechanisms that do exist:**
  - code-letter taxiway restrictions: EGLL "Taxiway Yankee … restricted to aircraft with a maximum size
    of Code C", A380 on charted routes only [V per agent, NATS AIP EGLL AD 2.20];
  - lower turn speeds for codes E/F (Doc 9157, §b row 11);
  - longer start-up for 4-engine aircraft on remote stands (A-CDM ASAT + 10, §b row 3);
  - wake separation paid by the follower: RECAT-EU 2.0 Table 4, e.g. 180 s for F behind A, 120 s for
    D behind B [V per agent].
- **Verdict:** a documentation-grade feature block, not a score lever. Its one score use is back-off
  for rare or unseen types, which are 0.1% of rows.

---

## (d) The ten airports

**Keys and rules for this table.**
- Taxi-out numbers are **2025 only**, from the PRU dataset (APDF-based, P10 stand × runway reference)
  [V: `txo_2025.csv` = https://www.eurocontrol.int/performance/data/download/csv/taxi_out_additional_time_2025.csv,
  and the PRU airport dashboards https://aiu-airport-dashboard.netlify.app/<ICAO>.html].
- The dashboards' **2026 columns are refused**: they aggregate the scored months' hidden off-block
  times (§f1). Nothing from them is shown here.
- "A-CDM class" comes from NM IN/25-005 [V].
- "proxy informativeness" is §1b's within-pair corr(proxy, y) [M].
- Other sources, by bracket key:
  - [AIP] national AIP AD 2: NATS UK, SIA France, ENAIRE Spain, DHMİ Turkey.
  - [DIP] the airport's de-icing procedure document.
  - [W] Wikipedia, a last resort.
  - [DASH] the PRU dashboard's 2025 runway-configuration shares.
  - Full URLs are listed below the table.

| airport | runways; 2025 departure runway shares (our data, Jan+Jul) | typical configuration & drivers | PRU 2025 ref / add (min/dep); add Jan → Jul | long-taxi departure runway (2025 ref min) | de-icing location | A-CDM class; since | proxy informativeness [M] | 2025–26 changes that move taxi geometry |
|---|---|---|---|---|---|---|---|---|
| **EHAM** | 6 runways [W]; dep 24 37.5%, 18L 25.5%, 36L 22.0%, 36C 11.2% | preferential runway system: Polderbaan 18R/36L and Kaagbaan 06/24 least noise; LVNL picks on weather, traffic, neighbour agreements [V, Schiphol news]. 2025: arr 18R/dep 18L+24 19%, arr 06+36R/dep 36L 15% [DASH] | 9.76 / 3.25; 4.30 → 3.04 | **36L 14.1** vs 24 7.9, 18L 9.0 [DASH] | gate and remote ("both gate and remote de-icing") [V, Schiphol A-CDM manual]; KLM runs 25 de-icing trucks [V]; "4 stations, 25 platforms" [U] | CDM; 16 May 2018 [V, EUROCONTROL news] | **0.61** (event-like) | Polderbaan maintenance May 2026 (outside Jan/Jul); snow and de-icing-fluid crisis 3–12 Jan 2026 [V, KLM news] |
| **EGLL** | 09L/27R 3,901 m, 09R/27L 3,658 m [AIP]; dep 27R 42.7%, 27L 41.6%, 09R 15.6% (2026 scored: 09R 42.2%) | segregated mode; 27 preferred if tailwind ≤ 5 kt and dry [AIP]; runway alternation at 15:00; easterly: all departures 09R [V, Heathrow] | 16.23 / 6.50; 5.89 → 7.41 | all long: 27R 16.7, 27L 16.2, 09R 15.8 | **on stand** by default; 4 remote pads (JEDI Delta/South, VADER North/South) [V, Heathrow DSP doc]; TSAT driven by the end of de-icing [V] | CDM; date [U] (2012) | 0.48 | 09L/27R night resurfacing Jun–Oct 2026, 23:00–06:00 [V, EG SUP 047/2026]; **easterly share much higher in the 2026 file** |
| **LFPG** | 4 runways [AIP]; dep 26R 39.8%, 27L 19.4%, 08L 19.0%, 09R 12.2%, 27R 8.8% | inner runways (08L/26R, 09R/27L) for departures, outer for arrivals; east/west [AIP] | 11.80 / 4.32; 3.99 → 4.75 | 09L 13.8, 09R 13.1, 27R 13.1 vs 26R 11.3 | **six remote de-icing areas** at runway thresholds + Romeo/Juliett, 15 Oct–15 May [AIP] | ANI; Nov 2010 [AIP] | 0.54 | **09R/27L closed 7 Aug–8 Dec 2025** [V, SUP 120/2025]; 09L/27R closed spring 2025 [U]: training-year geometry is atypical |
| **EDDF** | 07C/25C, 07R/25L 4,000 m; 18 take-off only; 07L/25R landing only [W]; dep 18 64.2%, 25C 26.8%, 07C 8.7% | west preferred for noise; 2025 east 47% [DASH]; night ban 23–05 [V, Fraport] | 10.68 / 3.54; 3.15 → 3.75 | 18 11.4 vs 25C 9.3 | on stand or remote pads DP1–DP4 (DP West only for RWY 18) [V, Fraport DIP 2024-25]; ≈ 50/50 [U] | ANI; 23 Feb 2011 [V, Munich A-CDM page] | 0.53 | 07C/25C closed 8–24 Mar 2026 [V]; **Terminal 3 opened 23 Apr 2026, 57 airlines moved by 9 Jun, T2 closed** [V, Fraport]: July 2026 has **12.9% of EDDF rows on stand×runway pairs with ≤ 50 rows in 2025, against 1.4% in January [M]** |
| **EDDM** | 08L/26R, 08R/26L 4,000 m [W]; dep 26L 42.2%, 26R 30.3%, 08R 14.7%, 08L 12.8%; mixed mode (92% of departures share a runway with landings [M]) | 2025: both 26 56%, both 08 39% [DASH] | 9.54 / 3.39; 2.45 → 4.08 | 08R 11.1 vs 26L 8.5 | **all jet de-icing on remote areas at the runway heads** (DA1–3, DA13–15); opposite-end areas "require an increase of taxi times" [V, Munich de-icing plan 2025-26] | CDM; 7 Jun 2007 (first in Europe) [V] | **0.09** (synthetic) | T1 non-Schengen pier opened 21 Apr 2026 [V]: **7.8% of EDDM July 2026 rows are on stands never seen in 2025 [M]** |
| **LEMD** | 4 runways; 32R/32L/18L/18R not for take-off [AIP]; dep 36L 38.4%, 36R 36.2%, 14L 15.1%, 14R 10.3% | north config preferred (dep 36L/36R); south (dep 14L/14R) at ≥ 10 kt tailwind or 20 kt crosswind; nights dep 36L [AIP]; 2025 north 77% [DASH] | 13.16 / 3.62; 3.14 → 4.07 | 36R 14.3, 14L 14.3 vs 36L 11.9 | two de-icing areas near THR 36L/36R, requested via CDM [AIP] | ANI; July 2014 [V, Aena] | **−0.04** (synthetic) | 18R/36L works from Sep 2026 (outside the scored months) [V] |
| **LEBL** | 06L/24R 3,352, 06R/24L 2,660, 02/20 2,528 m [AIP]; dep 24L 76.9%, 06R 21.3% | day west: arr 24R/dep 24L; east: arr 06L/dep 06R; **night 23–07: arr 02, dep 06R** [AIP] | 12.02 / 3.82; 2.82 → 4.52 | 24R 12.8 (long-aircraft only) | de-icing area on Ramp-17 stands up to 52 m span; code E+ on own stand [AIP] | CDM; Oct 2015 [V, Aena press] | **−0.07** (synthetic) | 06R/24L works Nov 2026 (outside) [V] |
| **LIRF** | 16L/34R, 16R/34L 3,902 m; 07/25 [U, conflicting lengths]; dep 25 85.8%, 16R 9.2%, 34L 4.6% | 25 is the main departure runway; arrivals 16L/16R or 34L/34R [U, VATSIM briefing]; 2025 arr 16L+16R/dep 25 58% [DASH] | 12.31 / **6.74**; 5.09 → **9.05** (highest) | **16R 16.0, 34L 15.2** vs 25 12.0 | **not found** (the Italian AIP needs a login) | ANI; trial Oct 2012, platform 2014 [V, airport-technology] | 0.23 (weak) | not found |
| **LTFM** | 6 runways incl. new 09 (departures only) [AIP AMDT 08/26]; dep 36 44.4%, 35L 30.2%, 18 13.1%, 17R 8.6% | preferential system, tailwind ≤ 10 kt; the departure runway on parallels is chosen by exit fix [AIP] | 12.14 / 4.80; 4.39 → 6.18 | 17R 16.3, 34L 16.2 vs 36 11.2 | five dedicated de-icing aprons (De-Icing 1–5) [AIP] | **not on IN/25-005 or the EUROCONTROL A-CDM list** (local A-CDM only) [V]. **Remote Holding Area for CTOT waits of 20–60 min since 31 Oct 2024** [V, AIC A07/24] | **0.12** (synthetic; NM taxi from CACD / flight plan) | triple independent runway ops from 17 Apr 2025 [V]; runway 09 planned Aug 2026 (outside) |
| **LSZH** | 10/28 2,500, 14/32 3,300, 16/34 3,700 m, all intersecting [W, V ZRH report]; dep 28 59.5%, 32 29.2%, 16 9.4% | noise-driven concepts by time of day: North 07–21 (dep 28/16), East 21–23:30 (dep 32/34), South early morning (dep 32/34); Bise dep 10 [V, ZRH operating concepts] | 8.61 / 3.30; 3.01 → 3.57 | 16 12.0, 34 11.7 vs 28 8.1 | **two remote de-icing pads "as main infrastructure"** + on stand; de-icing time built into TSAT [V, ZRH winter ops 2025; de-icing flyer] | CDM; partial May 2012, full Aug 2013 [V, ZRH A-CDM report] | 0.63 (event-like) | none found for 2025–26 |

**What the table says for features [I]:**
1. **The four synthetic airports are not a class of "non-CDM" airports.** Three of them are CDM or ANI
   airports. What they share is that NM's M3 off-block does not follow the event. The regime is an
   empirical, per-airport property (§1b), and it is stable into 2026.
2. **De-icing time is inside taxi-out** at EDDM (all remote), LSZH, LFPG, LEMD, LTFM and partly EDDF
   and EHAM. At EGLL, and for code E+ at LEBL, it is mostly on stand, i.e. before AOBT.
   - EDDM combines remote de-icing, a synthetic proxy and the winter delta < −600 tail
     (TRAIN_SERVE_AUDIT §5). Rank 1–2 features target exactly this.
3. **July 2026 has new stands at EDDF (T3) and EDDM (T1 pier)** that no 2025 row saw. This is the
   concrete case for §e rank 4.
   - Arrivals' taxi-in at those stands *is* in the scored file (arrival labels are not blanked), so a
     stand-level median taxi-in is a legitimate geometry proxy for new stands.
   - On the 2025 holdout it adds nothing at level: 32.7 MSE ceiling overall, 9.5 on rare pairs
     (`stand_taxiin.py`). Its value can only show on genuinely new stands.
4. **LIRF's July additional taxi-out (9.05 min/dep in 2025) is the highest of the ten**, and its
   non-25 departure runways carry references of 15–16 min. This agrees with LIRF being the weakest
   lane (PIPELINE §1b). LIRF remote hold under ATFM (§g: 36.5% held at 10–30 min delay) is the
   documented summer mechanism.

**Airport-table sources** [V unless marked].
- EHAM: https://news.schiphol.com/why-do-i-always-fly-from-de-polderbaan/ ; https://www.eurocontrol.int/news/amsterdam-schiphol-airport-adopts-cdm ; https://news.klm.com/impact-of-winter-weather-on-klm-flights/ ; Schiphol A-CDM manual https://assets.ctfassets.net/biom0eqyyi6b/7ERl8iHeLELDtgFsnK0mGi/474b9801a07e239cb41adc5f8b2fe8d2/A-CDM_Manual_Schiphol_Airport_v1.0.pdf
- EGLL: https://www.aurora.nats.co.uk/htmlAIP/Publications/2026-04-16-AIRAC/html/eAIP/EG-AD-2.EGLL-en-GB.html ; https://www.heathrow.com/company/local-community/noise/operations/runway-alternation ; https://www.heathrow.com/content/dam/heathrow/web/common/documents/company/team-heathrow/airside/winter-operations/Heathrow-AOP-and-DSPs.pdf ; https://nats-uk.ead-it.com/cms-nats/opencms/en/Publications/aip-supplements/EG_Sup_2026_047_en.pdf
- LFPG: https://www.sia.aviation-civile.gouv.fr/media/dvd/eAIP_03_SEP_2026/FRANCE/AIRAC-2026-09-03/html/eAIP/FR-AD-2.LFPG-fr-FR.html ; https://www.sia.aviation-civile.gouv.fr/media/store/documents/file/l/f/lf_sup_a_2025_120_fr.pdf
- EDDF: https://www.fraport.com/en/sustainability/dialog-with-neighbors/noise-and-air/flight-operations/runway-system-and-operating-hours.html ; Fraport DIP 2024-25 https://cdm.frankfurt-airport.com/content/dam/fraport-company-cdm/documents/binary/documents/deicing-procedure/EN-DIP%202024-2025.pdf/_jcr_content/renditions/original./EN-DIP%202024-2025.pdf ; https://www.fraport.com/de/newsroom/pressemitteilungen/2026/q1/sanierung-der-centerbahn-am-flughafen-frankfurt.html ; https://www.fraport.com/de/newsroom/pressemitteilungen/2026/q2/terminal-2-operativ-ausser-betrieb.html
- EDDM: https://www.munich-airport.de/_b/0000000000000035898332bb68e77735/deicing-plan-2025-2026.pdf ; https://www.munich-airport.com/airport-cdm/en/participating-airports ; https://www.passengerterminaltoday.com/news/operations-news/munich-airport-opens-terminal-1-pier-to-passengers.html
- LEMD/LEBL: https://aip.enaire.es/AIP/contenido_AIP/AD/AD2/LEMD/LE_AD_2_LEMD_en.pdf ; https://aip.enaire.es/AIP/contenido_AIP/AD/AD2/LEBL/LE_AD_2_LEBL_en.pdf ; Aena A-CDM press releases (2014, 2015)
- LIRF: https://www.airport-technology.com/news/newsrome-fiumicino-airport-introduces-collaborative-decision-making-platform-4190532/ ; runway rules [U] https://cdn.vatita.net/Pilot%20briefings/Briefing_LIRF_v1_8.pdf
- LTFM: https://www.dhmi.gov.tr/AIPDocuments/LT_AD_2_LTFM_en.pdf ; https://dhmi.gov.tr/AIPDocuments/LT_Circ_2024_A_07_en.pdf ; https://www.airport-technology.com/news/iga-triple-runway-operations-system/
- LSZH: https://www.flughafen-zuerich.ch/en/company/responsibility/noise-and-sound-insulation/flight-operations/operating-concepts ; https://media.flughafen-zuerich.ch/-/jssmedia/airport/portal/dokumente/business/airlines-and-handling/flight-operations/winter-operations/winter_operations_2025.pdf ; https://www.flughafen-zuerich.ch/-/jssmedia/airport/portal/dokumente/das-unternehmen/politics-and-responsibility/environmental-protection/technische-berichte/2015-10_a-cdm_env-benefits_zrh.pdf
- A-CDM list: https://www.nm.eurocontrol.int/STATIC/docs/pdf/IN-25-005.pdf ; https://www.eurocontrol.int/concept/airport-collaborative-decision-making
- Runway counts from Wikipedia where marked [W]: https://en.wikipedia.org/wiki/Amsterdam_Airport_Schiphol , /Frankfurt_Airport , /Munich_Airport , /Zurich_Airport

---

## (e) Ranked candidate features

**How the list is ranked.** Expected board MSE × confidence ÷ cost.

- "Expected value" is honest: a number only where §g gives a ceiling, otherwise **unknown**. Every
  ceiling in §g is an in-sample level-fix upper bound, not a gain.
- **Leakage rule applied.** Refuse anything that reads the row's own BLOCK or TAXITIME, anything built
  from other rows' labels in the scored months, and any external information about the scored period
  that is **derived from the hidden labels** (published aggregates of actual off-block or taxi-out
  times, §f1). Exogenous observations of the scored period, such as METAR, are not label-derived. The
  project already uses them in arm W under an as-of rule.
- Other rows' *serve-time* columns in the scored file (their MVT, AOBT_3, stand, arrivals' block times)
  are allowed. This is the same transductive rule Amendment 19.1 already uses.
- Column names refer to `stand_ab.py`: `proxy` = MVT − AOBT_3, `ars` = airport|stand|runway.

| rank | feature (definition in our columns) | mechanism | support | have it? | expected value | leakage |
|---|---|---|---|---|---|---|
| **1** | **NM taxi-parameter block**. (i) `exot_mode` = modal minute of round(proxy/60)·60 per `ars`, computed on the training fold, with back-off to (airport, `stand_pref`, runway) and then (airport, runway) below 20 rows. (ii) `px_dev` = proxy − `exot_mode`. (iii) `px_at_mode` = \|`px_dev`\| ≤ 30 s. (iv) `ars_mode_share` = the pair's training share at its mode, i.e. how "synthetic" the pair is | at synthetic rows delta = EXOT − y, so the model must predict physical taxi. `px_dev` > 0 is an extended EXOT: de-icing (EXOT_DI_rem, EXOT_gen_ext) or a runway closure (§b rows 2, 8) | [V] DPI Guide §4.7.3, §4.7.21, §4.7.26, §9.3, §11.4; ATFCM UM glossary; [M] §1b | **no**. `dep_dur`, `rwy_dur` and `q_rwy_ambient_proxy` are rolling medians of proxy per airport/runway, not per stand; `te_ars` and `de_ars` are target means | **unknown**. Largest *mechanism* found; the regime covers 41% of rows and is stable into 2026. Trees may already approximate it from proxy × encodings, so only an A/B settles it. One cheap arm | none: reads proxy (AOBT_3, MVT) and training-fold modes only |
| **2** | **Airport-hour declared-extension witness**. Over the airport's matched departures in the MVT hour: median `px_dev`, and the share with `px_dev` ≥ +300 s. Available to **unmatched** rows too, taken from their hour's matched rows | "General de-icing": airports "extend the taxi-times with an average value for all departures within the specified period" [V, DPI §11.4.1.1(3)]. Runway closures are pushed by updated A-DPI taxi times [V, §11.4.2] | [V] as cited; EHAM 2–8 Jan 2026 is this regime (PIPELINE §1d) | partial: `dep_dur` (rolling median proxy, not stand-normalised); the unmatched lane's `hprox_med` (hour median proxy) | unknown. January-specific (EDDM, LSZH, EHAM, EDDF). For the unmatched lane it is a *normalised* version of the witness E3R already uses, so it could be added to E3R's next registration | none (other rows' proxies, same file) |
| 3 | **Re-anchored runway queue (Idris Q, two-stage)**: Q̂ = take-offs on the own runway in [MVT − ŷ₁, MVT); pushed-not-airborne at MVT − ŷ₁. ŷ₁ is an **out-of-fold** first-stage taxi prediction | Q during the actual taxi is the literature's strongest single driver. Ours is anchored at AOBT_3, which at 4 airports is MVT − EXOT, i.e. a fixed-length window | [V] Idris 2002 (R² 0.59 vs 0.19 for N at pushback); Simaiakis & Balakrishnan | partial: `q_rwy_tko_in_taxi` and `q_apt_at_push` on the AOBT_3 anchor | unknown. Headroom limited: Idris-N bins show a 38 MSE level ceiling (§g), but the anchor error is concentrated at the 4 synthetic airports | **medium**: ŷ₁ must be out-of-fold by month for training rows; at serve time any fitted ŷ₁ is fine |
| 4 | **Hierarchical unimpeded / geometry prior**: PRU-style P10 of y per `ars` on training months (≥ 10 flights ≤ P10), backed off to (airport, `stand_pref`, runway) and (airport, runway); log support n(`ars`); for stands unseen in 2025, the stand's median **arrival taxi-in** from the same file (arrival labels are inputs, not blanked) | new or rare stand×runway pairs fall back to the *global* prior under the current smoothing (k = 50). 2026 adds new stands and runway mixes: EDDF Terminal 3 (Apr 2026) and the EDDM T1 pier (Apr 2026). **July 2026 rows on pairs with ≤ 50 rows in 2025: EDDF 12.9% (Jan 1.4%), EDDM 9.6%, LFPG 10.2%, LEMD 9.0% [M]**. EGLL 09R share 15.6% → 42.2% | [V] ATXOT methodology; [V] Lee et al. (NASA CLT) unimpeded = P10 per gate/spot/runway; [V] Lordan (LEBL) stand × runway dominates; [V] Fraport / Munich press (§d) | partial (`te_ars`, `de_ars`, `prev_arr_taxiin`) | **≤ 1.56k ceiling** on the 2025 holdout (§g); realistic 0.1–0.5k [I]. The 2026 July composition makes it more relevant than the holdout shows. The stand taxi-in back-off alone has only a 32.7 MSE level ceiling on 2025 rows (`stand_taxiin.py`) | low if fitted in-fold, as the encodings are (`infold_encodings`). Arrivals' taxi-in is a serve-time input |
| 5 | **Runway state at take-off**: gap to the previous take-off on the same runway; busy-period length and position (a run of take-offs ≤ 180 s apart); previous departure's RECAT-EU category (the follower pays the separation); landings on the same runway in the 20 min before take-off (mixed mode) | queue presence at the threshold; saturation; wake spacing; landings interleaved on mixed-mode runways | [V] Simaiakis TS; [V] Wang et al. 2021; [V] Clewlow 2010 (landings during taxi R² 0.71 at JFK); [V per agent] RECAT-EU 2.0 Table 4 | partial: `rwy_b30`, `rwy_rate`, `q_rwy_tko_sym10`, `arr_b30` | **≤ 0.1k** (level ceilings: gap 97, busy period 89, same-runway landings 59, airport landings 43, airport × wake 13 MSE; §g) | none (MVT of other rows) |
| 6 | **Remote-hold / ATFM interactions**: `aobt_eobt` ≥ 30 min × airport remote-hold practice (LTFM RHA from Oct 2024; FRA remote holding) × stand re-occupied before AOBT_3 | the A-DPI, and so NM's off-block, is postponed to leaving the remote hold; delta is then very negative | [V] DPI §11.3, §9.3(6); [V] DHMİ AIC A07/24; [V] FRA briefing 5.2; [M] §g P(delta < −600) up to 68.5% | components yes (`aobt_eobt`, `prev_arr_gap`); `n_sch_aobt` tested (+350, closed) | small. Arm F already has the level (bias ≤ 26 s per bin). The loss is the held/not-held classification, which no serve-time key has separated so far | none |
| 7 | **De-icing context**: remote-pad airport flag (§d) × freezing METAR × `px_dev` | remote de-icing adds 10–30 min to real taxi; at DPI airports it should also show in EXOT_DI_rem | [V] DPI §11.4; per-airport practice §d; [V] Vakaet 2021 drops < 3 °C flights to avoid de-icing delays | weather arm W (not in F) | small. Arm W measured small. The new ingredient is `px_dev` (rank 1) | none (METAR at or before the anchor) |
| 8 | **Aircraft physical attributes** from the FAA ACD: wingspan → ICAO code letter, engine count/class, MTOW, TDG | turn speeds by code letter; 4-engine start; taxiway restrictions for code E/F | [V] Doc 9157; literature says weak (Idris R² 0.01; PRU does not group by class; Pushback 2nd place: engine class near bottom; PRC 2024 ITA: aircraft characteristics +2.6%) | type and wake encodings yes | **≈ 0** (level ceiling ≤ 32 MSE; rare-type rows are 0.1%). Worth adding only for documentation and rare-type back-off | none |
| 9 | Airport integration class (ANI / CDM / none) and per-airport proxy informativeness (within-pair corr from training) | NM taxi-time source differs by class (DPI vs CACD default) | [V] IN/25-005 (§d) | via ADEP encodings | ≈ 0 as a feature. As a *routing key*, arm Y's per-airport numbers show y-target ≈ delta-target where corr ≈ 0 | none |
| 10 | Arrival-side congestion (taxi-in of arrivals near my take-off) | surface congestion witness; the only realised-taxi signal left in the scored file | [V] Wang; [V] Idris (arrivals R² < 0.02) | **yes** (`arr_ob60_taxiin`, `q_arr_taxiin_sym30_*`, `d_arr_*`) | do not rebuild | none |
| 11 | Recent realised taxi of other departures (Wang's "average speed of last 5 departures"; Vakaet's recent-error feedback; Futer 2006) | strongest dynamic congestion signal in the literature | [V] Wang 2021; [V] Vakaet 2021 | **not available**: departures' BLOCK is blank for the whole scored file. Neighbours' proxies stand in only where the proxy is informative (EHAM, LSZH, LFPG, EDDF, EGLL: `dep_dur`, `rwy_dur`) | structural limit at the 4 synthetic airports | using other scored rows' labels would be leakage, and they do not exist |

**What the ranking implies.** Ranks 1 and 2 are one arm, one cheap A/B, with a documented mechanism.
Ranks 3 and 4 are the only entries with an arguable path to ≥ 0.5k. Everything else is either already
in the model or has a measured level ceiling ≤ 0.1k. **None of these candidates has a located
mechanism for the ≈ 15k gap to 245.** That agrees with PIPELINE_TO_245 §5.

---

## (f) Refused inputs, and claims in our approach that the evidence contradicts

### f1. Refused as features, with the reason

| input | why it is refused |
|---|---|
| **EUROCONTROL PRU "taxi-out additional time" CSV for 2026** (same series as https://www.eurocontrol.int/performance/data/download/csv/taxi_out_additional_time_2025.csv; the 2026 file already holds Jan–Jun 2026 rows) | it is a monthly airport aggregate of the very APDF off-block times that are blanked in the scored file. Using the January 2026 row would leak the label at airport-month level; July 2026 is not published yet. **Refused.** A copy was fetched into scratch during this review. None of its 2026 values are reproduced in this report and none may be used for calibration or sanity-checking a submission. The **2025** file is label information *for the training year only*. It is usable for context and documentation (§d), but adds nothing a model fitted on 2025 does not already have. |
| PRU airport dashboards' 2026 series (https://aiu-airport-dashboard.netlify.app/<ICAO>.html: monthly and per-runway reference and additional taxi-out, "2026 = Jan–Jul") | same reason: aggregates of the hidden target for the scored months. **Refused.** The research sub-agent extracted them into scratch (`airports/eurocontrol_dashboard_extract.txt`). They are not reproduced here and must not be read for modelling decisions. Only the 2021–2025 series may be used, for context |
| EUROCONTROL CODA taxi-time planning values by season/WTC | label-derived for the seasons covered and non-commercial terms; 2025 seasons as a local sanity check only, never a feature or a committed file |
| other scored departures' BLOCK/TAXITIME | blank in the scored file; would be leakage in any case |
| OpenSky data | owner's rule |
| ICAO Doc 8643 data, BADA | licence (§c) |
| any weather or ADS-B observation *after* the take-off time | not used by any candidate; the as-of rule stays |

### f2. Claims in our current approach that the literature or the documents contradict

1. **"`AOBT_3_flt` is NM's actual off-block time" (DATA_DICTIONARY §2.29; the "push-anchored" wording in
   `stand_ab.py` for the queue block; the weather block's "anchor is the row's own pushback").**
   - **Contradicted at LEBL, LEMD, EDDM and LTFM, and in part elsewhere** (§1: DPI Guide §4.7.3,
     §4.7.26; ATFCM UM glossary; within-pair correlation −0.07 to 0.12).
   - At those airports AOBT_3 is take-off minus a planned taxi time, so "pushed-not-airborne at my
     push" counts a fixed window before take-off.
   - This does not invalidate the features, which were measured and they help. It changes what they
     mean, and it is where a re-anchored queue (§e rank 3) comes from.
2. **"Weather is the single highest-value external dataset" (plans/RESEARCH_2026_09_08.md §5 claim 3).**
   - Contradicted by the PRC 2024 2nd-place ablation (weather +1.8%) and by Pushback 2023 (wind and
     cloud dropped as unhelpful).
   - Contradicted by Idris and Diana (IMC weak or not significant). Wang finds weather matters only at
     specific airports.
   - Contradicted by our own arm W (small). De-icing is the exception, and it is airport-specific (§d).
3. **"The queue is the dominant feature family" (same file, claim 2).**
   - True in the US literature for *physical* taxi without a planned taxi time.
   - In A-CDM Europe the pre-departure sequencer absorbs waiting at the stand before AOBT (TSAT = TTOT
     − EXOT, A-CDM Spec §3.5). PRU's additional taxi-out at our airports is only 2.5–9 min/dep.
   - Our dominant "clock gap" family *is* the airport's planned taxi at four airports.
   - So the claim does not transfer. That is consistent with queue counts carrying 4% of gain.
4. **"The strongest dynamic signal can come from recent taxi times" (implicit in `dep_dur`, `rwy_dur`,
   `q_rwy_ambient_proxy`).**
   - The literature's version uses *realised* taxi (Wang, Balakrishna, Wu). Ours uses other rows'
     **proxies**.
   - At LEBL, LEMD, EDDM and LTFM those proxies are planned values. They measure configuration, not
     congestion.
   - So these features are near-blind to congestion exactly where the proxy is synthetic. This is an
     inference from §1 [I]; testable by a per-airport cut of their gain.
5. **ADS-B calibration against AOBT_3 (PIPELINE_TO_245 stage 2).**
   - **Not contradicted at EHAM**, where the proxy behaves as an event stamp (within-pair correlation
     0.61, modal share 0.21 ≈ control 0.16).
   - It **would be** contradicted if the same calibration were extended to LEBL, LEMD, EDDM or LTFM. Do
     not extend it without a re-check.
6. **Consistent, not contradicted.** These current readings match the literature:
   - aircraft size is weak (0.7% gain; Idris R² 0.01; PRU drops class);
   - runway-configuration regimes add nothing beyond stand × runway (Lordan: stand × runway dominates;
     VTT/EXOT already depends on configuration);
   - the EDDM winter / LIRF summer mechanisms of the delta < −600 tail (TRAIN_SERVE_AUDIT §5). The
     documents sharpen these: remote hold postpones the A-DPI (§g), and at EDDM an un-extended EXOT
     sends de-icing time straight into delta.

---

## (g) The descriptive residual cuts behind the "expected value" column [M]

**Data and method.**
- Rows: arm F's fold-A holdout. That is Jan+Jul 2025 matched rows, 339,015 of them, MSE 49,534, RMSE
  222.56, reproducing PIPELINE_TO_245 §0.
- Residual: r = y − max(proxy − treatment, 1).
- For each bin of a candidate quantity I report n, bias (mean r) and MSE.
- I also report a **level-fix ceiling**: the MSE removed if each bin's mean residual were subtracted,
  Σ n·bias²/N. It is in-sample on the holdout, so it is an **optimistic upper bound** for any feature
  acting only through that quantity's level. It says nothing about interactions.
- Script: `resid_cuts.py`, output in `resid_cuts.out`.

| candidate quantity (source of the idea) | level-fix ceiling (MSE of 49,534) | biggest bin effect |
|---|---:|---|
| ICAO code letter from FAA wingspan (Annex 14 code) | **31.9** | code D and code F bias −18 s |
| engine count | 28.1 | 4-engine −13 s on 5,691 rows |
| engine class | 27.7 | turboprop −9 s |
| NM wake category | 30.1 | J −19 s on 2,444 rows |
| training frequency of the type (rare types) | 28.0 | types with 1–100 rows: MSE 106k, but only 414 rows |
| gap to previous take-off on the same runway (queue at the runway) | **97.1** | 300–600 s gaps −24 s |
| busy-period length at take-off (Simaiakis saturation) | 88.8 | idle runway −17 s |
| Idris N = runway take-offs during taxi (already `q_rwy_tko_in_taxi`) | 37.8 | 10–20 −11 s |
| landings on the **same runway** in the 20 min before take-off (Clewlow 2010; `landings.py`) | 59.1 | 5–10 landings +26 s on 9,591 rows; mixed mode is mostly EDDM (92% of its departures) |
| landings at the airport in the 20 min before take-off | 43.0 | ≤ 5 landings −13 s |

**Remote hold / ATFM mechanism (DPI Guide §11.3).** This is P(delta < −600) against AOBT_3 − EOBT_1,
all airports:

| AOBT_3 − EOBT_1 | ≤ −10 min | −10…0 | 0…10 | 10…30 | 30…45 | 45…60 | 1–2 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(delta < −600) | 2.0% | 0.3% | 1.4% | 10.3% | 22.5% | 29.6% | **68.5%** |
| arm F bias (s) | −12 | −7 | −4 | −5 | −3 | +26 | +10 |
| arm F MSE | 51k | 25k | 33k | 71k | 169k | 315k | **819k** |

- The shape matches the documented procedure. Push-and-hold is "mainly applied for flights which have a
  'longer' ATFM delay (> 30–45min)". The A-DPI, and therefore NM's off-block, is postponed to the
  taxi-out from the hold [V, DPI Guide §11.3, §9.3(6)].
- At LIRF, 36.5% of rows in the 10–30 min bin are held. At EGLL, 87% in the 1–2 h bin.
- **Arm F already has the level** (bias ≤ 26 s in every bin). The error there is *which* delayed flights
  were held.
- The stand chain helps. P(delta < −600) is 14.7% when an arrival reached the stand between SCHED and
  AOBT_3 (`n_sch_aobt > 0`), against 4.0% otherwise. That feature family was A/B-tested and closed at
  +350 (MSE_LEDGER lever 2).

**Rare stand×runway pairs (`rare_ars.py`).**

| pair's training support | ≤ 50 rows | 51–200 | 201–1,000 | > 1,000 |
|---|---:|---:|---:|---:|
| holdout share | 5.2% | 15.2% | 61.2% | 18.4% |
| holdout MSE | 76k–108k | 54k | 44k | 57k |

- In the 2026 scored file, 5.9% of matched rows have a pair with ≤ 50 rows in all of 2025.
- By airport: LFPG 9.7%, LTFM 8.0%, EDDF 8.0%, LEMD 7.8%, EDDM 6.7%, EGLL 0.8%.
- Ceiling if those rows were brought to average error: **≈ 1.56k MSE**.

---

## (h) Sources and reproduction

**Operational primary sources, all read [V].** Text extracts of the PDFs are in scratch.
- DPI Implementation Guide 2.700 (June 2025):
  https://www.eurocontrol.int/sites/default/files/2025-06/eurocontrol-dpi-impl-guide-2-700.pdf.pdf
- NM Flight Progress Messages 3.100: https://www.eurocontrol.int/archive_download/all/node/11547
- ATFCM Users Manual MAINT-2:
  https://www.eurocontrol.int/sites/default/files/2024-05/eurocontrol-atfcm-users-manual-maint-2-wef-20240618.pdf
- A-CDM Specification (Jan 2025):
  https://www.eurocontrol.int/sites/default/files/2025-01/eurocontrol-specification-for-acdm.pdf
- NM Information Notice IN/25-005 (A-CDM airport list): https://www.nm.eurocontrol.int/STATIC/docs/pdf/IN-25-005.pdf
- PRU ATXOT methodology: https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf
- APDF data specification: https://www.eurocontrol.int/sites/default/files/2019-10/ao-data-flow-specs.pdf
- EUROCONTROL Sustainable Taxi Operations ConOps (2024):
  https://www.eurocontrol.int/sites/default/files/2024-07/eurocontrol-sustainable-taxi-operations-conops.pdf
- ICAO Doc 9157 Part 2 (BAZL copy): https://www.bazl.admin.ch/dam/de/sd-web/h7SYVJZ5kRTq/9157_p2_cons_en.pdf
- DHMİ AIC A07/24 (LTFM remote holding): https://dhmi.gov.tr/AIPDocuments/LT_Circ_2024_A_07_en.pdf
- PRC DC2026 data and eligibility pages: https://prc-data-challenge-2026.netlify.app/data.html ,
  https://prc-data-challenge-2026.netlify.app/eligibility.html
- NM B2B webinar Q&A (2022): https://www.eurocontrol.int/sites/default/files/2022-06/nm-b2b-webinar-202206-qanda.pdf

**Literature and competitions:** URLs are given per row in §(a).

**Reproduction.**
- Scratch directory: `/private/tmp/claude-501/-Users-saurabhdxt-Projects-prc-challenge/81b461ea-ab3b-47e3-926a-3198c439b0c1/scratchpad/research_lit/`.
- Command: `OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B <script>` from any directory.
- Inputs: `data/cache_stand/{training_2025-*,ranking,fold_preds_queue_order}.parquet`,
  `data/cache_queue/training_2025-{01,07}*.parquet`, `data/raw/training_2025-{01,07}*.parquet`.

| script | what it produces | output | peak RSS |
|---|---|---|---:|
| `within.py` | §1b test 1 | stdout, quoted above | not measured (smaller input than `resid_cuts.py`) |
| `synth.py` | §1b test 2, holdout | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |
| `synth2026.py` | §1b test 2, 2026 | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |
| `resid_cuts.py` | §g table and remote-hold table; joins the FAA ACD from `sheet1.xml.csv`, extracted from `faa_acd_oct2024.xlsx` | `resid_cuts.out` | 0.58 GB |
| `rare_ars.py` | §g rare pairs | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |
| `landings.py` | §g landings rows | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |
| `wake_ap.py` | §c airport × wake residuals | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |
| `stand_taxiin.py` | §d point 3 stand taxi-in back-off | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |
| `eddf_t3.py` | §d / §e rank 4, 2026 new-stand and low-support shares | stdout | not measured (each ran in < 60 s with the machine at load ≈ 3.5) |

- Sub-agent transcripts, flattened: `agent_dumps/{lit,acdm,acft,apt,comp}.txt`.
- **Tests:** none. These are read-only descriptive scripts in scratch. None of them feeds a model or
  a submission. Any feature built from §e must be written with tests first, per CLAUDE.md Part 1.

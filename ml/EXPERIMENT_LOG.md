# Experiment log: low-sugar alert

Dataset: ShanghaiT2DM only (100 patients; 79 development patients in 5 patient-wise folds, 21 locked test patients, `splits.csv`).
Low event = 2+ readings under 70 mg/dL starting within 60 min; first 12 h of each recording blank. Test group: 25 low events
from 6 patients over 248 patient-days. Every choice below was made on development patients only; the locked test group was
looked at only after the choice was fixed, and each variant is reported, including the ones that did not help.

Reproduce: run from `ml/` with `python <script> data/shanghai_t2dm_clean` (data not in the repo, see README for the source).

| # | Question | Script | Chosen on | Result on the locked test group | Verdict |
|---|---|---|---|---|---|
| 1 | Which alert design? | `final_test.py` (`final_test_results.txt`) | development | Forecast alert (lowest of +30 and +60 min <= 90, or 100 with heart disease): 1.50 alerts/patient-day, warned 24 of 25 lows, 7% of alerts followed by a real low. Plain sugar rule: 1.43/day, 22 of 25. Sugar + 0.5 x 30-min change: 2.22/day, 25 of 25. Constrained XGBoost warning model at 1.50/day: 19 of 25 | **Adopted** the forecast alert. A trained low classifier lost to the plain rule, because there are too few lows |
| 2 | Hold back repeat alerts (cooldown after an alert starts)? | `lows_cooldown.py` | development: longest cooldown losing at most 1 dev event = 90 min | 90 min cooldown: 1.50 -> 1.44 alerts/day, warned 24 -> 23 of 25. On development: 1.88 -> 1.78/day. 240 min: 1.42/day but 59 of 66 warned | **Not adopted.** False alarms are separate dips that recover, not repeats |
| 3 | Weight training rows where sugar is already low (< 120) so the forecast is sharper near the alert threshold? | `lows_weighted_forecast.py` (`lows_weighted_results.txt`) | development, weights 1 / 3 / 6 / 10 | Development MAE for sugar < 100 got **worse** with weight (+60 min: 12.51 -> 12.72 / 12.98 / 13.03); alerts at equal recall improved only through a threshold shift. Weight 1 (no change) was chosen. Test: offset -4 gave 1.12 alerts/day but warned 21 of 25 | **Not adopted.** Re-weighting adds no information, as expected with 103 low moments |
| 4 | Every combination of (1) risk gate, (2) k-in-a-row confirmation, (3) patient-day worklist, numbered 1-8 | `lows_combos.py` (`lows_combos_results.txt`) | development: lowest burden that still warns >= 60 of 66 events | Minute alerts: baseline 1.00 alerts/day, 20 of 25 warned; gate only 0.84/day, 18 of 25. Worklist: gate + worklist 43% of patient-days flagged, 21 of 25, precision 13.7%; all three 38% flagged, 19 of 25, precision 13.3%. Confirmation never helped minute alerts; the drug gate (insulin / sulfonylurea) was never picked, the 24 h recent-low gate (<= 80) was | **Not adopted.** Gains come from trading recall for fewer alerts (the dev rule pushed every combination to a -5 mg/dL shift); no combination beat the current design on both recall and burden |

## Notes for readers
- Numbers are small-sample: 25 low events, 6 patients, one patient (2094) has 13 of them. Treat recall as promising, not proven.
- "Alert followed by a low" counts an alert run that contains a moment with a low starting within 60 min.
- Not tried (kept honest): extra datasets (mostly type 1, different lows, licences unchecked) and synthetic data (cannot be scored fairly).

## Dataset scouting for more low events (8 Oct 2026; searched, not downloaded or used)
Checked: web search, PhysioNet, AWS Open Data Registry, data.gov / datahub.io / WHO GHO searches, IEEE DataPort, OMIX.
| Source | What it is | Access found | Fit |
|---|---|---|---|
| IEEE DataPort "Continuous glucose measurement for inpatient with type 2 diabetes" (DOI 10.21227/adzq-2y15) | 43 hospital inpatients with T2D, Dexcom G5/G6, 5-min, 7-10 days, readings clipped to 40-400 | "Requires an IEEE DataPort subscription" (licence not shown) | Closest to our population, but hospital setting, first day unreliable, access not free |
| OMIX002495 (CNCB) "CGM-based hypoglycemia prediction" | Hypoglycemia-focused CGM + clinical data; diabetes type not stated | Not confirmed on the page we read; sibling OMIX002493 (60 T2D patients) is controlled-access | Unknown until access is checked |
| CGMacros (PhysioNet) | 45 people (14 T2D, 16 pre-diabetes) | Open, CC BY-NC-SA 4.0 | Already used for demo patient B only; few lows, non-commercial |
| DiaTrend, OhioT1DM, T1DEXI | Type 1 | Reported as data-use-agreement restricted (secondary source) | Different physiology; not free |
| DiaData (T1D, 13 sets merged), HUPA-UCM | Type 1 | Reported open (secondary source) | Different physiology |
| AWS Open Data Registry, data.gov, datahub.io, WHO GHO | - | No patient-level CGM dataset found; WHO GHO is country-level aggregates | Not usable |
Rule if any is added later: training only, a "which dataset" flag, scoring stays on the locked ShanghaiT2DM test group.

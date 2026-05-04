# 2026-05-04 Session 01 — V4.1/V4.2 Teacher-Smoothed Baseline, Same-Data Reconciliation, and Next Model Leg

## Session Objective

Continue the V4.1/V4.2 forensic and research work around the recovered `20.84%` corrected V4 reference baseline, clarify whether the teacher pseudo-label method is reproducible, package the current best teacher-smoothed benchmark, and define the next research path toward a production-ready model for staged scale-up.

The broader production objective remains:

- continue validating current live V4 production,
- scale capital in stages: `$100K -> $1M -> $100M`,
- keep production safety first,
- use research outputs after they pass clear governance gates.

---

## 1. Starting Context

The recovered corrected V4 reference baseline had previously shown:

| Metric | Value |
|---|---:|
| CAGR | `20.840487%` |
| Sharpe | `1.002359` |
| Sortino | `1.372906` |
| Max DD | `-25.538514%` |
| Final Equity | `$12,883,454.99` |

The key governance caveat remains:

> The recovered `20.84%` baseline uses a full-sample teacher pseudo-label method. It is teacher-smoothed regime research, not strict point-in-time production evidence.

---

## 2. Faithful Legacy Master Data Clarification

The **faithful legacy grid** is a recently reconstructed forensic master dataset:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv
```

It is not simply the current production table:

```text
pawan_quant_research_1.v4.silver_bt_master_grid_1h
```

The faithful grid uses:

- recovered legacy/default market source,
- recovered legacy/default VIX/VIX3M source,
- original `first_releases_<indicator>.csv`,
- ALFRED vintage value correction,
- Finnhub/calendar release timing where encoded in the original CSVs,
- macro `+2h` availability assumption,
- 24/7 hourly forward-fill,
- recovered z-score / event / feature logic.

The macro data is not “just ALFRED.” It is:

```text
legacy first_releases CSV release schedule/timing
+ ALFRED vintage value correction
+ explicit fallback rules
```

A durable master-data note was added to Linear MUB-19 describing assumptions, source logic, release timing, ALFRED selection, fallback behavior, and production-awareness caveats.

---

## 3. Teacher Method Interpretation

The teacher method is used to create smoother regime labels, instead of relying only on the hardcoded next-week return rule:

```text
Expansion: next-week SPY return > +0.5%
Slowdown: between -0.5% and +0.5%
Recession: next-week SPY return < -0.5%
```

The teacher-smoothed approach is conceptually useful because regimes are not necessarily best represented as noisy timestamp-by-timestamp return buckets.

Working description:

> The teacher-smoothed method is a research baseline for smoother regime definition. It should not be described as strict point-in-time production evidence.

---

## 4. Initial V4.2 PIT Teacher-Student Experiment

A strict PIT teacher-student experiment was first run on current production silver:

```text
pawan_quant_research_1.v4.silver_bt_master_grid_1h
```

Output root:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/
```

Results showed that the PIT teacher-student variant only modestly beat the current production read-only reference and materially lagged SPY.

This was later determined to be **not apples-to-apples** against the recovered `20.84%` reference because it used current production silver, not the faithful legacy master grid.

---

## 5. Same-Data Apples-to-Apples Experiment

A corrected same-data experiment was run using the faithful legacy master grid.

Output root:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_2_same_data_teacher_student_2026-05-04/
```

Validation:

| Item | Value |
|---|---|
| Input grid | `full_legacy_grid_clone.csv` |
| Grid rows | `89,991` |
| Grid hash | `b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1` |
| Grid range | `2016-01-01 00:00:00` to `2026-04-07 14:00:00` |
| Canonical signal rows | `43,824` |
| Same timestamp set for all variants | `true` |
| B recovered control match vs recovered corrected WF | `100.0%` |

### Final Same-Data Results

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switches | Match vs 20.84 Ref |
|---|---:|---:|---:|---:|---:|---:|---:|
| A Raw-label WF | `5.4884%` | `0.3669` | `0.4735` | `-40.9836%` | `$6.531M` | `1066` | `69.8544%` |
| B Recovered control | `20.8405%` | `1.0024` | `1.3729` | `-25.5385%` | `$12.883M` | `296` | `100.0000%` |
| C New full-sample teacher | `14.6138%` | `0.8034` | `1.0186` | `-24.1275%` | `$9.889M` | `906` | `71.4563%` |
| D Strict PIT teacher-student | `8.4885%` | `0.5121` | `0.6663` | `-26.2756%` | `$7.514M` | `961` | `69.9388%` |
| E Conservative hybrid | `5.8086%` | `0.3917` | `0.5025` | `-28.7781%` | `$6.631M` | `1297` | `66.4841%` |
| F SPY benchmark | `12.7546%` | `0.7703` | `1.0480` | `-25.1969%` | `$9.113M` | `0` | `2.1929%` |

Interpretation:

- The prior weak `6.0667%` B-like result was caused mainly by using current `v4.silver_bt_master_grid_1h`, not the faithful legacy grid.
- On the correct faithful legacy data, B recovered control returns to `20.8405%`.
- Strict PIT teacher-student still degrades to `8.4885%`.
- Newly trained full-sample teacher C reaches `14.6138%`, beating SPY on CAGR and max drawdown, but remains non-PIT.

---

## 6. Variant C Package

Variant C was packaged as the current teacher-smoothed research benchmark.

Output root:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_2_c_full_sample_teacher_package_2026-05-04/
```

Databricks run:

```text
Run: 1019752531416219
Task: 396198615747320
```

Key outputs:

```text
README.md
code/03_V4_2_Same_Data_Teacher_Student.py
code/04_V4_2_C_Full_Sample_Teacher_Package.py
manifests/runtime_manifest.json
manifests/input_hashes.json
signals/full_signal_export.csv
signals/teacher_pseudo_labels.csv
signals/walkforward_student_signals.csv
performance/equity_curve_hourly_price_updates.csv
performance/equity_curve_daily_last.csv
performance/trades.csv
performance/daily_returns.csv
performance/weekly_returns.csv
performance/monthly_returns.csv
performance/yearly_returns.csv
performance/audit_pack.xlsx
forensics/comparison_vs_B_D_SPY.csv
forensics/variant_c_package_summary.md
```

C package metrics:

| Metric | Value |
|---|---:|
| CAGR | `14.6138%` |
| Sharpe | `0.8034` |
| Sortino | `1.0186` |
| Max DD | `-24.1275%` |
| Final Equity | `$9,889,062.68` |
| Switch Count | `906` |
| Teacher Label Distribution | `{'0': 43679, '1': 19242, '2': 26143}` |

At this stage, C was useful but still not reproducing B.

---

## 7. B-vs-C Lineage Ladder

A B-vs-C lineage audit was then run to understand why C did not match B despite using the same master data.

Output root:

```text
/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/
```

Key files:

```text
03_Research/02_Models/05_V4_2_B_vs_C_Lineage_Diff.py
session notes/2026-05-04_session-01_b_vs_c_lineage_diff.md
forensics/reproducibility_ladder.csv
forensics/b_vs_c_lineage_diff_summary.md
```

Databricks run:

```text
Run id: 653294628821948
Task run id: 77175825505828
```

### Root Cause

C current diverged because it changed the recovered training recipe:

| Item | C0 Current | Recovered/C2 |
|---|---|---|
| Teacher params | `500 trees / depth 6` | `200 trees / depth 4` |
| Preprocessing | current row filtering | recovered fillna / merge / schedule |
| Signal match vs B | `71.4563%` | `100.0000%` |
| CAGR | `14.6138%` | `20.8405%` |

### Reproducibility Ladder

| Variant | Match vs B | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switches |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 current | `71.4563%` | `14.6138%` | `0.8034` | `1.0186` | `-24.1275%` | `$9.889M` | `906` |
| C1 recovered params | `92.3877%` | `23.6492%` | `1.1197` | `1.5146` | `-21.8160%` | `$14.452M` | `276` |
| C2 recovered preprocessing | `100.0000%` | `20.8405%` | `1.0024` | `1.3729` | `-25.5385%` | `$12.883M` | `296` |
| C3 recovered teacher labels | `100.0000%` | `20.8405%` | `1.0024` | `1.3729` | `-25.5385%` | `$12.883M` | `296` |
| C4 recovered WF signals | `100.0000%` | `20.8405%` | `1.0024` | `1.3729` | `-25.5385%` | `$12.883M` | `296` |
| C5 cached models | `100.0000%` | `20.8405%` | `1.0024` | `1.3729` | `-25.5385%` | `$12.883M` | `296` |

### Key Conclusion

C **can** reproduce B and the `20.84%` result without directly using cached labels or cached walk-forward signals.

The minimum reproduction path is:

1. use recovered teacher params,
2. use recovered student params,
3. use recovered preprocessing,
4. use recovered fillna / merge / schedule semantics,
5. use faithful legacy master data.

This means:

> The 20.84% teacher-smoothed baseline is reproducible from code and faithful legacy master data when the recovered training recipe is used exactly.

It is not merely a cached artifact replay.

---

## 8. Updated Baseline Position

As of this session, **B/C2 is the baseline research benchmark** until a better model engine is researched.

Recommended wording:

> B/C2 is the current teacher-smoothed V4 research baseline. It is reproducible from code and faithful legacy master data using recovered parameters and preprocessing. It should be treated as the current benchmark until a better model engine is validated, such as TFT, BiLSTM, PPO, DQN, RL, or other DL/ML/NN architectures.

Important caveat:

> B/C2 is not strict PIT production evidence because regime labeling uses a full-sample teacher-student method. It is teacher-smoothed regime research, not direct production proof.

This balances both points:

1. It is our strongest current reproducible research benchmark.
2. It should not be over-claimed as strict PIT.

---

## 9. Production and Scale-Up Interpretation

The project moved from a more aggressive full-dataset XGBoost/PCA-style research posture to a more conservative 2016+ XGBoost retraining / walk-forward production setup.

That conservative setup is intentional and production-safe.

Current live production has been operating through a volatile market window and has provided real-world validation of the model/pipeline behavior.

Therefore, the scale-up plan remains on track, with the following posture:

1. Current production V4 remains the live validated baseline.
2. B/C2 is the current best teacher-smoothed research benchmark.
3. D strict PIT teacher-student is useful governance evidence but not strong enough for promotion.
4. TFT/DL is the next independent research leg.
5. Capital scaling should proceed through staged live/shadow gates:
   - `$100K`
   - `$1M`
   - `$100M`

No direct production promotion of B/C2 should happen solely from retrospective results.

---

## 10. Daily-Resolution Alternative

A separate hypothesis was discussed:

If longer clean hourly data before 2016 is unavailable, switch to **daily resolution** to access longer history.

Daily V4.2 could test:

1. daily teacher-smoothed regime labels,
2. daily strict PIT-only regime model,
3. longer ETF history,
4. ALFRED/Finnhub/FRED daily macro/volatility data,
5. defensive recession sleeves instead of cash.

Potential recession sleeves:

```text
BIL
SHV
SGOV, if history is sufficient
AGG
IEF
TLT
```

This could reduce reliance on teacher smoothing by exposing the model to more market regimes over longer history.

---

## 11. Next Model Engine Leg

The next independent model research leg should be separate from:

- XGBoost,
- teacher pseudo-labeling,
- recovered V4 lineage.

Candidate engines:

```text
TFT
BiLSTM
PPO
DQN
PCA + LSTM
RL allocation models
other DL / ML / NN models
```

TFT is the preferred next starting point.

Any new model must be tested under:

- strict rolling validation,
- point-in-time feature rules,
- realistic cost/slippage,
- comparison vs SPY,
- comparison vs current production V4,
- comparison vs B/C2 teacher-smoothed benchmark,
- capital scaling assumptions.

---

## 12. Tomorrow / Next Session Plan

1. Generate relevant Excel/audit files for the B/C2 baseline package:
   - daily returns,
   - weekly returns,
   - monthly returns,
   - quarterly returns,
   - yearly returns,
   - trade log,
   - equity curves,
   - signal files,
   - comparison vs SPY/current production reference/PIT variant.
2. Ensure the baseline package has clear README/governance wording and master-data assumptions.
3. Start independent model research leg on top of the robust V4 architecture:
   - TFT,
   - BiLSTM,
   - PPO / RL,
   - DQN / PCA+LSTM and other DL/ML/NN architectures.
4. Explore additional data/features that were previously excluded due to noise risk:
   - PCA or dimensionality reduction,
   - interest rates,
   - credit spreads,
   - additional macro/market features.
5. Explore additional asset classes / allocation sleeves:
   - MMF / money-market funds instead of cash,
   - bond ETFs,
   - futures vs CFDs,
   - swaps/carry/financing costs,
   - leverage assumptions and constraints.

---

## 13. Safety and Validation

Across the work completed in this session:

- no production Delta tables were modified,
- no Saxo execution logic was modified,
- no token refresh logic was modified,
- no realtime notebooks were modified,
- no dashboard notebooks were modified,
- no workflows or schedules were modified,
- no model registry assets were modified.

Research outputs were isolated under `/Volumes/pawan_quant_research_1/v4/market_data/...`.

Validation completed:

- `py_compile` passed for new research scripts,
- safety scans found no production write APIs,
- runtime manifests confirmed no production writes,
- all outputs were isolated to research roots.

---

## 14. Current State at End of Session

### Current production

Current production V4 remains the live validated baseline.

### Current research benchmark

B/C2 is the current teacher-smoothed research baseline:

| Metric | B/C2 |
|---|---:|
| CAGR | `20.8405%` |
| Sharpe | `1.0024` |
| Sortino | `1.3729` |
| Max DD | `-25.5385%` |
| Final Equity | `$12.883M` |
| Switches | `296` |

### Governance label

```text
Teacher-smoothed full-sample regime research baseline.
Reproducible from code and faithful legacy master data.
Not strict PIT production evidence.
Current benchmark until a better model engine is validated.
```

### Git note

The branch `research/v4-2-pit-teacher-student-2026-05-04` was pushed to origin and set to upstream tracking. The worktree may still contain uncommitted/untracked research files; this push only included commits already present on the branch.

### Next actions

1. Commit/push remaining research docs and scripts safely, excluding unrelated dirty files.
2. Keep B/C2 as current research benchmark.
3. Prepare TFT/BiLSTM/PPO next-leg research plan.
4. Consider daily-resolution V4.2 for longer history and defensive recession sleeves.
5. Continue current production monitoring and staged scale-up governance.

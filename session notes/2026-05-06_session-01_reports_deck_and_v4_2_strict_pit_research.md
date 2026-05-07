# 2026-05-06 Session 01 — Report Finalization, Deck Support, and V4.2 Strict-PIT Research

## Session objective

Continue from the 2026-05-05 dashboard/report fixes. Finalize stakeholder reports and deck-support outputs, then begin the V4.2 strict point-in-time research track.

Starting status at the beginning of the session:

| Workstream | Status |
|---|---|
| Dashboard PDF crash | Done |
| PDF formatting | Done, with final report polish still needed |
| Avik job-trigger PDF access | Done |
| Shared dashboard compute | Created; notebook ran successfully; startup latency > ~5 minutes |
| V4.1 support CSVs | Done |
| Deck update | Waiting for deck/source file and final numbers |
| SharePoint upload | Waiting for SharePoint folder/auth details |

Safety posture:

- Keep all report/deck/research changes away from Saxo execution.
- Do not change token refresh, model inference, RT jobs, monthly retraining, live workflows, or production cluster `0311-005159-pm2x3pce`.
- Keep Git state, Databricks Repo metadata, and actual workspace notebook state distinct.

---

## 1. Session tracking

Created the daily tracker:

```text
MUB-24 — 2026-05-06 Session 1: finalize reports, deck, SharePoint, and strict-PIT research plan
```

Session priorities:

1. Finalize reports.
2. Record shared compute operating decision.
3. Prepare deck numbers/charts.
4. Keep SharePoint upload blocked until folder/auth is ready.
5. If time permits, start strict-PIT V4.2 research.

---

## 2. MUB-25 — Final report polish and hardcoded catalyst watchlist

A new issue was created:

```text
MUB-25 — Finalize 04_Saxo_Institutional_Dashboard circulation report wording, hardcoded catalysts, and notebook warnings
```

Requested changes included:

- Rename report title to a V4 strategy-level report.
- Reduce first page density.
- Add hardcoded upcoming economic events from Pawan's weekly Google Sheet.
- Fix radar chart layout.
- Soften `+2h` historical PIT wording to stakeholder-safe language.
- Remove or simplify `Raw value evidence` in circulation.
- Add open-context rows to Appendices A/B.
- Fix repeated Matplotlib `findfont` warnings.
- Fix all-N/A probability table behavior.

### First MUB-25 attempt

Reported branch/commit:

```text
Branch: fix/mub-25-final-report-circulation-polish
Commit: a2a5f79
```

Reported validation:

```text
Interactive run: 405364944938593 / task 634541363735423 — SUCCESS
Overall PDF: run 652077992570817 / task 898395256472029 — SUCCESS
Circulation PDF: run 927496404193282 / task 621977913072105 — SUCCESS
```

However, a notebook syntax error was later identified:

```text
SyntaxError: f-string expression part cannot include a backslash
```

The problematic line computed a unicode suffix inside an f-string expression.

Additional problems found:

- Page 1 probabilities regressed to `N/A`.
- Feature importances became fake equal weights (`7.14%` across all features).
- Radar chart was not centered.
- Hardcoded events appeared but formatting was still dense.

MUB-25 remained open.

### Follow-up commit and partial recovery

Reported follow-up:

```text
Commit: ea795501058c54ac3b236029488219c3c43686af
```

Improvements:

- Syntax error fixed.
- Page 1 probabilities restored.
- Hardcoded catalyst watchlist appeared.
- Macro detail wording changed to `Historical timing rule` and `Validated live-source gate`.
- Raw evidence removed from circulation.
- Appendix open-context rows added.

Remaining issues:

- Feature importance was hidden or unavailable.
- Historical model probability charts disappeared or were incomplete.
- Radar chart still not centered.
- 4d table still too dense.
- Appendix open-context rows were too far from session open.

### Follow-up pass 2

Reported follow-up:

```text
Commit: 7e8c25e8395e91baab8dd3a5f46f7390b7bd7ff9
```

Reported validation:

```text
Interactive run: parent 280571341037640, task 857123660100476
Overall PDF run: parent 374214796816856, task 760886577579529
Circulation PDF run: parent 32743487453202, task 1119625949236648
```

Final improvements reported:

- Page 1 probabilities populated again.
- Historical probability line charts restored.
- Feature importance became real/non-uniform again.
- No fake equal-weight fallback.
- Brown diamond marker removed.
- `+2h` wording absent from circulation.

Latest reviewed PDF:

```text
saxo_institutional_circulation_2026-05-06_SGT (3).pdf
```

Remaining acceptance blockers after pass 2:

1. Radar chart still not visually centered.
2. Section 4d/upcoming-events repeated in interactive notebook: compact stakeholder view plus detailed diagnostics.
3. Historical feature monitoring charts still lacked EDT context on individual charts.
4. Appendix open-context rows were about two hours before first session row; target should be closer to one hour.
5. Some text/table overflow remained.

MUB-25 remained open pending final targeted report cleanup.

---

## 3. MUB-26 — Deck-ready V4.1 backtest validation charts and tables

A new issue was created:

```text
MUB-26 — Generate deck-ready V4.1 backtest validation charts and KPI tables
```

Purpose:

- Generate charts/tables matching the backtest validation slide style.
- Use V4.1/B-C2 support CSVs from MUB-21.
- Do not use live Saxo data.

Requested outputs:

- `deck_kpi_cards.csv`
- `yearly_return_comparison.csv`
- `trailing_1y_3y_5y_metrics.csv`
- deck chart images:
  - yearly return comparison
  - trailing CAGR
  - trailing Sharpe
  - trailing MaxDD
  - combined trailing performance panel
  - optional backtest validation composite visual

Slide values clarified:

| Metric | V4/B-C2 | SPY |
|---|---:|---:|
| CAGR | 20.8% | 12.8% |
| Sharpe | 1.0 | 0.8 / 0.77 |
| Sortino | 1.37 | about 1.05 |
| Max DD | -25.5% | -25.2% |
| Leverage | 0% | 0% |

Governance footnote:

```text
V4.1/B-C2 teacher-smoothed research baseline; reproducible from faithful legacy master data and recovered training recipe; not strict PIT production evidence.
```

---

## 4. V4.2 strict-PIT research begins — MUB-27

A new issue was created:

```text
MUB-27 — V4.2 strict-PIT research plan and next model architecture evaluation
```

Objective:

- Convert the current teacher-smoothed V4.1/B-C2 research benchmark into strict-PIT candidate paths.
- Keep all work isolated in research folders.

Current benchmark context:

| Benchmark | CAGR | Sharpe | Sortino | Max DD | Final Equity |
|---|---:|---:|---:|---:|---:|
| V4.1/B-C2 teacher-smoothed | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M |

Governance caveat:

- B/C2 is reproducible and useful as a research benchmark.
- It is not strict PIT production evidence because it relies on full-sample teacher-smoothed labels.

---

## 5. MUB-27 first experiment — strict-PIT XGBoost/PCA/rolling teacher-student

Reported implementation:

```text
Branch: research/v4-2-strict-pit-labeling-2026-05-06
Commit: 56491589013b39fc35771b839d1fdd6e5763d6b4
Script: 03_Research/02_Models/06_V4_2_Strict_PIT_Labeling_Research.py
Databricks run: 4760500849383 — SUCCESS
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_strict_pit_research_2026-05-06/
```

Input hashes:

```text
Faithful legacy grid: b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1
B/C2 walkforward_signals.csv: 24b9840ebb0c1bdb262a85519ad3c42812c92a8d24986310b6a1a477da3d1dc8
B/C2 full_signal_export.csv: 700eb50c4512edb2f81f59ca47704649e86b4e9ebb027ca1fff03f6cbe91f286
```

Variant comparison:

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switches | Match vs B/C2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A raw-label WF | 5.4884% | 0.3669 | 0.4735 | -40.9836% | $6.531M | 1066 | 69.8544% |
| B teacher-smoothed ref | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M | 296 | 100.0000% |
| C strict rolling teacher-student | 4.6656% | 0.3193 | 0.4269 | -32.0866% | $6.280M | 669 | 68.5492% |
| D train-only PCA XGBoost | 1.7847% | 0.1848 | 0.2552 | -42.2982% | $5.462M | 1494 | 71.4403% |
| E SPY benchmark | 12.7546% | 0.7703 | 1.0480 | -25.1969% | $9.113M | 0 | 2.1929% |

Conclusion:

- Best strict-PIT result was raw-label WF, but it did not close the B/C2 gap.
- Strict rolling teacher-student and PCA XGBoost underperformed raw-label WF.
- More XGBoost/PCA tuning was not justified as the next primary path.

---

## 6. BiLSTM strict-PIT leg

Reported implementation:

```text
Branch: research/v4-2-bilstm-strict-pit-2026-05-06
Commit: 920075bc5d526f94f018fe7012d7a6ce7e5f2007
Databricks run: 60400453222550
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_bilstm_strict_pit_2026-05-06/
```

Comparison:

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switches |
|---|---:|---:|---:|---:|---:|---:|
| A raw-label WF | 5.4884% | 0.3669 | 0.4735 | -40.9836% | $6.531M | 1066 |
| B/C2 ref | 20.8405% | 1.0024 | 1.3729 | -25.5385% | $12.883M | 296 |
| C strict teacher-student | 4.6656% | 0.3193 | 0.4269 | -32.0866% | $6.280M | 669 |
| D PCA XGBoost | 1.7847% | 0.1848 | 0.2552 | -42.2982% | $5.462M | 1494 |
| SPY | 12.7546% | 0.7703 | 1.0480 | -25.1969% | $9.113M | 0 |
| BiLSTM 24h | 6.0122% | 0.4417 | 0.5359 | -22.4240% | $6.695M | 401 |
| BiLSTM 72h | 1.5035% | 0.1723 | 0.1930 | -30.5153% | $5.387M | 390 |
| BiLSTM 168h | 4.9886% | 0.4077 | 0.4442 | -32.8779% | $6.378M | 386 |

Conclusion:

- BiLSTM 24h beat raw-label WF modestly and improved drawdown materially.
- It did not beat SPY or approach B/C2.
- Sequence memory helped, but labels remained the bottleneck.

---

## 7. TFT/attention strict-PIT leg

Reported implementation:

```text
Branch: research/v4-2-tft-strict-pit-2026-05-06
Commit: 6ac2b714202060be5da4130bd67eea1309428203
Databricks run: 832981978530082
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_strict_pit_2026-05-06/
```

Comparison:

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switches |
|---|---:|---:|---:|---:|---:|---:|
| TFT-attn 24h | 7.4604% | 0.5453 | 0.6295 | -26.0036% | $7.165M | 230 |
| TFT-attn 72h | 8.0455% | 0.5986 | 0.6758 | -24.4903% | $7.362M | 134 |

Conclusion:

- TFT attention improved materially over BiLSTM and raw-label WF.
- 72h context was better than 24h.
- Still below SPY and far from B/C2.
- Label redesign became the next main research path.

---

## 8. Defensive ETF / money-market sleeve experiment

Motivation:

- Current recession sleeve was cash.
- Pawan suggested testing BIL/SHV money-market/T-bill ETFs.
- Defensive ETF price history was not initially available in visible Volume artifacts.

Research-only Alpaca defensive ETF price pull was added.

Reported implementation:

```text
Branch: research/v4-2-defensive-etf-prices-2026-05-06
Commit: fffde9de795188c7862bbcfe9576f4e09f79156b
Databricks runs:
  Defensive ETF price pull: 660857949780826
  Updated TFT defensive-sleeve ablation: 821869985214803
Price output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_defensive_sleeve_prices_2026-05-06/
Ablation output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_window_defensive_sleeve_2026-05-06/
```

Coverage summary:

| Symbol | Rows | Days | First | Last | Coverage | Classification |
|---|---:|---:|---|---|---:|---|
| BIL | 7,309 | 1,242 | 2021-01-04 16:00 | 2025-12-31 20:00 | 99.9597% | usable_cash_like |
| SHV | 7,438 | 1,251 | 2021-01-04 15:00 | 2025-12-31 21:00 | 99.9648% | usable_cash_like |
| SGOV | 5,523 | 1,001 | 2021-01-20 14:00 | 2025-12-31 20:00 | 99.1040% | insufficient_history |
| IEF | 9,180 | 1,254 | 2021-01-04 14:00 | 2025-12-31 20:00 | 99.9698% | usable_duration_sleeve |
| AGG | 8,756 | 1,254 | 2021-01-04 14:00 | 2025-12-31 20:00 | 99.9698% | usable_duration_sleeve |
| TLT | 10,264 | 1,254 | 2021-01-04 14:00 | 2025-12-31 21:00 | 99.9698% | usable_duration_sleeve |

Window ablation under cash:

| Variant | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| TFT 72h CASH | 8.0455% | 0.5986 | -24.4903% |
| TFT 168h CASH | 1.0623% | 0.1430 | -40.1864% |
| TFT 336h CASH | 5.0849% | 0.3930 | -34.2918% |

Best defensive sleeve on best window:

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity |
|---|---:|---:|---:|---:|---:|
| TFT 72h BIL | 8.1612% | 0.6056 | 0.6933 | -24.4077% | $7.402M |
| TFT 72h CASH | 8.0455% | 0.5986 | 0.6758 | -24.4903% | $7.362M |
| TFT 72h SHV | 7.9975% | 0.5952 | 0.6871 | -24.4284% | $7.346M |

Conclusion:

- BIL slightly improved CAGR and drawdown versus cash.
- Longer windows beyond 72h did not help.
- No strict-PIT variant beat SPY or closed the B/C2 gap.
- Next step remained label redesign.

Operational note:

- Databricks `alpaca` secret scope/keys were reportedly created/updated from existing fallback credentials.
- No secret values were printed, committed, or written to outputs.

---

## 9. Leverage sensitivity on strict-PIT variants

Pawan asked whether the best strict-PIT model could beat SPY if ETF leverage was applied.

Reported implementation:

```text
Branch: research/v4-2-leverage-sensitivity-2026-05-06
Commit: c48bf557bfc4c3b6dae5d4369738cc5bcd2ec218
Databricks run: 833158008608424
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_leverage_sensitivity_2026-05-06/
```

1x reconciliation:

| Variant | CAGR | Sharpe | Max DD | Reconciles |
|---|---:|---:|---:|---|
| TFT_attention_raw_72h__BIL 1x / 0% financing | 8.1612% | 0.6056 | -24.4077% | yes |

Best TFT 72h BIL by financing:

| Financing | Best L | CAGR | Sharpe | Max DD | Risk Flag | Gap vs SPY | Gap vs B/C2 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0% | 5x | 19.9100% | 0.6056 | -82.4158% | severe | +7.1554pp | -0.9305pp |
| 3% | 3x | 11.6751% | 0.4644 | -62.7796% | high | -1.0795pp | -9.1654pp |
| 5% | 2x | 8.7875% | 0.4291 | -46.8532% | normal | -3.9671pp | -12.0530pp |
| 7% | 1x | 8.1612% | 0.6056 | -24.4077% | normal | -4.5934pp | -12.6793pp |
| 10% | 1x | 8.1612% | 0.6056 | -24.4077% | normal | -4.5934pp | -12.6793pp |

5x check:

| Financing | Beats unlevered SPY? | CAGR | Max DD | Risk |
|---:|---|---:|---:|---|
| 0% | yes | 19.9100% | -82.4158% | severe |
| 5% | no | -2.5550% | -86.2997% | severe |
| 10% | no | -20.8242% | -89.5045% | severe |

Conclusion:

- Leverage is not a solution to the strict-PIT alpha gap.
- 5x only works under unrealistic zero financing and creates severe drawdown.
- More realistic financing assumptions do not beat SPY.
- Leverage may remain a future risk-appetite overlay, not a model-signal solution.

---

## 10. Strict-PIT label redesign

This was the first major positive V4.2 result.

Reported implementation:

```text
Branch: research/v4-2-label-redesign-2026-05-06
Commit: 00834f8db07ad3b09d1bcbcd68063e6a0de8eb28
Databricks run: 560801199016430
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_label_redesign_strict_pit_2026-05-06/
```

Input hashes:

| Artifact | SHA256 |
|---|---|
| Faithful legacy grid | b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1 |
| Defensive ETF hourly prices | 3440dd73d7cd4c4980995d770b504448a2a65f46995c14ea5949ebc8803140d2 |
| B/C2 reference signals | 603b92155f46bcd45344f777f5732fa903b7342bd6608d5bee27abf871b78610 |
| Raw WF reference signals | f86b3a0b5c370a5e3f2a46e551acdacaff21e5eb1f32befe7f0f78f761863a50 |

Label comparison:

| Variant | CAGR | Sharpe | Max DD | Gap vs SPY | Gap vs B/C2 |
|---|---:|---:|---:|---:|---:|
| tft72_bil_drawdown_aware_dd_010_ret_005 | 11.0980% | 1.0233 | -15.6828% | -1.6566pp | -9.7425pp |
| xgboost_drawdown_aware_dd_010_ret_005 | 4.0439% | 0.4173 | -21.1586% | -8.7107pp | -16.7966pp |
| xgboost_triple_barrier_005 | 2.6371% | 0.2379 | -18.8088% | -10.1175pp | -18.2034pp |
| tft72_bil_triple_barrier_005 | 1.3359% | 0.1608 | -32.0572% | -11.4187pp | -19.5046pp |
| xgboost_raw_forward_return_baseline | -1.6205% | -0.0485 | -37.6314% | -14.3751pp | -22.4610pp |

Best strict-PIT variant:

```text
tft72_bil_drawdown_aware_dd_010_ret_005
```

Metrics:

| Metric | Value |
|---|---:|
| CAGR | 11.0980% |
| Sharpe | 1.0233 |
| Sortino | 1.1103 |
| Max DD | -15.6828% |
| Final equity | $8.463M |
| Switch count | 137 |
| Gap vs SPY | -1.6566pp CAGR |
| Gap vs B/C2 | -9.7425pp CAGR |

Interpretation:

- First strict-PIT path with meaningfully improved risk-adjusted performance.
- Drawdown-aware label was the key improvement.
- It still did not beat SPY on CAGR, but it beat SPY on Sharpe and max drawdown quality.
- The model became more institutionally usable, but upside capture remained insufficient.

K-12 explanation recorded:

- Old labels asked mostly: did SPY go up or down one week later?
- Drawdown-aware labels ask: did the market go up enough and avoid a bad drawdown path?
- This better matches the real objective: avoid dangerous losses while staying invested when conditions are healthy.

---

## 11. Feature clarification for V4.2

The current V4.2 strict-PIT experiments still used the original 14 V4 features.

Feature groups:

### Macro z-score features

```text
CPIAUCSL_z
PCEPI_z
PCE_z
PAYEMS_z
UNRATE_z
GDPC1_z
INDPRO_z
UMCSENT_z
```

### Volatility / term structure features

```text
VIXCLS_z
vix3m_z
VIX_Inversion_Z
```

### Price / technical feature

```text
Price_Mom_24H
```

### Event features

```text
CPI_Event_Active
NFP_Event_Active
```

Clarifications:

- `BIL` is not an input feature; it is a defensive allocation sleeve.
- The drawdown-aware label is the target definition, not an input feature.
- TFT/BiLSTM sequence models use windows of the same 14 features, e.g. 72h context.

---

## 12. Calendar/time feature idea and ablation

Pawan asked whether TFT's 72h memory already captures weekday/week-pattern effects and whether adding day-of-week and `week_number mod 3` could help.

Explanation:

- 72h lookback is roughly 3 days, not a full Monday-Friday week.
- Calendar features are strict-PIT safe because they are known in advance.
- But raw `year` should be avoided initially because it risks historical memorization.

A calendar/time feature ablation was run.

Reported implementation:

```text
Branch: research/v4-2-calendar-feature-ablation-2026-05-07
Commit: 9a69c0a934a9c9c9d24f0c977916422b5de7753d
Databricks run: 748784257343923
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_calendar_feature_ablation_2026-05-07/
```

Baseline reconciliation:

```text
A_baseline_14 reconciled exactly to prior best:
CAGR 11.0980%, Sharpe 1.0233, Max DD -15.6828%, Final equity $8.463M
```

Feature ablation summary:

| Feature Set | CAGR | Sharpe | Max DD | Calmar | Final Equity | Switches |
|---|---:|---:|---:|---:|---:|---:|
| A_baseline_14 | 11.0980% | 1.0233 | -15.6828% | 0.7077 | $8.463M | 137 |
| B_day_of_week | 6.9544% | 0.6724 | -16.8224% | 0.4134 | $6.998M | 179 |
| D_calendar_session | 5.2214% | 0.5058 | -19.3826% | 0.2694 | $6.449M | 193 |
| C_day_of_week_week_mod3 | 4.4943% | 0.4676 | -20.5449% | 0.2188 | $6.229M | 177 |

Conclusion:

- Calendar/time features hurt the current TFT 72h + BIL + drawdown-aware setup.
- Best CAGR, best Sharpe, and best balanced feature set all remained the original 14-feature baseline.
- Day-of-week reduced CAGR by about 4.14 percentage points.
- Week-number mod 3 and session/month-end/quarter-end additions degraded further.
- Calendar features should be dropped for the current candidate.

---

## 13. V4.2 TFT deck-support package

Pawan noted that existing leverage slide/table was for the XGBoost/V4.1/B-C2 baseline, not TFT.

A support script was created:

```text
03_Research/03_Support/03_V4_2_TFT_Deck_Support_CSVs.py
```

Reported initial commit:

```text
5bfe29a0091a0d9ec5931cf19779dd60ee0aab21
```

Initial blocker:

- Databricks CLI profiles had invalid refresh tokens, so script could not initially run against DBFS.

After reauth, reported completion:

```text
Branch: research/v4-2-calendar-feature-ablation-2026-05-07
Latest commit: a34bcd963b6e4304f7778faeb937182423f7442e
Databricks run: 1097622971680703
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_deck_support_2026-05-07/
```

Generated package:

```text
daily_performance.csv
monthly_performance.csv
quarterly_performance.csv
half_yearly_performance.csv
yearly_performance.csv
summary_metrics.csv
presentation_numbers.csv
leverage_sensitivity.csv
metric_reconciliation.csv
source_files.csv
runtime_manifest.json
README.md
v4_2_tft_deck_support.xlsx
```

Reconciled metrics:

| Metric | Value |
|---|---:|
| CAGR | 11.0980% |
| Sharpe | 1.0233 |
| Sortino | 1.1103 |
| Max DD | -15.6828% |
| Final equity | $8.463M |
| Switches | 137 |

Leverage assumption used:

- Fixed `4.50%` annual financing proxy.
- Explicitly documented because no exact 10Y Treasury series was present in inspected MUB-27 artifacts.
- Marked as proxy, not fabricated Treasury data.

Later Codex response outside this chat generated B-C1/XGBoost leverage CSVs with:

```text
Borrowing cost assumption: 10 Year US Treasury Bill + 50 bps premium
```

Reported value:

```text
FRED DGS10 latest available 10Y Treasury constant maturity yield: 4.43% as of 2026-05-05
+ 50 bps premium = 4.93% annual borrowing cost
```

Reported output files:

```text
B-C1-Xgboost_leverage_analysis_10Y_plus_50bps_table.csv
B-C1-Xgboost_leverage_analysis_10Y_plus_50bps_raw.csv
B-C1-Xgboost_leverage_analysis_10Y_plus_50bps_metadata.json
```

---

## 14. Research conclusions at end of 2026-05-06 session

### What is now known

1. Strict-PIT XGBoost/PCA is weak.
2. BiLSTM adds modest sequence signal but not enough.
3. TFT attention improves strict-PIT sequence modeling.
4. BIL is the best tested cash-like defensive sleeve, but only adds slight uplift.
5. Leverage does not solve the alpha gap under realistic financing and risk constraints.
6. Drawdown-aware labels are the first meaningful strict-PIT improvement.
7. Calendar/time features hurt the current candidate and should be dropped.
8. The current best strict-PIT candidate is:

```text
TFT_attention_72h + BIL + drawdown-aware label + original 14 features
```

### Current best V4.2 strict-PIT candidate

| Metric | Value |
|---|---:|
| CAGR | 11.0980% |
| Sharpe | 1.0233 |
| Sortino | 1.1103 |
| Max DD | -15.6828% |
| Final equity | $8.463M |
| Switches | 137 |

### Comparison

| Benchmark | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| V4.2 strict-PIT candidate | 11.0980% | 1.0233 | -15.6828% |
| SPY | 12.7546% | 0.7703 | -25.1969% |
| V4.1/B-C2 teacher-smoothed reference | 20.8405% | 1.0024 | -25.5385% |

Interpretation:

- V4.2 strict-PIT candidate does not beat SPY on CAGR yet.
- It beats SPY on Sharpe and drawdown quality.
- It is materially below B/C2 on CAGR, but B/C2 is non-PIT.
- This is the first strict-PIT version with institutionally attractive risk control.

---

## 15. Recommended next steps

1. Run drawdown-aware label threshold grid:
   - drawdown thresholds: 7.5%, 10%, 12.5%, 15%
   - return thresholds: 0.25%, 0.5%, 0.75%, 1.0%
   - keep model fixed: TFT 72h + BIL + original 14 features

2. Verify all reported research branches are pushed to origin.
   Several reported commit SHAs were not immediately visible through the GitHub connector.

3. Use `v4_2_tft_deck_support.xlsx` for V4.2 strict-PIT slides, but clearly separate it from V4.1/B-C2 XGBoost deck support.

4. Keep V4.2 strict-PIT as research only. Do not promote to production without additional robustness, threshold sensitivity, and forward/paper validation.

5. Keep SharePoint upload work separate and blocked until folder/auth is available.

## Production safety confirmation

Across the session's reported work:

- No production V4 Delta tables were intentionally modified.
- No Saxo execution logic was changed.
- No token refresher was changed.
- No RT/dashboard production jobs were changed for V4.2 research.
- No live schedules or model registry objects were changed.
- All research outputs were isolated under `/Volumes/pawan_quant_research_1/v4/market_data/...` research folders.

## Caution

Many research branch commits were reported locally, but some were not visible through the GitHub connector when checked. Before ending the broader workstream, verify branch pushes for:

```text
research/v4-2-strict-pit-labeling-2026-05-06
research/v4-2-bilstm-strict-pit-2026-05-06
research/v4-2-tft-strict-pit-2026-05-06
research/v4-2-defensive-etf-prices-2026-05-06
research/v4-2-leverage-sensitivity-2026-05-06
research/v4-2-label-redesign-2026-05-06
research/v4-2-calendar-feature-ablation-2026-05-07
```

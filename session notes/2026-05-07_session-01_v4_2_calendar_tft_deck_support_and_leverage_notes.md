# 2026-05-07 Session 01 — V4.2 Calendar Feature Ablation, TFT Deck Support, and Leverage Assumption Notes

## Session objective

Continue from the 2026-05-06 V4.2 strict-PIT research work. Analyze the current V4.2 candidate, clarify the feature set and TFT temporal-memory behavior, test calendar/time features, generate deck-support outputs for the TFT/V4.2 candidate, and clarify leverage-analysis assumptions for slide use.

Safety posture:

- V4.2 work remains research-only.
- No Saxo live execution changes.
- No token refresher changes.
- No production model registry changes.
- No RT/dashboard production workflow changes.
- No trading logic changes.
- All research outputs stay under isolated `/Volumes/pawan_quant_research_1/v4/market_data/...` folders.

---

## 1. Starting point from prior session

The best strict-PIT candidate at the start of the session was:

```text
TFT_attention_72h + BIL defensive sleeve + drawdown-aware label + original 14 V4 features
```

Best reported metrics:

| Metric | Value |
|---|---:|
| CAGR | 11.0980% |
| Sharpe | 1.0233 |
| Sortino | 1.1103 |
| Max DD | -15.6828% |
| Final equity | $8.463M |
| Switches | 137 |
| Gap vs SPY | -1.6566pp CAGR |
| Gap vs B/C2 | -9.7425pp CAGR |

Interpretation carried forward:

- The drawdown-aware label was the first meaningful strict-PIT improvement.
- V4.2 strict-PIT still did not beat SPY on CAGR.
- It did beat SPY on Sharpe and drawdown quality.
- Further work should tune labels/features, but remain research-only.

---

## 2. Explanation: what worked in the label redesign

The session clarified that the major improvement came from **label design**, not from adding new features or leverage.

K-12 explanation used:

- Old labels were like grading a school week only by whether the final score was slightly higher or lower on Friday.
- That can be noisy because the path may have included a scary fall before recovering.
- Drawdown-aware labels ask whether the market improved **and** whether it stayed safe along the way.
- This better matches the strategy objective: avoid major drawdowns while staying invested when conditions are healthy.

Trading interpretation:

- Old raw label mostly asked: did SPY go up or down one week later?
- New drawdown-aware label asks: did the market go up enough and avoid a bad drawdown path?
- This made the target less noisy and more aligned with risk-controlled regime behavior.

Comparison:

| Setup | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| TFT 72h + BIL, raw label | 8.1612% | 0.6056 | -24.4077% |
| TFT 72h + BIL, drawdown-aware label | 11.0980% | 1.0233 | -15.6828% |

Conclusion:

```text
The model learned better because the label described the regime we actually care about, not just a noisy one-week return bucket.
```

---

## 3. Feature set clarification

The current V4.2 strict-PIT experiments still used the original **14 V4 features**.

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

### Volatility / term-structure features

```text
VIXCLS_z
vix3m_z
VIX_Inversion_Z
```

### Price / technical feature

```text
Price_Mom_24H
```

### Event pulse features

```text
CPI_Event_Active
NFP_Event_Active
```

Clarifications:

- `BIL` is not an input feature; it is the defensive allocation sleeve when predicted regime is risk-off/recession.
- The drawdown-aware label is not an input feature; it is the training target.
- TFT/BiLSTM use a sequence window of these same 14 features, e.g. 72 hours.

---

## 4. TFT temporal memory discussion

The session clarified what TFT temporal memory means.

Current best setup:

```text
TFT_attention_72h
```

Meaning:

- Each prediction sees the previous 72 hourly rows.
- Because the grid is hourly and 24/7 forward-filled, 72h is roughly 3 calendar days.
- It can capture recent feature movement, such as VIX rising, price momentum fading, or macro states remaining stale/fresh.
- It does not automatically know broader calendar context like Monday vs Friday unless those features are added explicitly.

Simple explanation:

```text
72h lookback = what happened recently
calendar features = where we are in the market calendar
```

This motivated a feature ablation around calendar/time features.

---

## 5. Calendar feature idea

Candidate calendar/time features discussed:

- day of week
- `week_number mod 3`
- hour/session flags
- month-end / quarter-end flags
- first/last trading day of month

Important implementation principles:

- These features are strict-PIT safe because they are known in advance.
- Avoid raw `year`, because it can encourage period memorization.
- Prefer cyclical or one-hot encodings:
  - `dow_sin`, `dow_cos` or weekday one-hot
  - `week_mod3_0`, `week_mod3_1`, `week_mod3_2`
  - `hour_sin`, `hour_cos`

Recommended ablation design:

| Feature Set | Description |
|---|---|
| A_baseline_14 | Original 14 features |
| B_day_of_week | 14 + day-of-week features |
| C_day_of_week_week_mod3 | 14 + day-of-week + week-number mod 3 |
| D_calendar_session | 14 + day-of-week + hour/session + month-end/quarter-end flags |

Fixed setup for the test:

```text
TFT 72h + BIL + drawdown-aware label
```

---

## 6. Calendar/time feature ablation result

Reported implementation:

```text
Branch: research/v4-2-calendar-feature-ablation-2026-05-07
Commit: 9a69c0a934a9c9c9d24f0c977916422b5de7753d
Databricks run: 748784257343923
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_calendar_feature_ablation_2026-05-07/
```

Input hashes:

| Artifact | SHA256 |
|---|---|
| Faithful legacy grid | b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1 |
| Defensive ETF hourly prices | 3440dd73d7cd4c4980995d770b504448a2a65f46995c14ea5949ebc8803140d2 |
| Label file | d6ac55776acc848ae731fd5b06b2e7e7cf83636036f526ba0d04333e463b18f4 |
| B/C2 reference signals | 603b92155f46bcd45344f777f5732fa903b7342bd6608d5bee27abf871b78610 |

Baseline reconciliation:

```text
A_baseline_14 matched the prior best exactly:
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

- Calendar/time features hurt the current setup.
- Best CAGR, best Sharpe, and best balanced feature set all remained original 14 features.
- Day-of-week reduced CAGR by about 4.14 percentage points.
- Week-number mod 3 and session/month-end/quarter-end features degraded further.
- No calendar feature set beat SPY on CAGR.

Research conclusion:

```text
Drop calendar/time features for the current V4.2 strict-PIT candidate.
Keep original 14 V4 features.
The signal quality is coming from drawdown-aware labels plus TFT 72h context plus BIL sleeve.
```

---

## 7. V4.2 TFT deck-support package

Pawan clarified that the existing leverage slide/table was for the **XGBoost/V4.1/B-C2** baseline, not TFT.

Therefore a separate TFT/V4.2 deck-support package was needed.

Script created:

```text
03_Research/03_Support/03_V4_2_TFT_Deck_Support_CSVs.py
```

Reported initial commit:

```text
5bfe29a0091a0d9ec5931cf19779dd60ee0aab21
```

Initial blocker:

- Databricks CLI profiles had invalid refresh tokens, so script could not initially run against DBFS.

After Databricks re-authentication, the package was completed.

Reported completion:

```text
Branch: research/v4-2-calendar-feature-ablation-2026-05-07
Latest commit: a34bcd963b6e4304f7778faeb937182423f7442e
Databricks run: 1097622971680703
Output root: /Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_deck_support_2026-05-07/
```

Generated files:

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

Source files used:

```text
tft72_bil_drawdown_aware_dd_010_ret_005_daily_returns.csv
tft72_bil_drawdown_aware_dd_010_ret_005_signals.csv
label redesign variant_comparison.csv
label distribution summary
defensive ETF coverage summary
prior leverage summary for context only
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

Leverage assumption used in TFT support package:

```text
Fixed 4.50% annual financing proxy
```

This was explicitly documented because no exact 10Y Treasury series was present in the inspected MUB-27 research artifacts. It was marked as a proxy, not fabricated Treasury data.

Excel workbook validation:

- `v4_2_tft_deck_support.xlsx` was verified as a valid `.xlsx` zip package.
- Workbook contained 10 sheets.
- Package was declared ready for PPT update.

---

## 8. B-C1/XGBoost leverage analysis with 10Y + 50 bps borrowing cost

The final user-provided Codex note, generated outside this chat, updated the leverage analysis for the XGBoost/B-C1 slide.

Requested borrowing-cost assumption:

```text
Borrowing cost assumption: 10 Year US Treasury Bill + 50 bps premium
```

Reported outputs:

```text
B-C1-Xgboost_leverage_analysis_10Y_plus_50bps_table.csv
B-C1-Xgboost_leverage_analysis_10Y_plus_50bps_raw.csv
B-C1-Xgboost_leverage_analysis_10Y_plus_50bps_metadata.json
```

Borrowing cost used:

```text
Latest available FRED DGS10 10Y Treasury constant maturity yield: 4.43% as of 2026-05-05
+ 50 bps premium
= 4.93% annual borrowing cost
```

Reported method:

```text
strategy_daily_return = strategy_equity.pct_change()
levered_return = leverage * strategy_daily_return - max(leverage - 1, 0) * 4.93% / 252
```

Production safety:

- No production objects changed.

---

## 9. Updated interpretation at end of 2026-05-07 session

### What worked

1. Drawdown-aware labels materially improved strict-PIT performance.
2. TFT attention over 72h helped more than XGBoost or BiLSTM.
3. BIL defensive sleeve modestly improved over cash.
4. Original 14 V4 features remained best.

### What did not work

1. Calendar/time features degraded performance.
2. Longer windows beyond 72h degraded performance.
3. Leverage did not solve alpha gap under realistic financing assumptions.
4. PCA/XGBoost and rolling teacher-student variants remained weak.

### Current best V4.2 strict-PIT candidate

```text
TFT_attention_72h + BIL + drawdown-aware label + original 14 V4 features
```

Metrics:

| Metric | Value |
|---|---:|
| CAGR | 11.0980% |
| Sharpe | 1.0233 |
| Sortino | 1.1103 |
| Max DD | -15.6828% |
| Final equity | $8.463M |
| Switches | 137 |

Comparison:

| Benchmark | CAGR | Sharpe | Max DD |
|---|---:|---:|---:|
| V4.2 strict-PIT candidate | 11.0980% | 1.0233 | -15.6828% |
| SPY | 12.7546% | 0.7703 | -25.1969% |
| V4.1/B-C2 teacher-smoothed reference | 20.8405% | 1.0024 | -25.5385% |

Institutional read:

- V4.2 strict-PIT candidate is not yet a CAGR winner versus SPY.
- It has better Sharpe and much better drawdown than SPY.
- It is cleaner governance-wise than B/C2, but still needs more robustness and threshold tuning before any production consideration.

---

## 10. Recommended next steps

1. Run drawdown-aware label threshold grid:

```text
drawdown thresholds: 7.5%, 10%, 12.5%, 15%
return thresholds: 0.25%, 0.5%, 0.75%, 1.0%
```

Fixed setup:

```text
TFT_attention_72h + BIL + original 14 features
```

2. Do not add calendar/time features further unless a new hypothesis justifies it.

3. Verify branch pushes to origin for research branches. Several reported commits were not visible through GitHub connector when checked.

4. Use separate slide labels:

- V4.1/B-C2 XGBoost teacher-smoothed reference for original leverage/backtest slides.
- V4.2 TFT strict-PIT candidate for strict-PIT research slides.

5. Keep V4.2 TFT as research only. Do not promote to live model registry or Saxo execution.

## Production safety confirmation

Across the session's reported work:

- No production V4 Delta tables were intentionally modified.
- No Saxo execution logic changed.
- No token refresher changed.
- No RT/dashboard production job changed.
- No live schedules or model registry objects changed.
- Research outputs were isolated under `/Volumes/pawan_quant_research_1/v4/market_data/...`.

## Git/checkpoint caution

Several reported local commits were not visible through the GitHub connector during the chat. Before closing the broader research loop, verify or push:

```text
research/v4-2-label-redesign-2026-05-06
research/v4-2-calendar-feature-ablation-2026-05-07
research/v4-2-defensive-etf-prices-2026-05-06
research/v4-2-leverage-sensitivity-2026-05-06
research/v4-2-tft-strict-pit-2026-05-06
research/v4-2-bilstm-strict-pit-2026-05-06
research/v4-2-strict-pit-labeling-2026-05-06
```

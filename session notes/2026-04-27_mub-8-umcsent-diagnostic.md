# MUB-8 UMCSENT Diagnostic

Date: 2026-04-27

Branch: `jatalep2018/mub-8-bug-umcsent-latest-public-value-not-reaching-model-input`

Scope: diagnostic only. No production feed logic, model inference logic, or silver schema changes were made.

## Summary

The latest `silver_rt_1h` rows are receiving an updated UMCSENT z-score, and the persisted regime signal rows are using that same z-score. The apparent dashboard mismatch is caused by comparing a live April 2026 final value from Finnhub/UMich (`49.8`) against the latest ALFRED/FRED public reference row currently visible to the dashboard (`53.3`, March 2026 final).

Based on the current evidence, `49.8` is the model-eligible value for live realtime inference after the April final release. `53.3` is still a valid public reference, but it is the prior March observation and should not be treated as the current April model input after the validated live final row exists.

## Query Results

### 1. Live UMCSENT rows

Query target: `pawan_quant_research_1.v4.bronze_rt_macro_events_live`

Rows returned: 2

| scheduled_release_ts | observed_value | provider_event_name | release_stage | validation_status | status | ingestion_ts |
| --- | ---: | --- | --- | --- | --- | --- |
| 2026-04-24 18:00:00 | 49.8 | Michigan Consumer Sentiment Final | final | validated | observed | 2026-04-25 03:01:44.514079 |
| 2026-04-10 18:00:00 | 47.6 | Michigan Consumer Sentiment Prel | prelim | ignored_prelim | observed | 2026-04-11 03:01:42.145695 |

Findings:
- `49.8` exists in live macro and is `release_stage='final'`, `validation_status='validated'`.
- `53.3` does not exist in live macro.
- The parser correctly classified the preliminary row as `prelim` / `ignored_prelim` and the final row as `final` / `validated`.

### 2. Latest silver RT UMCSENT rows

Query target: `pawan_quant_research_1.v4.silver_rt_1h`

Latest row:

| timestamp | UMCSENT_z | source |
| --- | ---: | --- |
| 2026-04-24 20:00:00 | -0.6382303448 | rt_enrichment_live_recompute |

Recent change points:

| timestamp | UMCSENT_z | source |
| --- | ---: | --- |
| 2026-04-24 16:00:00 | -0.6382303448 | rt_enrichment_live_recompute |
| 2026-04-20 12:00:00 | 0.3135200945 | rt_enrichment_live_recompute |
| 2026-04-06 16:00:00 | -0.6529562024 | rt_enrichment |

Finding: UMCSENT is not stuck. It changed in `silver_rt_1h`, including a drop on 2026-04-24.

### 3. Batch macro UMCSENT rows

Query target: `pawan_quant_research_1.v4.bronze_bt_macro_events`

Latest batch row:

| timestamp | actual | event | ingestion_ts |
| --- | ---: | --- | --- |
| 2026-03-27 14:00:00 | 53.3 | First Release: UMCSENT | 2026-04-24 15:15:23.051986 |

Finding: batch macro PIT reconstruction at the latest silver timestamp resolves to `53.3`, because batch has March as the latest available UMCSENT observation.

### 4. Batch silver UMCSENT history

Query target: `pawan_quant_research_1.v4.silver_bt_master_grid_1h`

Latest returned batch silver row:

| timestamp | UMCSENT_z |
| --- | ---: |
| 2026-04-06 14:00:00 | -0.6382303448 |

Finding: batch silver history is older than realtime silver. It is not the current live model input after the April final live override.

### 5. Model/regime signal rows

Query target: `pawan_quant_research_1.v4.bronze_rt_regime_signal`

Rows around the UMCSENT change show the same z-score used for inference:

| timestamp | UMCSENT_z | regime_name | position | source |
| --- | ---: | --- | --- | --- |
| 2026-04-24 15:00:00 | 0.3135200945 | Recession | 100% CASH | rt_inference_v4_live_alias |
| 2026-04-24 16:00:00 | -0.6382303448 | Recession | 100% CASH | rt_inference_v4_live_alias |
| 2026-04-24 20:00:00 | -0.6382303448 | Recession | 100% CASH | rt_inference_v4_live_alias |

Finding: the UMCSENT z-score in `silver_rt_1h` is reaching the persisted model/regime signal table. Probability columns in this table are currently null, but the feature columns and regime output are present.

### 6. ALFRED/FRED public reference

Dashboard-style FRED/ALFRED query for `UMCSENT`, observation window 2026-01-01 through 2026-04-30:

| realtime_start | observation_date | value |
| --- | --- | ---: |
| 2026-01-23 | 2026-01-01 | 56.4 |
| 2026-02-20 | 2026-02-01 | 56.6 |
| 2026-03-27 | 2026-03-01 | 53.3 |

FRED/ALFRED did not return an April 2026 UMCSENT observation in this pull. The dashboard public vintage reference therefore compares the live April final value (`49.8`) to the latest public row known through ALFRED/FRED (`53.3`, March).

External cross-check:
- University of Michigan Surveys of Consumers shows Final Results for April 2026 with Index of Consumer Sentiment `49.8`, and March 2026 `53.3`: https://www.sca.isr.umich.edu/
- Investing.com calendar shows Apr 24, 2026 actual `49.8`, prior `53.3`, and Apr 10, 2026 preliminary `47.6`: https://www.investing.com/economic-calendar/michigan-consumer-sentiment-320

## Answers

1. Does `bronze_rt_macro_events_live` contain `49.8`?
   - Yes. It is the latest UMCSENT live row, `release_stage='final'`, `validation_status='validated'`.

2. Does `bronze_rt_macro_events_live` contain `53.3`?
   - No. `53.3` appears in batch/public history as the March 2026 final observation.

3. Which row has `release_stage='final'` and `validation_status='validated'`?
   - `Michigan Consumer Sentiment Final`, scheduled `2026-04-24 18:00:00`, observed value `49.8`.

4. Was `53.3` available before the latest `silver_rt_1h` timestamp?
   - Yes. Batch PIT resolves `53.3` from `bronze_bt_macro_events` at release timestamp `2026-03-27 14:00:00`, available under the +2h rule before latest model timestamp `2026-04-24 20:00:00`.
   - But it was superseded for live realtime inference by the validated April final live row `49.8`.

5. Did `silver_rt_1h` use a stale live override?
   - No evidence of a stale live override. The latest validated live row is `49.8`, and `silver_rt_1h` changed around 2026-04-24 to the current UMCSENT z-score.

6. Is the parser misclassifying UMich preliminary/final event names?
   - No evidence of misclassification. `Michigan Consumer Sentiment Prel` is `prelim` / `ignored_prelim`; `Michigan Consumer Sentiment Final` is `final` / `validated`.

7. Is ALFRED/FRED public reference comparing a different vintage/stage than Finnhub?
   - Yes. It is comparing latest public ALFRED/FRED row for March (`53.3`) against live vendor April final (`49.8`). This is a period/stage availability mismatch, not a direct raw-value mismatch for the same observation.

## Root Cause Hypothesis

The dashboard audit table is semantically overloading "public latest value." For UMCSENT, the live feed can capture the April final value before the ALFRED/FRED public reference path exposes the April observation. The audit then compares April live final `49.8` to March public ALFRED/FRED `53.3`, producing `CHECK_STAGE_OR_VENDOR_SEMANTICS`.

This is not evidence that the latest public value failed to reach `silver_rt_1h`. It is evidence that the public reference table is lagging or representing a different observation period than the live validated final override.

## Model Eligibility

`49.8` should be model-eligible after the validated live final event is available. It is the April 2026 final UMich sentiment value and is the correct current live input candidate.

`53.3` was model-eligible before the April final live event. It remains the correct March batch/public reference value, but it should not override the April final live value in realtime inference.

## Smallest Safe Fix If Needed

Diagnostic/dashboard patch:
- For live override rows, compare public reference only when `observation_date` / release month / stage aligns with the live event.
- If ALFRED/FRED latest public row lags the live event period, display `PUBLIC_REFERENCE_LAGS_LIVE_EVENT` or keep `CHECK_STAGE_OR_VENDOR_SEMANTICS` with explicit observation-period notes.
- Add live event fields to the audit display: `provider_event_name`, `scheduled_release_ts`, `release_stage`, `validation_status`, and whether ALFRED/FRED has an aligned observation.

Production patch, only if later approved:
- Persist an explicit `observation_date` / `period` on `bronze_rt_macro_events_live`.
- Reconcile live UMCSENT final rows to ALFRED/FRED once ALFRED exposes the matching observation, without blocking realtime model use of validated final live rows.

## Validation Plan

1. Re-run live UMCSENT query and confirm latest final row remains `49.8`, `final`, `validated`.
2. Re-run FRED/ALFRED UMCSENT query after FRED publishes April observation and confirm April `49.8` appears.
3. Confirm `silver_rt_1h.UMCSENT_z` continues from the live final value until the next eligible sentiment release.
4. Confirm `bronze_rt_regime_signal.UMCSENT_z` matches `silver_rt_1h.UMCSENT_z` for the same timestamps.
5. Update dashboard audit semantics so mismatched observation periods are not treated as same-period raw value mismatches.

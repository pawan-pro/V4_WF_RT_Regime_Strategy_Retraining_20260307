# V4.2 PIT Teacher-Student Research Experiment

This folder contains a research-only implementation of a strict point-in-time teacher-student path for V4.2. It converts the recovered corrected-reference teacher pseudo-label idea into comparable variants while avoiding production writes.

## Scope

Script:

`03_Research/02_Models/01_V4_2_PIT_Teacher_Student.py`

Finalizer:

`03_Research/02_Models/02_V4_2_PIT_Teacher_Student_Finalize.py`

Output root:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/`

Read-only source:

`pawan_quant_research_1.v4.silver_bt_master_grid_1h`

Optional read-only comparison:

`pawan_quant_research_1.v4.bt_walkforward_regime_signals`

## Production Guardrails

The script does not call `saveAsTable`, write Delta, register models, update aliases, edit jobs, or touch Saxo/RT/dashboard assets. It writes only CSV, JSON, XLSX, and joblib files under the research output root.

It must not overwrite:

- `pawan_quant_research_1.v4.bt_walkforward_regime_signals`
- `pawan_quant_research_1.v4.bronze_rt_regime_signal`
- `pawan_quant_research_1.v4.silver_rt_1h`
- Saxo tables
- dashboard feed tables
- model registry tables or registry aliases

## Variants

| Variant | Description | PIT status |
|---|---|---|
| A | Raw-label monthly walk-forward baseline | Strict PIT |
| B | Retrospective full-sample teacher reference | Not PIT; comparison only |
| C | PIT-safe teacher-student with fold-local teacher | Strict PIT |
| D | Conservative hybrid requiring A/C agreement for full allocation | Strict PIT derived signal |

For variant C, each monthly fold trains the teacher only on rows strictly before the fold start. The teacher generates pseudo-labels only for that historical fold training window. The student trains on those fold-safe pseudo-labels and predicts the next month.

## Performance Methodology

The script uses the corrected-reference methodology requested for this experiment:

- prior-bar / next-bar allocation
- price-update hourly rows only
- 5 bps switch cost
- daily-last equity reporting
- CAGR over exactly 5.0 years from `2021-01-01` through `2025-12-31`
- Sharpe/Sortino on daily returns with 252 annualization and risk-free 0

Regime allocation mapping:

- `0` Expansion: QQQ
- `1` Slowdown: SPY
- `2` Recession: cash

The conservative hybrid uses the more defensive of raw-label and PIT teacher-student regimes when they disagree and applies 50% allocation in disagreement rows.

## Outputs

Expected output families:

- `signals/*_signals.csv`
- `models/<variant>/.../*.joblib`
- `manifests/fold_manifest.csv`
- `manifests/fold_manifest.json`
- `manifests/runtime_manifest.json`
- `performance/<variant>/equity_curve_hourly_price_updates.csv`
- `performance/<variant>/equity_curve_daily_last.csv`
- `performance/<variant>/metrics.json`
- `performance/<variant>/annual_returns.csv`
- `performance/<variant>/trailing_metrics.csv`
- `performance/<variant>/slippage_sensitivity.csv`
- `performance/variant_summary.csv`
- `performance/scaling_readiness_diagnostics.csv`
- `performance/audit_pack.xlsx`
- `forensics/pit_teacher_student_research_summary.md`

## Governance Verdicts

The script assigns one of:

- `research_only`
- `shadow_candidate`
- `eligible_for_100k`
- `not_eligible`

The retrospective full-sample teacher reference is always `research_only` because it is not strict PIT. No output from this script is promoted automatically.

## Run Pattern

Run on Databricks as a one-off research job or notebook command on a cluster with read access to `pawan_quant_research_1.v4.silver_bt_master_grid_1h`.

Do not attach this script to a production workflow. Do not add model registry or Delta write cells.

## 2026-05-04 Research Run Result

The research run completed in the isolated output root. The finalizer writes only summary artifacts from already-produced research CSVs and does not train models.

| Variant | CAGR | Sharpe | Sortino | Max DD | Final Equity | Switch Count | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_raw_label_monthly_wf | 6.8925% | 0.4733 | 0.5834 | -38.4318% | $6,977,589.08 | 1058 | shadow_candidate |
| B_retrospective_full_sample_teacher_reference | 6.0667% | 0.4290 | 0.5483 | -40.1091% | $6,712,199.89 | 904 | research_only |
| C_pit_safe_teacher_student | 6.4963% | 0.4542 | 0.5835 | -38.0591% | $6,849,254.53 | 924 | shadow_candidate |
| D_conservative_hybrid | 5.3296% | 0.4009 | 0.4962 | -36.1099% | $6,482,197.35 | 1259 | research_only |
| production_v4_read_only_reference | 5.9886% | 0.4141 | 0.5146 | -39.7854% | $6,687,537.12 | 1002 | read_only_reference |
| SPY_benchmark_same_price_update_rows | 12.7658% | 0.8431 | 1.1842 | -25.2094% | $9,117,097.14 | 0 | benchmark |

The PIT-safe teacher-student variant is a `shadow_candidate`, not a promotion candidate. It did not beat SPY over the requested 2021-2025 performance window under the corrected reference performance methodology.

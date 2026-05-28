# 2026-05-04 Session 01 - C Teacher Package and TFT Next Leg

## Request

Create a research-only package for Variant C: newly trained full-sample teacher on the faithful legacy master grid. Then draft the next research plan for an independent TFT model with no XGBoost or teacher-label dependency.

## Scope

No production assets may be modified:

- no production Delta writes
- no Saxo execution changes
- no token refresh changes
- no RT notebook changes
- no dashboard changes
- no workflow or schedule changes
- no model registry writes

## Variant C Package

Output root:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_c_full_sample_teacher_package_2026-05-04/`

Input grid:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/full_legacy_grid_clone_2026-04-30/grids/full_legacy_grid_clone.csv`

Script:

`03_Research/02_Models/04_V4_2_C_Full_Sample_Teacher_Package.py`

The package uses the validated same-data Variant C artifacts from:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_same_data_teacher_student_2026-05-04/`

The purpose is to package the already-trained full-sample teacher/student path, recompute performance artifacts, and preserve code/model/signal manifests under an isolated root.

## Required Outputs

- code snapshot
- runtime manifest
- input hashes
- teacher label distribution
- full signal CSV
- walk-forward/student signal CSV
- trades/equity curve CSV
- daily, weekly, monthly, yearly returns
- Excel audit pack
- summary markdown
- README
- comparison against B recovered control, D strict PIT teacher-student, and SPY

## Variant C Governance

Variant C uses teacher-smoothed regime labels rather than directly training monthly students on hardcoded `+/-0.5%` next-week return labels. It is still full-sample/non-PIT because the teacher was trained across the full available historical sample.

This package is research-only and is not a production promotion artifact.

## TFT Next Leg

Plan file:

`03_Research/02_Models/README_V4_2_TFT_Next_Leg_Plan.md`

The TFT next leg should be independent:

- no XGBoost dependency
- no teacher-label dependency
- strict rolling validation
- compare against SPY, B, C, D, and production V4 read-only reference

## Pending Execution Fields

Fill after Databricks package run:

- package run id: `1019752531416219`
- package task run id: `396198615747320`
- package output confirmation: completed under `/Volumes/pawan_quant_research_1/v4/market_data/v4_2_c_full_sample_teacher_package_2026-05-04/`
- final C metrics:
  - CAGR: `14.613829906604114%`
  - Sharpe: `0.803388192149144`
  - Sortino: `1.0186063028886188`
  - Max DD: `-24.127472430194552%`
  - Final equity: `$9,889,062.68`
  - Switch count: `906`
  - Teacher label distribution: `{'0': 43679, '1': 19242, '2': 26143}`
- validation status:
  - package script compiled with `py_compile`
  - safety scan found no production write APIs
  - output root contains code, manifests, signals, models, performance artifacts, forensics, README, and audit pack
  - runtime manifest reports `production_delta_writes=false`, `model_registry_writes=false`, `saxo_or_rt_writes=false`, `dashboard_writes=false`, and `workflow_or_schedule_changes=false`

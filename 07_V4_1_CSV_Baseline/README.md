# V4.1 CSV-Only Research/Audit Baseline

This folder is an isolated local scaffold for the V4.1 CSV-only baseline. The scripts default to writing all runtime artifacts under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/`

They do not write to production Delta tables, Saxo execution paths, token refresh state, live inference notebooks, RT notebooks, dashboard production notebooks, Databricks workflows, or the production model registry.

## Phase-0 State Captured Locally

- Local branch: `feat/live-macro-phase1-pce-gdp-overrides`
- Latest commit: `b87af93 (HEAD -> feat/live-macro-phase1-pce-gdp-overrides, origin/feat/live-macro-phase1-pce-gdp-overrides) docs: record live macro semantic guard`
- Pre-existing local modifications were present before this scaffold:
  - `.gitignore`
  - `05_Context/04_Codex_Project_Memory.md`
  - `model_feature_upstream_lineage.csv`
  - `session notes/2026-04-30_session-01.md`
- Databricks CLI version: `0.295.0`
- Databricks profiles existed but were not authenticated locally. `databricks repos get --profile pawan-quant-1 ...` failed with an invalid refresh token.
- Local `/Volumes/pawan_quant_research_1/...` was not mounted in this shell.

## Run Inside Databricks

Run on an existing Databricks cluster with access to `/Volumes/pawan_quant_research_1/v4/market_data/` and read access to `pawan_quant_research_1.v4.bronze_bt_market_1h`.

```bash
python /Workspace/Repos/.../07_V4_1_CSV_Baseline/code/v4_1_run_all.py --fred-api-key "$FRED_API_KEY"
```

Or run phase by phase:

```bash
python 07_V4_1_CSV_Baseline/code/v4_1_setup_inputs_from_sources.py --fred-api-key "$FRED_API_KEY"
python 07_V4_1_CSV_Baseline/code/v4_1_build_grid_from_csv.py
python 07_V4_1_CSV_Baseline/code/v4_1_train_walkforward_from_csv.py
python 07_V4_1_CSV_Baseline/code/v4_1_backtest_performance_from_csv.py
python 07_V4_1_CSV_Baseline/code/v4_1_write_reports.py
```

## Main Scripts

- `code/v4_1_setup_inputs_from_sources.py`: creates V4.1 folders, timing CSVs, frozen market CSV, ALFRED first-release macro CSV, VIX/VIX3M PIT CSV, and input hash manifest.
- `code/v4_1_build_grid_from_csv.py`: builds one schema-compatible silver-grid CSV per timing variant from frozen CSVs only.
- `code/v4_1_train_walkforward_from_csv.py`: trains monthly expanding-window XGBoost models per variant from grid CSVs only.
- `code/v4_1_backtest_performance_from_csv.py`: writes equity curves, annual/trailing metrics, audit packs, and timing sensitivity outputs.
- `code/v4_1_write_reports.py`: writes `v4_1_rt_gap_analysis.md`, `v4_1_variant_comparison.xlsx`, and `v4_1_build_summary.md`.

## Safety Scope

The code contains path guards for writes under the V4.1 root and does not call `saveAsTable` or workflow APIs. The only production read currently implemented is the approved one-time market export from:

`pawan_quant_research_1.v4.bronze_bt_market_1h`

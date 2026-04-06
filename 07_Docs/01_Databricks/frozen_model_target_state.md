# Frozen-Model Target State

## Goal

Freeze the V4 model as of 2026-04-06 and move the repository to a single clean Unity Catalog schema: `pawan_quant_research_1.v4`.

This keeps the cutover simple:

- historical V4 training data lives in `v4`
- frozen model artifacts live in `/Volumes/pawan_quant_research_1/v4/market_data`
- live RT Databricks tables also live in `v4`

## Proposed Unity Catalog Layout

| Namespace | Purpose | Proposed path root |
|---|---|---|
| `pawan_quant_research_1.v4` | Frozen V4 historical data, artifacts, and RT serving tables | `/Volumes/pawan_quant_research_1/v4/market_data` |

## Proposed Object Strategy

Keep the existing table names and move them into `pawan_quant_research_1.v4`.

### Historical / Batch Objects

- `bronze_bt_market_1h`
- `bronze_bt_macro_events`
- `bronze_bt_rates_1h`
- `bronze_bt_vix_1h`
- `bronze_bt_vix3m_1h`
- `silver_bt_master_grid_1h`
- `bt_walkforward_regime_signals`
- `gold_bt_returns_*`

### Real-Time Serving Objects

- `bronze_rt_market_1h`
- `bronze_rt_vix_1d`
- `silver_rt_1h`
- `bronze_rt_regime_signal`
- `bronze_rt_paper_trade_log`
- `bronze_rt_signal_audit_log`
- `gold_rt_dashboard_feed`

### Frozen Volume Artifacts

- `bt_fred_1d_xgboost_v4_frozen_2026_04_06.joblib`
- `bt_fred_1d_regime_signals-v4.csv`
- walk-forward CSV outputs
- performance exports and deck-ready files

## Migration Rules

| Current object | Target object | Action |
|---|---|---|
| `default.bronze_bt_*` | `v4.bronze_bt_*` | Copy full historical data |
| `default.silver_bt_master_grid_1h` | `v4.silver_bt_master_grid_1h` | Copy full historical data |
| `default.bt_walkforward_regime_signals` | `v4.bt_walkforward_regime_signals` | Copy full historical data |
| `/Volumes/pawan_quant_research_1/default/market_data/*` | `/Volumes/pawan_quant_research_1/v4/market_data/*` | Copy frozen artifacts and exports |
| `default.bronze_rt_*`, `default.silver_rt_1h`, `default.gold_rt_dashboard_feed` | `v4.*` | Recreate empty RT tables, then repoint jobs |

## Repository Rule

The repo should now treat `pawan_quant_research_1.v4` as the canonical schema. New work should not write to `pawan_quant_research_1.default`.

## Recommended Order

1. Create volume `pawan_quant_research_1.v4.market_data`.
2. Copy frozen model and signal artifacts from `default` volume to `v4` volume.
3. Copy BT tables from `default` into `v4`.
4. Recreate clean RT tables in `v4`.
5. Run the repointed notebooks from this branch.
6. Version the actual Databricks workflow JSON in the repo.

## Cautions

- One schema is simpler, but naming discipline matters more.
- RT silver still carries forward some historical silver-derived values.
- External raw-source paths under `pawan_quant_research/default/...` were left unchanged in the code because those are upstream source locations, not the V4 target schema.

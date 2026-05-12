# 2026-05-12 Session 01: Saxo V4 Strategy Ledger

## Scope

- Added read-only broker-native Saxo history ingestion in `01_Processing/03_Realtime/11_RT_Saxo_Broker_Trade_History.py`.
- Added strategy-only V4 Saxo attribution and reporting in `01_Processing/03_Realtime/12_RT_Saxo_V4_Strategy_Ledger.py`.
- No live execution code was changed.
- No workflows were scheduled or modified.

## Broker-Native History Ingestion

- New target table: `pawan_quant_research_1.v4.bronze_rt_saxo_broker_trade_history`.
- The notebook reads the existing Saxo token state file and never refreshes or writes token state.
- It uses GET-only Saxo OpenAPI calls and does not place, modify, precheck, or cancel orders.
- Populated broker rows from:
  - `cs_audit_orderactivities`
  - `hist_transactions`
  - `cs_reports_trades`

## Strategy-Only Attribution

- V4 strategy attribution is now built from broker-native Saxo history when available.
- Strategy fills are restricted to `cs_reports_trades` rows for SPY and QQQ.
- Saxo account activity outside V4 SPY/QQQ strategy fills remains excluded from V4 strategy P&L.
- The latest validated reconstructed stance was `QQQ`, matching latest live `target_ticker=QQQ` and `target_position=100% QQQ`.

## Cost Framing

- Captured cost is broker-reported conversion cost from Saxo `ConversionCost` / `ConversionRate` fields.
- This is not confirmed all-in transaction cost.
- Saxo payloads did not populate commission, fee, or slippage fields in the validated run.
- Ledger rows explicitly mark:
  - `cost_observation_quality = broker_confirmed_conversion_cost` where conversion cost is attached.
  - `all_in_cost_confirmed = false`.
  - `cost_notes = 'Saxo payload populated ConversionCost/ConversionRate; commission/fee fields not populated.'`

## Validated Results

- Manual Databricks run id for updated ledger validation: `3618230298009`.
- `gold_saxo_v4_strategy_trade_ledger`: 181 rows.
- `gold_saxo_v4_strategy_equity_curve`: 241 rows.
- `gold_saxo_v4_transaction_cost_analysis`: 16 rows.
- V4 strategy trade rows: 23.
- All strategy trades came from `broker_native_saxo_history` / `cs_reports_trades`.
- Estimated cost rows: 0.
- Positive-cost V4 strategy rows: 23.
- `all_in_cost_confirmed=true` rows: 0.
- Total captured broker-reported conversion cost: 241.08737584.

## Account-Size Implication

- The 5k trial is materially cost-distorted by observed broker-reported conversion cost.
- Larger account sizes reduce observed cost drag materially:
  - At 5k, captured conversion cost was about 4.82% of account.
  - At 100k, captured conversion cost was about 0.24% of account.
  - At 500k, captured conversion cost was about 0.05% of account.

## Commit Plan

- Stage only:
  - `01_Processing/03_Realtime/11_RT_Saxo_Broker_Trade_History.py`
  - `01_Processing/03_Realtime/12_RT_Saxo_V4_Strategy_Ledger.py`
  - `session notes/2026-05-12_session-01_saxo_v4_strategy_ledger.md`
- Do not stage unrelated untracked files such as `WORKFLOW.md` or `databricks.yml`.
- Commit message: `analysis: add broker-native Saxo V4 strategy ledger`.
- Do not push until reviewed.

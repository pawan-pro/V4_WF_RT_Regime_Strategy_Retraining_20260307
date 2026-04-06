-- Bootstrap script for migrating the repo from pawan_quant_research_1.default
-- to pawan_quant_research_1.v4 after the V4 model freeze on 2026-04-06.

CREATE SCHEMA IF NOT EXISTS pawan_quant_research_1.v4;
CREATE VOLUME IF NOT EXISTS pawan_quant_research_1.v4.market_data;

-- ---------------------------------------------------------------------------
-- Historical objects: copy from default into v4
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_bt_market_1h AS
SELECT * FROM pawan_quant_research_1.default.bronze_bt_market_1h;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_bt_macro_events AS
SELECT * FROM pawan_quant_research_1.default.bronze_bt_macro_events;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_bt_rates_1h AS
SELECT * FROM pawan_quant_research_1.default.bronze_bt_rates_1h;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_bt_vix_1h AS
SELECT * FROM pawan_quant_research_1.default.bronze_bt_vix_1h;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_bt_vix3m_1h AS
SELECT * FROM pawan_quant_research_1.default.bronze_bt_vix3m_1h;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.silver_bt_master_grid_1h AS
SELECT * FROM pawan_quant_research_1.default.silver_bt_master_grid_1h;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bt_walkforward_regime_signals AS
SELECT * FROM pawan_quant_research_1.default.bt_walkforward_regime_signals;

-- ---------------------------------------------------------------------------
-- Real-time objects: create clean empty tables in v4
-- ---------------------------------------------------------------------------

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_rt_market_1h AS
SELECT * FROM pawan_quant_research_1.default.bronze_rt_market_1h WHERE 1 = 0;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_rt_vix_1d AS
SELECT * FROM pawan_quant_research_1.default.bronze_rt_vix_1d WHERE 1 = 0;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.silver_rt_1h AS
SELECT * FROM pawan_quant_research_1.default.silver_rt_1h WHERE 1 = 0;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_rt_regime_signal AS
SELECT * FROM pawan_quant_research_1.default.bronze_rt_regime_signal WHERE 1 = 0;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_rt_paper_trade_log AS
SELECT * FROM pawan_quant_research_1.default.bronze_rt_paper_trade_log WHERE 1 = 0;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.bronze_rt_signal_audit_log AS
SELECT * FROM pawan_quant_research_1.default.bronze_rt_signal_audit_log WHERE 1 = 0;

CREATE OR REPLACE TABLE pawan_quant_research_1.v4.gold_rt_dashboard_feed AS
SELECT * FROM pawan_quant_research_1.default.gold_rt_dashboard_feed WHERE 1 = 0;

-- Optional one-time seed if you want v4 RT tables to start with the latest
-- snapshot instead of a clean slate.
--
-- INSERT INTO pawan_quant_research_1.v4.bronze_rt_market_1h
-- SELECT * FROM pawan_quant_research_1.default.bronze_rt_market_1h;
--
-- INSERT INTO pawan_quant_research_1.v4.bronze_rt_vix_1d
-- SELECT * FROM pawan_quant_research_1.default.bronze_rt_vix_1d;

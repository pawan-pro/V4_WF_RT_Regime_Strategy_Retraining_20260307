"""Frozen-model Databricks manifest for the V4 schema migration."""

FREEZE_DATE = "2026-04-06"

CATALOG = "pawan_quant_research_1"
SOURCE_SCHEMA = "default"
TARGET_SCHEMA = "v4"
VOLUME = "market_data"

SOURCE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SOURCE_SCHEMA}/{VOLUME}"
TARGET_VOLUME_PATH = f"/Volumes/{CATALOG}/{TARGET_SCHEMA}/{VOLUME}"

MODEL_VERSION = "bt_fred_1d_xgboost_v4_frozen_2026_04_06"
MODEL_ARTIFACT_PATH = f"{TARGET_VOLUME_PATH}/{MODEL_VERSION}.joblib"
SIGNAL_EXPORT_PATH = f"{TARGET_VOLUME_PATH}/bt_fred_1d_regime_signals-v4.csv"
WALKFORWARD_EXPORT_PATH = f"{TARGET_VOLUME_PATH}/bt_walkforward_regime_signals.csv"

SOURCE_TABLES = {
    "bronze_bt_market_1h": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_bt_market_1h",
    "bronze_bt_macro_events": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_bt_macro_events",
    "bronze_bt_rates_1h": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_bt_rates_1h",
    "bronze_bt_vix_1h": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_bt_vix_1h",
    "bronze_bt_vix3m_1h": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_bt_vix3m_1h",
    "silver_bt_master_grid_1h": f"{CATALOG}.{SOURCE_SCHEMA}.silver_bt_master_grid_1h",
    "bt_walkforward_regime_signals": (
        f"{CATALOG}.{SOURCE_SCHEMA}.bt_walkforward_regime_signals"
    ),
    "bronze_rt_market_1h": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_rt_market_1h",
    "bronze_rt_vix_1d": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_rt_vix_1d",
    "silver_rt_1h": f"{CATALOG}.{SOURCE_SCHEMA}.silver_rt_1h",
    "bronze_rt_regime_signal": f"{CATALOG}.{SOURCE_SCHEMA}.bronze_rt_regime_signal",
    "bronze_rt_paper_trade_log": (
        f"{CATALOG}.{SOURCE_SCHEMA}.bronze_rt_paper_trade_log"
    ),
    "bronze_rt_signal_audit_log": (
        f"{CATALOG}.{SOURCE_SCHEMA}.bronze_rt_signal_audit_log"
    ),
    "gold_rt_dashboard_feed": f"{CATALOG}.{SOURCE_SCHEMA}.gold_rt_dashboard_feed",
}

TARGET_TABLES = {
    "bronze_bt_market_1h": f"{CATALOG}.{TARGET_SCHEMA}.bronze_bt_market_1h",
    "bronze_bt_macro_events": f"{CATALOG}.{TARGET_SCHEMA}.bronze_bt_macro_events",
    "bronze_bt_rates_1h": f"{CATALOG}.{TARGET_SCHEMA}.bronze_bt_rates_1h",
    "bronze_bt_vix_1h": f"{CATALOG}.{TARGET_SCHEMA}.bronze_bt_vix_1h",
    "bronze_bt_vix3m_1h": f"{CATALOG}.{TARGET_SCHEMA}.bronze_bt_vix3m_1h",
    "silver_bt_master_grid_1h": f"{CATALOG}.{TARGET_SCHEMA}.silver_bt_master_grid_1h",
    "bt_walkforward_regime_signals": (
        f"{CATALOG}.{TARGET_SCHEMA}.bt_walkforward_regime_signals"
    ),
    "bronze_rt_market_1h": f"{CATALOG}.{TARGET_SCHEMA}.bronze_rt_market_1h",
    "bronze_rt_vix_1d": f"{CATALOG}.{TARGET_SCHEMA}.bronze_rt_vix_1d",
    "silver_rt_1h": f"{CATALOG}.{TARGET_SCHEMA}.silver_rt_1h",
    "bronze_rt_regime_signal": f"{CATALOG}.{TARGET_SCHEMA}.bronze_rt_regime_signal",
    "bronze_rt_paper_trade_log": (
        f"{CATALOG}.{TARGET_SCHEMA}.bronze_rt_paper_trade_log"
    ),
    "bronze_rt_signal_audit_log": (
        f"{CATALOG}.{TARGET_SCHEMA}.bronze_rt_signal_audit_log"
    ),
    "gold_rt_dashboard_feed": f"{CATALOG}.{TARGET_SCHEMA}.gold_rt_dashboard_feed",
}

REBUILD_ACTIONS = {
    "volume": "create market_data volume in pawan_quant_research_1.v4",
    "historical": "copy historical BT tables from default into v4",
    "artifacts": "copy frozen model and signal exports into /Volumes/pawan_quant_research_1/v4/market_data",
    "realtime": "create empty RT tables in v4, then repoint Databricks jobs",
}

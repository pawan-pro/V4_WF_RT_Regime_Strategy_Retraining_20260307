# Databricks notebook source
"""Copy frozen V4 artifacts from default into the new v4 schema volume."""

FREEZE_DATE = "2026-04-06"
CATALOG = "pawan_quant_research_1"
SOURCE_SCHEMA = "default"
TARGET_SCHEMA = "v4"
VOLUME = "market_data"

SOURCE_BASE = f"/Volumes/{CATALOG}/{SOURCE_SCHEMA}/{VOLUME}"
TARGET_BASE = f"/Volumes/{CATALOG}/{TARGET_SCHEMA}/{VOLUME}"

SOURCE_MODEL = f"{SOURCE_BASE}/bt_fred_1d_xgboost_regime.joblib"
TARGET_MODEL = f"{TARGET_BASE}/bt_fred_1d_xgboost_v4_frozen_{FREEZE_DATE}.joblib"

SOURCE_SIGNAL = f"{SOURCE_BASE}/bt_fred_1d_regime_signals-v4.csv"
TARGET_SIGNAL = f"{TARGET_BASE}/bt_fred_1d_regime_signals-v4.csv"

dbutils.fs.cp(SOURCE_MODEL, TARGET_MODEL)
dbutils.fs.cp(SOURCE_SIGNAL, TARGET_SIGNAL)

print(f"Copied model to: {TARGET_MODEL}")
print(f"Copied signal export to: {TARGET_SIGNAL}")

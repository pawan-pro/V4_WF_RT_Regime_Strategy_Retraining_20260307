import json
from datetime import datetime, timezone

import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
SUFFIX = "pre_corrected_alfred_promotion_20260430"
PROMOTION_TAG = "corrected_alfred_baseline_20260430"
MARKET_DATA_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
REPORT_ROOT = f"{MARKET_DATA_ROOT}/reports"
BACKUP_ROOT = f"{MARKET_DATA_ROOT}/backups/{SUFFIX}"

CURRENT_SILVER = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
CURRENT_SIGNALS = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
CURRENT_REGISTRY = f"{CATALOG}.{SCHEMA}.model_registry_current"

CORRECTED_SILVER = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_recreated_corrected"
CORRECTED_SIGNALS = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_recreated_corrected"

PROMOTED_SILVER_VERSIONED = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_{PROMOTION_TAG}"
PROMOTED_SIGNALS_VERSIONED = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_{PROMOTION_TAG}"

CANONICAL_WF_CSV = f"{MARKET_DATA_ROOT}/bt_walkforward_regime_signals.csv"
CANONICAL_FULL_SIGNAL_CSV = f"{MARKET_DATA_ROOT}/bt_fred_1d_regime_signals-v4.csv"
CORRECTED_WF_CSV = (
    f"{MARKET_DATA_ROOT}/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "corrected_variant/walkforward_signals.csv"
)
CORRECTED_FULL_SIGNAL_CSV = (
    f"{MARKET_DATA_ROOT}/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "corrected_variant/full_signal_export.csv"
)

BACKUP_SILVER = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_{SUFFIX}"
BACKUP_SIGNALS = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_{SUFFIX}"
BACKUP_REGISTRY = f"{CATALOG}.{SCHEMA}.model_registry_current_{SUFFIX}"
BACKUP_WF_CSV = f"{BACKUP_ROOT}/bt_walkforward_regime_signals_{SUFFIX}.csv"
BACKUP_FULL_SIGNAL_CSV = f"{BACKUP_ROOT}/bt_fred_1d_regime_signals-v4_{SUFFIX}.csv"
BACKUP_REGISTRY_JSON = f"{BACKUP_ROOT}/model_registry_current_{SUFFIX}.json"
PROMOTION_MANIFEST = f"{REPORT_ROOT}/v4_corrected_alfred_promotion_manifest_2026-04-30.json"


def table_exists(name: str) -> bool:
    return spark.catalog.tableExists(name)


def table_columns(name: str):
    return [(f.name, f.dataType.simpleString(), f.nullable) for f in spark.table(name).schema.fields]


def table_row_range(name: str, ts_col: str = "timestamp") -> dict:
    row = (
        spark.sql(
            f"""
            SELECT COUNT(*) AS row_count,
                   MIN({ts_col}) AS min_ts,
                   MAX({ts_col}) AS max_ts
            FROM {name}
            """
        )
        .toPandas()
        .iloc[0]
    )
    return {
        "table": name,
        "row_count": int(row["row_count"]),
        "min_ts": str(row["min_ts"]) if pd.notna(row["min_ts"]) else None,
        "max_ts": str(row["max_ts"]) if pd.notna(row["max_ts"]) else None,
    }


def ensure_parent(path: str):
    parent = path.rsplit("/", 1)[0]
    dbutils.fs.mkdirs(parent)


def copy_file(src: str, dst: str):
    ensure_parent(dst)
    dbutils.fs.cp(src, dst, True)


def file_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def write_json(path: str, payload: dict):
    ensure_parent(path)
    dbutils.fs.put(path, json.dumps(payload, indent=2), True)


def main():
    required_tables = [CURRENT_SILVER, CURRENT_SIGNALS, CURRENT_REGISTRY, CORRECTED_SILVER, CORRECTED_SIGNALS]
    for table_name in required_tables:
        if not table_exists(table_name):
            raise ValueError(f"Required table missing: {table_name}")

    required_files = [CANONICAL_WF_CSV, CANONICAL_FULL_SIGNAL_CSV, CORRECTED_WF_CSV, CORRECTED_FULL_SIGNAL_CSV]
    for file_name in required_files:
        if not file_exists(file_name):
            raise ValueError(f"Required file missing: {file_name}")

    current_signal_schema = table_columns(CURRENT_SIGNALS)
    corrected_signal_schema = table_columns(CORRECTED_SIGNALS)
    current_silver_schema = table_columns(CURRENT_SILVER)
    corrected_silver_schema = table_columns(CORRECTED_SILVER)

    silver_schema_match = current_silver_schema == corrected_silver_schema

    corrected_wf_pdf = spark.read.option("header", True).csv(CORRECTED_WF_CSV).toPandas()
    corrected_full_signal_pdf = spark.read.option("header", True).csv(CORRECTED_FULL_SIGNAL_CSV).toPandas()
    corrected_wf_pdf["timestamp"] = pd.to_datetime(corrected_wf_pdf["timestamp"])
    corrected_full_signal_pdf["timestamp"] = pd.to_datetime(corrected_full_signal_pdf["timestamp"])
    corrected_wf_pdf["predicted_regime"] = pd.to_numeric(corrected_wf_pdf["predicted_regime"], errors="coerce").astype("Int64")
    corrected_full_signal_pdf["label"] = pd.to_numeric(corrected_full_signal_pdf["label"], errors="coerce").astype("Int64")

    canonical_signal_pdf = corrected_wf_pdf.merge(
        corrected_full_signal_pdf[["timestamp", "label"]],
        on="timestamp",
        how="left",
    )
    if canonical_signal_pdf["label"].isna().any():
        raise ValueError("Corrected full_signal_export is missing labels for some walkforward timestamps")
    canonical_signal_pdf = canonical_signal_pdf[["timestamp", "label", "predicted_regime"]].rename(
        columns={"label": "regime"}
    )
    canonical_signal_pdf["regime"] = canonical_signal_pdf["regime"].astype(int)
    canonical_signal_pdf["predicted_regime"] = canonical_signal_pdf["predicted_regime"].astype(int)
    canonical_signal_pdf = canonical_signal_pdf.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    canonical_signal_columns = [(str(col),) for col in canonical_signal_pdf.columns.tolist()]
    current_signal_column_names = [name for name, _, _ in current_signal_schema]
    signal_schema_match = current_signal_column_names == canonical_signal_pdf.columns.tolist()
    if not signal_schema_match:
        raise ValueError(
            f"Canonical walkforward compatibility frame columns {canonical_signal_pdf.columns.tolist()} "
            f"do not match current signal columns {current_signal_column_names}"
        )

    registry_pdf = spark.table(CURRENT_REGISTRY).toPandas()
    if registry_pdf.empty:
        raise ValueError("model_registry_current is empty")
    registry_row = registry_pdf.iloc[0].to_dict()
    model_version = registry_row.get("model_version")
    versioned_model_path = registry_row.get("versioned_model_path")
    live_model_path = registry_row.get("live_model_path")
    signal_path = registry_row.get("signal_path")
    wf_signal_table = registry_row.get("wf_signal_table")

    model_artifact_confirmed = bool(model_version and versioned_model_path)
    if model_artifact_confirmed:
        try:
            dbutils.fs.ls(str(versioned_model_path).rsplit("/", 1)[0])
        except Exception:
            model_artifact_confirmed = False

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "backup_suffix": SUFFIX,
        "current_state": {
            "current_silver": table_row_range(CURRENT_SILVER),
            "current_signals": table_row_range(CURRENT_SIGNALS),
            "current_registry_rows": int(len(registry_pdf)),
            "current_registry_row": {k: (None if pd.isna(v) else str(v)) for k, v in registry_row.items()},
        },
        "corrected_state": {
            "corrected_silver": table_row_range(CORRECTED_SILVER),
            "corrected_signals": table_row_range(CORRECTED_SIGNALS),
        },
        "schema_checks": {
            "signal_schema_match": signal_schema_match,
            "current_signal_schema": current_signal_schema,
            "corrected_signal_schema": corrected_signal_schema,
            "canonical_signal_compatibility_columns": canonical_signal_pdf.columns.tolist(),
            "silver_schema_match": silver_schema_match,
            "silver_missing_from_corrected": [name for name, _, _ in current_silver_schema if name not in {c[0] for c in corrected_silver_schema}],
            "silver_extra_in_corrected": [name for name, _, _ in corrected_silver_schema if name not in {c[0] for c in current_silver_schema}],
        },
        "backups": {},
        "promotions": {},
        "model_registry_decision": {},
    }

    spark.sql(f"CREATE OR REPLACE TABLE {BACKUP_SILVER} AS SELECT * FROM {CURRENT_SILVER}")
    spark.sql(f"CREATE OR REPLACE TABLE {BACKUP_SIGNALS} AS SELECT * FROM {CURRENT_SIGNALS}")
    spark.sql(f"CREATE OR REPLACE TABLE {BACKUP_REGISTRY} AS SELECT * FROM {CURRENT_REGISTRY}")
    copy_file(CANONICAL_WF_CSV, BACKUP_WF_CSV)
    copy_file(CANONICAL_FULL_SIGNAL_CSV, BACKUP_FULL_SIGNAL_CSV)
    write_json(BACKUP_REGISTRY_JSON, {k: (None if pd.isna(v) else str(v)) for k, v in registry_row.items()})

    manifest["backups"] = {
        "silver_table": BACKUP_SILVER,
        "signals_table": BACKUP_SIGNALS,
        "registry_table": BACKUP_REGISTRY,
        "walkforward_csv": BACKUP_WF_CSV,
        "full_signal_csv": BACKUP_FULL_SIGNAL_CSV,
        "registry_json": BACKUP_REGISTRY_JSON,
    }

    spark.sql(f"CREATE OR REPLACE TABLE {PROMOTED_SILVER_VERSIONED} AS SELECT * FROM {CORRECTED_SILVER}")
    spark.sql(f"CREATE OR REPLACE TABLE {PROMOTED_SIGNALS_VERSIONED} AS SELECT * FROM {CORRECTED_SIGNALS}")

    spark.createDataFrame(canonical_signal_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        CURRENT_SIGNALS
    )
    canonical_signal_pdf.to_csv("/tmp/canonical_promoted_bt_walkforward_regime_signals.csv", index=False)
    dbutils.fs.cp("file:/tmp/canonical_promoted_bt_walkforward_regime_signals.csv", CANONICAL_WF_CSV, True)
    copy_file(CORRECTED_FULL_SIGNAL_CSV, CANONICAL_FULL_SIGNAL_CSV)

    manifest["promotions"] = {
        "promoted_silver_versioned_table": PROMOTED_SILVER_VERSIONED,
        "promoted_signals_versioned_table": PROMOTED_SIGNALS_VERSIONED,
        "canonical_signal_table_overwritten": CURRENT_SIGNALS,
        "canonical_signal_table_row_count": int(len(canonical_signal_pdf)),
        "canonical_walkforward_csv_overwritten": CANONICAL_WF_CSV,
        "canonical_walkforward_csv_method": "corrected_walkforward_predictions joined with corrected full_signal_export labels",
        "canonical_full_signal_csv_overwritten": CANONICAL_FULL_SIGNAL_CSV,
        "canonical_silver_alias_overwritten": False,
    }

    if model_artifact_confirmed:
        manifest["model_registry_decision"] = {
            "action": "no_change",
            "reason": (
                "Current registry already points to a confirmed versioned model artifact; "
                "signal/table aliases changed underneath the existing canonical paths."
            ),
            "model_version": str(model_version),
            "versioned_model_path": str(versioned_model_path),
            "live_model_path": str(live_model_path),
            "signal_path": str(signal_path),
            "wf_signal_table": str(wf_signal_table),
        }
    else:
        manifest["model_registry_decision"] = {
            "action": "no_change",
            "reason": "Did not confirm a corrected model artifact/version path strongly enough to edit model_registry_current.",
            "model_version": None if pd.isna(model_version) else str(model_version),
            "versioned_model_path": None if pd.isna(versioned_model_path) else str(versioned_model_path),
        }

    write_json(PROMOTION_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


main()

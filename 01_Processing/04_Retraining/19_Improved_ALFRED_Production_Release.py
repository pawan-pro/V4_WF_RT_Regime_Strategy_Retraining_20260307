# Databricks notebook source
# Controlled improved-ALFRED production release helper.
# Modes:
# - snapshot: capture pre-release state only
# - validate: capture post-release state, validate, and write release artifacts

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from pyspark.sql import functions as F

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

RELEASE_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
RELEASE_ROOT_DEFAULT = (
    f"/Volumes/{CATALOG}/{SCHEMA}/market_data/comparison/"
    f"{RELEASE_DATE}_improved_alfred_production_release"
)
WORKSPACE_REPO_PATH = (
    "/Workspace/Users/pawan.jatale@mubama.group/"
    "V4_WF_RT_Regime_Strategy_Retraining_github"
)

MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"
BRONZE_MACRO_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
SILVER_BT_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
SILVER_RT_TABLE = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
REGIME_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"
LIVE_MODEL_PATH = (
    f"/Volumes/{CATALOG}/{SCHEMA}/market_data/models/"
    "bt_fred_1d_xgboost_regime_live.joblib"
)
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
KEY_INDICATORS = ["PAYEMS", "INDPRO", "CPIAUCSL", "UNRATE", "GDPC1"]
SILVER_MACRO_COLUMNS = [
    "CPIAUCSL_z",
    "PCEPI_z",
    "PCE_z",
    "PAYEMS_z",
    "UNRATE_z",
    "GDPC1_z",
    "INDPRO_z",
    "UMCSENT_z",
    "CPI_Event_Active",
    "NFP_Event_Active",
]


def parse_json_widget(raw_value, default):
    if not raw_value:
        return default
    return json.loads(raw_value)


def serialize(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def ensure_dir(path_str):
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path_str, payload):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(json.dumps(payload, indent=2, default=serialize))


def write_text(path_str, content):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(content)


def write_csv(path_str, pdf):
    ensure_dir(str(Path(path_str).parent))
    pdf.to_csv(path_str, index=False)


def file_info(path_str):
    path = Path(path_str)
    if not path.exists():
        return {
            "path": path_str,
            "exists": False,
            "size_bytes": None,
            "modified_utc": None,
        }
    stat = path.stat()
    return {
        "path": path_str,
        "exists": True,
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    }


def table_exists(table_name):
    return spark.catalog.tableExists(table_name)


def latest_registry_pdf():
    if not table_exists(MODEL_REGISTRY_TABLE):
        return pd.DataFrame()
    return spark.table(MODEL_REGISTRY_TABLE).toPandas()


def latest_registry_row():
    pdf = latest_registry_pdf()
    if pdf.empty:
        return {}
    if "promoted_ts" in pdf.columns:
        pdf["promoted_ts"] = pd.to_datetime(pdf["promoted_ts"], errors="coerce")
        pdf = pdf.sort_values("promoted_ts", ascending=False, na_position="last")
    return {key: serialize(value) for key, value in pdf.iloc[0].to_dict().items()}


def latest_row_pdf(table_name):
    if not table_exists(table_name):
        return pd.DataFrame()
    return spark.sql(
        f"""
        SELECT *
        FROM {table_name}
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).toPandas()


def write_snapshot_csv(table_name, output_path):
    pdf = latest_row_pdf(table_name)
    if pdf.empty:
        pdf = pd.DataFrame()
    write_csv(output_path, pdf)
    return pdf


def write_snapshot_json(table_name, output_path):
    pdf = latest_row_pdf(table_name)
    payload = {}
    if not pdf.empty:
        payload = {key: serialize(value) for key, value in pdf.iloc[0].to_dict().items()}
    write_json(output_path, payload)
    return payload


def write_registry_snapshot(output_path):
    pdf = latest_registry_pdf()
    write_csv(output_path, pdf if not pdf.empty else pd.DataFrame())
    return pdf


def snapshot_pre_release(release_root):
    registry_pdf = write_registry_snapshot(
        f"{release_root}/pre_release_registry_snapshot.csv"
    )
    latest_registry = (
        {}
        if registry_pdf.empty
        else {
            key: serialize(value)
            for key, value in registry_pdf.iloc[0].to_dict().items()
        }
    )
    model_metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_repo_path": WORKSPACE_REPO_PATH,
        "latest_registry_row": latest_registry,
        "live_model_alias_file": file_info(LIVE_MODEL_PATH),
    }
    write_json(
        f"{release_root}/pre_release_model_metadata.json",
        model_metadata,
    )
    latest_regime = write_snapshot_json(
        REGIME_SIGNAL_TABLE,
        f"{release_root}/pre_release_latest_regime_snapshot.json",
    )
    latest_feature = write_snapshot_csv(
        SILVER_RT_TABLE,
        f"{release_root}/pre_release_latest_feature_snapshot.csv",
    )
    result = {
        "status": "success",
        "phase": "pre_release_snapshot",
        "release_root": release_root,
        "artifacts": {
            "pre_release_model_metadata": (
                f"{release_root}/pre_release_model_metadata.json"
            ),
            "pre_release_registry_snapshot": (
                f"{release_root}/pre_release_registry_snapshot.csv"
            ),
            "pre_release_latest_regime_snapshot": (
                f"{release_root}/pre_release_latest_regime_snapshot.json"
            ),
            "pre_release_latest_feature_snapshot": (
                f"{release_root}/pre_release_latest_feature_snapshot.csv"
            ),
        },
        "latest_registry_row": latest_registry,
        "latest_regime_snapshot": latest_regime,
        "latest_feature_rows": int(len(latest_feature)),
    }
    write_json(f"{release_root}/run_metadata.json", result)
    return result


def query_fred_release_bucket(series_id, release_date):
    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "limit": 100000,
        },
        timeout=60,
    )
    response.raise_for_status()
    observations = pd.DataFrame(response.json().get("observations", []))
    if observations.empty:
        return None

    observations["realtime_start"] = pd.to_datetime(
        observations["realtime_start"],
        errors="coerce",
    ).dt.date
    observations["observation_date"] = pd.to_datetime(
        observations["date"],
        errors="coerce",
    )
    observations["value"] = pd.to_numeric(
        observations["value"],
        errors="coerce",
    )
    release_dt = pd.to_datetime(release_date).date()
    release_rows = observations[
        observations["realtime_start"] == release_dt
    ].dropna(subset=["observation_date", "value"])
    if release_rows.empty:
        return None

    max_obs = release_rows["observation_date"].max()
    selected = release_rows[
        release_rows["observation_date"] == max_obs
    ].iloc[-1]
    return {
        "expected_value": float(selected["value"]),
        "selected_observation_date": max_obs.date().isoformat(),
        "bucket_row_count": int(len(release_rows)),
    }


def validate_macro_selection():
    bronze_pdf = spark.sql(
        f"""
        WITH ranked AS (
            SELECT
                indicator_id,
                timestamp,
                actual,
                ROW_NUMBER() OVER (
                    PARTITION BY indicator_id
                    ORDER BY timestamp DESC
                ) AS rn
            FROM {BRONZE_MACRO_TABLE}
            WHERE indicator_id IN ({", ".join([f"'{x}'" for x in KEY_INDICATORS])})
        )
        SELECT indicator_id, timestamp, actual
        FROM ranked
        WHERE rn = 1
        ORDER BY indicator_id
        """
    ).toPandas()

    checks = []
    eastern = ZoneInfo("America/New_York")
    for _, row in bronze_pdf.iterrows():
        release_ts = pd.Timestamp(row["timestamp"]).tz_localize("UTC")
        release_date = release_ts.tz_convert(eastern).date().isoformat()
        check = {
            "check_name": f"macro_release_selection_{row['indicator_id']}",
            "category": "macro_release_selection",
            "indicator_id": row["indicator_id"],
            "release_timestamp_utc": serialize(row["timestamp"]),
            "release_date": release_date,
            "production_actual": float(row["actual"]),
            "status": "unknown",
            "details": "",
        }
        try:
            expected = query_fred_release_bucket(row["indicator_id"], release_date)
            if expected is None:
                check["status"] = "warning"
                check["details"] = "No ALFRED release bucket rows returned"
            else:
                check["expected_actual"] = expected["expected_value"]
                check["selected_observation_date"] = expected[
                    "selected_observation_date"
                ]
                check["bucket_row_count"] = expected["bucket_row_count"]
                is_match = abs(float(row["actual"]) - expected["expected_value"]) < 1e-9
                check["status"] = "pass" if is_match else "fail"
                check["details"] = (
                    "Production bronze row matches latest observation_date in release bucket"
                    if is_match
                    else "Production bronze row does not match latest observation_date in release bucket"
                )
        except Exception as exc:
            check["status"] = "warning"
            check["details"] = f"Validation lookup failed: {exc}"
        checks.append(check)

    return checks


def validate_silver_join_propagation():
    bronze_pdf = spark.sql(
        f"""
        WITH ranked AS (
            SELECT
                indicator_id,
                timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY indicator_id
                    ORDER BY timestamp DESC
                ) AS rn
            FROM {BRONZE_MACRO_TABLE}
            WHERE indicator_id IN ({", ".join([f"'{x}'" for x in KEY_INDICATORS])})
        )
        SELECT indicator_id, timestamp
        FROM ranked
        WHERE rn = 1
        ORDER BY indicator_id
        """
    ).toPandas()

    checks = []
    for _, row in bronze_pdf.iterrows():
        indicator = row["indicator_id"]
        feature_col = f"{indicator}_z"
        join_ts = (
            pd.Timestamp(row["timestamp"])
            .floor("H")
            + pd.Timedelta(hours=2)
        )
        silver_pdf = spark.sql(
            f"""
            SELECT timestamp, {feature_col}
            FROM {SILVER_BT_TABLE}
            WHERE timestamp = TIMESTAMP('{join_ts.strftime("%Y-%m-%d %H:%M:%S")}')
            """
        ).toPandas()
        status = "fail"
        details = "Silver PIT join row missing"
        feature_value = None
        if not silver_pdf.empty:
            feature_value = serialize(silver_pdf.iloc[0][feature_col])
            if pd.notna(silver_pdf.iloc[0][feature_col]):
                status = "pass"
                details = "Silver PIT join row exists with non-null z-score"
            else:
                details = "Silver PIT join row exists but z-score is null"
        checks.append(
            {
                "check_name": f"silver_pit_join_{indicator}",
                "category": "silver_join",
                "indicator_id": indicator,
                "release_timestamp_utc": serialize(row["timestamp"]),
                "expected_join_timestamp_utc": join_ts.isoformat(),
                "feature_column": feature_col,
                "feature_value": feature_value,
                "status": status,
                "details": details,
            }
        )
    return checks


def validate_rt_alignment():
    rt_pdf = latest_row_pdf(SILVER_RT_TABLE)
    if rt_pdf.empty:
        return [
            {
                "check_name": "rt_silver_alignment",
                "category": "rt_alignment",
                "status": "fail",
                "details": "silver_rt_1h is empty",
            }
        ]

    rt_row = rt_pdf.iloc[0]
    hist_pdf = spark.sql(
        f"""
        SELECT *
        FROM {SILVER_BT_TABLE}
        WHERE timestamp <= TIMESTAMP('{pd.Timestamp(rt_row['timestamp']).strftime("%Y-%m-%d %H:%M:%S")}')
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).toPandas()
    if hist_pdf.empty:
        return [
            {
                "check_name": "rt_silver_alignment",
                "category": "rt_alignment",
                "status": "fail",
                "details": "No matching historical silver row",
            }
        ]

    hist_row = hist_pdf.iloc[0]
    mismatches = []
    for column in SILVER_MACRO_COLUMNS:
        rt_val = rt_row.get(column)
        hist_val = hist_row.get(column)
        if pd.isna(rt_val) and pd.isna(hist_val):
            continue
        if isinstance(rt_val, (np.floating, float, int)) or isinstance(
            hist_val,
            (np.floating, float, int),
        ):
            if not np.isclose(float(rt_val), float(hist_val), atol=1e-9, rtol=0):
                mismatches.append(column)
        elif serialize(rt_val) != serialize(hist_val):
            mismatches.append(column)

    return [
        {
            "check_name": "rt_silver_alignment",
            "category": "rt_alignment",
            "status": "pass" if not mismatches else "fail",
            "details": (
                "Latest realtime silver macro carry-forward matches historical silver"
                if not mismatches
                else f"Mismatched columns: {', '.join(mismatches)}"
            ),
            "rt_timestamp": serialize(rt_row["timestamp"]),
            "hist_timestamp": serialize(hist_row["timestamp"]),
        }
    ]


def validate_model_and_signal(pre_release_metadata):
    current_registry = latest_registry_row()
    latest_regime = write_snapshot_json(
        REGIME_SIGNAL_TABLE,
        f"{release_root}/latest_regime_snapshot.json",
    )
    latest_feature = write_snapshot_csv(
        SILVER_RT_TABLE,
        f"{release_root}/latest_feature_snapshot.csv",
    )

    pre_registry = pre_release_metadata.get("latest_registry_row", {})
    promoted = bool(
        current_registry
        and current_registry.get("model_version") != pre_registry.get("model_version")
    )
    live_alias = file_info(LIVE_MODEL_PATH)

    checks = [
        {
            "check_name": "model_registry_promoted",
            "category": "model_promotion",
            "status": "pass" if promoted else "fail",
            "details": (
                f"Model version advanced from {pre_registry.get('model_version')} "
                f"to {current_registry.get('model_version')}"
                if promoted
                else "model_registry_current did not advance to a new model version"
            ),
        },
        {
            "check_name": "live_model_alias_exists",
            "category": "model_promotion",
            "status": "pass" if live_alias.get("exists") else "fail",
            "details": json.dumps(live_alias),
        },
        {
            "check_name": "latest_regime_matches_registry",
            "category": "signal_coherence",
            "status": (
                "pass"
                if latest_regime.get("model_version") == current_registry.get("model_version")
                else "fail"
            ),
            "details": (
                f"Latest regime model_version={latest_regime.get('model_version')} "
                f"registry model_version={current_registry.get('model_version')}"
            ),
        },
        {
            "check_name": "latest_feature_snapshot_present",
            "category": "signal_coherence",
            "status": "pass" if len(latest_feature) > 0 else "fail",
            "details": f"latest_feature_rows={len(latest_feature)}",
        },
    ]

    promoted_metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_registry_row": current_registry,
        "live_model_alias_file": live_alias,
        "latest_regime_snapshot": latest_regime,
    }
    write_json(
        f"{release_root}/promoted_model_metadata.json",
        promoted_metadata,
    )
    return checks, promoted_metadata


if "dbutils" in globals():
    dbutils.widgets.dropdown("mode", "snapshot", ["snapshot", "validate"])
    dbutils.widgets.text("release_root", RELEASE_ROOT_DEFAULT)
    dbutils.widgets.text("executed_runs_json", "")
    dbutils.widgets.text("repo_branch_before", "")
    dbutils.widgets.text("repo_branch_after", "")
    dbutils.widgets.text("repo_head_before", "")
    dbutils.widgets.text("repo_head_after", "")

mode = dbutils.widgets.get("mode").strip() if "dbutils" in globals() else "snapshot"
release_root = (
    dbutils.widgets.get("release_root").strip()
    if "dbutils" in globals()
    else RELEASE_ROOT_DEFAULT
)
executed_runs = (
    parse_json_widget(dbutils.widgets.get("executed_runs_json"), [])
    if "dbutils" in globals()
    else []
)
repo_branch_before = (
    dbutils.widgets.get("repo_branch_before").strip()
    if "dbutils" in globals()
    else ""
)
repo_branch_after = (
    dbutils.widgets.get("repo_branch_after").strip()
    if "dbutils" in globals()
    else ""
)
repo_head_before = (
    dbutils.widgets.get("repo_head_before").strip()
    if "dbutils" in globals()
    else ""
)
repo_head_after = (
    dbutils.widgets.get("repo_head_after").strip()
    if "dbutils" in globals()
    else ""
)

release_root = release_root or RELEASE_ROOT_DEFAULT
ensure_dir(release_root)

if mode == "snapshot":
    result = snapshot_pre_release(release_root)
elif mode == "validate":
    pre_release_model_metadata = json.loads(
        Path(f"{release_root}/pre_release_model_metadata.json").read_text()
    )
    validation_rows = []
    validation_rows.extend(validate_macro_selection())
    validation_rows.extend(validate_silver_join_propagation())
    validation_rows.extend(validate_rt_alignment())
    model_checks, promoted_metadata = validate_model_and_signal(
        pre_release_model_metadata
    )
    validation_rows.extend(model_checks)

    dashboard_runs = [
        run
        for run in executed_runs
        if run.get("label") == "dashboard_validation"
    ]
    if dashboard_runs:
        dashboard_run = dashboard_runs[-1]
        validation_rows.append(
            {
                "check_name": "dashboard_run_success",
                "category": "dashboard",
                "status": (
                    "pass"
                    if dashboard_run.get("result_state") == "SUCCESS"
                    else "fail"
                ),
                "details": json.dumps(dashboard_run),
            }
        )

    validation_pdf = pd.DataFrame(validation_rows)
    write_csv(f"{release_root}/validation_checks.csv", validation_pdf)

    changed_objects = {
        "workspace_repo_path": WORKSPACE_REPO_PATH,
        "repo_branch_before": repo_branch_before,
        "repo_branch_after": repo_branch_after,
        "repo_head_before": repo_head_before,
        "repo_head_after": repo_head_after,
        "code_files_changed": [
            "01_Processing/01_Ingestion/02_BT_FinnHub_Macro_Historical.py",
            "01_Processing/04_Retraining/19_Improved_ALFRED_Production_Release.py",
        ],
        "production_tables": [
            BRONZE_MACRO_TABLE,
            SILVER_BT_TABLE,
            MODEL_REGISTRY_TABLE,
            f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals",
            f"{CATALOG}.{SCHEMA}.gold_bt_returns_daily",
            f"{CATALOG}.{SCHEMA}.gold_bt_returns_weekly",
            f"{CATALOG}.{SCHEMA}.gold_bt_returns_monthly",
            f"{CATALOG}.{SCHEMA}.gold_bt_returns_quarterly",
            f"{CATALOG}.{SCHEMA}.gold_bt_returns_yearly",
            f"{CATALOG}.{SCHEMA}.gold_bt_performance_summary",
            SILVER_RT_TABLE,
            REGIME_SIGNAL_TABLE,
        ],
        "model_artifacts": {
            "pre_release_model_metadata": (
                f"{release_root}/pre_release_model_metadata.json"
            ),
            "promoted_model_metadata": (
                f"{release_root}/promoted_model_metadata.json"
            ),
            "live_model_alias_path": LIVE_MODEL_PATH,
        },
    }
    write_json(f"{release_root}/changed_objects.json", changed_objects)

    summary_lines = [
        "# Improved ALFRED Production Release",
        "",
        "## Release Shape",
        "",
        "- Historical macro refresh type: `broad full bootstrap overwrite`",
        "- Silver refresh type: `broad full rebuild`",
        "- Retrain type: `production monthly walk-forward retrain with promotion`",
        "",
        "## Repo State",
        "",
        f"- Workspace repo path: `{WORKSPACE_REPO_PATH}`",
        f"- Repo branch before: `{repo_branch_before}`",
        f"- Repo branch after: `{repo_branch_after}`",
        f"- Repo head before: `{repo_head_before}`",
        f"- Repo head after: `{repo_head_after}`",
        "",
        "## Promoted Model",
        "",
        f"- Model version: `{promoted_metadata['latest_registry_row'].get('model_version')}`",
        f"- Model path: `{promoted_metadata['latest_registry_row'].get('model_path')}`",
        f"- Versioned model path: `{promoted_metadata['latest_registry_row'].get('versioned_model_path')}`",
        "",
        "## Validation",
        "",
        validation_pdf.to_markdown(index=False),
    ]
    write_text(f"{release_root}/deployment_summary.md", "\n".join(summary_lines) + "\n")

    result = {
        "status": "success",
        "phase": "post_release_validation",
        "release_root": release_root,
        "promoted_model_version": promoted_metadata["latest_registry_row"].get(
            "model_version"
        ),
        "promoted_model_path": promoted_metadata["latest_registry_row"].get(
            "model_path"
        ),
        "repo_branch_before": repo_branch_before,
        "repo_branch_after": repo_branch_after,
        "repo_head_before": repo_head_before,
        "repo_head_after": repo_head_after,
        "validation_summary": validation_pdf[
            ["check_name", "status", "details"]
        ].to_dict("records"),
        "executed_runs": executed_runs,
        "artifacts": {
            "deployment_summary": f"{release_root}/deployment_summary.md",
            "changed_objects": f"{release_root}/changed_objects.json",
            "validation_checks": f"{release_root}/validation_checks.csv",
            "latest_feature_snapshot": f"{release_root}/latest_feature_snapshot.csv",
            "latest_regime_snapshot": f"{release_root}/latest_regime_snapshot.json",
            "promoted_model_metadata": f"{release_root}/promoted_model_metadata.json",
        },
    }
    write_json(f"{release_root}/run_metadata.json", result)
else:
    raise ValueError(f"Unsupported mode: {mode}")

print(json.dumps(result, indent=2, default=serialize))
if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=serialize))

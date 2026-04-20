# Databricks notebook source
# MAGIC %pip install fredapi xgboost

# COMMAND ----------

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytz
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from pyspark.sql import functions as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
LEGACY_SCHEMA = "default"

COMPARISON_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/comparison"
DEFAULT_OUTPUT_ROOT = (
    f"{COMPARISON_ROOT}/2026-04-20_legacy_recreated_signals_alfred_only"
)
LEGACY_VOLUME_PATH = f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data"

MARKET_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"
VIX3M_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix3m_1h"
VIX_SPOT_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix_1h"
REFERENCE_SIGNAL_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bt_walkforward_regime_signals"
REFERENCE_SUMMARY_PATH_DEFAULT = f"{LEGACY_VOLUME_PATH}/strategy_performance_summary.csv"

BRONZE_BASELINE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_legacy_recreated_baseline"
)
BRONZE_CORRECTED_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_legacy_recreated_corrected"
)
SILVER_BASELINE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_recreated_baseline"
)
SILVER_CORRECTED_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_recreated_corrected"
)
WF_SIGNAL_BASELINE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_recreated_baseline"
)
WF_SIGNAL_CORRECTED_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_recreated_corrected"
)

TRAIN_START_DEFAULT = "2016-01-01"
LIVE_START_DEFAULT = "2021-01-01"
LIVE_END_DEFAULT = "2026-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252
VIX_SOURCE = "fred_1d"
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

INDICATOR_CONFIG = {
    "CPIAUCSL": {"typical_time": "08:30:00"},
    "PCEPI": {"typical_time": "08:30:00"},
    "PCE": {"typical_time": "08:30:00"},
    "PAYEMS": {"typical_time": "08:30:00"},
    "UNRATE": {"typical_time": "08:30:00"},
    "GDPC1": {"typical_time": "08:30:00"},
    "INDPRO": {"typical_time": "09:15:00"},
    "UMCSENT": {"typical_time": "10:00:00"},
}

FEATURES = [
    "CPIAUCSL_z",
    "PCEPI_z",
    "PCE_z",
    "PAYEMS_z",
    "UNRATE_z",
    "GDPC1_z",
    "INDPRO_z",
    "UMCSENT_z",
    "VIXCLS_z",
    "vix3m_z",
    "VIX_Inversion_Z",
    "Price_Mom_24H",
    "CPI_Event_Active",
    "NFP_Event_Active",
]

REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}


def ensure_dir(path_str):
    Path(path_str).mkdir(parents=True, exist_ok=True)


def write_text(path_str, content):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(content)


def write_json(path_str, payload):
    ensure_dir(str(Path(path_str).parent))
    Path(path_str).write_text(json.dumps(payload, indent=2, default=str))


def normalize_timestamp(pdf, column="timestamp"):
    result = pdf.copy()
    if column in result.columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
        try:
            result[column] = result[column].dt.tz_localize(None)
        except Exception:
            pass
    return result


def load_csv(path_str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def table_exists(table_name):
    return spark.catalog.tableExists(table_name)


def table_row_range(table_name, ts_column="timestamp"):
    if not table_exists(table_name):
        return {
            "table": table_name,
            "exists": False,
            "row_count": None,
            "min_timestamp": None,
            "max_timestamp": None,
        }

    row = (
        spark.sql(
            f"""
            SELECT
                COUNT(*) AS row_count,
                MIN({ts_column}) AS min_timestamp,
                MAX({ts_column}) AS max_timestamp
            FROM {table_name}
            """
        )
        .toPandas()
        .iloc[0]
    )
    return {
        "table": table_name,
        "exists": True,
        "row_count": int(row["row_count"]),
        "min_timestamp": str(row["min_timestamp"]) if pd.notna(row["min_timestamp"]) else None,
        "max_timestamp": str(row["max_timestamp"]) if pd.notna(row["max_timestamp"]) else None,
    }


def source_indicator_csv_path(indicator_id):
    return f"{LEGACY_VOLUME_PATH}/first_releases_{indicator_id}.csv"


def load_source_indicator_csv(indicator_id):
    path_str = source_indicator_csv_path(indicator_id)
    pdf = load_csv(path_str)
    if pdf.empty:
        return pdf, path_str

    if "release_datetime" in pdf.columns:
        pdf["release_datetime"] = pd.to_datetime(pdf["release_datetime"], errors="coerce")
    else:
        typical_time = INDICATOR_CONFIG[indicator_id]["typical_time"]
        pdf["release_datetime"] = pd.to_datetime(
            pdf["release_date"].astype(str) + f" {typical_time}",
            errors="coerce",
        )
    pdf["release_date"] = (
        pd.to_datetime(pdf["release_date"], errors="coerce").dt.date.astype(str)
    )
    value_col = "value" if "value" in pdf.columns else "actual"
    pdf["actual"] = pd.to_numeric(pdf[value_col], errors="coerce")
    pdf = normalize_timestamp(pdf, "release_datetime")
    pdf = pdf.dropna(subset=["release_date", "release_datetime"]).drop_duplicates(
        subset=["release_date"],
        keep="last",
    )
    return pdf.sort_values("release_datetime"), path_str


def write_delta_table_from_pandas(pdf, table_name, partition_cols=None):
    sdf = spark.createDataFrame(pdf)
    writer = (
        sdf.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
    )
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(table_name)


output_root = DEFAULT_OUTPUT_ROOT
market_table = MARKET_TABLE_DEFAULT
vix3m_table = VIX3M_TABLE_DEFAULT
vix_spot_table = VIX_SPOT_TABLE_DEFAULT
reference_signal_table = REFERENCE_SIGNAL_TABLE_DEFAULT
reference_summary_path = REFERENCE_SUMMARY_PATH_DEFAULT
bronze_baseline_table = BRONZE_BASELINE_TABLE_DEFAULT
bronze_corrected_table = BRONZE_CORRECTED_TABLE_DEFAULT
silver_baseline_table = SILVER_BASELINE_TABLE_DEFAULT
silver_corrected_table = SILVER_CORRECTED_TABLE_DEFAULT
wf_signal_baseline_table = WF_SIGNAL_BASELINE_TABLE_DEFAULT
wf_signal_corrected_table = WF_SIGNAL_CORRECTED_TABLE_DEFAULT
train_start = TRAIN_START_DEFAULT
live_start = LIVE_START_DEFAULT
live_end = LIVE_END_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("output_root", DEFAULT_OUTPUT_ROOT)
    dbutils.widgets.text("market_table", MARKET_TABLE_DEFAULT)
    dbutils.widgets.text("vix3m_table", VIX3M_TABLE_DEFAULT)
    dbutils.widgets.text("vix_spot_table", VIX_SPOT_TABLE_DEFAULT)
    dbutils.widgets.text("reference_signal_table", REFERENCE_SIGNAL_TABLE_DEFAULT)
    dbutils.widgets.text("reference_summary_path", REFERENCE_SUMMARY_PATH_DEFAULT)
    dbutils.widgets.text("bronze_baseline_table", BRONZE_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("bronze_corrected_table", BRONZE_CORRECTED_TABLE_DEFAULT)
    dbutils.widgets.text("silver_baseline_table", SILVER_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("silver_corrected_table", SILVER_CORRECTED_TABLE_DEFAULT)
    dbutils.widgets.text("wf_signal_baseline_table", WF_SIGNAL_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("wf_signal_corrected_table", WF_SIGNAL_CORRECTED_TABLE_DEFAULT)
    dbutils.widgets.text("train_start", TRAIN_START_DEFAULT)
    dbutils.widgets.text("live_start", LIVE_START_DEFAULT)
    dbutils.widgets.text("live_end", LIVE_END_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    output_root = dbutils.widgets.get("output_root").strip() or DEFAULT_OUTPUT_ROOT
    market_table = dbutils.widgets.get("market_table").strip() or MARKET_TABLE_DEFAULT
    vix3m_table = dbutils.widgets.get("vix3m_table").strip() or VIX3M_TABLE_DEFAULT
    vix_spot_table = dbutils.widgets.get("vix_spot_table").strip() or VIX_SPOT_TABLE_DEFAULT
    reference_signal_table = (
        dbutils.widgets.get("reference_signal_table").strip()
        or REFERENCE_SIGNAL_TABLE_DEFAULT
    )
    reference_summary_path = (
        dbutils.widgets.get("reference_summary_path").strip()
        or REFERENCE_SUMMARY_PATH_DEFAULT
    )
    bronze_baseline_table = (
        dbutils.widgets.get("bronze_baseline_table").strip()
        or BRONZE_BASELINE_TABLE_DEFAULT
    )
    bronze_corrected_table = (
        dbutils.widgets.get("bronze_corrected_table").strip()
        or BRONZE_CORRECTED_TABLE_DEFAULT
    )
    silver_baseline_table = (
        dbutils.widgets.get("silver_baseline_table").strip()
        or SILVER_BASELINE_TABLE_DEFAULT
    )
    silver_corrected_table = (
        dbutils.widgets.get("silver_corrected_table").strip()
        or SILVER_CORRECTED_TABLE_DEFAULT
    )
    wf_signal_baseline_table = (
        dbutils.widgets.get("wf_signal_baseline_table").strip()
        or WF_SIGNAL_BASELINE_TABLE_DEFAULT
    )
    wf_signal_corrected_table = (
        dbutils.widgets.get("wf_signal_corrected_table").strip()
        or WF_SIGNAL_CORRECTED_TABLE_DEFAULT
    )
    train_start = dbutils.widgets.get("train_start").strip() or TRAIN_START_DEFAULT
    live_start = dbutils.widgets.get("live_start").strip() or LIVE_START_DEFAULT
    live_end = dbutils.widgets.get("live_end").strip() or LIVE_END_DEFAULT
    initial_capital = parse_float(
        dbutils.widgets.get("initial_capital"),
        INITIAL_CAPITAL_DEFAULT,
    )
    transaction_cost = parse_float(
        dbutils.widgets.get("transaction_cost"),
        TRANSACTION_COST_DEFAULT,
    )


ensure_dir(output_root)
baseline_dir = f"{output_root}/legacy_recreated_baseline"
corrected_dir = f"{output_root}/legacy_recreated_corrected_alfred"
comparison_dir = f"{output_root}/comparison"
for folder in [baseline_dir, corrected_dir, comparison_dir]:
    ensure_dir(folder)

fred = Fred(api_key=FRED_API_KEY)
alfred_release_cache = {}


def load_alfred_releases(series_id):
    if series_id in alfred_release_cache:
        return alfred_release_cache[series_id]

    vintages = fred.get_series_all_releases(series_id).copy()
    if "date" in vintages.columns and "observation_date" not in vintages.columns:
        vintages = vintages.rename(columns={"date": "observation_date"})
    vintages["realtime_start"] = pd.to_datetime(
        vintages["realtime_start"],
        errors="coerce",
    ).dt.date.astype(str)
    vintages["observation_date"] = pd.to_datetime(
        vintages["observation_date"],
        errors="coerce",
    )
    vintages["value_num"] = pd.to_numeric(vintages["value"], errors="coerce")
    alfred_release_cache[series_id] = vintages
    return vintages


def select_corrected_release_value(indicator_id, release_date, fallback_actual):
    vintages = load_alfred_releases(indicator_id)
    release_rows = vintages[vintages["realtime_start"] == str(release_date)].copy()
    if release_rows.empty:
        return {
            "actual": fallback_actual,
            "selected_observation_date": None,
            "selection_status": "fallback_no_release_bucket",
        }

    valid_rows = release_rows.dropna(subset=["observation_date"]).copy()
    if valid_rows.empty:
        first_row = release_rows.iloc[0]
        actual = pd.to_numeric(first_row.get("value_num"), errors="coerce")
        if pd.isna(actual):
            actual = fallback_actual
        return {
            "actual": float(actual) if pd.notna(actual) else np.nan,
            "selected_observation_date": None,
            "selection_status": "fallback_no_observation_date",
        }

    max_obs = valid_rows["observation_date"].max()
    selected = valid_rows[valid_rows["observation_date"] == max_obs].iloc[-1]
    actual = pd.to_numeric(selected.get("value_num"), errors="coerce")
    if pd.isna(actual):
        actual = fallback_actual
        status = "fallback_nan_value"
    else:
        status = "corrected_latest_observation"
    return {
        "actual": float(actual) if pd.notna(actual) else np.nan,
        "selected_observation_date": (
            max_obs.date().isoformat() if pd.notna(max_obs) else None
        ),
        "selection_status": status,
    }


def build_baseline_bronze(target_table, variant_dir):
    bronze_rows = []
    audit_rows = []
    for indicator_id in INDICATOR_CONFIG:
        source_pdf, source_path = load_source_indicator_csv(indicator_id)
        if source_pdf.empty:
            raise FileNotFoundError(f"Missing source CSV for {indicator_id}: {source_path}")

        numeric_pdf = source_pdf.dropna(subset=["release_datetime", "actual"]).copy()
        numeric_pdf["indicator_id"] = indicator_id
        numeric_pdf["timestamp"] = numeric_pdf["release_datetime"]
        numeric_pdf["event"] = f"First Release: {indicator_id}"
        numeric_pdf["year_partition"] = numeric_pdf["timestamp"].dt.year.astype(int)
        numeric_pdf["ingestion_ts"] = pd.Timestamp.utcnow().tz_localize(None)
        bronze_rows.append(
            numeric_pdf[
                [
                    "indicator_id",
                    "timestamp",
                    "actual",
                    "event",
                    "year_partition",
                    "ingestion_ts",
                ]
            ]
        )
        audit_rows.append(
            numeric_pdf[
                ["indicator_id", "release_date", "release_datetime", "actual"]
            ].assign(
                source_csv_path=source_path,
                selection_mode="legacy_source_csv",
                selected_observation_date=None,
                selection_status="source_csv",
            )
        )

    bronze_pdf = pd.concat(bronze_rows, ignore_index=True).sort_values(
        ["indicator_id", "timestamp"]
    )
    write_delta_table_from_pandas(
        bronze_pdf,
        target_table,
        partition_cols=["indicator_id"],
    )

    audit_pdf = pd.concat(audit_rows, ignore_index=True).sort_values(
        ["indicator_id", "release_datetime"]
    )
    audit_path = f"{variant_dir}/macro_selection_audit.csv"
    audit_pdf.to_csv(audit_path, index=False)

    summary_pdf = (
        bronze_pdf.groupby("indicator_id")
        .agg(
            row_count=("actual", "size"),
            min_timestamp=("timestamp", "min"),
            max_timestamp=("timestamp", "max"),
            avg_actual=("actual", "mean"),
        )
        .reset_index()
        .sort_values("indicator_id")
    )
    summary_path = f"{variant_dir}/macro_indicator_summary.csv"
    summary_pdf.to_csv(summary_path, index=False)
    return {
        "table_stats": table_row_range(target_table),
        "audit_path": audit_path,
        "summary_path": summary_path,
    }


def build_corrected_bronze(target_table, variant_dir):
    bronze_rows = []
    audit_rows = []
    for indicator_id in INDICATOR_CONFIG:
        source_pdf, source_path = load_source_indicator_csv(indicator_id)
        if source_pdf.empty:
            raise FileNotFoundError(f"Missing source CSV for {indicator_id}: {source_path}")

        numeric_pdf = source_pdf.dropna(subset=["release_datetime", "actual"]).copy()
        numeric_pdf = numeric_pdf.sort_values("release_datetime")
        selection_rows = []
        for row in numeric_pdf.itertuples(index=False):
            selection = select_corrected_release_value(
                indicator_id,
                row.release_date,
                row.actual,
            )
            selection_rows.append(
                {
                    "indicator_id": indicator_id,
                    "release_date": row.release_date,
                    "release_datetime": row.release_datetime,
                    "baseline_actual": row.actual,
                    "actual": selection["actual"],
                    "selected_observation_date": selection["selected_observation_date"],
                    "selection_status": selection["selection_status"],
                    "source_csv_path": source_path,
                }
            )
        selected_pdf = pd.DataFrame(selection_rows)
        selected_pdf["timestamp"] = selected_pdf["release_datetime"]
        selected_pdf["event"] = f"First Release: {indicator_id}"
        selected_pdf["year_partition"] = selected_pdf["timestamp"].dt.year.astype(int)
        selected_pdf["ingestion_ts"] = pd.Timestamp.utcnow().tz_localize(None)
        bronze_rows.append(
            selected_pdf[
                [
                    "indicator_id",
                    "timestamp",
                    "actual",
                    "event",
                    "year_partition",
                    "ingestion_ts",
                ]
            ]
        )
        audit_rows.append(selected_pdf)

    bronze_pdf = pd.concat(bronze_rows, ignore_index=True).sort_values(
        ["indicator_id", "timestamp"]
    )
    write_delta_table_from_pandas(
        bronze_pdf,
        target_table,
        partition_cols=["indicator_id"],
    )

    audit_pdf = pd.concat(audit_rows, ignore_index=True).sort_values(
        ["indicator_id", "release_datetime"]
    )
    audit_pdf["actual_changed_vs_baseline"] = (
        audit_pdf["actual"].round(12) != audit_pdf["baseline_actual"].round(12)
    )
    audit_path = f"{variant_dir}/macro_selection_audit.csv"
    audit_pdf.to_csv(audit_path, index=False)

    summary_pdf = (
        bronze_pdf.groupby("indicator_id")
        .agg(
            row_count=("actual", "size"),
            min_timestamp=("timestamp", "min"),
            max_timestamp=("timestamp", "max"),
            avg_actual=("actual", "mean"),
        )
        .reset_index()
        .sort_values("indicator_id")
    )
    summary_path = f"{variant_dir}/macro_indicator_summary.csv"
    summary_pdf.to_csv(summary_path, index=False)
    return {
        "table_stats": table_row_range(target_table),
        "audit_path": audit_path,
        "summary_path": summary_path,
    }


def build_legacy_silver(macro_table, target_table):
    start_date = pd.Timestamp(f"{train_start} 00:00:00")
    market_pdf = (
        spark.table(market_table)
        .filter(F.col("symbol") == "SPY")
        .select("timestamp", F.col("close").alias("SPY_close"))
        .toPandas()
    )
    market_pdf = normalize_timestamp(market_pdf)
    market_pdf = market_pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    market_pdf = market_pdf.sort_values("timestamp")
    end_date = market_pdf["timestamp"].max()

    backbone_pdf = pd.DataFrame(
        {"timestamp": pd.date_range(start=start_date, end=end_date, freq="H")}
    )

    macro_pdf = spark.table(macro_table).toPandas()
    macro_pdf = normalize_timestamp(macro_pdf)
    macro_pdf = macro_pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp", "indicator_id"],
        keep="last",
    )
    macro_pdf["join_ts"] = (macro_pdf["timestamp"] + pd.Timedelta(hours=2)).dt.floor("H")
    macro_prepared = (
        macro_pdf.pivot_table(
            index="join_ts",
            columns="indicator_id",
            values="actual",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"join_ts": "timestamp"})
    )

    vix3m_pdf = spark.table(vix3m_table).toPandas()
    vix3m_pdf = normalize_timestamp(vix3m_pdf)
    vix3m_pdf = vix3m_pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    if VIX_SOURCE == "portara_1h":
        vix3m_pdf["timestamp"] = vix3m_pdf["timestamp"] + pd.Timedelta(days=1)
        vix3m_pdf["vix3m"] = pd.to_numeric(vix3m_pdf["vix3m"], errors="coerce")
    else:
        vix3m_pdf["vix3m"] = pd.to_numeric(vix3m_pdf["close"], errors="coerce")
    vix3m_pdf = vix3m_pdf[["timestamp", "vix3m"]]

    vix_spot_pdf = spark.table(vix_spot_table).toPandas()
    vix_spot_pdf = normalize_timestamp(vix_spot_pdf)
    vix_spot_pdf = vix_spot_pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    vix_spot_pdf["VIXCLS"] = pd.to_numeric(vix_spot_pdf["close"], errors="coerce")
    vix_spot_pdf = vix_spot_pdf[["timestamp", "VIXCLS"]]

    silver_grid = backbone_pdf.merge(macro_prepared, on="timestamp", how="left")
    silver_grid = silver_grid.merge(market_pdf, on="timestamp", how="left")
    silver_grid = silver_grid.merge(vix3m_pdf, on="timestamp", how="left")
    silver_grid = silver_grid.merge(vix_spot_pdf, on="timestamp", how="left")

    required_macro_columns = list(INDICATOR_CONFIG.keys())
    for column_name in required_macro_columns:
        if column_name not in silver_grid.columns:
            silver_grid[column_name] = np.nan

    fill_cols = required_macro_columns + ["VIXCLS", "vix3m", "SPY_close"]
    silver_grid = silver_grid.sort_values("timestamp").reset_index(drop=True)
    silver_grid[fill_cols] = silver_grid[fill_cols].ffill()

    z_targets = required_macro_columns + ["VIXCLS", "vix3m"]
    window_size = 8761
    for col_name in z_targets:
        rolling_mean = silver_grid[col_name].rolling(window_size, min_periods=1).mean()
        rolling_std = silver_grid[col_name].rolling(window_size, min_periods=2).std()
        denominator = rolling_std.where(rolling_std != 0, 1)
        silver_grid[f"{col_name}_z"] = (silver_grid[col_name] - rolling_mean) / denominator

    silver_grid["CPI_Shock"] = (
        silver_grid["CPIAUCSL"] - silver_grid["CPIAUCSL"].shift(1)
    ).abs()
    silver_grid["NFP_Shock"] = (
        silver_grid["PAYEMS"] - silver_grid["PAYEMS"].shift(1)
    ).abs()
    silver_grid["CPI_Event_Active"] = (
        (silver_grid["CPI_Shock"] > 0).astype(int).rolling(49, min_periods=1).max()
    )
    silver_grid["NFP_Event_Active"] = (
        (silver_grid["NFP_Shock"] > 0).astype(int).rolling(49, min_periods=1).max()
    )
    silver_grid["VIX_Inversion_Z"] = silver_grid["VIXCLS_z"] - silver_grid["vix3m_z"]
    silver_grid["Price_Mom_24H"] = (
        silver_grid["SPY_close"].shift(1) / silver_grid["SPY_close"].shift(25)
    ) - 1

    final_silver = silver_grid[["timestamp", "SPY_close"] + FEATURES].copy()
    final_silver = final_silver[final_silver["SPY_close"].notna()].reset_index(drop=True)
    write_delta_table_from_pandas(final_silver, target_table)
    return table_row_range(target_table)


def train_full_signal_model(silver_table, output_dir):
    df = spark.table(silver_table).toPandas()
    df = normalize_timestamp(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    df["fwd_return_1w"] = df["SPY_close"].shift(-168) / df["SPY_close"] - 1

    def label_regime(ret):
        if pd.isna(ret):
            return None
        if ret > 0.005:
            return 0
        if ret < -0.005:
            return 2
        return 1

    df["label"] = df["fwd_return_1w"].apply(label_regime)
    df_ml = df.dropna(subset=["label"]).copy()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        objective="multi:softmax",
        num_class=3,
        tree_method="hist",
        random_state=42,
    )
    model.fit(df_ml[FEATURES], df_ml["label"])

    importance_pdf = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_pdf.to_csv(f"{output_dir}/feature_importance.csv", index=False)

    x_full = df[FEATURES].fillna(0)
    df["predicted_regime"] = model.predict(x_full)
    df["regime_name"] = df["predicted_regime"].map(REGIME_NAME_MAP)
    full_signal_pdf = df.reset_index()
    full_signal_path = f"{output_dir}/full_signal_export.csv"
    full_signal_pdf.to_csv(full_signal_path, index=False)
    return full_signal_pdf, full_signal_path


def build_walkforward_schedule(live_start_value, live_end_value):
    retrain_dates = []
    d = pd.Timestamp(live_start_value)
    live_end_ts = pd.Timestamp(live_end_value)
    while d <= live_end_ts:
        retrain_dates.append(d)
        d += relativedelta(months=1)
    return retrain_dates


def train_walkforward_model(train_df):
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=3,
        tree_method="hist",
        random_state=42,
    )
    model.fit(train_df[FEATURES], train_df["regime"])
    return model


def load_price_frame(table_name, start_value, end_value):
    pdf = (
        spark.sql(
            f"""
            SELECT timestamp, symbol, close
            FROM {table_name}
            WHERE timestamp >= TIMESTAMP('{start_value}')
              AND timestamp < TIMESTAMP('{end_value}')
              AND symbol IN ('SPY', 'QQQ')
            """
        )
        .toPandas()
    )
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp", "symbol"],
        keep="last",
    )
    pivoted = (
        pdf.pivot_table(
            index="timestamp",
            columns="symbol",
            values="close",
            aggfunc="last",
        )
        .reset_index()
        .sort_values("timestamp")
    )
    pivoted["SPY"] = pd.to_numeric(pivoted["SPY"], errors="coerce").ffill()
    pivoted["QQQ"] = pd.to_numeric(pivoted["QQQ"], errors="coerce").ffill()
    return pivoted


def run_legacy_backtest(signals_pdf, prices_pdf, initial_capital_value, transaction_cost_value):
    merged = signals_pdf.merge(prices_pdf, on="timestamp", how="left").sort_values("timestamp")
    merged["SPY"] = pd.to_numeric(merged["SPY"], errors="coerce").ffill()
    merged["QQQ"] = pd.to_numeric(merged["QQQ"], errors="coerce").ffill()
    merged = merged.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

    capital = float(initial_capital_value)
    active_asset = "START"
    portfolio = [capital]
    applied_assets = ["START"]
    switches = [False]
    net_returns = [0.0]

    for i in range(1, len(merged)):
        regime = int(merged["predicted_regime"].iloc[i - 1])
        if regime == 0:
            target = "QQQ"
        elif regime == 1:
            target = "SPY"
        else:
            target = "CASH"

        if target == "QQQ":
            ret = merged["QQQ"].iloc[i] / merged["QQQ"].iloc[i - 1] - 1
        elif target == "SPY":
            ret = merged["SPY"].iloc[i] / merged["SPY"].iloc[i - 1] - 1
        else:
            ret = 0.0

        cost = transaction_cost_value if target != active_asset else 0.0
        capital *= 1 + ret - cost
        portfolio.append(capital)
        applied_assets.append(target)
        switches.append(target != active_asset)
        net_returns.append(ret - cost)
        active_asset = target

    merged["portfolio"] = portfolio
    merged["active_asset"] = applied_assets
    merged["position_changed"] = switches
    merged["net_return"] = net_returns
    return merged


def compute_legacy_metrics(series):
    daily = series.resample("D").last().dropna()
    returns = daily.pct_change(fill_method=None).dropna()
    years = len(returns) / TRADING_DAYS if len(returns) else np.nan

    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    total_return = end / start - 1
    cagr = (end / start) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol_ann = returns.std() * np.sqrt(TRADING_DAYS) if len(returns) else np.nan
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)
        if len(returns) and returns.std() != 0
        else np.nan
    )
    downside = returns[returns < 0]
    sortino = (
        (returns.mean() / downside.std()) * np.sqrt(TRADING_DAYS)
        if len(downside) and downside.std() != 0
        else np.nan
    )
    drawdown = series / series.cummax() - 1
    return {
        "Start Value": start,
        "Final Value": end,
        "Total Return": float(total_return),
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Vol Annual": float(vol_ann) if np.isfinite(vol_ann) else np.nan,
        "Sharpe Annual": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Sortino Annual": float(sortino) if np.isfinite(sortino) else np.nan,
        "Max Drawdown": float(drawdown.min()) if len(series) else np.nan,
    }


def regime_stats_from_signals(signals_pdf):
    ordered = signals_pdf.sort_values("timestamp").reset_index(drop=True).copy()
    ordered["switch_group"] = ordered["predicted_regime"].ne(
        ordered["predicted_regime"].shift()
    ).cumsum()
    holding_lengths = ordered.groupby("switch_group").size()
    regime_mix = (
        ordered["predicted_regime"]
        .value_counts(normalize=True)
        .rename_axis("predicted_regime")
        .reset_index(name="pct_time")
    )
    regime_mix["regime_name"] = regime_mix["predicted_regime"].map(REGIME_NAME_MAP)
    return {
        "total_switches": int(ordered["predicted_regime"].ne(ordered["predicted_regime"].shift()).sum() - 1),
        "average_holding_bars": float(holding_lengths.mean()) if not holding_lengths.empty else np.nan,
        "regime_mix": regime_mix.sort_values("predicted_regime"),
    }


def run_walkforward_variant(silver_table, label_path, signal_table, output_dir, variant_label):
    silver_df = (
        spark.table(silver_table)
        .filter(f"timestamp >= '{train_start}'")
        .select(["timestamp"] + FEATURES)
        .toPandas()
    )
    silver_df = normalize_timestamp(silver_df)
    silver_df = silver_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    labels_df = pd.read_csv(label_path, parse_dates=["timestamp"])
    labels_df = normalize_timestamp(labels_df)
    labels_df = labels_df[["timestamp", "predicted_regime"]].rename(
        columns={"predicted_regime": "regime"}
    )

    dataset = silver_df.merge(labels_df, on="timestamp", how="left")
    dataset = dataset.dropna(subset=["regime"]).copy()
    dataset["regime"] = pd.to_numeric(dataset["regime"], errors="coerce").astype(int)
    dataset = dataset[
        (dataset["timestamp"] >= pd.Timestamp(train_start))
        & (dataset["timestamp"] < pd.Timestamp(live_end))
    ].copy()

    retrain_dates = build_walkforward_schedule(live_start, live_end)
    signals = []
    monthly_rows = []
    feature_importance_history = []

    for i in range(len(retrain_dates) - 1):
        train_end = retrain_dates[i]
        test_end = retrain_dates[i + 1]

        train_df = dataset[
            (dataset["timestamp"] >= pd.Timestamp(train_start))
            & (dataset["timestamp"] < train_end)
        ].copy()
        test_df = dataset[
            (dataset["timestamp"] >= train_end)
            & (dataset["timestamp"] < test_end)
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        model = train_walkforward_model(train_df)
        probs = model.predict_proba(test_df[FEATURES])
        preds = np.argmax(probs, axis=1)

        test_df["prob_regime0"] = probs[:, 0]
        test_df["prob_regime1"] = probs[:, 1]
        test_df["prob_regime2"] = probs[:, 2]
        test_df["predicted_regime"] = preds

        monthly_rows.append(
            {
                "variant": variant_label,
                "train_end": train_end,
                "test_end": test_end,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "monthly_accuracy": float(
                    (test_df["predicted_regime"] == test_df["regime"]).mean()
                ),
            }
        )

        importance_dict = model.get_booster().get_score(importance_type="gain")
        imp_df = pd.DataFrame(
            {
                "feature": list(importance_dict.keys()),
                "importance": list(importance_dict.values()),
            }
        )
        imp_df["date"] = train_end
        feature_importance_history.append(imp_df)

        signals.append(
            test_df[
                [
                    "timestamp",
                    "predicted_regime",
                    "prob_regime0",
                    "prob_regime1",
                    "prob_regime2",
                ]
            ]
        )

    if not signals:
        raise ValueError(f"No walk-forward signals generated for {variant_label}")

    signals_df = pd.concat(signals, ignore_index=True).sort_values("timestamp")
    signals_df["regime_name"] = signals_df["predicted_regime"].map(REGIME_NAME_MAP)
    signals_df.to_csv(f"{output_dir}/walkforward_signals.csv", index=False)

    feature_importance_pdf = pd.concat(feature_importance_history, ignore_index=True)
    if not feature_importance_pdf.empty:
        feature_importance_pdf["importance"] = feature_importance_pdf.groupby("date")[
            "importance"
        ].transform(lambda x: x / x.sum() if x.sum() else x)
    feature_importance_pdf.to_csv(
        f"{output_dir}/feature_importance_monthly.csv",
        index=False,
    )

    monthly_pdf = pd.DataFrame(monthly_rows)
    monthly_pdf.to_csv(f"{output_dir}/monthly_validation.csv", index=False)

    write_delta_table_from_pandas(
        signals_df[["timestamp", "predicted_regime"]],
        signal_table,
    )

    prices_df = load_price_frame(market_table, live_start, live_end)
    backtest_df = run_legacy_backtest(
        signals_df[["timestamp", "predicted_regime"]].copy(),
        prices_df,
        initial_capital,
        transaction_cost,
    )
    backtest_df["regime_name"] = backtest_df["predicted_regime"].map(REGIME_NAME_MAP)
    backtest_df.to_csv(f"{output_dir}/hourly_strategy_curve.csv", index=False)

    strategy_series = backtest_df.set_index("timestamp")["portfolio"]
    metrics = compute_legacy_metrics(strategy_series)
    metrics["Variant"] = variant_label
    metrics["Signal Rows"] = int(len(signals_df))
    metrics["Mean Monthly Accuracy"] = (
        float(monthly_pdf["monthly_accuracy"].mean())
        if not monthly_pdf.empty
        else np.nan
    )
    regime_stats = regime_stats_from_signals(signals_df[["timestamp", "predicted_regime"]])
    metrics["Total Switches"] = regime_stats["total_switches"]
    metrics["Average Holding Bars"] = regime_stats["average_holding_bars"]

    daily_curve = strategy_series.resample("D").last().dropna()
    daily_drawdown = daily_curve / daily_curve.cummax() - 1
    return {
        "signals": signals_df,
        "backtest": backtest_df,
        "daily_curve": daily_curve,
        "daily_drawdown": daily_drawdown,
        "metrics": metrics,
        "monthly_validation": monthly_pdf,
        "regime_stats": regime_stats,
    }


def load_reference_walkforward_signals():
    pdf = spark.table(reference_signal_table).toPandas()
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    pdf["predicted_regime"] = pd.to_numeric(
        pdf["predicted_regime"],
        errors="coerce",
    ).fillna(2).astype(int)
    pdf = pdf[
        (pdf["timestamp"] >= pd.Timestamp(live_start))
        & (pdf["timestamp"] < pd.Timestamp(live_end))
    ].copy()
    return pdf[["timestamp", "predicted_regime"]].sort_values("timestamp")


def load_reference_metrics():
    pdf = load_csv(reference_summary_path)
    row_name_col = "Unnamed: 0" if "Unnamed: 0" in pdf.columns else pdf.columns[0]
    row = pdf[pdf[row_name_col] == "WalkForward_V4"].iloc[0]
    return {
        "Variant": "trusted_legacy_reference",
        "Start Value": float(row["Start Value"]),
        "Final Value": float(row["Final Value"]),
        "Total Return": float(row["Total Return"]),
        "CAGR": float(row["CAGR"]),
        "Vol Annual": float(row["Vol Annual"]),
        "Sharpe Annual": float(row["Sharpe Annual"]),
        "Sortino Annual": float(row["Sortino Annual"]),
        "Max Drawdown": float(row["Max Drawdown"]),
    }


def metric_frame(metric_dict):
    ordered_columns = [
        "Variant",
        "Start Value",
        "Final Value",
        "Total Return",
        "CAGR",
        "Vol Annual",
        "Sharpe Annual",
        "Sortino Annual",
        "Max Drawdown",
        "Signal Rows",
        "Mean Monthly Accuracy",
        "Total Switches",
        "Average Holding Bars",
    ]
    return pd.DataFrame([{key: metric_dict.get(key) for key in ordered_columns}])


def markdown_table(pdf):
    headers = list(pdf.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in pdf.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def recommendation_text(baseline_metrics, corrected_metrics):
    cagr_delta = corrected_metrics["CAGR"] - baseline_metrics["CAGR"]
    sharpe_delta = corrected_metrics["Sharpe Annual"] - baseline_metrics["Sharpe Annual"]
    max_dd_delta = corrected_metrics["Max Drawdown"] - baseline_metrics["Max Drawdown"]

    if cagr_delta >= 0.005 and sharpe_delta >= 0.02 and max_dd_delta >= -0.02:
        return "prefer recreated corrected ALFRED"
    if cagr_delta <= -0.005 and sharpe_delta <= -0.02:
        return "keep recreated baseline"
    return "further review needed"


baseline_macro_info = build_baseline_bronze(bronze_baseline_table, baseline_dir)
corrected_macro_info = build_corrected_bronze(bronze_corrected_table, corrected_dir)

baseline_silver_stats = build_legacy_silver(bronze_baseline_table, silver_baseline_table)
corrected_silver_stats = build_legacy_silver(bronze_corrected_table, silver_corrected_table)

baseline_full_signal_pdf, baseline_full_signal_path = train_full_signal_model(
    silver_baseline_table,
    baseline_dir,
)
corrected_full_signal_pdf, corrected_full_signal_path = train_full_signal_model(
    silver_corrected_table,
    corrected_dir,
)

baseline_walkforward = run_walkforward_variant(
    silver_baseline_table,
    baseline_full_signal_path,
    wf_signal_baseline_table,
    baseline_dir,
    "legacy_recreated_baseline",
)
corrected_walkforward = run_walkforward_variant(
    silver_corrected_table,
    corrected_full_signal_path,
    wf_signal_corrected_table,
    corrected_dir,
    "legacy_recreated_corrected_alfred",
)

reference_signals = load_reference_walkforward_signals()
reference_metrics = load_reference_metrics()
baseline_reference_compare = (
    baseline_walkforward["signals"][["timestamp", "predicted_regime"]]
    .merge(
        reference_signals.rename(columns={"predicted_regime": "reference_regime"}),
        on="timestamp",
        how="inner",
    )
)
baseline_reference_compare["signal_match"] = (
    baseline_reference_compare["predicted_regime"]
    == baseline_reference_compare["reference_regime"]
)
baseline_reference_match_rate = (
    float(baseline_reference_compare["signal_match"].mean())
    if not baseline_reference_compare.empty
    else np.nan
)

signal_diff_pdf = (
    baseline_walkforward["signals"]
    .rename(
        columns={
            "predicted_regime": "predicted_regime_baseline",
            "prob_regime0": "prob_regime0_baseline",
            "prob_regime1": "prob_regime1_baseline",
            "prob_regime2": "prob_regime2_baseline",
            "regime_name": "regime_name_baseline",
        }
    )
    .merge(
        corrected_walkforward["signals"].rename(
            columns={
                "predicted_regime": "predicted_regime_corrected",
                "prob_regime0": "prob_regime0_corrected",
                "prob_regime1": "prob_regime1_corrected",
                "prob_regime2": "prob_regime2_corrected",
                "regime_name": "regime_name_corrected",
            }
        ),
        on="timestamp",
        how="outer",
    )
    .sort_values("timestamp")
)
signal_diff_pdf["regime_changed"] = (
    signal_diff_pdf["predicted_regime_baseline"]
    != signal_diff_pdf["predicted_regime_corrected"]
)
signal_diff_pdf["max_probability_abs_diff"] = signal_diff_pdf[
    [
        "prob_regime0_baseline",
        "prob_regime1_baseline",
        "prob_regime2_baseline",
    ]
].sub(
    signal_diff_pdf[
        [
            "prob_regime0_corrected",
            "prob_regime1_corrected",
            "prob_regime2_corrected",
        ]
    ].to_numpy()
).abs().max(axis=1)
signal_match_rate = 1.0 - float(signal_diff_pdf["regime_changed"].fillna(True).mean())
regime_mismatch_count = int(signal_diff_pdf["regime_changed"].fillna(True).sum())

baseline_metrics_pdf = metric_frame(baseline_walkforward["metrics"])
corrected_metrics_pdf = metric_frame(corrected_walkforward["metrics"])
baseline_metrics_path = f"{comparison_dir}/recreated_baseline_metrics.csv"
corrected_metrics_path = f"{comparison_dir}/recreated_corrected_metrics.csv"
baseline_metrics_pdf.to_csv(baseline_metrics_path, index=False)
corrected_metrics_pdf.to_csv(corrected_metrics_path, index=False)

metrics_diff_rows = []
for metric_name in [
    "Final Value",
    "Total Return",
    "CAGR",
    "Vol Annual",
    "Sharpe Annual",
    "Sortino Annual",
    "Max Drawdown",
    "Signal Rows",
    "Mean Monthly Accuracy",
    "Total Switches",
    "Average Holding Bars",
]:
    baseline_value = baseline_walkforward["metrics"].get(metric_name)
    corrected_value = corrected_walkforward["metrics"].get(metric_name)
    metrics_diff_rows.append(
        {
            "metric": metric_name,
            "recreated_baseline": baseline_value,
            "recreated_corrected_alfred": corrected_value,
            "corrected_minus_baseline": (
                corrected_value - baseline_value
                if pd.notna(baseline_value) and pd.notna(corrected_value)
                else np.nan
            ),
        }
    )
metrics_diff_pdf = pd.DataFrame(metrics_diff_rows)
metrics_diff_path = f"{comparison_dir}/metrics_diff.csv"
metrics_diff_pdf.to_csv(metrics_diff_path, index=False)

equity_curve_pdf = pd.concat(
    [
        baseline_walkforward["daily_curve"].rename("equity").reset_index().assign(
            variant="legacy_recreated_baseline"
        ),
        corrected_walkforward["daily_curve"].rename("equity").reset_index().assign(
            variant="legacy_recreated_corrected_alfred"
        ),
    ],
    ignore_index=True,
)
equity_curve_path = f"{comparison_dir}/equity_curve.csv"
equity_curve_pdf.to_csv(equity_curve_path, index=False)

drawdown_curve_pdf = pd.concat(
    [
        baseline_walkforward["daily_drawdown"].rename("drawdown").reset_index().assign(
            variant="legacy_recreated_baseline"
        ),
        corrected_walkforward["daily_drawdown"].rename("drawdown").reset_index().assign(
            variant="legacy_recreated_corrected_alfred"
        ),
    ],
    ignore_index=True,
)
drawdown_curve_path = f"{comparison_dir}/drawdown_curve.csv"
drawdown_curve_pdf.to_csv(drawdown_curve_path, index=False)

regime_stats_rows = []
for variant_name, payload in [
    ("legacy_recreated_baseline", baseline_walkforward),
    ("legacy_recreated_corrected_alfred", corrected_walkforward),
]:
    stats = payload["regime_stats"]
    regime_mix = stats["regime_mix"].copy()
    regime_mix["variant"] = variant_name
    regime_mix["total_switches"] = stats["total_switches"]
    regime_mix["average_holding_bars"] = stats["average_holding_bars"]
    regime_stats_rows.append(regime_mix)
regime_stats_pdf = pd.concat(regime_stats_rows, ignore_index=True)
regime_stats_path = f"{comparison_dir}/regime_stats.csv"
regime_stats_pdf.to_csv(regime_stats_path, index=False)

baseline_corrected_bronze_compare = (
    pd.read_csv(f"{corrected_dir}/macro_selection_audit.csv")
    .assign(
        actual_changed_vs_baseline=lambda pdf: (
            pd.to_numeric(pdf["actual"], errors="coerce").round(12)
            != pd.to_numeric(pdf["baseline_actual"], errors="coerce").round(12)
        )
    )
)
bronze_diff_count = int(baseline_corrected_bronze_compare["actual_changed_vs_baseline"].sum())
bronze_diff_pct = float(baseline_corrected_bronze_compare["actual_changed_vs_baseline"].mean())

recommendation = recommendation_text(
    baseline_walkforward["metrics"],
    corrected_walkforward["metrics"],
)

methodology_notes = [
    "# Methodology Notes",
    "",
    "- Recreated signals are used on both sides.",
    "- Only intended difference is ALFRED logic.",
    "- Frozen legacy signal file is reference-only, not control.",
    f"- Legacy baseline reference signal table for sanity only: `{reference_signal_table}`",
    f"- Baseline bronze table: `{bronze_baseline_table}`",
    f"- Corrected bronze table: `{bronze_corrected_table}`",
    f"- Baseline silver table: `{silver_baseline_table}`",
    f"- Corrected silver table: `{silver_corrected_table}`",
    f"- Baseline walk-forward signal table: `{wf_signal_baseline_table}`",
    f"- Corrected walk-forward signal table: `{wf_signal_corrected_table}`",
    f"- Train start: `{train_start}`",
    f"- Live start: `{live_start}`",
    f"- Live end: `{live_end}`",
    f"- Initial capital: `{initial_capital}`",
    f"- Transaction cost: `{transaction_cost}`",
    "",
    "## Controlled Swap",
    "",
    f"- Release schedule source on both sides: legacy `first_releases_*.csv` under `{LEGACY_VOLUME_PATH}`",
    "- Baseline uses the legacy source CSV values.",
    "- Corrected variant preserves the same release timestamps and swaps only the ALFRED bucket value to the latest `observation_date` when available.",
]
methodology_notes_path = f"{comparison_dir}/methodology_notes.md"
write_text(methodology_notes_path, "\n".join(methodology_notes) + "\n")

summary_table = pd.DataFrame(
    [
        baseline_walkforward["metrics"],
        corrected_walkforward["metrics"],
        reference_metrics,
    ]
)[
    [
        "Variant",
        "Final Value",
        "CAGR",
        "Vol Annual",
        "Sharpe Annual",
        "Sortino Annual",
        "Max Drawdown",
        "Signal Rows",
        "Mean Monthly Accuracy",
        "Total Switches",
        "Average Holding Bars",
    ]
]

summary_lines = [
    "# Recreated Legacy Signals ALFRED-Only Comparison",
    "",
    "## Baseline Sanity vs Trusted Legacy Reference",
    "",
    f"- Recreated baseline signal match vs trusted reference table: `{baseline_reference_match_rate:.6f}`",
    f"- Recreated baseline CAGR: `{baseline_walkforward['metrics']['CAGR']:.6f}`",
    f"- Trusted legacy reference CAGR: `{reference_metrics['CAGR']:.6f}`",
    f"- Recreated baseline Sharpe: `{baseline_walkforward['metrics']['Sharpe Annual']:.6f}`",
    f"- Trusted legacy reference Sharpe: `{reference_metrics['Sharpe Annual']:.6f}`",
    f"- Recreated baseline max drawdown: `{baseline_walkforward['metrics']['Max Drawdown']:.6f}`",
    f"- Trusted legacy reference max drawdown: `{reference_metrics['Max Drawdown']:.6f}`",
    "",
    "## Variant Metrics",
    "",
    markdown_table(summary_table),
    "",
    "## ALFRED-Only Effect Using Recreated Signals",
    "",
    f"- Signal match rate between recreated variants: `{signal_match_rate:.6f}`",
    f"- Regime mismatch count between recreated variants: `{regime_mismatch_count}`",
    f"- Bronze rows changed by ALFRED swap: `{bronze_diff_count}`",
    f"- Bronze change rate: `{bronze_diff_pct:.6f}`",
    f"- CAGR delta (corrected - baseline): `{corrected_walkforward['metrics']['CAGR'] - baseline_walkforward['metrics']['CAGR']:.6f}`",
    f"- Annual vol delta (corrected - baseline): `{corrected_walkforward['metrics']['Vol Annual'] - baseline_walkforward['metrics']['Vol Annual']:.6f}`",
    f"- Sharpe delta (corrected - baseline): `{corrected_walkforward['metrics']['Sharpe Annual'] - baseline_walkforward['metrics']['Sharpe Annual']:.6f}`",
    f"- Sortino delta (corrected - baseline): `{corrected_walkforward['metrics']['Sortino Annual'] - baseline_walkforward['metrics']['Sortino Annual']:.6f}`",
    f"- Max drawdown delta (corrected - baseline): `{corrected_walkforward['metrics']['Max Drawdown'] - baseline_walkforward['metrics']['Max Drawdown']:.6f}`",
    f"- Total switch delta (corrected - baseline): `{corrected_walkforward['metrics']['Total Switches'] - baseline_walkforward['metrics']['Total Switches']}`",
    f"- Average holding bars delta (corrected - baseline): `{corrected_walkforward['metrics']['Average Holding Bars'] - baseline_walkforward['metrics']['Average Holding Bars']:.6f}`",
    "",
    "## Recommendation",
    "",
    f"- `{recommendation}`",
]
summary_path = f"{comparison_dir}/summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "recommendation": recommendation,
    "baseline_reference_sanity": {
        "signal_match_rate": baseline_reference_match_rate,
        "cagr_gap": baseline_walkforward["metrics"]["CAGR"] - reference_metrics["CAGR"],
        "sharpe_gap": baseline_walkforward["metrics"]["Sharpe Annual"] - reference_metrics["Sharpe Annual"],
        "max_drawdown_gap": baseline_walkforward["metrics"]["Max Drawdown"] - reference_metrics["Max Drawdown"],
    },
    "alfred_only_effect": {
        "signal_match_rate": signal_match_rate,
        "regime_mismatch_count": regime_mismatch_count,
        "bronze_diff_count": bronze_diff_count,
        "bronze_diff_pct": bronze_diff_pct,
        "cagr_delta": corrected_walkforward["metrics"]["CAGR"] - baseline_walkforward["metrics"]["CAGR"],
        "vol_annual_delta": corrected_walkforward["metrics"]["Vol Annual"] - baseline_walkforward["metrics"]["Vol Annual"],
        "sharpe_delta": corrected_walkforward["metrics"]["Sharpe Annual"] - baseline_walkforward["metrics"]["Sharpe Annual"],
        "sortino_delta": corrected_walkforward["metrics"]["Sortino Annual"] - baseline_walkforward["metrics"]["Sortino Annual"],
        "max_drawdown_delta": corrected_walkforward["metrics"]["Max Drawdown"] - baseline_walkforward["metrics"]["Max Drawdown"],
        "switch_delta": corrected_walkforward["metrics"]["Total Switches"] - baseline_walkforward["metrics"]["Total Switches"],
        "average_holding_bars_delta": corrected_walkforward["metrics"]["Average Holding Bars"] - baseline_walkforward["metrics"]["Average Holding Bars"],
    },
}
recommendation_path = f"{comparison_dir}/recommendation.json"
write_json(recommendation_path, recommendation_payload)

plt.figure(figsize=(14, 7))
for variant, group in equity_curve_pdf.groupby("variant"):
    plt.plot(group["timestamp"], group["equity"], label=variant)
plt.title("Legacy Recreated Signals Equity Curve Comparison")
plt.ylabel("Portfolio Value")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
equity_plot_path = f"{comparison_dir}/equity_curve_comparison.png"
plt.savefig(equity_plot_path, dpi=180)
plt.close()

plt.figure(figsize=(14, 7))
for variant, group in drawdown_curve_pdf.groupby("variant"):
    plt.plot(group["timestamp"], group["drawdown"], label=variant)
plt.title("Legacy Recreated Signals Drawdown Comparison")
plt.ylabel("Drawdown")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
drawdown_plot_path = f"{comparison_dir}/drawdown_comparison.png"
plt.savefig(drawdown_plot_path, dpi=180)
plt.close()

timeline_pdf = (
    signal_diff_pdf[
        ["timestamp", "predicted_regime_baseline", "predicted_regime_corrected"]
    ]
    .dropna()
    .copy()
)
plt.figure(figsize=(16, 5))
plt.step(
    timeline_pdf["timestamp"],
    timeline_pdf["predicted_regime_baseline"],
    where="post",
    label="legacy_recreated_baseline",
)
plt.step(
    timeline_pdf["timestamp"],
    timeline_pdf["predicted_regime_corrected"] + 0.05,
    where="post",
    label="legacy_recreated_corrected_alfred",
)
plt.yticks([0, 1, 2], ["Expansion", "Slowdown", "Recession"])
plt.title("Legacy Recreated Signals Regime Timeline")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
regime_timeline_path = f"{comparison_dir}/regime_timeline.png"
plt.savefig(regime_timeline_path, dpi=180)
plt.close()

signal_diff_path = f"{comparison_dir}/signal_diff.csv"
signal_diff_pdf.to_csv(signal_diff_path, index=False)

run_metadata = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_root": output_root,
    "train_start": train_start,
    "live_start": live_start,
    "live_end": live_end,
    "initial_capital": initial_capital,
    "transaction_cost": transaction_cost,
    "reference_signal_table": reference_signal_table,
    "reference_summary_path": reference_summary_path,
    "tables": {
        "market_table": market_table,
        "vix3m_table": vix3m_table,
        "vix_spot_table": vix_spot_table,
        "bronze_baseline_table": table_row_range(bronze_baseline_table),
        "bronze_corrected_table": table_row_range(bronze_corrected_table),
        "silver_baseline_table": baseline_silver_stats,
        "silver_corrected_table": corrected_silver_stats,
        "wf_signal_baseline_table": table_row_range(wf_signal_baseline_table),
        "wf_signal_corrected_table": table_row_range(wf_signal_corrected_table),
    },
    "baseline_reference_sanity": recommendation_payload["baseline_reference_sanity"],
    "alfred_only_effect": recommendation_payload["alfred_only_effect"],
    "artifacts": {
        "recreated_baseline_metrics": baseline_metrics_path,
        "recreated_corrected_metrics": corrected_metrics_path,
        "metrics_diff": metrics_diff_path,
        "signal_diff": signal_diff_path,
        "equity_curve": equity_curve_path,
        "drawdown_curve": drawdown_curve_path,
        "regime_stats": regime_stats_path,
        "summary": summary_path,
        "recommendation": recommendation_path,
        "methodology_notes": methodology_notes_path,
        "equity_curve_plot": equity_plot_path,
        "drawdown_plot": drawdown_plot_path,
        "regime_timeline_plot": regime_timeline_path,
        "baseline_macro_audit": baseline_macro_info["audit_path"],
        "corrected_macro_audit": corrected_macro_info["audit_path"],
    },
}
run_metadata_path = f"{comparison_dir}/run_metadata.json"
write_json(run_metadata_path, run_metadata)

result = {
    "phase": "legacy_recreated_signals_alfred_only",
    "status": "success",
    "recommendation": recommendation,
    "output_root": output_root,
    "artifacts": run_metadata["artifacts"],
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

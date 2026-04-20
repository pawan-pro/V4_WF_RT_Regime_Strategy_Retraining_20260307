# Databricks notebook source
# MAGIC %pip install fredapi xgboost

# COMMAND ----------

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
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
BASELINE_CONTROL_ROOT_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_canonical_legacy_v4_performance"
)
COMPARISON_OUTPUT_ROOT_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
)
PERFORMANCE_NOTEBOOK_PATH_DEFAULT = (
    "/Users/pawan.jatale@mubama.group/tmp/"
    "freeze_canonical_baseline_and_rerun_alfred_corrected_2026_04_20/"
    "01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator"
)

LEGACY_VOLUME_PATH = f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data"
MARKET_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"
VIX3M_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix3m_1h"
VIX_SPOT_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix_1h"

CORRECTED_BRONZE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_legacy_recreated_corrected"
)
CORRECTED_SILVER_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_recreated_corrected"
)
BASELINE_SIGNAL_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_recreated_baseline"
)
CORRECTED_SIGNAL_TABLE_DEFAULT = (
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

REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession/Contraction"}


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


def file_sha256(path_str):
    path = Path(path_str)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    pdf["release_date"] = pd.to_datetime(
        pdf["release_date"], errors="coerce"
    ).dt.date.astype(str)
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


baseline_control_root = BASELINE_CONTROL_ROOT_DEFAULT
comparison_output_root = COMPARISON_OUTPUT_ROOT_DEFAULT
performance_notebook_path = PERFORMANCE_NOTEBOOK_PATH_DEFAULT
market_table = MARKET_TABLE_DEFAULT
vix3m_table = VIX3M_TABLE_DEFAULT
vix_spot_table = VIX_SPOT_TABLE_DEFAULT
corrected_bronze_table = CORRECTED_BRONZE_TABLE_DEFAULT
corrected_silver_table = CORRECTED_SILVER_TABLE_DEFAULT
baseline_signal_table = BASELINE_SIGNAL_TABLE_DEFAULT
corrected_signal_table = CORRECTED_SIGNAL_TABLE_DEFAULT
train_start = TRAIN_START_DEFAULT
live_start = LIVE_START_DEFAULT
live_end = LIVE_END_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("baseline_control_root", BASELINE_CONTROL_ROOT_DEFAULT)
    dbutils.widgets.text("comparison_output_root", COMPARISON_OUTPUT_ROOT_DEFAULT)
    dbutils.widgets.text("performance_notebook_path", PERFORMANCE_NOTEBOOK_PATH_DEFAULT)
    dbutils.widgets.text("market_table", MARKET_TABLE_DEFAULT)
    dbutils.widgets.text("vix3m_table", VIX3M_TABLE_DEFAULT)
    dbutils.widgets.text("vix_spot_table", VIX_SPOT_TABLE_DEFAULT)
    dbutils.widgets.text("corrected_bronze_table", CORRECTED_BRONZE_TABLE_DEFAULT)
    dbutils.widgets.text("corrected_silver_table", CORRECTED_SILVER_TABLE_DEFAULT)
    dbutils.widgets.text("baseline_signal_table", BASELINE_SIGNAL_TABLE_DEFAULT)
    dbutils.widgets.text("corrected_signal_table", CORRECTED_SIGNAL_TABLE_DEFAULT)
    dbutils.widgets.text("train_start", TRAIN_START_DEFAULT)
    dbutils.widgets.text("live_start", LIVE_START_DEFAULT)
    dbutils.widgets.text("live_end", LIVE_END_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    baseline_control_root = (
        dbutils.widgets.get("baseline_control_root").strip()
        or BASELINE_CONTROL_ROOT_DEFAULT
    )
    comparison_output_root = (
        dbutils.widgets.get("comparison_output_root").strip()
        or COMPARISON_OUTPUT_ROOT_DEFAULT
    )
    performance_notebook_path = (
        dbutils.widgets.get("performance_notebook_path").strip()
        or PERFORMANCE_NOTEBOOK_PATH_DEFAULT
    )
    market_table = dbutils.widgets.get("market_table").strip() or MARKET_TABLE_DEFAULT
    vix3m_table = dbutils.widgets.get("vix3m_table").strip() or VIX3M_TABLE_DEFAULT
    vix_spot_table = dbutils.widgets.get("vix_spot_table").strip() or VIX_SPOT_TABLE_DEFAULT
    corrected_bronze_table = (
        dbutils.widgets.get("corrected_bronze_table").strip()
        or CORRECTED_BRONZE_TABLE_DEFAULT
    )
    corrected_silver_table = (
        dbutils.widgets.get("corrected_silver_table").strip()
        or CORRECTED_SILVER_TABLE_DEFAULT
    )
    baseline_signal_table = (
        dbutils.widgets.get("baseline_signal_table").strip()
        or BASELINE_SIGNAL_TABLE_DEFAULT
    )
    corrected_signal_table = (
        dbutils.widgets.get("corrected_signal_table").strip()
        or CORRECTED_SIGNAL_TABLE_DEFAULT
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


corrected_variant_root = f"{comparison_output_root}/corrected_variant"
comparison_dir = f"{comparison_output_root}/comparison"
ensure_dir(corrected_variant_root)
ensure_dir(comparison_dir)

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
        pd.to_numeric(audit_pdf["actual"], errors="coerce").round(12)
        != pd.to_numeric(audit_pdf["baseline_actual"], errors="coerce").round(12)
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
        "changed_rows": int(audit_pdf["actual_changed_vs_baseline"].sum()),
        "changed_pct": float(audit_pdf["actual_changed_vs_baseline"].mean()),
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


def run_walkforward_variant(silver_table, label_path, signal_table, output_dir):
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

    for index in range(len(retrain_dates) - 1):
        train_end = retrain_dates[index]
        test_end = retrain_dates[index + 1]

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
        raise ValueError("No walk-forward signals generated for corrected variant.")

    signals_df = pd.concat(signals, ignore_index=True).sort_values("timestamp")
    signals_df["regime_name"] = signals_df["predicted_regime"].map(REGIME_NAME_MAP)
    signals_path = f"{output_dir}/walkforward_signals.csv"
    signals_df.to_csv(signals_path, index=False)

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
    monthly_path = f"{output_dir}/monthly_validation.csv"
    monthly_pdf.to_csv(monthly_path, index=False)

    write_delta_table_from_pandas(
        signals_df[["timestamp", "predicted_regime"]],
        signal_table,
    )

    return {
        "signals_path": signals_path,
        "monthly_validation_path": monthly_path,
        "signal_rows": int(len(signals_df)),
        "mean_monthly_accuracy": float(monthly_pdf["monthly_accuracy"].mean())
        if not monthly_pdf.empty
        else np.nan,
        "signal_table_stats": table_row_range(signal_table),
    }


def load_performance_metrics(root_path):
    return pd.read_csv(f"{root_path}/performance_metrics.csv").iloc[0].to_dict()


def load_curve(root_path, filename, value_column):
    pdf = pd.read_csv(f"{root_path}/{filename}", parse_dates=["timestamp"])
    pdf = normalize_timestamp(pdf)
    return pdf[["timestamp", value_column]].sort_values("timestamp").reset_index(drop=True)


def load_signal_frame_from_table(table_name):
    pdf = spark.table(table_name).select("timestamp", "predicted_regime").toPandas()
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    pdf["predicted_regime"] = pd.to_numeric(
        pdf["predicted_regime"],
        errors="coerce",
    ).fillna(2).astype(int)
    return pdf.sort_values("timestamp").reset_index(drop=True)


def recommendation_text(metrics_diff, signal_match_rate):
    cagr_delta = metrics_diff.get("CAGR")
    sharpe_delta = metrics_diff.get("Sharpe")
    max_dd_delta = metrics_diff.get("Max Drawdown")

    if (
        pd.notna(cagr_delta)
        and pd.notna(sharpe_delta)
        and pd.notna(max_dd_delta)
        and cagr_delta > 0.005
        and sharpe_delta > 0.03
        and max_dd_delta >= -0.02
    ):
        return "prefer corrected ALFRED"

    if (
        pd.notna(cagr_delta)
        and pd.notna(sharpe_delta)
        and cagr_delta < -0.005
        and sharpe_delta < -0.03
    ):
        return "keep baseline"

    if signal_match_rate < 0.90:
        return "further review needed"

    return "further review needed"


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


baseline_required_files = [
    "performance_metrics.csv",
    "daily_equity_curve.csv",
    "hourly_equity_curve.csv",
    "drawdown_curve.csv",
    "regime_stats.csv",
    "positive_periods.csv",
    "summary.md",
    "methodology_notes.md",
    "run_metadata.json",
]

missing_baseline = [
    filename
    for filename in baseline_required_files
    if not Path(f"{baseline_control_root}/{filename}").exists()
]
if missing_baseline:
    raise FileNotFoundError(
        f"Baseline control root missing required files: {missing_baseline}"
    )

baseline_performance_metrics = load_performance_metrics(baseline_control_root)
baseline_run_metadata = json.loads(Path(f"{baseline_control_root}/run_metadata.json").read_text())
baseline_hashes = {
    filename: file_sha256(f"{baseline_control_root}/{filename}")
    for filename in baseline_required_files
}

control_manifest_path = f"{baseline_control_root}/control_manifest.json"
control_notes_path = f"{baseline_control_root}/control_notes.md"

control_manifest = {
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "generator_notebook_repo_path": "01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator.py",
    "generator_workspace_path": performance_notebook_path,
    "baseline_output_root": baseline_control_root,
    "signal_source": baseline_run_metadata.get("signal_source"),
    "price_source": baseline_run_metadata.get("input_tables", {}).get("market_table"),
    "execution_assumptions": baseline_run_metadata.get("methodology"),
    "performance_metrics": baseline_performance_metrics,
    "artifact_hashes": baseline_hashes,
}
write_json(control_manifest_path, control_manifest)

control_notes_lines = [
    "# Canonical Legacy V4 Baseline Control Notes",
    "",
    "- This control captures the frozen canonical legacy-V4 performance-table generator output for the recreated baseline strategy.",
    "- The baseline methodology is not modified in this task.",
    f"- Baseline output root: `{baseline_control_root}`",
    f"- Signal source: `{baseline_run_metadata.get('signal_source', {}).get('source')}`",
    f"- Market price source: `{market_table}`",
    "- Execution assumptions remain: hourly signals, hourly prices, prior-bar execution, QQQ/SPY/CASH mapping, 5 bps subtractive switch cost, hourly equity path, daily-resampled reporting, 365-day CAGR convention.",
    f"- Frozen baseline final value: `{baseline_performance_metrics['Final Value']}`",
    f"- Frozen baseline CAGR: `{baseline_performance_metrics['CAGR']}`",
    f"- Frozen baseline Sharpe: `{baseline_performance_metrics['Sharpe']}`",
    f"- Frozen baseline Max Drawdown: `{baseline_performance_metrics['Max Drawdown']}`",
]
write_text(control_notes_path, "\n".join(control_notes_lines) + "\n")

corrected_macro_info = build_corrected_bronze(corrected_bronze_table, corrected_variant_root)
corrected_silver_stats = build_legacy_silver(corrected_bronze_table, corrected_silver_table)
corrected_full_signal_pdf, corrected_full_signal_path = train_full_signal_model(
    corrected_silver_table,
    corrected_variant_root,
)
corrected_walkforward_info = run_walkforward_variant(
    corrected_silver_table,
    corrected_full_signal_path,
    corrected_signal_table,
    corrected_variant_root,
)

performance_run_result = None
if "dbutils" in globals():
    performance_run_result = json.loads(
        dbutils.notebook.run(
            performance_notebook_path,
            7200,
            {
                "output_root": corrected_variant_root,
                "signal_table": corrected_signal_table,
                "signal_csv_path": corrected_walkforward_info["signals_path"],
                "market_table": market_table,
                "live_start": live_start,
                "live_end": live_end,
                "initial_capital": str(initial_capital),
                "transaction_cost": str(transaction_cost),
            },
        )
    )
else:
    raise RuntimeError("This notebook must run on Databricks because it calls dbutils.notebook.run.")

corrected_performance_metrics = load_performance_metrics(corrected_variant_root)
corrected_run_metadata = json.loads(Path(f"{corrected_variant_root}/run_metadata.json").read_text())

baseline_current_hashes = {
    filename: file_sha256(f"{baseline_control_root}/{filename}")
    for filename in baseline_required_files
}
baseline_control_unchanged = baseline_current_hashes == baseline_hashes

baseline_signal_pdf = load_signal_frame_from_table(baseline_signal_table)
corrected_signal_pdf = load_signal_frame_from_table(corrected_signal_table)
signal_diff_pdf = baseline_signal_pdf.rename(
    columns={"predicted_regime": "predicted_regime_baseline"}
).merge(
    corrected_signal_pdf.rename(columns={"predicted_regime": "predicted_regime_corrected"}),
    on="timestamp",
    how="outer",
).sort_values("timestamp")
signal_diff_pdf["regime_changed"] = (
    signal_diff_pdf["predicted_regime_baseline"]
    != signal_diff_pdf["predicted_regime_corrected"]
)
signal_match_rate = float(1.0 - signal_diff_pdf["regime_changed"].fillna(True).mean())
regime_mismatch_count = int(signal_diff_pdf["regime_changed"].fillna(True).sum())
signal_diff_path = f"{comparison_dir}/signal_diff.csv"
signal_diff_pdf.to_csv(signal_diff_path, index=False)

metric_names = [
    "Final Value",
    "Total Return",
    "CAGR",
    "Volatility",
    "Sharpe",
    "Sortino",
    "Max Drawdown",
    "Total Regime Switches",
    "Average Holding Bars",
    "Positive Weeks",
    "Positive Months",
    "Positive Years",
]
metrics_diff_rows = []
metrics_delta_lookup = {}
for metric_name in metric_names:
    baseline_value = baseline_performance_metrics.get(metric_name)
    corrected_value = corrected_performance_metrics.get(metric_name)
    delta_value = (
        corrected_value - baseline_value
        if pd.notna(baseline_value) and pd.notna(corrected_value)
        else np.nan
    )
    metrics_diff_rows.append(
        {
            "metric": metric_name,
            "baseline": baseline_value,
            "corrected_alfred": corrected_value,
            "corrected_minus_baseline": delta_value,
        }
    )
    metrics_delta_lookup[metric_name] = delta_value

metrics_diff_pdf = pd.DataFrame(metrics_diff_rows)
metrics_diff_path = f"{comparison_dir}/metrics_diff.csv"
metrics_diff_pdf.to_csv(metrics_diff_path, index=False)

baseline_equity_pdf = load_curve(baseline_control_root, "daily_equity_curve.csv", "equity").assign(
    variant="baseline"
)
corrected_equity_pdf = load_curve(corrected_variant_root, "daily_equity_curve.csv", "equity").assign(
    variant="corrected_alfred"
)
equity_curve_pdf = pd.concat(
    [baseline_equity_pdf, corrected_equity_pdf],
    ignore_index=True,
)
equity_curve_path = f"{comparison_dir}/equity_curve.csv"
equity_curve_pdf.to_csv(equity_curve_path, index=False)

baseline_drawdown_pdf = load_curve(
    baseline_control_root,
    "drawdown_curve.csv",
    "drawdown",
).assign(variant="baseline")
corrected_drawdown_pdf = load_curve(
    corrected_variant_root,
    "drawdown_curve.csv",
    "drawdown",
).assign(variant="corrected_alfred")
drawdown_curve_pdf = pd.concat(
    [baseline_drawdown_pdf, corrected_drawdown_pdf],
    ignore_index=True,
)
drawdown_curve_path = f"{comparison_dir}/drawdown_curve.csv"
drawdown_curve_pdf.to_csv(drawdown_curve_path, index=False)

recommendation = recommendation_text(metrics_delta_lookup, signal_match_rate)

summary_lines = [
    "# Canonical Legacy V4 Baseline vs Corrected ALFRED",
    "",
    "## Control Validation",
    "",
    f"- Baseline control root: `{baseline_control_root}`",
    f"- Baseline control unchanged: `{baseline_control_unchanged}`",
    f"- Frozen control manifest: `{control_manifest_path}`",
    "",
    "## Corrected Variant Rebuild",
    "",
    f"- Corrected bronze table: `{corrected_bronze_table}`",
    f"- Corrected silver table: `{corrected_silver_table}`",
    f"- Corrected signal table: `{corrected_signal_table}`",
    f"- Corrected bronze changed rows vs baseline release CSV layer: `{corrected_macro_info['changed_rows']}`",
    f"- Corrected bronze changed row rate: `{corrected_macro_info['changed_pct']:.6f}`",
    f"- Canonical generator notebook used for corrected performance: `{performance_notebook_path}`",
    "",
    "## Performance Comparison",
    "",
    markdown_table(metrics_diff_pdf),
    "",
    "## Signal Comparison",
    "",
    f"- Signal match rate: `{signal_match_rate:.6f}`",
    f"- Regime mismatch count: `{regime_mismatch_count}`",
    "",
    "## Recommendation",
    "",
    f"- `{recommendation}`",
]
summary_path = f"{comparison_dir}/summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "recommendation": recommendation,
    "baseline_control_unchanged": baseline_control_unchanged,
    "corrected_variant_metrics": corrected_performance_metrics,
    "metrics_delta": metrics_delta_lookup,
    "signal_match_rate": signal_match_rate,
    "regime_mismatch_count": regime_mismatch_count,
}
recommendation_path = f"{comparison_dir}/recommendation.json"
write_json(recommendation_path, recommendation_payload)

plt.figure(figsize=(14, 7))
for variant, group in equity_curve_pdf.groupby("variant"):
    plt.plot(group["timestamp"], group["equity"], label=variant)
plt.title("Baseline vs Corrected ALFRED Daily Equity Curve")
plt.ylabel("Equity")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
equity_plot_path = f"{comparison_dir}/equity_curve_comparison.png"
plt.savefig(equity_plot_path, dpi=180)
plt.close()

plt.figure(figsize=(14, 7))
for variant, group in drawdown_curve_pdf.groupby("variant"):
    plt.plot(group["timestamp"], group["drawdown"], label=variant)
plt.title("Baseline vs Corrected ALFRED Daily Drawdown")
plt.ylabel("Drawdown")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
drawdown_plot_path = f"{comparison_dir}/drawdown_curve.png"
plt.savefig(drawdown_plot_path, dpi=180)
plt.close()

timeline_pdf = signal_diff_pdf[
    ["timestamp", "predicted_regime_baseline", "predicted_regime_corrected"]
].dropna().copy()
plt.figure(figsize=(16, 5))
plt.step(
    timeline_pdf["timestamp"],
    timeline_pdf["predicted_regime_baseline"],
    where="post",
    label="baseline",
)
plt.step(
    timeline_pdf["timestamp"],
    timeline_pdf["predicted_regime_corrected"] + 0.05,
    where="post",
    label="corrected_alfred",
)
plt.yticks([0, 1, 2], ["Expansion", "Slowdown", "Recession/Contraction"])
plt.title("Baseline vs Corrected ALFRED Regime Timeline")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
regime_timeline_path = f"{comparison_dir}/regime_timeline.png"
plt.savefig(regime_timeline_path, dpi=180)
plt.close()

comparison_run_metadata = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "baseline_control_root": baseline_control_root,
    "comparison_output_root": comparison_output_root,
    "baseline_control_unchanged": baseline_control_unchanged,
    "same_canonical_generator_used": True,
    "generator_repo_path": "01_Processing/04_Retraining/17_Canonical_Legacy_V4_Performance_Table_Generator.py",
    "generator_workspace_path": performance_notebook_path,
    "input_tables": {
        "market_table": table_row_range(market_table),
        "corrected_bronze_table": corrected_macro_info["table_stats"],
        "corrected_silver_table": corrected_silver_stats,
        "baseline_signal_table": table_row_range(baseline_signal_table),
        "corrected_signal_table": corrected_walkforward_info["signal_table_stats"],
    },
    "corrected_rebuild": {
        "corrected_bronze_table": corrected_bronze_table,
        "corrected_silver_table": corrected_silver_table,
        "corrected_signal_table": corrected_signal_table,
        "changed_rows": corrected_macro_info["changed_rows"],
        "changed_pct": corrected_macro_info["changed_pct"],
        "signal_rows": corrected_walkforward_info["signal_rows"],
        "mean_monthly_accuracy": corrected_walkforward_info["mean_monthly_accuracy"],
        "performance_run_result": performance_run_result,
    },
    "comparison": {
        "signal_match_rate": signal_match_rate,
        "regime_mismatch_count": regime_mismatch_count,
        "recommendation": recommendation,
        "metrics_delta": metrics_delta_lookup,
    },
    "artifacts": {
        "control_manifest": control_manifest_path,
        "control_notes": control_notes_path,
        "corrected_variant_root": corrected_variant_root,
        "metrics_diff": metrics_diff_path,
        "signal_diff": signal_diff_path,
        "equity_curve": equity_curve_path,
        "drawdown_curve": drawdown_curve_path,
        "summary": summary_path,
        "recommendation": recommendation_path,
        "equity_plot": equity_plot_path,
        "drawdown_plot": drawdown_plot_path,
        "regime_timeline_plot": regime_timeline_path,
    },
}
comparison_run_metadata_path = f"{comparison_dir}/run_metadata.json"
write_json(comparison_run_metadata_path, comparison_run_metadata)

result = {
    "phase": "freeze_canonical_baseline_and_rerun_alfred_corrected",
    "status": "success",
    "baseline_control_unchanged": baseline_control_unchanged,
    "recommendation": recommendation,
    "baseline_control_root": baseline_control_root,
    "corrected_variant_root": corrected_variant_root,
    "comparison_root": comparison_dir,
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

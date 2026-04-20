# Databricks notebook source
# MAGIC %pip install fredapi xgboost

# COMMAND ----------

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from pyspark.sql import functions as F

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
LEGACY_SCHEMA = "default"

COMPARISON_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/comparison"
DEFAULT_OUTPUT_ROOT = f"{COMPARISON_ROOT}/2026-04-20_legacy_macro_repair"
LEGACY_VOLUME_PATH = f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data"
HAYSTACK_PATH_DEFAULT = f"{LEGACY_VOLUME_PATH}/raw_finnhub_haystack.csv"
LEGACY_WF_SUMMARY_PATH_DEFAULT = f"{LEGACY_VOLUME_PATH}/strategy_performance_summary.csv"
LEGACY_SIGNAL_LABEL_PATH_DEFAULT = f"{LEGACY_VOLUME_PATH}/bt_fred_1d_regime_signals-v4.csv"
PARTIAL_BASELINE_METRICS_PATH_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_legacy_v4_alfred_swap/comparison/legacy_baseline_metrics.csv"
)

MARKET_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"
VIX3M_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix3m_1h"
VIX_SPOT_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix_1h"
REFERENCE_WF_SIGNAL_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bt_walkforward_regime_signals"

BRONZE_BASELINE_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_legacy_baseline"
SILVER_BASELINE_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_baseline"
WF_SIGNAL_BASELINE_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_baseline"

TRAIN_START_DEFAULT = "2016-01-01"
LIVE_START_DEFAULT = "2021-01-01"
LIVE_END_DEFAULT = "2026-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252
VIX_SOURCE = "fred_1d"
FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

INDICATOR_CONFIG = {
    "CPIAUCSL": {
        "typical_time": "08:30:00",
        "patterns": [r"\bcpi mm, sa\b", r"consumer price index"],
        "feature_column": "CPIAUCSL_z",
        "note": "CPI schedule comes from monthly CPI MoM release rows.",
    },
    "PCEPI": {
        "typical_time": "08:30:00",
        "patterns": [
            r"core pce price index",
            r"pce price index mom",
            r"consumption-core pce",
        ],
        "feature_column": "PCEPI_z",
        "note": "Finnhub naming drift uses MoM / YY variants rather than the old exact strings.",
    },
    "PCE": {
        "typical_time": "08:30:00",
        "patterns": [
            r"personal consumption expenditures",
            r"pce price index mom",
            r"core pce price index",
            r"consumption-core pce",
        ],
        "schedule_from": "PCEPI",
        "feature_column": "PCE_z",
        "note": "Legacy stack uses the same release bucket schedule as the PCE/PCEPI report.",
    },
    "PAYEMS": {
        "typical_time": "08:30:00",
        "patterns": [r"non[ -]?farm payrolls"],
        "feature_column": "PAYEMS_z",
        "note": "Legacy stack must capture both 'Non-Farm' and 'Non Farm' naming.",
    },
    "UNRATE": {
        "typical_time": "08:30:00",
        "patterns": [r"unemployment rate"],
        "feature_column": "UNRATE_z",
        "note": "Unemployment rate rows remain direct matches.",
    },
    "GDPC1": {
        "typical_time": "08:30:00",
        "patterns": [
            r"\bgdp advance\b",
            r"\bgdp 2nd estimate\b",
            r"\bgdp final\b",
            r"gdp growth rate qoq",
            r"united states-gdp",
        ],
        "feature_column": "GDPC1_z",
        "note": "GDP requires broad release-name matching across advance / second / final variants.",
    },
    "INDPRO": {
        "typical_time": "09:15:00",
        "patterns": [r"industrial production"],
        "feature_column": "INDPRO_z",
        "note": "Finnhub uses MoM / YoY naming for Industrial Production.",
    },
    "UMCSENT": {
        "typical_time": "10:00:00",
        "patterns": [r"u mich sentiment", r"michigan consumer sentiment"],
        "feature_column": "UMCSENT_z",
        "note": "Legacy stack includes both preliminary and final Michigan sentiment releases.",
    },
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


def parse_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


output_root = DEFAULT_OUTPUT_ROOT
haystack_path = HAYSTACK_PATH_DEFAULT
market_table = MARKET_TABLE_DEFAULT
vix3m_table = VIX3M_TABLE_DEFAULT
vix_spot_table = VIX_SPOT_TABLE_DEFAULT
reference_wf_signal_table = REFERENCE_WF_SIGNAL_TABLE_DEFAULT
legacy_wf_summary_path = LEGACY_WF_SUMMARY_PATH_DEFAULT
legacy_signal_label_path = LEGACY_SIGNAL_LABEL_PATH_DEFAULT
partial_baseline_metrics_path = PARTIAL_BASELINE_METRICS_PATH_DEFAULT
bronze_baseline_table = BRONZE_BASELINE_TABLE_DEFAULT
silver_baseline_table = SILVER_BASELINE_TABLE_DEFAULT
wf_signal_baseline_table = WF_SIGNAL_BASELINE_TABLE_DEFAULT
train_start = TRAIN_START_DEFAULT
live_start = LIVE_START_DEFAULT
live_end = LIVE_END_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("output_root", DEFAULT_OUTPUT_ROOT)
    dbutils.widgets.text("haystack_path", HAYSTACK_PATH_DEFAULT)
    dbutils.widgets.text("market_table", MARKET_TABLE_DEFAULT)
    dbutils.widgets.text("vix3m_table", VIX3M_TABLE_DEFAULT)
    dbutils.widgets.text("vix_spot_table", VIX_SPOT_TABLE_DEFAULT)
    dbutils.widgets.text("reference_wf_signal_table", REFERENCE_WF_SIGNAL_TABLE_DEFAULT)
    dbutils.widgets.text("legacy_wf_summary_path", LEGACY_WF_SUMMARY_PATH_DEFAULT)
    dbutils.widgets.text("legacy_signal_label_path", LEGACY_SIGNAL_LABEL_PATH_DEFAULT)
    dbutils.widgets.text(
        "partial_baseline_metrics_path",
        PARTIAL_BASELINE_METRICS_PATH_DEFAULT,
    )
    dbutils.widgets.text("bronze_baseline_table", BRONZE_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("silver_baseline_table", SILVER_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("wf_signal_baseline_table", WF_SIGNAL_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("train_start", TRAIN_START_DEFAULT)
    dbutils.widgets.text("live_start", LIVE_START_DEFAULT)
    dbutils.widgets.text("live_end", LIVE_END_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    output_root = dbutils.widgets.get("output_root").strip() or DEFAULT_OUTPUT_ROOT
    haystack_path = dbutils.widgets.get("haystack_path").strip() or HAYSTACK_PATH_DEFAULT
    market_table = dbutils.widgets.get("market_table").strip() or MARKET_TABLE_DEFAULT
    vix3m_table = dbutils.widgets.get("vix3m_table").strip() or VIX3M_TABLE_DEFAULT
    vix_spot_table = dbutils.widgets.get("vix_spot_table").strip() or VIX_SPOT_TABLE_DEFAULT
    reference_wf_signal_table = (
        dbutils.widgets.get("reference_wf_signal_table").strip()
        or REFERENCE_WF_SIGNAL_TABLE_DEFAULT
    )
    legacy_wf_summary_path = (
        dbutils.widgets.get("legacy_wf_summary_path").strip()
        or LEGACY_WF_SUMMARY_PATH_DEFAULT
    )
    legacy_signal_label_path = (
        dbutils.widgets.get("legacy_signal_label_path").strip()
        or LEGACY_SIGNAL_LABEL_PATH_DEFAULT
    )
    partial_baseline_metrics_path = (
        dbutils.widgets.get("partial_baseline_metrics_path").strip()
        or PARTIAL_BASELINE_METRICS_PATH_DEFAULT
    )
    bronze_baseline_table = (
        dbutils.widgets.get("bronze_baseline_table").strip()
        or BRONZE_BASELINE_TABLE_DEFAULT
    )
    silver_baseline_table = (
        dbutils.widgets.get("silver_baseline_table").strip()
        or SILVER_BASELINE_TABLE_DEFAULT
    )
    wf_signal_baseline_table = (
        dbutils.widgets.get("wf_signal_baseline_table").strip()
        or WF_SIGNAL_BASELINE_TABLE_DEFAULT
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
baseline_dir = f"{output_root}/repaired_legacy_baseline"
comparison_dir = f"{output_root}/comparison"
indicator_csv_dir = f"{baseline_dir}/indicator_csvs"
for folder in [baseline_dir, comparison_dir, indicator_csv_dir]:
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
    ).dt.date
    vintages["observation_date"] = pd.to_datetime(
        vintages["observation_date"],
        errors="coerce",
    )
    vintages["value_num"] = pd.to_numeric(vintages["value"], errors="coerce")
    alfred_release_cache[series_id] = vintages
    return vintages


def apply_precise_timing(indicator_id, release_date):
    wall_clock_time = INDICATOR_CONFIG[indicator_id]["typical_time"]
    naive_dt = datetime.strptime(
        f"{release_date} {wall_clock_time}",
        "%Y-%m-%d %H:%M:%S",
    )
    eastern = pytz.timezone("US/Eastern")
    localized_dt = eastern.localize(naive_dt)
    return localized_dt.astimezone(pytz.utc).replace(tzinfo=None)


def load_haystack(path_str):
    pdf = load_csv(path_str)
    if pdf.empty:
        raise FileNotFoundError(f"Missing or empty haystack file: {path_str}")
    pdf["time"] = pd.to_datetime(pdf["time"], errors="coerce")
    pdf = pdf.dropna(subset=["time"])
    pdf = pdf.drop_duplicates(subset=["time", "event"], keep="last")
    pdf["release_date"] = pdf["time"].dt.date.astype(str)
    return pdf.sort_values("time")


def build_haystack_match_mask(haystack_df, indicator_id):
    mask = pd.Series(False, index=haystack_df.index)
    for pattern in INDICATOR_CONFIG[indicator_id]["patterns"]:
        mask = mask | haystack_df["event"].astype(str).str.contains(
            pattern,
            case=False,
            na=False,
            regex=True,
        )
    return mask


def collect_haystack_matches(haystack_df, indicator_id, release_date_override=None):
    source_indicator = release_date_override or indicator_id
    mask = build_haystack_match_mask(haystack_df, source_indicator)
    matches = haystack_df.loc[mask].copy()
    if matches.empty:
        return matches
    matches = matches.sort_values(["release_date", "time", "event"])
    matches = matches.drop_duplicates(subset=["release_date"], keep="first")
    matches["indicator_id"] = indicator_id
    matches["schedule_source_indicator"] = source_indicator
    return matches


def source_indicator_csv_path(indicator_id):
    return f"{LEGACY_VOLUME_PATH}/first_releases_{indicator_id}.csv"


def load_source_indicator_csv(indicator_id):
    path_str = source_indicator_csv_path(indicator_id)
    pdf = load_csv(path_str)
    if pdf.empty:
        return pdf, path_str
    pdf["release_date"] = pd.to_datetime(pdf["release_date"], errors="coerce").dt.date.astype(str)
    pdf["release_datetime"] = pd.to_datetime(pdf["release_datetime"], errors="coerce")
    pdf["actual"] = pd.to_numeric(pdf["value"], errors="coerce")
    return pdf, path_str


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


def copy_source_indicator_csvs():
    status_rows = []
    for indicator_id in INDICATOR_CONFIG:
        source_pdf, source_path = load_source_indicator_csv(indicator_id)
        target_path = f"{indicator_csv_dir}/first_releases_{indicator_id}.csv"
        if Path(source_path).exists():
            shutil.copyfile(source_path, target_path)
        status_rows.append(
            {
                "indicator_id": indicator_id,
                "source_csv_path": source_path,
                "source_csv_exists": Path(source_path).exists(),
                "source_csv_rows": int(len(source_pdf)),
                "source_csv_numeric_rows": int(source_pdf["actual"].notna().sum()) if "actual" in source_pdf.columns else 0,
                "source_csv_release_dates": int(source_pdf["release_date"].nunique()) if "release_date" in source_pdf.columns else 0,
                "copied_csv_path": target_path if Path(source_path).exists() else None,
            }
        )
    status_pdf = pd.DataFrame(status_rows).sort_values("indicator_id")
    status_path = f"{comparison_dir}/indicator_csv_status.csv"
    status_pdf.to_csv(status_path, index=False)
    return status_pdf, status_path


def build_repaired_macro_bronze(target_table):
    bronze_rows = []
    bronze_summary_rows = []
    for indicator_id in INDICATOR_CONFIG:
        source_pdf, _ = load_source_indicator_csv(indicator_id)
        if source_pdf.empty:
            raise FileNotFoundError(f"Missing legacy source CSV for {indicator_id}")

        numeric_pdf = source_pdf.dropna(subset=["release_datetime", "actual"]).copy()
        numeric_pdf = normalize_timestamp(numeric_pdf, "release_datetime")
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
        bronze_summary_rows.append(
            {
                "indicator_id": indicator_id,
                "bronze_rows_from_source_csv": int(len(numeric_pdf)),
                "bronze_unique_timestamps": int(numeric_pdf["timestamp"].nunique()),
                "bronze_min_timestamp": numeric_pdf["timestamp"].min(),
                "bronze_max_timestamp": numeric_pdf["timestamp"].max(),
            }
        )

    bronze_pdf = pd.concat(bronze_rows, ignore_index=True).sort_values(
        ["indicator_id", "timestamp"]
    )
    write_delta_table_from_pandas(
        bronze_pdf,
        target_table,
        partition_cols=["indicator_id"],
    )

    bronze_indicator_summary_pdf = pd.DataFrame(bronze_summary_rows).sort_values("indicator_id")
    bronze_indicator_summary_path = f"{comparison_dir}/bronze_indicator_summary.csv"
    bronze_indicator_summary_pdf.to_csv(bronze_indicator_summary_path, index=False)
    return table_row_range(target_table), bronze_indicator_summary_pdf, bronze_indicator_summary_path


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
        subset=["timestamp"], keep="last"
    )
    market_pdf = market_pdf.sort_values("timestamp")
    end_date = market_pdf["timestamp"].max()

    backbone_pdf = pd.DataFrame(
        {"timestamp": pd.date_range(start=start_date, end=end_date, freq="H")}
    )

    macro_pdf = spark.table(macro_table).toPandas()
    macro_pdf = normalize_timestamp(macro_pdf)
    macro_pdf = macro_pdf.dropna(subset=["timestamp"])
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
        subset=["timestamp"], keep="last"
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
        subset=["timestamp"], keep="last"
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

    final_silver = silver_grid[
        ["timestamp", "SPY_close"] + FEATURES
    ].copy()
    final_silver = final_silver[final_silver["SPY_close"].notna()].reset_index(drop=True)
    write_delta_table_from_pandas(final_silver, target_table)
    return table_row_range(target_table), final_silver


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

    X_full = df[FEATURES].fillna(0)
    df["predicted_regime"] = model.predict(X_full)
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
    pdf = pdf.dropna(subset=["timestamp"])
    pdf = pdf.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    pivoted = (
        pdf.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
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
    feature_importance_history = []
    signals = []
    monthly_rows = []

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
            {"feature": list(importance_dict.keys()), "importance": list(importance_dict.values())}
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
        raise ValueError("No walk-forward signals generated")

    signals_df = pd.concat(signals, ignore_index=True).sort_values("timestamp")
    signals_df["regime_name"] = signals_df["predicted_regime"].map(REGIME_NAME_MAP)
    signals_df.to_csv(f"{output_dir}/walkforward_signals.csv", index=False)

    fi_df = pd.concat(feature_importance_history, ignore_index=True)
    if not fi_df.empty:
        fi_df["importance"] = fi_df.groupby("date")["importance"].transform(
            lambda x: x / x.sum() if x.sum() else x
        )
    fi_df.to_csv(f"{output_dir}/feature_importance_monthly.csv", index=False)

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
    metrics["Signal Rows"] = int(len(signals_df))
    metrics["Mean Monthly Accuracy"] = (
        float(monthly_pdf["monthly_accuracy"].mean()) if not monthly_pdf.empty else np.nan
    )
    metrics["Total Switches"] = int(backtest_df["position_changed"].sum())
    holding_lengths = (
        backtest_df["active_asset"].ne(backtest_df["active_asset"].shift()).cumsum().value_counts()
    )
    metrics["Average Holding Bars"] = (
        float(holding_lengths.mean()) if not holding_lengths.empty else np.nan
    )

    return {
        "signals": signals_df,
        "backtest": backtest_df,
        "daily_curve": strategy_series.resample("D").last().dropna(),
        "metrics": metrics,
        "monthly_validation": monthly_pdf,
    }


def load_reference_walkforward_signals():
    pdf = spark.table(reference_wf_signal_table).toPandas()
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf[
        (pdf["timestamp"] >= pd.Timestamp(live_start))
        & (pdf["timestamp"] < pd.Timestamp(live_end))
    ].copy()
    pdf["predicted_regime"] = pd.to_numeric(pdf["predicted_regime"], errors="coerce").fillna(2).astype(int)
    return pdf[["timestamp", "predicted_regime"]]


def load_reference_performance_summary():
    pdf = load_csv(legacy_wf_summary_path)
    if pdf.empty:
        return {}
    row_name_col = "Unnamed: 0" if "Unnamed: 0" in pdf.columns else pdf.columns[0]
    row = pdf[pdf[row_name_col] == "WalkForward_V4"]
    if row.empty:
        return {}
    row = row.iloc[0].to_dict()
    return {
        "Start Value": float(row.get("Start Value", np.nan)),
        "Final Value": float(row.get("Final Value", np.nan)),
        "Total Return": float(row.get("Total Return", np.nan)),
        "CAGR": float(row.get("CAGR", np.nan)),
        "Vol Annual": float(row.get("Vol Annual", np.nan)),
        "Sharpe Annual": float(row.get("Sharpe Annual", np.nan)),
        "Sortino Annual": float(row.get("Sortino Annual", np.nan)),
        "Max Drawdown": float(row.get("Max Drawdown", np.nan)),
    }


def load_partial_baseline_metrics():
    pdf = load_csv(partial_baseline_metrics_path)
    if pdf.empty:
        return {}
    row = pdf.iloc[0].to_dict()
    return {
        "Final Value": float(row.get("Final Value", np.nan)),
        "CAGR": float(row.get("CAGR", np.nan)),
        "Sharpe Annual": float(row.get("Sharpe Annual", np.nan)),
        "Max Drawdown": float(row.get("Max Drawdown", np.nan)),
    }


def build_silver_feature_presence(final_silver_pdf):
    rows = []
    for indicator_id, cfg in INDICATOR_CONFIG.items():
        feature_col = cfg["feature_column"]
        rows.append(
            {
                "indicator_id": indicator_id,
                "silver_feature": feature_col,
                "silver_feature_non_null_rows": int(final_silver_pdf[feature_col].notna().sum()),
            }
        )
    rows.extend(
        [
            {
                "indicator_id": "VIXCLS",
                "silver_feature": "VIXCLS_z",
                "silver_feature_non_null_rows": int(final_silver_pdf["VIXCLS_z"].notna().sum()),
            },
            {
                "indicator_id": "vix3m",
                "silver_feature": "vix3m_z",
                "silver_feature_non_null_rows": int(final_silver_pdf["vix3m_z"].notna().sum()),
            },
        ]
    )
    pdf = pd.DataFrame(rows)
    path = f"{comparison_dir}/silver_feature_presence.csv"
    pdf.to_csv(path, index=False)
    return pdf, path


def indicator_failure_note(indicator_id, haystack_matches, source_csv_pdf):
    if source_csv_pdf.empty:
        return "csv_missing", "Legacy per-indicator CSV is missing; cannot rebuild this series faithfully."

    source_release_dates = int(source_csv_pdf["release_date"].nunique())
    source_numeric_rows = int(source_csv_pdf["actual"].notna().sum())
    haystack_release_dates = int(haystack_matches["release_date"].nunique()) if not haystack_matches.empty else 0

    if haystack_matches.empty:
        return (
            "haystack_naming_or_schedule_gap",
            "Frozen Finnhub haystack does not expose this series under the legacy exact names; current recreation drops the series before ALFRED lookup.",
        )
    if haystack_release_dates < source_release_dates:
        return (
            "haystack_release_date_gap",
            f"Haystack release-date coverage is {haystack_release_dates} vs {source_release_dates} in the legacy CSV, so direct recreation misses releases.",
        )
    if source_numeric_rows > source_release_dates + 5:
        return (
            "alfred_bucket_collapse",
            "Legacy source CSV preserves the full ALFRED release bucket, while the weak recreation collapsed each bucket to one row.",
        )
    return ("none", INDICATOR_CONFIG[indicator_id]["note"])


def build_indicator_audit(haystack_df, bronze_indicator_summary_pdf, silver_feature_presence_pdf):
    bronze_map = bronze_indicator_summary_pdf.set_index("indicator_id").to_dict("index")
    silver_map = silver_feature_presence_pdf.set_index("indicator_id").to_dict("index")
    audit_rows = []

    for indicator_id, cfg in INDICATOR_CONFIG.items():
        schedule_from = cfg.get("schedule_from", indicator_id)
        haystack_matches = collect_haystack_matches(
            haystack_df,
            indicator_id,
            release_date_override=schedule_from,
        )
        source_csv_pdf, source_csv_path = load_source_indicator_csv(indicator_id)
        alfred_rows = load_alfred_releases(indicator_id)
        csv_release_dates = (
            pd.to_datetime(source_csv_pdf["release_date"], errors="coerce").dt.date.dropna().unique()
            if not source_csv_pdf.empty
            else []
        )
        alfred_release_dates_found = (
            alfred_rows["realtime_start"].isin(csv_release_dates).sum() if len(csv_release_dates) else 0
        )
        failure_point, note = indicator_failure_note(indicator_id, haystack_matches, source_csv_pdf)
        bronze_info = bronze_map.get(indicator_id, {})
        silver_info = silver_map.get(indicator_id, {})

        audit_rows.append(
            {
                "indicator_id": indicator_id,
                "expected_in_old_stack": True,
                "schedule_source_indicator": schedule_from,
                "matched_in_haystack": not haystack_matches.empty,
                "haystack_match_rows": int(len(haystack_matches)),
                "haystack_release_dates": int(haystack_matches["release_date"].nunique()) if not haystack_matches.empty else 0,
                "haystack_event_examples": " | ".join(sorted(haystack_matches["event"].dropna().unique())[:6]) if not haystack_matches.empty else "",
                "alfred_rows_found": int(alfred_release_dates_found) > 0,
                "alfred_rows_found_count": int(alfred_release_dates_found),
                "csv_created": not source_csv_pdf.empty,
                "source_csv_path": source_csv_path,
                "source_csv_rows": int(len(source_csv_pdf)),
                "source_csv_numeric_rows": int(source_csv_pdf["actual"].notna().sum()) if not source_csv_pdf.empty else 0,
                "source_csv_release_dates": int(source_csv_pdf["release_date"].nunique()) if not source_csv_pdf.empty else 0,
                "bronze_rows_written": int(bronze_info.get("bronze_rows_from_source_csv", 0)),
                "silver_feature": silver_info.get("silver_feature"),
                "silver_feature_non_null_rows": int(silver_info.get("silver_feature_non_null_rows", 0) or 0),
                "failure_point": failure_point,
                "note": note,
            }
        )

    audit_pdf = pd.DataFrame(audit_rows).sort_values("indicator_id")
    audit_csv_path = f"{comparison_dir}/legacy_macro_series_audit.csv"
    audit_pdf.to_csv(audit_csv_path, index=False)

    md_lines = [
        "# Legacy Macro Series Audit",
        "",
        "| Indicator | Haystack Match Rows | Haystack Release Dates | Legacy CSV Rows | Bronze Rows Written | Silver Non-Null Rows | Failure Point | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in audit_pdf.iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["indicator_id"],
                    str(row["haystack_match_rows"]),
                    str(row["haystack_release_dates"]),
                    str(row["source_csv_rows"]),
                    str(row["bronze_rows_written"]),
                    str(row["silver_feature_non_null_rows"]),
                    str(row["failure_point"]),
                    str(row["note"]).replace("\n", " "),
                ]
            )
            + " |"
        )
    audit_md_path = f"{comparison_dir}/legacy_macro_series_audit.md"
    write_text(audit_md_path, "\n".join(md_lines) + "\n")
    return audit_pdf, audit_csv_path, audit_md_path


def make_metric_frame(metric_dict):
    keys = [
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
    return pd.DataFrame([{key: metric_dict.get(key) for key in keys}])


def recovery_verdict(repaired_metrics, reference_metrics, signal_match_rate, regime_mismatch_count, prior_partial_metrics):
    cagr_gap = abs(repaired_metrics["CAGR"] - reference_metrics["CAGR"])
    sharpe_gap = abs(repaired_metrics["Sharpe Annual"] - reference_metrics["Sharpe Annual"])
    max_dd_gap = abs(repaired_metrics["Max Drawdown"] - reference_metrics["Max Drawdown"])
    final_value_gap_pct = abs(repaired_metrics["Final Value"] - reference_metrics["Final Value"]) / reference_metrics["Final Value"]

    repaired_gap_score = cagr_gap + (sharpe_gap / 4.0) + max_dd_gap + final_value_gap_pct
    prior_gap_score = None
    materially_closer_than_partial = None
    if prior_partial_metrics:
        prior_gap_score = (
            abs(prior_partial_metrics["CAGR"] - reference_metrics["CAGR"])
            + (abs(prior_partial_metrics["Sharpe Annual"] - reference_metrics["Sharpe Annual"]) / 4.0)
            + abs(prior_partial_metrics["Max Drawdown"] - reference_metrics["Max Drawdown"])
            + (abs(prior_partial_metrics["Final Value"] - reference_metrics["Final Value"]) / reference_metrics["Final Value"])
        )
        materially_closer_than_partial = bool(repaired_gap_score <= prior_gap_score * 0.6)

    if (
        signal_match_rate >= 0.95
        and cagr_gap <= 0.03
        and sharpe_gap <= 0.15
        and max_dd_gap <= 0.05
    ):
        verdict = "recovery_succeeded"
    elif materially_closer_than_partial or (
        signal_match_rate >= 0.85 and cagr_gap <= 0.08 and sharpe_gap <= 0.30
    ):
        verdict = "recovery_partial"
    else:
        verdict = "recovery_failed"

    return {
        "verdict": verdict,
        "signal_match_rate": float(signal_match_rate),
        "regime_mismatch_count": int(regime_mismatch_count),
        "cagr_gap": float(cagr_gap),
        "sharpe_gap": float(sharpe_gap),
        "max_drawdown_gap": float(max_dd_gap),
        "final_value_gap_pct": float(final_value_gap_pct),
        "repaired_gap_score": float(repaired_gap_score),
        "prior_gap_score": float(prior_gap_score) if prior_gap_score is not None else None,
        "materially_closer_than_partial": materially_closer_than_partial,
    }


haystack_df = load_haystack(haystack_path)
indicator_csv_status_pdf, indicator_csv_status_path = copy_source_indicator_csvs()
bronze_stats, bronze_indicator_summary_pdf, bronze_indicator_summary_path = build_repaired_macro_bronze(
    bronze_baseline_table
)
silver_stats, final_silver_pdf = build_legacy_silver(bronze_baseline_table, silver_baseline_table)
silver_feature_presence_pdf, silver_feature_presence_path = build_silver_feature_presence(final_silver_pdf)
audit_pdf, audit_csv_path, audit_md_path = build_indicator_audit(
    haystack_df,
    bronze_indicator_summary_pdf,
    silver_feature_presence_pdf,
)

full_signal_pdf, full_signal_path = train_full_signal_model(silver_baseline_table, baseline_dir)
walkforward = run_walkforward_variant(
    silver_baseline_table,
    full_signal_path,
    wf_signal_baseline_table,
    baseline_dir,
)

reference_signals = load_reference_walkforward_signals()
reference_summary_metrics = load_reference_performance_summary()
prior_partial_metrics = load_partial_baseline_metrics()

signal_diff_pdf = (
    walkforward["signals"][["timestamp", "predicted_regime"]]
    .rename(columns={"predicted_regime": "predicted_regime_repaired"})
    .merge(
        reference_signals.rename(columns={"predicted_regime": "predicted_regime_reference"}),
        on="timestamp",
        how="outer",
    )
    .sort_values("timestamp")
)
signal_diff_pdf["regime_match"] = (
    signal_diff_pdf["predicted_regime_repaired"] == signal_diff_pdf["predicted_regime_reference"]
)
signal_diff_pdf["regime_mismatch"] = ~signal_diff_pdf["regime_match"].fillna(False)
signal_diff_path = f"{comparison_dir}/repaired_legacy_baseline_signal_diff.csv"
signal_diff_pdf.to_csv(signal_diff_path, index=False)

overlap_mask = signal_diff_pdf["predicted_regime_repaired"].notna() & signal_diff_pdf["predicted_regime_reference"].notna()
signal_match_rate = (
    float(signal_diff_pdf.loc[overlap_mask, "regime_match"].mean()) if overlap_mask.any() else 0.0
)
regime_mismatch_count = int(signal_diff_pdf.loc[overlap_mask, "regime_mismatch"].sum()) if overlap_mask.any() else 0

metrics_pdf = make_metric_frame(walkforward["metrics"])
metrics_path = f"{comparison_dir}/repaired_legacy_baseline_metrics.csv"
metrics_pdf.to_csv(metrics_path, index=False)

recovery_status = recovery_verdict(
    walkforward["metrics"],
    reference_summary_metrics,
    signal_match_rate,
    regime_mismatch_count,
    prior_partial_metrics,
)

ready_for_next_step = recovery_status["verdict"] == "recovery_succeeded"
next_step_status = (
    "swap only ALFRED logic and compare"
    if ready_for_next_step
    else "still blocked on legacy baseline recovery"
)

summary_lines = [
    "# Repaired Legacy Macro Recovery Summary",
    "",
    "## Recovery Verdict",
    "",
    f"- Verdict: `{recovery_status['verdict']}`",
    f"- Ready for next step: `{next_step_status}`",
    f"- Signal match rate vs legacy walk-forward table: `{recovery_status['signal_match_rate']:.6f}`",
    f"- Regime mismatch count vs legacy walk-forward table: `{recovery_status['regime_mismatch_count']}`",
    "",
    "## Repaired Baseline Metrics",
    "",
    f"- Final equity: `{walkforward['metrics']['Final Value']:.6f}`",
    f"- CAGR: `{walkforward['metrics']['CAGR']:.6f}`",
    f"- Sharpe: `{walkforward['metrics']['Sharpe Annual']:.6f}`",
    f"- Max drawdown: `{walkforward['metrics']['Max Drawdown']:.6f}`",
    "",
    "## Trusted Reference Metrics",
    "",
    f"- Final equity: `{reference_summary_metrics.get('Final Value', np.nan):.6f}`",
    f"- CAGR: `{reference_summary_metrics.get('CAGR', np.nan):.6f}`",
    f"- Sharpe: `{reference_summary_metrics.get('Sharpe Annual', np.nan):.6f}`",
    f"- Max drawdown: `{reference_summary_metrics.get('Max Drawdown', np.nan):.6f}`",
    "",
    "## Gap vs Trusted Reference",
    "",
    f"- CAGR gap: `{recovery_status['cagr_gap']:.6f}`",
    f"- Sharpe gap: `{recovery_status['sharpe_gap']:.6f}`",
    f"- Max drawdown gap: `{recovery_status['max_drawdown_gap']:.6f}`",
    f"- Final value gap pct: `{recovery_status['final_value_gap_pct']:.6f}`",
    "",
    "## Failure Analysis",
    "",
]
for _, row in audit_pdf.iterrows():
    summary_lines.append(
        f"- `{row['indicator_id']}`: `{row['failure_point']}` — {row['note']}"
    )

summary_path = f"{comparison_dir}/repaired_legacy_summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "recovery_verdict": recovery_status["verdict"],
    "ready_for_next_step": ready_for_next_step,
    "next_step_status": next_step_status,
    "trusted_reference_metrics": reference_summary_metrics,
    "repaired_metrics": walkforward["metrics"],
    "signal_match_rate": signal_match_rate,
    "regime_mismatch_count": regime_mismatch_count,
    "repair_summary": {
        "repair_method": "comparison-safe rebuild from authoritative legacy per-indicator CSVs, preserving old bronze consolidation and silver averaging behavior",
        "current_recreation_failure": "weak recreation relied on sparse exact haystack matches and collapsed legacy release-bucket richness before bronze consolidation",
    },
}
recommendation_path = f"{comparison_dir}/repaired_legacy_recommendation.json"
write_json(recommendation_path, recommendation_payload)

run_metadata = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_root": output_root,
    "tables": {
        "bronze_baseline_table": bronze_stats,
        "silver_baseline_table": silver_stats,
        "wf_signal_baseline_table": table_row_range(wf_signal_baseline_table),
    },
    "paths": {
        "indicator_csv_status": indicator_csv_status_path,
        "bronze_indicator_summary": bronze_indicator_summary_path,
        "silver_feature_presence": silver_feature_presence_path,
        "legacy_macro_series_audit_csv": audit_csv_path,
        "legacy_macro_series_audit_md": audit_md_path,
        "repaired_legacy_baseline_metrics": metrics_path,
        "repaired_legacy_baseline_signal_diff": signal_diff_path,
        "repaired_legacy_summary": summary_path,
        "repaired_legacy_recommendation": recommendation_path,
    },
    "trusted_reference_metrics": reference_summary_metrics,
    "prior_partial_metrics": prior_partial_metrics,
    "recovery_status": recovery_status,
    "next_step_status": next_step_status,
}
run_metadata_path = f"{comparison_dir}/run_metadata.json"
write_json(run_metadata_path, run_metadata)

result = {
    "phase": "legacy_macro_series_repair",
    "status": "success",
    "recovery_verdict": recovery_status["verdict"],
    "ready_for_next_step": ready_for_next_step,
    "next_step_status": next_step_status,
    "output_root": output_root,
}
print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

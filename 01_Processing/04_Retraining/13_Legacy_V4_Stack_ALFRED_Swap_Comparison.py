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
from pyspark.sql import Window
from pyspark.sql import functions as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
LEGACY_SCHEMA = "default"

COMPARISON_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/comparison"
DEFAULT_OUTPUT_ROOT = (
    f"{COMPARISON_ROOT}/2026-04-20_legacy_v4_alfred_swap"
)

HAYSTACK_PATH_DEFAULT = (
    f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data/raw_finnhub_haystack.csv"
)
MARKET_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"
VIX3M_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix3m_1h"
VIX_SPOT_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_vix_1h"

LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT = (
    f"{CATALOG}.{LEGACY_SCHEMA}.bt_walkforward_regime_signals"
)
LEGACY_REFERENCE_FULL_SIGNAL_PATH_DEFAULT = (
    f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data/bt_fred_1d_regime_signals-v4.csv"
)
LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT = (
    f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data/strategy_performance_summary.csv"
)

BRONZE_BASELINE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_legacy_baseline"
)
BRONZE_CORRECTED_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_legacy_corrected"
)
SILVER_BASELINE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_baseline"
)
SILVER_CORRECTED_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_corrected"
)
WF_SIGNAL_BASELINE_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_baseline"
)
WF_SIGNAL_CORRECTED_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_corrected"
)

TRAIN_START_DEFAULT = "2016-01-01"
LIVE_START_DEFAULT = "2021-01-01"
LIVE_END_DEFAULT = "2026-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252
VIX_SOURCE = "fred_1d"

FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"

INDICATOR_MAP = {
    "PAYEMS": [
        "Non-Farm Payrolls",
        "United States-Employment-Non-Farm Payrolls",
        "Nonfarm Payrolls",
    ],
    "UNRATE": [
        "Unemployment Rate",
        "United States-Employment-Unemployment Rate",
    ],
    "CPIAUCSL": [
        "CPI MM, SA",
        "United States-CPI-CPI MM, SA",
        "Consumer Price Index (CPI)",
    ],
    "PCEPI": [
        "Core PCE Price Index",
        "United States-Consumption-Core PCE Price Index MM",
        "PCE Price Index",
    ],
    "PCE": [
        "Personal Consumption Expenditures",
        "United States-Consumption-Personal Consumption Expenditures",
    ],
    "INDPRO": ["Industrial Production"],
    "GDPC1": ["Real GDP", "GDP Growth Rate"],
    "UMCSENT": [
        "U Mich Sentiment",
        "Michigan Consumer Sentiment",
        "U Mich Sentiment Final",
    ],
}

TYPICAL_TIMES = {
    "PAYEMS": "08:30:00",
    "UNRATE": "08:30:00",
    "CPIAUCSL": "08:30:00",
    "PCEPI": "08:30:00",
    "PCE": "08:30:00",
    "INDPRO": "09:15:00",
    "GDPC1": "08:30:00",
    "UMCSENT": "10:00:00",
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
POSITION_MAP = {0: "QQQ", 1: "SPY", 2: "CASH"}


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
legacy_reference_signal_table = LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT
legacy_reference_full_signal_path = LEGACY_REFERENCE_FULL_SIGNAL_PATH_DEFAULT
legacy_reference_summary_path = LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT
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
    dbutils.widgets.text("haystack_path", HAYSTACK_PATH_DEFAULT)
    dbutils.widgets.text("market_table", MARKET_TABLE_DEFAULT)
    dbutils.widgets.text("vix3m_table", VIX3M_TABLE_DEFAULT)
    dbutils.widgets.text("vix_spot_table", VIX_SPOT_TABLE_DEFAULT)
    dbutils.widgets.text(
        "legacy_reference_signal_table",
        LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT,
    )
    dbutils.widgets.text(
        "legacy_reference_full_signal_path",
        LEGACY_REFERENCE_FULL_SIGNAL_PATH_DEFAULT,
    )
    dbutils.widgets.text(
        "legacy_reference_summary_path",
        LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT,
    )
    dbutils.widgets.text("bronze_baseline_table", BRONZE_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("bronze_corrected_table", BRONZE_CORRECTED_TABLE_DEFAULT)
    dbutils.widgets.text("silver_baseline_table", SILVER_BASELINE_TABLE_DEFAULT)
    dbutils.widgets.text("silver_corrected_table", SILVER_CORRECTED_TABLE_DEFAULT)
    dbutils.widgets.text(
        "wf_signal_baseline_table",
        WF_SIGNAL_BASELINE_TABLE_DEFAULT,
    )
    dbutils.widgets.text(
        "wf_signal_corrected_table",
        WF_SIGNAL_CORRECTED_TABLE_DEFAULT,
    )
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
    legacy_reference_signal_table = (
        dbutils.widgets.get("legacy_reference_signal_table").strip()
        or LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT
    )
    legacy_reference_full_signal_path = (
        dbutils.widgets.get("legacy_reference_full_signal_path").strip()
        or LEGACY_REFERENCE_FULL_SIGNAL_PATH_DEFAULT
    )
    legacy_reference_summary_path = (
        dbutils.widgets.get("legacy_reference_summary_path").strip()
        or LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT
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
baseline_dir = f"{output_root}/legacy_baseline"
corrected_dir = f"{output_root}/legacy_corrected_alfred"
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
    ).dt.date
    vintages["observation_date"] = pd.to_datetime(
        vintages["observation_date"],
        errors="coerce",
    )
    vintages["value_num"] = pd.to_numeric(vintages["value"], errors="coerce")
    alfred_release_cache[series_id] = vintages
    return vintages


def select_release_row(series_id, release_date, mode):
    vintages = load_alfred_releases(series_id)
    release_dt = pd.to_datetime(release_date).date()
    release_rows = vintages[vintages["realtime_start"] == release_dt].copy()
    if release_rows.empty:
        return None

    if mode == "legacy_first":
        selected = release_rows.iloc[0]
    elif mode == "corrected_latest":
        valid_rows = release_rows.dropna(subset=["observation_date"]).copy()
        if valid_rows.empty:
            selected = release_rows.iloc[0]
        else:
            max_obs = valid_rows["observation_date"].max()
            selected = valid_rows[valid_rows["observation_date"] == max_obs].iloc[-1]
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    selected_observation_date = (
        selected["observation_date"].date().isoformat()
        if pd.notna(selected.get("observation_date"))
        else None
    )
    actual = pd.to_numeric(selected.get("value_num"), errors="coerce")
    return {
        "actual": float(actual) if pd.notna(actual) else np.nan,
        "selected_observation_date": selected_observation_date,
    }


def apply_precise_timing(indicator_id, release_date):
    wall_clock_time = TYPICAL_TIMES.get(indicator_id, "08:30:00")
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
    return pdf.sort_values("time")


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


def build_macro_variant(haystack_df, selection_mode, target_table, variant_dir):
    rows = []
    audit_rows = []

    for indicator_id, vendor_names in INDICATOR_MAP.items():
        indicator_releases = haystack_df[haystack_df["event"].isin(vendor_names)].copy()
        indicator_releases["release_date"] = indicator_releases["time"].dt.date
        indicator_releases = indicator_releases.sort_values("time")

        for _, row in indicator_releases.iterrows():
            selection = select_release_row(
                indicator_id,
                row["release_date"],
                selection_mode,
            )
            if selection is None:
                continue

            timestamp_utc = apply_precise_timing(indicator_id, str(row["release_date"]))
            rows.append(
                {
                    "indicator_id": indicator_id,
                    "timestamp": timestamp_utc,
                    "actual": selection["actual"],
                    "event": f"First Release: {indicator_id}",
                }
            )
            audit_rows.append(
                {
                    "indicator_id": indicator_id,
                    "source_event": row["event"],
                    "source_time": row["time"],
                    "release_date": str(row["release_date"]),
                    "timestamp_utc": timestamp_utc,
                    "selection_mode": selection_mode,
                    "selected_observation_date": selection["selected_observation_date"],
                    "actual": selection["actual"],
                }
            )

    if not rows:
        raise ValueError(f"No macro rows produced for {selection_mode}")

    macro_pdf = pd.DataFrame(rows)
    macro_pdf = normalize_timestamp(macro_pdf)
    macro_pdf = macro_pdf.dropna(subset=["timestamp", "actual"])
    macro_pdf["year_partition"] = macro_pdf["timestamp"].dt.year.astype(int)
    macro_pdf["ingestion_ts"] = pd.Timestamp.utcnow().tz_localize(None)
    macro_pdf = macro_pdf.sort_values(["indicator_id", "timestamp"])

    write_delta_table_from_pandas(
        macro_pdf,
        target_table,
        partition_cols=["indicator_id"],
    )

    audit_pdf = pd.DataFrame(audit_rows)
    audit_pdf = normalize_timestamp(audit_pdf, "timestamp_utc")
    audit_pdf = normalize_timestamp(audit_pdf, "source_time")
    audit_pdf = audit_pdf.sort_values(["indicator_id", "timestamp_utc", "source_event"])
    audit_path = f"{variant_dir}/macro_selection_audit.csv"
    audit_pdf.to_csv(audit_path, index=False)

    summary_pdf = (
        macro_pdf.groupby("indicator_id")
        .agg(
            row_count=("actual", "size"),
            min_timestamp=("timestamp", "min"),
            max_timestamp=("timestamp", "max"),
            avg_actual=("actual", "mean"),
        )
        .reset_index()
    )
    summary_pdf.to_csv(f"{variant_dir}/macro_indicator_summary.csv", index=False)
    return {
        "table": target_table,
        "table_stats": table_row_range(target_table),
        "audit_path": audit_path,
        "summary_path": f"{variant_dir}/macro_indicator_summary.csv",
    }


def build_legacy_silver(macro_table, target_table):
    start_date = pd.Timestamp("2016-01-01 00:00:00")
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
        {
            "timestamp": pd.date_range(
                start=start_date,
                end=end_date,
                freq="H",
            )
        }
    )

    macro_pdf = spark.table(macro_table).toPandas()
    macro_pdf = normalize_timestamp(macro_pdf)
    macro_pdf = macro_pdf.dropna(subset=["timestamp"])
    macro_pdf = macro_pdf.drop_duplicates(
        subset=["timestamp", "indicator_id"],
        keep="last",
    )
    macro_pdf["join_ts"] = (
        macro_pdf["timestamp"] + pd.Timedelta(hours=2)
    ).dt.floor("H")
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

    required_macro_columns = [
        "CPIAUCSL",
        "PCEPI",
        "PCE",
        "PAYEMS",
        "UNRATE",
        "GDPC1",
        "INDPRO",
        "UMCSENT",
    ]
    for column_name in required_macro_columns:
        if column_name not in silver_grid.columns:
            silver_grid[column_name] = np.nan

    fill_cols = [
        "CPIAUCSL",
        "PCEPI",
        "PCE",
        "PAYEMS",
        "UNRATE",
        "GDPC1",
        "INDPRO",
        "UMCSENT",
        "VIXCLS",
        "vix3m",
        "SPY_close",
    ]
    silver_grid = silver_grid.sort_values("timestamp").reset_index(drop=True)
    silver_grid[fill_cols] = silver_grid[fill_cols].ffill()

    z_targets = [
        "CPIAUCSL",
        "PCEPI",
        "PCE",
        "PAYEMS",
        "UNRATE",
        "GDPC1",
        "INDPRO",
        "UMCSENT",
        "VIXCLS",
        "vix3m",
    ]
    window_size = 8761
    for col_name in z_targets:
        rolling_mean = silver_grid[col_name].rolling(window_size, min_periods=1).mean()
        rolling_std = silver_grid[col_name].rolling(window_size, min_periods=2).std()
        denominator = rolling_std.where(rolling_std != 0, 1)
        silver_grid[f"{col_name}_z"] = (
            silver_grid[col_name] - rolling_mean
        ) / denominator

    silver_grid["CPI_Shock"] = (
        silver_grid["CPIAUCSL"] - silver_grid["CPIAUCSL"].shift(1)
    ).abs()
    silver_grid["NFP_Shock"] = (
        silver_grid["PAYEMS"] - silver_grid["PAYEMS"].shift(1)
    ).abs()
    silver_grid["CPI_Event_Active"] = (
        (silver_grid["CPI_Shock"] > 0)
        .astype(int)
        .rolling(49, min_periods=1)
        .max()
    )
    silver_grid["NFP_Event_Active"] = (
        (silver_grid["NFP_Shock"] > 0)
        .astype(int)
        .rolling(49, min_periods=1)
        .max()
    )
    silver_grid["VIX_Inversion_Z"] = (
        silver_grid["VIXCLS_z"] - silver_grid["vix3m_z"]
    )
    silver_grid["Price_Mom_24H"] = (
        silver_grid["SPY_close"].shift(1) / silver_grid["SPY_close"].shift(25)
    ) - 1

    final_silver = silver_grid[
        [
            "timestamp",
            "SPY_close",
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
    ].copy()
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
        pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": model.feature_importances_,
            }
        )
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
        pdf.pivot_table(
            index="timestamp",
            columns="symbol",
            values="close",
            aggfunc="last",
        )
        .reset_index()
        .rename(columns={"SPY": "SPY", "QQQ": "QQQ"})
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
    vol_month = returns.std() * np.sqrt(21) if len(returns) else np.nan
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
    max_dd = float(drawdown.min()) if len(series) else np.nan
    return {
        "Start Value": start,
        "Final Value": end,
        "Total Return": float(total_return),
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Vol Monthly": float(vol_month) if np.isfinite(vol_month) else np.nan,
        "Vol Annual": float(vol_ann) if np.isfinite(vol_ann) else np.nan,
        "Sharpe Annual": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Sortino Annual": float(sortino) if np.isfinite(sortino) else np.nan,
        "Max Drawdown": max_dd,
    }


def run_walkforward_variant(
    silver_table,
    label_path,
    signal_table,
    output_dir,
    variant_label,
):
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

        monthly_accuracy = (
            test_df["predicted_regime"] == test_df["regime"]
        ).mean()
        monthly_rows.append(
            {
                "variant": variant_label,
                "train_end": train_end,
                "test_end": test_end,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "monthly_accuracy": float(monthly_accuracy),
            }
        )

        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type="gain")
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
    fi_df = pd.concat(feature_importance_history, ignore_index=True)
    if not fi_df.empty:
        fi_df["importance"] = fi_df.groupby("date")["importance"].transform(
            lambda x: x / x.sum() if x.sum() else x
        )
    fi_df.to_csv(f"{output_dir}/feature_importance_monthly.csv", index=False)

    monthly_pdf = pd.DataFrame(monthly_rows)
    monthly_pdf.to_csv(f"{output_dir}/monthly_validation.csv", index=False)

    signals_df["regime_name"] = signals_df["predicted_regime"].map(REGIME_NAME_MAP)
    signals_df.to_csv(f"{output_dir}/walkforward_signals.csv", index=False)

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
    metrics["Total Switches"] = int(backtest_df["position_changed"].sum())
    holding_lengths = (
        backtest_df["active_asset"]
        .ne(backtest_df["active_asset"].shift())
        .cumsum()
        .value_counts()
    )
    metrics["Average Holding Bars"] = (
        float(holding_lengths.mean()) if not holding_lengths.empty else np.nan
    )

    daily_curve = strategy_series.resample("D").last().dropna()
    daily_drawdown = daily_curve / daily_curve.cummax() - 1

    return {
        "signals": signals_df,
        "backtest": backtest_df,
        "daily_curve": daily_curve,
        "daily_drawdown": daily_drawdown,
        "metrics": metrics,
        "monthly_validation": monthly_pdf,
    }


def load_reference_walkforward_signals():
    pdf = spark.table(legacy_reference_signal_table).toPandas()
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf[
        (pdf["timestamp"] >= pd.Timestamp(live_start))
        & (pdf["timestamp"] < pd.Timestamp(live_end))
    ].copy()
    pdf["predicted_regime"] = pd.to_numeric(
        pdf["predicted_regime"],
        errors="coerce",
    ).fillna(2).astype(int)
    return pdf[["timestamp", "predicted_regime"]]


def baseline_recovery_verdict(recreated_metrics, reference_metrics, signal_match_rate):
    cagr_gap = abs(recreated_metrics["CAGR"] - reference_metrics["CAGR"])
    sharpe_gap = abs(
        recreated_metrics["Sharpe Annual"] - reference_metrics["Sharpe Annual"]
    )
    max_dd_gap = abs(
        recreated_metrics["Max Drawdown"] - reference_metrics["Max Drawdown"]
    )
    succeeded = (
        signal_match_rate >= 0.95
        and cagr_gap <= 0.03
        and sharpe_gap <= 0.15
        and max_dd_gap <= 0.05
    )
    return {
        "succeeded": bool(succeeded),
        "signal_match_rate": float(signal_match_rate),
        "cagr_gap": float(cagr_gap),
        "sharpe_gap": float(sharpe_gap),
        "max_drawdown_gap": float(max_dd_gap),
    }


def make_metric_frame(metric_dict):
    keys = [
        "Variant",
        "Start Value",
        "Final Value",
        "Total Return",
        "CAGR",
        "Vol Monthly",
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


def markdown_table(pdf):
    headers = list(pdf.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in pdf.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def recommendation_text(recovery_status, baseline_metrics, corrected_metrics):
    if not recovery_status["succeeded"]:
        return "further review needed"
    if (
        corrected_metrics["CAGR"] > baseline_metrics["CAGR"]
        and corrected_metrics["Sharpe Annual"] > baseline_metrics["Sharpe Annual"]
        and corrected_metrics["Max Drawdown"] >= baseline_metrics["Max Drawdown"] - 0.02
    ):
        return "prefer legacy_plus_corrected_alfred for next-stage evaluation"
    if (
        corrected_metrics["CAGR"] < baseline_metrics["CAGR"]
        and corrected_metrics["Sharpe Annual"] < baseline_metrics["Sharpe Annual"]
    ):
        return "keep legacy_baseline as the control"
    return "further review needed"


haystack_df = load_haystack(haystack_path)
source_event_counts = (
    haystack_df.groupby("event")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="row_count")
)
source_event_counts.to_csv(f"{comparison_dir}/haystack_event_counts.csv", index=False)

legacy_macro_info = build_macro_variant(
    haystack_df,
    "legacy_first",
    bronze_baseline_table,
    baseline_dir,
)
corrected_macro_info = build_macro_variant(
    haystack_df,
    "corrected_latest",
    bronze_corrected_table,
    corrected_dir,
)

legacy_silver_stats = build_legacy_silver(bronze_baseline_table, silver_baseline_table)
corrected_silver_stats = build_legacy_silver(bronze_corrected_table, silver_corrected_table)

legacy_full_signal_pdf, legacy_full_signal_path = train_full_signal_model(
    silver_baseline_table,
    baseline_dir,
)
corrected_full_signal_pdf, corrected_full_signal_path = train_full_signal_model(
    silver_corrected_table,
    corrected_dir,
)

legacy_walkforward = run_walkforward_variant(
    silver_baseline_table,
    legacy_full_signal_path,
    wf_signal_baseline_table,
    baseline_dir,
    "legacy_baseline",
)
corrected_walkforward = run_walkforward_variant(
    silver_corrected_table,
    corrected_full_signal_path,
    wf_signal_corrected_table,
    corrected_dir,
    "legacy_plus_corrected_alfred",
)

reference_signals = load_reference_walkforward_signals()
reference_prices = load_price_frame(market_table, live_start, live_end)
reference_backtest = run_legacy_backtest(
    reference_signals,
    reference_prices,
    initial_capital,
    transaction_cost,
)
reference_metrics = compute_legacy_metrics(
    reference_backtest.set_index("timestamp")["portfolio"]
)
reference_metrics["Variant"] = "legacy_reference_replay"
reference_metrics["Signal Rows"] = int(len(reference_signals))

baseline_signal_compare = (
    legacy_walkforward["signals"][["timestamp", "predicted_regime"]]
    .merge(
        reference_signals.rename(
            columns={"predicted_regime": "predicted_regime_reference"}
        ),
        on="timestamp",
        how="inner",
    )
)
baseline_signal_compare["signal_match"] = (
    baseline_signal_compare["predicted_regime"]
    == baseline_signal_compare["predicted_regime_reference"]
)
signal_match_rate = (
    float(baseline_signal_compare["signal_match"].mean())
    if not baseline_signal_compare.empty
    else 0.0
)

recovery_status = baseline_recovery_verdict(
    legacy_walkforward["metrics"],
    reference_metrics,
    signal_match_rate,
)

legacy_baseline_metrics_pdf = make_metric_frame(legacy_walkforward["metrics"])
legacy_corrected_metrics_pdf = make_metric_frame(corrected_walkforward["metrics"])

legacy_baseline_metrics_path = f"{comparison_dir}/legacy_baseline_metrics.csv"
legacy_corrected_metrics_path = f"{comparison_dir}/legacy_corrected_metrics.csv"
legacy_baseline_metrics_pdf.to_csv(legacy_baseline_metrics_path, index=False)
legacy_corrected_metrics_pdf.to_csv(legacy_corrected_metrics_path, index=False)

metric_diff_rows = []
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
    baseline_value = legacy_walkforward["metrics"].get(metric_name)
    corrected_value = corrected_walkforward["metrics"].get(metric_name)
    if pd.isna(baseline_value) or pd.isna(corrected_value):
        delta_value = np.nan
    else:
        delta_value = corrected_value - baseline_value
    metric_diff_rows.append(
        {
            "metric": metric_name,
            "legacy_baseline": baseline_value,
            "legacy_corrected_alfred": corrected_value,
            "corrected_minus_baseline": delta_value,
        }
    )

metrics_diff_pdf = pd.DataFrame(metric_diff_rows)
metrics_diff_path = f"{comparison_dir}/metrics_diff.csv"
metrics_diff_pdf.to_csv(metrics_diff_path, index=False)

signal_diff_pdf = (
    legacy_walkforward["signals"]
    .rename(
        columns={
            "predicted_regime": "predicted_regime_baseline",
            "prob_regime0": "prob_regime0_baseline",
            "prob_regime1": "prob_regime1_baseline",
            "prob_regime2": "prob_regime2_baseline",
        }
    )
    .merge(
        corrected_walkforward["signals"].rename(
            columns={
                "predicted_regime": "predicted_regime_corrected",
                "prob_regime0": "prob_regime0_corrected",
                "prob_regime1": "prob_regime1_corrected",
                "prob_regime2": "prob_regime2_corrected",
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
signal_diff_path = f"{comparison_dir}/signal_diff.csv"
signal_diff_pdf.to_csv(signal_diff_path, index=False)

equity_curve_pdf = pd.concat(
    [
        legacy_walkforward["daily_curve"].rename("equity").reset_index().assign(
            variant="legacy_baseline"
        ),
        corrected_walkforward["daily_curve"].rename("equity").reset_index().assign(
            variant="legacy_plus_corrected_alfred"
        ),
        reference_backtest.set_index("timestamp")["portfolio"]
        .resample("D")
        .last()
        .dropna()
        .rename("equity")
        .reset_index()
        .assign(variant="legacy_reference_replay"),
    ],
    ignore_index=True,
)
equity_curve_path = f"{comparison_dir}/equity_curve.csv"
equity_curve_pdf.to_csv(equity_curve_path, index=False)

drawdown_curve_pdf = pd.concat(
    [
        legacy_walkforward["daily_drawdown"].rename("drawdown").reset_index().assign(
            variant="legacy_baseline"
        ),
        corrected_walkforward["daily_drawdown"].rename("drawdown").reset_index().assign(
            variant="legacy_plus_corrected_alfred"
        ),
        (
            reference_backtest.set_index("timestamp")["portfolio"]
            .resample("D")
            .last()
            .dropna()
            .pipe(lambda s: s / s.cummax() - 1)
            .rename("drawdown")
            .reset_index()
            .assign(variant="legacy_reference_replay")
        ),
    ],
    ignore_index=True,
)
drawdown_curve_path = f"{comparison_dir}/drawdown_curve.csv"
drawdown_curve_pdf.to_csv(drawdown_curve_path, index=False)

bronze_baseline_pdf = (
    spark.table(bronze_baseline_table)
    .select("indicator_id", "timestamp", "actual")
    .toPandas()
)
bronze_corrected_pdf = (
    spark.table(bronze_corrected_table)
    .select("indicator_id", "timestamp", "actual")
    .toPandas()
)
bronze_baseline_pdf = normalize_timestamp(bronze_baseline_pdf).drop_duplicates(
    subset=["indicator_id", "timestamp"],
    keep="last",
)
bronze_corrected_pdf = normalize_timestamp(bronze_corrected_pdf).drop_duplicates(
    subset=["indicator_id", "timestamp"],
    keep="last",
)
bronze_compare = bronze_baseline_pdf.merge(
    bronze_corrected_pdf,
    on=["indicator_id", "timestamp"],
    how="outer",
    suffixes=("_baseline", "_corrected"),
)
bronze_compare["actual_changed"] = (
    bronze_compare["actual_baseline"].round(12)
    != bronze_compare["actual_corrected"].round(12)
)

silver_baseline_pdf = normalize_timestamp(spark.table(silver_baseline_table).toPandas())
silver_corrected_pdf = normalize_timestamp(spark.table(silver_corrected_table).toPandas())
silver_compare = silver_baseline_pdf.merge(
    silver_corrected_pdf,
    on="timestamp",
    how="inner",
    suffixes=("_baseline", "_corrected"),
)
silver_changed_mask = pd.Series(False, index=silver_compare.index)
for feature in [
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
]:
    left = silver_compare[f"{feature}_baseline"]
    right = silver_compare[f"{feature}_corrected"]
    silver_changed_mask = silver_changed_mask | (left.round(12) != right.round(12))

bronze_diff_count = int(bronze_compare["actual_changed"].fillna(True).sum())
signal_diff_count = int(signal_diff_pdf["regime_changed"].fillna(True).sum())
signal_diff_pct = (
    float(signal_diff_pdf["regime_changed"].mean())
    if len(signal_diff_pdf) > 0
    else np.nan
)
silver_diff_count = int(silver_changed_mask.sum())

methodology_lines = [
    "# Legacy V4 ALFRED Swap Methodology",
    "",
    "## Source of Truth",
    "",
    "- Attached legacy notebooks were treated as the authority for macro ingestion, silver construction, full-signal training, and walk-forward evaluation.",
    "- This notebook recreates the old stack inside the current repo without mutating production objects.",
    "",
    "## Controlled Variants",
    "",
    "- `legacy_baseline`: old Finnhub schedule source + old ALFRED bucket selection (`iloc[0]`).",
    "- `legacy_plus_corrected_alfred`: same legacy pipeline, but ALFRED selects the latest `observation_date` inside each release bucket.",
    "",
    "## Shared Inputs",
    "",
    f"- Finnhub haystack source: `{haystack_path}`",
    f"- Market price source: `{market_table}`",
    f"- VIX3M source: `{vix3m_table}` with `VIX_SOURCE={VIX_SOURCE}`",
    f"- VIX spot source: `{vix_spot_table}`",
    f"- Train start: `{train_start}`",
    f"- Live start: `{live_start}`",
    f"- Live end: `{live_end}`",
    f"- Initial capital: `{initial_capital}`",
    f"- Transaction cost: `{transaction_cost}`",
    "",
    "## Controlled-Swap Validation",
    "",
    f"- Bronze rows with changed `actual`: `{bronze_diff_count}`",
    f"- Silver timestamps with any changed feature: `{silver_diff_count}`",
    f"- Walk-forward timestamps with changed regime: `{signal_diff_count}`",
    f"- Walk-forward regime change rate: `{signal_diff_pct}`",
    "",
    "## Baseline Recovery",
    "",
    f"- Signal match rate vs legacy reference table: `{recovery_status['signal_match_rate']}`",
    f"- CAGR gap vs reference replay: `{recovery_status['cagr_gap']}`",
    f"- Sharpe gap vs reference replay: `{recovery_status['sharpe_gap']}`",
    f"- Max drawdown gap vs reference replay: `{recovery_status['max_drawdown_gap']}`",
    f"- Recovery succeeded: `{recovery_status['succeeded']}`",
]
methodology_spec_path = f"{comparison_dir}/methodology_spec.md"
write_text(methodology_spec_path, "\n".join(methodology_lines) + "\n")

summary_pdf = pd.DataFrame(
    [
        legacy_walkforward["metrics"],
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

recommendation = recommendation_text(
    recovery_status,
    legacy_walkforward["metrics"],
    corrected_walkforward["metrics"],
)

summary_lines = [
    "# Legacy V4 ALFRED Swap Summary",
    "",
    "## Baseline Recovery",
    "",
    f"- Recovery succeeded: `{recovery_status['succeeded']}`",
    f"- Signal match rate vs legacy reference table: `{recovery_status['signal_match_rate']:.6f}`",
    f"- Legacy baseline CAGR: `{legacy_walkforward['metrics']['CAGR']:.6f}`",
    f"- Legacy reference replay CAGR: `{reference_metrics['CAGR']:.6f}`",
    f"- Legacy baseline Sharpe: `{legacy_walkforward['metrics']['Sharpe Annual']:.6f}`",
    f"- Legacy reference replay Sharpe: `{reference_metrics['Sharpe Annual']:.6f}`",
    "",
    "## Variant Comparison",
    "",
    markdown_table(summary_pdf),
    "",
    "## ALFRED-Only Effect",
    "",
    f"- Bronze rows changed by ALFRED swap: `{bronze_diff_count}`",
    f"- Silver timestamps changed by ALFRED swap: `{silver_diff_count}`",
    f"- Walk-forward regime timestamps changed: `{signal_diff_count}`",
    f"- Walk-forward regime change rate: `{signal_diff_pct:.6f}`",
    f"- CAGR delta (corrected - baseline): `{corrected_walkforward['metrics']['CAGR'] - legacy_walkforward['metrics']['CAGR']:.6f}`",
    f"- Sharpe delta (corrected - baseline): `{corrected_walkforward['metrics']['Sharpe Annual'] - legacy_walkforward['metrics']['Sharpe Annual']:.6f}`",
    f"- Max drawdown delta (corrected - baseline): `{corrected_walkforward['metrics']['Max Drawdown'] - legacy_walkforward['metrics']['Max Drawdown']:.6f}`",
    "",
    "## Recommendation",
    "",
    f"- `{recommendation}`",
]
summary_path = f"{comparison_dir}/summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "recommendation": recommendation,
    "baseline_recovery": recovery_status,
    "alfred_only_effect": {
        "bronze_diff_count": bronze_diff_count,
        "silver_diff_count": silver_diff_count,
        "signal_diff_count": signal_diff_count,
        "signal_diff_pct": signal_diff_pct,
        "cagr_delta": corrected_walkforward["metrics"]["CAGR"]
        - legacy_walkforward["metrics"]["CAGR"],
        "sharpe_delta": corrected_walkforward["metrics"]["Sharpe Annual"]
        - legacy_walkforward["metrics"]["Sharpe Annual"],
        "max_drawdown_delta": corrected_walkforward["metrics"]["Max Drawdown"]
        - legacy_walkforward["metrics"]["Max Drawdown"],
    },
}
recommendation_path = f"{comparison_dir}/recommendation.json"
write_json(recommendation_path, recommendation_payload)

plt.figure(figsize=(14, 7))
for variant, group in equity_curve_pdf.groupby("variant"):
    plt.plot(group["timestamp"], group["equity"], label=variant)
plt.title("Legacy V4 Baseline vs ALFRED-Corrected Equity Curve")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
equity_plot_path = f"{comparison_dir}/equity_curve_comparison.png"
plt.savefig(equity_plot_path, dpi=180)
plt.close()

plt.figure(figsize=(14, 7))
for variant, group in drawdown_curve_pdf.groupby("variant"):
    plt.plot(group["timestamp"], group["drawdown"], label=variant)
plt.title("Legacy V4 Baseline vs ALFRED-Corrected Drawdown")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
drawdown_plot_path = f"{comparison_dir}/drawdown_comparison.png"
plt.savefig(drawdown_plot_path, dpi=180)
plt.close()

timeline_pdf = (
    signal_diff_pdf[["timestamp", "predicted_regime_baseline", "predicted_regime_corrected"]]
    .dropna()
    .copy()
)
plt.figure(figsize=(16, 5))
plt.step(
    timeline_pdf["timestamp"],
    timeline_pdf["predicted_regime_baseline"],
    where="post",
    label="legacy_baseline",
)
plt.step(
    timeline_pdf["timestamp"],
    timeline_pdf["predicted_regime_corrected"] + 0.05,
    where="post",
    label="legacy_plus_corrected_alfred",
)
plt.yticks([0, 1, 2], ["Expansion", "Slowdown", "Recession"])
plt.title("Regime Timeline Comparison")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
regime_timeline_path = f"{comparison_dir}/regime_timeline.png"
plt.savefig(regime_timeline_path, dpi=180)
plt.close()

run_metadata = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_root": output_root,
    "train_start": train_start,
    "live_start": live_start,
    "live_end": live_end,
    "initial_capital": initial_capital,
    "transaction_cost": transaction_cost,
    "legacy_reference_signal_table": legacy_reference_signal_table,
    "legacy_reference_full_signal_path": legacy_reference_full_signal_path,
    "legacy_reference_summary_path": legacy_reference_summary_path,
    "source_tables": {
        "market_table": market_table,
        "vix3m_table": vix3m_table,
        "vix_spot_table": vix_spot_table,
    },
    "variant_tables": {
        "bronze_baseline_table": legacy_macro_info["table_stats"],
        "bronze_corrected_table": corrected_macro_info["table_stats"],
        "silver_baseline_table": legacy_silver_stats,
        "silver_corrected_table": corrected_silver_stats,
        "wf_signal_baseline_table": table_row_range(wf_signal_baseline_table),
        "wf_signal_corrected_table": table_row_range(wf_signal_corrected_table),
    },
    "baseline_recovery": recovery_status,
    "alfred_only_effect": recommendation_payload["alfred_only_effect"],
    "artifacts": {
        "methodology_spec": methodology_spec_path,
        "legacy_baseline_metrics": legacy_baseline_metrics_path,
        "legacy_corrected_metrics": legacy_corrected_metrics_path,
        "metrics_diff": metrics_diff_path,
        "signal_diff": signal_diff_path,
        "equity_curve": equity_curve_path,
        "drawdown_curve": drawdown_curve_path,
        "summary": summary_path,
        "recommendation": recommendation_path,
        "equity_curve_plot": equity_plot_path,
        "drawdown_plot": drawdown_plot_path,
        "regime_timeline_plot": regime_timeline_path,
    },
}
run_metadata_path = f"{comparison_dir}/run_metadata.json"
write_json(run_metadata_path, run_metadata)

result = {
    "phase": "legacy_v4_stack_rebuild_and_alfred_swap",
    "status": "success",
    "baseline_recovery_succeeded": recovery_status["succeeded"],
    "recommendation": recommendation,
    "output_root": output_root,
    "artifacts": run_metadata["artifacts"],
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

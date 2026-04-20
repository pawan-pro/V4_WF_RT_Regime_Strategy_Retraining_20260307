# Databricks notebook source
# MAGIC %md
# MAGIC master_time_grid_silver

# COMMAND ----------

import json

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

DEFAULT_MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"
DEFAULT_MACRO_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events"
DEFAULT_TARGET_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
DEFAULT_VIX3M_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_vix3m_1h"
DEFAULT_VIX_SPOT_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_vix_1h"
DEFAULT_START_DATE = "2016-01-01 00:00:00"


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


VIX_SOURCE = "fred_1d"
execution_engine = "spark"
market_table = DEFAULT_MARKET_TABLE
macro_table = DEFAULT_MACRO_TABLE
target_table = DEFAULT_TARGET_TABLE
vix3m_table = DEFAULT_VIX3M_TABLE
vix_spot_table = DEFAULT_VIX_SPOT_TABLE
start_date = DEFAULT_START_DATE
allow_production_overwrite = False

if "dbutils" in globals():
    dbutils.widgets.text("market_table", DEFAULT_MARKET_TABLE)
    dbutils.widgets.text("macro_table", DEFAULT_MACRO_TABLE)
    dbutils.widgets.text("target_table", DEFAULT_TARGET_TABLE)
    dbutils.widgets.text("vix3m_table", DEFAULT_VIX3M_TABLE)
    dbutils.widgets.text("vix_spot_table", DEFAULT_VIX_SPOT_TABLE)
    dbutils.widgets.text("start_date", DEFAULT_START_DATE)
    dbutils.widgets.dropdown("vix_source", VIX_SOURCE, ["fred_1d", "portara_1h"])
    dbutils.widgets.dropdown("execution_engine", execution_engine, ["spark", "pandas"])
    dbutils.widgets.dropdown(
        "allow_production_overwrite",
        "false",
        ["false", "true"],
    )
    market_table = dbutils.widgets.get("market_table").strip() or DEFAULT_MARKET_TABLE
    macro_table = dbutils.widgets.get("macro_table").strip() or DEFAULT_MACRO_TABLE
    target_table = dbutils.widgets.get("target_table").strip() or DEFAULT_TARGET_TABLE
    vix3m_table = dbutils.widgets.get("vix3m_table").strip() or DEFAULT_VIX3M_TABLE
    vix_spot_table = dbutils.widgets.get("vix_spot_table").strip() or DEFAULT_VIX_SPOT_TABLE
    start_date = dbutils.widgets.get("start_date").strip() or DEFAULT_START_DATE
    VIX_SOURCE = dbutils.widgets.get("vix_source").strip() or VIX_SOURCE
    execution_engine = dbutils.widgets.get("execution_engine").strip() or execution_engine
    allow_production_overwrite = parse_bool(
        dbutils.widgets.get("allow_production_overwrite"),
        default=False,
    )

if (
    target_table == DEFAULT_TARGET_TABLE
    and macro_table != DEFAULT_MACRO_TABLE
    and not allow_production_overwrite
):
    raise ValueError(
        "Refusing to overwrite production silver table from a non-production macro "
        "source without allow_production_overwrite=true"
    )

max_ts_row = spark.sql(
    f"SELECT MAX(timestamp) AS max_ts FROM {market_table}"
).collect()[0]
end_date = pd.Timestamp(max_ts_row["max_ts"])

print(
    f"Building silver grid from {start_date} to {end_date} | "
    f"macro_table={macro_table} | target_table={target_table} | "
    f"VIX_SOURCE={VIX_SOURCE} | execution_engine={execution_engine}"
)

feature_columns = [
    "timestamp",
    "SPY_close",
    "QQQ_close",
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


def write_result_frame(final_pdf):
    final_pdf = final_pdf.copy()
    final_pdf["timestamp"] = pd.to_datetime(final_pdf["timestamp"])
    spark.createDataFrame(final_pdf).write.mode("overwrite").option(
        "overwriteSchema", "true"
    ).format("delta").saveAsTable(target_table)

    summary_row = spark.sql(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            MIN(timestamp) AS first_timestamp,
            MAX(timestamp) AS last_timestamp,
            COUNT(CASE WHEN CPIAUCSL_z IS NOT NULL THEN 1 END) AS rows_with_cpi,
            COUNT(CASE WHEN SPY_close IS NOT NULL THEN 1 END) AS rows_with_spy
        FROM {target_table}
        """
    ).collect()[0]

    result = {
        "phase": "silver_rebuild",
        "status": "success",
        "macro_table": macro_table,
        "target_table": target_table,
        "execution_engine": execution_engine,
        "production_table_unchanged": target_table != DEFAULT_TARGET_TABLE
        or macro_table == DEFAULT_MACRO_TABLE,
        "vix_source": VIX_SOURCE,
        "total_rows": int(summary_row["total_rows"]),
        "first_timestamp": str(summary_row["first_timestamp"]),
        "last_timestamp": str(summary_row["last_timestamp"]),
        "rows_with_cpi": int(summary_row["rows_with_cpi"]),
        "rows_with_spy": int(summary_row["rows_with_spy"]),
        "feature_validation": {
            "QQQ_close_non_null_rows": int(final_pdf["QQQ_close"].notna().sum()),
            "CPIAUCSL_z_non_null_rows": int(final_pdf["CPIAUCSL_z"].notna().sum()),
            "PAYEMS_z_non_null_rows": int(final_pdf["PAYEMS_z"].notna().sum()),
            "Price_Mom_24H_non_null_rows": int(final_pdf["Price_Mom_24H"].notna().sum()),
            "CPIAUCSL_z_min": float(final_pdf["CPIAUCSL_z"].min()),
            "CPIAUCSL_z_max": float(final_pdf["CPIAUCSL_z"].max()),
            "PAYEMS_z_min": float(final_pdf["PAYEMS_z"].min()),
            "PAYEMS_z_max": float(final_pdf["PAYEMS_z"].max()),
            "Price_Mom_24H_min": float(final_pdf["Price_Mom_24H"].min()),
            "Price_Mom_24H_max": float(final_pdf["Price_Mom_24H"].max()),
        },
    }
    print(json.dumps(result, indent=2))
    if "dbutils" in globals():
        dbutils.notebook.exit(json.dumps(result))


def build_with_spark():
    backbone = (
        spark.range(0, 15 * 365 * 24)
        .withColumn(
            "timestamp",
            F.to_timestamp(F.lit(start_date))
            + (F.col("id") * F.expr("INTERVAL 1 HOUR")),
        )
        .filter(F.col("timestamp") <= F.lit(end_date.to_pydatetime()))
        .select("timestamp")
    )

    macro_df = spark.table(macro_table).dropDuplicates(["timestamp", "indicator_id"])
    market_df = (
        spark.table(market_table)
        .filter(F.col("symbol") == "SPY")
        .dropDuplicates(["timestamp"])
    )
    qqq_df = (
        spark.table(market_table)
        .filter(F.col("symbol") == "QQQ")
        .dropDuplicates(["timestamp"])
    )
    vix3m_df = spark.table(vix3m_table).dropDuplicates(["timestamp"])
    vix_spot_df = spark.table(vix_spot_table).dropDuplicates(["timestamp"])

    if VIX_SOURCE == "portara_1h":
        vix3m_lagged = vix3m_df.withColumn(
            "v3m_join_ts",
            F.to_timestamp(F.col("timestamp")) + F.expr("INTERVAL 1 DAY"),
        )
    else:
        vix3m_lagged = (
            vix3m_df.withColumnRenamed("close", "vix3m")
            .withColumn("v3m_join_ts", F.col("timestamp"))
        )

    macro_prepared = (
        macro_df.withColumn(
            "join_ts",
            F.date_trunc("hour", F.col("timestamp") + F.expr("INTERVAL 2 HOUR")),
        )
        .select("join_ts", "indicator_id", "actual")
        .groupBy("join_ts")
        .pivot("indicator_id")
        .avg("actual")
    )

    silver_grid = (
        backbone.join(
            macro_prepared, backbone.timestamp == macro_prepared.join_ts, "left"
        )
        .join(
            market_df.select(
                F.col("timestamp").alias("m_ts"),
                F.col("close").alias("SPY_close"),
            ),
            F.col("timestamp") == F.col("m_ts"),
            "left",
        )
        .join(
            qqq_df.select(
                F.col("timestamp").alias("q_ts"),
                F.col("close").alias("QQQ_close"),
            ),
            F.col("timestamp") == F.col("q_ts"),
            "left",
        )
        .join(
            vix3m_lagged.select(
                F.col("v3m_join_ts").alias("v3m_ts"),
                F.col("vix3m"),
            ),
            F.col("timestamp") == F.col("v3m_ts"),
            "left",
        )
        .join(
            vix_spot_df.select(
                F.col("timestamp").alias("vs_ts"),
                F.col("close").alias("VIXCLS"),
            ),
            F.col("timestamp") == F.col("vs_ts"),
            "left",
        )
    )

    win_ffill = Window.orderBy("timestamp").rowsBetween(
        Window.unboundedPreceding, 0
    )
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
        "QQQ_close",
    ]
    for col_name in fill_cols:
        silver_grid = silver_grid.withColumn(
            col_name,
            F.last(F.col(col_name), True).over(win_ffill),
        )

    win_lag = Window.orderBy("timestamp")
    win_z = Window.orderBy("timestamp").rowsBetween(-8760, 0)
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
    for col_name in z_targets:
        avg = F.avg(F.col(col_name)).over(win_z)
        std = F.stddev(F.col(col_name)).over(win_z)
        silver_grid = silver_grid.withColumn(
            f"{col_name}_z",
            (F.col(col_name) - avg) / F.when(std == 0, 1).otherwise(std),
        )

    silver_grid = (
        silver_grid.withColumn(
            "CPI_Shock",
            F.abs(F.col("CPIAUCSL") - F.lag("CPIAUCSL", 1).over(win_lag)),
        )
        .withColumn(
            "NFP_Shock",
            F.abs(F.col("PAYEMS") - F.lag("PAYEMS", 1).over(win_lag)),
        )
    )

    def pulse_event(col_name, hours=48):
        is_event = F.when(F.col(col_name) > 0, 1).otherwise(0)
        return F.max(is_event).over(
            Window.orderBy("timestamp").rowsBetween(-hours, 0)
        )

    final_silver = (
        silver_grid.withColumn("CPI_Event_Active", pulse_event("CPI_Shock"))
        .withColumn("NFP_Event_Active", pulse_event("NFP_Shock"))
        .withColumn("VIX_Inversion_Z", F.col("VIXCLS_z") - F.col("vix3m_z"))
        .withColumn(
            "Price_Mom_24H",
            (F.lag(F.col("SPY_close"), 1).over(win_lag)
             / F.lag(F.col("SPY_close"), 25).over(win_lag)) - 1,
        )
        .select(*feature_columns)
        .filter(F.col("SPY_close").isNotNull())
        .toPandas()
    )
    write_result_frame(final_silver)


def build_with_pandas():
    start_ts = pd.Timestamp(start_date)
    backbone = pd.DataFrame(
        {"timestamp": pd.date_range(start=start_ts, end=end_date, freq="h")}
    )

    macro_pdf = (
        spark.table(macro_table)
        .select("timestamp", "indicator_id", "actual")
        .toPandas()
        .drop_duplicates(subset=["timestamp", "indicator_id"])
    )
    macro_pdf["timestamp"] = pd.to_datetime(macro_pdf["timestamp"], errors="coerce")
    macro_pdf["join_ts"] = (
        macro_pdf["timestamp"] + pd.Timedelta(hours=2)
    ).dt.floor("h")
    macro_pivot = (
        macro_pdf.pivot_table(
            index="join_ts",
            columns="indicator_id",
            values="actual",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"join_ts": "timestamp"})
    )

    market_pdf = (
        spark.table(market_table)
        .filter(F.col("symbol") == "SPY")
        .select("timestamp", F.col("close").alias("SPY_close"))
        .toPandas()
        .drop_duplicates(subset=["timestamp"])
    )
    qqq_pdf = (
        spark.table(market_table)
        .filter(F.col("symbol") == "QQQ")
        .select("timestamp", F.col("close").alias("QQQ_close"))
        .toPandas()
        .drop_duplicates(subset=["timestamp"])
    )
    vix3m_pdf = spark.table(vix3m_table).toPandas().drop_duplicates(subset=["timestamp"])
    vix_spot_pdf = (
        spark.table(vix_spot_table)
        .select("timestamp", F.col("close").alias("VIXCLS"))
        .toPandas()
        .drop_duplicates(subset=["timestamp"])
    )

    for frame in [market_pdf, qqq_pdf, vix3m_pdf, vix_spot_pdf]:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")

    if VIX_SOURCE == "portara_1h":
        if "vix3m" not in vix3m_pdf.columns and "close" in vix3m_pdf.columns:
            vix3m_pdf = vix3m_pdf.rename(columns={"close": "vix3m"})
        vix3m_pdf["timestamp"] = vix3m_pdf["timestamp"] + pd.Timedelta(days=1)
    else:
        vix3m_pdf = vix3m_pdf.rename(columns={"close": "vix3m"})

    silver_pdf = (
        backbone.merge(macro_pivot, on="timestamp", how="left")
        .merge(market_pdf, on="timestamp", how="left")
        .merge(qqq_pdf, on="timestamp", how="left")
        .merge(vix3m_pdf[["timestamp", "vix3m"]], on="timestamp", how="left")
        .merge(vix_spot_pdf[["timestamp", "VIXCLS"]], on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

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
        "QQQ_close",
    ]
    silver_pdf[fill_cols] = silver_pdf[fill_cols].ffill()

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
    z_window = 8761
    for col_name in z_targets:
        rolling_mean = silver_pdf[col_name].rolling(z_window, min_periods=1).mean()
        rolling_std = silver_pdf[col_name].rolling(z_window, min_periods=1).std(ddof=1)
        rolling_std = rolling_std.replace(0, np.nan).fillna(1.0)
        silver_pdf[f"{col_name}_z"] = (
            silver_pdf[col_name] - rolling_mean
        ) / rolling_std

    silver_pdf["CPI_Shock"] = silver_pdf["CPIAUCSL"].diff().abs()
    silver_pdf["NFP_Shock"] = silver_pdf["PAYEMS"].diff().abs()
    silver_pdf["CPI_Event_Active"] = (
        silver_pdf["CPI_Shock"].fillna(0).gt(0).astype(int).rolling(49, min_periods=1).max()
    )
    silver_pdf["NFP_Event_Active"] = (
        silver_pdf["NFP_Shock"].fillna(0).gt(0).astype(int).rolling(49, min_periods=1).max()
    )
    silver_pdf["VIX_Inversion_Z"] = silver_pdf["VIXCLS_z"] - silver_pdf["vix3m_z"]
    silver_pdf["Price_Mom_24H"] = (
        silver_pdf["SPY_close"].shift(1) / silver_pdf["SPY_close"].shift(25)
    ) - 1

    final_silver = silver_pdf.loc[
        silver_pdf["SPY_close"].notna(),
        feature_columns,
    ].copy()
    write_result_frame(final_silver)


if execution_engine == "pandas":
    build_with_pandas()
else:
    build_with_spark()

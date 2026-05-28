# Databricks notebook source
# RT_13: Excel-ready CSV exports for validated Saxo V4 strategy ledger
#
# Production-safety scope:
# - Reads existing gold analysis tables only.
# - Writes CSV report files only under the Saxo V4 Excel export volume path.
# - Does not call Saxo APIs, token state, order endpoints, or Databricks Jobs APIs.
# - Does not modify workflows or live execution notebooks.

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

LEDGER_TABLE = f"{CATALOG}.{SCHEMA}.gold_saxo_v4_strategy_trade_ledger"
EQUITY_TABLE = f"{CATALOG}.{SCHEMA}.gold_saxo_v4_strategy_equity_curve"
TCA_TABLE = f"{CATALOG}.{SCHEMA}.gold_saxo_v4_transaction_cost_analysis"
INTERNAL_TRADE_LOG = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"

EXPORT_ROOT_VOLUME = "/Volumes/pawan_quant_research_1/v4/market_data/reports/saxo_v4_excel_exports"
EXPORT_ROOT_DBFS = f"dbfs:{EXPORT_ROOT_VOLUME}"
LATEST_DIR_DBFS = f"{EXPORT_ROOT_DBFS}/latest"

EXPORT_FILES = {
    "trades": "saxo_v4_etf_trades_switches.csv",
    "equity": "saxo_v4_strategy_equity_curve.csv",
    "combined": "saxo_v4_trades_with_equity_curve.csv",
    "tca": "saxo_v4_transaction_cost_analysis.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dbfs_to_volume_path(path: str) -> str:
    return path.replace("dbfs:", "", 1) if path.startswith("dbfs:") else path


def file_exists(path: str) -> bool:
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def rows_as_dicts(dataframe, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for row in dataframe.limit(limit).collect():
        item = {}
        for key, value in row.asDict(recursive=True).items():
            item[key] = value.isoformat(sep=" ") if isinstance(value, datetime) else value
        rows.append(item)
    return rows


def scalar_value(sql: str, field: str):
    rows = spark.sql(sql).collect()
    if not rows:
        return None
    return rows[0][field]


def write_single_csv(dataframe, output_dir_dbfs: str, filename: str) -> str:
    dbutils.fs.mkdirs(output_dir_dbfs)
    tmp_dir = f"{output_dir_dbfs}/_tmp_{filename.replace('.csv', '')}_{uuid.uuid4().hex}"
    final_path = f"{output_dir_dbfs}/{filename}"

    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(tmp_dir)
    )

    part_files = [
        item.path
        for item in dbutils.fs.ls(tmp_dir)
        if item.name.endswith(".csv") and item.name.startswith("part-")
    ]
    if len(part_files) != 1:
        raise RuntimeError(f"Expected exactly one CSV part file in {tmp_dir}, found {len(part_files)}")

    if file_exists(final_path):
        dbutils.fs.rm(final_path, True)
    dbutils.fs.mv(part_files[0], final_path)
    dbutils.fs.rm(tmp_dir, True)
    return dbfs_to_volume_path(final_path)


def write_to_latest_and_timestamp(dataframe, timestamped_dir_dbfs: str, filename: str) -> dict[str, str]:
    return {
        "latest": write_single_csv(dataframe, LATEST_DIR_DBFS, filename),
        "timestamped": write_single_csv(dataframe, timestamped_dir_dbfs, filename),
    }


# ---------------------------------------------------------------------------
# Export dataframes
# ---------------------------------------------------------------------------
trades_df = spark.sql(
    f"""
    SELECT
        event_ts AS trade_ts,
        CAST(event_ts AS DATE) AS trade_date,
        strategy_stance_before,
        strategy_stance_after,
        symbol,
        side,
        quantity,
        price,
        gross_notional,
        total_transaction_cost AS captured_conversion_cost,
        cost_source,
        cost_observation_quality,
        all_in_cost_confirmed,
        cost_notes,
        trade_id,
        position_id,
        source,
        event_action
    FROM {LEDGER_TABLE}
    WHERE is_v4_strategy_trade = true
    ORDER BY event_ts ASC, event_sequence ASC
    """
)

equity_df = spark.sql(
    f"""
    SELECT
        timestamp,
        strategy_stance,
        strategy_cash,
        strategy_position_symbol,
        strategy_position_qty,
        strategy_position_market_value,
        strategy_equity_gross,
        strategy_equity_net,
        cumulative_transaction_cost AS cumulative_captured_conversion_cost,
        strategy_return_gross,
        strategy_return_net,
        source
    FROM {EQUITY_TABLE}
    ORDER BY timestamp ASC
    """
)

combined_df = spark.sql(
    f"""
    WITH trades AS (
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY event_ts ASC, COALESCE(trade_id, ''), COALESCE(position_id, ''), source_record_id, event_sequence
            ) AS trade_row_id,
            event_ts AS trade_ts,
            CAST(event_ts AS DATE) AS trade_date,
            strategy_stance_before,
            strategy_stance_after,
            symbol,
            side,
            quantity,
            price,
            gross_notional,
            total_transaction_cost AS captured_conversion_cost,
            cost_source,
            cost_observation_quality,
            all_in_cost_confirmed,
            cost_notes,
            trade_id,
            position_id,
            source,
            event_action
        FROM {LEDGER_TABLE}
        WHERE is_v4_strategy_trade = true
    ),
    joined AS (
        SELECT
            t.*,
            e.timestamp AS equity_ts,
            e.strategy_stance AS equity_stance,
            e.strategy_equity_gross,
            e.strategy_equity_net,
            e.cumulative_transaction_cost AS cumulative_captured_conversion_cost,
            e.strategy_return_gross,
            e.strategy_return_net,
            ROW_NUMBER() OVER (
                PARTITION BY t.trade_row_id
                ORDER BY e.timestamp ASC NULLS LAST
            ) AS rn
        FROM trades t
        LEFT JOIN {EQUITY_TABLE} e
          ON e.timestamp >= t.trade_ts
    )
    SELECT
        trade_ts,
        trade_date,
        strategy_stance_before,
        strategy_stance_after,
        symbol,
        side,
        quantity,
        price,
        gross_notional,
        captured_conversion_cost,
        cost_source,
        cost_observation_quality,
        all_in_cost_confirmed,
        cost_notes,
        trade_id,
        position_id,
        source,
        event_action,
        equity_ts,
        equity_stance,
        strategy_equity_gross,
        strategy_equity_net,
        cumulative_captured_conversion_cost,
        strategy_return_gross,
        strategy_return_net
    FROM joined
    WHERE rn = 1
    ORDER BY trade_ts ASC, COALESCE(trade_id, ''), COALESCE(position_id, '')
    """
)

tca_df = spark.sql(
    f"""
    SELECT *
    FROM {TCA_TABLE}
    ORDER BY account_size, cost_model
    """
)


# ---------------------------------------------------------------------------
# Write exports
# ---------------------------------------------------------------------------
run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
TIMESTAMPED_DIR_DBFS = f"{EXPORT_ROOT_DBFS}/{run_tag}"

exports = {
    "trades": write_to_latest_and_timestamp(trades_df, TIMESTAMPED_DIR_DBFS, EXPORT_FILES["trades"]),
    "equity": write_to_latest_and_timestamp(equity_df, TIMESTAMPED_DIR_DBFS, EXPORT_FILES["equity"]),
    "combined": write_to_latest_and_timestamp(combined_df, TIMESTAMPED_DIR_DBFS, EXPORT_FILES["combined"]),
    "tca": write_to_latest_and_timestamp(tca_df, TIMESTAMPED_DIR_DBFS, EXPORT_FILES["tca"]),
}

row_counts = {
    "trades": trades_df.count(),
    "equity": equity_df.count(),
    "combined": combined_df.count(),
    "tca": tca_df.count(),
}

print("Export files")
for export_name, paths in exports.items():
    print(f"{export_name}: latest={paths['latest']} timestamped={paths['timestamped']}")

print("Row counts")
print(json.dumps(row_counts, indent=2))


# ---------------------------------------------------------------------------
# Displays
# ---------------------------------------------------------------------------
print("ETF trade / switch log")
display(trades_df.limit(20))

print("Strategy equity curve")
display(equity_df.limit(20))

print("Trades with nearest equity curve row")
display(combined_df.limit(20))

print("Transaction cost analysis")
display(tca_df.limit(20))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
latest_files = [f"{LATEST_DIR_DBFS}/{filename}" for filename in EXPORT_FILES.values()]
timestamped_files = [f"{TIMESTAMPED_DIR_DBFS}/{filename}" for filename in EXPORT_FILES.values()]

source_trade_count = scalar_value(
    f"""
    SELECT COUNT(*) AS n
    FROM {LEDGER_TABLE}
    WHERE is_v4_strategy_trade = true
    """,
    "n",
)
source_equity_count = scalar_value(f"SELECT COUNT(*) AS n FROM {EQUITY_TABLE}", "n")
source_tca_count = scalar_value(f"SELECT COUNT(*) AS n FROM {TCA_TABLE}", "n")
all_in_true_count = scalar_value(
    f"""
    SELECT COUNT(*) AS n
    FROM {LEDGER_TABLE}
    WHERE is_v4_strategy_trade = true
      AND all_in_cost_confirmed = true
    """,
    "n",
)
positive_cost_quality_missing = scalar_value(
    f"""
    SELECT COUNT(*) AS n
    FROM {LEDGER_TABLE}
    WHERE is_v4_strategy_trade = true
      AND total_transaction_cost > 0
      AND (cost_observation_quality IS NULL OR trim(cost_observation_quality) = '')
    """,
    "n",
)
latest_stance = spark.sql(
    f"""
    WITH reconstructed AS (
        SELECT strategy_stance_after AS reconstructed_stance, event_ts AS reconstructed_event_ts
        FROM {LEDGER_TABLE}
        WHERE is_v4_strategy_trade = true
        ORDER BY event_ts DESC NULLS LAST, event_sequence DESC NULLS LAST
        LIMIT 1
    ),
    target AS (
        SELECT target_ticker, target_position, execution_mode, trade_ts
        FROM {INTERNAL_TRADE_LOG}
        ORDER BY trade_ts DESC
        LIMIT 1
    )
    SELECT *
    FROM reconstructed
    CROSS JOIN target
    """
)
latest_stance_rows = rows_as_dicts(latest_stance, 1)
latest_stance_match = (
    bool(latest_stance_rows)
    and latest_stance_rows[0]["reconstructed_stance"] == latest_stance_rows[0]["target_ticker"]
)

validation = {
    "latest_files_exist": {dbfs_to_volume_path(path): file_exists(path) for path in latest_files},
    "timestamped_files_exist": {dbfs_to_volume_path(path): file_exists(path) for path in timestamped_files},
    "trade_csv_matches_source_trade_count": row_counts["trades"] == source_trade_count,
    "combined_csv_matches_trade_csv_count": row_counts["combined"] == row_counts["trades"],
    "equity_csv_matches_source_count": row_counts["equity"] == source_equity_count,
    "tca_csv_matches_source_count": row_counts["tca"] == source_tca_count,
    "all_in_cost_confirmed_false_for_strategy_trades": all_in_true_count == 0,
    "positive_cost_rows_have_quality": positive_cost_quality_missing == 0,
    "latest_reconstructed_stance_matches_target": latest_stance_match,
    "latest_stance_check": latest_stance_rows,
    "production_safety": {
        "read_existing_delta_tables_only": True,
        "wrote_only_excel_export_csv_files": True,
        "called_saxo_api": False,
        "touched_token_state": False,
        "modified_orders": False,
        "modified_workflows": False,
        "scheduled_notebook": False,
    },
}

print("Validation")
print(json.dumps(validation, indent=2, default=str))

dbutils.notebook.exit(
    json.dumps(
        {
            "latest_dir": dbfs_to_volume_path(LATEST_DIR_DBFS),
            "timestamped_dir": dbfs_to_volume_path(TIMESTAMPED_DIR_DBFS),
            "exports": exports,
            "row_counts": row_counts,
            "validation": validation,
            "samples": {
                "trades": rows_as_dicts(trades_df, 20),
                "equity": rows_as_dicts(equity_df, 20),
                "combined": rows_as_dicts(combined_df, 20),
                "tca": rows_as_dicts(tca_df, 20),
            },
        },
        default=str,
    )
)

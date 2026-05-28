import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.sql import functions as F


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
CORRECTED_SILVER = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_legacy_recreated_corrected"
CORRECTED_SIGNALS = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_recreated_corrected"
MARKET_TABLE = f"{CATALOG}.default.bronze_bt_market_1h"
CORRECTED_WALKFORWARD_CSV = (
    "/Volumes/pawan_quant_research_1/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "corrected_variant/walkforward_signals.csv"
)
CORRECTED_FULL_SIGNAL_CSV = (
    "/Volumes/pawan_quant_research_1/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "corrected_variant/full_signal_export.csv"
)
CORRECTED_OUTPUT_ROOT = (
    "/Volumes/pawan_quant_research_1/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "codex_confirmation_2026_04_30"
)
HOURLY_EQUITY_ARTIFACT = (
    "/Volumes/pawan_quant_research_1/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "corrected_variant/hourly_equity_curve.csv"
)
DAILY_EQUITY_ARTIFACT = (
    "/Volumes/pawan_quant_research_1/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected/"
    "corrected_variant/daily_equity_curve.csv"
)
EXPECTED = {
    "cagr": 0.208405,
    "sharpe": 1.0024,
    "sortino": 1.3729,
    "max_drawdown": -0.255385,
    "final_equity": 12883454.99,
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
}
TOLERANCE = {
    "cagr": 0.0005,
    "sharpe": 0.01,
    "sortino": 0.02,
    "max_drawdown": 0.005,
    "final_equity": 5000.0,
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
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005


def compute_metrics(nav_df: pd.DataFrame, return_col: str, ann_factor: float) -> dict:
    if nav_df.empty:
        raise ValueError("No rows available for metric computation")
    start_ts = pd.Timestamp(nav_df["timestamp"].iloc[0])
    end_ts = pd.Timestamp(nav_df["timestamp"].iloc[-1])
    cal_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1e-9)
    start_equity = float(nav_df["portfolio_value"].iloc[0])
    final_equity = float(nav_df["portfolio_value"].iloc[-1])
    total_return = final_equity / start_equity - 1.0
    cagr = (final_equity / start_equity) ** (365.0 / cal_days) - 1.0
    returns = nav_df[return_col].dropna()
    if returns.empty:
        ann_vol = sharpe = sortino = 0.0
    else:
        ann_vol = float(returns.std(ddof=1) * math.sqrt(ann_factor))
        mean_ret = float(returns.mean())
        sharpe = 0.0 if ann_vol == 0 else mean_ret / returns.std(ddof=1) * math.sqrt(ann_factor)
        downside = returns[returns < 0]
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = 0.0 if downside_std == 0 else mean_ret / downside_std * math.sqrt(ann_factor)
    running_max = nav_df["portfolio_value"].cummax()
    drawdown = nav_df["portfolio_value"] / running_max - 1.0
    return {
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "obs": int(len(nav_df)),
        "start_equity": start_equity,
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
    }


def trailing_metrics(daily_df: pd.DataFrame, years: int) -> dict:
    end_ts = pd.Timestamp(daily_df["timestamp"].iloc[-1])
    start_cutoff = end_ts - pd.DateOffset(years=years)
    window = daily_df[daily_df["timestamp"] >= start_cutoff].copy()
    if window.empty:
        return {}
    return compute_metrics(window, "daily_return", 252.0)


def annual_returns(daily_df: pd.DataFrame, benchmark_col: str) -> pd.DataFrame:
    tmp = daily_df.copy()
    tmp["year"] = tmp["ny_date"].dt.year
    rows = []
    for year, group in tmp.groupby("year", sort=True):
        v4_ret = float(group["portfolio_value"].iloc[-1] / group["portfolio_value"].iloc[0] - 1.0)
        spy_ret = float(group[benchmark_col].iloc[-1] / group[benchmark_col].iloc[0] - 1.0)
        rows.append({"year": int(year), "v4_return": v4_ret, "spy_return": spy_ret})
    return pd.DataFrame(rows)


def material_diff(actual: float, expected: float, tol: float) -> bool:
    return abs(actual - expected) > tol


def main():
    silver_sdf = spark.table(CORRECTED_SILVER)
    signal_sdf = spark.table(CORRECTED_SIGNALS)

    signal_cols = signal_sdf.columns
    regime_col = None
    for candidate in ["predicted_regime", "regime", "signal", "predicted_label"]:
        if candidate in signal_cols:
            regime_col = candidate
            break
    if regime_col is None:
        raise ValueError(f"Could not locate regime column in {signal_cols}")

    required_silver_cols = {"timestamp", "SPY_close", *FEATURES}
    missing_silver = sorted(required_silver_cols.difference(silver_sdf.columns))
    if missing_silver:
        raise ValueError(f"Missing required corrected silver columns: {missing_silver}")

    table_signals = (
        signal_sdf.select("timestamp", F.col(regime_col).cast("int").alias("regime"))
        .orderBy("timestamp")
        .toPandas()
    )
    corrected_silver = (
        silver_sdf.select("timestamp", "SPY_close", *FEATURES)
        .orderBy("timestamp")
        .toPandas()
    )
    walkforward_csv = spark.read.option("header", True).csv(CORRECTED_WALKFORWARD_CSV).toPandas()
    full_signal_csv = spark.read.option("header", True).csv(CORRECTED_FULL_SIGNAL_CSV).toPandas()
    hourly_equity_artifact = spark.read.option("header", True).csv(HOURLY_EQUITY_ARTIFACT).toPandas()
    daily_eod_artifact = spark.read.option("header", True).csv(DAILY_EQUITY_ARTIFACT).toPandas()

    table_signals["timestamp"] = pd.to_datetime(table_signals["timestamp"], utc=True).dt.tz_localize(None)
    corrected_silver["timestamp"] = pd.to_datetime(corrected_silver["timestamp"], utc=True).dt.tz_localize(None)
    hourly_equity_artifact["timestamp"] = pd.to_datetime(hourly_equity_artifact["timestamp"], utc=True).dt.tz_localize(None)
    daily_eod_artifact["timestamp"] = pd.to_datetime(daily_eod_artifact["timestamp"])

    wf_ts_col = "timestamp" if "timestamp" in walkforward_csv.columns else walkforward_csv.columns[0]
    wf_regime_col = next(
        (c for c in ["predicted_regime", "regime", "signal", "predicted_label"] if c in walkforward_csv.columns),
        None,
    )
    if wf_regime_col is None:
        raise ValueError(f"Could not locate regime column in walkforward CSV {list(walkforward_csv.columns)}")
    walkforward_csv = walkforward_csv[[wf_ts_col, wf_regime_col]].rename(
        columns={wf_ts_col: "timestamp", wf_regime_col: "regime"}
    )
    walkforward_csv["timestamp"] = pd.to_datetime(walkforward_csv["timestamp"], utc=True).dt.tz_localize(None)
    walkforward_csv["regime"] = walkforward_csv["regime"].astype(int)

    required_full_signal_cols = {"timestamp", "SPY_close", "predicted_regime", *FEATURES}
    missing_full_signal = sorted(required_full_signal_cols.difference(full_signal_csv.columns))
    if missing_full_signal:
        raise ValueError(f"Missing required full signal CSV columns: {missing_full_signal}")

    market_prices = (
        spark.sql(
            f"""
            SELECT timestamp, symbol, close
            FROM {MARKET_TABLE}
            WHERE timestamp >= TIMESTAMP('2021-01-01')
              AND timestamp < TIMESTAMP('2026-01-01')
              AND symbol IN ('SPY', 'QQQ')
            """
        )
        .toPandas()
    )
    market_prices["timestamp"] = pd.to_datetime(market_prices["timestamp"], utc=True).dt.tz_localize(None)
    market_prices = (
        market_prices.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .reset_index()
        .sort_values("timestamp")
    )
    market_prices["SPY"] = pd.to_numeric(market_prices["SPY"], errors="coerce").ffill()
    market_prices["QQQ"] = pd.to_numeric(market_prices["QQQ"], errors="coerce").ffill()
    market_prices = market_prices.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

    signal_merge = table_signals.merge(
        walkforward_csv, on="timestamp", how="outer", suffixes=("_table", "_csv"), indicator=True
    )
    mismatch_mask = (
        signal_merge["_merge"] != "both"
    ) | (signal_merge["regime_table"].fillna(-999) != signal_merge["regime_csv"].fillna(-999))
    signal_mismatch_count = int(mismatch_mask.sum())

    data = market_prices.merge(table_signals, on="timestamp", how="inner")
    data = data[(data["timestamp"] >= pd.Timestamp("2021-01-01")) & (data["timestamp"] <= pd.Timestamp("2025-12-31 23:59:59"))]
    data = data.sort_values("timestamp").reset_index(drop=True)

    for col in ["SPY", "QQQ"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["spy_ret"] = data["SPY"].pct_change().fillna(0.0)
    data["qqq_ret"] = data["QQQ"].pct_change().fillna(0.0)
    data["prev_regime"] = data["regime"].shift(1)
    data["trade_regime"] = data["prev_regime"].fillna(data["regime"]).astype(int)
    data["asset_return"] = np.select(
        [data["trade_regime"] == 0, data["trade_regime"] == 1, data["trade_regime"] == 2],
        [data["qqq_ret"], data["spy_ret"], 0.0],
        default=0.0,
    )
    switch_mask = data["trade_regime"] != data["trade_regime"].shift(1)
    switch_mask.iloc[0] = False
    data["net_return"] = data["asset_return"] - np.where(switch_mask, SWITCH_COST, 0.0)
    data["portfolio_value"] = INITIAL_CAPITAL * (1.0 + data["net_return"]).cumprod()
    data["spy_value"] = INITIAL_CAPITAL * (1.0 + data["spy_ret"]).cumprod()
    data["benchmark_value"] = data["spy_value"]

    hourly_metrics = compute_metrics(data, "net_return", ann_factor=252.0 * 7.0)
    spy_hourly = data[["timestamp", "spy_value", "spy_ret"]].copy()
    spy_hourly = spy_hourly.rename(columns={"spy_value": "portfolio_value", "spy_ret": "hourly_return"})
    spy_hourly["hourly_return"] = spy_hourly["hourly_return"].fillna(0.0)
    spy_hourly_metrics = compute_metrics(spy_hourly, "hourly_return", ann_factor=252.0 * 7.0)

    ny_ts = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert("America/New_York")
    data["ny_date"] = pd.to_datetime(ny_ts.dt.date)
    daily_eod = data.groupby("ny_date", as_index=False).tail(1).copy()
    daily_eod["timestamp"] = pd.to_datetime(daily_eod["timestamp"])
    daily_eod["daily_return"] = daily_eod["portfolio_value"].pct_change().fillna(0.0)
    daily_eod["spy_daily_return"] = daily_eod["spy_value"].pct_change().fillna(0.0)

    daily_metrics = compute_metrics(daily_eod, "daily_return", ann_factor=252.0)
    spy_daily = daily_eod[["timestamp", "spy_value", "spy_daily_return"]].copy()
    spy_daily = spy_daily.rename(columns={"spy_value": "portfolio_value", "spy_daily_return": "daily_return"})
    spy_daily_metrics = compute_metrics(spy_daily, "daily_return", ann_factor=252.0)

    annual = annual_returns(daily_eod, "spy_value")
    trailing_rows = []
    for years in [1, 3, 5]:
        v4 = trailing_metrics(daily_eod, years)
        spy = trailing_metrics(
            spy_daily.assign(ny_date=daily_eod["ny_date"], timestamp=daily_eod["timestamp"]),
            years,
        )
        trailing_rows.append(
            {
                "window_years": years,
                "v4_cagr": v4.get("cagr"),
                "v4_sharpe": v4.get("sharpe"),
                "v4_sortino": v4.get("sortino"),
                "v4_max_drawdown": v4.get("max_drawdown"),
                "spy_cagr": spy.get("cagr"),
                "spy_sharpe": spy.get("sharpe"),
                "spy_sortino": spy.get("sortino"),
                "spy_max_drawdown": spy.get("max_drawdown"),
            }
        )
    trailing_df = pd.DataFrame(trailing_rows)

    artifact_hourly = hourly_equity_artifact.copy()
    for col in ["portfolio_value", "strategy_return", "switch_cost", "gross_asset_return", "SPY", "QQQ"]:
        if col in artifact_hourly.columns:
            artifact_hourly[col] = pd.to_numeric(artifact_hourly[col], errors="coerce")
    artifact_daily = daily_eod_artifact.copy()
    artifact_daily["equity"] = pd.to_numeric(artifact_daily["equity"], errors="coerce")
    artifact_daily["portfolio_value"] = artifact_daily["equity"]
    artifact_daily["daily_return"] = artifact_daily["portfolio_value"].pct_change().fillna(0.0)
    artifact_daily_metrics = compute_metrics(
        artifact_daily.rename(columns={"timestamp": "timestamp"}), "daily_return", ann_factor=252.0
    )

    hourly_compare = data[["timestamp", "portfolio_value", "net_return"]].merge(
        artifact_hourly[["timestamp", "portfolio_value", "strategy_return"]],
        on="timestamp",
        how="inner",
        suffixes=("_recomputed", "_artifact"),
    )
    hourly_value_max_abs_diff = float(
        (hourly_compare["portfolio_value_recomputed"] - hourly_compare["portfolio_value_artifact"]).abs().max()
    ) if not hourly_compare.empty else None
    hourly_return_max_abs_diff = float(
        (hourly_compare["net_return"] - hourly_compare["strategy_return"]).abs().max()
    ) if not hourly_compare.empty else None

    confirmation = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "source_tables": {
            "corrected_silver": CORRECTED_SILVER,
            "corrected_signals": CORRECTED_SIGNALS,
            "market_table": MARKET_TABLE,
        },
        "source_files": {
            "walkforward_signals_csv": CORRECTED_WALKFORWARD_CSV,
            "full_signal_export_csv": CORRECTED_FULL_SIGNAL_CSV,
            "hourly_equity_artifact": HOURLY_EQUITY_ARTIFACT,
            "daily_equity_artifact": DAILY_EQUITY_ARTIFACT,
        },
        "row_counts": {
            "corrected_signals_table": int(len(table_signals)),
            "corrected_silver_table": int(len(corrected_silver)),
            "market_price_rows": int(len(market_prices)),
            "walkforward_signals_csv": int(len(walkforward_csv)),
            "full_signal_export_csv": int(len(full_signal_csv)),
            "hourly_equity_artifact": int(len(hourly_equity_artifact)),
            "daily_eod_artifact": int(len(daily_eod_artifact)),
            "daily_eod_rows": int(len(daily_eod)),
        },
        "signal_csv_match": {
            "mismatch_count": signal_mismatch_count,
            "match_rate": 0.0
            if len(signal_merge) == 0
            else float((len(signal_merge) - signal_mismatch_count) / len(signal_merge)),
        },
        "feature_schema": {
            "expected_count": len(FEATURES),
            "all_present": True,
            "features": FEATURES,
        },
        "methodology": {
            "execution_semantics": "prior_bar",
            "transaction_cost": SWITCH_COST,
            "hourly_reporting": "hourly_returns_with_252x7_annualization",
            "daily_eod_reporting": "new_york_date_last_timestamp_with_252_annualization",
            "cagr_basis": "calendar_days_divided_by_365",
        },
        "hourly_metrics": {
            "v4": hourly_metrics,
            "spy": spy_hourly_metrics,
        },
        "daily_eod_metrics": {
            "v4": daily_metrics,
            "spy": spy_daily_metrics,
        },
        "artifact_daily_eod_metrics": artifact_daily_metrics,
        "recompute_vs_artifact": {
            "hourly_value_max_abs_diff": hourly_value_max_abs_diff,
            "hourly_return_max_abs_diff": hourly_return_max_abs_diff,
            "artifact_daily_cagr_diff": float(daily_metrics["cagr"] - artifact_daily_metrics["cagr"]),
            "artifact_daily_final_equity_diff": float(daily_metrics["final_equity"] - artifact_daily_metrics["final_equity"]),
        },
        "annual_returns": annual.to_dict(orient="records"),
        "trailing_metrics": trailing_df.to_dict(orient="records"),
        "expected_check": {
            "cagr_within_tolerance": not material_diff(daily_metrics["cagr"], EXPECTED["cagr"], TOLERANCE["cagr"]),
            "max_drawdown_within_tolerance": not material_diff(
                daily_metrics["max_drawdown"], EXPECTED["max_drawdown"], TOLERANCE["max_drawdown"]
            ),
            "final_equity_within_tolerance": not material_diff(
                daily_metrics["final_equity"], EXPECTED["final_equity"], TOLERANCE["final_equity"]
            ),
            "matches_expected_period": (
                daily_metrics["start_date"] == EXPECTED["start_date"]
                and daily_metrics["end_date"] == EXPECTED["end_date"]
            ),
        },
    }
    confirmation["expected_check"]["all_pass"] = all(confirmation["expected_check"].values())

    daily_eod["ny_date"] = daily_eod["ny_date"].astype(str)
    annual["year"] = annual["year"].astype(int)

    dbutils.fs.mkdirs(CORRECTED_OUTPUT_ROOT)
    dbutils.fs.put(
        f"{CORRECTED_OUTPUT_ROOT}/codex_corrected_alfred_confirmation.json",
        json.dumps(confirmation, indent=2),
        overwrite=True,
    )
    spark.createDataFrame(
        daily_eod[
            [
                "timestamp",
                "ny_date",
                "regime",
                "trade_regime",
                "portfolio_value",
                "spy_value",
                "daily_return",
                "spy_daily_return",
            ]
        ]
    ).coalesce(1).write.mode("overwrite").option("header", True).csv(
        f"{CORRECTED_OUTPUT_ROOT}/daily_eod_equity_curve_csv"
    )
    spark.createDataFrame(annual).coalesce(1).write.mode("overwrite").option("header", True).csv(
        f"{CORRECTED_OUTPUT_ROOT}/annual_returns_csv"
    )
    spark.createDataFrame(trailing_df).coalesce(1).write.mode("overwrite").option("header", True).csv(
        f"{CORRECTED_OUTPUT_ROOT}/trailing_metrics_csv"
    )
    print(json.dumps(confirmation, indent=2))


main()

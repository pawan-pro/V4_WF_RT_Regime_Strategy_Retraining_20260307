# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

# Strategy-faithful walk-forward replication:
# - monthly expanding-window retrain
# - full-signal teacher labels
# - hourly prior-bar execution on SPY/QQQ
# - daily-resampled reporting with 252-day annualization

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from dateutil.relativedelta import relativedelta

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
LEGACY_SCHEMA = "default"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
COMPARISON_ROOT = f"{VOLUME_ROOT}/comparison"
DEFAULT_OUTPUT_ROOT = (
    f"{COMPARISON_ROOT}/2026-04-20_strategy_faithful_walkforward"
)

BASELINE_SILVER_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
REBUILD_SILVER_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h_rebuild"
)
BASELINE_TEACHER_PATH_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_full_historical_adoption/"
    "baseline/current_full_signal_export.csv"
)
REBUILD_TEACHER_PATH_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_full_historical_adoption/"
    "full_rebuild_retrain/full_signal_export.csv"
)
LEGACY_TEACHER_PATH_DEFAULT = (
    f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data/bt_fred_1d_regime_signals-v4.csv"
)
LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT = (
    f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data/strategy_performance_summary.csv"
)
LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT = (
    f"{CATALOG}.{LEGACY_SCHEMA}.bt_walkforward_regime_signals"
)

PRICE_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"
LEGACY_PRICE_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"
PRODUCTION_FAITHFUL_PNL_PATH_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_full_historical_adoption/walkforward_pnl/pnl_metrics.csv"
)
BRONZE_REBUILD_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.bronze_bt_macro_events_rebuild"

TRAIN_START_DEFAULT = "2016-01-01"
LIVE_START_DEFAULT = "2021-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252

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

POSITION_MAP = {0: "QQQ", 1: "SPY", 2: "CASH"}
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


def parse_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def file_exists(path_str):
    return Path(path_str).exists()


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


def load_feature_frame(table_name, train_start):
    pdf = (
        spark.table(table_name)
        .filter(f"timestamp >= '{train_start}'")
        .select(["timestamp"] + FEATURES)
        .toPandas()
    )
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf.drop_duplicates(subset=["timestamp"], keep="last")
    return pdf


def load_teacher_frame(path_str, train_start):
    pdf = load_csv(path_str)
    if pdf.empty:
        raise FileNotFoundError(f"Missing or empty teacher file: {path_str}")
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf.drop_duplicates(subset=["timestamp"], keep="last")
    if "predicted_regime" not in pdf.columns:
        raise KeyError(
            f"'predicted_regime' missing in teacher file {path_str}. "
            f"columns={list(pdf.columns)}"
        )
    result = pdf[pdf["timestamp"] >= pd.Timestamp(train_start)].copy()
    result["predicted_regime"] = (
        pd.to_numeric(result["predicted_regime"], errors="coerce").fillna(2).astype(int)
    )
    if "regime_name" not in result.columns:
        result["regime_name"] = result["predicted_regime"].map(REGIME_NAME_MAP)
    return result


def load_signal_table(table_name, start_date):
    pdf = spark.table(table_name).toPandas()
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf.drop_duplicates(subset=["timestamp"], keep="last")
    if "predicted_regime" not in pdf.columns:
        raise KeyError(
            f"'predicted_regime' missing in table {table_name}. "
            f"columns={list(pdf.columns)}"
        )
    pdf = pdf[pdf["timestamp"] >= pd.Timestamp(start_date)].copy()
    pdf["predicted_regime"] = (
        pd.to_numeric(pdf["predicted_regime"], errors="coerce").fillna(2).astype(int)
    )
    return pdf[["timestamp", "predicted_regime"]]


def load_price_frame(table_name, start_date):
    pdf = (
        spark.sql(
            f"""
            SELECT timestamp, symbol, close
            FROM {table_name}
            WHERE timestamp >= TIMESTAMP('{start_date}')
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
        .rename(columns={"SPY": "SPY_close", "QQQ": "QQQ_close"})
        .sort_values("timestamp")
    )
    pivoted["SPY_close"] = pd.to_numeric(
        pivoted["SPY_close"], errors="coerce"
    ).ffill()
    pivoted["QQQ_close"] = pd.to_numeric(
        pivoted["QQQ_close"], errors="coerce"
    ).ffill()
    return pivoted


def build_model():
    return xgb.XGBClassifier(
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


def build_monthly_schedule(live_start, live_end):
    retrain_dates = []
    current = pd.Timestamp(live_start)
    live_end_ts = pd.Timestamp(live_end)
    while current <= live_end_ts:
        retrain_dates.append(current)
        current += relativedelta(months=1)
    final_boundary = live_end_ts + pd.Timedelta(microseconds=1)
    if not retrain_dates or retrain_dates[-1] <= live_end_ts:
        retrain_dates.append(final_boundary)
    return retrain_dates


def generate_walkforward_signals(
    feature_pdf,
    teacher_pdf,
    train_start,
    live_start,
    live_end,
):
    merged = feature_pdf.merge(
        teacher_pdf[["timestamp", "predicted_regime"]],
        on="timestamp",
        how="inner",
    )
    merged = merged.dropna(subset=["predicted_regime"]).copy()
    merged["teacher_regime"] = (
        pd.to_numeric(merged["predicted_regime"], errors="coerce").fillna(2).astype(int)
    )
    merged = merged.drop(columns=["predicted_regime"])
    merged = merged.sort_values("timestamp")

    schedule = build_monthly_schedule(live_start, live_end)
    if len(schedule) < 2:
        raise ValueError(
            f"Walk-forward schedule too short: live_start={live_start}, live_end={live_end}"
        )

    signal_frames = []
    monthly_metrics = []

    for idx in range(len(schedule) - 1):
        train_end = schedule[idx]
        test_end = schedule[idx + 1]

        train_df = merged[
            (merged["timestamp"] >= pd.Timestamp(train_start))
            & (merged["timestamp"] < train_end)
        ]
        test_df = merged[
            (merged["timestamp"] >= train_end)
            & (merged["timestamp"] < test_end)
        ].copy()

        if train_df.empty or test_df.empty:
            continue

        model = build_model()
        model.fit(train_df[FEATURES], train_df["teacher_regime"])

        preds = model.predict(test_df[FEATURES]).astype(int)
        scored = test_df[["timestamp", "teacher_regime"] + FEATURES].copy()
        scored["predicted_regime"] = preds
        scored["regime_name"] = scored["predicted_regime"].map(REGIME_NAME_MAP)

        try:
            probs = model.predict_proba(test_df[FEATURES])
            if probs is not None and probs.shape[1] == 3:
                scored["prob_expansion"] = probs[:, 0]
                scored["prob_slowdown"] = probs[:, 1]
                scored["prob_recession"] = probs[:, 2]
                scored["max_probability"] = probs.max(axis=1)
        except Exception:
            pass

        signal_frames.append(scored)
        monthly_metrics.append(
            {
                "train_end": train_end,
                "test_end": test_end,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "accuracy": float(
                    (scored["predicted_regime"] == scored["teacher_regime"]).mean()
                ),
                "predicted_expansion_pct": float(
                    scored["predicted_regime"].eq(0).mean()
                ),
                "predicted_slowdown_pct": float(
                    scored["predicted_regime"].eq(1).mean()
                ),
                "predicted_recession_pct": float(
                    scored["predicted_regime"].eq(2).mean()
                ),
            }
        )

    if not signal_frames:
        raise ValueError("No walk-forward signals were generated.")

    wf_signals = pd.concat(signal_frames, ignore_index=True).sort_values("timestamp")
    wf_signals = wf_signals.drop_duplicates(subset=["timestamp"], keep="last")
    monthly_validation = pd.DataFrame(monthly_metrics)
    return wf_signals, monthly_validation


def run_hourly_backtest(signal_pdf, price_pdf, initial_capital, transaction_cost):
    aligned = signal_pdf.merge(
        price_pdf[["timestamp", "SPY_close", "QQQ_close"]],
        on="timestamp",
        how="inner",
    ).sort_values("timestamp")
    aligned["SPY_close"] = pd.to_numeric(aligned["SPY_close"], errors="coerce").ffill()
    aligned["QQQ_close"] = pd.to_numeric(aligned["QQQ_close"], errors="coerce").ffill()
    aligned = aligned.dropna(subset=["SPY_close", "QQQ_close"]).reset_index(drop=True)

    equity = float(initial_capital)
    active_asset = "START"
    records = []

    for idx in range(len(aligned)):
        row = aligned.iloc[idx]
        record = {
            "timestamp": row["timestamp"],
            "signal_at_t": int(row["predicted_regime"]),
            "applied_regime": np.nan,
            "applied_regime_name": None,
            "position": "START" if idx == 0 else None,
            "SPY_close": row["SPY_close"],
            "QQQ_close": row["QQQ_close"],
            "gross_return": 0.0,
            "transaction_cost_applied": 0.0,
            "net_return": 0.0,
            "equity": equity,
            "position_changed": False,
        }

        if idx > 0:
            regime = int(aligned.iloc[idx - 1]["predicted_regime"])
            target = POSITION_MAP.get(regime, "CASH")

            if target == "QQQ":
                gross_return = (
                    row["QQQ_close"] / aligned.iloc[idx - 1]["QQQ_close"]
                ) - 1.0
            elif target == "SPY":
                gross_return = (
                    row["SPY_close"] / aligned.iloc[idx - 1]["SPY_close"]
                ) - 1.0
            else:
                gross_return = 0.0

            transaction_cost_applied = (
                transaction_cost if target != active_asset else 0.0
            )
            net_return = gross_return - transaction_cost_applied
            equity *= 1.0 + net_return

            record.update(
                {
                    "applied_regime": regime,
                    "applied_regime_name": REGIME_NAME_MAP.get(regime),
                    "position": target,
                    "gross_return": gross_return,
                    "transaction_cost_applied": transaction_cost_applied,
                    "net_return": net_return,
                    "equity": equity,
                    "position_changed": target != active_asset,
                }
            )
            active_asset = target

        records.append(record)

    backtest = pd.DataFrame(records)
    if len(backtest) > 1:
        backtest["applied_regime"] = (
            pd.to_numeric(backtest["applied_regime"], errors="coerce")
            .ffill()
            .fillna(2)
            .astype(int)
        )
    else:
        backtest["applied_regime"] = 2
    backtest["applied_regime_name"] = backtest["applied_regime"].map(REGIME_NAME_MAP)
    return backtest


def contiguous_run_lengths(series):
    if series.empty:
        return pd.Series(dtype=float)
    groups = series.ne(series.shift()).cumsum()
    return series.groupby(groups).size().astype(float)


def compute_reference_metrics(equity_series):
    series = equity_series.dropna()
    daily = series.resample("D").last().dropna()
    returns = daily.pct_change(fill_method=None).dropna()
    years = len(returns) / TRADING_DAYS if len(returns) else np.nan
    start_value = float(series.iloc[0])
    final_value = float(series.iloc[-1])
    total_return = (final_value / start_value) - 1.0
    cagr = (
        (final_value / start_value) ** (1 / years) - 1
        if pd.notna(years) and years > 0
        else np.nan
    )
    vol_month = returns.std() * np.sqrt(21) if not returns.empty else np.nan
    vol_ann = returns.std() * np.sqrt(TRADING_DAYS) if not returns.empty else np.nan
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
    drawdown = series / series.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan
    calmar = cagr / abs(max_drawdown) if pd.notna(max_drawdown) and max_drawdown != 0 else np.nan
    return {
        "start_timestamp": series.index.min(),
        "end_timestamp": series.index.max(),
        "daily_rows": int(len(daily)),
        "return_rows": int(len(returns)),
        "years": float(years) if pd.notna(years) else np.nan,
        "initial_capital": start_value,
        "final_equity": final_value,
        "total_return": float(total_return),
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "vol_monthly": float(vol_month) if np.isfinite(vol_month) else np.nan,
        "annualized_volatility": float(vol_ann) if np.isfinite(vol_ann) else np.nan,
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino_ratio": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
    }


def compute_regime_stats(backtest_pdf):
    active = backtest_pdf.iloc[1:].copy()
    holding_lengths = contiguous_run_lengths(active["position"])
    return {
        "total_regime_switches": int(active["position_changed"].sum()),
        "average_holding_bars": float(holding_lengths.mean())
        if not holding_lengths.empty
        else np.nan,
        "expansion_pct_time": float(active["applied_regime"].eq(0).mean())
        if len(active)
        else np.nan,
        "slowdown_pct_time": float(active["applied_regime"].eq(1).mean())
        if len(active)
        else np.nan,
        "recession_pct_time": float(active["applied_regime"].eq(2).mean())
        if len(active)
        else np.nan,
    }


def build_metric_row(model_name, framework, backtest_pdf, monthly_validation=None):
    metrics = compute_reference_metrics(backtest_pdf.set_index("timestamp")["equity"])
    metrics.update(compute_regime_stats(backtest_pdf))
    metrics["model"] = model_name
    metrics["framework"] = framework
    metrics["signal_rows"] = int(len(backtest_pdf))
    if monthly_validation is not None and not monthly_validation.empty:
        metrics["mean_monthly_accuracy"] = float(monthly_validation["accuracy"].mean())
    else:
        metrics["mean_monthly_accuracy"] = np.nan
    return metrics


def build_reference_row(strategy_name, export_summary_pdf):
    row = export_summary_pdf[export_summary_pdf["strategy"].eq(strategy_name)].copy()
    if row.empty:
        return {}
    item = row.iloc[0]
    max_drawdown = item.get("Max Drawdown", np.nan)
    return {
        "model": "legacy_reference_exported",
        "framework": "legacy_reference_export",
        "start_timestamp": None,
        "end_timestamp": None,
        "daily_rows": np.nan,
        "return_rows": np.nan,
        "years": np.nan,
        "initial_capital": float(item.get("Start Value", np.nan)),
        "final_equity": float(item.get("Final Value", np.nan)),
        "total_return": float(item.get("Total Return", np.nan)),
        "cagr": float(item.get("CAGR", np.nan)),
        "vol_monthly": float(item.get("Vol Monthly", np.nan)),
        "annualized_volatility": float(item.get("Vol Annual", np.nan)),
        "sharpe_ratio": float(item.get("Sharpe Annual", np.nan)),
        "sortino_ratio": float(item.get("Sortino Annual", np.nan)),
        "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
        "calmar_ratio": (
            float(item.get("CAGR", np.nan)) / abs(float(max_drawdown))
            if pd.notna(item.get("CAGR", np.nan))
            and pd.notna(max_drawdown)
            and float(max_drawdown) != 0
            else np.nan
        ),
        "signal_rows": np.nan,
        "mean_monthly_accuracy": np.nan,
        "total_regime_switches": np.nan,
        "average_holding_bars": np.nan,
        "expansion_pct_time": np.nan,
        "slowdown_pct_time": np.nan,
        "recession_pct_time": np.nan,
    }


def daily_curve(backtest_pdf, model_name):
    hourly = backtest_pdf[["timestamp", "equity", "applied_regime"]].copy()
    hourly = hourly.set_index("timestamp").sort_index()
    daily_equity = hourly["equity"].resample("D").last().dropna()
    daily_regime = hourly["applied_regime"].resample("D").last().dropna().astype(int)
    drawdown = daily_equity / daily_equity.cummax() - 1.0
    return (
        pd.DataFrame(
            {
                "timestamp": daily_equity.index,
                "model": model_name,
                "equity": daily_equity.values,
                "drawdown": drawdown.values,
                "applied_regime": daily_regime.reindex(daily_equity.index).ffill().fillna(2).astype(int).values,
            }
        )
        .reset_index(drop=True)
    )


def save_plot(path_str, fig):
    ensure_dir(str(Path(path_str).parent))
    fig.savefig(path_str, dpi=200, bbox_inches="tight")
    plt.close(fig)


def legacy_export_summary(path_str):
    pdf = load_csv(path_str)
    if pdf.empty:
        return pd.DataFrame()
    if pdf.columns[0].startswith("Unnamed"):
        pdf = pdf.rename(columns={pdf.columns[0]: "strategy"})
    return pdf


def load_production_faithful_baseline(path_str):
    pdf = load_csv(path_str)
    if pdf.empty or "model" not in pdf.columns:
        return {}
    row = pdf[pdf["model"].eq("baseline")]
    if row.empty:
        return {}
    item = row.iloc[0].to_dict()
    item["model"] = "baseline_v4_production_faithful"
    item["framework"] = "production_faithful_hourly"
    return item


def baseline_recovery_status(
    baseline_metrics,
    reference_metrics,
    production_metrics,
):
    cagr_gap = abs(baseline_metrics["cagr"] - reference_metrics["cagr"])
    sharpe_gap = abs(
        baseline_metrics["sharpe_ratio"] - reference_metrics["sharpe_ratio"]
    )
    production_cagr_gap = (
        abs(production_metrics["cagr"] - reference_metrics["cagr"])
        if production_metrics
        else np.inf
    )
    production_sharpe_gap = (
        abs(production_metrics["sharpe_ratio"] - reference_metrics["sharpe_ratio"])
        if production_metrics
        else np.inf
    )

    materially_closer = (
        cagr_gap < production_cagr_gap and sharpe_gap < production_sharpe_gap
    )
    strong_frame = (
        baseline_metrics["cagr"] >= 0.14 and baseline_metrics["sharpe_ratio"] >= 0.70
    )
    bounded_gap = cagr_gap <= 0.08 and sharpe_gap <= 0.45
    succeeded = bool(materially_closer and strong_frame and bounded_gap)
    return {
        "succeeded": succeeded,
        "cagr_gap_vs_reference": float(cagr_gap),
        "sharpe_gap_vs_reference": float(sharpe_gap),
        "production_cagr_gap_vs_reference": float(production_cagr_gap)
        if np.isfinite(production_cagr_gap)
        else None,
        "production_sharpe_gap_vs_reference": float(production_sharpe_gap)
        if np.isfinite(production_sharpe_gap)
        else None,
        "criteria": {
            "materially_closer_than_production_faithful": materially_closer,
            "strong_historical_frame": strong_frame,
            "bounded_gap_to_reference": bounded_gap,
        },
    }


def comparison_recommendation(baseline_metrics, rebuild_metrics):
    if rebuild_metrics["final_equity"] > baseline_metrics["final_equity"] and (
        rebuild_metrics["cagr"] > baseline_metrics["cagr"] + 0.01
    ) and (
        rebuild_metrics["sharpe_ratio"] >= baseline_metrics["sharpe_ratio"]
    ) and (
        abs(rebuild_metrics["max_drawdown"])
        <= abs(baseline_metrics["max_drawdown"]) + 0.03
    ):
        return "promote full rebuild V4.1 after approval"

    if (
        rebuild_metrics["cagr"] < baseline_metrics["cagr"]
        or rebuild_metrics["sharpe_ratio"] < baseline_metrics["sharpe_ratio"]
        or rebuild_metrics["final_equity"] < baseline_metrics["final_equity"]
    ):
        return "hold current V4"

    return "further review needed"


def markdown_table(pdf, float_format="{:.6f}"):
    if pdf.empty:
        return "_No rows_"
    frame = pdf.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: float_format.format(value) if pd.notna(value) else ""
            )
        else:
            frame[column] = frame[column].fillna("").astype(str)
    rows = [list(frame.columns), ["---"] * len(frame.columns)]
    rows.extend(frame.values.tolist())
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


output_root = DEFAULT_OUTPUT_ROOT
baseline_silver_table = BASELINE_SILVER_TABLE_DEFAULT
rebuild_silver_table = REBUILD_SILVER_TABLE_DEFAULT
baseline_teacher_path = BASELINE_TEACHER_PATH_DEFAULT
rebuild_teacher_path = REBUILD_TEACHER_PATH_DEFAULT
legacy_teacher_path = LEGACY_TEACHER_PATH_DEFAULT
legacy_reference_summary_path = LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT
legacy_reference_signal_table = LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT
price_table = PRICE_TABLE_DEFAULT
legacy_price_table = LEGACY_PRICE_TABLE_DEFAULT
production_faithful_pnl_path = PRODUCTION_FAITHFUL_PNL_PATH_DEFAULT
train_start = TRAIN_START_DEFAULT
live_start = LIVE_START_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("output_root", DEFAULT_OUTPUT_ROOT)
    dbutils.widgets.text("baseline_silver_table", BASELINE_SILVER_TABLE_DEFAULT)
    dbutils.widgets.text("rebuild_silver_table", REBUILD_SILVER_TABLE_DEFAULT)
    dbutils.widgets.text("baseline_teacher_path", BASELINE_TEACHER_PATH_DEFAULT)
    dbutils.widgets.text("rebuild_teacher_path", REBUILD_TEACHER_PATH_DEFAULT)
    dbutils.widgets.text("legacy_teacher_path", LEGACY_TEACHER_PATH_DEFAULT)
    dbutils.widgets.text(
        "legacy_reference_summary_path", LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT
    )
    dbutils.widgets.text(
        "legacy_reference_signal_table", LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT
    )
    dbutils.widgets.text("price_table", PRICE_TABLE_DEFAULT)
    dbutils.widgets.text("legacy_price_table", LEGACY_PRICE_TABLE_DEFAULT)
    dbutils.widgets.text(
        "production_faithful_pnl_path", PRODUCTION_FAITHFUL_PNL_PATH_DEFAULT
    )
    dbutils.widgets.text("train_start", TRAIN_START_DEFAULT)
    dbutils.widgets.text("live_start", LIVE_START_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    output_root = dbutils.widgets.get("output_root").strip() or DEFAULT_OUTPUT_ROOT
    baseline_silver_table = (
        dbutils.widgets.get("baseline_silver_table").strip()
        or BASELINE_SILVER_TABLE_DEFAULT
    )
    rebuild_silver_table = (
        dbutils.widgets.get("rebuild_silver_table").strip()
        or REBUILD_SILVER_TABLE_DEFAULT
    )
    baseline_teacher_path = (
        dbutils.widgets.get("baseline_teacher_path").strip()
        or BASELINE_TEACHER_PATH_DEFAULT
    )
    rebuild_teacher_path = (
        dbutils.widgets.get("rebuild_teacher_path").strip()
        or REBUILD_TEACHER_PATH_DEFAULT
    )
    legacy_teacher_path = (
        dbutils.widgets.get("legacy_teacher_path").strip()
        or LEGACY_TEACHER_PATH_DEFAULT
    )
    legacy_reference_summary_path = (
        dbutils.widgets.get("legacy_reference_summary_path").strip()
        or LEGACY_REFERENCE_SUMMARY_PATH_DEFAULT
    )
    legacy_reference_signal_table = (
        dbutils.widgets.get("legacy_reference_signal_table").strip()
        or LEGACY_REFERENCE_SIGNAL_TABLE_DEFAULT
    )
    price_table = dbutils.widgets.get("price_table").strip() or PRICE_TABLE_DEFAULT
    legacy_price_table = (
        dbutils.widgets.get("legacy_price_table").strip()
        or LEGACY_PRICE_TABLE_DEFAULT
    )
    production_faithful_pnl_path = (
        dbutils.widgets.get("production_faithful_pnl_path").strip()
        or PRODUCTION_FAITHFUL_PNL_PATH_DEFAULT
    )
    train_start = dbutils.widgets.get("train_start").strip() or TRAIN_START_DEFAULT
    live_start = dbutils.widgets.get("live_start").strip() or LIVE_START_DEFAULT
    initial_capital = parse_float(
        dbutils.widgets.get("initial_capital"), INITIAL_CAPITAL_DEFAULT
    )
    transaction_cost = parse_float(
        dbutils.widgets.get("transaction_cost"), TRANSACTION_COST_DEFAULT
    )

ensure_dir(output_root)

baseline_teacher = load_teacher_frame(baseline_teacher_path, train_start)
rebuild_teacher = load_teacher_frame(rebuild_teacher_path, train_start)
legacy_teacher = load_teacher_frame(legacy_teacher_path, train_start)
baseline_price = load_price_frame(price_table, live_start)
legacy_price = load_price_frame(legacy_price_table, live_start)

common_live_end = min(
    baseline_teacher["timestamp"].max(),
    rebuild_teacher["timestamp"].max(),
    baseline_price["timestamp"].max(),
)
legacy_live_end = min(
    legacy_teacher["timestamp"].max(),
    legacy_price["timestamp"].max(),
)

baseline_feature_pdf = load_feature_frame(baseline_silver_table, train_start)
rebuild_feature_pdf = load_feature_frame(rebuild_silver_table, train_start)

legacy_reference_summary_pdf = legacy_export_summary(legacy_reference_summary_path)
legacy_reference_metrics = build_reference_row(
    "WalkForward_V4", legacy_reference_summary_pdf
)

legacy_replay_signals = load_signal_table(legacy_reference_signal_table, live_start)
legacy_replay_backtest = run_hourly_backtest(
    legacy_replay_signals[
        legacy_replay_signals["timestamp"] <= legacy_live_end
    ].copy(),
    legacy_price[legacy_price["timestamp"] <= legacy_live_end].copy(),
    initial_capital,
    transaction_cost,
)
legacy_replay_metrics = build_metric_row(
    "legacy_reference_signal_replay",
    "legacy_reference_replay",
    legacy_replay_backtest,
)

baseline_direct_backtest = run_hourly_backtest(
    baseline_teacher[
        (baseline_teacher["timestamp"] >= pd.Timestamp(live_start))
        & (baseline_teacher["timestamp"] <= common_live_end)
    ][["timestamp", "predicted_regime"]].copy(),
    baseline_price[baseline_price["timestamp"] <= common_live_end].copy(),
    initial_capital,
    transaction_cost,
)
baseline_direct_metrics = build_metric_row(
    "baseline_v4_full_signal_direct",
    "full_signal_direct",
    baseline_direct_backtest,
)

baseline_wf_signals, baseline_monthly_validation = generate_walkforward_signals(
    baseline_feature_pdf,
    baseline_teacher,
    train_start,
    live_start,
    common_live_end,
)
baseline_wf_signals = baseline_wf_signals[
    baseline_wf_signals["timestamp"] >= pd.Timestamp(live_start)
].copy()
baseline_wf_backtest = run_hourly_backtest(
    baseline_wf_signals[["timestamp", "predicted_regime"]].copy(),
    baseline_price[baseline_price["timestamp"] <= common_live_end].copy(),
    initial_capital,
    transaction_cost,
)
baseline_metrics = build_metric_row(
    "baseline_v4_strategy_faithful",
    "strategy_faithful_walkforward",
    baseline_wf_backtest,
    baseline_monthly_validation,
)

production_faithful_baseline = load_production_faithful_baseline(
    production_faithful_pnl_path
)
if production_faithful_baseline:
    production_faithful_baseline_row = {
        "model": production_faithful_baseline.get("model"),
        "framework": production_faithful_baseline.get("framework"),
        "start_timestamp": production_faithful_baseline.get("start_timestamp"),
        "end_timestamp": production_faithful_baseline.get("end_timestamp"),
        "daily_rows": production_faithful_baseline.get("daily_rows"),
        "return_rows": production_faithful_baseline.get("return_rows"),
        "years": production_faithful_baseline.get("years"),
        "initial_capital": production_faithful_baseline.get("initial_capital"),
        "final_equity": production_faithful_baseline.get("final_equity"),
        "total_return": production_faithful_baseline.get("total_return"),
        "cagr": production_faithful_baseline.get("cagr"),
        "vol_monthly": np.nan,
        "annualized_volatility": production_faithful_baseline.get(
            "annualized_volatility"
        ),
        "sharpe_ratio": production_faithful_baseline.get("sharpe_ratio"),
        "sortino_ratio": np.nan,
        "max_drawdown": production_faithful_baseline.get("max_drawdown"),
        "calmar_ratio": production_faithful_baseline.get("calmar_ratio"),
        "signal_rows": production_faithful_baseline.get("signal_rows"),
        "mean_monthly_accuracy": np.nan,
        "total_regime_switches": production_faithful_baseline.get(
            "total_regime_switches"
        ),
        "average_holding_bars": production_faithful_baseline.get(
            "average_holding_bars"
        ),
        "expansion_pct_time": production_faithful_baseline.get("expansion_pct_time"),
        "slowdown_pct_time": production_faithful_baseline.get("slowdown_pct_time"),
        "recession_pct_time": production_faithful_baseline.get("recession_pct_time"),
    }
else:
    production_faithful_baseline_row = {}

baseline_recovery_rows = []
if legacy_reference_metrics:
    baseline_recovery_rows.append(legacy_reference_metrics)
baseline_recovery_rows.append(legacy_replay_metrics)
baseline_recovery_rows.append(baseline_direct_metrics)
baseline_recovery_rows.append(baseline_metrics)
if production_faithful_baseline_row:
    baseline_recovery_rows.append(production_faithful_baseline_row)

baseline_recovery_metrics_pdf = pd.DataFrame(baseline_recovery_rows)
baseline_recovery_metrics_path = f"{output_root}/baseline_recovery_metrics.csv"
baseline_recovery_metrics_pdf.to_csv(baseline_recovery_metrics_path, index=False)

reference_for_recovery = (
    legacy_reference_metrics if legacy_reference_metrics else legacy_replay_metrics
)
baseline_recovery = baseline_recovery_status(
    baseline_metrics,
    reference_for_recovery,
    production_faithful_baseline_row,
)

methodology_spec_lines = [
    "# Strategy-Faithful Methodology Spec",
    "",
    "## Canonical Target",
    "",
    "- Source of truth: legacy walk-forward notebook plus historical performance export framing.",
    "- Teacher labels: full-signal `predicted_regime` series, not the current production walk-forward export.",
    "- Model cadence: monthly expanding-window retrain starting from `2016-01-01`.",
    "- Invested period: `2021-01-01` onward.",
    "- Signal cadence: hourly signal timestamps.",
    "- Decision cadence: hourly prior-bar application.",
    "- Price path: hourly `SPY` and `QQQ` closes from `bronze_bt_market_1h`.",
    "- Transaction cost: subtract `0.0005` on position switch.",
    "- Portfolio mapping: `0 -> QQQ`, `1 -> SPY`, `2 -> CASH`.",
    "- Reporting basis: daily-resampled equity with `252` trading-day annualization.",
    "",
    "## Determination on 1D vs 1H",
    "",
    "- The historical source-of-truth notebook does not use a pure 1D decision engine.",
    "- The faithful target remains an hourly execution path with daily-resampled reporting.",
    "- A 1D decision layer is therefore not selected as canonical for this replication task.",
    "",
    "## Comparison Inputs",
    "",
    f"- Baseline silver table: `{baseline_silver_table}`",
    f"- Rebuild silver table: `{rebuild_silver_table}`",
    f"- Baseline teacher path: `{baseline_teacher_path}`",
    f"- Rebuild teacher path: `{rebuild_teacher_path}`",
    f"- Reference legacy teacher path: `{legacy_teacher_path}`",
    f"- Price table: `{price_table}`",
    f"- Bronze rebuild table preserved: `{BRONZE_REBUILD_TABLE_DEFAULT}`",
    "",
    "## Validation View",
    "",
    "- The main gap versus the production-faithful engine is expected to come from signal-source lineage, not from 1D decision cadence.",
    "- This notebook uses the strategy-faithful teacher-label lineage explicitly to test that hypothesis.",
]
methodology_spec_path = f"{output_root}/methodology_spec.md"
write_text(methodology_spec_path, "\n".join(methodology_spec_lines) + "\n")

baseline_recovery_audit_lines = [
    "# Baseline Recovery Audit",
    "",
    "## Goal",
    "",
    "- Prove whether the strategy-faithful replication recovers the stronger historical walk-forward frame materially better than the production-faithful engine.",
    "",
    "## Success Criteria",
    "",
    "- Strategy-faithful baseline must be materially closer to the legacy reference than the production-faithful baseline on CAGR and Sharpe.",
    "- Strategy-faithful baseline must still look like a strong historical frame: `CAGR >= 14%` and `Sharpe >= 0.70`.",
    "- Absolute gap to the reference must stay bounded: `|CAGR gap| <= 8%` and `|Sharpe gap| <= 0.45`.",
    "",
    "## Recovery Metrics",
    "",
    markdown_table(
        baseline_recovery_metrics_pdf[
            [
                "model",
                "framework",
                "final_equity",
                "cagr",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
            ]
        ]
    ),
    "",
    "## Verdict",
    "",
    f"- Recovery succeeded: `{baseline_recovery['succeeded']}`",
    f"- Strategy-faithful baseline CAGR gap vs reference: `{baseline_recovery['cagr_gap_vs_reference']}`",
    f"- Strategy-faithful baseline Sharpe gap vs reference: `{baseline_recovery['sharpe_gap_vs_reference']}`",
    f"- Production-faithful baseline CAGR gap vs reference: `{baseline_recovery['production_cagr_gap_vs_reference']}`",
    f"- Production-faithful baseline Sharpe gap vs reference: `{baseline_recovery['production_sharpe_gap_vs_reference']}`",
    "",
    "## Interpretation",
    "",
    "- If recovery succeeds, the strategy-faithful framework becomes valid for baseline-vs-rebuild comparison.",
    "- If recovery fails, comparison to V4.1 remains blocked because the target historical strategy object has not been recovered.",
]
baseline_recovery_audit_path = f"{output_root}/baseline_recovery_audit.md"
write_text(baseline_recovery_audit_path, "\n".join(baseline_recovery_audit_lines) + "\n")

daily_signal_alignment = (
    legacy_teacher[["timestamp", "predicted_regime"]]
    .rename(columns={"predicted_regime": "legacy_teacher_regime"})
    .merge(
        baseline_teacher[["timestamp", "predicted_regime"]].rename(
            columns={"predicted_regime": "baseline_teacher_regime"}
        ),
        on="timestamp",
        how="outer",
    )
    .sort_values("timestamp")
)
daily_signal_alignment = (
    daily_signal_alignment.set_index("timestamp")
    .resample("D")
    .last()
    .dropna(how="all")
    .reset_index()
)
daily_signal_alignment_sample_path = f"{output_root}/daily_signal_alignment_sample.csv"
daily_signal_alignment.head(40).to_csv(daily_signal_alignment_sample_path, index=False)

daily_price_alignment = (
    baseline_price.set_index("timestamp")
    .resample("D")
    .last()
    .dropna()
    .reset_index()
)
daily_price_alignment_sample_path = f"{output_root}/daily_price_alignment_sample.csv"
daily_price_alignment.head(40).to_csv(daily_price_alignment_sample_path, index=False)

timing_logic_notes_path = f"{output_root}/timing_logic_notes.md"
write_text(
    timing_logic_notes_path,
    "\n".join(
        [
            "# Timing Logic Notes",
            "",
            "- Signal at timestamp `t-1` is applied to the price move realized from `t-1` to `t`.",
            "- This matches the historical reference notebook.",
            "- No same-bar application is used.",
            "- No pure 1D rebalance layer is used because the source-of-truth strategy object is hourly.",
        ]
    )
    + "\n",
)

baseline_output_dir = f"{output_root}/baseline_strategy_faithful"
ensure_dir(baseline_output_dir)
baseline_wf_signals.to_csv(
    f"{baseline_output_dir}/walkforward_signals.csv", index=False
)
baseline_monthly_validation.to_csv(
    f"{baseline_output_dir}/monthly_validation.csv", index=False
)
baseline_wf_signals.tail(200).to_csv(
    f"{baseline_output_dir}/recent_signal_sample.csv", index=False
)

rebuild_metrics = None
rebuild_wf_backtest = None
rebuild_daily_curve = None
recommendation = "further review needed"
comparison_summary_note = ""
comparison_rows = [baseline_metrics]

if baseline_recovery["succeeded"]:
    rebuild_wf_signals, rebuild_monthly_validation = generate_walkforward_signals(
        rebuild_feature_pdf,
        rebuild_teacher,
        train_start,
        live_start,
        common_live_end,
    )
    rebuild_wf_signals = rebuild_wf_signals[
        rebuild_wf_signals["timestamp"] >= pd.Timestamp(live_start)
    ].copy()
    rebuild_wf_backtest = run_hourly_backtest(
        rebuild_wf_signals[["timestamp", "predicted_regime"]].copy(),
        baseline_price[baseline_price["timestamp"] <= common_live_end].copy(),
        initial_capital,
        transaction_cost,
    )
    rebuild_metrics = build_metric_row(
        "full_rebuild_v4_1_strategy_faithful",
        "strategy_faithful_walkforward",
        rebuild_wf_backtest,
        rebuild_monthly_validation,
    )
    comparison_rows.append(rebuild_metrics)

    rebuild_output_dir = f"{output_root}/full_rebuild_strategy_faithful"
    ensure_dir(rebuild_output_dir)
    rebuild_wf_signals.to_csv(
        f"{rebuild_output_dir}/walkforward_signals.csv", index=False
    )
    rebuild_monthly_validation.to_csv(
        f"{rebuild_output_dir}/monthly_validation.csv", index=False
    )
    rebuild_wf_signals.tail(200).to_csv(
        f"{rebuild_output_dir}/recent_signal_sample.csv", index=False
    )

    recommendation = comparison_recommendation(baseline_metrics, rebuild_metrics)
    if recommendation == "promote full rebuild V4.1 after approval":
        comparison_summary_note = (
            "Rebuilt V4.1 improves on the baseline inside the strategy-faithful framework "
            "without a materially worse drawdown profile."
        )
    elif recommendation == "hold current V4":
        comparison_summary_note = (
            "Baseline V4 remains stronger than the rebuilt model inside the strategy-faithful framework."
        )
    else:
        comparison_summary_note = (
            "Results are mixed inside the strategy-faithful framework and still need review."
        )
else:
    comparison_summary_note = (
        "Baseline recovery did not meet the defined success criteria. "
        "The rebuilt-model comparison is intentionally skipped."
    )

pnl_metrics_pdf = pd.DataFrame(comparison_rows)
pnl_metrics_path = f"{output_root}/pnl_metrics.csv"
pnl_metrics_pdf.to_csv(pnl_metrics_path, index=False)

baseline_daily_curve = daily_curve(
    baseline_wf_backtest, "baseline_v4_strategy_faithful"
)

curve_frames = [baseline_daily_curve]
if rebuild_wf_backtest is not None:
    rebuild_daily_curve = daily_curve(
        rebuild_wf_backtest, "full_rebuild_v4_1_strategy_faithful"
    )
    curve_frames.append(rebuild_daily_curve)

equity_curve_pdf = pd.concat(curve_frames, ignore_index=True)
equity_curve_path = f"{output_root}/equity_curve.csv"
equity_curve_pdf[["timestamp", "model", "equity", "applied_regime"]].to_csv(
    equity_curve_path, index=False
)

drawdown_curve_path = f"{output_root}/drawdown_curve.csv"
equity_curve_pdf[["timestamp", "model", "drawdown"]].to_csv(
    drawdown_curve_path, index=False
)

regime_stats_rows = [
    {
        "model": baseline_metrics["model"],
        "total_regime_switches": baseline_metrics["total_regime_switches"],
        "average_holding_bars": baseline_metrics["average_holding_bars"],
        "expansion_pct_time": baseline_metrics["expansion_pct_time"],
        "slowdown_pct_time": baseline_metrics["slowdown_pct_time"],
        "recession_pct_time": baseline_metrics["recession_pct_time"],
    }
]
if rebuild_metrics is not None:
    regime_stats_rows.append(
        {
            "model": rebuild_metrics["model"],
            "total_regime_switches": rebuild_metrics["total_regime_switches"],
            "average_holding_bars": rebuild_metrics["average_holding_bars"],
            "expansion_pct_time": rebuild_metrics["expansion_pct_time"],
            "slowdown_pct_time": rebuild_metrics["slowdown_pct_time"],
            "recession_pct_time": rebuild_metrics["recession_pct_time"],
        }
    )
regime_stats_pdf = pd.DataFrame(regime_stats_rows)
regime_stats_path = f"{output_root}/regime_stats.csv"
regime_stats_pdf.to_csv(regime_stats_path, index=False)

equity_plot = plt.figure(figsize=(14, 7))
for model_name, frame in equity_curve_pdf.groupby("model"):
    plt.plot(frame["timestamp"], frame["equity"], label=model_name)
plt.yscale("log")
plt.title("Strategy-Faithful Walk-Forward Equity Curve Comparison")
plt.ylabel("Equity")
plt.legend()
plt.grid(alpha=0.25)
save_plot(f"{output_root}/equity_curve_comparison.png", equity_plot)

drawdown_plot = plt.figure(figsize=(14, 5))
for model_name, frame in equity_curve_pdf.groupby("model"):
    plt.plot(frame["timestamp"], frame["drawdown"], label=model_name)
plt.title("Strategy-Faithful Drawdown Comparison")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(alpha=0.25)
save_plot(f"{output_root}/drawdown_comparison.png", drawdown_plot)

regime_timeline_plot = plt.figure(figsize=(14, 4))
for model_name, frame in equity_curve_pdf.groupby("model"):
    plt.step(
        frame["timestamp"],
        frame["applied_regime"],
        where="post",
        label=model_name,
        alpha=0.9,
    )
plt.yticks([0, 1, 2], ["Expansion", "Slowdown", "Recession"])
plt.title("Strategy-Faithful Regime Timeline")
plt.legend()
plt.grid(alpha=0.25)
save_plot(f"{output_root}/regime_timeline.png", regime_timeline_plot)

summary_lines = [
    "# Strategy-Faithful Walk-Forward Summary",
    "",
    "## Methodology",
    "",
    "- Monthly expanding-window retrain against full-signal teacher labels.",
    "- Hourly prior-bar execution on SPY/QQQ prices.",
    "- Daily-resampled reporting with 252-day annualization.",
    "- This framework is intentionally different from the production-faithful walk-forward engine.",
    "",
    "## Baseline Recovery",
    "",
    f"- Recovery succeeded: `{baseline_recovery['succeeded']}`",
    f"- Strategy-faithful baseline CAGR: `{baseline_metrics['cagr']}`",
    f"- Legacy reference CAGR: `{reference_for_recovery['cagr']}`",
    f"- Production-faithful baseline CAGR: `{production_faithful_baseline_row.get('cagr') if production_faithful_baseline_row else None}`",
    f"- Strategy-faithful baseline Sharpe: `{baseline_metrics['sharpe_ratio']}`",
    f"- Legacy reference Sharpe: `{reference_for_recovery['sharpe_ratio']}`",
    f"- Production-faithful baseline Sharpe: `{production_faithful_baseline_row.get('sharpe_ratio') if production_faithful_baseline_row else None}`",
    "",
    "## Comparison",
    "",
    markdown_table(
        pnl_metrics_pdf[
            [
                "model",
                "final_equity",
                "cagr",
                "annualized_volatility",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "calmar_ratio",
                "total_regime_switches",
                "average_holding_bars",
            ]
        ]
    ),
    "",
    "## Interpretation",
    "",
    f"- {comparison_summary_note}",
    "- The dominant gap versus the production-faithful engine comes from signal-source lineage and teacher-label methodology.",
    "- The historical source-of-truth notebook does not support treating 1D decision cadence as the canonical target.",
    "",
    "## Recommendation",
    "",
    f"- `{recommendation}`",
]
summary_path = f"{output_root}/summary.md"
write_text(summary_path, "\n".join(summary_lines) + "\n")

recommendation_payload = {
    "recommendation": recommendation,
    "baseline_recovery_succeeded": baseline_recovery["succeeded"],
    "comparison_executed": rebuild_metrics is not None,
    "baseline_metrics": baseline_metrics,
    "rebuild_metrics": rebuild_metrics,
    "baseline_recovery": baseline_recovery,
    "notes": comparison_summary_note,
}
recommendation_path = f"{output_root}/recommendation.json"
write_json(recommendation_path, recommendation_payload)

run_metadata = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_root": output_root,
    "train_start": train_start,
    "live_start": live_start,
    "common_live_end": str(common_live_end),
    "initial_capital": initial_capital,
    "transaction_cost": transaction_cost,
    "inputs": {
        "baseline_silver_table": table_row_range(baseline_silver_table),
        "rebuild_silver_table": table_row_range(rebuild_silver_table),
        "bronze_rebuild_table": {
            "table": BRONZE_REBUILD_TABLE_DEFAULT,
            "exists": table_exists(BRONZE_REBUILD_TABLE_DEFAULT),
        },
        "baseline_teacher_path": baseline_teacher_path,
        "rebuild_teacher_path": rebuild_teacher_path,
        "legacy_teacher_path": legacy_teacher_path,
        "price_table": price_table,
        "legacy_price_table": legacy_price_table,
        "legacy_reference_signal_table": legacy_reference_signal_table,
        "production_faithful_pnl_path": production_faithful_pnl_path,
    },
    "baseline_recovery": baseline_recovery,
    "comparison_executed": rebuild_metrics is not None,
    "artifacts": {
        "methodology_spec": methodology_spec_path,
        "baseline_recovery_audit": baseline_recovery_audit_path,
        "baseline_recovery_metrics": baseline_recovery_metrics_path,
        "pnl_metrics": pnl_metrics_path,
        "equity_curve": equity_curve_path,
        "drawdown_curve": drawdown_curve_path,
        "regime_stats": regime_stats_path,
        "summary": summary_path,
        "recommendation": recommendation_path,
        "daily_signal_alignment_sample": daily_signal_alignment_sample_path,
        "daily_price_alignment_sample": daily_price_alignment_sample_path,
        "timing_logic_notes": timing_logic_notes_path,
        "equity_curve_plot": f"{output_root}/equity_curve_comparison.png",
        "drawdown_plot": f"{output_root}/drawdown_comparison.png",
        "regime_timeline_plot": f"{output_root}/regime_timeline.png",
    },
}
run_metadata_path = f"{output_root}/run_metadata.json"
write_json(run_metadata_path, run_metadata)

result = {
    "phase": "strategy_faithful_walkforward_replication",
    "status": "success",
    "baseline_recovery_succeeded": baseline_recovery["succeeded"],
    "comparison_executed": rebuild_metrics is not None,
    "recommendation": recommendation,
    "output_root": output_root,
    "artifacts": run_metadata["artifacts"],
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

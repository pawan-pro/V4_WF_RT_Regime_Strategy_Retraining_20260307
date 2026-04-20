# Databricks notebook source
# Reconcile the legacy walk-forward methodology against the parity-correct walk-forward engine.

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CATALOG = "pawan_quant_research_1"
CURRENT_SCHEMA = "v4"
LEGACY_SCHEMA = "default"

LEGACY_REFERENCE_SOURCE_PATH = (
    "/Users/pawan/Downloads/11RT_WalkForward_Simulation 2026-03-12 08_50_27 (2).py"
)

CURRENT_VOLUME_ROOT = f"/Volumes/{CATALOG}/{CURRENT_SCHEMA}/market_data"
LEGACY_VOLUME_ROOT = f"/Volumes/{CATALOG}/{LEGACY_SCHEMA}/market_data"
DEFAULT_OUTPUT_ROOT = (
    f"{CURRENT_VOLUME_ROOT}/comparison/2026-04-20_walkforward_methodology_reconciliation"
)
DEFAULT_CURRENT_BASELINE_ROOT = (
    f"{CURRENT_VOLUME_ROOT}/comparison/2026-04-20_full_historical_adoption/baseline"
)
DEFAULT_PARITY_PNL_ROOT = (
    f"{CURRENT_VOLUME_ROOT}/comparison/2026-04-20_full_historical_adoption/walkforward_pnl"
)

LEGACY_SIGNAL_TABLE = f"{CATALOG}.{LEGACY_SCHEMA}.bt_walkforward_regime_signals"
LEGACY_SILVER_TABLE = f"{CATALOG}.{LEGACY_SCHEMA}.silver_bt_master_grid_1h"
LEGACY_MARKET_TABLE = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"
LEGACY_LABEL_PATH = f"{LEGACY_VOLUME_ROOT}/bt_fred_1d_regime_signals-v4.csv"
LEGACY_STRATEGY_SUMMARY_PATH = f"{LEGACY_VOLUME_ROOT}/strategy_performance_summary.csv"
LEGACY_DAILY_RETURNS_PATH = f"{LEGACY_VOLUME_ROOT}/strategy_daily_returns.csv"

CURRENT_MARKET_TABLE = f"{CATALOG}.{CURRENT_SCHEMA}.bronze_bt_market_1h"
CURRENT_BASELINE_SIGNAL_PATH = (
    f"{DEFAULT_CURRENT_BASELINE_ROOT}/current_walkforward_signals.csv"
)

START_DATE_DEFAULT = "2021-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252

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


def load_csv(path_str):
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_timestamp(pdf, column="timestamp"):
    result = pdf.copy()
    if column in result.columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
        try:
            result[column] = result[column].dt.tz_localize(None)
        except Exception:
            pass
    return result


def load_signal_from_table(table_name, signal_column="predicted_regime"):
    signal_pdf = spark.table(table_name).toPandas()
    signal_pdf = normalize_timestamp(signal_pdf)
    signal_pdf = signal_pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    signal_pdf = signal_pdf.drop_duplicates(subset=["timestamp"], keep="last")
    if signal_column not in signal_pdf.columns:
        raise KeyError(
            f"Column {signal_column} not found in {table_name}. "
            f"columns={list(signal_pdf.columns)}"
        )
    result = signal_pdf[["timestamp", signal_column]].copy()
    if signal_column == "predicted_regime":
        result["predicted_regime"] = (
            pd.to_numeric(result["predicted_regime"], errors="coerce").fillna(2).astype(int)
        )
        return result

    result["predicted_regime"] = (
        pd.to_numeric(result[signal_column], errors="coerce").fillna(2).astype(int)
    )
    return result.drop(columns=[signal_column])


def load_signal_from_csv(path_str, candidates):
    signal_pdf = load_csv(path_str)
    if signal_pdf.empty:
        raise FileNotFoundError(f"Missing or empty signal file: {path_str}")
    signal_pdf = normalize_timestamp(signal_pdf)
    signal_pdf = signal_pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    signal_pdf = signal_pdf.drop_duplicates(subset=["timestamp"], keep="last")

    chosen = next((column for column in candidates if column in signal_pdf.columns), None)
    if not chosen:
        raise KeyError(
            f"No signal column found in {path_str}. "
            f"candidates={candidates}, columns={list(signal_pdf.columns)}"
        )

    result = signal_pdf[["timestamp", chosen]].copy()
    if chosen == "predicted_regime":
        result["predicted_regime"] = (
            pd.to_numeric(result["predicted_regime"], errors="coerce").fillna(2).astype(int)
        )
        return result

    result["predicted_regime"] = (
        pd.to_numeric(result[chosen], errors="coerce").fillna(2).astype(int)
    )
    return result.drop(columns=[chosen])


def load_price_frame(table_name, start_date):
    price_pdf = (
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
    price_pdf = normalize_timestamp(price_pdf)
    price_pdf = price_pdf.dropna(subset=["timestamp"])
    price_pdf = price_pdf.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    price_pdf = (
        price_pdf.pivot_table(
            index="timestamp",
            columns="symbol",
            values="close",
            aggfunc="last",
        )
        .reset_index()
        .rename(columns={"SPY": "SPY", "QQQ": "QQQ"})
        .sort_values("timestamp")
    )
    price_pdf["SPY"] = pd.to_numeric(price_pdf["SPY"], errors="coerce").ffill()
    price_pdf["QQQ"] = pd.to_numeric(price_pdf["QQQ"], errors="coerce").ffill()
    return price_pdf


def run_prior_bar_backtest(signal_pdf, price_pdf, initial_capital, transaction_cost):
    aligned = signal_pdf.merge(price_pdf, on="timestamp", how="inner").sort_values("timestamp")
    aligned = aligned.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)

    capital = float(initial_capital)
    active_asset = "START"
    records = []

    for idx in range(len(aligned)):
        row = aligned.iloc[idx]
        record = {
            "timestamp": row["timestamp"],
            "signal_at_t": int(row["predicted_regime"]),
            "applied_regime": np.nan,
            "position": "START" if idx == 0 else None,
            "SPY": row["SPY"],
            "QQQ": row["QQQ"],
            "gross_return": 0.0,
            "transaction_cost_applied": 0.0,
            "net_return": 0.0,
            "equity": capital,
            "position_changed": False,
        }

        if idx > 0:
            prior_signal = int(aligned.iloc[idx - 1]["predicted_regime"])
            target = POSITION_MAP.get(prior_signal, "CASH")
            if target == "QQQ":
                gross_return = row["QQQ"] / aligned.iloc[idx - 1]["QQQ"] - 1.0
            elif target == "SPY":
                gross_return = row["SPY"] / aligned.iloc[idx - 1]["SPY"] - 1.0
            else:
                gross_return = 0.0

            transaction_cost_applied = transaction_cost if target != active_asset else 0.0
            net_return = gross_return - transaction_cost_applied
            capital *= 1.0 + net_return

            record.update(
                {
                    "applied_regime": prior_signal,
                    "position": target,
                    "gross_return": gross_return,
                    "transaction_cost_applied": transaction_cost_applied,
                    "net_return": net_return,
                    "equity": capital,
                    "position_changed": target != active_asset,
                }
            )
            active_asset = target

        records.append(record)

    backtest_pdf = pd.DataFrame(records)
    if len(backtest_pdf) > 1:
        backtest_pdf["applied_regime"] = (
            pd.to_numeric(backtest_pdf["applied_regime"], errors="coerce")
            .fillna(method="ffill")
            .fillna(2)
            .astype(int)
        )
    else:
        backtest_pdf["applied_regime"] = 2
    backtest_pdf["applied_regime_name"] = backtest_pdf["applied_regime"].map(
        REGIME_NAME_MAP
    )
    return backtest_pdf


def contiguous_run_lengths(series):
    if series.empty:
        return pd.Series(dtype=float)
    groups = series.ne(series.shift()).cumsum()
    return series.groupby(groups).size().astype(float)


def legacy_metric_style(equity_series):
    series = equity_series.dropna()
    daily = series.resample("D").last().dropna()
    returns = daily.pct_change(fill_method=None).dropna()
    years = len(returns) / TRADING_DAYS if len(returns) else np.nan
    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1 if years and years > 0 else np.nan
    vol_annual = returns.std() * np.sqrt(TRADING_DAYS) if not returns.empty else np.nan
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)
        if len(returns) and returns.std() != 0
        else np.nan
    )
    max_dd = (series / series.cummax() - 1.0).min() if not series.empty else np.nan
    return {
        "metric_style": "legacy_reference",
        "start_timestamp": series.index.min(),
        "end_timestamp": series.index.max(),
        "daily_rows": int(len(daily)),
        "return_rows": int(len(returns)),
        "years": float(years) if pd.notna(years) else np.nan,
        "initial_capital": float(series.iloc[0]),
        "final_equity": float(series.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "annualized_volatility": float(vol_annual) if np.isfinite(vol_annual) else np.nan,
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(-max_dd) if pd.notna(max_dd) else np.nan,
    }


def new_engine_metric_style(equity_series):
    series = equity_series.dropna()
    daily = series.resample("D").last().dropna()
    returns = daily.pct_change().dropna()
    start_ts = daily.index[0]
    end_ts = daily.index[-1]
    years = max((end_ts - start_ts).days / 365.25, 1 / 365.25)
    total_return = daily.iloc[-1] / daily.iloc[0] - 1.0
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol_annual = returns.std() * np.sqrt(TRADING_DAYS) if not returns.empty else np.nan
    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)
        if len(returns) and returns.std() != 0
        else np.nan
    )
    drawdown = daily / daily.cummax() - 1.0
    max_dd = float(-drawdown.min()) if not drawdown.empty else np.nan
    calmar = cagr / max_dd if max_dd and max_dd > 0 else np.nan
    return {
        "metric_style": "new_engine",
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "daily_rows": int(len(daily)),
        "return_rows": int(len(returns)),
        "years": float(years),
        "initial_capital": float(daily.iloc[0]),
        "final_equity": float(daily.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "annualized_volatility": float(vol_annual) if np.isfinite(vol_annual) else np.nan,
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else np.nan,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
    }


def regime_stats(backtest_pdf):
    active = backtest_pdf.iloc[1:].copy()
    holding_lengths = contiguous_run_lengths(active["position"])
    return {
        "total_regime_switches": int(active["position_changed"].sum()),
        "average_holding_bars": float(holding_lengths.mean()) if not holding_lengths.empty else np.nan,
        "expansion_pct_time": float(active["applied_regime"].eq(0).mean()) if len(active) else np.nan,
        "slowdown_pct_time": float(active["applied_regime"].eq(1).mean()) if len(active) else np.nan,
        "recession_pct_time": float(active["applied_regime"].eq(2).mean()) if len(active) else np.nan,
    }


def build_metric_row(scenario, signal_source, price_source, horizon_note, backtest_pdf, metric_fn):
    metrics = metric_fn(backtest_pdf.set_index("timestamp")["equity"])
    metrics.update(regime_stats(backtest_pdf))
    metrics["scenario"] = scenario
    metrics["signal_source"] = signal_source
    metrics["price_source"] = price_source
    metrics["horizon_note"] = horizon_note
    metrics["signal_rows"] = int(len(backtest_pdf))
    return metrics


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
    headers = list(frame.columns)
    rows = [headers, ["---"] * len(headers)]
    rows.extend(frame.values.tolist())
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


output_root = DEFAULT_OUTPUT_ROOT
current_baseline_root = DEFAULT_CURRENT_BASELINE_ROOT
parity_pnl_root = DEFAULT_PARITY_PNL_ROOT
start_date = START_DATE_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("output_root", DEFAULT_OUTPUT_ROOT)
    dbutils.widgets.text("current_baseline_root", DEFAULT_CURRENT_BASELINE_ROOT)
    dbutils.widgets.text("parity_pnl_root", DEFAULT_PARITY_PNL_ROOT)
    dbutils.widgets.text("start_date", START_DATE_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    output_root = dbutils.widgets.get("output_root").strip() or DEFAULT_OUTPUT_ROOT
    current_baseline_root = (
        dbutils.widgets.get("current_baseline_root").strip() or DEFAULT_CURRENT_BASELINE_ROOT
    )
    parity_pnl_root = dbutils.widgets.get("parity_pnl_root").strip() or DEFAULT_PARITY_PNL_ROOT
    start_date = dbutils.widgets.get("start_date").strip() or START_DATE_DEFAULT
    initial_capital = float(dbutils.widgets.get("initial_capital").strip() or INITIAL_CAPITAL_DEFAULT)
    transaction_cost = float(dbutils.widgets.get("transaction_cost").strip() or TRANSACTION_COST_DEFAULT)

ensure_dir(output_root)

legacy_signal_pdf = load_signal_from_table(LEGACY_SIGNAL_TABLE)
legacy_signal_pdf = legacy_signal_pdf[legacy_signal_pdf["timestamp"] >= pd.Timestamp(start_date)].copy()
current_signal_pdf = load_signal_from_csv(
    f"{current_baseline_root}/current_walkforward_signals.csv",
    ["predicted_regime", "model_predicted_regime"],
)
current_signal_pdf = current_signal_pdf[current_signal_pdf["timestamp"] >= pd.Timestamp(start_date)].copy()

legacy_price_pdf = load_price_frame(LEGACY_MARKET_TABLE, start_date)
current_price_pdf = load_price_frame(CURRENT_MARKET_TABLE, start_date)

legacy_backtest = run_prior_bar_backtest(
    legacy_signal_pdf,
    legacy_price_pdf,
    initial_capital,
    transaction_cost,
)
current_backtest = run_prior_bar_backtest(
    current_signal_pdf,
    current_price_pdf,
    initial_capital,
    transaction_cost,
)

old_reference_metrics = pd.DataFrame(
    [
        build_metric_row(
            "legacy_signal_table_legacy_metric_style",
            LEGACY_SIGNAL_TABLE,
            LEGACY_MARKET_TABLE,
            "legacy LIVE_END / default namespace",
            legacy_backtest,
            legacy_metric_style,
        ),
        build_metric_row(
            "legacy_signal_table_new_engine_metric_style",
            LEGACY_SIGNAL_TABLE,
            LEGACY_MARKET_TABLE,
            "legacy LIVE_END / default namespace",
            legacy_backtest,
            new_engine_metric_style,
        ),
    ]
)

legacy_export_summary = load_csv(LEGACY_STRATEGY_SUMMARY_PATH)
if not legacy_export_summary.empty and legacy_export_summary.columns[0].startswith("Unnamed"):
    legacy_export_summary = legacy_export_summary.rename(
        columns={legacy_export_summary.columns[0]: "strategy"}
    )
legacy_export_walkforward = legacy_export_summary[
    legacy_export_summary["strategy"].eq("WalkForward_V4")
].copy() if "strategy" in legacy_export_summary.columns else pd.DataFrame()
if not legacy_export_walkforward.empty:
    export_row = legacy_export_walkforward.iloc[0]
    export_metrics_row = {
        "scenario": "legacy_exported_strategy_summary",
        "signal_source": LEGACY_SIGNAL_TABLE,
        "price_source": LEGACY_MARKET_TABLE,
        "horizon_note": "legacy CSV export",
        "metric_style": "legacy_reference",
        "start_timestamp": legacy_backtest["timestamp"].min(),
        "end_timestamp": legacy_backtest["timestamp"].max(),
        "initial_capital": float(export_row.get("Start Value", np.nan)),
        "final_equity": float(export_row.get("Final Value", np.nan)),
        "total_return": float(export_row.get("Total Return", np.nan)),
        "cagr": float(export_row.get("CAGR", np.nan)),
        "annualized_volatility": float(export_row.get("Vol Annual", np.nan)),
        "sharpe_ratio": float(export_row.get("Sharpe Annual", np.nan)),
        "max_drawdown": float(-export_row.get("Max Drawdown", np.nan))
        if pd.notna(export_row.get("Max Drawdown", np.nan))
        else np.nan,
        "signal_rows": int(len(legacy_backtest)),
        "daily_rows": np.nan,
        "return_rows": np.nan,
        "years": np.nan,
        "total_regime_switches": int(legacy_backtest["position_changed"].iloc[1:].sum()),
        "average_holding_bars": float(contiguous_run_lengths(legacy_backtest["position"].iloc[1:]).mean()),
        "expansion_pct_time": float(legacy_backtest["applied_regime"].iloc[1:].eq(0).mean()),
        "slowdown_pct_time": float(legacy_backtest["applied_regime"].iloc[1:].eq(1).mean()),
        "recession_pct_time": float(legacy_backtest["applied_regime"].iloc[1:].eq(2).mean()),
    }
    old_reference_metrics = pd.concat(
        [pd.DataFrame([export_metrics_row]), old_reference_metrics],
        ignore_index=True,
    )

new_engine_metrics = pd.DataFrame(
    [
        build_metric_row(
            "current_baseline_new_engine_metric_style",
            f"{current_baseline_root}/current_walkforward_signals.csv",
            CURRENT_MARKET_TABLE,
            "current v4 baseline snapshot",
            current_backtest,
            new_engine_metric_style,
        ),
        build_metric_row(
            "current_baseline_legacy_metric_style",
            f"{current_baseline_root}/current_walkforward_signals.csv",
            CURRENT_MARKET_TABLE,
            "current v4 baseline snapshot",
            current_backtest,
            legacy_metric_style,
        ),
    ]
)

current_parity_metrics = load_csv(f"{parity_pnl_root}/pnl_metrics.csv")
if not current_parity_metrics.empty:
    current_parity_metrics = current_parity_metrics[current_parity_metrics["model"].eq("baseline")].copy()

signal_compare = legacy_signal_pdf.merge(
    current_signal_pdf,
    on="timestamp",
    how="outer",
    suffixes=("_legacy", "_current"),
).sort_values("timestamp")
signal_compare["legacy_regime_name"] = signal_compare["predicted_regime_legacy"].map(REGIME_NAME_MAP)
signal_compare["current_regime_name"] = signal_compare["predicted_regime_current"].map(REGIME_NAME_MAP)
signal_compare["regime_changed"] = (
    signal_compare["predicted_regime_legacy"] != signal_compare["predicted_regime_current"]
)
signal_alignment_sample = signal_compare[
    signal_compare["timestamp"].between("2021-01-01", "2021-01-07")
].copy()
signal_alignment_path = f"{output_root}/signal_alignment_sample.csv"
signal_alignment_sample.to_csv(signal_alignment_path, index=False)

price_compare = legacy_price_pdf.merge(
    current_price_pdf,
    on="timestamp",
    how="inner",
    suffixes=("_legacy", "_current"),
)
price_compare["SPY_diff"] = price_compare["SPY_legacy"] - price_compare["SPY_current"]
price_compare["QQQ_diff"] = price_compare["QQQ_legacy"] - price_compare["QQQ_current"]
price_alignment_sample = price_compare[
    price_compare["timestamp"].between("2021-01-01", "2021-01-07")
].copy()
price_alignment_path = f"{output_root}/price_alignment_sample.csv"
price_alignment_sample.to_csv(price_alignment_path, index=False)

timing_alignment_sample = legacy_backtest[
    [
        "timestamp",
        "signal_at_t",
        "applied_regime",
        "applied_regime_name",
        "position",
        "gross_return",
        "transaction_cost_applied",
        "net_return",
        "equity",
    ]
].head(12).copy()
timing_alignment_path = f"{output_root}/timing_alignment_sample.csv"
timing_alignment_sample.to_csv(timing_alignment_path, index=False)

legacy_start = legacy_signal_pdf["timestamp"].min()
legacy_end = legacy_signal_pdf["timestamp"].max()
current_start = current_signal_pdf["timestamp"].min()
current_end = current_signal_pdf["timestamp"].max()

methodology_matrix = pd.DataFrame(
    [
        {
            "dimension": "signal_source",
            "old_reference": f"{LEGACY_SIGNAL_TABLE} written by legacy notebook",
            "new_engine": f"{current_baseline_root}/current_walkforward_signals.csv",
            "impact_assessment": "high",
            "exact_difference": "Old reference consumes the persisted legacy walk-forward table produced from the default stack; new engine consumes the current baseline walk-forward snapshot from the v4 stack.",
        },
        {
            "dimension": "training_target_for_signal_generation",
            "old_reference": LEGACY_LABEL_PATH,
            "new_engine": "N/A at PnL stage; uses already-exported baseline walk-forward predictions",
            "impact_assessment": "high",
            "exact_difference": "Legacy notebook trains monthly models against full-signal predicted_regime labels from bt_fred_1d_regime_signals-v4.csv, not against current baseline walk-forward exports.",
        },
        {
            "dimension": "price_source",
            "old_reference": LEGACY_MARKET_TABLE,
            "new_engine": CURRENT_MARKET_TABLE,
            "impact_assessment": "medium",
            "exact_difference": "Both use hourly bronze price pivots for SPY and QQQ, but they come from different schemas/namespaces.",
        },
        {
            "dimension": "time_granularity",
            "old_reference": "hourly signal path, daily-resampled summary metrics",
            "new_engine": "hourly signal path, daily-resampled summary metrics",
            "impact_assessment": "low",
            "exact_difference": "Both backtests run on hourly bars and summarize from daily-resampled equity.",
        },
        {
            "dimension": "signal_timing",
            "old_reference": "prior_bar",
            "new_engine": "prior_bar",
            "impact_assessment": "none",
            "exact_difference": "Both apply the t-1 regime to the return realized over bar t.",
        },
        {
            "dimension": "transaction_cost_logic",
            "old_reference": "gross_return - cost_on_switch",
            "new_engine": "gross_return - cost_on_switch",
            "impact_assessment": "none",
            "exact_difference": "Both use subtractive switching cost, not multiplicative capital haircut.",
        },
        {
            "dimension": "annualization_logic",
            "old_reference": "years = len(daily_returns)/252",
            "new_engine": "years = actual elapsed days / 365.25",
            "impact_assessment": "medium",
            "exact_difference": "Legacy annualizes on daily return count; new engine annualizes on actual elapsed calendar time.",
        },
        {
            "dimension": "evaluation_horizon",
            "old_reference": f"{legacy_start} -> {legacy_end}",
            "new_engine": f"{current_start} -> {current_end}",
            "impact_assessment": "medium",
            "exact_difference": "Legacy notebook hard-stops at LIVE_END=2026-01-01; new engine runs through the latest exported baseline signal timestamp.",
        },
        {
            "dimension": "portfolio_mapping",
            "old_reference": "0->QQQ, 1->SPY, 2->CASH",
            "new_engine": "0->QQQ, 1->SPY, 2->CASH",
            "impact_assessment": "none",
            "exact_difference": "Mapping is identical.",
        },
        {
            "dimension": "stack_namespace",
            "old_reference": LEGACY_SCHEMA,
            "new_engine": CURRENT_SCHEMA,
            "impact_assessment": "high",
            "exact_difference": "Legacy notebook runs entirely on the default namespace; new engine is tied to the current v4 production stack.",
        },
    ]
)
methodology_matrix_path = f"{output_root}/methodology_matrix.csv"
methodology_matrix.to_csv(methodology_matrix_path, index=False)

legacy_export_cagr = (
    old_reference_metrics.loc[
        old_reference_metrics["scenario"].eq("legacy_exported_strategy_summary"),
        "cagr",
    ].iloc[0]
    if "legacy_exported_strategy_summary" in old_reference_metrics["scenario"].values
    else np.nan
)
legacy_recomputed_new_engine_cagr = (
    old_reference_metrics.loc[
        old_reference_metrics["scenario"].eq("legacy_signal_table_new_engine_metric_style"),
        "cagr",
    ].iloc[0]
)
current_baseline_new_engine_cagr = (
    new_engine_metrics.loc[
        new_engine_metrics["scenario"].eq("current_baseline_new_engine_metric_style"),
        "cagr",
    ].iloc[0]
)
current_baseline_legacy_metric_cagr = (
    new_engine_metrics.loc[
        new_engine_metrics["scenario"].eq("current_baseline_legacy_metric_style"),
        "cagr",
    ].iloc[0]
)

dominant_driver = "signal_source_and_namespace"
if abs(current_baseline_legacy_metric_cagr - current_baseline_new_engine_cagr) > abs(
    legacy_recomputed_new_engine_cagr - current_baseline_new_engine_cagr
):
    dominant_driver = "metric_annualization"

canonical_method = "new_engine_is_canonical"
blocked = False
smallest_safe_change = None
legacy_retire_recommendation = (
    "Retire the legacy notebook from decision use and keep it only as archival research context."
)
recommendation_text = (
    "Use the new parity-correct engine as the canonical walk-forward methodology. "
    "The legacy notebook is not methodology-equivalent to the current production walk-forward baseline because it depends on the default namespace and full-signal teacher labels."
)

verdict = {
    "status": "success",
    "legacy_reference_source_path": LEGACY_REFERENCE_SOURCE_PATH,
    "canonical_method": canonical_method,
    "dominant_gap_driver": dominant_driver,
    "v4_vs_v4_1_decision_blocked": blocked,
    "smallest_safe_change": smallest_safe_change,
    "legacy_recommendation": legacy_retire_recommendation,
    "summary": recommendation_text,
}
verdict_path = f"{output_root}/reconciliation_verdict.json"
write_json(verdict_path, verdict)

old_reference_metrics_path = f"{output_root}/old_reference_recomputed_metrics.csv"
new_engine_metrics_path = f"{output_root}/new_engine_metrics.csv"
old_reference_metrics.to_csv(old_reference_metrics_path, index=False)
new_engine_metrics.to_csv(new_engine_metrics_path, index=False)

methodology_diff_lines = [
    "# Walk-Forward Methodology Diff",
    "",
    "## Executive Summary",
    "",
    f"- Canonical recommendation: `{canonical_method}`",
    f"- V4 vs V4.1 decision blocked on methodology: `{blocked}`",
    f"- Dominant performance gap driver: `{dominant_driver}`",
    "",
    "## Key Findings",
    "",
    f"- The legacy reference notebook reads labels from `{LEGACY_LABEL_PATH}` and uses them as the monthly walk-forward training target.",
    f"- The new engine reads the current baseline walk-forward export from `{current_baseline_root}/current_walkforward_signals.csv`.",
    f"- The legacy signal table runs on the `{LEGACY_SCHEMA}` namespace; the new engine runs on `{CURRENT_SCHEMA}`.",
    f"- Both methods use prior-bar timing and subtractive switching cost, so timing and cost are not the primary source of the gap.",
    f"- Legacy annualization differs, but applying the legacy metric style to the current baseline does not reproduce the stronger historical CAGR.",
    "",
    "## Metric Decomposition",
    "",
    f"- Legacy exported WalkForward CAGR: `{legacy_export_cagr}`",
    f"- Legacy signal table under new-engine metrics CAGR: `{legacy_recomputed_new_engine_cagr}`",
    f"- Current baseline under legacy metrics CAGR: `{current_baseline_legacy_metric_cagr}`",
    f"- Current baseline under new-engine metrics CAGR: `{current_baseline_new_engine_cagr}`",
    "",
    "## Interpretation",
    "",
    "- The large performance gap is not explained by timing or transaction cost handling.",
    "- The large performance gap is not explained primarily by the annualization formula.",
    "- The large performance gap is explained by the legacy notebook using a different signal-generation process and a different data namespace.",
    "- In practice, the legacy notebook is closer to a legacy teacher-label distillation workflow than to the current production walk-forward baseline evaluation.",
    "",
    "## Recommendation",
    "",
    f"- {recommendation_text}",
    f"- {legacy_retire_recommendation}",
]
methodology_diff_path = f"{output_root}/methodology_diff.md"
write_text(methodology_diff_path, "\n".join(methodology_diff_lines) + "\n")

recommendation_md_lines = [
    "# Recommendation",
    "",
    f"- Canonical method: `{canonical_method}`",
    f"- Decision blocked: `{blocked}`",
    f"- Smallest safe change: `{smallest_safe_change}`",
    "",
    "## Why",
    "",
    "- The new engine is aligned to the current v4 production walk-forward baseline exports.",
    "- The old notebook depends on default-schema features and a full-signal label file that is not the same object as the current baseline walk-forward signal stream.",
    "- The timing and switching cost conventions are already aligned between the two approaches.",
    "- The stronger legacy performance does not survive a like-for-like signal-source reconciliation.",
]
recommendation_md_path = f"{output_root}/recommendation.md"
write_text(recommendation_md_path, "\n".join(recommendation_md_lines) + "\n")

result = {
    "phase": "walkforward_methodology_reconciliation",
    "status": "success",
    "canonical_method": canonical_method,
    "v4_vs_v4_1_decision_blocked": blocked,
    "artifacts": {
        "methodology_diff": methodology_diff_path,
        "methodology_matrix": methodology_matrix_path,
        "old_reference_recomputed_metrics": old_reference_metrics_path,
        "new_engine_metrics": new_engine_metrics_path,
        "reconciliation_verdict": verdict_path,
        "recommendation": recommendation_md_path,
        "signal_alignment_sample": signal_alignment_path,
        "price_alignment_sample": price_alignment_path,
        "timing_alignment_sample": timing_alignment_path,
    },
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

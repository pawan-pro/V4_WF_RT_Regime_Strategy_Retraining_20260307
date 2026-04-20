# Databricks notebook source

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
LEGACY_SCHEMA = "default"

COMPARISON_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/comparison"
DEFAULT_OUTPUT_ROOT = (
    f"{COMPARISON_ROOT}/2026-04-20_canonical_legacy_v4_performance"
)

SIGNAL_TABLE_DEFAULT = (
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals_legacy_recreated_baseline"
)
SIGNAL_CSV_PATH_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-20_legacy_recreated_signals_alfred_only/"
    "legacy_recreated_baseline/walkforward_signals.csv"
)
MARKET_TABLE_DEFAULT = f"{CATALOG}.{LEGACY_SCHEMA}.bronze_bt_market_1h"

LIVE_START_DEFAULT = "2021-01-01"
LIVE_END_DEFAULT = "2026-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252

REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}


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


output_root = DEFAULT_OUTPUT_ROOT
signal_table = SIGNAL_TABLE_DEFAULT
signal_csv_path = SIGNAL_CSV_PATH_DEFAULT
market_table = MARKET_TABLE_DEFAULT
live_start = LIVE_START_DEFAULT
live_end = LIVE_END_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("output_root", DEFAULT_OUTPUT_ROOT)
    dbutils.widgets.text("signal_table", SIGNAL_TABLE_DEFAULT)
    dbutils.widgets.text("signal_csv_path", SIGNAL_CSV_PATH_DEFAULT)
    dbutils.widgets.text("market_table", MARKET_TABLE_DEFAULT)
    dbutils.widgets.text("live_start", LIVE_START_DEFAULT)
    dbutils.widgets.text("live_end", LIVE_END_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    output_root = dbutils.widgets.get("output_root").strip() or DEFAULT_OUTPUT_ROOT
    signal_table = dbutils.widgets.get("signal_table").strip() or SIGNAL_TABLE_DEFAULT
    signal_csv_path = dbutils.widgets.get("signal_csv_path").strip() or SIGNAL_CSV_PATH_DEFAULT
    market_table = dbutils.widgets.get("market_table").strip() or MARKET_TABLE_DEFAULT
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


def load_signals():
    source_used = None

    if table_exists(signal_table):
        pdf = spark.table(signal_table).select("timestamp", "predicted_regime").toPandas()
        pdf = normalize_timestamp(pdf)
        pdf = pdf.dropna(subset=["timestamp"]).drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        if not pdf.empty:
            source_used = {"type": "delta_table", "source": signal_table}
            return (
                pdf.sort_values("timestamp").reset_index(drop=True),
                source_used,
            )

    signal_path = Path(signal_csv_path)
    if not signal_path.exists():
        raise FileNotFoundError(
            f"Signal table empty/missing and fallback CSV not found: {signal_csv_path}"
        )

    pdf = pd.read_csv(signal_csv_path, parse_dates=["timestamp"])
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )
    if "predicted_regime" not in pdf.columns:
        raise ValueError(f"Signal CSV missing predicted_regime column: {signal_csv_path}")

    source_used = {"type": "csv", "source": signal_csv_path}
    return pdf[["timestamp", "predicted_regime"]].sort_values("timestamp").reset_index(
        drop=True
    ), source_used


def load_prices():
    pdf = (
        spark.sql(
            f"""
            SELECT timestamp, symbol, close
            FROM {market_table}
            WHERE timestamp >= TIMESTAMP('{live_start}')
              AND timestamp < TIMESTAMP('{live_end}')
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
    pivoted = pivoted.dropna(subset=["SPY", "QQQ"]).reset_index(drop=True)
    return pivoted


def align_inputs(signals_pdf, prices_pdf):
    signals = signals_pdf.copy()
    signals["predicted_regime"] = pd.to_numeric(
        signals["predicted_regime"],
        errors="coerce",
    ).fillna(2).astype(int)
    signals = signals[
        (signals["timestamp"] >= pd.Timestamp(live_start))
        & (signals["timestamp"] < pd.Timestamp(live_end))
    ].copy()

    merged = signals.merge(prices_pdf, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def run_hourly_backtest(aligned_pdf):
    if aligned_pdf.empty:
        raise ValueError("Aligned signal/price frame is empty.")

    portfolio_values = [float(initial_capital)]
    target_assets = ["START"]
    position_changed = [False]
    asset_returns = [0.0]
    transaction_costs = [0.0]
    applied_regimes = [np.nan]
    strategy_returns = [0.0]

    active_asset = "START"

    for idx in range(1, len(aligned_pdf)):
        prior_regime = int(aligned_pdf["predicted_regime"].iloc[idx - 1])
        target_asset = REGIME_TO_ASSET.get(prior_regime, "CASH")

        if target_asset == "QQQ":
            gross_return = aligned_pdf["QQQ"].iloc[idx] / aligned_pdf["QQQ"].iloc[idx - 1] - 1
        elif target_asset == "SPY":
            gross_return = aligned_pdf["SPY"].iloc[idx] / aligned_pdf["SPY"].iloc[idx - 1] - 1
        else:
            gross_return = 0.0

        switch_cost = transaction_cost if target_asset != active_asset else 0.0
        net_return = gross_return - switch_cost
        next_value = portfolio_values[-1] * (1 + net_return)

        portfolio_values.append(next_value)
        target_assets.append(target_asset)
        position_changed.append(target_asset != active_asset)
        asset_returns.append(gross_return)
        transaction_costs.append(switch_cost)
        applied_regimes.append(prior_regime)
        strategy_returns.append(net_return)
        active_asset = target_asset

    result = aligned_pdf.copy()
    result["applied_regime"] = applied_regimes
    result["target_asset"] = target_assets
    result["gross_asset_return"] = asset_returns
    result["switch_cost"] = transaction_costs
    result["strategy_return"] = strategy_returns
    result["position_changed"] = position_changed
    result["portfolio_value"] = portfolio_values
    return result


def build_daily_curves(hourly_pdf):
    daily_equity = (
        hourly_pdf.set_index("timestamp")["portfolio_value"].resample("D").last().dropna()
    )
    daily_drawdown = daily_equity / daily_equity.cummax() - 1
    return daily_equity, daily_drawdown


def annualized_ratio(returns, downside_only=False):
    series = returns.copy()
    if downside_only:
        series = series[series < 0]
    if len(series) == 0 or series.std() == 0 or np.isnan(series.std()):
        return np.nan
    return float((returns.mean() / series.std()) * np.sqrt(TRADING_DAYS))


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


def compute_metrics(daily_equity, hourly_pdf):
    daily_returns = daily_equity.pct_change(fill_method=None).dropna()
    start_value = float(daily_equity.iloc[0])
    end_value = float(daily_equity.iloc[-1])
    total_return = end_value / start_value - 1

    first_date = daily_equity.index.min()
    last_date = daily_equity.index.max()
    elapsed_days = max((last_date - first_date).days, 1)
    years = elapsed_days / 365
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan

    volatility = (
        float(daily_returns.std() * np.sqrt(TRADING_DAYS))
        if len(daily_returns)
        else np.nan
    )
    sharpe = annualized_ratio(daily_returns, downside_only=False)
    sortino = annualized_ratio(daily_returns, downside_only=True)
    drawdown = daily_equity / daily_equity.cummax() - 1

    applied_assets = hourly_pdf["target_asset"].iloc[1:].reset_index(drop=True)
    holding_groups = applied_assets.ne(applied_assets.shift()).cumsum()
    holding_lengths = applied_assets.groupby(holding_groups).size()

    return {
        "Strategy Start Date": str(first_date.date()),
        "Strategy End Date": str(last_date.date()),
        "Daily Observations": int(len(daily_equity)),
        "Start Value": start_value,
        "Final Value": end_value,
        "Total Return": float(total_return),
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "Total Regime Switches": int(hourly_pdf["position_changed"].sum()),
        "Average Holding Bars": float(holding_lengths.mean()) if len(holding_lengths) else np.nan,
    }


def build_regime_stats(hourly_pdf):
    executable = hourly_pdf.iloc[1:].copy()
    executable["applied_regime_name"] = executable["applied_regime"].map(
        {0: "Expansion", 1: "Slowdown", 2: "Recession/Contraction"}
    )
    mix = (
        executable["applied_regime"]
        .value_counts(normalize=True)
        .rename_axis("applied_regime")
        .reset_index(name="pct_time")
        .sort_values("applied_regime")
    )
    mix["applied_regime_name"] = mix["applied_regime"].map(
        {0: "Expansion", 1: "Slowdown", 2: "Recession/Contraction"}
    )
    mix["target_asset"] = mix["applied_regime"].map(REGIME_TO_ASSET)
    mix["bars"] = mix["applied_regime"].map(
        executable["applied_regime"].value_counts().to_dict()
    ).fillna(0).astype(int)
    mix["total_switches"] = int(hourly_pdf["position_changed"].sum())
    mix["average_holding_bars"] = compute_metrics(
        hourly_pdf.set_index("timestamp")["portfolio_value"].resample("D").last().dropna(),
        hourly_pdf,
    )["Average Holding Bars"]
    return mix


def build_positive_periods(daily_equity):
    period_frames = []
    period_config = {
        "week": "W-FRI",
        "month": "M",
        "year": "Y",
    }

    for period_type, freq in period_config.items():
        period_equity = daily_equity.resample(freq).last().dropna()
        period_returns = period_equity.pct_change(fill_method=None).dropna()
        if period_returns.empty:
            continue

        frame = pd.DataFrame(
            {
                "period_type": period_type,
                "period_end": period_returns.index,
                "start_value": period_equity.shift(1).loc[period_returns.index].values,
                "end_value": period_equity.loc[period_returns.index].values,
                "period_return": period_returns.values,
            }
        )
        frame["is_positive"] = frame["period_return"] > 0
        period_frames.append(frame)

    if not period_frames:
        return pd.DataFrame(
            columns=[
                "period_type",
                "period_end",
                "start_value",
                "end_value",
                "period_return",
                "is_positive",
            ]
        )
    return pd.concat(period_frames, ignore_index=True)


signals_pdf, signal_source = load_signals()
prices_pdf = load_prices()
aligned_pdf = align_inputs(signals_pdf, prices_pdf)
hourly_equity_pdf = run_hourly_backtest(aligned_pdf)
daily_equity, daily_drawdown = build_daily_curves(hourly_equity_pdf)
metrics = compute_metrics(daily_equity, hourly_equity_pdf)
regime_stats_pdf = build_regime_stats(hourly_equity_pdf)
positive_periods_pdf = build_positive_periods(daily_equity)

metrics["Positive Weeks"] = int(
    positive_periods_pdf.query("period_type == 'week' and is_positive").shape[0]
)
metrics["Positive Months"] = int(
    positive_periods_pdf.query("period_type == 'month' and is_positive").shape[0]
)
metrics["Positive Years"] = int(
    positive_periods_pdf.query("period_type == 'year' and is_positive").shape[0]
)

performance_metrics_pdf = pd.DataFrame([metrics])
daily_equity_curve_pdf = daily_equity.rename("equity").reset_index()
drawdown_curve_pdf = daily_drawdown.rename("drawdown").reset_index()

performance_metrics_path = f"{output_root}/performance_metrics.csv"
daily_equity_curve_path = f"{output_root}/daily_equity_curve.csv"
hourly_equity_curve_path = f"{output_root}/hourly_equity_curve.csv"
drawdown_curve_path = f"{output_root}/drawdown_curve.csv"
regime_stats_path = f"{output_root}/regime_stats.csv"
positive_periods_path = f"{output_root}/positive_periods.csv"
summary_path = f"{output_root}/summary.md"
methodology_notes_path = f"{output_root}/methodology_notes.md"
run_metadata_path = f"{output_root}/run_metadata.json"
equity_plot_path = f"{output_root}/equity_curve.png"
drawdown_plot_path = f"{output_root}/drawdown_curve.png"

performance_metrics_pdf.to_csv(performance_metrics_path, index=False)
daily_equity_curve_pdf.to_csv(daily_equity_curve_path, index=False)
hourly_equity_pdf.to_csv(hourly_equity_curve_path, index=False)
drawdown_curve_pdf.to_csv(drawdown_curve_path, index=False)
regime_stats_pdf.to_csv(regime_stats_path, index=False)
positive_periods_pdf.to_csv(positive_periods_path, index=False)

methodology_lines = [
    "# Canonical Legacy V4 Performance Reporting Methodology",
    "",
    "- Hourly signals are used.",
    "- Hourly prices are used.",
    "- Prior-bar execution is used.",
    "- `0=Expansion -> QQQ`, `1=Slowdown -> SPY`, `2=Recession/Contraction -> CASH`.",
    "- Transaction cost is `5 bps` on switch.",
    "- Equity is built hourly.",
    "- Metrics are computed from daily-resampled equity.",
    "- CAGR uses `365-day` convention.",
    "- Volatility, Sharpe, and Sortino are annualized from daily returns using `sqrt(252)`.",
    "- The default signal source is the comparison-safe recreated legacy baseline walk-forward signal table, with CSV fallback only if the table is unavailable.",
]
write_text(methodology_notes_path, "\n".join(methodology_lines) + "\n")

summary_lines = [
    "# Canonical Legacy V4 Performance Summary",
    "",
    "## Inputs",
    "",
    f"- Signal source used: `{signal_source['type']}` → `{signal_source['source']}`",
    f"- Market price table: `{market_table}`",
    f"- Strategy start: `{metrics['Strategy Start Date']}`",
    f"- Strategy end: `{metrics['Strategy End Date']}`",
    f"- Initial capital: `{initial_capital}`",
    f"- Transaction cost: `{transaction_cost}`",
    "",
    "## Alignment",
    "",
    f"- Signal rows loaded: `{len(signals_pdf)}`",
    f"- Price rows loaded: `{len(prices_pdf)}`",
    f"- Overlapping hourly rows used: `{len(aligned_pdf)}`",
    f"- Daily observations: `{metrics['Daily Observations']}`",
    "",
    "## Performance Table",
    "",
    markdown_table(performance_metrics_pdf),
    "",
    "## Interpretation",
    "",
    "- This output is the canonical reusable performance table for the recreated legacy V4 baseline.",
    "- It is comparison-safe and does not mutate production tables, live aliases, registry objects, workflow JSON, or RT pipelines.",
]
write_text(summary_path, "\n".join(summary_lines) + "\n")

plt.figure(figsize=(14, 7))
plt.plot(daily_equity_curve_pdf["timestamp"], daily_equity_curve_pdf["equity"], color="navy")
plt.title("Canonical Legacy V4 Daily Equity Curve")
plt.ylabel("Equity")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(equity_plot_path, dpi=180)
plt.close()

plt.figure(figsize=(14, 7))
plt.plot(drawdown_curve_pdf["timestamp"], drawdown_curve_pdf["drawdown"], color="darkred")
plt.title("Canonical Legacy V4 Daily Drawdown")
plt.ylabel("Drawdown")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(drawdown_plot_path, dpi=180)
plt.close()

run_metadata = {
    "status": "success",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "output_root": output_root,
    "signal_source": signal_source,
    "input_tables": {
        "signal_table": table_row_range(signal_table),
        "market_table": table_row_range(market_table),
    },
    "input_counts": {
        "signal_rows_loaded": int(len(signals_pdf)),
        "price_rows_loaded": int(len(prices_pdf)),
        "aligned_rows_used": int(len(aligned_pdf)),
        "daily_observations": int(metrics["Daily Observations"]),
    },
    "methodology": {
        "signal_frequency": "hourly",
        "price_frequency": "hourly",
        "execution_timing": "prior_bar",
        "regime_to_asset": REGIME_TO_ASSET,
        "transaction_cost": transaction_cost,
        "transaction_cost_application": "subtractive_on_switch",
        "equity_path": "hourly",
        "reporting_layer": "daily_resampled",
        "cagr_convention": "calendar_days_divided_by_365",
    },
    "artifacts": {
        "performance_metrics": performance_metrics_path,
        "daily_equity_curve": daily_equity_curve_path,
        "hourly_equity_curve": hourly_equity_curve_path,
        "drawdown_curve": drawdown_curve_path,
        "regime_stats": regime_stats_path,
        "positive_periods": positive_periods_path,
        "summary": summary_path,
        "methodology_notes": methodology_notes_path,
        "equity_curve_plot": equity_plot_path,
        "drawdown_curve_plot": drawdown_plot_path,
    },
}
write_json(run_metadata_path, run_metadata)

result = {
    "phase": "canonical_legacy_v4_performance",
    "status": "success",
    "output_root": output_root,
    "signal_source": signal_source,
    "performance_metrics": metrics,
}

print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

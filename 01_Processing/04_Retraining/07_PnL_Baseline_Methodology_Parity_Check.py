# Databricks notebook source
# Audit whether 06_PnL_Comparison_V4_vs_V4_1.py matches accepted V4 baseline methodology

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
COMPARISON_ROOT = f"{VOLUME_ROOT}/comparison"
DEFAULT_FULL_REBUILD_ROOT = f"{COMPARISON_ROOT}/2026-04-18_full_rebuild"
DEFAULT_PARITY_ROOT = f"{COMPARISON_ROOT}/2026-04-18_pnl_parity_check"
DEFAULT_MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"

INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
START_DATE_DEFAULT = "2021-01-01"

POSITION_MAP = {0: "QQQ", 1: "SPY", 2: "CASH"}
REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
TRADING_DAYS = 252


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


def load_json(path_str):
    return json.loads(Path(path_str).read_text())


def load_signal_frame(path_str, candidate_columns, output_column):
    pdf = load_csv(path_str)
    if pdf.empty:
        raise FileNotFoundError(f"Signal file missing or empty: {path_str}")
    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf.drop_duplicates(subset=["timestamp"], keep="last")

    chosen_column = next(
        (column for column in candidate_columns if column in pdf.columns), None
    )
    if not chosen_column:
        raise KeyError(
            f"No regime column in {path_str}. candidates={candidate_columns}, columns={list(pdf.columns)}"
        )

    signal_pdf = pdf.copy()
    signal_pdf[output_column] = (
        pd.to_numeric(signal_pdf[chosen_column], errors="coerce").fillna(2).astype(int)
    )
    return signal_pdf


def load_price_frame(table_name, start_date=None):
    where_clause = ""
    if start_date:
        where_clause = f"WHERE timestamp >= TIMESTAMP('{start_date}')"
    price_pdf = (
        spark.sql(
            f"""
            SELECT timestamp, symbol, close
            FROM {table_name}
            {where_clause}
            """
        )
        .toPandas()
    )
    price_pdf = normalize_timestamp(price_pdf)
    price_pdf = price_pdf.dropna(subset=["timestamp"])
    price_pdf = price_pdf[price_pdf["symbol"].isin(["SPY", "QQQ"])].copy()
    price_pdf = price_pdf.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    pivoted = (
        price_pdf.pivot_table(
            index="timestamp", columns="symbol", values="close", aggfunc="last"
        )
        .reset_index()
        .rename(columns={"SPY": "SPY_close", "QQQ": "QQQ_close"})
        .sort_values("timestamp")
    )
    pivoted["SPY_close"] = pd.to_numeric(pivoted["SPY_close"], errors="coerce").ffill()
    pivoted["QQQ_close"] = pd.to_numeric(pivoted["QQQ_close"], errors="coerce").ffill()
    pivoted["SPY_return"] = pivoted["SPY_close"].pct_change().fillna(0.0)
    pivoted["QQQ_return"] = pivoted["QQQ_close"].pct_change().fillna(0.0)
    return pivoted


def load_artifact_metrics(path_str, model_name):
    pdf = load_csv(path_str)
    if pdf.empty:
        return {}
    row = pdf.loc[pdf["model"] == model_name]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def contiguous_run_lengths(series):
    if series.empty:
        return pd.Series(dtype=float)
    groups = series.ne(series.shift()).cumsum()
    return series.groupby(groups).size().astype(float)


def artifact_style_metrics(backtest_pdf, initial_capital):
    start_ts = backtest_pdf["timestamp"].iloc[0]
    end_ts = backtest_pdf["timestamp"].iloc[-1]
    days = max((end_ts - start_ts).days, 1)
    years = days / 365
    final_equity = float(backtest_pdf["equity"].iloc[-1])
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan
    bars_per_year = len(backtest_pdf) / years if years > 0 else np.nan

    net_returns = backtest_pdf["net_return"].fillna(0.0)
    return_std = float(net_returns.std(ddof=0))
    annualized_volatility = (
        return_std * np.sqrt(bars_per_year) if np.isfinite(bars_per_year) else np.nan
    )
    average_return = float(net_returns.mean())
    sharpe_ratio = (
        average_return / return_std * np.sqrt(bars_per_year)
        if return_std > 0 and np.isfinite(bars_per_year)
        else np.nan
    )

    drawdown = backtest_pdf["equity"] / backtest_pdf["equity"].cummax() - 1.0
    max_drawdown = float(-drawdown.min()) if not drawdown.empty else np.nan
    calmar_ratio = cagr / max_drawdown if max_drawdown and max_drawdown > 0 else np.nan

    holding_lengths = contiguous_run_lengths(backtest_pdf["position"])
    return {
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "rows": int(len(backtest_pdf)),
        "days": int(days),
        "years": float(years),
        "bars_per_year": float(bars_per_year) if np.isfinite(bars_per_year) else np.nan,
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": final_equity / initial_capital - 1.0,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "annualized_volatility": float(annualized_volatility)
        if np.isfinite(annualized_volatility)
        else np.nan,
        "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
        "max_drawdown": float(max_drawdown) if np.isfinite(max_drawdown) else np.nan,
        "calmar_ratio": float(calmar_ratio) if np.isfinite(calmar_ratio) else np.nan,
        "total_regime_switches": int(
            backtest_pdf["position"].ne(backtest_pdf["position"].shift()).iloc[1:].sum()
        ),
        "transaction_cost_events": int(backtest_pdf["transaction_cost_applied"].sum() > 0)
        if "transaction_cost_applied" not in backtest_pdf.columns
        else int(backtest_pdf["position_changed"].sum()),
        "average_holding_bars": float(holding_lengths.mean())
        if not holding_lengths.empty
        else np.nan,
        "expansion_pct_time": float(backtest_pdf["regime"].eq(0).mean()),
        "slowdown_pct_time": float(backtest_pdf["regime"].eq(1).mean()),
        "recession_pct_time": float(backtest_pdf["regime"].eq(2).mean()),
    }


def daily_metrics_from_equity(equity_series, initial_capital):
    daily_equity = equity_series.resample("D").last().dropna()
    start_ts = daily_equity.index[0]
    end_ts = daily_equity.index[-1]
    years = max((end_ts - start_ts).days / 365.25, 1 / 365.25)
    daily_ret = daily_equity.pct_change().dropna()
    sigma = daily_ret.std()
    mean_ret = daily_ret.mean()
    vol = sigma * np.sqrt(TRADING_DAYS) if pd.notna(sigma) else np.nan
    sharpe = (
        (mean_ret / sigma) * np.sqrt(TRADING_DAYS)
        if pd.notna(sigma) and sigma != 0
        else np.nan
    )
    rolling_max = daily_equity.cummax()
    drawdown = (daily_equity / rolling_max) - 1
    max_dd = float(-drawdown.min()) if not drawdown.empty else np.nan
    final_val = float(daily_equity.iloc[-1])
    cagr = (final_val / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan
    calmar = cagr / max_dd if max_dd and max_dd > 0 else np.nan
    return {
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "rows": int(len(daily_equity)),
        "days": int((end_ts - start_ts).days),
        "years": float(years),
        "initial_capital": float(initial_capital),
        "final_equity": final_val,
        "total_return": final_val / initial_capital - 1.0,
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "annualized_volatility": float(vol) if np.isfinite(vol) else np.nan,
        "sharpe_ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else np.nan,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else np.nan,
    }


def run_samebar_06_style(signal_pdf, price_pdf, initial_capital, transaction_cost):
    aligned = signal_pdf.merge(
        price_pdf[
            ["timestamp", "SPY_close", "QQQ_close", "SPY_return", "QQQ_return"]
        ],
        on="timestamp",
        how="inner",
    ).sort_values("timestamp")
    aligned = aligned.reset_index(drop=True)

    backtest = aligned[
        ["timestamp", "predicted_regime", "SPY_close", "QQQ_close", "SPY_return", "QQQ_return"]
    ].copy()
    backtest["regime"] = (
        pd.to_numeric(backtest["predicted_regime"], errors="coerce").fillna(2).astype(int)
    )
    backtest["regime_name"] = backtest["regime"].map(REGIME_NAME_MAP)
    backtest["position"] = backtest["regime"].map(POSITION_MAP).fillna("CASH")
    previous_position = backtest["position"].shift(1).fillna("CASH")
    backtest["position_changed"] = backtest["position"] != previous_position
    backtest["transaction_cost_applied"] = np.where(
        backtest["position_changed"], transaction_cost, 0.0
    )
    backtest["gross_return"] = np.select(
        [backtest["position"].eq("SPY"), backtest["position"].eq("QQQ")],
        [backtest["SPY_return"], backtest["QQQ_return"]],
        default=0.0,
    )
    backtest["net_return"] = (
        (1.0 + backtest["gross_return"])
        * (1.0 - backtest["transaction_cost_applied"])
        - 1.0
    )
    backtest["equity"] = initial_capital * (1.0 + backtest["net_return"]).cumprod()
    backtest["drawdown"] = backtest["equity"] / backtest["equity"].cummax() - 1.0
    metrics = artifact_style_metrics(backtest, initial_capital)
    return aligned, backtest, metrics


def run_canonical_lagged(signal_pdf, price_pdf, initial_capital, transaction_cost):
    merged = signal_pdf.merge(
        price_pdf[["timestamp", "SPY_close", "QQQ_close"]],
        on="timestamp",
        how="left",
    ).sort_values("timestamp")
    merged["SPY_close"] = pd.to_numeric(merged["SPY_close"], errors="coerce").ffill()
    merged["QQQ_close"] = pd.to_numeric(merged["QQQ_close"], errors="coerce").ffill()
    merged = merged.dropna(subset=["SPY_close", "QQQ_close"]).reset_index(drop=True)

    portfolio = [initial_capital]
    records = []
    current_value = initial_capital
    active_asset = "START"

    for i in range(len(merged)):
        row = merged.iloc[i]
        record = {
            "timestamp": row["timestamp"],
            "signal_regime_at_t": int(row["predicted_regime"]),
            "applied_regime": np.nan,
            "applied_position": "START" if i == 0 else None,
            "SPY_close": row["SPY_close"],
            "QQQ_close": row["QQQ_close"],
            "gross_return": 0.0,
            "transaction_cost_applied": 0.0,
            "net_return": 0.0,
            "equity": current_value if i == 0 else None,
        }
        if i > 0:
            prior_regime = int(merged.iloc[i - 1]["predicted_regime"])
            target = POSITION_MAP.get(prior_regime, "CASH")
            spy_ret = (merged.iloc[i]["SPY_close"] / merged.iloc[i - 1]["SPY_close"]) - 1
            qqq_ret = (merged.iloc[i]["QQQ_close"] / merged.iloc[i - 1]["QQQ_close"]) - 1
            if target == "QQQ":
                gross_return = qqq_ret
            elif target == "SPY":
                gross_return = spy_ret
            else:
                gross_return = 0.0
            friction = transaction_cost if target != active_asset else 0.0
            net_return = gross_return - friction
            current_value *= (1.0 + net_return)

            record.update(
                {
                    "applied_regime": prior_regime,
                    "applied_position": target,
                    "SPY_return": spy_ret,
                    "QQQ_return": qqq_ret,
                    "gross_return": gross_return,
                    "transaction_cost_applied": friction,
                    "net_return": net_return,
                    "equity": current_value,
                    "position_changed": target != active_asset,
                }
            )
            active_asset = target
        records.append(record)
        portfolio.append(current_value)

    backtest = pd.DataFrame(records)
    if "position_changed" not in backtest.columns:
        backtest["position_changed"] = False
    backtest["applied_position"] = backtest["applied_position"].fillna("CASH")
    backtest["regime"] = (
        backtest["applied_regime"].fillna(backtest["signal_regime_at_t"]).astype(int)
    )
    backtest["position"] = backtest["applied_position"]
    backtest["drawdown"] = backtest["equity"] / backtest["equity"].cummax() - 1.0

    metrics = daily_metrics_from_equity(
        backtest.set_index("timestamp")["equity"], initial_capital
    )
    metrics.update(
        {
            "bars_per_year": float(len(backtest) / max(metrics["years"], 1e-9)),
            "total_regime_switches": int(backtest["position_changed"].sum()),
            "transaction_cost_events": int(backtest["position_changed"].sum()),
            "average_holding_bars": float(
                contiguous_run_lengths(backtest["position"]).mean()
            ),
            "expansion_pct_time": float(backtest["regime"].eq(0).mean()),
            "slowdown_pct_time": float(backtest["regime"].eq(1).mean()),
            "recession_pct_time": float(backtest["regime"].eq(2).mean()),
        }
    )
    return merged, backtest, metrics


def summarize_source(label, pdf, source_kind, path_or_table, regime_column=None):
    if pdf.empty:
        return {
            "source_label": label,
            "source_kind": source_kind,
            "path_or_table": path_or_table,
            "rows": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "regime_column": regime_column,
        }
    return {
        "source_label": label,
        "source_kind": source_kind,
        "path_or_table": path_or_table,
        "rows": int(len(pdf)),
        "first_timestamp": pdf["timestamp"].min(),
        "last_timestamp": pdf["timestamp"].max(),
        "regime_column": regime_column,
    }


def to_metric_row(label, methodology, signal_source, price_source, metric_basis, metrics):
    row = {
        "label": label,
        "methodology": methodology,
        "signal_source": signal_source,
        "price_source": price_source,
        "metric_basis": metric_basis,
    }
    row.update(metrics)
    return row


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


comparison_root = DEFAULT_FULL_REBUILD_ROOT
parity_root = DEFAULT_PARITY_ROOT
market_table = DEFAULT_MARKET_TABLE
start_date = START_DATE_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("comparison_root", DEFAULT_FULL_REBUILD_ROOT)
    dbutils.widgets.text("parity_root", DEFAULT_PARITY_ROOT)
    dbutils.widgets.text("market_table", DEFAULT_MARKET_TABLE)
    dbutils.widgets.text("start_date", START_DATE_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    comparison_root = dbutils.widgets.get("comparison_root").strip() or DEFAULT_FULL_REBUILD_ROOT
    parity_root = dbutils.widgets.get("parity_root").strip() or DEFAULT_PARITY_ROOT
    market_table = dbutils.widgets.get("market_table").strip() or DEFAULT_MARKET_TABLE
    start_date = dbutils.widgets.get("start_date").strip() or START_DATE_DEFAULT
    initial_capital = float(dbutils.widgets.get("initial_capital").strip() or INITIAL_CAPITAL_DEFAULT)
    transaction_cost = float(dbutils.widgets.get("transaction_cost").strip() or TRANSACTION_COST_DEFAULT)

ensure_dir(parity_root)

baseline_root = f"{comparison_root}/baseline"
full_rebuild_root = f"{comparison_root}/full_rebuild_retrain"
pnl_root = f"{comparison_root}/pnl_comparison"

baseline_signal_path = f"{baseline_root}/current_walkforward_signals.csv"
baseline_full_signal_path = f"{baseline_root}/current_full_signal_export.csv"
full_rebuild_signal_path = f"{full_rebuild_root}/walkforward_signals.csv"
pnl_metrics_path = f"{pnl_root}/pnl_metrics.csv"
equity_curve_path = f"{pnl_root}/equity_curve.csv"
run_metadata_path = f"{pnl_root}/run_metadata.json"
baseline_metadata_path = f"{baseline_root}/current_model_metadata.json"

baseline_wf = load_signal_frame(
    baseline_signal_path,
    ["predicted_regime", "model_predicted_regime"],
    "predicted_regime",
)
baseline_wf = baseline_wf[baseline_wf["timestamp"] >= pd.Timestamp(start_date)].copy()

baseline_full = load_signal_frame(
    baseline_full_signal_path,
    ["predicted_regime", "model_predicted_regime"],
    "predicted_regime",
)

full_rebuild_wf = load_signal_frame(
    full_rebuild_signal_path,
    ["predicted_regime", "model_predicted_regime"],
    "predicted_regime",
)
full_rebuild_wf = full_rebuild_wf[
    full_rebuild_wf["timestamp"] >= pd.Timestamp(start_date)
].copy()

price_pdf = load_price_frame(market_table, start_date=None)
price_2021 = price_pdf[price_pdf["timestamp"] >= pd.Timestamp(start_date)].copy()

artifact_metrics = load_artifact_metrics(pnl_metrics_path, "baseline")
equity_curve_artifact = normalize_timestamp(load_csv(equity_curve_path))
run_metadata = load_json(run_metadata_path)
baseline_metadata = load_json(baseline_metadata_path)

artifact_first_ts = (
    equity_curve_artifact["timestamp"].min() if not equity_curve_artifact.empty else None
)
artifact_last_ts = (
    equity_curve_artifact["timestamp"].max() if not equity_curve_artifact.empty else None
)

artifact_signal_common = baseline_wf[["timestamp", "predicted_regime"]].merge(
    full_rebuild_wf[["timestamp"]], on="timestamp", how="inner"
)
artifact_signal_common = artifact_signal_common.merge(
    price_2021[["timestamp", "SPY_close", "QQQ_close", "SPY_return", "QQQ_return"]],
    on="timestamp",
    how="inner",
).sort_values("timestamp")

_, exact_artifact_backtest, exact_artifact_metrics = run_samebar_06_style(
    artifact_signal_common[["timestamp", "predicted_regime"]].copy(),
    artifact_signal_common[
        ["timestamp", "SPY_close", "QQQ_close", "SPY_return", "QQQ_return"]
    ].copy(),
    initial_capital,
    transaction_cost,
)

_, samebar_baseline_only_backtest, samebar_baseline_only_metrics = run_samebar_06_style(
    baseline_wf[["timestamp", "predicted_regime"]].copy(),
    price_2021.copy(),
    initial_capital,
    transaction_cost,
)

canonical_wf_merged, canonical_wf_backtest, canonical_wf_metrics = run_canonical_lagged(
    baseline_wf[["timestamp", "predicted_regime"]].copy(),
    price_2021.copy(),
    initial_capital,
    transaction_cost,
)

baseline_full_2021 = baseline_full[
    baseline_full["timestamp"] >= pd.Timestamp(start_date)
].copy()
_, canonical_full_2021_backtest, canonical_full_2021_metrics = run_canonical_lagged(
    baseline_full_2021[["timestamp", "predicted_regime"]].copy(),
    price_2021.copy(),
    initial_capital,
    transaction_cost,
)

full_price_end = min(
    baseline_full["timestamp"].max(),
    price_pdf["timestamp"].max(),
) if not baseline_full.empty and not price_pdf.empty else None
baseline_full_trimmed = baseline_full.copy()
if full_price_end is not None:
    baseline_full_trimmed = baseline_full_trimmed[
        baseline_full_trimmed["timestamp"] <= full_price_end
    ].copy()
_, canonical_full_fullhist_backtest, canonical_full_fullhist_metrics = run_canonical_lagged(
    baseline_full_trimmed[["timestamp", "predicted_regime"]].copy(),
    price_pdf.copy(),
    initial_capital,
    transaction_cost,
)

baseline_vs_canonical_metrics = pd.DataFrame(
    [
        to_metric_row(
            "artifact_baseline_from_existing_pnl_pack",
            "06_pnl_notebook_existing_output",
            baseline_signal_path,
            run_metadata.get("resolved_market_table"),
            "hourly_samebar_metrics",
            artifact_metrics,
        ),
        to_metric_row(
            "recomputed_06_exact_common_horizon",
            "samebar_signal_applied_to_same_row_return_with_common_horizon",
            baseline_signal_path,
            market_table,
            "hourly_samebar_metrics",
            exact_artifact_metrics,
        ),
        to_metric_row(
            "recomputed_06_baseline_only",
            "samebar_signal_applied_to_same_row_return_without_full_rebuild_truncation",
            baseline_signal_path,
            market_table,
            "hourly_samebar_metrics",
            samebar_baseline_only_metrics,
        ),
        to_metric_row(
            "canonical_walkforward_baseline",
            "lagged_signal_prior_bar_walkforward_baseline",
            baseline_signal_path,
            market_table,
            "daily_metrics_from_intraday_equity",
            canonical_wf_metrics,
        ),
        to_metric_row(
            "canonical_fullsignal_2021_plus",
            "lagged_signal_prior_bar_promoted_model_full_signal_2021_plus",
            baseline_full_signal_path,
            market_table,
            "daily_metrics_from_intraday_equity",
            canonical_full_2021_metrics,
        ),
        to_metric_row(
            "canonical_fullsignal_full_history",
            "lagged_signal_prior_bar_promoted_model_full_signal_full_history",
            baseline_full_signal_path,
            market_table,
            "daily_metrics_from_intraday_equity",
            canonical_full_fullhist_metrics,
        ),
    ]
)

signal_source_audit = pd.DataFrame(
    [
        summarize_source(
            "baseline_walkforward_snapshot",
            baseline_wf,
            "csv",
            baseline_signal_path,
            "predicted_regime",
        ),
        summarize_source(
            "baseline_full_signal_snapshot",
            baseline_full,
            "csv",
            baseline_full_signal_path,
            "predicted_regime",
        ),
        summarize_source(
            "full_rebuild_walkforward_snapshot",
            full_rebuild_wf,
            "csv",
            full_rebuild_signal_path,
            "predicted_regime",
        ),
        summarize_source(
            "bronze_market_prices",
            price_pdf[["timestamp"]].copy(),
            "table",
            market_table,
            None,
        ),
    ]
)
signal_source_audit["matches_live_start_2021_01_01"] = (
    signal_source_audit["first_timestamp"].astype(str).str.startswith("2021-01-01")
)

timing_alignment_audit = pd.DataFrame(
    [
        {
            "method": "06_existing_pnl_artifact",
            "signal_source": baseline_signal_path,
            "price_source": run_metadata.get("resolved_market_table"),
            "position_map": "0->QQQ,1->SPY,2->CASH",
            "signal_timing": "same_row_signal",
            "return_window": "return_t_over_t_minus_1",
            "applied_bar": "signal_at_t controls return_t_minus_1_to_t",
            "transaction_cost_logic": "multiplicative_(1+r)*(1-cost)-1",
            "metric_basis": "hourly_returns",
            "first_timestamp": artifact_first_ts,
            "last_timestamp": artifact_last_ts,
            "rows": int(len(equity_curve_artifact)) if not equity_curve_artifact.empty else 0,
            "notes": "Also truncated to timestamps common with full rebuild output.",
        },
        {
            "method": "canonical_walkforward_baseline",
            "signal_source": baseline_signal_path,
            "price_source": market_table,
            "position_map": "0->QQQ,1->SPY,2->CASH",
            "signal_timing": "prior_row_signal",
            "return_window": "return_t_over_t_minus_1",
            "applied_bar": "signal_at_t_minus_1 controls return_t_minus_1_to_t",
            "transaction_cost_logic": "linear_return_minus_cost",
            "metric_basis": "daily_resampled_equity",
            "first_timestamp": canonical_wf_backtest["timestamp"].min(),
            "last_timestamp": canonical_wf_backtest["timestamp"].max(),
            "rows": int(len(canonical_wf_backtest)),
            "notes": "Matches 03_Performance_Exports.py walk-forward NAV construction.",
        },
        {
            "method": "accepted_framing_fullsignal_notebook",
            "signal_source": baseline_full_signal_path,
            "price_source": market_table,
            "position_map": "0->QQQ,1->SPY,2->CASH",
            "signal_timing": "prior_row_signal",
            "return_window": "return_t_over_t_minus_1",
            "applied_bar": "signal_at_t_minus_1 controls return_t_minus_1_to_t",
            "transaction_cost_logic": "linear_return_minus_cost",
            "metric_basis": "daily_resampled_equity",
            "first_timestamp": canonical_full_fullhist_backtest["timestamp"].min(),
            "last_timestamp": canonical_full_fullhist_backtest["timestamp"].max(),
            "rows": int(len(canonical_full_fullhist_backtest)),
            "notes": "Matches 01_Performance_Deep_Dive.py / 02_Performance_Analysis.py structure.",
        },
    ]
)

hourly_return_audit_sample = samebar_baseline_only_backtest[
    [
        "timestamp",
        "regime",
        "position",
        "SPY_return",
        "QQQ_return",
        "gross_return",
        "transaction_cost_applied",
        "net_return",
        "equity",
    ]
].head(48).copy()
hourly_return_audit_sample = hourly_return_audit_sample.merge(
    canonical_wf_backtest[
        [
            "timestamp",
            "signal_regime_at_t",
            "applied_regime",
            "applied_position",
            "gross_return",
            "transaction_cost_applied",
            "net_return",
            "equity",
        ]
    ].rename(
        columns={
            "signal_regime_at_t": "canonical_signal_regime_at_t",
            "applied_regime": "canonical_applied_regime",
            "applied_position": "canonical_applied_position",
            "gross_return": "canonical_gross_return",
            "transaction_cost_applied": "canonical_transaction_cost",
            "net_return": "canonical_net_return",
            "equity": "canonical_equity",
        }
    ),
    on="timestamp",
    how="left",
)

switch_cost_audit_sample = canonical_wf_backtest[
    canonical_wf_backtest["position_changed"].fillna(False)
].head(25).copy()

artifact_row = baseline_vs_canonical_metrics.loc[
    baseline_vs_canonical_metrics["label"] == "artifact_baseline_from_existing_pnl_pack"
].iloc[0]
canonical_row = baseline_vs_canonical_metrics.loc[
    baseline_vs_canonical_metrics["label"] == "canonical_walkforward_baseline"
].iloc[0]
fullsignal_row = baseline_vs_canonical_metrics.loc[
    baseline_vs_canonical_metrics["label"] == "canonical_fullsignal_full_history"
].iloc[0]

reasons = []
if artifact_row["signal_source"] == baseline_signal_path:
    reasons.append("Signal source in 06 uses the baseline walk-forward snapshot, which matches the monthly retrain output source.")
if str(artifact_row["start_timestamp"])[:10] == START_DATE_DEFAULT:
    reasons.append("The 06 baseline artifact begins at the intended live start date 2021-01-01.")
if artifact_row["end_timestamp"] != canonical_row["end_timestamp"]:
    reasons.append(
        "The 06 baseline artifact is truncated to the overlap with full rebuild outputs instead of evaluating the baseline on its own full horizon."
    )
reasons.append(
    "The 06 notebook applies the signal on the same row as the realized return, while canonical V4 performance notebooks use the prior row signal for the next bar return."
)
reasons.append(
    "The 06 notebook computes volatility and Sharpe from hourly returns, while canonical V4 performance notebooks compute metrics from daily-resampled equity."
)
reasons.append(
    "The 06 notebook applies transaction cost multiplicatively, while canonical V4 performance notebooks subtract the switching cost directly from return."
)

equity_gap_vs_canonical = float(artifact_row["final_equity"] - canonical_row["final_equity"])
sharpe_gap_vs_canonical = float(artifact_row["sharpe_ratio"] - canonical_row["sharpe_ratio"])
equity_gap_vs_fullsignal = float(artifact_row["final_equity"] - fullsignal_row["final_equity"])

horizon_mismatch = artifact_row["end_timestamp"] != canonical_row["end_timestamp"]
major_metric_gap = (
    abs(equity_gap_vs_canonical) > initial_capital * 0.01
    or abs(sharpe_gap_vs_canonical) > 0.05
)

if horizon_mismatch or major_metric_gap:
    verdict = "parity_not_confirmed"
else:
    verdict = "parity_confirmed"

if abs(equity_gap_vs_fullsignal) > initial_capital * 0.05:
    reasons.append(
        "The 06 baseline result is also materially below the higher promoted-model full-signal framing used in the deep-dive performance notebooks."
    )

parity_confirmed = verdict == "parity_confirmed"
prior_hold_recommendation_provisional = not parity_confirmed

audit_lines = [
    "# Baseline Methodology Audit",
    "",
    "## Verdict",
    f"- Verdict: `{verdict}`",
    f"- Prior `hold current V4` decision should be treated as provisional: `{prior_hold_recommendation_provisional}`",
    "",
    "## Scope",
    f"- PnL notebook audited: `01_Processing/04_Retraining/06_PnL_Comparison_V4_vs_V4_1.py`",
    f"- Retrain notebook audited: `01_Processing/04_Retraining/01_Monthly_Walkforward_Retrain.py`",
    f"- Comparison pack audited: `{pnl_root}`",
    f"- Baseline metadata source: `{baseline_metadata_path}`",
    "",
    "## Checklist Findings",
    f"- Evaluation horizon in existing PnL artifact: `{artifact_first_ts}` -> `{artifact_last_ts}`",
    f"- Baseline signal source used by 06: `{baseline_signal_path}`",
    f"- Canonical walk-forward signal source: `{baseline_metadata.get('wf_signal_export_path')}`",
    f"- Position mapping in 06: `0->QQQ, 1->SPY, 2->CASH`",
    f"- Canonical performance mapping: `0->QQQ, 1->SPY, 2->CASH`",
    f"- 06 rebalance timing: `same-row signal applied to return_t_minus_1_to_t`",
    f"- Canonical rebalance timing: `prior-row signal applied to return_t_minus_1_to_t`",
    f"- 06 price source actually used in run: `{run_metadata.get('resolved_market_table')}`",
    f"- Canonical price source: `{market_table}`",
    f"- 06 transaction cost logic: `multiplicative`",
    f"- Canonical transaction cost logic: `linear return minus cost`",
    f"- 06 CAGR formula: calendar-day based on first/last timestamp, matching the requested formula.",
    "",
    "## Metrics Comparison",
    markdown_table(
        baseline_vs_canonical_metrics[
            [
                "label",
                "metric_basis",
                "start_timestamp",
                "end_timestamp",
                "final_equity",
                "cagr",
                "annualized_volatility",
                "sharpe_ratio",
                "max_drawdown",
            ]
        ]
    ),
    "",
    "## Reasons",
]
audit_lines.extend([f"- {reason}" for reason in reasons])
audit_lines.extend(
    [
        "",
        "## Smallest Safe Correction To 06",
        "- Use the baseline walk-forward signal series without intersecting to the rebuild horizon when auditing baseline parity.",
        "- Shift the applied regime by one bar so signal at `t-1` drives return from `t-1` to `t`.",
        "- Match canonical transaction cost logic and compute summary risk metrics from daily-resampled equity.",
        "",
        "## Output Files",
        f"- `baseline_vs_canonical_metrics.csv`: `{parity_root}/baseline_vs_canonical_metrics.csv`",
        f"- `signal_source_audit.csv`: `{parity_root}/signal_source_audit.csv`",
        f"- `timing_alignment_audit.csv`: `{parity_root}/timing_alignment_audit.csv`",
        f"- `hourly_return_audit_sample.csv`: `{parity_root}/hourly_return_audit_sample.csv`",
        f"- `switch_cost_audit_sample.csv`: `{parity_root}/switch_cost_audit_sample.csv`",
        f"- `parity_verdict.json`: `{parity_root}/parity_verdict.json`",
    ]
)

baseline_methodology_audit_path = f"{parity_root}/baseline_methodology_audit.md"
baseline_vs_canonical_metrics_path = f"{parity_root}/baseline_vs_canonical_metrics.csv"
signal_source_audit_path = f"{parity_root}/signal_source_audit.csv"
timing_alignment_audit_path = f"{parity_root}/timing_alignment_audit.csv"
hourly_return_audit_sample_path = f"{parity_root}/hourly_return_audit_sample.csv"
switch_cost_audit_sample_path = f"{parity_root}/switch_cost_audit_sample.csv"
parity_verdict_path = f"{parity_root}/parity_verdict.json"

baseline_vs_canonical_metrics.to_csv(baseline_vs_canonical_metrics_path, index=False)
signal_source_audit.to_csv(signal_source_audit_path, index=False)
timing_alignment_audit.to_csv(timing_alignment_audit_path, index=False)
hourly_return_audit_sample.to_csv(hourly_return_audit_sample_path, index=False)
switch_cost_audit_sample.to_csv(switch_cost_audit_sample_path, index=False)
write_text(baseline_methodology_audit_path, "\n".join(audit_lines) + "\n")

parity_verdict = {
    "verdict": verdict,
    "prior_hold_recommendation_provisional": prior_hold_recommendation_provisional,
    "comparison_root": comparison_root,
    "parity_root": parity_root,
    "artifact_first_timestamp": artifact_first_ts,
    "artifact_last_timestamp": artifact_last_ts,
    "baseline_signal_source": baseline_signal_path,
    "canonical_walkforward_signal_source": baseline_metadata.get("wf_signal_export_path"),
    "resolved_06_price_source": run_metadata.get("resolved_market_table"),
    "canonical_price_source": market_table,
    "equity_gap_vs_canonical_walkforward": equity_gap_vs_canonical,
    "sharpe_gap_vs_canonical_walkforward": sharpe_gap_vs_canonical,
    "equity_gap_vs_fullsignal_framing": equity_gap_vs_fullsignal,
    "reasons": reasons,
    "recommended_smallest_safe_correction": [
        "Remove full-rebuild overlap truncation when auditing baseline.",
        "Use prior-bar signal application.",
        "Use canonical daily-equity metrics for summary statistics.",
    ],
    "artifacts": {
        "baseline_methodology_audit": baseline_methodology_audit_path,
        "baseline_vs_canonical_metrics": baseline_vs_canonical_metrics_path,
        "signal_source_audit": signal_source_audit_path,
        "timing_alignment_audit": timing_alignment_audit_path,
        "hourly_return_audit_sample": hourly_return_audit_sample_path,
        "switch_cost_audit_sample": switch_cost_audit_sample_path,
    },
    "production_tables_modified": False,
    "model_registry_updated": False,
    "workflow_json_modified": False,
    "live_alias_changed": False,
}
write_json(parity_verdict_path, parity_verdict)

result = {
    "phase": "pnl_baseline_parity_check",
    "status": "success",
    "verdict": verdict,
    "prior_hold_recommendation_provisional": prior_hold_recommendation_provisional,
    "artifact_first_timestamp": artifact_first_ts,
    "artifact_last_timestamp": artifact_last_ts,
    "artifacts": {
        "baseline_methodology_audit": baseline_methodology_audit_path,
        "baseline_vs_canonical_metrics": baseline_vs_canonical_metrics_path,
        "signal_source_audit": signal_source_audit_path,
        "timing_alignment_audit": timing_alignment_audit_path,
        "hourly_return_audit_sample": hourly_return_audit_sample_path,
        "switch_cost_audit_sample": switch_cost_audit_sample_path,
        "parity_verdict": parity_verdict_path,
    },
}
print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

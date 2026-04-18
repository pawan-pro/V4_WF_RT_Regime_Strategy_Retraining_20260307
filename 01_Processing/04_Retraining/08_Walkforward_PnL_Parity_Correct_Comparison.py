# Databricks notebook source
# Walk-forward-only, parity-correct PnL engine for baseline, targeted-fix, and full-rebuild comparison

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

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data"
COMPARISON_ROOT = f"{VOLUME_ROOT}/comparison"

BASELINE_ROOT_DEFAULT = f"{COMPARISON_ROOT}/2026-04-18_full_rebuild/baseline"
TARGETED_FIX_ROOT_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-18_baseline_vs_corrected/corrected_retrain"
)
FULL_REBUILD_ROOT_DEFAULT = (
    f"{COMPARISON_ROOT}/2026-04-18_full_rebuild/full_rebuild_retrain"
)
PARITY_AUDIT_ROOT_DEFAULT = f"{COMPARISON_ROOT}/2026-04-18_pnl_parity_check"
OUTPUT_ROOT_DEFAULT = f"{COMPARISON_ROOT}/2026-04-18_walkforward_pnl_parity_correct"
PRICE_TABLE_DEFAULT = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"

START_DATE_DEFAULT = "2021-01-01"
INITIAL_CAPITAL_DEFAULT = 5_000_000
TRANSACTION_COST_DEFAULT = 0.0005
TRADING_DAYS = 252

POSITION_MAP = {0: "QQQ", 1: "SPY", 2: "CASH"}
REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
MODEL_LABELS = {
    "baseline": "Current V4",
    "targeted_fix": "Targeted Fix",
    "full_rebuild": "Full Rebuild V4.1",
}


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


def load_signal_frame(path_str, model_name, candidate_columns):
    pdf = load_csv(path_str)
    if pdf.empty:
        raise FileNotFoundError(f"Missing or empty signal file: {path_str}")

    pdf = normalize_timestamp(pdf)
    pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp")
    pdf = pdf.drop_duplicates(subset=["timestamp"], keep="last")

    chosen_column = next((col for col in candidate_columns if col in pdf.columns), None)
    if not chosen_column:
        raise KeyError(
            f"No usable signal column in {path_str}. candidates={candidate_columns}, columns={list(pdf.columns)}"
        )

    signal_pdf = pdf[["timestamp", chosen_column]].copy()
    predicted_regime = (
        pd.to_numeric(signal_pdf[chosen_column], errors="coerce").fillna(2).astype(int)
    )
    if chosen_column == "predicted_regime":
        signal_pdf["predicted_regime"] = predicted_regime
    else:
        signal_pdf["predicted_regime"] = predicted_regime
        signal_pdf = signal_pdf.drop(columns=[chosen_column])
    signal_pdf["model"] = model_name
    return signal_pdf


def signal_only_frame(pdf):
    if "predicted_regime" not in pdf.columns:
        raise KeyError(f"'predicted_regime' missing in columns: {list(pdf.columns)}")
    result = pd.DataFrame(
        {
            "timestamp": pdf["timestamp"].values,
            "predicted_regime": pd.to_numeric(
                pdf["predicted_regime"], errors="coerce"
            ).fillna(2).astype(int),
        }
    )
    return result.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )


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

    pivoted = (
        price_pdf.pivot_table(
            index="timestamp",
            columns="symbol",
            values="close",
            aggfunc="last",
        )
        .reset_index()
        .rename(columns={"SPY": "SPY_close", "QQQ": "QQQ_close"})
        .sort_values("timestamp")
    )

    pivoted["SPY_close"] = pd.to_numeric(pivoted["SPY_close"], errors="coerce").ffill()
    pivoted["QQQ_close"] = pd.to_numeric(pivoted["QQQ_close"], errors="coerce").ffill()
    return pivoted


def contiguous_run_lengths(series):
    if series.empty:
        return pd.Series(dtype=float)
    groups = series.ne(series.shift()).cumsum()
    return series.groupby(groups).size().astype(float)


def compute_daily_metrics(daily_equity, initial_capital):
    daily_equity = daily_equity.dropna()
    start_ts = daily_equity.index[0]
    end_ts = daily_equity.index[-1]
    years = max((end_ts - start_ts).days / 365.25, 1 / 365.25)

    daily_returns = daily_equity.pct_change().dropna()
    sigma = daily_returns.std()
    mean_return = daily_returns.mean()
    annualized_volatility = (
        sigma * np.sqrt(TRADING_DAYS) if pd.notna(sigma) else np.nan
    )
    sharpe_ratio = (
        (mean_return / sigma) * np.sqrt(TRADING_DAYS)
        if pd.notna(sigma) and sigma != 0
        else np.nan
    )

    rolling_peak = daily_equity.cummax()
    drawdown = daily_equity / rolling_peak - 1.0
    max_drawdown = float(-drawdown.min()) if not drawdown.empty else np.nan

    final_equity = float(daily_equity.iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    cagr = (final_equity / initial_capital) ** (1 / years) - 1 if years > 0 else np.nan
    calmar_ratio = cagr / max_drawdown if max_drawdown and max_drawdown > 0 else np.nan

    return {
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "daily_rows": int(len(daily_equity)),
        "days": int((end_ts - start_ts).days),
        "years": float(years),
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        "annualized_volatility": float(annualized_volatility)
        if np.isfinite(annualized_volatility)
        else np.nan,
        "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
        "max_drawdown": float(max_drawdown) if np.isfinite(max_drawdown) else np.nan,
        "calmar_ratio": float(calmar_ratio) if np.isfinite(calmar_ratio) else np.nan,
    }


def run_walkforward_engine(signal_pdf, price_pdf, model_name, initial_capital, transaction_cost):
    merged = signal_pdf.merge(
        price_pdf[["timestamp", "SPY_close", "QQQ_close"]],
        on="timestamp",
        how="left",
    ).sort_values("timestamp")

    merged["SPY_close"] = pd.to_numeric(merged["SPY_close"], errors="coerce").ffill()
    merged["QQQ_close"] = pd.to_numeric(merged["QQQ_close"], errors="coerce").ffill()
    merged = merged.dropna(subset=["SPY_close", "QQQ_close"]).reset_index(drop=True)

    current_equity = float(initial_capital)
    active_asset = "START"
    records = []

    for idx in range(len(merged)):
        row = merged.iloc[idx]
        record = {
            "timestamp": row["timestamp"],
            "model": model_name,
            "signal_regime_at_t": int(row["predicted_regime"]),
            "applied_regime": np.nan,
            "applied_regime_name": None,
            "position": "START" if idx == 0 else None,
            "SPY_close": row["SPY_close"],
            "QQQ_close": row["QQQ_close"],
            "SPY_return": np.nan,
            "QQQ_return": np.nan,
            "gross_return": 0.0,
            "transaction_cost_applied": 0.0,
            "net_return": 0.0,
            "equity": current_equity,
            "position_changed": False,
        }

        if idx > 0:
            # Apply the signal from t-1 to the return realized over bar t.
            prior_signal = int(merged.iloc[idx - 1]["predicted_regime"])
            target = POSITION_MAP.get(prior_signal, "CASH")
            spy_return = (row["SPY_close"] / merged.iloc[idx - 1]["SPY_close"]) - 1
            qqq_return = (row["QQQ_close"] / merged.iloc[idx - 1]["QQQ_close"]) - 1

            if target == "QQQ":
                gross_return = qqq_return
            elif target == "SPY":
                gross_return = spy_return
            else:
                gross_return = 0.0

            # Switching cost is modeled as return minus cost, not multiplicative slippage.
            transaction_cost_applied = transaction_cost if target != active_asset else 0.0
            net_return = gross_return - transaction_cost_applied
            current_equity *= (1.0 + net_return)

            record.update(
                {
                    "applied_regime": prior_signal,
                    "applied_regime_name": REGIME_NAME_MAP.get(prior_signal),
                    "position": target,
                    "SPY_return": spy_return,
                    "QQQ_return": qqq_return,
                    "gross_return": gross_return,
                    "transaction_cost_applied": transaction_cost_applied,
                    "net_return": net_return,
                    "equity": current_equity,
                    "position_changed": target != active_asset,
                }
            )
            active_asset = target

        records.append(record)

    hourly_curve = pd.DataFrame(records)
    valid_positions = hourly_curve.iloc[1:].copy()
    valid_positions["applied_regime"] = valid_positions["applied_regime"].astype(int)

    regime_stats = {
        "model": model_name,
        "label": MODEL_LABELS.get(model_name, model_name),
        "hourly_rows": int(len(hourly_curve)),
        "invested_rows": int(len(valid_positions)),
        "total_regime_switches": int(valid_positions["position_changed"].sum()),
        "average_holding_bars": float(
            contiguous_run_lengths(valid_positions["position"]).mean()
        ),
        "expansion_pct_time": float(valid_positions["applied_regime"].eq(0).mean()),
        "slowdown_pct_time": float(valid_positions["applied_regime"].eq(1).mean()),
        "recession_pct_time": float(valid_positions["applied_regime"].eq(2).mean()),
    }

    daily_equity = (
        hourly_curve.set_index("timestamp")["equity"].resample("D").last().dropna()
    )
    daily_curve = pd.DataFrame(
        {
            "timestamp": daily_equity.index,
            "model": model_name,
            "daily_equity": daily_equity.values,
        }
    )
    daily_curve["daily_drawdown"] = (
        daily_curve["daily_equity"] / daily_curve["daily_equity"].cummax() - 1.0
    )

    metrics = compute_daily_metrics(daily_equity, initial_capital)
    metrics.update(regime_stats)
    metrics["model"] = model_name
    metrics["label"] = MODEL_LABELS.get(model_name, model_name)
    metrics["signal_timing"] = "prior_bar"
    metrics["transaction_cost_logic"] = "return_minus_cost_on_switch"
    metrics["metric_basis"] = "daily_resampled_equity"

    return hourly_curve, daily_curve, metrics


def make_equity_plot(daily_curve_wide, output_path):
    plt.figure(figsize=(14, 6))
    for model_name in ["baseline", "targeted_fix", "full_rebuild"]:
        plt.plot(
            daily_curve_wide["timestamp"],
            daily_curve_wide[f"{model_name}_daily_equity"],
            label=MODEL_LABELS[model_name],
        )
    plt.title("Walk-Forward PnL Comparison")
    plt.xlabel("Date")
    plt.ylabel("Daily Equity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_drawdown_plot(drawdown_wide, output_path):
    plt.figure(figsize=(14, 6))
    for model_name in ["baseline", "targeted_fix", "full_rebuild"]:
        plt.plot(
            drawdown_wide["timestamp"],
            drawdown_wide[f"{model_name}_daily_drawdown"],
            label=MODEL_LABELS[model_name],
        )
    plt.title("Walk-Forward Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_regime_plot(hourly_curve_wide, output_path):
    plt.figure(figsize=(14, 4.5))
    for model_name in ["baseline", "targeted_fix", "full_rebuild"]:
        plt.step(
            hourly_curve_wide["timestamp"],
            hourly_curve_wide[f"{model_name}_applied_regime"],
            where="post",
            label=MODEL_LABELS[model_name],
        )
    plt.yticks([0, 1, 2], ["Expansion", "Slowdown", "Recession"])
    plt.title("Applied Regime Timeline")
    plt.xlabel("Timestamp")
    plt.ylabel("Regime")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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


def recommendation_from_common(metrics_pdf):
    common = metrics_pdf[metrics_pdf["scope"] == "common_overlap"].copy()
    baseline = common.loc[common["model"] == "baseline"].iloc[0]
    alternatives = common[common["model"] != "baseline"].copy()

    promotable = []
    for _, row in alternatives.iterrows():
        better_return = row["final_equity"] > baseline["final_equity"] and row["cagr"] > baseline["cagr"]
        better_risk = (
            row["sharpe_ratio"] >= baseline["sharpe_ratio"]
            and row["max_drawdown"] <= baseline["max_drawdown"] * 1.10
        )
        if better_return and better_risk:
            promotable.append(row)

    if promotable:
        promotable_pdf = pd.DataFrame(promotable).sort_values(
            ["final_equity", "sharpe_ratio"], ascending=False
        )
        best = promotable_pdf.iloc[0]
        if best["model"] == "targeted_fix":
            return "Promote targeted-fix version", (
                "Targeted-fix improves walk-forward return and meets the risk guardrails versus the baseline."
            )
        return "Promote full-rebuild version", (
            "Full rebuild improves walk-forward return and meets the risk guardrails versus the baseline."
        )

    all_underperform = (
        alternatives["final_equity"].le(baseline["final_equity"]).all()
        and alternatives["cagr"].le(baseline["cagr"]).all()
        and alternatives["sharpe_ratio"].le(baseline["sharpe_ratio"]).all()
    )
    if all_underperform:
        return "Hold current V4", (
            "Neither alternative beats the baseline on walk-forward return and Sharpe on the aligned comparison horizon."
        )

    return "Further review needed", (
        "At least one alternative improves part of the walk-forward profile, but the return and risk signals are mixed."
    )


baseline_root = BASELINE_ROOT_DEFAULT
targeted_fix_root = TARGETED_FIX_ROOT_DEFAULT
full_rebuild_root = FULL_REBUILD_ROOT_DEFAULT
parity_audit_root = PARITY_AUDIT_ROOT_DEFAULT
output_root = OUTPUT_ROOT_DEFAULT
price_table = PRICE_TABLE_DEFAULT
start_date = START_DATE_DEFAULT
initial_capital = INITIAL_CAPITAL_DEFAULT
transaction_cost = TRANSACTION_COST_DEFAULT

if "dbutils" in globals():
    dbutils.widgets.text("baseline_root", BASELINE_ROOT_DEFAULT)
    dbutils.widgets.text("targeted_fix_root", TARGETED_FIX_ROOT_DEFAULT)
    dbutils.widgets.text("full_rebuild_root", FULL_REBUILD_ROOT_DEFAULT)
    dbutils.widgets.text("parity_audit_root", PARITY_AUDIT_ROOT_DEFAULT)
    dbutils.widgets.text("output_root", OUTPUT_ROOT_DEFAULT)
    dbutils.widgets.text("price_table", PRICE_TABLE_DEFAULT)
    dbutils.widgets.text("start_date", START_DATE_DEFAULT)
    dbutils.widgets.text("initial_capital", str(INITIAL_CAPITAL_DEFAULT))
    dbutils.widgets.text("transaction_cost", str(TRANSACTION_COST_DEFAULT))

    baseline_root = dbutils.widgets.get("baseline_root").strip() or BASELINE_ROOT_DEFAULT
    targeted_fix_root = dbutils.widgets.get("targeted_fix_root").strip() or TARGETED_FIX_ROOT_DEFAULT
    full_rebuild_root = dbutils.widgets.get("full_rebuild_root").strip() or FULL_REBUILD_ROOT_DEFAULT
    parity_audit_root = dbutils.widgets.get("parity_audit_root").strip() or PARITY_AUDIT_ROOT_DEFAULT
    output_root = dbutils.widgets.get("output_root").strip() or OUTPUT_ROOT_DEFAULT
    price_table = dbutils.widgets.get("price_table").strip() or PRICE_TABLE_DEFAULT
    start_date = dbutils.widgets.get("start_date").strip() or START_DATE_DEFAULT
    initial_capital = float(
        dbutils.widgets.get("initial_capital").strip() or INITIAL_CAPITAL_DEFAULT
    )
    transaction_cost = float(
        dbutils.widgets.get("transaction_cost").strip() or TRANSACTION_COST_DEFAULT
    )

ensure_dir(output_root)

baseline_signal_path = f"{baseline_root}/current_walkforward_signals.csv"
targeted_fix_signal_path = f"{targeted_fix_root}/walkforward_signals.csv"
full_rebuild_signal_path = f"{full_rebuild_root}/walkforward_signals.csv"
prior_non_parity_metrics_path = (
    f"{COMPARISON_ROOT}/2026-04-18_full_rebuild/pnl_comparison/pnl_metrics.csv"
)
prior_parity_metrics_path = (
    f"{parity_audit_root}/baseline_vs_canonical_metrics.csv"
)

baseline_signal = load_signal_frame(
    baseline_signal_path,
    "baseline",
    ["predicted_regime", "model_predicted_regime"],
)
targeted_fix_signal = load_signal_frame(
    targeted_fix_signal_path,
    "targeted_fix",
    ["predicted_regime", "model_predicted_regime"],
)
full_rebuild_signal = load_signal_frame(
    full_rebuild_signal_path,
    "full_rebuild",
    ["predicted_regime", "model_predicted_regime"],
)

start_ts = pd.Timestamp(start_date)
baseline_signal = baseline_signal[baseline_signal["timestamp"] >= start_ts].copy()
targeted_fix_signal = targeted_fix_signal[targeted_fix_signal["timestamp"] >= start_ts].copy()
full_rebuild_signal = full_rebuild_signal[full_rebuild_signal["timestamp"] >= start_ts].copy()

price_frame = load_price_frame(price_table, start_date)

common_timestamps = pd.DataFrame(
    {
        "timestamp": sorted(
            set(baseline_signal["timestamp"])
            .intersection(set(targeted_fix_signal["timestamp"]))
            .intersection(set(full_rebuild_signal["timestamp"]))
        )
    }
)

if common_timestamps.empty:
    raise ValueError("No common timestamps found across baseline, targeted-fix, and full-rebuild walk-forward signals.")

baseline_common = signal_only_frame(
    baseline_signal[baseline_signal["timestamp"].isin(common_timestamps["timestamp"])]
)
targeted_fix_common = signal_only_frame(
    targeted_fix_signal[
        targeted_fix_signal["timestamp"].isin(common_timestamps["timestamp"])
    ]
)
full_rebuild_common = signal_only_frame(
    full_rebuild_signal[
        full_rebuild_signal["timestamp"].isin(common_timestamps["timestamp"])
    ]
)

baseline_hourly_common, baseline_daily_common, baseline_metrics_common = run_walkforward_engine(
    baseline_common.copy(),
    price_frame.copy(),
    "baseline",
    initial_capital,
    transaction_cost,
)
targeted_fix_hourly_common, targeted_fix_daily_common, targeted_fix_metrics_common = run_walkforward_engine(
    targeted_fix_common.copy(),
    price_frame.copy(),
    "targeted_fix",
    initial_capital,
    transaction_cost,
)
full_rebuild_hourly_common, full_rebuild_daily_common, full_rebuild_metrics_common = run_walkforward_engine(
    full_rebuild_common.copy(),
    price_frame.copy(),
    "full_rebuild",
    initial_capital,
    transaction_cost,
)

baseline_hourly_full, baseline_daily_full, baseline_metrics_full = run_walkforward_engine(
    signal_only_frame(baseline_signal),
    price_frame.copy(),
    "baseline",
    initial_capital,
    transaction_cost,
)

baseline_metrics_common["scope"] = "common_overlap"
targeted_fix_metrics_common["scope"] = "common_overlap"
full_rebuild_metrics_common["scope"] = "common_overlap"
baseline_metrics_full["scope"] = "baseline_parity_full_horizon"

pnl_metrics = pd.DataFrame(
    [
        baseline_metrics_common,
        targeted_fix_metrics_common,
        full_rebuild_metrics_common,
        baseline_metrics_full,
    ]
)

regime_stats = pnl_metrics[
    [
        "scope",
        "model",
        "label",
        "hourly_rows",
        "invested_rows",
        "total_regime_switches",
        "average_holding_bars",
        "expansion_pct_time",
        "slowdown_pct_time",
        "recession_pct_time",
    ]
].copy()

hourly_equity_curve = baseline_hourly_common[
    [
        "timestamp",
        "signal_regime_at_t",
        "applied_regime",
        "applied_regime_name",
        "position",
        "equity",
        "net_return",
        "transaction_cost_applied",
    ]
].rename(
    columns={
        "signal_regime_at_t": "baseline_signal_regime_at_t",
        "applied_regime": "baseline_applied_regime",
        "applied_regime_name": "baseline_applied_regime_name",
        "position": "baseline_position",
        "equity": "baseline_equity",
        "net_return": "baseline_net_return",
        "transaction_cost_applied": "baseline_transaction_cost",
    }
)
hourly_equity_curve = hourly_equity_curve.merge(
    targeted_fix_hourly_common[
        [
            "timestamp",
            "signal_regime_at_t",
            "applied_regime",
            "applied_regime_name",
            "position",
            "equity",
            "net_return",
            "transaction_cost_applied",
        ]
    ].rename(
        columns={
            "signal_regime_at_t": "targeted_fix_signal_regime_at_t",
            "applied_regime": "targeted_fix_applied_regime",
            "applied_regime_name": "targeted_fix_applied_regime_name",
            "position": "targeted_fix_position",
            "equity": "targeted_fix_equity",
            "net_return": "targeted_fix_net_return",
            "transaction_cost_applied": "targeted_fix_transaction_cost",
        }
    ),
    on="timestamp",
    how="inner",
)
hourly_equity_curve = hourly_equity_curve.merge(
    full_rebuild_hourly_common[
        [
            "timestamp",
            "signal_regime_at_t",
            "applied_regime",
            "applied_regime_name",
            "position",
            "equity",
            "net_return",
            "transaction_cost_applied",
        ]
    ].rename(
        columns={
            "signal_regime_at_t": "full_rebuild_signal_regime_at_t",
            "applied_regime": "full_rebuild_applied_regime",
            "applied_regime_name": "full_rebuild_applied_regime_name",
            "position": "full_rebuild_position",
            "equity": "full_rebuild_equity",
            "net_return": "full_rebuild_net_return",
            "transaction_cost_applied": "full_rebuild_transaction_cost",
        }
    ),
    on="timestamp",
    how="inner",
)

daily_drawdown_curve = baseline_daily_common[["timestamp", "daily_equity", "daily_drawdown"]].rename(
    columns={
        "daily_equity": "baseline_daily_equity",
        "daily_drawdown": "baseline_daily_drawdown",
    }
)
daily_drawdown_curve = daily_drawdown_curve.merge(
    targeted_fix_daily_common[["timestamp", "daily_equity", "daily_drawdown"]].rename(
        columns={
            "daily_equity": "targeted_fix_daily_equity",
            "daily_drawdown": "targeted_fix_daily_drawdown",
        }
    ),
    on="timestamp",
    how="inner",
)
daily_drawdown_curve = daily_drawdown_curve.merge(
    full_rebuild_daily_common[["timestamp", "daily_equity", "daily_drawdown"]].rename(
        columns={
            "daily_equity": "full_rebuild_daily_equity",
            "daily_drawdown": "full_rebuild_daily_drawdown",
        }
    ),
    on="timestamp",
    how="inner",
)

prior_non_parity_metrics = load_csv(prior_non_parity_metrics_path)
prior_non_parity_baseline = (
    prior_non_parity_metrics.loc[prior_non_parity_metrics["model"] == "baseline"].iloc[0].to_dict()
    if not prior_non_parity_metrics.empty
    else {}
)

prior_parity_metrics = load_csv(prior_parity_metrics_path)
canonical_walkforward_row = (
    prior_parity_metrics.loc[
        prior_parity_metrics["label"] == "canonical_walkforward_baseline"
    ].iloc[0].to_dict()
    if not prior_parity_metrics.empty
    else {}
)

baseline_parity_validation = {
    "old_non_parity_final_equity": prior_non_parity_baseline.get("final_equity"),
    "old_non_parity_cagr": prior_non_parity_baseline.get("cagr"),
    "old_non_parity_sharpe": prior_non_parity_baseline.get("sharpe_ratio"),
    "corrected_full_horizon_final_equity": baseline_metrics_full["final_equity"],
    "corrected_full_horizon_cagr": baseline_metrics_full["cagr"],
    "corrected_full_horizon_sharpe": baseline_metrics_full["sharpe_ratio"],
    "canonical_walkforward_final_equity": canonical_walkforward_row.get("final_equity"),
    "canonical_walkforward_cagr": canonical_walkforward_row.get("cagr"),
    "canonical_walkforward_sharpe": canonical_walkforward_row.get("sharpe_ratio"),
}
baseline_parity_validation["equity_gap_vs_canonical"] = (
    baseline_metrics_full["final_equity"] - canonical_walkforward_row.get("final_equity", np.nan)
)
baseline_parity_validation["cagr_gap_vs_canonical"] = (
    baseline_metrics_full["cagr"] - canonical_walkforward_row.get("cagr", np.nan)
)
baseline_parity_validation["sharpe_gap_vs_canonical"] = (
    baseline_metrics_full["sharpe_ratio"] - canonical_walkforward_row.get("sharpe_ratio", np.nan)
)
baseline_parity_validation["materially_closer_than_old"] = bool(
    abs(baseline_parity_validation["cagr_gap_vs_canonical"])
    < abs((prior_non_parity_baseline.get("cagr") or 0) - (canonical_walkforward_row.get("cagr") or 0))
)

recommendation, recommendation_reason = recommendation_from_common(pnl_metrics)

methodology_notes = [
    "# Methodology Notes",
    "",
    "- Scope: walk-forward only.",
    "- Investment start date: `2021-01-01`.",
    "- Initial capital: `5,000,000`.",
    "- Regime mapping: `0 -> QQQ`, `1 -> SPY`, `2 -> CASH`.",
    "- Signal timing: prior-bar application only.",
    "- Transaction cost logic: subtract cost from return on position switch.",
    "- Metrics basis: daily-resampled equity only.",
    "- Price source: `pawan_quant_research_1.v4.bronze_bt_market_1h`.",
    "- Out of scope: full-dataset/full-signal framing and the 50%+ CAGR analysis.",
]

comparison_metrics_view = pnl_metrics[pnl_metrics["scope"] == "common_overlap"][
    [
        "model",
        "label",
        "start_timestamp",
        "end_timestamp",
        "final_equity",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "total_regime_switches",
        "average_holding_bars",
    ]
].copy()

summary_lines = [
    "# Walk-Forward PnL Comparison",
    "",
    "## Scope",
    "- Walk-forward only.",
    "- Full-dataset/full-signal framing is explicitly out of scope.",
    "",
    "## Methodology",
    f"- Start date: `{start_date}`",
    f"- Initial capital: `{initial_capital}`",
    f"- Price source: `{price_table}`",
    "- Signal timing: `prior_bar`",
    "- Transaction cost logic: `return_minus_cost_on_switch`",
    "- Metrics basis: `daily_resampled_equity`",
    f"- Common comparison horizon: `{common_timestamps['timestamp'].min()}` -> `{common_timestamps['timestamp'].max()}`",
    "",
    "## Baseline Parity Validation",
    f"- Old non-parity baseline CAGR: `{baseline_parity_validation['old_non_parity_cagr']}`",
    f"- Corrected baseline full-horizon CAGR: `{baseline_parity_validation['corrected_full_horizon_cagr']}`",
    f"- Canonical walk-forward CAGR reference: `{baseline_parity_validation['canonical_walkforward_cagr']}`",
    f"- Corrected baseline materially closer than old result: `{baseline_parity_validation['materially_closer_than_old']}`",
    f"- Corrected baseline equity gap vs canonical: `{baseline_parity_validation['equity_gap_vs_canonical']}`",
    f"- Corrected baseline Sharpe gap vs canonical: `{baseline_parity_validation['sharpe_gap_vs_canonical']}`",
    "",
    "## Common Overlap Metrics",
    markdown_table(comparison_metrics_view),
    "",
    "## Regime Stats",
    markdown_table(
        regime_stats[regime_stats["scope"] == "common_overlap"][
            [
                "model",
                "total_regime_switches",
                "average_holding_bars",
                "expansion_pct_time",
                "slowdown_pct_time",
                "recession_pct_time",
            ]
        ]
    ),
    "",
    "## Recommendation",
    f"- Decision: `{recommendation}`",
    f"- Reason: `{recommendation_reason}`",
]

methodology_notes_path = f"{output_root}/methodology_notes.md"
pnl_metrics_path = f"{output_root}/pnl_metrics.csv"
equity_curve_path = f"{output_root}/equity_curve.csv"
drawdown_curve_path = f"{output_root}/drawdown_curve.csv"
regime_stats_path = f"{output_root}/regime_stats.csv"
summary_path = f"{output_root}/summary.md"
run_metadata_path = f"{output_root}/run_metadata.json"
equity_plot_path = f"{output_root}/equity_curve_comparison.png"
drawdown_plot_path = f"{output_root}/drawdown_comparison.png"
regime_plot_path = f"{output_root}/regime_timeline.png"

write_text(methodology_notes_path, "\n".join(methodology_notes) + "\n")
pnl_metrics.to_csv(pnl_metrics_path, index=False)
hourly_equity_curve.to_csv(equity_curve_path, index=False)
daily_drawdown_curve.to_csv(drawdown_curve_path, index=False)
regime_stats.to_csv(regime_stats_path, index=False)
write_text(summary_path, "\n".join(summary_lines) + "\n")

make_equity_plot(daily_drawdown_curve, equity_plot_path)
make_drawdown_plot(daily_drawdown_curve, drawdown_plot_path)
make_regime_plot(
    hourly_equity_curve[
        [
            "timestamp",
            "baseline_applied_regime",
            "targeted_fix_applied_regime",
            "full_rebuild_applied_regime",
        ]
    ],
    regime_plot_path,
)

run_metadata = {
    "phase": "walkforward_pnl_parity_correct",
    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "walkforward_only",
    "baseline_root": baseline_root,
    "targeted_fix_root": targeted_fix_root,
    "full_rebuild_root": full_rebuild_root,
    "parity_audit_root": parity_audit_root,
    "output_root": output_root,
    "price_source_table": price_table,
    "baseline_signal_source": baseline_signal_path,
    "targeted_fix_signal_source": targeted_fix_signal_path,
    "full_rebuild_signal_source": full_rebuild_signal_path,
    "start_date": start_date,
    "initial_capital": initial_capital,
    "transaction_cost": transaction_cost,
    "signal_timing": "prior_bar",
    "transaction_cost_logic": "return_minus_cost_on_switch",
    "metric_basis": "daily_resampled_equity",
    "common_horizon_start": common_timestamps["timestamp"].min(),
    "common_horizon_end": common_timestamps["timestamp"].max(),
    "common_signal_rows": int(len(common_timestamps)),
    "baseline_parity_validation": baseline_parity_validation,
    "recommendation": recommendation,
    "recommendation_reason": recommendation_reason,
    "artifacts": {
        "methodology_notes": methodology_notes_path,
        "pnl_metrics": pnl_metrics_path,
        "equity_curve": equity_curve_path,
        "drawdown_curve": drawdown_curve_path,
        "regime_stats": regime_stats_path,
        "summary": summary_path,
        "equity_plot": equity_plot_path,
        "drawdown_plot": drawdown_plot_path,
        "regime_plot": regime_plot_path,
    },
    "production_tables_modified": False,
    "model_registry_updated": False,
    "workflow_json_modified": False,
    "live_alias_changed": False,
}
write_json(run_metadata_path, run_metadata)

result = {
    "phase": "walkforward_pnl_parity_correct",
    "status": "success",
    "recommendation": recommendation,
    "recommendation_reason": recommendation_reason,
    "common_horizon_start": common_timestamps["timestamp"].min(),
    "common_horizon_end": common_timestamps["timestamp"].max(),
    "baseline_parity_validation": baseline_parity_validation,
    "artifacts": run_metadata["artifacts"],
}
print(json.dumps(result, indent=2, default=str))

if "dbutils" in globals():
    dbutils.notebook.exit(json.dumps(result, default=str))

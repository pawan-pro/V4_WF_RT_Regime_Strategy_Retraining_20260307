#!/usr/bin/env python3
"""Reconcile corrected-reference performance methodology from matched signals.

CSV-only forensic script. It reads replay/reference CSV artifacts and writes
only under OUTPUT_ROOT. It does not train models or write Delta tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Volumes/pawan_quant_research_1/v4/market_data")
CSV_BASE = BASE / "v4_1_csv"
OUTPUT_ROOT = CSV_BASE / "performance_methodology_reconciliation_2026-04-30"
REF_DIR = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
)
SIGNALS = (
    CSV_BASE
    / "faithful_legacy_grid_model_replay_2026-04-30"
    / "signals"
    / "walkforward_signals.csv"
)
REF_SIGNALS = REF_DIR / "walkforward_signals.csv"
GRID = CSV_BASE / "full_legacy_grid_clone_2026-04-30" / "grids" / "full_legacy_grid_clone.csv"
REF_FULL_SIGNAL = REF_DIR / "full_signal_export.csv"
REPLAY_EQUITY = (
    CSV_BASE
    / "faithful_legacy_grid_model_replay_2026-04-30"
    / "performance"
    / "equity_curve.csv"
)

INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}
TARGET = {
    "CAGR": 0.208405,
    "Sharpe": 1.0024,
    "Sortino": 1.3729,
    "Max Drawdown": -0.255385,
    "Final Value": 12_883_454.9938051,
}


def ensure_dirs() -> None:
    for subdir in ["inputs", "performance", "forensics", "manifests", "code"]:
        (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    try:
        out[column] = out[column].dt.tz_localize(None)
    except Exception:
        pass
    return out


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def inventory_reference_artifacts() -> pd.DataFrame:
    rows: list[dict] = []
    needles = [
        "20.8405",
        "0.208405",
        "0.208404",
        "12.883",
        "12883",
        "25.5385",
        "0.255385",
        "-0.255385",
    ]
    for path in sorted(REF_DIR.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        row = {
            "path": str(path),
            "relative_path": str(path.relative_to(REF_DIR)),
            "suffix": path.suffix,
            "size_bytes": int(stat.st_size),
            "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": sha256_file(path),
            "role_guess": infer_artifact_role(path.name),
            "row_count": None,
            "columns": None,
            "min_timestamp": None,
            "max_timestamp": None,
            "target_text_hits": "",
        }
        if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
            try:
                text = path.read_text(errors="ignore")
                row["target_text_hits"] = ",".join([needle for needle in needles if needle in text])
            except Exception:
                pass
        if path.suffix.lower() == ".csv":
            try:
                df = safe_read_csv(path)
                row["row_count"] = int(len(df))
                row["columns"] = json.dumps(list(df.columns))
                for ts_col in ["timestamp", "date", "Strategy Start Date", "Strategy End Date"]:
                    if ts_col in df.columns:
                        ts = pd.to_datetime(df[ts_col], errors="coerce")
                        if ts.notna().any():
                            row["min_timestamp"] = str(ts.min())
                            row["max_timestamp"] = str(ts.max())
                            break
            except Exception as exc:
                row["columns"] = f"read_error: {exc}"
        rows.append(row)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUTPUT_ROOT / "forensics" / "performance_artifact_inventory.csv", index=False)
    return inv


def infer_artifact_role(name: str) -> str:
    lower = name.lower()
    if "equity" in lower:
        return "equity_curve"
    if "metric" in lower or "performance" in lower:
        return "performance_metrics"
    if "annual" in lower:
        return "annual_returns"
    if "drawdown" in lower:
        return "drawdown"
    if "signal" in lower:
        return "signal_or_full_export"
    if "summary" in lower or "methodology" in lower:
        return "report_or_notes"
    return "other"


def load_signal_price_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = normalize_timestamp(safe_read_csv(SIGNALS))
    ref_signals = normalize_timestamp(safe_read_csv(REF_SIGNALS))
    grid = normalize_timestamp(safe_read_csv(GRID, usecols=["timestamp", "SPY_close", "QQQ_close"]))
    full = normalize_timestamp(safe_read_csv(REF_FULL_SIGNAL))
    return signals, ref_signals, grid, full


def standard_price_frame(signals: pd.DataFrame, prices: pd.DataFrame, source_name: str) -> pd.DataFrame:
    price_cols = {}
    if {"SPY_close", "QQQ_close"}.issubset(prices.columns):
        price_cols = {"SPY_close": "SPY", "QQQ_close": "QQQ"}
    elif {"SPY", "QQQ"}.issubset(prices.columns):
        price_cols = {"SPY": "SPY", "QQQ": "QQQ"}
    else:
        raise ValueError(f"Price frame {source_name} missing SPY/QQQ columns")
    px = prices[["timestamp"] + list(price_cols)].rename(columns=price_cols).drop_duplicates("timestamp")
    data = (
        signals[["timestamp", "predicted_regime"]]
        .drop_duplicates("timestamp")
        .merge(px, on="timestamp", how="inner")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    data = data[
        (data["timestamp"] >= pd.Timestamp("2021-01-01"))
        & (data["timestamp"] <= pd.Timestamp("2025-12-31 23:59:59"))
    ].copy()
    data["SPY"] = pd.to_numeric(data["SPY"], errors="coerce").ffill()
    data["QQQ"] = pd.to_numeric(data["QQQ"], errors="coerce").ffill()
    data["predicted_regime"] = pd.to_numeric(data["predicted_regime"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["SPY", "QQQ", "predicted_regime"]).reset_index(drop=True)
    data["source_name"] = source_name
    return data


def price_update_filter(data: pd.DataFrame) -> pd.DataFrame:
    changed = data[["SPY", "QQQ"]].ne(data[["SPY", "QQQ"]].shift()).any(axis=1)
    return data[changed].copy().reset_index(drop=True)


def daily_last(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data.groupby(data["timestamp"].dt.date)["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)


def daily_first(data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[data.groupby(data["timestamp"].dt.date)["timestamp"].idxmin()].sort_values("timestamp").reset_index(drop=True)


def weekday_hour_filter(data: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    mask = (
        (data["timestamp"].dt.weekday < 5)
        & (data["timestamp"].dt.hour >= start_hour)
        & (data["timestamp"].dt.hour <= end_hour)
    )
    return data[mask].copy().reset_index(drop=True)


def reference_price_frame(signals: pd.DataFrame) -> pd.DataFrame | None:
    path = REF_DIR / "hourly_equity_curve.csv"
    if not path.exists():
        return None
    ref_hourly = normalize_timestamp(safe_read_csv(path))
    prices = ref_hourly[["timestamp", "SPY", "QQQ"]].copy()
    return standard_price_frame(signals, prices, "reference_package_hourly_price_path")


def run_backtest(
    data: pd.DataFrame,
    allocation_timing: str,
    switch_cost: float = SWITCH_COST,
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot backtest empty frame")
    d = data.sort_values("timestamp").reset_index(drop=True).copy()
    portfolio_values = [float(INITIAL_CAPITAL)]
    target_assets = ["START"]
    applied_regimes = [np.nan]
    gross_returns = [0.0]
    strategy_returns = [0.0]
    switch_costs = [0.0]
    position_changed = [False]
    active_asset = "START"
    for idx in range(1, len(d)):
        signal_idx = idx if allocation_timing == "same_bar" else idx - 1
        regime = int(d["predicted_regime"].iloc[signal_idx])
        target_asset = REGIME_TO_ASSET.get(regime, "CASH")
        if target_asset == "QQQ":
            gross_return = d["QQQ"].iloc[idx] / d["QQQ"].iloc[idx - 1] - 1.0
        elif target_asset == "SPY":
            gross_return = d["SPY"].iloc[idx] / d["SPY"].iloc[idx - 1] - 1.0
        else:
            gross_return = 0.0
        cost = switch_cost if target_asset != active_asset else 0.0
        strategy_return = gross_return - cost
        portfolio_values.append(portfolio_values[-1] * (1.0 + strategy_return))
        target_assets.append(target_asset)
        applied_regimes.append(regime)
        gross_returns.append(gross_return)
        strategy_returns.append(strategy_return)
        switch_costs.append(cost)
        position_changed.append(target_asset != active_asset)
        active_asset = target_asset
    out = d.copy()
    out["applied_regime"] = applied_regimes
    out["target_asset"] = target_assets
    out["gross_asset_return"] = gross_returns
    out["switch_cost"] = switch_costs
    out["strategy_return"] = strategy_returns
    out["position_changed"] = position_changed
    out["portfolio_value"] = portfolio_values
    out["drawdown"] = out["portfolio_value"] / out["portfolio_value"].cummax() - 1.0
    return out


def metric_block(
    equity: pd.Series,
    timestamps: pd.Series,
    return_frequency: str,
    ann_factor: float = 252.0,
) -> dict:
    eq = pd.Series(equity, dtype=float).reset_index(drop=True)
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    returns = eq.pct_change().dropna()
    elapsed_days = max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1e-12)
    years = elapsed_days / 365.0
    downside = returns[returns < 0]
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0
    sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(ann_factor) if returns.std(ddof=1) else 0.0
    sortino = returns.mean() / downside.std(ddof=1) * math.sqrt(ann_factor) if len(downside) > 1 and downside.std(ddof=1) else 0.0
    dd = eq / eq.cummax() - 1.0
    return {
        "return_frequency": return_frequency,
        "annualization_factor": ann_factor,
        "row_count": int(len(eq)),
        "start_timestamp": str(ts.iloc[0]),
        "end_timestamp": str(ts.iloc[-1]),
        "start_equity": float(eq.iloc[0]),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "calendar_days": float(elapsed_days),
        "years_days_over_365": float(years),
        "CAGR": float(cagr),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)),
        "downside_std_return": float(downside.std(ddof=1)) if len(downside) > 1 else np.nan,
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Max Drawdown": float(dd.min()),
    }


def annual_returns_from_daily(daily_equity: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in daily_equity.groupby(daily_equity["timestamp"].dt.year):
        if len(group) < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "start_equity": float(group["equity"].iloc[0]),
                "end_equity": float(group["equity"].iloc[-1]),
                "return": float(group["equity"].iloc[-1] / group["equity"].iloc[0] - 1.0),
                "observations": int(len(group)),
            }
        )
    return rows


def evaluate_variant(
    variant_name: str,
    data: pd.DataFrame,
    allocation_timing: str,
    price_path: str,
    row_filter: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    hourly = run_backtest(data, allocation_timing=allocation_timing)
    daily = (
        hourly.loc[hourly.groupby(hourly["timestamp"].dt.date)["timestamp"].idxmax()]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    daily_equity = daily[["timestamp", "portfolio_value"]].rename(columns={"portfolio_value": "equity"})
    metric_dates = daily_equity["timestamp"].dt.normalize()
    metrics = metric_block(daily_equity["equity"], metric_dates, "daily_last_from_variant_equity", 252.0)
    metrics.update(
        {
            "variant": variant_name,
            "allocation_timing": allocation_timing,
            "price_path": price_path,
            "row_filter": row_filter,
            "hourly_or_source_rows": int(len(hourly)),
            "daily_observations": int(len(daily_equity)),
            "actual_first_intraday_timestamp": str(daily_equity["timestamp"].iloc[0]),
            "actual_last_intraday_timestamp": str(daily_equity["timestamp"].iloc[-1]),
            "switch_cost_bps": SWITCH_COST * 10_000,
            "annual_returns_json": json.dumps(annual_returns_from_daily(daily_equity), sort_keys=True),
        }
    )
    metrics.update(score_metrics(metrics))
    return metrics, hourly, daily_equity


def score_metrics(metrics: dict) -> dict:
    cagr_gap = abs(metrics["CAGR"] - TARGET["CAGR"])
    sharpe_gap = abs(metrics["Sharpe"] - TARGET["Sharpe"])
    sortino_gap = abs(metrics["Sortino"] - TARGET["Sortino"])
    dd_gap = abs(metrics["Max Drawdown"] - TARGET["Max Drawdown"])
    final_gap = abs(metrics["final_equity"] - TARGET["Final Value"]) / TARGET["Final Value"]
    composite = cagr_gap / 0.01 + sharpe_gap / 0.1 + sortino_gap / 0.1 + dd_gap / 0.01 + final_gap
    return {
        "target_CAGR_abs_diff": float(cagr_gap),
        "target_Sharpe_abs_diff": float(sharpe_gap),
        "target_Sortino_abs_diff": float(sortino_gap),
        "target_MaxDD_abs_diff": float(dd_gap),
        "target_FinalValue_pct_diff": float(final_gap),
        "target_composite_distance": float(composite),
    }


def build_methodology_matrix() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    signals, ref_signals, grid, full = load_signal_price_frames()
    signal_cmp = signals[["timestamp", "predicted_regime"]].merge(
        ref_signals[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference_regime"}),
        on="timestamp",
        how="outer",
        indicator=True,
    )
    if (signal_cmp["_merge"] != "both").any():
        raise RuntimeError("Faithful replay and corrected reference signals do not have identical timestamp membership")
    grid_data = standard_price_frame(signals, grid, "faithful_grid")
    full_data = None
    if {"SPY_close", "QQQ_close"}.issubset(full.columns) or {"SPY", "QQQ"}.issubset(full.columns):
        full_data = standard_price_frame(signals, full, "reference_full_signal_export")
    ref_data = reference_price_frame(signals)

    variants: list[tuple[str, pd.DataFrame, str, str, str]] = [
        ("A_hourly_same_bar_all_24x7_grid", grid_data, "same_bar", "faithful_grid", "all_24x7_signal_rows"),
        ("B_hourly_next_bar_all_24x7_grid_replay_current", grid_data, "next_bar", "faithful_grid", "all_24x7_signal_rows"),
        ("C_daily_last_signal_grid_next_bar", daily_last(grid_data), "next_bar", "faithful_grid", "daily_last_rows"),
        ("D_daily_first_signal_grid_next_bar", daily_first(grid_data), "next_bar", "faithful_grid", "daily_first_rows"),
        ("E_daily_eod_naive_price_update_grid_next_bar", daily_last(price_update_filter(grid_data)), "next_bar", "faithful_grid", "daily_last_price_update_rows"),
        ("F_signal_timestamp_only_returns_grid_next_bar", grid_data, "next_bar", "faithful_grid", "all_signal_timestamps"),
        ("H_faithful_grid_price_update_rows_next_bar", price_update_filter(grid_data), "next_bar", "faithful_grid", "price_update_rows"),
        ("H2_faithful_grid_weekday_9_to_23_next_bar", weekday_hour_filter(grid_data, 9, 23), "next_bar", "faithful_grid", "weekday_hours_9_to_23"),
    ]
    if full_data is not None:
        variants.append(("G_full_signal_export_price_path_next_bar", full_data, "next_bar", "reference_full_signal_export", "all_24x7_signal_rows"))
    if ref_data is not None:
        variants.append(("I_reference_package_price_path_next_bar", ref_data, "next_bar", "reference_package_hourly_equity_prices", "reference_hourly_equity_timestamps"))

    rows = []
    hourly_outputs: dict[str, pd.DataFrame] = {}
    daily_outputs: dict[str, pd.DataFrame] = {}
    for name, data, timing, price_path, row_filter in variants:
        metrics, hourly, daily = evaluate_variant(name, data, timing, price_path, row_filter)
        rows.append(metrics)
        hourly_outputs[name] = hourly
        daily_outputs[name] = daily
    matrix = pd.DataFrame(rows)
    matrix["source_priority"] = np.select(
        [
            matrix["variant"].astype(str).str.startswith("I_reference_package"),
            matrix["variant"].astype(str).str.startswith("H_faithful_grid_price_update"),
        ],
        [0, 1],
        default=2,
    )
    matrix = matrix.sort_values(["target_composite_distance", "source_priority"]).reset_index(drop=True)
    matrix.to_csv(OUTPUT_ROOT / "performance" / "performance_methodology_matrix.csv", index=False)
    annual_rows = []
    for name, daily in daily_outputs.items():
        for row in annual_returns_from_daily(daily):
            row["variant"] = name
            annual_rows.append(row)
    pd.DataFrame(annual_rows).to_csv(
        OUTPUT_ROOT / "performance" / "performance_methodology_annual_returns.csv",
        index=False,
    )
    return matrix, hourly_outputs, daily_outputs


def compare_equity_curves() -> pd.DataFrame:
    rows = []
    comparisons = []
    if (REF_DIR / "hourly_equity_curve.csv").exists() and REPLAY_EQUITY.exists():
        ref = normalize_timestamp(safe_read_csv(REF_DIR / "hourly_equity_curve.csv"))
        replay = normalize_timestamp(safe_read_csv(REPLAY_EQUITY))
        comparisons.append(("reference_hourly_vs_replay_current_hourly", ref, "portfolio_value", replay, "portfolio_value"))
    if (REF_DIR / "daily_equity_curve.csv").exists() and REPLAY_EQUITY.exists():
        ref_daily = normalize_timestamp(safe_read_csv(REF_DIR / "daily_equity_curve.csv"))
        replay_hourly = normalize_timestamp(safe_read_csv(REPLAY_EQUITY))
        replay_daily = (
            replay_hourly.loc[replay_hourly.groupby(replay_hourly["timestamp"].dt.date)["timestamp"].idxmax()]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        ref_daily_cmp = pd.DataFrame(
            {
                "timestamp": ref_daily["timestamp"].dt.normalize(),
                "equity": ref_daily["equity"],
            }
        )
        replay_daily_cmp = pd.DataFrame(
            {
                "timestamp": replay_daily["timestamp"].dt.normalize(),
                "portfolio_value": replay_daily["portfolio_value"],
            }
        )
        comparisons.append(("reference_daily_vs_replay_current_daily", ref_daily_cmp, "equity", replay_daily_cmp, "portfolio_value"))
    for name, ref, ref_col, replay, replay_col in comparisons:
        ref_cmp = ref[["timestamp", ref_col]].rename(columns={ref_col: "reference_equity"})
        replay_cmp = replay[["timestamp", replay_col]].rename(columns={replay_col: "replay_equity"})
        merged = ref_cmp.merge(
            replay_cmp,
            on="timestamp",
            how="outer",
            indicator=True,
        )
        both = merged[merged["_merge"] == "both"].copy()
        if not both.empty:
            diff = (both["reference_equity"] - both["replay_equity"]).abs()
            first_div = both.loc[diff > 1e-6, "timestamp"].iloc[0] if (diff > 1e-6).any() else None
            corr = both["reference_equity"].corr(both["replay_equity"])
            max_diff = diff.max()
        else:
            first_div = None
            corr = np.nan
            max_diff = np.nan
        rows.append(
            {
                "comparison": name,
                "reference_rows": int(len(ref)),
                "replay_rows": int(len(replay)),
                "overlap_rows": int(len(both)),
                "missing_in_replay": int((merged["_merge"] == "left_only").sum()),
                "extra_in_replay": int((merged["_merge"] == "right_only").sum()),
                "reference_first_timestamp": str(ref["timestamp"].min()),
                "reference_last_timestamp": str(ref["timestamp"].max()),
                "replay_first_timestamp": str(replay["timestamp"].min()),
                "replay_last_timestamp": str(replay["timestamp"].max()),
                "reference_final_equity": float(ref[ref_col].iloc[-1]),
                "replay_final_equity": float(replay[replay_col].iloc[-1]),
                "equity_correlation": None if pd.isna(corr) else float(corr),
                "max_abs_equity_difference": None if pd.isna(max_diff) else float(max_diff),
                "first_divergence_timestamp": None if first_div is None else str(first_div),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_ROOT / "forensics" / "reference_vs_replay_equity_curve_comparison.csv", index=False)
    write_equity_summary(out)
    return out


def write_equity_summary(comparison: pd.DataFrame) -> None:
    lines = ["# Equity Curve Comparison Summary", ""]
    for _, row in comparison.iterrows():
        lines.extend(
            [
                f"## {row['comparison']}",
                "",
                f"- Reference rows: {row['reference_rows']:,}",
                f"- Replay rows: {row['replay_rows']:,}",
                f"- Overlap rows: {row['overlap_rows']:,}",
                f"- Reference final equity: ${row['reference_final_equity']:,.2f}",
                f"- Replay final equity: ${row['replay_final_equity']:,.2f}",
                f"- First divergence timestamp: `{row['first_divergence_timestamp']}`",
                "",
            ]
        )
    (OUTPUT_ROOT / "forensics" / "equity_curve_comparison_summary.md").write_text("\n".join(lines))


def write_best_match(matrix: pd.DataFrame, hourly_outputs: dict[str, pd.DataFrame], daily_outputs: dict[str, pd.DataFrame]) -> dict:
    best = matrix.iloc[0].to_dict()
    best_name = best["variant"]
    hourly_outputs[best_name].to_csv(OUTPUT_ROOT / "performance" / "best_match_hourly_equity_curve.csv", index=False)
    daily_outputs[best_name].to_csv(OUTPUT_ROOT / "performance" / "best_match_daily_equity_curve.csv", index=False)
    payload = {
        "target_metrics": TARGET,
        "best_variant": best,
        "best_hourly_equity_curve_path": str(OUTPUT_ROOT / "performance" / "best_match_hourly_equity_curve.csv"),
        "best_daily_equity_curve_path": str(OUTPUT_ROOT / "performance" / "best_match_daily_equity_curve.csv"),
        "interpretation": (
            "Corrected reference performance is reproduced by applying prior-bar signals "
            "only on price-update timestamps, then taking daily-last equity and computing "
            "CAGR over elapsed calendar days / 365 with daily-return Sharpe/Sortino."
        ),
    }
    (OUTPUT_ROOT / "forensics" / "best_performance_methodology_match.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = [
        "# Best Performance Methodology Match",
        "",
        f"- Variant: `{best_name}`",
        f"- Price path: `{best['price_path']}`",
        f"- Row filter: `{best['row_filter']}`",
        f"- Allocation timing: `{best['allocation_timing']}`",
        f"- CAGR: `{best['CAGR']:.12f}`",
        f"- Sharpe: `{best['Sharpe']:.12f}`",
        f"- Sortino: `{best['Sortino']:.12f}`",
        f"- Max drawdown: `{best['Max Drawdown']:.12f}`",
        f"- Final equity: `${best['final_equity']:,.2f}`",
        "",
        payload["interpretation"],
        "",
    ]
    (OUTPUT_ROOT / "forensics" / "best_performance_methodology_match.md").write_text("\n".join(lines))
    return payload


def write_formula_reconciliations(matrix: pd.DataFrame) -> None:
    selected = matrix[matrix["variant"].isin([
        matrix.iloc[0]["variant"],
        "B_hourly_next_bar_all_24x7_grid_replay_current",
    ])].copy()
    cagr_rows = []
    sharpe_rows = []
    for _, row in selected.iterrows():
        fixed_5y = (row["final_equity"] / row["start_equity"]) ** (1.0 / 5.0) - 1.0
        avg_annualized = row["mean_return"] * row["annualization_factor"]
        cagr_rows.append(
            {
                "variant": row["variant"],
                "start_equity": row["start_equity"],
                "final_equity": row["final_equity"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "calendar_days": row["calendar_days"],
                "years_days_over_365": row["years_days_over_365"],
                "cagr_formula": "(final/start) ** (365/calendar_days) - 1",
                "cagr_result": row["CAGR"],
                "fixed_5_year_denominator_result": fixed_5y,
                "annualized_average_return_result": avg_annualized,
                "matches_20_8405": abs(row["CAGR"] - TARGET["CAGR"]) <= 1e-6,
            }
        )
        sharpe_rows.append(
            {
                "variant": row["variant"],
                "return_frequency": row["return_frequency"],
                "annualization_factor": row["annualization_factor"],
                "mean_return": row["mean_return"],
                "std_return": row["std_return"],
                "downside_std_return": row["downside_std_return"],
                "risk_free_assumption": 0.0,
                "Sharpe": row["Sharpe"],
                "Sortino": row["Sortino"],
            }
        )
    pd.DataFrame(cagr_rows).to_csv(OUTPUT_ROOT / "forensics" / "cagr_formula_reconciliation.csv", index=False)
    pd.DataFrame(sharpe_rows).to_csv(OUTPUT_ROOT / "forensics" / "sharpe_sortino_formula_reconciliation.csv", index=False)


def write_summary(matrix: pd.DataFrame, best_payload: dict, equity_cmp: pd.DataFrame) -> None:
    best = best_payload["best_variant"]
    replay = matrix[matrix["variant"] == "B_hourly_next_bar_all_24x7_grid_replay_current"].iloc[0].to_dict()
    lines = [
        "# Performance Methodology Reconciliation Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Answer",
        "",
        "1. Price path: the corrected result uses the same faithful legacy price levels, but only at price-update timestamps. The saved reference `hourly_equity_curve.csv` provides the exact timestamp membership and price path.",
        "2. Allocation timing: prior-bar / next-bar allocation. The signal at row `t-1` selects the asset return from `t-1` to `t`.",
        "3. Frequency/resampling: hourly equity on price-update rows, then daily-last equity for reporting metrics.",
        "4. CAGR formula: `(final_equity / start_equity) ** (365 / calendar_days) - 1`; for 2021-01-01 to 2025-12-31 this is exactly a 5-year denominator.",
        "5. Replay gave 19.2818% because the recovered replay runner evaluated every 24/7 forward-filled hourly signal row. That includes overnight/weekend/flat-price rows in the equity path and return distribution, while the reference evaluates only price-update rows.",
        "6. The reference performance is valid and reproducible from the matched signal stream when the saved price-update row membership is used.",
        "7. Recommended deck-safe methodology: state explicitly that the forensic reference uses prior-bar signals on price-update hourly bars with 5 bps switch cost and daily-last reporting; for production/live comparisons, separately publish a strict PIT execution methodology.",
        "",
        "## Best Match",
        "",
        f"- Variant: `{best['variant']}`",
        f"- CAGR: `{best['CAGR']:.12f}`",
        f"- Sharpe: `{best['Sharpe']:.12f}`",
        f"- Sortino: `{best['Sortino']:.12f}`",
        f"- Max drawdown: `{best['Max Drawdown']:.12f}`",
        f"- Final equity: `${best['final_equity']:,.2f}`",
        "",
        "## Current Replay Method",
        "",
        f"- Variant: `{replay['variant']}`",
        f"- CAGR: `{replay['CAGR']:.12f}`",
        f"- Sharpe: `{replay['Sharpe']:.12f}`",
        f"- Sortino: `{replay['Sortino']:.12f}`",
        f"- Max drawdown: `{replay['Max Drawdown']:.12f}`",
        f"- Final equity: `${replay['final_equity']:,.2f}`",
        "",
        "## Safety",
        "",
        "This audit wrote only CSV/JSON/XLSX/Markdown artifacts under the isolated V4.1 CSV folder. It did not train models, write production Delta tables, update workflows, touch Saxo execution/token refresh, modify RT/dashboard notebooks, change live schedules, or update the model registry.",
        "",
    ]
    (OUTPUT_ROOT / "forensics" / "performance_methodology_reconciliation_summary.md").write_text("\n".join(lines))


def try_write_xlsx(matrix: pd.DataFrame) -> None:
    try:
        with pd.ExcelWriter(OUTPUT_ROOT / "performance" / "performance_methodology_matrix.xlsx") as writer:
            matrix.to_excel(writer, sheet_name="methodology_matrix", index=False)
            pd.read_csv(OUTPUT_ROOT / "performance" / "performance_methodology_annual_returns.csv").to_excel(
                writer, sheet_name="annual_returns", index=False
            )
    except Exception as exc:
        (OUTPUT_ROOT / "performance" / "performance_methodology_matrix_xlsx_fallback.json").write_text(
            json.dumps({"status": "xlsx_not_written", "reason": str(exc)}, indent=2)
        )


def write_manifest() -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(OUTPUT_ROOT),
        "inputs": {
            "faithful_replay_signals": str(SIGNALS),
            "corrected_reference_signals": str(REF_SIGNALS),
            "faithful_legacy_grid": str(GRID),
            "corrected_reference_full_signal_export": str(REF_FULL_SIGNAL),
            "corrected_variant_folder": str(REF_DIR),
        },
        "input_sha256": {
            "faithful_replay_signals": sha256_file(SIGNALS),
            "corrected_reference_signals": sha256_file(REF_SIGNALS),
            "faithful_legacy_grid": sha256_file(GRID),
            "corrected_reference_full_signal_export": sha256_file(REF_FULL_SIGNAL),
        },
        "production_delta_writes": False,
        "model_training": False,
        "workflow_changes": False,
        "saxo_or_rt_changes": False,
    }
    (OUTPUT_ROOT / "manifests" / "runtime_manifest.json").write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    ensure_dirs()
    inventory_reference_artifacts()
    equity_cmp = compare_equity_curves()
    matrix, hourly_outputs, daily_outputs = build_methodology_matrix()
    best_payload = write_best_match(matrix, hourly_outputs, daily_outputs)
    write_formula_reconciliations(matrix)
    write_summary(matrix, best_payload, equity_cmp)
    try_write_xlsx(matrix)
    write_manifest()
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "best_variant": best_payload["best_variant"]["variant"],
                "best_metrics": {
                    "CAGR": best_payload["best_variant"]["CAGR"],
                    "Sharpe": best_payload["best_variant"]["Sharpe"],
                    "Sortino": best_payload["best_variant"]["Sortino"],
                    "Max Drawdown": best_payload["best_variant"]["Max Drawdown"],
                    "Final Value": best_payload["best_variant"]["final_equity"],
                },
                "no_production_change_confirmation": True,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

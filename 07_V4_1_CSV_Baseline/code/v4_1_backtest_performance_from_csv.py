"""Backtest V4.1 CSV-only walk-forward signals and write audit artifacts."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v4_1_common import (
    INITIAL_CAPITAL,
    REFERENCE_METRICS,
    SWITCH_COST,
    TIMING_VARIANTS,
    V4_1_ROOT,
    compute_metric_block,
    ensure_v4_1_dirs,
    market_open_mask_utc,
    parse_utc,
    print_json,
    write_csv,
    write_json,
)


REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REGIME_TARGET = {0: "QQQ", 1: "SPY", 2: "CASH"}


def annual_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, grp in daily.groupby(pd.to_datetime(daily["ny_market_date"]).dt.year, sort=True):
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(grp["strategy_equity"].iloc[-1] / grp["strategy_equity"].iloc[0] - 1.0),
                "spy_return": float(grp["spy_equity"].iloc[-1] / grp["spy_equity"].iloc[0] - 1.0),
                "observations": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def trailing_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    end = pd.Timestamp(daily["timestamp"].iloc[-1])
    for years in [1, 3, 5]:
        window = daily[daily["timestamp"] >= end - pd.DateOffset(years=years)].copy()
        if len(window) < 2:
            continue
        strategy = compute_metric_block(window.rename(columns={"strategy_equity": "equity"}), "equity", "strategy_daily_return", 252.0)
        spy = compute_metric_block(window.rename(columns={"spy_equity": "equity"}), "equity", "spy_daily_return", 252.0)
        rows.append(
            {
                "window_years": years,
                "strategy_cagr": strategy["cagr"],
                "strategy_sharpe": strategy["sharpe"],
                "strategy_sortino": strategy["sortino"],
                "strategy_max_drawdown": strategy["max_drawdown"],
                "spy_cagr": spy["cagr"],
                "spy_sharpe": spy["sharpe"],
                "spy_sortino": spy["sortino"],
                "spy_max_drawdown": spy["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def backtest_variant(root: Path, variant: str) -> dict:
    grid = pd.read_csv(root / "grids" / variant / "v4_1_silver_grid.csv")
    signals = pd.read_csv(root / "signals" / variant / "v4_1_walkforward_signals.csv")
    full = pd.read_csv(root / "signals" / variant / "v4_1_full_signal_export.csv")
    grid["timestamp"] = parse_utc(grid["timestamp"])
    signals["timestamp"] = parse_utc(signals["timestamp"])
    full["timestamp"] = parse_utc(full["timestamp"])

    data = (
        signals[["timestamp", "predicted_regime"]]
        .drop_duplicates("timestamp")
        .merge(grid[["timestamp", "SPY_close", "QQQ_close"]].drop_duplicates("timestamp"), on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    data = data[(data["timestamp"] >= pd.Timestamp("2021-01-01")) & (data["timestamp"] <= pd.Timestamp("2025-12-31 23:59:59"))].copy()
    data = data[market_open_mask_utc(data["timestamp"])].reset_index(drop=True)
    for col in ["SPY_close", "QQQ_close"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").ffill()
    data["predicted_regime"] = pd.to_numeric(data["predicted_regime"], errors="coerce").astype("Int64")
    data = data.dropna(subset=["SPY_close", "QQQ_close", "predicted_regime"]).reset_index(drop=True)
    if data.empty:
        raise RuntimeError(f"No market-open backtest rows for {variant}")

    data["regime_name"] = data["predicted_regime"].astype(int).map(REGIME_NAME)
    data["target_asset"] = data["predicted_regime"].astype(int).map(REGIME_TARGET)
    data["spy_return"] = data["SPY_close"].pct_change().fillna(0.0)
    data["qqq_return"] = data["QQQ_close"].pct_change().fillna(0.0)
    data["trade_regime"] = data["predicted_regime"].shift(1).fillna(data["predicted_regime"]).astype(int)
    data["trade_asset"] = data["trade_regime"].map(REGIME_TARGET)
    data["gross_strategy_return"] = np.select(
        [data["trade_regime"] == 0, data["trade_regime"] == 1, data["trade_regime"] == 2],
        [data["qqq_return"], data["spy_return"], 0.0],
        default=0.0,
    )
    switched = data["trade_regime"] != data["trade_regime"].shift(1)
    switched.iloc[0] = False
    data["switch_cost"] = np.where(switched, SWITCH_COST, 0.0)
    data["strategy_return"] = data["gross_strategy_return"] - data["switch_cost"]
    data["strategy_equity"] = INITIAL_CAPITAL * (1.0 + data["strategy_return"]).cumprod()
    data["spy_equity"] = INITIAL_CAPITAL * (1.0 + data["spy_return"]).cumprod()
    data["strategy_drawdown"] = data["strategy_equity"] / data["strategy_equity"].cummax() - 1.0
    data["spy_drawdown"] = data["spy_equity"] / data["spy_equity"].cummax() - 1.0
    data["ny_market_date"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)

    daily = data.loc[data.groupby("ny_market_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)

    hourly_strategy = compute_metric_block(data.rename(columns={"strategy_equity": "equity"}), "equity", "strategy_return", 252.0 * 7.0)
    hourly_spy = compute_metric_block(data.rename(columns={"spy_equity": "equity"}), "equity", "spy_return", 252.0 * 7.0)
    daily_strategy = compute_metric_block(daily.rename(columns={"strategy_equity": "equity"}), "equity", "strategy_daily_return", 252.0)
    daily_spy = compute_metric_block(daily.rename(columns={"spy_equity": "equity"}), "equity", "spy_daily_return", 252.0)

    annual = annual_returns(daily)
    trailing = trailing_metrics(daily)
    out_dir = root / "performance" / variant
    write_csv(data, out_dir / "v4_1_equity_curve.csv", root)
    write_csv(annual, out_dir / "v4_1_annual_returns.csv", root)
    write_csv(trailing, out_dir / "v4_1_trailing_metrics.csv", root)

    reference_diff = {
        "cagr_abs_diff": abs(daily_strategy["cagr"] - REFERENCE_METRICS["cagr"]),
        "max_drawdown_abs_diff": abs(daily_strategy["max_drawdown"] - REFERENCE_METRICS["max_drawdown"]),
        "final_equity_abs_diff": abs(daily_strategy["final_equity"] - REFERENCE_METRICS["final_equity"]),
        "period_matches_reference": daily_strategy["start_date"] == REFERENCE_METRICS["start_date"] and daily_strategy["end_date"] == REFERENCE_METRICS["end_date"],
    }
    signal_gap_check = {
        "signal_rows": int(len(signals)),
        "full_signal_rows": int(len(full)),
        "backtest_rows": int(len(data)),
        "daily_rows": int(len(daily)),
        "unexpected_signal_gaps_gt_3h": int((signals["timestamp"].sort_values().diff() > pd.Timedelta(hours=3)).sum()),
    }
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "methodology": {
            "execution_semantics": "prior market-open bar signal selects next market-open bar return",
            "regime_mapping": {"0": "QQQ", "1": "SPY", "2": "CASH"},
            "initial_capital": INITIAL_CAPITAL,
            "switch_cost": SWITCH_COST,
            "cagr_basis": "elapsed calendar days / 365",
            "reference_compatible": "market-open hourly returns with 252*7 risk annualization plus NY daily EOD audit",
            "strict_ny_daily_eod": "last available regular-hours timestamp per New York market date",
        },
        "hourly_market_open_metrics": {"strategy": hourly_strategy, "spy": hourly_spy},
        "daily_eod_metrics": {"strategy": daily_strategy, "spy": daily_spy},
        "annual_returns": annual.to_dict(orient="records"),
        "trailing_metrics": trailing.to_dict(orient="records"),
        "reference_20_8405_check": reference_diff,
        "signal_gap_check": signal_gap_check,
        "production_delta_aliases_modified": False,
    }
    write_json(metrics, out_dir / "v4_1_metrics.json", root)
    write_json(
        {
            "variant": variant,
            "daily_eod_metrics": metrics["daily_eod_metrics"],
            "annual_returns": metrics["annual_returns"],
            "trailing_metrics": metrics["trailing_metrics"],
            "reference_20_8405_check": reference_diff,
        },
        out_dir / "v4_1_slide_metrics.json",
        root,
    )
    final_xlsx = out_dir / "v4_1_audit_pack.xlsx"
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_xlsx = Path(tmp.name)
    with pd.ExcelWriter(tmp_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([daily_strategy]).to_excel(writer, sheet_name="daily_strategy_metrics", index=False)
        pd.DataFrame([daily_spy]).to_excel(writer, sheet_name="daily_spy_metrics", index=False)
        annual.to_excel(writer, sheet_name="annual_returns", index=False)
        trailing.to_excel(writer, sheet_name="trailing_metrics", index=False)
        data.tail(5000).to_excel(writer, sheet_name="recent_equity_curve", index=False)
    shutil.copy2(tmp_xlsx, final_xlsx)
    tmp_xlsx.unlink(missing_ok=True)
    return metrics


def write_sensitivity(root: Path, results: dict[str, dict]) -> dict:
    rows = []
    for variant in TIMING_VARIANTS:
        if variant not in results:
            continue
        metrics = results[variant]["daily_eod_metrics"]["strategy"]
        annual = pd.DataFrame(results[variant]["annual_returns"])
        row = {
            "variant": variant,
            "CAGR": metrics["cagr"],
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "final_equity": metrics["final_equity"],
            "notes_on_timing_realism": "main realistic candidate" if variant == TIMING_VARIANTS[0] else "timing sensitivity variant",
        }
        for year in range(2021, 2026):
            match = annual[annual["year"] == year]
            row[f"annual_return_{year}"] = None if match.empty else float(match["strategy_return"].iloc[0])
        rows.append(row)
    frame = pd.DataFrame(rows)
    write_csv(frame, root / "performance" / "v4_1_timing_sensitivity_comparison.csv", root)
    final_xlsx = root / "performance" / "v4_1_timing_sensitivity_comparison.xlsx"
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_xlsx = Path(tmp.name)
    with pd.ExcelWriter(tmp_xlsx, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="timing_sensitivity", index=False)
    shutil.copy2(tmp_xlsx, final_xlsx)
    tmp_xlsx.unlink(missing_ok=True)

    main = frame[frame["variant"] == TIMING_VARIANTS[0]]
    interpretation = "Main variant not available"
    if not main.empty:
        cagr = float(main["CAGR"].iloc[0])
        if abs(cagr - REFERENCE_METRICS["cagr"]) <= 0.01:
            interpretation = "Main realistic timing remains within about 1 percentage point CAGR of the corrected-ALFRED reference; robust candidate pending RT dry-run."
        else:
            best = frame.iloc[(frame["CAGR"] - REFERENCE_METRICS["cagr"]).abs().argsort()].iloc[0]
            interpretation = f"Main timing differs materially from reference; closest variant is {best['variant']}. Treat this as timing sensitivity risk."
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": REFERENCE_METRICS,
        "rows": frame.to_dict(orient="records"),
        "interpretation": interpretation,
    }
    write_json(payload, root / "performance" / "v4_1_timing_sensitivity_comparison.json", root)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(V4_1_ROOT))
    parser.add_argument("--variant", action="append", choices=TIMING_VARIANTS)
    args = parser.parse_args()
    root = ensure_v4_1_dirs(args.root)
    results = {variant: backtest_variant(root, variant) for variant in (args.variant or TIMING_VARIANTS)}
    sensitivity = write_sensitivity(root, results)
    print_json({"root": root, "results": results, "sensitivity": sensitivity})


if __name__ == "__main__":
    main()

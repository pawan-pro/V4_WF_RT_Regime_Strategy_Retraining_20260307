# Databricks notebook source
"""Generate V4.1 presentation support CSVs for MUB-21.

Research/support only. Reads reconciled V4.1 forensic CSV artifacts and writes
only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/presentation_support_2026-05-05/`

No production Delta tables, Saxo execution, token refresh, realtime notebooks,
dashboard notebooks, workflows, schedules, model registry, or trading logic are
modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE = Path("/Volumes/pawan_quant_research_1/v4/market_data")
OUTPUT_ROOT = BASE / "v4_1_csv" / "presentation_support_2026-05-05"
SIGNAL_DIR = OUTPUT_ROOT / "signal_files"

PERF_RECON_ROOT = BASE / "v4_1_csv" / "performance_methodology_reconciliation_2026-04-30"
SAME_DATA_ROOT = BASE / "v4_2_same_data_teacher_student_2026-05-04"
LINEAGE_ROOT = BASE / "v4_2_b_vs_c_lineage_diff_2026-05-04"
REF_ROOT = BASE / "comparison" / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected" / "corrected_variant"

HOURLY_EQUITY_PATH = PERF_RECON_ROOT / "performance" / "best_match_hourly_equity_curve.csv"
DAILY_EQUITY_PATH = PERF_RECON_ROOT / "performance" / "best_match_daily_equity_curve.csv"
BEST_MATCH_PATH = PERF_RECON_ROOT / "forensics" / "best_performance_methodology_match.json"
VARIANT_SUMMARY_PATH = SAME_DATA_ROOT / "performance" / "variant_summary.csv"
LINEAGE_LADDER_PATH = LINEAGE_ROOT / "forensics" / "reproducibility_ladder.csv"
REF_FULL_SIGNAL_PATH = REF_ROOT / "full_signal_export.csv"
REF_WALKFORWARD_PATH = REF_ROOT / "walkforward_signals.csv"

INITIAL_CAPITAL = 5_000_000.0
SOURCE_VARIANT = "B_C2_teacher_smoothed_baseline"
GOVERNANCE_CAVEAT = (
    "Teacher-smoothed full-sample regime research baseline. Reproducible from code and "
    "faithful legacy master data. Not strict PIT production evidence. Current benchmark "
    "until a better model engine is validated."
)
METHODOLOGY = (
    "prior-bar / next-bar allocation; price-update hourly rows only; 5 bps switch cost; "
    "daily-last equity reporting; CAGR over exactly 5.0 years from 2021-01-01 to "
    "2025-12-31; daily returns with 252 annualization; risk-free 0"
)

TARGETS = {
    "CAGR": 0.20840487336126445,
    "Sharpe": 1.0023589403061952,
    "Sortino": 1.3729057801036,
    "Max DD": -0.2553851434504447,
    "Final equity": 12_883_454.9938051,
    "Switches": 296,
}

REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}


def assert_output(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing to write outside presentation output root: {path}")


def ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, SIGNAL_DIR]:
        assert_output(path)
        path.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    assert_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(payload: dict[str, Any], path: Path) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return None if math.isnan(float(obj)) else float(obj)
        return obj

    assert_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    row = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() and path.is_file() else None,
        "row_count": None,
        "min_timestamp": None,
        "max_timestamp": None,
        "columns": None,
    }
    if path.exists() and path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        row["row_count"] = len(df)
        row["columns"] = json.dumps(list(df.columns))
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            row["min_timestamp"] = ts.min()
            row["max_timestamp"] = ts.max()
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"], errors="coerce")
            row["min_timestamp"] = row["min_timestamp"] or dt.min()
            row["max_timestamp"] = row["max_timestamp"] or dt.max()
    return row


def normalize_timestamp(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    try:
        out[col] = out[col].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=[col]).sort_values(col).reset_index(drop=True)


def load_hourly() -> pd.DataFrame:
    hourly = normalize_timestamp(pd.read_csv(HOURLY_EQUITY_PATH))
    required = ["timestamp", "predicted_regime", "SPY", "QQQ", "strategy_return", "portfolio_value", "drawdown"]
    missing = [col for col in required if col not in hourly.columns]
    if missing:
        raise ValueError(f"Missing required columns in hourly equity curve: {missing}")
    for col in ["predicted_regime", "SPY", "QQQ", "strategy_return", "switch_cost", "portfolio_value", "drawdown"]:
        if col in hourly.columns:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce")
    hourly["date"] = hourly["timestamp"].dt.date.astype(str)
    return hourly


def build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    daily = hourly.loc[hourly.groupby("date")["timestamp"].idxmax()].copy().sort_values("timestamp").reset_index(drop=True)
    daily["strategy_equity"] = daily["portfolio_value"]
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["SPY"].pct_change().fillna(0.0)
    daily["spy_benchmark_equity"] = INITIAL_CAPITAL * (1.0 + daily["spy_daily_return"]).cumprod()
    daily["excess_daily_return"] = daily["strategy_daily_return"] - daily["spy_daily_return"]
    daily["drawdown"] = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    daily["spy_drawdown"] = daily["spy_benchmark_equity"] / daily["spy_benchmark_equity"].cummax() - 1.0
    daily["regime_int"] = pd.to_numeric(daily["predicted_regime"], errors="coerce").astype("Int64")
    daily["regime_name"] = daily["regime_int"].map(REGIME_NAME)
    daily["position"] = daily.get("target_asset", pd.Series(index=daily.index, dtype=object)).fillna("")
    daily["source_variant"] = SOURCE_VARIANT
    cols = [
        "date",
        "strategy_equity",
        "spy_benchmark_equity",
        "strategy_daily_return",
        "spy_daily_return",
        "excess_daily_return",
        "drawdown",
        "spy_drawdown",
        "regime_int",
        "regime_name",
        "position",
        "source_variant",
    ]
    return daily[cols]


def period_table(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    d = daily.copy()
    d["date_dt"] = pd.to_datetime(d["date"])
    if period == "weekly":
        iso = d["date_dt"].dt.isocalendar()
        d["_period"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    elif period == "monthly":
        d["_period"] = d["date_dt"].dt.to_period("M").astype(str)
    elif period == "quarterly":
        d["_period"] = d["date_dt"].dt.to_period("Q").astype(str)
    elif period == "half_yearly":
        d["_period"] = d["date_dt"].dt.year.astype(str) + "-H" + np.where(d["date_dt"].dt.month <= 6, "1", "2")
    elif period == "yearly":
        d["_period"] = d["date_dt"].dt.year.astype(str)
    else:
        raise ValueError(period)

    rows = []
    for key, g in d.groupby("_period", sort=True):
        g = g.sort_values("date_dt")
        row = {
            "strategy_return": float((1.0 + g["strategy_daily_return"]).prod() - 1.0),
            "spy_return": float((1.0 + g["spy_daily_return"]).prod() - 1.0),
            "excess_return": float((1.0 + g["strategy_daily_return"]).prod() - (1.0 + g["spy_daily_return"]).prod()),
            "ending_equity": float(g["strategy_equity"].iloc[-1]),
            "ending_drawdown": float(g["drawdown"].iloc[-1]),
            "source_variant": SOURCE_VARIANT,
        }
        if period == "weekly":
            row.update({"week_start": g["date"].iloc[0], "week_end": g["date"].iloc[-1]})
        elif period == "monthly":
            row.update({"month": key})
        elif period == "quarterly":
            row.update({"year": int(g["date_dt"].dt.year.iloc[-1]), "quarter": f"Q{int(g['date_dt'].dt.quarter.iloc[-1])}"})
        elif period == "half_yearly":
            row.update({"year": int(g["date_dt"].dt.year.iloc[-1]), "half": f"H{1 if int(g['date_dt'].dt.month.iloc[-1]) <= 6 else 2}"})
        elif period == "yearly":
            row.update({"year": int(key)})
        rows.append(row)
    out = pd.DataFrame(rows)
    if period == "weekly":
        return out[["week_start", "week_end", "strategy_return", "spy_return", "excess_return", "ending_equity", "ending_drawdown", "source_variant"]]
    if period == "monthly":
        return out[["month", "strategy_return", "spy_return", "excess_return", "ending_equity", "ending_drawdown", "source_variant"]]
    if period == "quarterly":
        return out[["year", "quarter", "strategy_return", "spy_return", "excess_return", "ending_equity", "ending_drawdown", "source_variant"]]
    if period == "half_yearly":
        return out[["year", "half", "strategy_return", "spy_return", "excess_return", "ending_equity", "ending_drawdown", "source_variant"]]
    return out[["year", "strategy_return", "spy_return", "excess_return", "ending_equity", "ending_drawdown", "source_variant"]]


def build_trades(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy().sort_values("timestamp").reset_index(drop=True)
    changed = h["position_changed"].astype(str).str.lower().eq("true") if "position_changed" in h.columns else h["target_asset"].ne(h["target_asset"].shift())
    trades = h[changed].copy()
    trades["trade_timestamp"] = trades["timestamp"]
    trades["prior_regime"] = h["applied_regime"].shift(1).reindex(trades.index)
    trades["new_regime"] = trades.get("applied_regime", trades["predicted_regime"])
    trades["prior_position"] = h.get("target_asset", pd.Series(index=h.index, dtype=object)).shift(1).reindex(trades.index).fillna("START")
    trades["new_position"] = trades.get("target_asset", pd.Series(index=trades.index, dtype=object)).fillna("")
    trades["cost"] = trades.get("switch_cost", pd.Series(index=trades.index, dtype=float)).fillna(0.0)
    trades["equity_before"] = h["portfolio_value"].shift(1).reindex(trades.index)
    trades["equity_after"] = trades["portfolio_value"]
    trades["source_variant"] = SOURCE_VARIANT
    return trades[
        [
            "trade_timestamp",
            "prior_regime",
            "new_regime",
            "prior_position",
            "new_position",
            "cost",
            "equity_before",
            "equity_after",
            "source_variant",
        ]
    ].reset_index(drop=True)


def metrics_from_daily(daily: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float]:
    ret = daily["strategy_daily_return"].iloc[1:]
    downside = ret[ret < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1])
    return {
        "CAGR": (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": float(ret.mean() / ret.std(ddof=1) * math.sqrt(252.0)),
        "Sortino": float(ret.mean() / downside.std(ddof=1) * math.sqrt(252.0)),
        "Max DD": float(daily["drawdown"].min()),
        "Final equity": final_equity,
        "Switches": int(len(trades)),
    }


def build_summary_metrics(metrics: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"metric": "CAGR", "exact_value": metrics["CAGR"], "deck_value": "20.8%", "target_value": TARGETS["CAGR"]},
        {"metric": "Sharpe", "exact_value": metrics["Sharpe"], "deck_value": "1.00", "target_value": TARGETS["Sharpe"]},
        {"metric": "Sortino", "exact_value": metrics["Sortino"], "deck_value": "1.37", "target_value": TARGETS["Sortino"]},
        {"metric": "Max DD", "exact_value": metrics["Max DD"], "deck_value": "-25.5%", "target_value": TARGETS["Max DD"]},
        {"metric": "Final equity", "exact_value": metrics["Final equity"], "deck_value": "$12.88M", "target_value": TARGETS["Final equity"]},
        {"metric": "Switch count", "exact_value": metrics["Switches"], "deck_value": "296", "target_value": TARGETS["Switches"]},
        {"metric": "Start date", "exact_value": "2021-01-01", "deck_value": "2021-01-01", "target_value": "2021-01-01"},
        {"metric": "End date", "exact_value": "2025-12-31", "deck_value": "2025-12-31", "target_value": "2025-12-31"},
        {"metric": "Methodology", "exact_value": METHODOLOGY, "deck_value": METHODOLOGY, "target_value": METHODOLOGY},
        {"metric": "Governance caveat", "exact_value": GOVERNANCE_CAVEAT, "deck_value": GOVERNANCE_CAVEAT, "target_value": GOVERNANCE_CAVEAT},
    ]
    out = pd.DataFrame(rows)
    out["source_variant"] = SOURCE_VARIANT
    return out


def build_presentation_numbers(variant_summary: pd.DataFrame, metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    specs = [
        (
            "Corrected V4.1 / B-C2 teacher-smoothed baseline",
            "B_recovered_full_sample_teacher_replay_control",
            "External only with governance caveat; not strict PIT production evidence.",
            SOURCE_VARIANT,
        ),
        ("SPY benchmark", "F_SPY_benchmark_same_price_rows", "Deck-safe benchmark on same price rows.", "benchmark"),
        (
            "C0 / newly trained full-sample teacher",
            "C_new_full_sample_teacher_faithful_grid",
            "Research comparison only; shows current C implementation before recovered preprocessing correction.",
            "research_only",
        ),
        (
            "D strict PIT teacher-student",
            "D_pit_teacher_student_faithful_grid",
            "Governance comparison; strict PIT research variant underperformed B/C2.",
            "research_only",
        ),
    ]
    for label, variant, note, tag in specs:
        src = variant_summary[variant_summary["Variant"].eq(variant)]
        if src.empty and label.startswith("Corrected"):
            rows.append(
                {
                    "deck_row": label,
                    "source_variant": tag,
                    "CAGR_exact": metrics["CAGR"],
                    "CAGR_deck": "20.8%",
                    "Sharpe_exact": metrics["Sharpe"],
                    "Sharpe_deck": "1.00",
                    "Sortino_exact": metrics["Sortino"],
                    "Sortino_deck": "1.37",
                    "Max_DD_exact": metrics["Max DD"],
                    "Max_DD_deck": "-25.5%",
                    "Final_Equity_exact": metrics["Final equity"],
                    "Final_Equity_deck": "$12.88M",
                    "Switch_Count": metrics["Switches"],
                    "notes": note,
                }
            )
            continue
        if src.empty:
            continue
        r = src.iloc[0]
        rows.append(
            {
                "deck_row": label,
                "source_variant": tag,
                "CAGR_exact": r["CAGR"],
                "CAGR_deck": f"{float(r['CAGR']) * 100:.1f}%",
                "Sharpe_exact": r["Sharpe"],
                "Sharpe_deck": f"{float(r['Sharpe']):.2f}",
                "Sortino_exact": r.get("Sortino", np.nan),
                "Sortino_deck": f"{float(r.get('Sortino', np.nan)):.2f}" if pd.notna(r.get("Sortino", np.nan)) else "",
                "Max_DD_exact": r["Max DD"],
                "Max_DD_deck": f"{float(r['Max DD']) * 100:.1f}%",
                "Final_Equity_exact": r["Final Equity"],
                "Final_Equity_deck": f"${float(r['Final Equity']) / 1_000_000:.2f}M",
                "Switch_Count": r.get("Switch Count", ""),
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def copy_signal_files() -> list[dict[str, Any]]:
    records = []
    copies = [
        (REF_FULL_SIGNAL_PATH, SIGNAL_DIR / "full_signal_export.csv", "Recovered corrected full-sample teacher signal export"),
        (REF_WALKFORWARD_PATH, SIGNAL_DIR / "walkforward_student_signals.csv", "Recovered corrected walk-forward signal stream"),
        (REF_WALKFORWARD_PATH, SIGNAL_DIR / "walkforward_signals.csv", "Alias of recovered corrected walk-forward signal stream"),
    ]
    for src, dst, role in copies:
        if src.exists():
            assert_output(dst)
            shutil.copy2(src, dst)
            records.append(file_record(dst, role))

    if REF_FULL_SIGNAL_PATH.exists():
        full = normalize_timestamp(pd.read_csv(REF_FULL_SIGNAL_PATH))
        cols = [c for c in ["timestamp", "predicted_regime"] if c in full.columns]
        pseudo = full[cols].copy()
        pseudo = pseudo.rename(columns={"predicted_regime": "teacher_pseudo_label"})
        pseudo["source_variant"] = SOURCE_VARIANT
        pseudo["notes"] = "Derived from recovered full_signal_export.csv.predicted_regime; this is the B/C2 student label source."
        write_csv(pseudo, SIGNAL_DIR / "teacher_pseudo_labels.csv")
        records.append(file_record(SIGNAL_DIR / "teacher_pseudo_labels.csv", "Derived B/C2 teacher pseudo labels"))
    return records


def validate(metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    tolerances = {
        "CAGR": 5e-7,
        "Sharpe": 5e-6,
        "Sortino": 5e-6,
        "Max DD": 5e-7,
        "Final equity": 0.02,
        "Switches": 0,
    }
    for metric, target in TARGETS.items():
        actual = metrics[metric]
        diff = actual - target
        tol = tolerances[metric]
        rows.append(
            {
                "metric": metric,
                "actual": actual,
                "target": target,
                "difference": diff,
                "tolerance": tol,
                "passes": abs(diff) <= tol,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    hourly = load_hourly()
    daily = build_daily(hourly)
    trades = build_trades(hourly)
    metrics = metrics_from_daily(daily, trades)

    write_csv(daily, OUTPUT_ROOT / "daily_performance.csv")
    write_csv(period_table(daily, "weekly"), OUTPUT_ROOT / "weekly_performance.csv")
    write_csv(period_table(daily, "monthly"), OUTPUT_ROOT / "monthly_performance.csv")
    write_csv(period_table(daily, "quarterly"), OUTPUT_ROOT / "quarterly_performance.csv")
    write_csv(period_table(daily, "half_yearly"), OUTPUT_ROOT / "half_yearly_performance.csv")
    write_csv(period_table(daily, "yearly"), OUTPUT_ROOT / "yearly_performance.csv")
    write_csv(trades, OUTPUT_ROOT / "trades.csv")
    write_csv(build_summary_metrics(metrics), OUTPUT_ROOT / "summary_metrics.csv")

    variant_summary = pd.read_csv(VARIANT_SUMMARY_PATH)
    write_csv(build_presentation_numbers(variant_summary, metrics), OUTPUT_ROOT / "presentation_numbers.csv")
    validation = validate(metrics)
    write_csv(validation, OUTPUT_ROOT / "metric_reconciliation.csv")

    signal_records = copy_signal_files()
    input_records = [
        file_record(HOURLY_EQUITY_PATH, "primary reconciled hourly equity curve"),
        file_record(DAILY_EQUITY_PATH, "reference daily equity curve"),
        file_record(BEST_MATCH_PATH, "best methodology diagnosis"),
        file_record(VARIANT_SUMMARY_PATH, "same-data presentation comparison variants"),
        file_record(LINEAGE_LADDER_PATH, "B-vs-C reproducibility ladder"),
        file_record(REF_FULL_SIGNAL_PATH, "recovered full_signal_export"),
        file_record(REF_WALKFORWARD_PATH, "recovered walkforward_signals"),
    ]
    write_csv(pd.DataFrame(input_records + signal_records), OUTPUT_ROOT / "source_artifact_inventory.csv")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "output_root": OUTPUT_ROOT,
        "source_variant": SOURCE_VARIANT,
        "methodology": METHODOLOGY,
        "governance_caveat": GOVERNANCE_CAVEAT,
        "primary_input": file_record(HOURLY_EQUITY_PATH, "primary reconciled hourly equity curve"),
        "daily_rows": len(daily),
        "hourly_rows": len(hourly),
        "trade_rows": len(trades),
        "metrics": metrics,
        "validation_all_passed": bool(validation["passes"].all()),
        "production_delta_writes": False,
        "saxo_or_rt_changes": False,
        "workflow_or_model_registry_changes": False,
    }
    write_json(manifest, OUTPUT_ROOT / "presentation_support_manifest.json")

    files = [
        "daily_performance.csv",
        "weekly_performance.csv",
        "monthly_performance.csv",
        "quarterly_performance.csv",
        "half_yearly_performance.csv",
        "yearly_performance.csv",
        "trades.csv",
        "summary_metrics.csv",
        "presentation_numbers.csv",
        "metric_reconciliation.csv",
        "source_artifact_inventory.csv",
        "presentation_support_manifest.json",
        "signal_files/full_signal_export.csv",
        "signal_files/teacher_pseudo_labels.csv",
        "signal_files/walkforward_student_signals.csv",
        "signal_files/walkforward_signals.csv",
    ]
    readme = f"""# V4.1 Presentation Support - MUB-21

Generated: {datetime.now(timezone.utc).isoformat()}

## Source

Primary source: `{HOURLY_EQUITY_PATH}`

Input SHA256: `{sha256(HOURLY_EQUITY_PATH)}`

Rows:
- hourly source rows: `{len(hourly)}`
- daily output rows: `{len(daily)}`
- trades: `{len(trades)}`

Date range:
- first daily date: `{daily['date'].min()}`
- last daily date: `{daily['date'].max()}`

## Methodology

{METHODOLOGY}

## Governance Caveat

{GOVERNANCE_CAVEAT}

## Metric Reconciliation

| Metric | Actual | Target | Difference | Pass |
| --- | ---: | ---: | ---: | --- |
""" + "\n".join(
        f"| {r.metric} | {r.actual} | {r.target} | {r.difference} | {r.passes} |"
        for r in validation.itertuples(index=False)
    ) + f"""

## File Inventory

""" + "\n".join(f"- `{f}`" for f in files) + """

## Production Safety

This package reads only research/forensic CSV artifacts. It does not recompute
from live Saxo tables and does not write production Delta tables, model registry
entries, workflows, schedules, realtime notebooks, dashboard notebooks, or trading
logic.
"""
    write_text(readme, OUTPUT_ROOT / "README.md")

    print(json.dumps(manifest, indent=2, default=str))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()

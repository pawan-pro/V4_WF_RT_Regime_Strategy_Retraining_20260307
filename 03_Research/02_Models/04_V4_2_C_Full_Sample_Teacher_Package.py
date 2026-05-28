# Databricks notebook source
"""Package Variant C: newly trained full-sample teacher on faithful legacy grid.

Research-only. This packages the already validated same-data Variant C
artifacts trained on the faithful legacy master grid and recomputes/report
performance under an isolated package root.

No production Delta writes, model registry writes, workflow changes, Saxo
changes, RT changes, or dashboard changes are performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_c_full_sample_teacher_package_2026-05-04"
CODE_DIR = OUTPUT_ROOT / "code"
INPUTS_DIR = OUTPUT_ROOT / "inputs"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
MODELS_DIR = OUTPUT_ROOT / "models"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"

FAITHFUL_GRID_PATH = (
    BASE
    / "v4_1_csv"
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)
SAME_DATA_ROOT = BASE / "v4_2_same_data_teacher_student_2026-05-04"
SAME_DATA_SCRIPT = SAME_DATA_ROOT / "code" / "03_V4_2_Same_Data_Teacher_Student.py"
C_VARIANT = "C_new_full_sample_teacher_faithful_grid"
B_VARIANT = "B_recovered_full_sample_teacher_replay_control"
D_VARIANT = "D_pit_teacher_student_faithful_grid"
F_VARIANT = "F_SPY_benchmark_same_price_rows"

SOURCE_C_TEACHER = SAME_DATA_ROOT / "models" / C_VARIANT / "teacher" / "teacher_model.joblib"
SOURCE_C_FOLDS = SAME_DATA_ROOT / "models" / C_VARIANT / "folds"
SOURCE_C_SIGNALS = SAME_DATA_ROOT / "signals" / f"{C_VARIANT}_signals.csv"
SOURCE_C_PSEUDO = SAME_DATA_ROOT / "signals" / "C_new_full_sample_teacher_pseudo_labels.csv"
SOURCE_VARIANT_SUMMARY = SAME_DATA_ROOT / "performance" / "variant_summary.csv"
SOURCE_SIGNAL_MATCHES = SAME_DATA_ROOT / "forensics" / "signal_match_rates.csv"

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005

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
REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("joblib", "joblib"), ("xgboost", "xgboost"), ("openpyxl", "openpyxl")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    import joblib
    import xgboost as xgb

    return joblib, xgb


joblib, xgb = ensure_python_deps()


def ensure_dirs() -> None:
    for path in [CODE_DIR, INPUTS_DIR, MANIFESTS_DIR, SIGNALS_DIR, MODELS_DIR, PERFORMANCE_DIR, FORENSICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_output(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside package root: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, tuple):
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


def md_table(df: pd.DataFrame) -> str:
    out = df.copy().astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.sort_index(axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_csv(path: Path) -> pd.DataFrame:
    return normalize_timestamp(pd.read_csv(path))


def label_regime(ret: float) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def load_grid() -> pd.DataFrame:
    grid = load_csv(FAITHFUL_GRID_PATH)
    cols = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing columns: {missing}")
    grid = grid[cols].copy()
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["fwd_return_1w"] = grid["SPY_close"].shift(-FWD_HOURS) / grid["SPY_close"] - 1.0
    grid["raw_regime"] = grid["fwd_return_1w"].apply(label_regime)
    return grid


def trainable_frame(grid: pd.DataFrame) -> pd.DataFrame:
    need = FEATURES + ["SPY_close", "QQQ_close", "raw_regime"]
    out = grid[(grid["timestamp"] >= TRAIN_START) & grid[need].notna().all(axis=1)].copy()
    out["raw_regime"] = out["raw_regime"].astype(int)
    return out.reset_index(drop=True)


def dist(series: pd.Series) -> dict[str, int]:
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    assert_output(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return "directory_copied"
    shutil.copy2(src, dst)
    return file_sha256(dst)


def build_full_signal_export(trainable: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    teacher = joblib.load(SOURCE_C_TEACHER)
    full = trainable[["timestamp", "raw_regime", "fwd_return_1w", "SPY_close", "QQQ_close", *FEATURES]].copy()
    full["teacher_pseudo_regime"] = teacher.predict(full[FEATURES]).astype(int)
    try:
        proba = teacher.predict_proba(full[FEATURES])
        for idx in range(proba.shape[1]):
            full[f"teacher_prob_regime{idx}"] = proba[:, idx]
    except Exception:
        pass
    full["teacher_regime_name"] = full["teacher_pseudo_regime"].map(REGIME_NAME)
    return full, dist(full["teacher_pseudo_regime"])


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    sig = signals[["timestamp", "predicted_regime", "allocation_scale"]].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close"]] = data[["SPY_close", "QQQ_close"]].ffill()
    data["price_update"] = data[["SPY_close", "QQQ_close"]].diff().abs().sum(axis=1).gt(0)
    if not data.empty:
        data.loc[data.index[0], "price_update"] = True
    data = data[data["price_update"]].dropna(subset=["SPY_close", "QQQ_close", "predicted_regime"]).reset_index(drop=True)
    data["predicted_regime"] = data["predicted_regime"].astype(int)
    data["allocation_scale"] = pd.to_numeric(data["allocation_scale"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    return data


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = price_update_frame(grid, signals)
    equity = [INITIAL_CAPITAL]
    applied_regimes = [np.nan]
    target_assets = ["START"]
    gross_returns = [0.0]
    strategy_returns = [0.0]
    switch_costs = [0.0]
    position_changed = [False]
    active_asset = "START"
    for idx in range(1, len(data)):
        regime = int(data["predicted_regime"].iloc[idx - 1])
        scale = float(data["allocation_scale"].iloc[idx - 1])
        target_asset = REGIME_TO_ASSET.get(regime, "CASH")
        if target_asset == "QQQ":
            gross_ret = scale * (data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0)
        elif target_asset == "SPY":
            gross_ret = scale * (data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0)
        else:
            gross_ret = 0.0
        changed = target_asset != active_asset
        cost = SWITCH_COST if changed else 0.0
        strat_ret = gross_ret - cost
        equity.append(equity[-1] * (1.0 + strat_ret))
        applied_regimes.append(regime)
        target_assets.append(target_asset)
        gross_returns.append(gross_ret)
        strategy_returns.append(strat_ret)
        switch_costs.append(cost)
        position_changed.append(changed)
        active_asset = target_asset
    out = data.copy()
    out["variant"] = C_VARIANT
    out["applied_regime"] = applied_regimes
    out["target_asset"] = target_assets
    out["gross_strategy_return"] = gross_returns
    out["strategy_return"] = strategy_returns
    out["switch_cost"] = switch_costs
    out["position_changed"] = position_changed
    out["strategy_equity"] = equity
    out["drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["daily_return"] = daily["strategy_equity"].pct_change()
    metrics = metrics_from_daily(daily)
    metrics["switch_count"] = int(out["position_changed"].astype(bool).sum())
    metrics["hourly_price_update_rows"] = len(out)
    metrics["daily_rows"] = len(daily)
    return out, daily, metrics


def metrics_from_daily(daily: pd.DataFrame) -> dict[str, Any]:
    ret = daily["strategy_equity"].astype(float).pct_change().dropna()
    downside = ret[ret < 0]
    eq = daily["strategy_equity"].astype(float).reset_index(drop=True)
    dd = eq / eq.cummax() - 1.0
    final = float(eq.iloc[-1])
    return {
        "CAGR": (final / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max DD": float(dd.min()),
        "Final Equity": final,
    }


def period_returns(daily: pd.DataFrame, freq: str, label: str) -> pd.DataFrame:
    ts = pd.to_datetime(daily["timestamp"])
    frame = daily[["timestamp", "strategy_equity"]].copy()
    frame["period"] = ts.dt.to_period(freq).astype(str)
    rows = []
    for period, group in frame.groupby("period"):
        rows.append(
            {
                label: period,
                "start_timestamp": group["timestamp"].iloc[0],
                "end_timestamp": group["timestamp"].iloc[-1],
                "start_equity": float(group["strategy_equity"].iloc[0]),
                "end_equity": float(group["strategy_equity"].iloc[-1]),
                "return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows)


def write_audit_pack(
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    teacher_dist: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
) -> None:
    path = PERFORMANCE_DIR / "audit_pack.xlsx"
    assert_output(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "audit_pack.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="variant_c_metrics", index=False)
            comparison.to_excel(writer, sheet_name="comparison", index=False)
            teacher_dist.to_excel(writer, sheet_name="teacher_distribution", index=False)
            daily.to_excel(writer, sheet_name="daily_returns", index=False)
            weekly.to_excel(writer, sheet_name="weekly_returns", index=False)
            monthly.to_excel(writer, sheet_name="monthly_returns", index=False)
            yearly.to_excel(writer, sheet_name="yearly_returns", index=False)
        shutil.copyfile(tmp, path)


def write_readme(metrics: dict[str, Any], comparison: pd.DataFrame) -> None:
    path = OUTPUT_ROOT / "README.md"
    lines = [
        "# V4.2 Variant C Full-Sample Teacher Package",
        "",
        "This is a research-only package for Variant C: a newly trained full-sample teacher on the faithful legacy master grid.",
        "",
        "## What C Is",
        "",
        "Variant C trains a full-sample teacher on raw next-week return regimes, then uses the teacher's smoothed `teacher_pseudo_regime` labels to train monthly walk-forward student models. The student signals are therefore based on teacher-smoothed regime labels rather than directly hardcoded `+/-0.5%` next-week return labels.",
        "",
        "## Faithful Legacy Data Assumptions",
        "",
        f"- Input grid: `{FAITHFUL_GRID_PATH}`",
        "- The grid is the faithful legacy clone that reproduced the recovered corrected reference feature set.",
        "- Performance uses faithful-grid price-update hourly rows, prior-bar allocation, 5 bps switch cost on every asset transition, daily-last reporting, and a fixed five-year 2021-2025 CAGR window.",
        "",
        "## Governance",
        "",
        "Variant C is full-sample and non-PIT because the teacher is trained on the full available historical sample. It is research-only and is not eligible for production promotion without a strict PIT redesign.",
        "",
        "## Variant C Metrics",
        "",
        md_table(pd.DataFrame([metrics])),
        "",
        "## Comparison",
        "",
        md_table(comparison),
        "",
        "## Safety",
        "",
        "No production Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, workflows, schedules, or model registry assets were modified.",
        "",
    ]
    assert_output(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    required = [FAITHFUL_GRID_PATH, SAME_DATA_SCRIPT, SOURCE_C_TEACHER, SOURCE_C_SIGNALS, SOURCE_VARIANT_SUMMARY]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required package inputs: {missing}")

    grid = load_grid()
    trainable = trainable_frame(grid)

    # Code and model snapshots.
    copy_if_exists(SAME_DATA_SCRIPT, CODE_DIR / "03_V4_2_Same_Data_Teacher_Student.py")
    self_file = globals().get("__file__")
    self_path = Path(self_file) if self_file else None
    if self_path and self_path.is_file():
        copy_if_exists(self_path, CODE_DIR / "04_V4_2_C_Full_Sample_Teacher_Package.py")
    copy_if_exists(SOURCE_C_TEACHER, MODELS_DIR / "teacher" / "teacher_model.joblib")
    copy_if_exists(SOURCE_C_FOLDS, MODELS_DIR / "folds")

    full_signal, teacher_distribution = build_full_signal_export(trainable)
    write_csv(full_signal, SIGNALS_DIR / "full_signal_export.csv")
    if SOURCE_C_PSEUDO.exists():
        copy_if_exists(SOURCE_C_PSEUDO, SIGNALS_DIR / "teacher_pseudo_labels.csv")

    c_signals = load_csv(SOURCE_C_SIGNALS)
    write_csv(c_signals, SIGNALS_DIR / "walkforward_student_signals.csv")
    hourly, daily, metrics = run_backtest(grid, c_signals)
    write_csv(hourly, PERFORMANCE_DIR / "equity_curve_hourly_price_updates.csv")
    write_csv(daily, PERFORMANCE_DIR / "equity_curve_daily_last.csv")
    write_csv(hourly[hourly["position_changed"].astype(bool)].copy(), PERFORMANCE_DIR / "trades.csv")

    daily_returns = daily[["timestamp", "strategy_equity", "daily_return", "drawdown"]].copy()
    weekly_returns = period_returns(daily, "W", "week")
    monthly_returns = period_returns(daily, "M", "month")
    yearly_returns = period_returns(daily, "Y", "year")
    write_csv(daily_returns, PERFORMANCE_DIR / "daily_returns.csv")
    write_csv(weekly_returns, PERFORMANCE_DIR / "weekly_returns.csv")
    write_csv(monthly_returns, PERFORMANCE_DIR / "monthly_returns.csv")
    write_csv(yearly_returns, PERFORMANCE_DIR / "yearly_returns.csv")

    source_summary = pd.read_csv(SOURCE_VARIANT_SUMMARY)
    comparison = source_summary[source_summary["Variant"].isin([B_VARIANT, C_VARIANT, D_VARIANT, F_VARIANT])].copy()
    write_csv(comparison, FORENSICS_DIR / "comparison_vs_B_D_SPY.csv")
    if SOURCE_SIGNAL_MATCHES.exists():
        copy_if_exists(SOURCE_SIGNAL_MATCHES, FORENSICS_DIR / "same_data_signal_match_rates.csv")

    teacher_dist_df = pd.DataFrame(
        [{"regime": k, "count": v} for k, v in teacher_distribution.items()]
    )
    metrics_payload = {
        "Variant": C_VARIANT,
        **metrics,
        "teacher_label_distribution": teacher_distribution,
        "teacher_model_sha256": file_sha256(SOURCE_C_TEACHER),
        "student_signal_sha256": file_sha256(SOURCE_C_SIGNALS),
        "full_signal_sha256": file_sha256(SIGNALS_DIR / "full_signal_export.csv"),
        "walkforward_signal_sha256": file_sha256(SIGNALS_DIR / "walkforward_student_signals.csv"),
    }
    metrics_df = pd.DataFrame([metrics_payload])
    write_csv(metrics_df, PERFORMANCE_DIR / "variant_c_metrics.csv")
    write_audit_pack(metrics_df, comparison, teacher_dist_df, daily_returns, weekly_returns, monthly_returns, yearly_returns)

    input_hashes = {
        "faithful_grid_path": FAITHFUL_GRID_PATH,
        "faithful_grid_sha256": file_sha256(FAITHFUL_GRID_PATH),
        "faithful_grid_rows": len(grid),
        "faithful_grid_min_timestamp": grid["timestamp"].min(),
        "faithful_grid_max_timestamp": grid["timestamp"].max(),
        "trainable_rows": len(trainable),
        "trainable_min_timestamp": trainable["timestamp"].min(),
        "trainable_max_timestamp": trainable["timestamp"].max(),
        "source_same_data_root": SAME_DATA_ROOT,
        "source_c_teacher_sha256": file_sha256(SOURCE_C_TEACHER),
        "source_c_signals_sha256": file_sha256(SOURCE_C_SIGNALS),
        "full_signal_hash": frame_hash(full_signal),
        "walkforward_signal_hash": frame_hash(c_signals),
    }
    write_json(input_hashes, MANIFESTS_DIR / "input_hashes.json")

    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "joblib_version": joblib.__version__,
        "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "output_root": OUTPUT_ROOT,
        "input_grid": FAITHFUL_GRID_PATH,
        "source_same_data_root": SAME_DATA_ROOT,
        "variant": C_VARIANT,
        "production_delta_writes": False,
        "model_registry_writes": False,
        "saxo_or_rt_writes": False,
        "dashboard_writes": False,
        "workflow_or_schedule_changes": False,
        "research_only": True,
        "full_sample_non_pit": True,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")
    write_readme(metrics_payload, comparison)

    lines = [
        "# Variant C Full-Sample Teacher Package Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output root: `{OUTPUT_ROOT}`",
        f"- Input grid: `{FAITHFUL_GRID_PATH}`",
        f"- Variant: `{C_VARIANT}`",
        "- Status: research-only, full-sample/non-PIT",
        "",
        "## Metrics",
        "",
        md_table(metrics_df),
        "",
        "## Comparison",
        "",
        md_table(comparison),
        "",
        "## Safety",
        "",
        "No production Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, workflows, schedules, or model registry assets were modified.",
        "",
    ]
    (FORENSICS_DIR / "variant_c_package_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

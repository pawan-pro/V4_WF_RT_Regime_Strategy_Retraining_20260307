# Databricks notebook source
"""V4.2 PIT-safe teacher-student research experiment.

Research-only. Reads `pawan_quant_research_1.v4.silver_bt_master_grid_1h`
and optional read-only production signal references. Writes only under
`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_pit_teacher_student_2026-05-04/`.

No Delta writes, model-registry writes, workflow changes, Saxo changes, or RT
table writes are performed by this script.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession, functions as F


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
PRODUCTION_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"

OUTPUT_ROOT = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data/v4_2_pit_teacher_student_2026-05-04")
SIGNALS_DIR = OUTPUT_ROOT / "signals"
MODELS_DIR = OUTPUT_ROOT / "models"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005
SLIPPAGE_BPS = [5, 10, 25]

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

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "merror",
    "tree_method": "hist",
    "random_state": 42,
}


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
    for path in [SIGNALS_DIR, MODELS_DIR, MANIFESTS_DIR, PERFORMANCE_DIR, FORENSICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_research_output_path(path: Path) -> None:
    resolved = str(path)
    allowed = str(OUTPUT_ROOT)
    if not resolved.startswith(allowed):
        raise ValueError(f"Refusing write outside research output root: {resolved}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_research_output_path(path)
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
        if obj is None:
            return None
        if isinstance(obj, (str, bool, int)):
            return obj
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(float(obj)) else float(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj

    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    out = df.copy().astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    use = df[cols].copy() if cols else df.copy()
    use = use.sort_index(axis=1)
    payload = use.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model() -> Any:
    return xgb.XGBClassifier(**MODEL_PARAMS)


def label_regime(ret: float) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_grid(spark: SparkSession) -> pd.DataFrame:
    cols = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in cols if c not in spark.table(SILVER_TABLE).columns]
    if missing:
        raise ValueError(f"{SILVER_TABLE} missing required columns: {missing}")
    pdf = spark.table(SILVER_TABLE).select(*cols).toPandas()
    pdf = normalize_timestamp(pdf)
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    pdf["fwd_return_1w"] = pdf["SPY_close"].shift(-FWD_HOURS) / pdf["SPY_close"] - 1.0
    pdf["raw_regime"] = pdf["fwd_return_1w"].apply(label_regime)
    pdf["row_id"] = np.arange(len(pdf))
    return pdf


def trainable_frame(grid: pd.DataFrame) -> pd.DataFrame:
    need = FEATURES + ["SPY_close", "QQQ_close", "raw_regime"]
    out = grid[(grid["timestamp"] >= TRAIN_START) & grid[need].notna().all(axis=1)].copy()
    out["raw_regime"] = out["raw_regime"].astype(int)
    return out.reset_index(drop=True)


def monthly_folds() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = EVAL_START
    while start < FINAL_FOLD_END:
        end = start + relativedelta(months=1)
        folds.append((start, end))
        start = end
    return folds


def dist(series: pd.Series) -> dict[str, int]:
    if series is None or len(series) == 0:
        return {}
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def dump_model(obj: Any, path: Path) -> str:
    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return file_sha256(path)


def fit_or_load_model(path: Path, features: pd.DataFrame, labels: pd.Series) -> tuple[Any, str, str]:
    """Reuse prior research-run models when present; otherwise fit and persist."""
    assert_research_output_path(path)
    if path.exists():
        return joblib.load(path), file_sha256(path), "loaded_existing_research_model"
    clf = model()
    clf.fit(features, labels)
    return clf, dump_model(clf, path), "trained_new_research_model"


@dataclass
class VariantResult:
    name: str
    signals: pd.DataFrame
    fold_manifest: pd.DataFrame


def signal_columns(df: pd.DataFrame, variant: str, label_col: str = "raw_regime") -> pd.DataFrame:
    out = df[["timestamp", label_col, "predicted_regime"]].rename(columns={label_col: "raw_regime"}).copy()
    out["variant"] = variant
    out["regime_name"] = out["predicted_regime"].astype(int).map(REGIME_NAME)
    out["allocation_scale"] = 1.0
    return out[["timestamp", "variant", "raw_regime", "predicted_regime", "regime_name", "allocation_scale"]]


def run_raw_label_wf(trainable: pd.DataFrame) -> VariantResult:
    variant = "A_raw_label_monthly_wf"
    signals = []
    manifests = []
    for fold_start, fold_end in monthly_folds():
        train_df = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)].copy()
        test_df = trainable[(trainable["timestamp"] >= fold_start) & (trainable["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        model_path = MODELS_DIR / variant / "folds" / fold_start.strftime("%Y_%m") / "model.joblib"
        clf, model_hash, model_action = fit_or_load_model(model_path, train_df[FEATURES], train_df["raw_regime"])
        test_df["predicted_regime"] = clf.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
        signal_hash = frame_hash(fold_signal)
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "teacher_train_start": None,
                "teacher_train_end": None,
                "student_train_start": train_df["timestamp"].min(),
                "student_train_end": train_df["timestamp"].max(),
                "train_row_count": len(train_df),
                "test_row_count": len(test_df),
                "teacher_label_distribution": None,
                "student_label_distribution": dist(train_df["raw_regime"]),
                "student_prediction_distribution": dist(test_df["predicted_regime"]),
                "feature_list": FEATURES,
                "model_parameters": MODEL_PARAMS,
                "input_data_hash": frame_hash(train_df[["timestamp", "raw_regime", *FEATURES]]),
                "output_signal_hash": signal_hash,
                "model_path": str(model_path),
                "model_sha256": model_hash,
                "model_action": model_action,
            }
        )
    return VariantResult(variant, pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def run_full_sample_teacher_reference(trainable: pd.DataFrame) -> VariantResult:
    variant = "B_retrospective_full_sample_teacher_reference"
    teacher_path = MODELS_DIR / variant / "teacher" / "teacher_model.joblib"
    teacher, teacher_hash, teacher_action = fit_or_load_model(teacher_path, trainable[FEATURES], trainable["raw_regime"])
    pseudo = trainable[["timestamp", "raw_regime", *FEATURES]].copy()
    pseudo["teacher_pseudo_regime"] = teacher.predict(pseudo[FEATURES]).astype(int)
    write_csv(pseudo[["timestamp", "raw_regime", "teacher_pseudo_regime"]], SIGNALS_DIR / "full_sample_teacher_pseudo_labels.csv")

    signals = []
    manifests = []
    for fold_start, fold_end in monthly_folds():
        train_df = pseudo[(pseudo["timestamp"] >= TRAIN_START) & (pseudo["timestamp"] < fold_start)].copy()
        test_df = pseudo[(pseudo["timestamp"] >= fold_start) & (pseudo["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        student_path = MODELS_DIR / variant / "folds" / fold_start.strftime("%Y_%m") / "student_model.joblib"
        student, student_hash, student_action = fit_or_load_model(student_path, train_df[FEATURES], train_df["teacher_pseudo_regime"])
        test_df["predicted_regime"] = student.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
        signal_hash = frame_hash(fold_signal)
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "teacher_train_start": trainable["timestamp"].min(),
                "teacher_train_end": trainable["timestamp"].max(),
                "student_train_start": train_df["timestamp"].min(),
                "student_train_end": train_df["timestamp"].max(),
                "train_row_count": len(train_df),
                "test_row_count": len(test_df),
                "teacher_label_distribution": dist(trainable["raw_regime"]),
                "student_label_distribution": dist(train_df["teacher_pseudo_regime"]),
                "student_prediction_distribution": dist(test_df["predicted_regime"]),
                "feature_list": FEATURES,
                "model_parameters": MODEL_PARAMS,
                "input_data_hash": frame_hash(train_df[["timestamp", "teacher_pseudo_regime", *FEATURES]]),
                "output_signal_hash": signal_hash,
                "teacher_model_path": str(teacher_path),
                "teacher_model_sha256": teacher_hash,
                "teacher_model_action": teacher_action,
                "student_model_path": str(student_path),
                "student_model_sha256": student_hash,
                "student_model_action": student_action,
                "leakage_flag": "full_sample_teacher_reference_not_pit",
            }
        )
    return VariantResult(variant, pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def run_pit_teacher_student(trainable: pd.DataFrame) -> VariantResult:
    variant = "C_pit_safe_teacher_student"
    signals = []
    manifests = []
    for fold_start, fold_end in monthly_folds():
        teacher_train = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)].copy()
        student_train = teacher_train.copy()
        test_df = trainable[(trainable["timestamp"] >= fold_start) & (trainable["timestamp"] < fold_end)].copy()
        if teacher_train.empty or student_train.empty or test_df.empty:
            continue
        fold_dir = MODELS_DIR / variant / "folds" / fold_start.strftime("%Y_%m")
        teacher, teacher_hash, teacher_action = fit_or_load_model(fold_dir / "teacher_model.joblib", teacher_train[FEATURES], teacher_train["raw_regime"])
        student_train["teacher_pseudo_regime"] = teacher.predict(student_train[FEATURES]).astype(int)
        student, student_hash, student_action = fit_or_load_model(fold_dir / "student_model.joblib", student_train[FEATURES], student_train["teacher_pseudo_regime"])
        test_df["predicted_regime"] = student.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
        signal_hash = frame_hash(fold_signal)
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "teacher_train_start": teacher_train["timestamp"].min(),
                "teacher_train_end": teacher_train["timestamp"].max(),
                "student_train_start": student_train["timestamp"].min(),
                "student_train_end": student_train["timestamp"].max(),
                "train_row_count": len(student_train),
                "test_row_count": len(test_df),
                "teacher_label_distribution": dist(teacher_train["raw_regime"]),
                "student_label_distribution": dist(student_train["teacher_pseudo_regime"]),
                "student_prediction_distribution": dist(test_df["predicted_regime"]),
                "feature_list": FEATURES,
                "model_parameters": MODEL_PARAMS,
                "input_data_hash": frame_hash(student_train[["timestamp", "teacher_pseudo_regime", *FEATURES]]),
                "output_signal_hash": signal_hash,
                "teacher_model_path": str(fold_dir / "teacher_model.joblib"),
                "teacher_model_sha256": teacher_hash,
                "teacher_model_action": teacher_action,
                "student_model_path": str(fold_dir / "student_model.joblib"),
                "student_model_sha256": student_hash,
                "student_model_action": student_action,
                "leakage_flag": "pit_safe_teacher_train_rows_strictly_before_fold_start",
            }
        )
    return VariantResult(variant, pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def run_hybrid(raw_signals: pd.DataFrame, pit_signals: pd.DataFrame) -> VariantResult:
    variant = "D_conservative_hybrid"
    raw = raw_signals[["timestamp", "raw_regime", "predicted_regime"]].rename(columns={"predicted_regime": "raw_pred"})
    pit = pit_signals[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "pit_pred"})
    merged = raw.merge(pit, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    merged["agree"] = merged["raw_pred"].astype(int) == merged["pit_pred"].astype(int)
    merged["predicted_regime"] = np.where(merged["agree"], merged["raw_pred"], np.maximum(merged["raw_pred"], merged["pit_pred"]))
    merged["allocation_scale"] = np.where(merged["agree"], 1.0, 0.5)
    merged["variant"] = variant
    merged["regime_name"] = merged["predicted_regime"].astype(int).map(REGIME_NAME)
    signals = merged[["timestamp", "variant", "raw_regime", "predicted_regime", "regime_name", "allocation_scale", "raw_pred", "pit_pred", "agree"]].copy()
    manifests = []
    for fold_start, fold_end in monthly_folds():
        fold = signals[(signals["timestamp"] >= fold_start) & (signals["timestamp"] < fold_end)].copy()
        if fold.empty:
            continue
        manifests.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "teacher_train_start": None,
                "teacher_train_end": None,
                "student_train_start": None,
                "student_train_end": None,
                "train_row_count": None,
                "test_row_count": len(fold),
                "teacher_label_distribution": None,
                "student_label_distribution": None,
                "student_prediction_distribution": dist(fold["predicted_regime"]),
                "feature_list": FEATURES,
                "model_parameters": {"hybrid_rule": "if raw and PIT agree use full allocation; otherwise choose more defensive regime and 50% allocation"},
                "input_data_hash": frame_hash(fold[["timestamp", "raw_pred", "pit_pred", "allocation_scale"]]),
                "output_signal_hash": frame_hash(fold),
                "agreement_rate": float(fold["agree"].mean()),
            }
        )
    return VariantResult(variant, signals, pd.DataFrame(manifests))


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


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame, variant: str, switch_cost: float = SWITCH_COST) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = price_update_frame(grid, signals)
    if len(data) < 2:
        raise ValueError(f"Not enough price-update rows for {variant}")
    equity = [INITIAL_CAPITAL]
    spy_equity = [INITIAL_CAPITAL]
    applied_regimes = [np.nan]
    target_assets = ["START"]
    target_scales = [0.0]
    gross_returns = [0.0]
    strategy_returns = [0.0]
    switch_costs = [0.0]
    position_changed = [False]
    active_asset = "START"
    active_scale = 0.0
    for idx in range(1, len(data)):
        signal_idx = idx - 1
        regime = int(data["predicted_regime"].iloc[signal_idx])
        scale = float(data["allocation_scale"].iloc[signal_idx])
        target_asset = REGIME_TO_ASSET.get(regime, "CASH")
        spy_ret = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        qqq_ret = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        if target_asset == "QQQ":
            gross_ret = scale * qqq_ret
        elif target_asset == "SPY":
            gross_ret = scale * spy_ret
        else:
            gross_ret = 0.0
            scale = 0.0
        changed = target_asset != active_asset or abs(scale - active_scale) > 1e-12
        cost = switch_cost * scale if changed else 0.0
        strat_ret = gross_ret - cost
        equity.append(equity[-1] * (1.0 + strat_ret))
        spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))
        applied_regimes.append(regime)
        target_assets.append(target_asset)
        target_scales.append(scale)
        gross_returns.append(gross_ret)
        strategy_returns.append(strat_ret)
        switch_costs.append(cost)
        position_changed.append(changed)
        active_asset = target_asset
        active_scale = scale
    out = data.copy()
    out["variant"] = variant
    out["applied_regime"] = applied_regimes
    out["target_asset"] = target_assets
    out["target_scale"] = target_scales
    out["gross_strategy_return"] = gross_returns
    out["switch_cost"] = switch_costs
    out["strategy_return"] = strategy_returns
    out["position_changed"] = position_changed
    out["strategy_equity"] = equity
    out["spy_return"] = out["SPY_close"].pct_change().fillna(0.0)
    out["spy_equity"] = spy_equity
    out["drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["ny_date"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    daily = out.loc[out.groupby("ny_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
    metrics = metrics_from_daily(daily, variant)
    return out, daily, metrics


def metrics_from_daily(daily: pd.DataFrame, variant: str) -> dict[str, Any]:
    ret = daily["strategy_daily_return"].astype(float)
    spy_ret = daily["spy_daily_return"].astype(float)
    downside = ret[ret < 0]
    spy_downside = spy_ret[spy_ret < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1])
    spy_final = float(daily["spy_equity"].iloc[-1])
    dd = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    spy_dd = daily["spy_equity"] / daily["spy_equity"].cummax() - 1.0
    return {
        "variant": variant,
        "start_timestamp": daily["timestamp"].iloc[0],
        "end_timestamp": daily["timestamp"].iloc[-1],
        "daily_rows": len(daily),
        "start_equity": INITIAL_CAPITAL,
        "final_equity": final_equity,
        "CAGR": (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max DD": float(dd.min()),
        "SPY_final_equity": spy_final,
        "SPY_CAGR": (spy_final / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "SPY_Sharpe": spy_ret.mean() / spy_ret.std(ddof=1) * math.sqrt(252.0) if spy_ret.std(ddof=1) else 0.0,
        "SPY_Sortino": spy_ret.mean() / spy_downside.std(ddof=1) * math.sqrt(252.0) if len(spy_downside) > 1 and spy_downside.std(ddof=1) else 0.0,
        "SPY_Max DD": float(spy_dd.min()),
        "methodology": "prior-bar allocation, price-update hourly rows, daily-last equity, 5.0-year CAGR denominator, daily Sharpe/Sortino 252 rf=0",
    }


def annual_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in daily.groupby(pd.to_datetime(daily["timestamp"]).dt.year):
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "spy_return": float(group["spy_equity"].iloc[-1] / group["spy_equity"].iloc[0] - 1.0),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def trailing_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    end = pd.Timestamp(daily["timestamp"].iloc[-1])
    for years in [1, 3, 5]:
        sub = daily[daily["timestamp"] >= end - pd.DateOffset(years=years)].copy()
        if len(sub) < 3:
            continue
        rows.append(metrics_from_daily(sub, f"trailing_{years}y"))
        rows[-1]["window_years"] = years
    return pd.DataFrame(rows)


def scaling_diagnostics(hourly: pd.DataFrame, daily: pd.DataFrame, variant: str) -> dict[str, Any]:
    switches = hourly["position_changed"].astype(bool)
    switch_count = int(switches.sum())
    holding_lengths = hourly.groupby(switches.cumsum()).size()
    daily_ret = daily["strategy_daily_return"].astype(float)
    crisis_windows = {
        "2022_rate_shock": ("2022-01-01", "2022-12-31"),
        "2023_bank_stress": ("2023-03-01", "2023-05-31"),
        "2024_q3_vol": ("2024-07-01", "2024-09-30"),
        "2025_full_year": ("2025-01-01", "2025-12-31"),
    }
    crisis = {}
    for name, (start, end) in crisis_windows.items():
        sub = daily[(daily["timestamp"] >= pd.Timestamp(start)) & (daily["timestamp"] <= pd.Timestamp(end))]
        if len(sub) > 1:
            crisis[name] = {
                "return": float(sub["strategy_equity"].iloc[-1] / sub["strategy_equity"].iloc[0] - 1.0),
                "max_drawdown": float((sub["strategy_equity"] / sub["strategy_equity"].cummax() - 1.0).min()),
            }
    notes = {
        "100K": "Operationally feasible if ETF-only execution remains liquid and signal turnover is accepted.",
        "1M": "Operationally feasible in SPY/QQQ under normal conditions; monitor switch clustering and close/next-bar execution drift.",
        "100M": "Requires dedicated liquidity, market-impact, participation-rate, and execution-window analysis before eligibility.",
    }
    return {
        "variant": variant,
        "turnover_switch_count": switch_count,
        "average_holding_period_price_update_rows": float(holding_lengths.mean()) if len(holding_lengths) else np.nan,
        "median_holding_period_price_update_rows": float(holding_lengths.median()) if len(holding_lengths) else np.nan,
        "max_drawdown": float((daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0).min()),
        "worst_1_day_return": float(daily_ret.min()),
        "worst_5_day_return": float((1.0 + daily_ret).rolling(5).apply(np.prod, raw=True).min() - 1.0),
        "worst_20_day_return": float((1.0 + daily_ret).rolling(20).apply(np.prod, raw=True).min() - 1.0),
        "crisis_windows": crisis,
        "notional_readiness_notes": notes,
    }


def slippage_sensitivity(grid: pd.DataFrame, signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for bps in SLIPPAGE_BPS:
        _, _, metrics = run_backtest(grid, signals, variant, switch_cost=bps / 10_000.0)
        rows.append({"variant": variant, "switch_cost_bps": bps, **metrics})
    return pd.DataFrame(rows)


def governance_verdict(metrics: dict[str, Any], diag: dict[str, Any], variant: str) -> str:
    if "full_sample_teacher" in variant:
        return "research_only"
    if metrics["CAGR"] <= 0 or metrics["Sharpe"] < 0.2 or metrics["Max DD"] < -0.50:
        return "not_eligible"
    if metrics["CAGR"] > metrics["SPY_CAGR"] and metrics["Sharpe"] >= 0.75 and metrics["Max DD"] >= -0.30 and diag["turnover_switch_count"] <= 250:
        return "eligible_for_100k"
    if metrics["CAGR"] > 0.05 and metrics["Sharpe"] >= 0.45 and metrics["Max DD"] >= -0.40:
        return "shadow_candidate"
    return "research_only"


def optional_production_signals(spark: SparkSession) -> pd.DataFrame | None:
    try:
        sdf = spark.table(PRODUCTION_SIGNAL_TABLE)
        if "timestamp" not in sdf.columns or "predicted_regime" not in sdf.columns:
            return None
        pdf = sdf.select("timestamp", "predicted_regime").toPandas()
    except Exception:
        return None
    pdf = normalize_timestamp(pdf)
    pdf["variant"] = "production_v4_read_only_reference"
    pdf["raw_regime"] = np.nan
    pdf["regime_name"] = pd.to_numeric(pdf["predicted_regime"], errors="coerce").astype("Int64").map(REGIME_NAME)
    pdf["allocation_scale"] = 1.0
    return pdf[["timestamp", "variant", "raw_regime", "predicted_regime", "regime_name", "allocation_scale"]].dropna(subset=["predicted_regime"])


def write_audit_pack(summary: pd.DataFrame, annual: dict[str, pd.DataFrame], trailing: dict[str, pd.DataFrame], diagnostics: pd.DataFrame) -> None:
    path = PERFORMANCE_DIR / "audit_pack.xlsx"
    assert_research_output_path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "audit_pack.xlsx"
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            diagnostics.to_excel(writer, sheet_name="scaling_diagnostics", index=False)
            for variant, frame in annual.items():
                frame.to_excel(writer, sheet_name=f"annual_{variant[:20]}", index=False)
            for variant, frame in trailing.items():
                frame.to_excel(writer, sheet_name=f"trailing_{variant[:18]}", index=False)
        shutil.copyfile(tmp_path, path)


def main() -> None:
    ensure_dirs()
    spark = SparkSession.builder.getOrCreate()
    grid = load_grid(spark)
    trainable = trainable_frame(grid)
    input_hash = frame_hash(grid[["timestamp", "SPY_close", "QQQ_close", *FEATURES]])

    results = [
        run_raw_label_wf(trainable),
        run_full_sample_teacher_reference(trainable),
        run_pit_teacher_student(trainable),
    ]
    results.append(run_hybrid(results[0].signals, results[2].signals))

    all_fold_manifest = []
    summary_rows = []
    diagnostics_rows = []
    annual_map = {}
    trailing_map = {}

    for result in results:
        signal_path = SIGNALS_DIR / f"{result.name}_signals.csv"
        write_csv(result.signals, signal_path)
        result.fold_manifest["variant_signal_path"] = str(signal_path)
        result.fold_manifest["variant_signal_sha256"] = file_sha256(signal_path)
        all_fold_manifest.append(result.fold_manifest)
        hourly, daily, metrics = run_backtest(grid, result.signals, result.name, SWITCH_COST)
        write_csv(hourly, PERFORMANCE_DIR / result.name / "equity_curve_hourly_price_updates.csv")
        write_csv(daily, PERFORMANCE_DIR / result.name / "equity_curve_daily_last.csv")
        ann = annual_returns(daily)
        trail = trailing_metrics(daily)
        write_csv(ann, PERFORMANCE_DIR / result.name / "annual_returns.csv")
        write_csv(trail, PERFORMANCE_DIR / result.name / "trailing_metrics.csv")
        diag = scaling_diagnostics(hourly, daily, result.name)
        sens = slippage_sensitivity(grid, result.signals, result.name)
        write_csv(sens, PERFORMANCE_DIR / result.name / "slippage_sensitivity.csv")
        verdict = governance_verdict(metrics, diag, result.name)
        write_json({"metrics": metrics, "scaling_diagnostics": diag, "verdict": verdict}, PERFORMANCE_DIR / result.name / "metrics.json")
        summary_rows.append(
            {
                "Variant": result.name,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max DD": metrics["Max DD"],
                "Final Equity": metrics["final_equity"],
                "Switch Count": diag["turnover_switch_count"],
                "Verdict": verdict,
            }
        )
        diagnostics_rows.append({"Variant": result.name, **diag})
        annual_map[result.name] = ann
        trailing_map[result.name] = trail

    prod = optional_production_signals(spark)
    if prod is not None and not prod.empty:
        write_csv(prod, SIGNALS_DIR / "production_v4_read_only_reference_signals.csv")
        hourly, daily, metrics = run_backtest(grid, prod, "production_v4_read_only_reference", SWITCH_COST)
        diag = scaling_diagnostics(hourly, daily, "production_v4_read_only_reference")
        write_csv(hourly, PERFORMANCE_DIR / "production_v4_read_only_reference" / "equity_curve_hourly_price_updates.csv")
        write_csv(daily, PERFORMANCE_DIR / "production_v4_read_only_reference" / "equity_curve_daily_last.csv")
        write_json({"metrics": metrics, "scaling_diagnostics": diag, "verdict": "read_only_reference"}, PERFORMANCE_DIR / "production_v4_read_only_reference" / "metrics.json")
        summary_rows.append(
            {
                "Variant": "production_v4_read_only_reference",
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max DD": metrics["Max DD"],
                "Final Equity": metrics["final_equity"],
                "Switch Count": diag["turnover_switch_count"],
                "Verdict": "read_only_reference",
            }
        )

    # SPY benchmark row from the raw-label daily path, which shares row membership.
    raw_daily = pd.read_csv(PERFORMANCE_DIR / results[0].name / "equity_curve_daily_last.csv")
    spy_ret = pd.to_numeric(raw_daily["spy_equity"], errors="coerce").pct_change().fillna(0.0)
    spy_down = spy_ret[spy_ret < 0]
    spy_final = float(raw_daily["spy_equity"].iloc[-1])
    spy_dd = pd.to_numeric(raw_daily["spy_equity"], errors="coerce") / pd.to_numeric(raw_daily["spy_equity"], errors="coerce").cummax() - 1.0
    summary_rows.append(
        {
            "Variant": "SPY_benchmark_same_price_update_rows",
            "CAGR": (spy_final / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
            "Sharpe": spy_ret.mean() / spy_ret.std(ddof=1) * math.sqrt(252.0) if spy_ret.std(ddof=1) else 0.0,
            "Sortino": spy_ret.mean() / spy_down.std(ddof=1) * math.sqrt(252.0) if len(spy_down) > 1 and spy_down.std(ddof=1) else 0.0,
            "Max DD": float(spy_dd.min()),
            "Final Equity": spy_final,
            "Switch Count": 0,
            "Verdict": "benchmark",
        }
    )

    fold_manifest = pd.concat(all_fold_manifest, ignore_index=True)
    write_csv(fold_manifest, MANIFESTS_DIR / "fold_manifest.csv")
    write_json({"folds": fold_manifest.to_dict("records")}, MANIFESTS_DIR / "fold_manifest.json")

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostics_rows)
    write_csv(summary, PERFORMANCE_DIR / "variant_summary.csv")
    write_csv(diagnostics, PERFORMANCE_DIR / "scaling_readiness_diagnostics.csv")
    write_audit_pack(summary, annual_map, trailing_map, diagnostics)

    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "joblib_version": joblib.__version__,
        "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "source_table": SILVER_TABLE,
        "read_only_reference_table": PRODUCTION_SIGNAL_TABLE,
        "output_root": str(OUTPUT_ROOT),
        "feature_list": FEATURES,
        "model_parameters": MODEL_PARAMS,
        "input_grid_hash": input_hash,
        "performance_methodology": {
            "allocation": "prior-bar / next-bar",
            "row_filter": "price-update hourly rows only",
            "switch_cost_bps": 5,
            "daily_reporting": "daily-last equity",
            "cagr_period": "2021-01-01 through 2025-12-31, exactly 5.0 years",
            "sharpe_sortino": "daily returns, 252 annualization, risk-free 0",
        },
        "production_delta_writes": False,
        "model_registry_writes": False,
        "saxo_or_rt_writes": False,
        "workflow_or_schedule_changes": False,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")

    lines = [
        "# V4.2 PIT Teacher-Student Research Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output root: `{OUTPUT_ROOT}`",
        f"- Source table: `{SILVER_TABLE}` read only",
        "- Production writes: none",
        "- Model registry writes: none",
        "",
        "## Variant Summary",
        "",
        md_table(summary),
        "",
        "## Governance",
        "",
        "The retrospective full-sample teacher reference is marked `research_only` by construction because it is not PIT. Promotion is explicitly out of scope.",
        "",
    ]
    (FORENSICS_DIR / "pit_teacher_student_research_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

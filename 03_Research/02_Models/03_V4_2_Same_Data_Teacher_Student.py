# Databricks notebook source
"""Same-data V4.2 teacher-student comparison on the faithful legacy grid.

Research-only. Reads CSV/Volume artifacts that reproduced the recovered
20.840487% forensic reference and writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_same_data_teacher_student_2026-05-04/`

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_same_data_teacher_student_2026-05-04"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
MODELS_DIR = OUTPUT_ROOT / "models"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"

FAITHFUL_GRID_PATH = (
    BASE
    / "v4_1_csv"
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)
REFERENCE_FULL_SIGNAL_PATH = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "full_signal_export.csv"
)
REFERENCE_WALKFORWARD_PATH = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "walkforward_signals.csv"
)
FAITHFUL_REPLAY_ROOT = BASE / "v4_1_csv" / "faithful_legacy_grid_model_replay_2026-04-30"
PERF_RECON_ROOT = BASE / "v4_1_csv" / "performance_methodology_reconciliation_2026-04-30"

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
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

TARGET_REFERENCE = {
    "CAGR": 0.20840487336126445,
    "Sharpe": 1.0023589403061952,
    "Sortino": 1.3729057801036,
    "Max DD": -0.2553851434504447,
    "Final Equity": 12883454.9938051,
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
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


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
        if isinstance(obj, (np.floating, float)):
            return None if math.isnan(float(obj)) else float(obj)
        return obj

    assert_research_output_path(path)
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


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    use = df[cols].copy() if cols else df.copy()
    use = use.sort_index(axis=1)
    return hashlib.sha256(use.to_csv(index=False).encode("utf-8")).hexdigest()


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


def model() -> Any:
    return xgb.XGBClassifier(**MODEL_PARAMS)


def fit_or_load_model(path: Path, x: pd.DataFrame, y: pd.Series) -> tuple[Any, str, str]:
    assert_research_output_path(path)
    if path.exists():
        return joblib.load(path), file_sha256(path) or "", "loaded_existing_research_model"
    clf = model()
    clf.fit(x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    return clf, file_sha256(path) or "", "trained_new_research_model"


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
        raise ValueError(f"Faithful grid missing required columns: {missing}")
    grid = grid[cols].copy()
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["fwd_return_1w"] = grid["SPY_close"].shift(-FWD_HOURS) / grid["SPY_close"] - 1.0
    grid["raw_regime"] = grid["fwd_return_1w"].apply(label_regime)
    grid["row_id"] = np.arange(len(grid))
    return grid


def trainable_frame(grid: pd.DataFrame) -> pd.DataFrame:
    need = FEATURES + ["SPY_close", "QQQ_close", "raw_regime"]
    out = grid[(grid["timestamp"] >= TRAIN_START) & grid[need].notna().all(axis=1)].copy()
    out["raw_regime"] = out["raw_regime"].astype(int)
    return out.reset_index(drop=True)


def reference_walkforward() -> pd.DataFrame:
    ref = load_csv(REFERENCE_WALKFORWARD_PATH)
    if "predicted_regime" not in ref.columns:
        raise ValueError(f"Reference walkforward missing predicted_regime: {REFERENCE_WALKFORWARD_PATH}")
    ref = ref[(ref["timestamp"] >= EVAL_START) & (ref["timestamp"] <= EVAL_END)].copy()
    ref["predicted_regime"] = pd.to_numeric(ref["predicted_regime"], errors="coerce").astype("Int64")
    return ref.dropna(subset=["predicted_regime"]).reset_index(drop=True)


def canonical_eval_frame(trainable: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    ref_ts = set(ref["timestamp"])
    out = trainable[
        (trainable["timestamp"] >= EVAL_START)
        & (trainable["timestamp"] <= EVAL_END)
        & trainable["timestamp"].isin(ref_ts)
    ].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


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


@dataclass
class VariantResult:
    name: str
    data_source: str
    signal_source: str
    signals: pd.DataFrame
    fold_manifest: pd.DataFrame


def signal_columns(df: pd.DataFrame, variant: str, label_col: str = "raw_regime") -> pd.DataFrame:
    out = df[["timestamp", label_col, "predicted_regime"]].rename(columns={label_col: "raw_regime"}).copy()
    out["variant"] = variant
    out["regime_name"] = out["predicted_regime"].astype(int).map(REGIME_NAME)
    out["allocation_scale"] = 1.0
    return out[["timestamp", "variant", "raw_regime", "predicted_regime", "regime_name", "allocation_scale"]]


def run_raw_label_wf(trainable: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "A_raw_label_monthly_wf_faithful_grid"
    signals, manifests = [], []
    for fold_start, fold_end in monthly_folds():
        train_df = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)].copy()
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        model_path = MODELS_DIR / variant / "folds" / fold_start.strftime("%Y_%m") / "model.joblib"
        clf, model_hash, model_action = fit_or_load_model(model_path, train_df[FEATURES], train_df["raw_regime"])
        test_df["predicted_regime"] = clf.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
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
                "output_signal_hash": frame_hash(fold_signal),
                "model_path": model_path,
                "model_sha256": model_hash,
                "model_action": model_action,
            }
        )
    return VariantResult(variant, str(FAITHFUL_GRID_PATH), "new raw-label monthly walk-forward", pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def run_recovered_control(trainable: pd.DataFrame, ref: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "B_recovered_full_sample_teacher_replay_control"
    sig = canonical[["timestamp", "raw_regime"]].merge(
        ref[["timestamp", "predicted_regime"]], on="timestamp", how="left"
    )
    if sig["predicted_regime"].isna().any():
        raise ValueError("Recovered reference signal missing canonical timestamps")
    sig["predicted_regime"] = sig["predicted_regime"].astype(int)
    signals = signal_columns(sig, variant)
    manifest = pd.DataFrame(
        [
            {
                "variant": variant,
                "fold_start": EVAL_START,
                "fold_end": FINAL_FOLD_END,
                "teacher_train_start": None,
                "teacher_train_end": None,
                "student_train_start": None,
                "student_train_end": None,
                "train_row_count": None,
                "test_row_count": len(signals),
                "teacher_label_distribution": "loaded_recovered_reference",
                "student_label_distribution": None,
                "student_prediction_distribution": dist(signals["predicted_regime"]),
                "feature_list": FEATURES,
                "model_parameters": "recovered/cached corrected walkforward_signals.csv",
                "input_data_hash": frame_hash(ref[["timestamp", "predicted_regime"]]),
                "output_signal_hash": frame_hash(signals),
                "signal_source_path": str(REFERENCE_WALKFORWARD_PATH),
                "signal_source_sha256": file_sha256(REFERENCE_WALKFORWARD_PATH),
                "leakage_flag": "recovered_forensic_control_not_retrained",
            }
        ]
    )
    return VariantResult(variant, str(FAITHFUL_GRID_PATH), str(REFERENCE_WALKFORWARD_PATH), signals, manifest)


def run_new_full_sample_teacher(trainable: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "C_new_full_sample_teacher_faithful_grid"
    teacher_path = MODELS_DIR / variant / "teacher" / "teacher_model.joblib"
    teacher, teacher_hash, teacher_action = fit_or_load_model(teacher_path, trainable[FEATURES], trainable["raw_regime"])
    pseudo = trainable[["timestamp", "raw_regime", *FEATURES]].copy()
    pseudo["teacher_pseudo_regime"] = teacher.predict(pseudo[FEATURES]).astype(int)
    write_csv(pseudo[["timestamp", "raw_regime", "teacher_pseudo_regime"]], SIGNALS_DIR / "C_new_full_sample_teacher_pseudo_labels.csv")
    signals, manifests = [], []
    for fold_start, fold_end in monthly_folds():
        train_df = pseudo[(pseudo["timestamp"] >= TRAIN_START) & (pseudo["timestamp"] < fold_start)].copy()
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        model_path = MODELS_DIR / variant / "folds" / fold_start.strftime("%Y_%m") / "student_model.joblib"
        student, student_hash, student_action = fit_or_load_model(model_path, train_df[FEATURES], train_df["teacher_pseudo_regime"])
        test_df["predicted_regime"] = student.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
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
                "output_signal_hash": frame_hash(fold_signal),
                "teacher_model_path": teacher_path,
                "teacher_model_sha256": teacher_hash,
                "teacher_model_action": teacher_action,
                "student_model_path": model_path,
                "student_model_sha256": student_hash,
                "student_model_action": student_action,
                "leakage_flag": "new_full_sample_teacher_not_pit",
            }
        )
    return VariantResult(variant, str(FAITHFUL_GRID_PATH), "new full-sample teacher trained on faithful grid", pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def run_pit_teacher_student(trainable: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "D_pit_teacher_student_faithful_grid"
    signals, manifests = [], []
    for fold_start, fold_end in monthly_folds():
        teacher_train = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)].copy()
        student_train = teacher_train.copy()
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        if teacher_train.empty or student_train.empty or test_df.empty:
            continue
        fold_dir = MODELS_DIR / variant / "folds" / fold_start.strftime("%Y_%m")
        teacher, teacher_hash, teacher_action = fit_or_load_model(fold_dir / "teacher_model.joblib", teacher_train[FEATURES], teacher_train["raw_regime"])
        student_train["teacher_pseudo_regime"] = teacher.predict(student_train[FEATURES]).astype(int)
        student, student_hash, student_action = fit_or_load_model(fold_dir / "student_model.joblib", student_train[FEATURES], student_train["teacher_pseudo_regime"])
        test_df["predicted_regime"] = student.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
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
                "output_signal_hash": frame_hash(fold_signal),
                "teacher_model_path": fold_dir / "teacher_model.joblib",
                "teacher_model_sha256": teacher_hash,
                "teacher_model_action": teacher_action,
                "student_model_path": fold_dir / "student_model.joblib",
                "student_model_sha256": student_hash,
                "student_model_action": student_action,
                "leakage_flag": "pit_safe_teacher_train_rows_strictly_before_fold_start",
            }
        )
    return VariantResult(variant, str(FAITHFUL_GRID_PATH), "fold-local PIT teacher/student trained on faithful grid", pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def run_hybrid(raw: pd.DataFrame, pit: pd.DataFrame) -> VariantResult:
    variant = "E_conservative_hybrid_raw_and_pit_agreement"
    a = raw[["timestamp", "raw_regime", "predicted_regime"]].rename(columns={"predicted_regime": "raw_pred"})
    d = pit[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "pit_pred"})
    merged = a.merge(d, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
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
                "test_row_count": len(fold),
                "student_prediction_distribution": dist(fold["predicted_regime"]),
                "feature_list": FEATURES,
                "model_parameters": {"hybrid_rule": "full allocation when raw and PIT agree; otherwise more defensive regime at 50% allocation"},
                "input_data_hash": frame_hash(fold[["timestamp", "raw_pred", "pit_pred", "allocation_scale"]]),
                "output_signal_hash": frame_hash(fold),
                "agreement_rate": float(fold["agree"].mean()),
            }
        )
    return VariantResult(variant, str(FAITHFUL_GRID_PATH), "deterministic hybrid of A and D", signals, pd.DataFrame(manifests))


def run_spy_signal(canonical: pd.DataFrame) -> VariantResult:
    variant = "F_SPY_benchmark_same_price_rows"
    sig = canonical[["timestamp", "raw_regime"]].copy()
    sig["predicted_regime"] = 1
    signals = signal_columns(sig, variant)
    manifest = pd.DataFrame(
        [
            {
                "variant": variant,
                "fold_start": EVAL_START,
                "fold_end": FINAL_FOLD_END,
                "test_row_count": len(signals),
                "student_prediction_distribution": dist(signals["predicted_regime"]),
                "model_parameters": "constant SPY allocation benchmark",
                "output_signal_hash": frame_hash(signals),
            }
        ]
    )
    return VariantResult(variant, str(FAITHFUL_GRID_PATH), "constant SPY benchmark", signals, manifest)


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
        cost = switch_cost if changed else 0.0
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
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
    metrics = metrics_from_daily(daily, variant)
    metrics["hourly_price_update_rows"] = len(out)
    metrics["daily_rows"] = len(daily)
    return out, daily, metrics


def metrics_from_daily(daily: pd.DataFrame, variant: str) -> dict[str, Any]:
    ret = daily["strategy_equity"].astype(float).pct_change().dropna()
    downside = ret[ret < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1])
    dd = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    return {
        "variant": variant,
        "start_timestamp": daily["timestamp"].iloc[0],
        "end_timestamp": daily["timestamp"].iloc[-1],
        "start_equity": INITIAL_CAPITAL,
        "final_equity": final_equity,
        "CAGR": (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max DD": float(dd.min()),
        "methodology": "prior-bar / next-bar allocation, faithful-grid price-update hourly rows, 5 bps switch cost, daily-last equity, fixed 5.0-year CAGR, daily Sharpe/Sortino 252 rf=0",
    }


def annual_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in daily.groupby(pd.to_datetime(daily["timestamp"]).dt.year):
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def signal_match(candidate: pd.DataFrame, reference: pd.DataFrame, name: str) -> dict[str, Any]:
    cand = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "candidate"})
    ref = reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference"})
    cmp = cand.merge(ref, on="timestamp", how="outer", indicator=True)
    both = cmp[cmp["_merge"] == "both"].copy()
    if not both.empty:
        both["candidate"] = pd.to_numeric(both["candidate"], errors="coerce").astype("Int64")
        both["reference"] = pd.to_numeric(both["reference"], errors="coerce").astype("Int64")
        both["match"] = both["candidate"] == both["reference"]
    return {
        "comparison": name,
        "candidate_rows": len(cand),
        "reference_rows": len(ref),
        "overlap_rows": len(both),
        "candidate_only_rows": int((cmp["_merge"] == "left_only").sum()),
        "reference_only_rows": int((cmp["_merge"] == "right_only").sum()),
        "match_rate": float(both["match"].mean()) if len(both) else np.nan,
        "mismatch_count": int((both["match"] != True).sum()) if len(both) else None,
    }


def governance_verdict(name: str, metrics: dict[str, Any], signal_match_ref: float | None) -> str:
    if name.startswith("B_"):
        return "forensic_reference_control"
    if name.startswith("C_"):
        return "research_only"
    if name.startswith("F_"):
        return "benchmark"
    if metrics["CAGR"] > 0.10 and metrics["Sharpe"] >= 0.75 and metrics["Max DD"] >= -0.30:
        return "shadow_candidate"
    if metrics["CAGR"] > 0 and metrics["Max DD"] > -0.50:
        return "research_only"
    return "not_eligible"


def write_audit_pack(summary: pd.DataFrame, signal_matches: pd.DataFrame, annual_map: dict[str, pd.DataFrame]) -> None:
    path = PERFORMANCE_DIR / "audit_pack.xlsx"
    assert_research_output_path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "audit_pack.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            signal_matches.to_excel(writer, sheet_name="signal_matches", index=False)
            for name, frame in annual_map.items():
                frame.to_excel(writer, sheet_name=f"annual_{name[:20]}", index=False)
        shutil.copyfile(tmp, path)


def main() -> None:
    ensure_dirs()
    grid = load_grid()
    trainable = trainable_frame(grid)
    ref = reference_walkforward()
    canonical = canonical_eval_frame(trainable, ref)
    if len(canonical) != len(ref):
        print(f"Warning: canonical timestamps {len(canonical)} differ from reference rows {len(ref)}")

    results = [
        run_raw_label_wf(trainable, canonical),
        run_recovered_control(trainable, ref, canonical),
        run_new_full_sample_teacher(trainable, canonical),
        run_pit_teacher_student(trainable, canonical),
    ]
    results.append(run_hybrid(results[0].signals, results[3].signals))
    results.append(run_spy_signal(canonical))

    ref_signal = results[1].signals
    all_fold_manifest = []
    summary_rows = []
    annual_map = {}
    signal_match_rows = []
    canonical_ts_hashes = {}

    for result in results:
        signal_path = SIGNALS_DIR / f"{result.name}_signals.csv"
        result.signals = result.signals.sort_values("timestamp").reset_index(drop=True)
        write_csv(result.signals, signal_path)
        result.fold_manifest["variant_signal_path"] = str(signal_path)
        result.fold_manifest["variant_signal_sha256"] = file_sha256(signal_path)
        all_fold_manifest.append(result.fold_manifest)
        hourly, daily, metrics = run_backtest(grid, result.signals, result.name, SWITCH_COST)
        write_csv(hourly, PERFORMANCE_DIR / result.name / "equity_curve_hourly_price_updates.csv")
        write_csv(daily, PERFORMANCE_DIR / result.name / "equity_curve_daily_last.csv")
        ann = annual_returns(daily)
        write_csv(ann, PERFORMANCE_DIR / result.name / "annual_returns.csv")
        annual_map[result.name] = ann
        switch_count = int(hourly["position_changed"].astype(bool).sum())
        if result.name.startswith("F_"):
            switch_count = 0
        match_ref = signal_match(result.signals, ref_signal, f"{result.name}_vs_20_84_reference")
        signal_match_rows.append(match_ref)
        canonical_ts_hashes[result.name] = frame_hash(result.signals[["timestamp"]])
        summary_rows.append(
            {
                "Variant": result.name,
                "Data source": result.data_source,
                "Signal source": result.signal_source,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max DD": metrics["Max DD"],
                "Final Equity": metrics["final_equity"],
                "Switch Count": switch_count,
                "Signal Match vs 20.84 Ref": match_ref["match_rate"],
                "Verdict": governance_verdict(result.name, metrics, match_ref["match_rate"]),
            }
        )

    # Requested direct signal comparisons.
    signal_match_rows.extend(
        [
            signal_match(results[2].signals, ref_signal, "C_vs_recovered_corrected_walkforward"),
            signal_match(results[3].signals, ref_signal, "D_vs_recovered_corrected_walkforward"),
            signal_match(results[0].signals, results[3].signals, "A_vs_D"),
        ]
    )

    fold_manifest = pd.concat(all_fold_manifest, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    signal_matches = pd.DataFrame(signal_match_rows).drop_duplicates("comparison")
    write_csv(fold_manifest, MANIFESTS_DIR / "fold_manifest.csv")
    write_json({"folds": fold_manifest.to_dict("records")}, MANIFESTS_DIR / "fold_manifest.json")
    write_csv(summary, PERFORMANCE_DIR / "variant_summary.csv")
    write_csv(signal_matches, FORENSICS_DIR / "signal_match_rates.csv")
    write_audit_pack(summary, signal_matches, annual_map)

    grid_info = {
        "faithful_grid_path": FAITHFUL_GRID_PATH,
        "faithful_grid_sha256": file_sha256(FAITHFUL_GRID_PATH),
        "faithful_grid_rows": len(grid),
        "faithful_grid_min_timestamp": grid["timestamp"].min(),
        "faithful_grid_max_timestamp": grid["timestamp"].max(),
        "reference_walkforward_path": REFERENCE_WALKFORWARD_PATH,
        "reference_walkforward_sha256": file_sha256(REFERENCE_WALKFORWARD_PATH),
        "reference_walkforward_rows": len(ref),
        "reference_full_signal_path": REFERENCE_FULL_SIGNAL_PATH,
        "reference_full_signal_sha256": file_sha256(REFERENCE_FULL_SIGNAL_PATH),
        "canonical_signal_rows": len(canonical),
        "canonical_signal_min_timestamp": canonical["timestamp"].min(),
        "canonical_signal_max_timestamp": canonical["timestamp"].max(),
        "canonical_signal_timestamp_hash": frame_hash(canonical[["timestamp"]]),
        "variant_timestamp_hashes": canonical_ts_hashes,
        "all_variants_same_timestamp_hash": len(set(canonical_ts_hashes.values())) == 1,
        "faithful_replay_root": FAITHFUL_REPLAY_ROOT,
        "performance_reconciliation_root": PERF_RECON_ROOT,
    }
    write_json(grid_info, MANIFESTS_DIR / "input_data_hashes.json")

    b_row = summary[summary["Variant"].str.startswith("B_")].iloc[0].to_dict()
    gap = {
        "target_reference": TARGET_REFERENCE,
        "B_recovered_control": b_row,
        "B_CAGR_abs_gap": float(b_row["CAGR"] - TARGET_REFERENCE["CAGR"]),
        "B_Final_Equity_gap": float(b_row["Final Equity"] - TARGET_REFERENCE["Final Equity"]),
        "interpretation": "B should reproduce the recovered 20.840487% control when the faithful legacy grid price-update row membership and recovered walkforward signals are used.",
    }
    write_json(gap, FORENSICS_DIR / "B_recovered_control_vs_20_84_reference.json")

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
        "reference_walkforward": REFERENCE_WALKFORWARD_PATH,
        "reference_full_signal": REFERENCE_FULL_SIGNAL_PATH,
        "features": FEATURES,
        "model_parameters": MODEL_PARAMS,
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
        "dashboard_writes": False,
        "workflow_or_schedule_changes": False,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")

    d_cagr = float(summary.loc[summary["Variant"].str.startswith("D_"), "CAGR"].iloc[0])
    b_cagr = float(summary.loc[summary["Variant"].str.startswith("B_"), "CAGR"].iloc[0])
    prior_v42_b = 0.060666800409978405
    key_answer = (
        "The prior 6.0667% result was not apples-to-apples because it used current V4 silver. "
        f"On the faithful legacy dataset, the recovered B control CAGR is {b_cagr:.6%}; "
        f"the strict PIT teacher-student CAGR is {d_cagr:.6%}. "
        "Compare those two rows to determine the same-data degradation."
    )
    lines = [
        "# Same-Data Teacher-Student Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output root: `{OUTPUT_ROOT}`",
        f"- Input grid: `{FAITHFUL_GRID_PATH}`",
        f"- Grid rows: `{len(grid)}`",
        f"- Grid timestamp range: `{grid['timestamp'].min()}` to `{grid['timestamp'].max()}`",
        f"- Canonical signal rows: `{len(canonical)}`",
        f"- All variants same timestamp set: `{grid_info['all_variants_same_timestamp_hash']}`",
        "- Production writes: none",
        "",
        "## Variant Summary",
        "",
        md_table(summary),
        "",
        "## Signal Match Rates",
        "",
        md_table(signal_matches),
        "",
        "## Key Answer",
        "",
        key_answer,
        "",
        "## Safety",
        "",
        "Research-only. No production Delta tables, Saxo execution, RT notebooks, dashboard notebooks, workflows/schedules, or model registry assets were modified.",
        "",
    ]
    summary_path = FORENSICS_DIR / "same_data_teacher_student_summary.md"
    assert_research_output_path(summary_path)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(key_answer)
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

# Databricks notebook source
"""MUB-27 V4.2 strict-PIT labeling research runner.

Research-only. This script reads faithful legacy CSV artifacts and writes only
under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_strict_pit_research_2026-05-06/`

It does not write production Delta tables, mutate Databricks jobs/workflows,
change Saxo/token state, update the model registry, or edit live/dashboard/RT
notebooks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
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
OUTPUT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
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
PRIOR_PRESENTATION_SUPPORT_ROOT = BASE / "v4_1_csv" / "presentation_support_2026-05-05"
PRIOR_LINEAGE_ROOT = BASE / "v4_2_b_vs_c_lineage_diff_2026-05-04"

TRAIN_START = pd.Timestamp("2016-01-01")
STRICT_TEACHER_CACHE_START = pd.Timestamp("2017-01-01")
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

RAW_STUDENT_PARAMS = {
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
RECOVERED_TEACHER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.03,
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "merror",
    "tree_method": "hist",
    "random_state": 42,
}
RECOVERED_STUDENT_PARAMS = RAW_STUDENT_PARAMS.copy()

TARGET_REFERENCE = {
    "CAGR": 0.20840487336126445,
    "Sharpe": 1.0023589403061952,
    "Sortino": 1.3729057801036,
    "Max DD": -0.2553851434504447,
    "Final Equity": 12883454.9938051,
}


def ensure_python_deps():
    missing = []
    for module_name, package_name in [
        ("xgboost", "xgboost"),
        ("sklearn", "scikit-learn"),
        ("openpyxl", "openpyxl"),
    ]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    import sklearn
    import xgboost as xgb
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    return xgb, sklearn, PCA, StandardScaler


xgb, sklearn, PCA, StandardScaler = ensure_python_deps()


def ensure_dirs() -> None:
    for path in [MANIFESTS_DIR, SIGNALS_DIR, PERFORMANCE_DIR, FORENSICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_research_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(payload: dict[str, Any], path: Path) -> None:
    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    out = df.copy().astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    use = df[cols].copy() if cols else df.copy()
    return hashlib.sha256(use.sort_index(axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("Expected timestamp column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return normalize_timestamp(pd.read_csv(path, **kwargs))


def label_regime(ret: float) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def classifier(params: dict[str, Any]) -> Any:
    return xgb.XGBClassifier(**params)


def load_grid() -> pd.DataFrame:
    grid = read_csv(FAITHFUL_GRID_PATH)
    cols = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing required columns: {missing}")
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


def reference_walkforward() -> pd.DataFrame:
    ref = read_csv(REFERENCE_WALKFORWARD_PATH)
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


def monthly_folds(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    fold_start = start
    while fold_start < end:
        fold_end = fold_start + relativedelta(months=1)
        folds.append((fold_start, fold_end))
        fold_start = fold_end
    return folds


def dist(series: pd.Series) -> dict[str, int]:
    if series is None or len(series) == 0:
        return {}
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


@dataclass
class VariantResult:
    name: str
    label: str
    data_source: str
    signal_source: str
    pit_status: str
    signals: pd.DataFrame
    fold_manifest: pd.DataFrame


def signal_columns(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = df[["timestamp", "raw_regime", "predicted_regime"]].copy()
    out["variant"] = variant
    out["raw_regime"] = out["raw_regime"].astype(int)
    out["predicted_regime"] = out["predicted_regime"].astype(int)
    out["regime_name"] = out["predicted_regime"].map(REGIME_NAME)
    out["allocation_scale"] = 1.0
    return out[["timestamp", "variant", "raw_regime", "predicted_regime", "regime_name", "allocation_scale"]]


def run_raw_label_wf(trainable: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "A_raw_label_walkforward"
    signals, manifests = [], []
    for fold_start, fold_end in monthly_folds(EVAL_START, FINAL_FOLD_END):
        train_df = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)].copy()
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        clf = classifier(RAW_STUDENT_PARAMS)
        clf.fit(train_df[FEATURES], train_df["raw_regime"])
        test_df["predicted_regime"] = clf.predict(test_df[FEATURES]).astype(int)
        fold_signal = signal_columns(test_df, variant)
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "train_row_count": len(train_df),
                "test_row_count": len(test_df),
                "train_start": train_df["timestamp"].min(),
                "train_end": train_df["timestamp"].max(),
                "label_distribution": dist(train_df["raw_regime"]),
                "prediction_distribution": dist(test_df["predicted_regime"]),
                "feature_mode": "raw_features",
                "model_parameters": RAW_STUDENT_PARAMS,
                "strict_pit": True,
                "input_hash": frame_hash(train_df[["timestamp", "raw_regime", *FEATURES]]),
                "output_signal_hash": frame_hash(fold_signal),
            }
        )
    return VariantResult(
        variant,
        "A raw-label monthly walk-forward",
        str(FAITHFUL_GRID_PATH),
        "fresh strict monthly XGBoost on forward SPY return raw labels",
        "strict_pit",
        pd.concat(signals, ignore_index=True),
        pd.DataFrame(manifests),
    )


def run_reference_control(ref: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "B_reference_teacher_smoothed"
    sig = canonical[["timestamp", "raw_regime"]].merge(
        ref[["timestamp", "predicted_regime"]], on="timestamp", how="left"
    )
    if sig["predicted_regime"].isna().any():
        raise ValueError("Recovered B/C2 reference signal missing canonical timestamps")
    sig["predicted_regime"] = sig["predicted_regime"].astype(int)
    signals = signal_columns(sig, variant)
    manifest = pd.DataFrame(
        [
            {
                "variant": variant,
                "fold_start": EVAL_START,
                "fold_end": FINAL_FOLD_END,
                "test_row_count": len(signals),
                "prediction_distribution": dist(signals["predicted_regime"]),
                "signal_source_path": str(REFERENCE_WALKFORWARD_PATH),
                "signal_source_sha256": file_sha256(REFERENCE_WALKFORWARD_PATH),
                "full_signal_export_path": str(REFERENCE_FULL_SIGNAL_PATH),
                "full_signal_export_sha256": file_sha256(REFERENCE_FULL_SIGNAL_PATH),
                "strict_pit": False,
                "governance_caveat": "non-PIT recovered full-sample teacher-smoothed reference benchmark; loaded for comparison only",
                "output_signal_hash": frame_hash(signals),
            }
        ]
    )
    return VariantResult(
        variant,
        "B/C2 non-PIT teacher-smoothed reference",
        str(FAITHFUL_GRID_PATH),
        str(REFERENCE_WALKFORWARD_PATH),
        "non_pit_reference_only",
        signals,
        manifest,
    )


def run_strict_rolling_teacher_student(trainable: pd.DataFrame, canonical: pd.DataFrame) -> VariantResult:
    variant = "C_strict_rolling_teacher_student"
    signals, manifests, pseudo_batches = [], [], []
    prior_teacher_labels = pd.DataFrame(columns=["timestamp", "teacher_pseudo_regime", *FEATURES])
    for fold_start, fold_end in monthly_folds(STRICT_TEACHER_CACHE_START, FINAL_FOLD_END):
        teacher_train = trainable[
            (trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)
        ].copy()
        fold_rows = trainable[(trainable["timestamp"] >= fold_start) & (trainable["timestamp"] < fold_end)].copy()
        if teacher_train.empty or fold_rows.empty:
            continue
        teacher = classifier(RECOVERED_TEACHER_PARAMS)
        teacher.fit(teacher_train[FEATURES], teacher_train["raw_regime"])

        teacher_fold_labels = fold_rows[["timestamp", "raw_regime", *FEATURES]].copy()
        teacher_fold_labels["teacher_pseudo_regime"] = teacher.predict(fold_rows[FEATURES]).astype(int)
        pseudo_batches.append(
            teacher_fold_labels[["timestamp", "raw_regime", "teacher_pseudo_regime"]].assign(
                fold_start=fold_start, fold_end=fold_end
            )
        )

        eval_test = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        prior_train = prior_teacher_labels[prior_teacher_labels["timestamp"] < fold_start].copy()
        if not eval_test.empty and not prior_train.empty:
            student = classifier(RECOVERED_STUDENT_PARAMS)
            student.fit(prior_train[FEATURES], prior_train["teacher_pseudo_regime"].astype(int))
            eval_test["predicted_regime"] = student.predict(eval_test[FEATURES]).astype(int)
            fold_signal = signal_columns(eval_test, variant)
            signals.append(fold_signal)
            manifests.append(
                {
                    "variant": variant,
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "teacher_train_start": teacher_train["timestamp"].min(),
                    "teacher_train_end": teacher_train["timestamp"].max(),
                    "teacher_train_row_count": len(teacher_train),
                    "teacher_label_target": "current_fold_only",
                    "student_train_start": prior_train["timestamp"].min(),
                    "student_train_end": prior_train["timestamp"].max(),
                    "student_train_row_count": len(prior_train),
                    "test_row_count": len(eval_test),
                    "teacher_raw_label_distribution": dist(teacher_train["raw_regime"]),
                    "student_teacher_label_distribution": dist(prior_train["teacher_pseudo_regime"]),
                    "fold_teacher_pseudo_distribution": dist(teacher_fold_labels["teacher_pseudo_regime"]),
                    "prediction_distribution": dist(eval_test["predicted_regime"]),
                    "teacher_parameters": RECOVERED_TEACHER_PARAMS,
                    "student_parameters": RECOVERED_STUDENT_PARAMS,
                    "strict_pit": True,
                    "pit_rule": "teacher, student, scaler-free raw features use rows strictly before fold_start; current fold pseudo-labels are appended only after prediction",
                    "student_input_hash": frame_hash(prior_train[["timestamp", "teacher_pseudo_regime", *FEATURES]]),
                    "output_signal_hash": frame_hash(fold_signal),
                }
            )

        append_labels = teacher_fold_labels[["timestamp", "teacher_pseudo_regime", *FEATURES]].copy()
        prior_teacher_labels = pd.concat([prior_teacher_labels, append_labels], ignore_index=True)

    if pseudo_batches:
        write_csv(
            pd.concat(pseudo_batches, ignore_index=True),
            SIGNALS_DIR / "C_strict_rolling_teacher_student_fold_teacher_pseudo_labels.csv",
        )
    return VariantResult(
        variant,
        "C strict rolling teacher-student",
        str(FAITHFUL_GRID_PATH),
        "fold-local teachers plus prior-fold-only pseudo-label student",
        "strict_pit",
        pd.concat(signals, ignore_index=True),
        pd.DataFrame(manifests),
    )


def run_pca_strict_xgboost(canonical: pd.DataFrame, trainable: pd.DataFrame) -> VariantResult:
    variant = "D_pca_strict_xgboost"
    signals, manifests = [], []
    for fold_start, fold_end in monthly_folds(EVAL_START, FINAL_FOLD_END):
        train_df = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)].copy()
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(train_df[FEATURES])
        x_test_scaled = scaler.transform(test_df[FEATURES])
        pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
        clf = classifier(RAW_STUDENT_PARAMS)
        clf.fit(x_train_pca, train_df["raw_regime"])
        test_df["predicted_regime"] = clf.predict(x_test_pca).astype(int)
        fold_signal = signal_columns(test_df, variant)
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "train_row_count": len(train_df),
                "test_row_count": len(test_df),
                "train_start": train_df["timestamp"].min(),
                "train_end": train_df["timestamp"].max(),
                "feature_mode": "train-only StandardScaler + train-only PCA retaining 95pct variance",
                "pca_components": int(pca.n_components_),
                "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
                "label_distribution": dist(train_df["raw_regime"]),
                "prediction_distribution": dist(test_df["predicted_regime"]),
                "model_parameters": RAW_STUDENT_PARAMS,
                "strict_pit": True,
                "pit_rule": "scaler and PCA fit on train rows strictly before fold_start only",
                "input_hash": frame_hash(train_df[["timestamp", "raw_regime", *FEATURES]]),
                "output_signal_hash": frame_hash(fold_signal),
            }
        )
    return VariantResult(
        variant,
        "D strict XGBoost with train-only PCA",
        str(FAITHFUL_GRID_PATH),
        "fresh strict monthly XGBoost on train-only PCA features",
        "strict_pit",
        pd.concat(signals, ignore_index=True),
        pd.DataFrame(manifests),
    )


def run_spy_benchmark(canonical: pd.DataFrame) -> VariantResult:
    variant = "E_spy_benchmark"
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
                "prediction_distribution": dist(signals["predicted_regime"]),
                "model_parameters": "constant SPY allocation benchmark",
                "strict_pit": True,
                "output_signal_hash": frame_hash(signals),
            }
        ]
    )
    return VariantResult(
        variant,
        "E SPY benchmark",
        str(FAITHFUL_GRID_PATH),
        "constant SPY allocation on same price-update rows",
        "benchmark",
        signals,
        manifest,
    )


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
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0)
        if len(downside) > 1 and downside.std(ddof=1)
        else 0.0,
        "Max DD": float(dd.min()),
        "methodology": "prior-bar / next-bar allocation, price-update hourly rows, 5 bps switch cost, daily-last equity, fixed 5.0-year CAGR, daily Sharpe/Sortino 252 rf=0",
    }


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
        cost = SWITCH_COST if changed else 0.0
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
    metrics["switch_count"] = int(out["position_changed"].astype(bool).sum())
    if variant == "E_spy_benchmark":
        metrics["switch_count"] = 0
    return out, daily, metrics


def yearly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in daily.groupby(pd.to_datetime(daily["timestamp"]).dt.year):
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "start_equity": float(group["strategy_equity"].iloc[0]),
                "end_equity": float(group["strategy_equity"].iloc[-1]),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    month_key = pd.to_datetime(daily["timestamp"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in daily.groupby(month_key):
        rows.append(
            {
                "month": month,
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "start_equity": float(group["strategy_equity"].iloc[0]),
                "end_equity": float(group["strategy_equity"].iloc[-1]),
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


def confusion_matrix(candidate: pd.DataFrame, actual_col: str, pred_col: str, name: str) -> pd.DataFrame:
    if actual_col not in candidate.columns or pred_col not in candidate.columns:
        return pd.DataFrame()
    crosstab = pd.crosstab(
        candidate[actual_col].astype(int),
        candidate[pred_col].astype(int),
        rownames=[actual_col],
        colnames=[pred_col],
        dropna=False,
    )
    crosstab = crosstab.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    out = crosstab.reset_index().rename(columns={0: "pred_0", 1: "pred_1", 2: "pred_2"})
    out.insert(0, "comparison", name)
    return out


def reference_confusion(candidate: pd.DataFrame, reference: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "candidate"}).merge(
        reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference"}),
        on="timestamp",
        how="inner",
    )
    return confusion_matrix(merged, "reference", "candidate", name)


def regime_distribution(signals: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    vc = signals["predicted_regime"].astype(int).value_counts().sort_index()
    total = int(vc.sum())
    for regime in [0, 1, 2]:
        count = int(vc.get(regime, 0))
        rows.append(
            {
                "variant": variant,
                "regime": regime,
                "regime_name": REGIME_NAME[regime],
                "count": count,
                "share": count / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def governance_verdict(result: VariantResult, metrics: dict[str, Any]) -> str:
    if result.name.startswith("B_"):
        return "non_pit_reference_control"
    if result.name.startswith("E_"):
        return "benchmark"
    if metrics["CAGR"] > 0.10 and metrics["Sharpe"] >= 0.75 and metrics["Max DD"] >= -0.30:
        return "strict_pit_shadow_candidate"
    if metrics["CAGR"] > 0 and metrics["Max DD"] > -0.50:
        return "strict_pit_research_only"
    return "not_eligible"


def write_readme(summary: pd.DataFrame, match_table: pd.DataFrame, best: dict[str, Any]) -> None:
    lines = [
        "# MUB-27 V4.2 Strict-PIT Labeling Research",
        "",
        "This folder is isolated research output. It does not represent a production promotion.",
        "",
        "## Governance Caveats",
        "",
        "- Production V4 Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, live schedules, model registry, and trading logic were not modified.",
        "- Variant B is loaded only as the recovered B/C2 non-PIT reference. It uses full-sample teacher-smoothed labels and is not strict point-in-time evidence.",
        "- Strict-PIT variants train labels, scalers, PCA transforms, and models using rows available strictly before each monthly fold start.",
        "- Labels use next-week SPY forward return thresholds: Expansion > +0.5%, Slowdown between -0.5% and +0.5%, Recession < -0.5%.",
        "- Performance uses prior-bar / next-bar allocation, price-update hourly rows only, 5 bps switch cost, daily-last equity, and 2021-01-01 through 2025-12-31 CAGR.",
        "",
        "## Variant Comparison",
        "",
        md_table(summary),
        "",
        "## Match vs B/C2",
        "",
        md_table(match_table),
        "",
        "## Best Strict-PIT Result",
        "",
        json.dumps(jsonable(best), indent=2, sort_keys=True),
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "README.md")


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
        run_reference_control(ref, canonical),
        run_strict_rolling_teacher_student(trainable, canonical),
        run_pca_strict_xgboost(canonical, trainable),
        run_spy_benchmark(canonical),
    ]

    all_fold_manifests, summary_rows, match_rows, confusion_rows, regime_rows = [], [], [], [], []
    reference_signal = results[1].signals
    daily_paths: dict[str, str] = {}

    for result in results:
        result.signals = result.signals.sort_values("timestamp").reset_index(drop=True)
        signal_path = SIGNALS_DIR / f"{result.name}_signals.csv"
        write_csv(result.signals, signal_path)
        result.fold_manifest["variant_signal_path"] = str(signal_path)
        result.fold_manifest["variant_signal_sha256"] = file_sha256(signal_path)
        all_fold_manifests.append(result.fold_manifest)

        hourly, daily, metrics = run_backtest(grid, result.signals, result.name)
        write_csv(hourly, PERFORMANCE_DIR / f"{result.name}_equity_curve_hourly_price_updates.csv")
        write_csv(daily, PERFORMANCE_DIR / f"{result.name}_equity_curve_daily.csv")
        daily_returns = daily[["timestamp", "report_date", "strategy_daily_return", "spy_daily_return", "strategy_equity", "spy_equity"]].copy()
        write_csv(daily_returns, PERFORMANCE_DIR / f"{result.name}_daily_returns.csv")
        write_csv(yearly_returns(daily), PERFORMANCE_DIR / f"{result.name}_yearly_returns.csv")
        write_csv(monthly_returns(daily), PERFORMANCE_DIR / f"{result.name}_monthly_returns.csv")
        daily_paths[result.name] = str(PERFORMANCE_DIR / f"{result.name}_daily_returns.csv")

        match_ref = signal_match(result.signals, reference_signal, f"{result.name}_vs_B_C2")
        match_rows.append(match_ref)
        confusion_rows.append(confusion_matrix(result.signals, "raw_regime", "predicted_regime", f"{result.name}_vs_raw_labels"))
        if result.name != "B_reference_teacher_smoothed":
            confusion_rows.append(reference_confusion(result.signals, reference_signal, f"{result.name}_vs_B_C2_labels"))
        regime_rows.append(regime_distribution(result.signals, result.name))

        summary_rows.append(
            {
                "variant": result.name,
                "label": result.label,
                "pit_status": result.pit_status,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max DD": metrics["Max DD"],
                "Final Equity": metrics["final_equity"],
                "Switch Count": metrics["switch_count"],
                "Signal Match vs B/C2": match_ref["match_rate"],
                "Verdict": governance_verdict(result, metrics),
            }
        )

    fold_manifest = pd.concat(all_fold_manifests, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    match_table = pd.DataFrame(match_rows)
    confusion = pd.concat([x for x in confusion_rows if not x.empty], ignore_index=True)
    regime_dist = pd.concat(regime_rows, ignore_index=True)

    strict = summary[summary["pit_status"].eq("strict_pit")].copy()
    best_strict = strict.sort_values(["CAGR", "Sharpe"], ascending=False).iloc[0].to_dict()
    b = summary[summary["variant"].eq("B_reference_teacher_smoothed")].iloc[0].to_dict()
    spy = summary[summary["variant"].eq("E_spy_benchmark")].iloc[0].to_dict()
    best_strict["gap_vs_B_C2_CAGR"] = float(best_strict["CAGR"] - b["CAGR"])
    best_strict["gap_vs_B_C2_final_equity"] = float(best_strict["Final Equity"] - b["Final Equity"])
    best_strict["gap_vs_SPY_CAGR"] = float(best_strict["CAGR"] - spy["CAGR"])
    best_strict["gap_vs_SPY_final_equity"] = float(best_strict["Final Equity"] - spy["Final Equity"])

    a_cagr = float(summary.loc[summary["variant"].eq("A_raw_label_walkforward"), "CAGR"].iloc[0])
    c_cagr = float(summary.loc[summary["variant"].eq("C_strict_rolling_teacher_student"), "CAGR"].iloc[0])
    d_cagr = float(summary.loc[summary["variant"].eq("D_pca_strict_xgboost"), "CAGR"].iloc[0])
    conclusion = {
        "strict_C_improves_over_A_raw_label_WF": c_cagr > a_cagr,
        "strict_D_improves_over_A_raw_label_WF": d_cagr > a_cagr,
        "strict_C_CAGR_minus_A": c_cagr - a_cagr,
        "strict_D_CAGR_minus_A": d_cagr - a_cagr,
        "best_strict_variant": best_strict["variant"],
        "best_strict_gap_vs_B_C2_CAGR": best_strict["gap_vs_B_C2_CAGR"],
        "best_strict_gap_vs_SPY_CAGR": best_strict["gap_vs_SPY_CAGR"],
        "recommendation": "strict XGBoost/PCA" if d_cagr > max(a_cagr, c_cagr) else "BiLSTM or TFT strict sequence model",
    }

    write_csv(fold_manifest, MANIFESTS_DIR / "fold_manifest.csv")
    write_json({"folds": fold_manifest.to_dict("records")}, MANIFESTS_DIR / "fold_manifest.json")
    write_csv(summary, FORENSICS_DIR / "variant_comparison.csv")
    write_csv(match_table, FORENSICS_DIR / "match_vs_b_c2.csv")
    write_csv(confusion, FORENSICS_DIR / "confusion_matrices.csv")
    write_csv(regime_dist, FORENSICS_DIR / "regime_distribution.csv")
    write_json({"best_strict": best_strict, "conclusion": conclusion}, FORENSICS_DIR / "strict_pit_conclusion.json")

    input_hashes = {
        "faithful_legacy_grid": {
            "path": FAITHFUL_GRID_PATH,
            "sha256": file_sha256(FAITHFUL_GRID_PATH),
            "rows": len(grid),
            "min_timestamp": grid["timestamp"].min(),
            "max_timestamp": grid["timestamp"].max(),
        },
        "reference_walkforward_signals": {
            "path": REFERENCE_WALKFORWARD_PATH,
            "sha256": file_sha256(REFERENCE_WALKFORWARD_PATH),
            "rows": len(ref),
        },
        "reference_full_signal_export": {
            "path": REFERENCE_FULL_SIGNAL_PATH,
            "sha256": file_sha256(REFERENCE_FULL_SIGNAL_PATH),
        },
        "prior_presentation_support_root": {
            "path": PRIOR_PRESENTATION_SUPPORT_ROOT,
            "exists": PRIOR_PRESENTATION_SUPPORT_ROOT.exists(),
        },
        "prior_lineage_root": {
            "path": PRIOR_LINEAGE_ROOT,
            "exists": PRIOR_LINEAGE_ROOT.exists(),
        },
        "canonical_eval": {
            "rows": len(canonical),
            "min_timestamp": canonical["timestamp"].min(),
            "max_timestamp": canonical["timestamp"].max(),
            "timestamp_hash": frame_hash(canonical[["timestamp"]]),
        },
    }
    write_json(input_hashes, MANIFESTS_DIR / "input_hashes.json")

    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
        "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "output_root": OUTPUT_ROOT,
        "features": FEATURES,
        "raw_label_definition": {
            "forward_hours": FWD_HOURS,
            "expansion": "SPY forward return > +0.5%",
            "slowdown": "-0.5% <= SPY forward return <= +0.5%",
            "recession": "SPY forward return < -0.5%",
        },
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
        "daily_return_paths": daily_paths,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")

    write_readme(summary, match_table, {"best_strict": best_strict, "conclusion": conclusion})

    print(summary.to_string(index=False))
    print(json.dumps(jsonable({"best_strict": best_strict, "conclusion": conclusion}), indent=2, sort_keys=True))
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

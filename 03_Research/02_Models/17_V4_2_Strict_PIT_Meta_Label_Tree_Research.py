# Databricks notebook source
"""MUB-27 V4.2 strict-PIT meta-label tree research.

Research-only walk-forward study that keeps the current strict-PIT XGBoost-style
base signal as the primary allocator, then trains a tree meta-label model to
decide whether risk-on exposure should be accepted, reduced, or vetoed.

The script writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_strict_pit_meta_label_tree_2026-05-13/`

It does not mutate production V4 Delta tables, workflows/jobs, Saxo/token
objects, RT/dashboard notebooks, model registry entries, schedules, or trading
logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_strict_pit_meta_label_tree_2026-05-13"

FAITHFUL_GRID_PATH = (
    BASE
    / "v4_1_csv"
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)
STRICT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
STRICT_COMPARISON_PATH = STRICT_ROOT / "forensics" / "variant_comparison.csv"
STRICT_SIGNAL_DIR = STRICT_ROOT / "signals"
DEFENSIVE_PRICE_ROOT = BASE / "v4_2_defensive_sleeve_prices_2026-05-06"
DEFENSIVE_PRICE_PATH = DEFENSIVE_PRICE_ROOT / "prices" / "defensive_etf_hourly_prices.csv"
LABEL_REDESIGN_ROOT = BASE / "v4_2_label_redesign_strict_pit_2026-05-06"

BASELINE_SIGNAL_FILES = {
    "A_raw_label_walkforward": STRICT_SIGNAL_DIR / "A_raw_label_walkforward_signals.csv",
    "B_reference_teacher_smoothed_non_pit_reference_only": STRICT_SIGNAL_DIR / "B_reference_teacher_smoothed_signals.csv",
    "C_strict_rolling_teacher_student": STRICT_SIGNAL_DIR / "C_strict_rolling_teacher_student_signals.csv",
    "D_pca_strict_xgboost": STRICT_SIGNAL_DIR / "D_pca_strict_xgboost_signals.csv",
    "E_spy_benchmark": STRICT_SIGNAL_DIR / "E_spy_benchmark_signals.csv",
}

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
CAGR_YEARS = 5.0
TRADING_DAYS = 252.0
SWITCH_COST = 0.0005
MAX_ACCEPTED_DRAWDOWN = -0.75

BASE_SIGNAL_NAME = "D_pca_strict_xgboost"
SPY_VARIANT = "E_spy_benchmark"
XGBOOST_VARIANT = "D_pca_strict_xgboost"
B_C2_VARIANT = "B_reference_teacher_smoothed"
HEADLINE_FINANCING_RATE = 0.045
HEADLINE_FINANCING_LABEL = "10Y Treasury + 50bps proxy"
FINANCING_GRID = [HEADLINE_FINANCING_RATE, 0.0, 0.07, 0.10]

RETURN_THRESHOLDS = [0.0025, 0.0050, 0.0075]
DRAWDOWN_THRESHOLDS = [0.050, 0.075, 0.100]
POLICY_A_THRESHOLDS = [0.45, 0.55, 0.65]
POLICY_B_LOW_THRESHOLDS = [0.40, 0.45]
POLICY_B_HIGH_THRESHOLDS = [0.55, 0.65]
POLICY_C_HIGH_THRESHOLDS = [0.55, 0.65]
POLICY_C_LEVERAGE = [1.25, 1.5, 2.0]
MIN_HOLD_DAYS = [1, 3]

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
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "BIL"}


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("sklearn", "scikit-learn")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import brier_score_loss, log_loss, precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    return sklearn, StandardScaler, HistGradientBoostingClassifier, RandomForestClassifier, {
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
        "precision_score": precision_score,
        "roc_auc_score": roc_auc_score,
    }


sklearn, StandardScaler, HistGradientBoostingClassifier, RandomForestClassifier, SK_METRICS = ensure_python_deps()


def optional_imports() -> dict[str, Any]:
    availability: dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier

        availability["xgboost"] = {"available": True, "class": XGBClassifier}
    except Exception as exc:
        availability["xgboost"] = {"available": False, "error": str(exc)}
    try:
        from lightgbm import LGBMClassifier

        availability["lightgbm"] = {"available": True, "class": LGBMClassifier}
    except Exception as exc:
        availability["lightgbm"] = {"available": False, "error": str(exc)}
    try:
        from catboost import CatBoostClassifier

        availability["catboost"] = {"available": True, "class": CatBoostClassifier}
    except Exception as exc:
        availability["catboost"] = {"available": False, "error": str(exc)}
    return availability


OPTIONAL_MODELS = optional_imports()


@dataclass(frozen=True)
class LabelSpec:
    label_type: str
    return_threshold: float
    drawdown_threshold: float

    @property
    def key(self) -> str:
        ret = int(round(self.return_threshold * 10000))
        dd = int(round(self.drawdown_threshold * 1000))
        return f"{self.label_type}_ret_{ret:03d}_dd_{dd:03d}"


@dataclass(frozen=True)
class ModelSpec:
    model_family: str
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def assert_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_text(text: str, name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    path.write_text(text, encoding="utf-8")
    return path


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
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(payload: dict[str, Any], name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def forward_path_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    future_prices = pd.concat([price.shift(-i) for i in range(1, horizon + 1)], axis=1)
    return future_prices.min(axis=1) / price - 1.0


def forward_return(price: pd.Series, horizon: int) -> pd.Series:
    return price.shift(-horizon) / price - 1.0


def load_grid() -> pd.DataFrame:
    grid = read_csv(FAITHFUL_GRID_PATH)
    required = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in required if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing columns: {missing}")
    optional = [c for c in ["BIL_close", "bil_close", "BIL_Close", "BIL_adj_close"] if c in grid.columns]
    grid = grid[required + optional].copy()
    for col in [c for c in grid.columns if c != "timestamp"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid = grid.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    grid["feature_ok"] = grid[FEATURES + ["SPY_close", "QQQ_close"]].notna().all(axis=1)
    grid["row_pos"] = np.arange(len(grid))
    return grid


def load_bil_prices(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    for col in ["BIL_close", "bil_close", "BIL_Close", "BIL_adj_close"]:
        if col in grid.columns:
            out = grid[["timestamp", col]].rename(columns={col: "BIL_close"}).copy()
            out["BIL_close"] = pd.to_numeric(out["BIL_close"], errors="coerce")
            out = out.dropna(subset=["timestamp", "BIL_close"]).sort_values("timestamp").drop_duplicates("timestamp")
            return out, {"source": f"faithful_grid:{col}", "rows": len(out), "sha256": file_sha256(FAITHFUL_GRID_PATH)}
    prices = read_csv(DEFENSIVE_PRICE_PATH)
    if "symbol" in prices.columns:
        prices = prices[prices["symbol"].astype(str).str.upper().eq("BIL")].copy()
    close_col = next((c for c in ["close", "adj_close", "adjusted_close", "BIL_close"] if c in prices.columns), None)
    if close_col is None:
        raise ValueError(f"No BIL close column in {DEFENSIVE_PRICE_PATH}")
    out = prices[["timestamp", close_col]].rename(columns={close_col: "BIL_close"}).copy()
    out["BIL_close"] = pd.to_numeric(out["BIL_close"], errors="coerce")
    out = out.dropna(subset=["timestamp", "BIL_close"]).sort_values("timestamp").drop_duplicates("timestamp")
    return out, {"source": str(DEFENSIVE_PRICE_PATH), "rows": len(out), "sha256": file_sha256(DEFENSIVE_PRICE_PATH)}


def attach_bil(grid: pd.DataFrame, bil_prices: pd.DataFrame) -> pd.DataFrame:
    out = grid.merge(bil_prices[["timestamp", "BIL_close"]], on="timestamp", how="left", suffixes=("", "_bil"))
    if "BIL_close_bil" in out.columns:
        out["BIL_close"] = out["BIL_close"].combine_first(out["BIL_close_bil"]) if "BIL_close" in out.columns else out["BIL_close_bil"]
        out = out.drop(columns=["BIL_close_bil"])
    out["BIL_close"] = out["BIL_close"].ffill()
    return out


def load_base_signals() -> pd.DataFrame:
    path = BASELINE_SIGNAL_FILES[BASE_SIGNAL_NAME]
    sig = read_csv(path)
    if "predicted_regime" not in sig.columns:
        raise ValueError(f"{path} missing predicted_regime")
    sig = sig[["timestamp", "predicted_regime"]].dropna().copy()
    sig["predicted_regime"] = pd.to_numeric(sig["predicted_regime"], errors="coerce").astype("Int64")
    sig = sig.dropna(subset=["predicted_regime"]).copy()
    sig["predicted_regime"] = sig["predicted_regime"].astype(int)
    sig["base_asset"] = sig["predicted_regime"].map(REGIME_TO_ASSET).fillna("BIL")
    sig = sig.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return sig


def load_benchmark_rows() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    prior = read_csv(STRICT_COMPARISON_PATH)
    spy = prior[prior["variant"].eq(SPY_VARIANT)]
    if spy.empty:
        raise ValueError(f"Missing {SPY_VARIANT} in {STRICT_COMPARISON_PATH}")
    xgb = prior[prior["variant"].eq(XGBOOST_VARIANT)]
    bc2 = prior[prior["variant"].astype(str).str.contains("B_reference_teacher_smoothed", regex=False, na=False)]
    return prior, spy.iloc[0].to_dict(), xgb.iloc[0].to_dict() if not xgb.empty else None, bc2.iloc[0].to_dict() if not bc2.empty else None


def build_work_frame(grid: pd.DataFrame, base_signals: pd.DataFrame) -> pd.DataFrame:
    work = grid.merge(base_signals, on="timestamp", how="inner").copy()
    work = work[(work["timestamp"] >= TRAIN_START) & (work["feature_ok"])].copy()
    for asset, col in [("SPY", "SPY_close"), ("QQQ", "QQQ_close"), ("BIL", "BIL_close")]:
        work[f"{asset}_fwd_return_1w"] = forward_return(work[col], FWD_HOURS)
        work[f"{asset}_fwd_path_drawdown_1w"] = forward_path_drawdown(work[col], FWD_HOURS)
    work["label_available_ts"] = work["timestamp"].shift(-FWD_HOURS)
    work["base_fwd_return_1w"] = np.select(
        [work["base_asset"].eq("QQQ"), work["base_asset"].eq("SPY"), work["base_asset"].eq("BIL")],
        [work["QQQ_fwd_return_1w"], work["SPY_fwd_return_1w"], work["BIL_fwd_return_1w"]],
        default=np.nan,
    )
    work["base_fwd_path_drawdown_1w"] = np.select(
        [work["base_asset"].eq("QQQ"), work["base_asset"].eq("SPY"), work["base_asset"].eq("BIL")],
        [work["QQQ_fwd_path_drawdown_1w"], work["SPY_fwd_path_drawdown_1w"], work["BIL_fwd_path_drawdown_1w"]],
        default=np.nan,
    )
    fallback_asset = work["base_asset"].map({"QQQ": "SPY", "SPY": "BIL", "BIL": "BIL"}).fillna("BIL")
    work["fallback_asset"] = fallback_asset
    work["fallback_fwd_return_1w"] = np.select(
        [fallback_asset.eq("SPY"), fallback_asset.eq("BIL")],
        [work["SPY_fwd_return_1w"], work["BIL_fwd_return_1w"]],
        default=work["BIL_fwd_return_1w"],
    )
    work = work.dropna(subset=["label_available_ts", "base_fwd_return_1w", "base_fwd_path_drawdown_1w", "fallback_fwd_return_1w"]).copy()
    work["work_pos"] = np.arange(len(work))
    return work.reset_index(drop=True)


def make_meta_label(work: pd.DataFrame, spec: LabelSpec) -> pd.Series:
    base_ret = work["base_fwd_return_1w"]
    base_dd = work["base_fwd_path_drawdown_1w"]
    excess = base_ret - work["fallback_fwd_return_1w"]
    acceptable_dd = base_dd > -spec.drawdown_threshold
    if spec.label_type == "risk_on_trust":
        label = (work["base_asset"].eq("QQQ") & (base_ret > spec.return_threshold) & acceptable_dd).astype(float)
        label = label.mask(~work["base_asset"].eq("QQQ"), np.nan)
    elif spec.label_type == "general_trust":
        label = ((excess > spec.return_threshold) & acceptable_dd).astype(float)
    elif spec.label_type == "drawdown_veto":
        label = (acceptable_dd & (base_ret > -spec.return_threshold)).astype(float)
    else:
        raise ValueError(f"Unknown label type: {spec.label_type}")
    return label


def label_specs(full_label_grid: bool = False) -> list[LabelSpec]:
    all_specs = [
        LabelSpec(label_type, ret, dd)
        for label_type in ["risk_on_trust", "general_trust", "drawdown_veto"]
        for ret in RETURN_THRESHOLDS
        for dd in DRAWDOWN_THRESHOLDS
    ]
    if full_label_grid:
        return all_specs
    return [LabelSpec("general_trust", 0.0025, 0.100)]


def model_specs(max_models: int | None = None, full_model_grid: bool = False, include_xgboost: bool = False) -> list[ModelSpec]:
    families = ["hist_gradient_boosting"]
    if include_xgboost and OPTIONAL_MODELS["xgboost"]["available"]:
        families.insert(0, "xgboost")
    if full_model_grid:
        if OPTIONAL_MODELS["xgboost"]["available"]:
            families.insert(0, "xgboost")
        if OPTIONAL_MODELS["lightgbm"]["available"]:
            families.append("lightgbm")
        if OPTIONAL_MODELS["catboost"]["available"]:
            families.append("catboost")
        families.append("random_forest")
    deduped = []
    for family in families:
        if family not in deduped:
            deduped.append(family)
    if max_models is not None:
        deduped = deduped[:max_models]
    return [ModelSpec(family) for family in deduped]


def build_model(spec: ModelSpec):
    seed_everything(spec.seed)
    if spec.model_family == "xgboost":
        cls = OPTIONAL_MODELS["xgboost"]["class"]
        return cls(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=spec.seed,
            n_jobs=2,
        )
    if spec.model_family == "lightgbm":
        cls = OPTIONAL_MODELS["lightgbm"]["class"]
        return cls(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=spec.seed,
            n_jobs=2,
            verbose=-1,
        )
    if spec.model_family == "catboost":
        cls = OPTIONAL_MODELS["catboost"]["class"]
        return cls(
            iterations=140,
            depth=4,
            learning_rate=0.04,
            loss_function="Logloss",
            random_seed=spec.seed,
            verbose=False,
            allow_writing_files=False,
        )
    if spec.model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=80,
            random_state=spec.seed,
            n_jobs=2,
            class_weight="balanced_subsample",
        )
    return HistGradientBoostingClassifier(
        max_iter=120,
        max_leaf_nodes=16,
        learning_rate=0.045,
        l2_regularization=0.02,
        random_state=spec.seed,
    )


def predict_probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        return proba.reshape(-1).astype(float)
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        return (1.0 / (1.0 + np.exp(-raw))).astype(float)
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def feature_importance_from_model(model: Any) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).reshape(-1))
    else:
        return {}
    if len(values) != len(FEATURES):
        return {}
    total = float(np.nansum(values))
    if total > 0:
        values = values / total
    return {feature: float(value) for feature, value in zip(FEATURES, values)}


def monthly_folds(work: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(EVAL_START.normalize(), EVAL_END.normalize(), freq="MS")
    folds = []
    for start in starts:
        end = start + pd.offsets.MonthBegin(1)
        if start <= EVAL_END:
            folds.append((pd.Timestamp(start), min(pd.Timestamp(end), FINAL_FOLD_END)))
    return folds


def run_walkforward(work: pd.DataFrame, label_spec: LabelSpec, model_spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = work.copy()
    frame["meta_label"] = make_meta_label(frame, label_spec)
    frame = frame[frame["meta_label"].notna()].copy()
    frame["meta_label"] = frame["meta_label"].astype(int)
    predictions = []
    fold_rows = []
    calibration_rows = []
    importance_rows = []
    for fold_start, fold_end in monthly_folds(frame):
        train = frame[
            (frame["timestamp"] < fold_start)
            & (frame["label_available_ts"] < fold_start)
            & frame["timestamp"].ge(TRAIN_START)
        ].copy()
        test = frame[(frame["timestamp"] >= fold_start) & (frame["timestamp"] < fold_end)].copy()
        if test.empty:
            continue
        if len(train) < 500 or train["meta_label"].nunique() < 2:
            pred = pd.Series(0.5, index=test.index, dtype=float)
            status = "fallback_constant_probability"
            importance = {}
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train[FEATURES].astype(float))
            y_train = train["meta_label"].astype(int).to_numpy()
            x_test = scaler.transform(test[FEATURES].astype(float))
            model = build_model(model_spec)
            model.fit(x_train, y_train)
            pred = pd.Series(predict_probability(model, x_test), index=test.index).clip(0.0, 1.0)
            status = "fit"
            importance = feature_importance_from_model(model)
        out = test[
            [
                "timestamp",
                "predicted_regime",
                "base_asset",
                "fallback_asset",
                "meta_label",
                "base_fwd_return_1w",
                "fallback_fwd_return_1w",
                "base_fwd_path_drawdown_1w",
            ]
        ].copy()
        out["meta_probability"] = pred.to_numpy()
        out["label_spec"] = label_spec.key
        out["label_type"] = label_spec.label_type
        out["return_threshold"] = label_spec.return_threshold
        out["drawdown_threshold"] = label_spec.drawdown_threshold
        out["model_family"] = model_spec.model_family
        out["seed"] = model_spec.seed
        predictions.append(out)

        y_true = out["meta_label"].astype(int)
        y_prob = out["meta_probability"].astype(float)
        try:
            auc = SK_METRICS["roc_auc_score"](y_true, y_prob) if y_true.nunique() > 1 else np.nan
        except Exception:
            auc = np.nan
        try:
            ll = SK_METRICS["log_loss"](y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])
        except Exception:
            ll = np.nan
        fold_rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "label_spec": label_spec.key,
                "label_type": label_spec.label_type,
                "model_family": model_spec.model_family,
                "seed": model_spec.seed,
                "status": status,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_positive_rate": float(train["meta_label"].mean()) if len(train) else np.nan,
                "test_positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
                "brier": float(SK_METRICS["brier_score_loss"](y_true, y_prob)) if len(y_true) else np.nan,
                "log_loss": ll,
                "roc_auc": auc,
                "precision_at_0_5": float(SK_METRICS["precision_score"](y_true, y_prob >= 0.5, zero_division=0)) if len(y_true) else np.nan,
                "strict_pit_train_max_label_available_ts": train["label_available_ts"].max() if len(train) else None,
            }
        )
        for feature, value in importance.items():
            importance_rows.append(
                {
                    "label_spec": label_spec.key,
                    "model_family": model_spec.model_family,
                    "seed": model_spec.seed,
                    "fold_start": fold_start,
                    "feature": feature,
                    "importance": value,
                }
            )
        bins = pd.cut(y_prob, bins=np.linspace(0.0, 1.0, 11), include_lowest=True, duplicates="drop")
        for bin_label, group in out.groupby(bins, observed=False):
            if group.empty:
                continue
            calibration_rows.append(
                {
                    "label_spec": label_spec.key,
                    "model_family": model_spec.model_family,
                    "probability_bin": str(bin_label),
                    "rows": len(group),
                    "mean_probability": float(group["meta_probability"].mean()),
                    "realized_positive_rate": float(group["meta_label"].mean()),
                }
            )
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pred_df, pd.DataFrame(fold_rows), pd.DataFrame(calibration_rows), pd.DataFrame(importance_rows)


def apply_min_hold(target: pd.Series, risk_off: pd.Series, hold_days: int) -> pd.Series:
    if hold_days <= 1 or target.empty:
        return target
    out = target.copy().astype(str)
    hold_rows = max(1, hold_days)
    active = out.iloc[0]
    remaining = 0
    for i in range(1, len(out)):
        if bool(risk_off.iloc[i]):
            active = out.iloc[i]
            remaining = 0
        elif remaining > 0:
            out.iloc[i] = active
            remaining -= 1
        elif out.iloc[i] != active:
            active = out.iloc[i]
            remaining = hold_rows - 1
    return out


def build_policy_signals(pred: pd.DataFrame, policy_name: str, params: dict[str, Any]) -> pd.DataFrame:
    out = pred.copy()
    base = out["base_asset"].astype(str)
    p = pd.to_numeric(out["meta_probability"], errors="coerce").fillna(0.5)
    target = base.copy()
    leverage = pd.Series(1.0, index=out.index, dtype=float)
    if policy_name == "A_hard_veto":
        threshold = float(params["threshold"])
        safe_asset = str(params["safe_asset"])
        target = target.mask(base.eq("QQQ") & p.lt(threshold), safe_asset)
    elif policy_name == "B_confidence_scaling":
        low = float(params["low_threshold"])
        high = float(params["high_threshold"])
        target = target.mask(base.eq("QQQ") & p.lt(high) & p.ge(low), "SPY")
        target = target.mask(base.eq("QQQ") & p.lt(low), "BIL")
        target = target.mask(base.eq("SPY") & p.lt(low), "BIL")
    elif policy_name == "C_leverage_permission":
        high = float(params["high_threshold"])
        max_lev = float(params["max_leverage"])
        target = target.mask(base.eq("QQQ") & p.lt(high), "SPY")
        leverage = leverage.mask(base.eq("QQQ") & p.ge(high), max_lev)
    elif policy_name == "D_min_hold_smoothing":
        threshold = float(params["threshold"])
        safe_asset = str(params["safe_asset"])
        hold_days = int(params["min_hold_days"])
        raw = target.mask(base.eq("QQQ") & p.lt(threshold), safe_asset)
        target = apply_min_hold(raw.reset_index(drop=True), (base.eq("BIL") | raw.eq("BIL")).reset_index(drop=True), hold_days)
        target.index = out.index
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    out["candidate_base"] = out["label_spec"] + "__" + out["model_family"] + "__" + policy_name + "__" + pd.Series(json.dumps(params, sort_keys=True), index=out.index)
    out["policy_name"] = policy_name
    out["policy_params"] = json.dumps(params, sort_keys=True)
    out["target_asset_signal"] = target
    out["daily_leverage_signal"] = leverage
    return out


def policy_specs(include_min_hold: bool = False) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    for threshold in POLICY_A_THRESHOLDS:
        for safe_asset in ["SPY", "BIL"]:
            specs.append(("A_hard_veto", {"threshold": threshold, "safe_asset": safe_asset}))
    for low in POLICY_B_LOW_THRESHOLDS:
        for high in POLICY_B_HIGH_THRESHOLDS:
            if low < high:
                specs.append(("B_confidence_scaling", {"low_threshold": low, "high_threshold": high}))
    for high in POLICY_C_HIGH_THRESHOLDS:
        for lev in POLICY_C_LEVERAGE:
            specs.append(("C_leverage_permission", {"high_threshold": high, "max_leverage": lev}))
    if include_min_hold:
        for hold in MIN_HOLD_DAYS:
            specs.append(("D_min_hold_smoothing", {"threshold": 0.55, "safe_asset": "SPY", "min_hold_days": hold}))
    return specs


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        "timestamp",
        "candidate_base",
        "label_spec",
        "label_type",
        "model_family",
        "seed",
        "policy_name",
        "policy_params",
        "predicted_regime",
        "base_asset",
        "target_asset_signal",
        "daily_leverage_signal",
        "meta_probability",
        "meta_label",
    ]
    sig = signals[sig_cols].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close", "BIL_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close", "BIL_close"]] = data[["SPY_close", "QQQ_close", "BIL_close"]].ffill()
    data["price_update"] = data[["SPY_close", "QQQ_close"]].diff().abs().sum(axis=1).gt(0)
    if not data.empty:
        data.loc[data.index[0], "price_update"] = True
    return data[data["price_update"]].dropna(subset=["SPY_close", "QQQ_close", "BIL_close"]).reset_index(drop=True)


def run_unlevered_backtest(grid: pd.DataFrame, policy_signals: pd.DataFrame) -> pd.DataFrame:
    data = price_update_frame(grid, policy_signals)
    if len(data) < 2:
        return pd.DataFrame()
    equity = [INITIAL_CAPITAL]
    spy_equity = [INITIAL_CAPITAL]
    returns = [0.0]
    changed = [False]
    assets = ["START"]
    lev_signal = [1.0]
    active_asset = "START"
    for idx in range(1, len(data)):
        signal_idx = idx - 1
        target_asset = str(data["target_asset_signal"].iloc[signal_idx])
        spy_ret = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        qqq_ret = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        bil_ret = data["BIL_close"].iloc[idx] / data["BIL_close"].iloc[idx - 1] - 1.0
        raw_ret = qqq_ret if target_asset == "QQQ" else spy_ret if target_asset == "SPY" else bil_ret
        is_changed = target_asset != active_asset
        ret = raw_ret - (SWITCH_COST if is_changed else 0.0)
        equity.append(equity[-1] * (1.0 + ret))
        spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))
        returns.append(ret)
        changed.append(is_changed)
        assets.append(target_asset)
        lev_signal.append(float(data["daily_leverage_signal"].iloc[signal_idx]))
        active_asset = target_asset
    out = data.copy()
    out["target_asset"] = assets
    out["strategy_return"] = returns
    out["position_changed"] = changed
    out["strategy_equity"] = equity
    out["spy_equity"] = spy_equity
    out["daily_leverage_signal_applied"] = lev_signal
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
    return daily


def fixed_or_dynamic_leverage_curve(daily: pd.DataFrame, leverage_policy: str, leverage: float | None, financing: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    returns: list[float] = []
    equities: list[float] = []
    drawdowns: list[float] = []
    applied: list[float] = []
    breached_zero = False
    for row in daily.itertuples(index=False):
        lev = float(leverage) if leverage_policy == "fixed" else float(getattr(row, "daily_leverage_signal_applied", 1.0))
        raw = float(getattr(row, "strategy_daily_return"))
        ret = lev * raw - max(lev - 1.0, 0.0) * financing / TRADING_DAYS
        if breached_zero:
            ret = 0.0
            equity = 0.0
        elif 1.0 + ret <= 0.0:
            breached_zero = True
            ret = -1.0
            equity = 0.0
        else:
            equity *= 1.0 + ret
            peak = max(peak, equity)
        returns.append(float(ret))
        equities.append(float(equity))
        drawdowns.append(-1.0 if breached_zero else float(equity / peak - 1.0))
        applied.append(lev)
    return pd.Series(returns), pd.Series(equities), pd.Series(drawdowns), pd.Series(applied), breached_zero


def metric_row(
    candidate_id: str,
    daily: pd.DataFrame,
    returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    applied_leverage: pd.Series,
    breached_zero: bool,
    leverage_policy: str,
    leverage: float | None,
    financing: float,
) -> dict[str, Any]:
    risk_returns = returns.iloc[1:] if len(returns) > 1 else returns
    downside = risk_returns[risk_returns < 0]
    final_equity = float(equity.iloc[-1]) if len(equity) else INITIAL_CAPITAL
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    cagr = -1.0 if breached_zero else float((final_equity / INITIAL_CAPITAL) ** (1.0 / CAGR_YEARS) - 1.0)
    ret_std = risk_returns.std(ddof=1)
    sharpe = float(risk_returns.mean() / ret_std * math.sqrt(TRADING_DAYS)) if ret_std else np.nan
    down_std = downside.std(ddof=1)
    sortino = float(risk_returns.mean() / down_std * math.sqrt(TRADING_DAYS)) if len(downside) > 1 and down_std else np.nan
    vol = float(ret_std * math.sqrt(TRADING_DAYS)) if ret_std else np.nan
    return {
        "candidate_id": candidate_id,
        "label_spec": str(daily["label_spec"].iloc[0]),
        "label_type": str(daily["label_type"].iloc[0]),
        "model_family": str(daily["model_family"].iloc[0]),
        "seed": int(daily["seed"].iloc[0]),
        "policy_name": str(daily["policy_name"].iloc[0]),
        "policy_params": str(daily["policy_params"].iloc[0]),
        "leverage_policy": leverage_policy,
        "leverage": leverage,
        "annual_financing_rate": financing,
        "headline_financing": bool(abs(financing - HEADLINE_FINANCING_RATE) < 1e-12),
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "Final Equity": final_equity,
        "annualized_volatility": vol,
        "min_equity": float(equity.min()) if len(equity) else np.nan,
        "breached_zero": bool(breached_zero),
        "max_dd_floor_breach": bool(max_dd < MAX_ACCEPTED_DRAWDOWN),
        "Switch Count": int(daily["position_changed"].astype(bool).sum()),
        "daily_rows": int(len(daily)),
        "mean_meta_probability": float(pd.to_numeric(daily["meta_probability"], errors="coerce").mean()),
        "trust_label_precision_at_0_5": float(SK_METRICS["precision_score"](daily["meta_label"].astype(int), daily["meta_probability"].ge(0.5), zero_division=0)),
        "mean_applied_leverage": float(applied_leverage.mean()) if len(applied_leverage) else leverage,
    }


def evaluate_candidate(grid: pd.DataFrame, signals: pd.DataFrame, policy_name: str, params: dict[str, Any]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    policy_signals = build_policy_signals(signals, policy_name, params)
    daily = run_unlevered_backtest(grid, policy_signals)
    if daily.empty:
        return [], []
    rows = []
    curves = []
    candidate_base = str(policy_signals["candidate_base"].iloc[0])
    if policy_name == "C_leverage_permission":
        leverage_runs = [("dynamic", None)]
    else:
        leverage_runs = [("fixed", 1.0)]
    for leverage_policy, leverage in leverage_runs:
        for financing in FINANCING_GRID:
            returns, equity, drawdown, applied, breached = fixed_or_dynamic_leverage_curve(daily, leverage_policy, leverage, financing)
            lev_tag = "dynamic" if leverage is None else str(leverage).replace(".", "_")
            candidate_id = f"{candidate_base}__{leverage_policy}_{lev_tag}__fin_{financing:g}"
            rows.append(metric_row(candidate_id, daily, returns, equity, drawdown, applied, breached, leverage_policy, leverage, financing))
            curves.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidate_id,
                        "report_date": daily["report_date"].astype(str).to_numpy(),
                        "candidate_daily_return": returns.to_numpy(),
                        "candidate_equity": equity.to_numpy(),
                        "drawdown": drawdown.to_numpy(),
                        "applied_leverage": applied.to_numpy(),
                    }
                )
            )
    return rows, curves


def add_benchmark_gaps(summary: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None, bc2_row: dict[str, Any] | None) -> pd.DataFrame:
    out = summary.copy()
    out["SPY_CAGR"] = float(spy_row["CAGR"])
    out["SPY_Sharpe"] = float(spy_row["Sharpe"])
    out["SPY_Max DD"] = float(spy_row["Max DD"])
    out["beats_spy_cagr"] = out["CAGR"] > out["SPY_CAGR"]
    out["beats_spy_sharpe"] = out["Sharpe"] > out["SPY_Sharpe"]
    out["gap_vs_spy_cagr"] = out["CAGR"] - out["SPY_CAGR"]
    out["gap_vs_spy_sharpe"] = out["Sharpe"] - out["SPY_Sharpe"]
    if xgb_row is not None:
        out["XGBoost_CAGR"] = float(xgb_row["CAGR"])
        out["XGBoost_Sharpe"] = float(xgb_row["Sharpe"])
        out["gap_vs_xgboost_cagr"] = out["CAGR"] - out["XGBoost_CAGR"]
        out["gap_vs_xgboost_sharpe"] = out["Sharpe"] - out["XGBoost_Sharpe"]
    else:
        out["XGBoost_CAGR"] = np.nan
        out["XGBoost_Sharpe"] = np.nan
        out["gap_vs_xgboost_cagr"] = np.nan
        out["gap_vs_xgboost_sharpe"] = np.nan
    if bc2_row is not None:
        out["B_C2_non_pit_CAGR"] = float(bc2_row["CAGR"])
        out["B_C2_non_pit_Sharpe"] = float(bc2_row["Sharpe"])
        out["gap_vs_b_c2_non_pit_cagr"] = out["CAGR"] - out["B_C2_non_pit_CAGR"]
        out["gap_vs_b_c2_non_pit_sharpe"] = out["Sharpe"] - out["B_C2_non_pit_Sharpe"]
    return out


def candidate_passes(row: pd.Series) -> bool:
    xgb_ok = True
    if pd.notna(row.get("XGBoost_CAGR", np.nan)):
        xgb_ok = bool(row["CAGR"] > row["XGBoost_CAGR"] and row["Sharpe"] > row["XGBoost_Sharpe"])
    return bool(
        row["beats_spy_cagr"]
        and row["beats_spy_sharpe"]
        and xgb_ok
        and row["Max DD"] >= MAX_ACCEPTED_DRAWDOWN
        and not row["breached_zero"]
        and not row["max_dd_floor_breach"]
        and row["headline_financing"]
    )


def yearly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in curves.groupby("candidate_id"):
        g = group.sort_values("report_date").copy()
        years = pd.to_datetime(g["report_date"]).dt.year
        for year, y in g.groupby(years):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "year": int(year),
                    "strategy_return": float(y["candidate_equity"].iloc[-1] / y["candidate_equity"].iloc[0] - 1.0),
                    "start_equity": float(y["candidate_equity"].iloc[0]),
                    "end_equity": float(y["candidate_equity"].iloc[-1]),
                    "rows": len(y),
                }
            )
    return pd.DataFrame(rows)


def monthly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in curves.groupby("candidate_id"):
        g = group.sort_values("report_date").copy()
        months = pd.to_datetime(g["report_date"]).dt.to_period("M").astype(str)
        for month, m in g.groupby(months):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "month": str(month),
                    "strategy_return": float(m["candidate_equity"].iloc[-1] / m["candidate_equity"].iloc[0] - 1.0),
                    "start_equity": float(m["candidate_equity"].iloc[0]),
                    "end_equity": float(m["candidate_equity"].iloc[-1]),
                    "rows": len(m),
                }
            )
    return pd.DataFrame(rows)


def drawdown_windows(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, group in curves.groupby("candidate_id"):
        g = group.sort_values("report_date").reset_index(drop=True)
        if g.empty:
            continue
        trough_idx = int(g["drawdown"].astype(float).idxmin())
        peak_idx = int(g.loc[:trough_idx, "candidate_equity"].astype(float).idxmax())
        recovery = g[(g.index > trough_idx) & (g["candidate_equity"] >= g["candidate_equity"].iloc[peak_idx])]
        rows.append(
            {
                "candidate_id": candidate_id,
                "peak_date": str(g["report_date"].iloc[peak_idx]),
                "trough_date": str(g["report_date"].iloc[trough_idx]),
                "recovery_date": str(recovery["report_date"].iloc[0]) if len(recovery) else None,
                "max_drawdown": float(g["drawdown"].min()),
                "peak_equity": float(g["candidate_equity"].iloc[peak_idx]),
                "trough_equity": float(g["candidate_equity"].iloc[trough_idx]),
            }
        )
    return pd.DataFrame(rows)


def regime_distribution(policy_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["label_spec", "model_family", "policy_name", "policy_params"]
    for keys, group in policy_predictions.groupby(group_cols):
        total = len(group)
        base = dict(zip(group_cols, keys))
        for asset, count in group["target_asset_signal"].astype(str).value_counts().sort_index().items():
            rows.append({**base, "target_asset": asset, "count": int(count), "share": float(count / total) if total else np.nan})
    return pd.DataFrame(rows)


def policy_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["policy_name", "headline_financing"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
        )
        .sort_values(["headline_financing", "best_Sharpe", "best_CAGR"], ascending=[False, False, False])
    )


def leverage_policy_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["leverage_policy", "leverage", "annual_financing_rate"], dropna=False, as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
        )
        .sort_values(["leverage_policy", "leverage", "annual_financing_rate"])
    )


def financing_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("annual_financing_rate", as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
        )
        .sort_values("annual_financing_rate")
    )


def add_acceptance_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["passes_acceptance"] = out.apply(candidate_passes, axis=1)
    return out


def candidate_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    out = add_acceptance_flags(summary)
    out["financing_sensitivity_penalty"] = out.groupby(["label_spec", "model_family", "policy_name", "policy_params", "leverage_policy", "leverage"], dropna=False)["CAGR"].transform("std").fillna(0.0)
    passing = out[out["passes_acceptance"]].copy()
    rest = out[~out["passes_acceptance"]].copy()
    sort_cols = ["Sharpe", "CAGR", "Max DD", "Switch Count", "financing_sensitivity_penalty"]
    passing = passing.sort_values(sort_cols, ascending=[False, False, False, True, True])
    rest = rest.sort_values(sort_cols, ascending=[False, False, False, True, True])
    return pd.concat([passing, rest], ignore_index=True)


def compact_vs(summary: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if benchmark == "spy":
        cols = ["candidate_id", "policy_name", "model_family", "leverage_policy", "leverage", "annual_financing_rate", "CAGR", "SPY_CAGR", "gap_vs_spy_cagr", "Sharpe", "SPY_Sharpe", "gap_vs_spy_sharpe", "Max DD", "SPY_Max DD", "breached_zero", "passes_acceptance"]
    elif benchmark == "xgboost":
        cols = ["candidate_id", "policy_name", "model_family", "leverage_policy", "leverage", "annual_financing_rate", "CAGR", "XGBoost_CAGR", "gap_vs_xgboost_cagr", "Sharpe", "XGBoost_Sharpe", "gap_vs_xgboost_sharpe", "Max DD", "breached_zero", "passes_acceptance"]
    else:
        cols = ["candidate_id", "policy_name", "model_family", "leverage_policy", "leverage", "annual_financing_rate", "CAGR", "B_C2_non_pit_CAGR", "gap_vs_b_c2_non_pit_cagr", "Sharpe", "B_C2_non_pit_Sharpe", "gap_vs_b_c2_non_pit_sharpe", "Max DD", "breached_zero", "passes_acceptance"]
        cols = [c for c in cols if c in summary.columns]
    return summary[cols].copy()


def write_runtime_manifest(args: argparse.Namespace, started_at: datetime, bil_meta: dict[str, Any], selected_models: list[ModelSpec]) -> None:
    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "output_root": OUTPUT_ROOT,
            "research_leg": "strict-PIT meta-label tree",
            "base_signal": BASE_SIGNAL_NAME,
            "features": FEATURES,
            "label_types": ["risk_on_trust", "general_trust", "drawdown_veto"],
            "return_thresholds": RETURN_THRESHOLDS,
            "drawdown_thresholds": DRAWDOWN_THRESHOLDS,
            "selected_label_specs": [spec.key for spec in label_specs(full_label_grid=args.full_label_grid)],
            "horizon_hours": FWD_HOURS,
            "policy_specs": policy_specs(include_min_hold=args.include_min_hold),
            "selected_model_families": [spec.model_family for spec in selected_models],
            "full_model_grid": bool(args.full_model_grid),
            "full_label_grid": bool(args.full_label_grid),
            "bounded_first_run_note": "Default run uses sklearn HistGradientBoosting on the primary general_trust_ret_025_dd_100 spec selected from checkpointed diagnostics. XGBoost and full label/model grids are available behind --include-xgboost, --full-label-grid, and --full-model-grid for an overnight run.",
            "financing_grid": FINANCING_GRID,
            "headline_financing_rate": HEADLINE_FINANCING_RATE,
            "headline_financing_label": HEADLINE_FINANCING_LABEL,
            "package_availability": {k: {kk: vv for kk, vv in v.items() if kk != "class"} for k, v in OPTIONAL_MODELS.items()},
            "sklearn_version": sklearn.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "bil_price_source": bil_meta,
            "strict_pit_controls": [
                "all fold scalers fit on training rows only",
                "training rows require label_available_ts strictly before fold_start",
                "test fold is never used for feature scaling, model fitting, or threshold selection",
                "folds are monthly from 2021-01-01 through 2025-12-31",
            ],
            "production_delta_writes": False,
            "registry_writes": False,
            "workflow_or_schedule_changes": False,
            "execution_notebook_changes": False,
            "dashboard_notebook_changes": False,
            "trading_logic_changes": False,
            "runtime_args": vars(args),
        },
        "runtime_manifest.json",
    )


def write_input_hashes(grid: pd.DataFrame, bil_meta: dict[str, Any], prior: pd.DataFrame) -> None:
    write_json(
        {
            "faithful_legacy_grid": {"path": FAITHFUL_GRID_PATH, "sha256": file_sha256(FAITHFUL_GRID_PATH), "rows": len(grid)},
            "strict_pit_comparison": {"path": STRICT_COMPARISON_PATH, "sha256": file_sha256(STRICT_COMPARISON_PATH), "rows": len(prior)},
            "baseline_signal_files": {name: {"path": path, "sha256": file_sha256(path)} for name, path in BASELINE_SIGNAL_FILES.items()},
            "defensive_bil_prices": bil_meta,
            "label_redesign_root": {"path": LABEL_REDESIGN_ROOT, "exists": LABEL_REDESIGN_ROOT.exists()},
            "financing_assumption": {
                "headline_label": HEADLINE_FINANCING_LABEL,
                "headline_rate": HEADLINE_FINANCING_RATE,
                "note": "Exact dated 10Y Treasury series was not required by the first run; 4.5% is retained as the prior deck-consistent 10Y+50bps proxy unless a future artifact supplies a dated series.",
            },
        },
        "input_hashes.json",
    )


def md_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "_None_"
    view = df.head(max_rows).copy()
    columns = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in view.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6g}")
            else:
                text = "" if pd.isna(value) else str(value)
                values.append(text.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_readme(summary: pd.DataFrame, rankings: pd.DataFrame, meta: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None) -> None:
    accepted = rankings[rankings["passes_acceptance"].eq(True)] if not rankings.empty else pd.DataFrame()
    top = rankings.head(10) if not rankings.empty else pd.DataFrame()
    lines = [
        "# MUB-27 V4.2 Strict-PIT Meta-Label Tree Research",
        "",
        "Research-only test of tree meta-label models that decide whether to trust, reduce, or veto the base strict-PIT XGBoost-style allocation.",
        "",
        "## Scope",
        "",
        f"- Base signal: `{BASE_SIGNAL_NAME}`.",
        "- Original 14 V4 features only; no calendar features.",
        f"- Headline financing: {HEADLINE_FINANCING_LABEL} = {HEADLINE_FINANCING_RATE:.2%}.",
        "- B/C2 teacher-smoothed comparison, when present, is non-PIT reference only.",
        "",
        "## Benchmarks",
        "",
        f"- SPY CAGR: {float(spy_row['CAGR']):.6f}",
        f"- SPY Sharpe: {float(spy_row['Sharpe']):.6f}",
        f"- XGBoost reference available: {xgb_row is not None}",
        "",
        "## Accepted Candidates",
        "",
        md_table(accepted),
        "",
        "## Top Candidates",
        "",
        md_table(top),
        "",
        "## Meta-Label Diagnostics",
        "",
        md_table(meta.sort_values(["roc_auc", "brier"], ascending=[False, True]) if not meta.empty else meta),
        "",
        "## Production Safety",
        "",
        "No production V4 tables, jobs, Saxo/token objects, RT/dashboard notebooks, model registry entries, schedules, or trading logic were changed.",
    ]
    write_text("\n".join(lines), "README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict-PIT V4.2 meta-label tree research.")
    parser.add_argument("--max-label-specs", type=int, default=None, help="Optional debug cap for label specs.")
    parser.add_argument("--max-models", type=int, default=None, help="Optional debug cap for model families.")
    parser.add_argument("--full-label-grid", action="store_true", help="Run all label families across all threshold combinations.")
    parser.add_argument("--full-model-grid", action="store_true", help="Run optional LightGBM/CatBoost/RandomForest families when available.")
    parser.add_argument("--include-xgboost", action="store_true", help="Include XGBoost in addition to the default fast sklearn tree model.")
    parser.add_argument("--include-min-hold", action="store_true", help="Run optional minimum-hold smoothing policy.")
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    ensure_dirs()
    grid_raw = load_grid()
    bil_prices, bil_meta = load_bil_prices(grid_raw)
    grid = attach_bil(grid_raw, bil_prices)
    base_signals = load_base_signals()
    prior, spy_row, xgb_row, bc2_row = load_benchmark_rows()
    models = model_specs(args.max_models, full_model_grid=args.full_model_grid, include_xgboost=args.include_xgboost)
    write_runtime_manifest(args, started_at, bil_meta, models)
    write_input_hashes(grid, bil_meta, prior)

    work = build_work_frame(grid, base_signals)
    specs = label_specs(full_label_grid=args.full_label_grid)
    if args.max_label_specs is not None:
        specs = specs[: args.max_label_specs]
    policies = policy_specs(include_min_hold=args.include_min_hold)

    all_pred_frames: list[pd.DataFrame] = []
    meta_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    policy_signal_frames: list[pd.DataFrame] = []
    total = len(specs) * len(models)
    completed = 0
    write_readme(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), spy_row, xgb_row)
    for label_spec in specs:
        for model_spec in models:
            completed += 1
            print(f"[{completed}/{total}] {label_spec.key} {model_spec.model_family}", flush=True)
            pred, meta, calibration, importance = run_walkforward(work, label_spec, model_spec)
            if pred.empty:
                continue
            all_pred_frames.append(pred)
            meta_frames.append(meta)
            calibration_frames.append(calibration)
            importance_frames.append(importance)
            for policy_name, params in policies:
                policy_signals = build_policy_signals(pred, policy_name, params)
                policy_signal_frames.append(policy_signals[["label_spec", "model_family", "policy_name", "policy_params", "target_asset_signal"]].copy())
                rows, curves = evaluate_candidate(grid, pred, policy_name, params)
                summary_rows.extend(rows)
                curve_frames.extend(curves)
            if summary_rows:
                partial = add_acceptance_flags(add_benchmark_gaps(pd.DataFrame(summary_rows), spy_row, xgb_row, bc2_row))
                partial_rankings = candidate_rankings(partial)
                write_csv(partial, "candidate_summary.csv")
                write_csv(partial_rankings, "candidate_rankings.csv")
                write_readme(partial, partial_rankings, pd.concat(meta_frames, ignore_index=True), spy_row, xgb_row)

    summary = add_acceptance_flags(add_benchmark_gaps(pd.DataFrame(summary_rows), spy_row, xgb_row, bc2_row)) if summary_rows else pd.DataFrame()
    rankings = candidate_rankings(summary) if not summary.empty else pd.DataFrame()
    meta_results = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    if not importance.empty:
        importance = (
            importance.groupby(["label_spec", "model_family", "feature"], as_index=False)
            .agg(mean_importance=("importance", "mean"), folds=("importance", "count"))
            .sort_values(["mean_importance"], ascending=False)
        )
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    policy_predictions = pd.concat(policy_signal_frames, ignore_index=True) if policy_signal_frames else pd.DataFrame()

    write_csv(summary, "candidate_summary.csv")
    write_csv(rankings, "candidate_rankings.csv")
    write_csv(meta_results, "meta_label_results.csv")
    write_csv(policy_results(summary), "policy_results.csv")
    write_csv(leverage_policy_results(summary), "leverage_policy_results.csv")
    write_csv(financing_sensitivity(summary), "financing_sensitivity.csv")
    write_csv(calibration, "calibration_results.csv")
    write_csv(importance, "feature_importance.csv")
    if not curves.empty:
        write_csv(yearly_returns(curves), "yearly_returns.csv")
        write_csv(monthly_returns(curves), "monthly_returns.csv")
        write_csv(drawdown_windows(curves), "drawdown_windows.csv")
    else:
        write_csv(pd.DataFrame(), "yearly_returns.csv")
        write_csv(pd.DataFrame(), "monthly_returns.csv")
        write_csv(pd.DataFrame(), "drawdown_windows.csv")
    write_csv(compact_vs(summary, "spy"), "candidate_vs_spy.csv")
    write_csv(compact_vs(summary, "xgboost"), "candidate_vs_xgboost.csv")
    write_csv(compact_vs(summary, "bc2"), "candidate_vs_b_c2_reference.csv")
    write_csv(regime_distribution(policy_predictions), "regime_distribution.csv")
    write_readme(summary, rankings, meta_results, spy_row, xgb_row)
    print(json.dumps(jsonable({"output_root": OUTPUT_ROOT, "top_10": rankings.head(10).to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()

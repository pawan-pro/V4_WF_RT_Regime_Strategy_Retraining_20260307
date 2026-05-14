# Databricks notebook source
"""MUB-27 V4.2 strict-PIT direct-allocation model-family comparison.

Research-only walk-forward study that trains models to choose the next
allocation directly among QQQ, SPY, and BIL using forward utility labels.

The script writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_direct_allocation_model_family_2026-05-14/`

It does not mutate production V4 Delta tables, workflows/jobs, Saxo/token
objects, RT/dashboard notebooks, model registry entries, schedules, or trading
logic.

No candidate can be accepted unless the input grid validation report passes.
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
OUTPUT_ROOT = BASE / "v4_2_direct_allocation_model_family_2026-05-14"
EXPECTED_GRID_SHA256 = "b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1"

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
GRID_VALIDATION_STATUS = "not_run"

RETURN_THRESHOLDS = [0.0025, 0.0050, 0.0075]
DRAWDOWN_THRESHOLDS = [0.050, 0.075, 0.100]
UTILITY_DRAWDOWN_LAMBDAS = [0.25, 0.50, 1.00]
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
ASSET_TO_CLASS = {"QQQ": 0, "SPY": 1, "BIL": 2}
CLASS_TO_ASSET = {v: k for k, v in ASSET_TO_CLASS.items()}


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
    from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    return sklearn, StandardScaler, HistGradientBoostingClassifier, RandomForestClassifier, {
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
        "precision_score": precision_score,
        "roc_auc_score": roc_auc_score,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
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
    try:
        import hmmlearn

        availability["hmmlearn"] = {"available": True, "version": getattr(hmmlearn, "__version__", "unknown")}
    except Exception as exc:
        availability["hmmlearn"] = {"available": False, "error": str(exc)}
    return availability


OPTIONAL_MODELS = optional_imports()


@dataclass(frozen=True)
class LabelSpec:
    label_type: str
    return_threshold: float
    drawdown_threshold: float
    drawdown_lambda: float = 0.50

    @property
    def key(self) -> str:
        ret = int(round(self.return_threshold * 10000))
        dd = int(round(self.drawdown_threshold * 1000))
        lam = int(round(self.drawdown_lambda * 100))
        return f"{self.label_type}_ret_{ret:03d}_dd_{dd:03d}_lam_{lam:03d}"


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


def _validation_row(check: str, status: str, details: str, value: Any = None, expected: Any = None) -> dict[str, Any]:
    return {
        "check_name": check,
        "status": status,
        "actual_value": json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else value,
        "expected_value": json.dumps(jsonable(expected), sort_keys=True) if isinstance(expected, (dict, list, tuple)) else expected,
        "details": details,
    }


def _find_known_hashes() -> dict[str, Any]:
    known: dict[str, Any] = {}
    for root in [STRICT_ROOT, LABEL_REDESIGN_ROOT, DEFENSIVE_PRICE_ROOT]:
        if not root.exists():
            continue
        for path in list(root.rglob("input_hashes.json"))[:20] + list(root.rglob("runtime_manifest.json"))[:20]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            known[str(path)] = payload
    return known


def validate_grid_inputs(
    grid_raw: pd.DataFrame,
    grid: pd.DataFrame,
    bil_meta: dict[str, Any],
    base_signals: pd.DataFrame,
    prior: pd.DataFrame,
    work: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_hash = file_sha256(FAITHFUL_GRID_PATH)
    rows.append(
        _validation_row(
            "grid_source_path_and_hash",
            "pass" if FAITHFUL_GRID_PATH.exists() and grid_hash == EXPECTED_GRID_SHA256 else "fail",
            "Input grid must be the faithful legacy clone used by prior V4.1/V4.2 research and match the known reconciled hash.",
            {"path": str(FAITHFUL_GRID_PATH), "sha256": grid_hash},
            EXPECTED_GRID_SHA256,
        )
    )
    ts = pd.to_datetime(grid_raw["timestamp"], errors="coerce")
    rows.append(
        _validation_row(
            "row_count_and_timestamp_range",
            "pass" if len(grid_raw) > 0 and ts.notna().all() else "fail",
            "Grid must have parseable timestamps and non-zero rows.",
            {"rows": len(grid_raw), "timestamp_min": ts.min(), "timestamp_max": ts.max()},
        )
    )
    eval_mask = ts.between(EVAL_START, EVAL_END)
    rows.append(
        _validation_row(
            "expected_2021_2025_coverage",
            "pass" if ts.min() <= EVAL_START and ts.max() >= EVAL_END.normalize() and int(eval_mask.sum()) > 0 else "fail",
            "Evaluation window must cover 2021-01-01 through 2025-12-31.",
            {"eval_rows": int(eval_mask.sum()), "timestamp_min": ts.min(), "timestamp_max": ts.max()},
        )
    )
    missing_features = [c for c in FEATURES if c not in grid_raw.columns]
    rows.append(
        _validation_row(
            "original_14_features_present",
            "pass" if not missing_features else "fail",
            "All original 14 V4 features are required; no calendar features are added.",
            {"missing_features": missing_features, "feature_count": len([c for c in FEATURES if c in grid_raw.columns])},
            FEATURES,
        )
    )
    vix_cols = ["VIXCLS_z", "vix3m_z", "VIX_Inversion_Z"]
    coverage = {}
    for col in vix_cols:
        coverage[col] = float(pd.to_numeric(grid_raw.get(col), errors="coerce").loc[eval_mask].notna().mean()) if col in grid_raw.columns and eval_mask.any() else 0.0
    rows.append(
        _validation_row(
            "vix_vix3m_non_null_coverage",
            "pass" if all(v >= 0.95 for v in coverage.values()) else "fail",
            "Reconciled VIX/VIX3M columns must be present and substantially populated over the evaluation period.",
            coverage,
            ">= 95% non-null coverage for each VIX/VIX3M feature",
        )
    )
    cadence = ts.sort_values().diff().dropna()
    cadence_hours = cadence.dt.total_seconds().div(3600.0)
    same_day_check = bool(FAITHFUL_GRID_PATH.name == "full_legacy_grid_clone.csv" and all(c in grid_raw.columns for c in vix_cols))
    rows.append(
        _validation_row(
            "pit_safe_vix_vix3m_alignment",
            "pass" if same_day_check and all(v >= 0.95 for v in coverage.values()) else "grid_unverified",
            "This script does not recompute or same-day join VIX/VIX3M. It requires the reconciled faithful grid artifact and validates VIX/VIX3M feature presence/cadence before training.",
            {
                "grid_file": FAITHFUL_GRID_PATH.name,
                "median_timestamp_step_hours": float(cadence_hours.median()) if len(cadence_hours) else np.nan,
                "vix_columns": vix_cols,
            },
            "faithful legacy grid with reconciled VIX/VIX3M treatment",
        )
    )
    price_cols = ["SPY_close", "QQQ_close", "BIL_close"]
    eval_grid = grid.loc[pd.to_datetime(grid["timestamp"]).between(EVAL_START, EVAL_END)].copy()
    price_non_null = {col: float(pd.to_numeric(eval_grid.get(col), errors="coerce").notna().mean()) if col in eval_grid.columns and len(eval_grid) else 0.0 for col in price_cols}
    price_updates = {
        "SPY_price_update_rows": int(pd.to_numeric(eval_grid["SPY_close"], errors="coerce").diff().abs().gt(0).sum()) if "SPY_close" in eval_grid else 0,
        "QQQ_price_update_rows": int(pd.to_numeric(eval_grid["QQQ_close"], errors="coerce").diff().abs().gt(0).sum()) if "QQQ_close" in eval_grid else 0,
        "BIL_price_update_rows": int(pd.to_numeric(eval_grid["BIL_close"], errors="coerce").diff().abs().gt(0).sum()) if "BIL_close" in eval_grid else 0,
    }
    rows.append(
        _validation_row(
            "spy_qqq_bil_price_alignment",
            "pass" if all(v >= 0.95 for v in price_non_null.values()) and min(price_updates.values()) > 0 else "fail",
            "SPY/QQQ/BIL prices must align to grid timestamps with price-update rows.",
            {"non_null_coverage": price_non_null, "price_update_rows": price_updates, "bil_source": bil_meta},
        )
    )
    grid_ts = set(pd.to_datetime(grid["timestamp"]).dropna())
    sig_ts = set(pd.to_datetime(base_signals["timestamp"]).dropna())
    overlap = len(grid_ts.intersection(sig_ts))
    rows.append(
        _validation_row(
            "benchmark_reference_signal_alignment",
            "pass" if overlap > 0 and not prior.empty else "fail",
            "Base strict-PIT/reference signals must overlap grid timestamps and benchmark comparison rows must exist.",
            {"base_signal": BASE_SIGNAL_NAME, "signal_rows": len(base_signals), "grid_signal_overlap_rows": overlap, "comparison_rows": len(prior)},
        )
    )
    label_ok = bool((work["label_available_ts"] > work["timestamp"]).all() and work["label_available_ts"].notna().all())
    folds_ok = True
    for fold_start, _fold_end in monthly_folds(work):
        train = work[(work["timestamp"] < fold_start) & (work["label_available_ts"] < fold_start) & work["timestamp"].ge(TRAIN_START)]
        if len(train) and not bool((train["label_available_ts"] < fold_start).all()):
            folds_ok = False
            break
    rows.append(
        _validation_row(
            "label_availability_strict_pit",
            "pass" if label_ok and folds_ok else "fail",
            "Training rows may only use labels whose one-week horizon is strictly before fold start.",
            {"work_rows": len(work), "max_label_available_ts": work["label_available_ts"].max() if len(work) else None},
        )
    )
    known_hashes = _find_known_hashes()
    rows.append(
        _validation_row(
            "known_prior_hash_comparison",
            "pass" if known_hashes else "warn",
            "Known prior input/runtime hash artifacts were scanned where available.",
            {"known_artifact_count": len(known_hashes), "current_grid_hash": grid_hash, "defensive_bil_hash": bil_meta.get("sha256"), "b_c2_signal_hash": file_sha256(BASELINE_SIGNAL_FILES["B_reference_teacher_smoothed_non_pit_reference_only"])},
        )
    )
    report = pd.DataFrame(rows)
    hard_fail = report["status"].eq("fail").any()
    grid_unverified = report["status"].eq("grid_unverified").any()
    overall_status = "fail" if hard_fail else "grid_unverified" if grid_unverified else "pass"
    summary = {
        "overall_status": overall_status,
        "grid_source_path": str(FAITHFUL_GRID_PATH),
        "grid_sha256": grid_hash,
        "grid_rows": len(grid_raw),
        "timestamp_min": ts.min(),
        "timestamp_max": ts.max(),
        "eval_rows": int(eval_mask.sum()),
        "vix_vix3m_coverage": coverage,
        "price_non_null_coverage": price_non_null,
        "price_update_rows": price_updates,
        "known_prior_hash_artifacts": len(known_hashes),
    }
    return report, summary


def make_direct_allocation_label(work: pd.DataFrame, spec: LabelSpec) -> pd.Series:
    utilities: dict[str, pd.Series] = {}
    recent_vol = (
        pd.to_numeric(work["SPY_close"], errors="coerce")
        .pct_change()
        .rolling(72, min_periods=24)
        .std()
        .replace(0.0, np.nan)
        .fillna(method="bfill")
        .fillna(0.01)
    )
    for asset in ["QQQ", "SPY", "BIL"]:
        ret = pd.to_numeric(work[f"{asset}_fwd_return_1w"], errors="coerce")
        dd = pd.to_numeric(work[f"{asset}_fwd_path_drawdown_1w"], errors="coerce").clip(upper=0.0).abs()
        switch_penalty = np.where(work["base_asset"].astype(str).eq(asset), 0.0, SWITCH_COST)
        if spec.label_type == "utility_vol_adjusted":
            utility = ret / recent_vol - spec.drawdown_lambda * dd - switch_penalty
        else:
            utility = ret - spec.drawdown_lambda * dd - switch_penalty
        if spec.label_type == "utility_drawdown_capped":
            utility = utility.mask(pd.to_numeric(work[f"{asset}_fwd_path_drawdown_1w"], errors="coerce") < -spec.drawdown_threshold, utility - 10.0)
        utilities[asset] = utility
    util = pd.DataFrame(utilities)
    label = util.idxmax(axis=1).map(ASSET_TO_CLASS).astype(float)
    invalid = util.isna().all(axis=1)
    return label.mask(invalid, np.nan)


def label_specs(full_label_grid: bool = False) -> list[LabelSpec]:
    all_specs = [
        LabelSpec(label_type, 0.0, dd, lam)
        for label_type in ["utility_basic", "utility_drawdown_capped", "utility_vol_adjusted"]
        for dd in DRAWDOWN_THRESHOLDS
        for lam in UTILITY_DRAWDOWN_LAMBDAS
    ]
    if full_label_grid:
        return all_specs
    return [
        LabelSpec("utility_basic", 0.0, 0.100, 0.25),
        LabelSpec("utility_drawdown_capped", 0.0, 0.075, 0.50),
    ]


def model_specs(max_models: int | None = None, full_model_grid: bool = False, include_xgboost: bool = True) -> list[ModelSpec]:
    families = ["hist_gradient_boosting"]
    if include_xgboost and OPTIONAL_MODELS["xgboost"]["available"]:
        families.append("xgboost")
    else:
        families.append("random_forest")
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


def build_model(spec: ModelSpec, n_classes: int):
    seed_everything(spec.seed)
    if spec.model_family == "xgboost":
        cls = OPTIONAL_MODELS["xgboost"]["class"]
        params = dict(
            n_estimators=70,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=spec.seed,
            n_jobs=2,
        )
        if n_classes > 2:
            params["num_class"] = n_classes
        return cls(**params)
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
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=80,
            random_state=spec.seed,
            n_jobs=2,
            class_weight="balanced_subsample",
        )
    return HistGradientBoostingClassifier(
        max_iter=70,
        max_leaf_nodes=16,
        learning_rate=0.045,
        l2_regularization=0.02,
        random_state=spec.seed,
    )


def predict_class_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), 3), dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = getattr(model, "classes_", np.arange(proba.shape[1]))
        for idx, cls in enumerate(classes):
            cls_int = int(cls)
            if 0 <= cls_int <= 2:
                out[:, cls_int] = proba[:, idx]
        row_sum = out.sum(axis=1)
        missing = row_sum <= 0
        out[~missing] = out[~missing] / row_sum[~missing, None]
        out[missing, :] = 1.0 / 3.0
        return out
    pred = model.predict(x)
    for i, cls in enumerate(np.asarray(pred, dtype=int)):
        out[i, cls if 0 <= cls <= 2 else 1] = 1.0
    return out


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
    frame["allocation_label"] = make_direct_allocation_label(frame, label_spec)
    frame = frame[frame["allocation_label"].notna()].copy()
    frame["allocation_label"] = frame["allocation_label"].astype(int)
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
        if len(train) < 500 or train["allocation_label"].nunique() < 2:
            prior = train["allocation_label"].value_counts(normalize=True).to_dict() if len(train) else {}
            proba = np.tile([prior.get(0, 1 / 3), prior.get(1, 1 / 3), prior.get(2, 1 / 3)], (len(test), 1))
            proba = proba / proba.sum(axis=1, keepdims=True)
            status = "fallback_constant_probability"
            importance = {}
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train[FEATURES].astype(float))
            y_train = train["allocation_label"].astype(int).to_numpy()
            x_test = scaler.transform(test[FEATURES].astype(float))
            model = build_model(model_spec, int(len(np.unique(y_train))))
            model.fit(x_train, y_train)
            proba = predict_class_probabilities(model, x_test)
            status = "fit"
            importance = feature_importance_from_model(model)
        out = test[
            [
                "timestamp",
                "predicted_regime",
                "base_asset",
                "fallback_asset",
                "allocation_label",
                "base_fwd_return_1w",
                "fallback_fwd_return_1w",
                "base_fwd_path_drawdown_1w",
            ]
        ].copy()
        out["prob_qqq"] = proba[:, 0]
        out["prob_spy"] = proba[:, 1]
        out["prob_bil"] = proba[:, 2]
        out["predicted_allocation_class"] = np.argmax(proba, axis=1)
        out["predicted_allocation_asset"] = out["predicted_allocation_class"].map(CLASS_TO_ASSET)
        out["allocation_confidence"] = np.max(proba, axis=1)
        out["label_spec"] = label_spec.key
        out["label_type"] = label_spec.label_type
        out["return_threshold"] = label_spec.return_threshold
        out["drawdown_threshold"] = label_spec.drawdown_threshold
        out["drawdown_lambda"] = label_spec.drawdown_lambda
        out["model_family"] = model_spec.model_family
        out["seed"] = model_spec.seed
        predictions.append(out)

        y_true = out["allocation_label"].astype(int)
        y_pred = out["predicted_allocation_class"].astype(int)
        try:
            ll = SK_METRICS["log_loss"](y_true, np.clip(proba, 1e-6, 1.0), labels=[0, 1, 2])
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
                "train_qqq_share": float(train["allocation_label"].eq(0).mean()) if len(train) else np.nan,
                "train_spy_share": float(train["allocation_label"].eq(1).mean()) if len(train) else np.nan,
                "train_bil_share": float(train["allocation_label"].eq(2).mean()) if len(train) else np.nan,
                "test_qqq_share": float(y_true.eq(0).mean()) if len(y_true) else np.nan,
                "test_spy_share": float(y_true.eq(1).mean()) if len(y_true) else np.nan,
                "test_bil_share": float(y_true.eq(2).mean()) if len(y_true) else np.nan,
                "accuracy": float(SK_METRICS["accuracy_score"](y_true, y_pred)) if len(y_true) else np.nan,
                "macro_f1": float(SK_METRICS["f1_score"](y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else np.nan,
                "log_loss": ll,
                "mean_confidence": float(out["allocation_confidence"].mean()) if len(out) else np.nan,
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
        bins = pd.cut(out["allocation_confidence"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True, duplicates="drop")
        for bin_label, group in out.groupby(bins, observed=False):
            if group.empty:
                continue
            calibration_rows.append(
                {
                    "label_spec": label_spec.key,
                    "model_family": model_spec.model_family,
                    "probability_bin": str(bin_label),
                    "rows": len(group),
                    "mean_probability": float(group["allocation_confidence"].mean()),
                    "realized_accuracy": float(group["predicted_allocation_class"].eq(group["allocation_label"]).mean()),
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
    p_qqq = pd.to_numeric(out["prob_qqq"], errors="coerce").fillna(1 / 3)
    p_spy = pd.to_numeric(out["prob_spy"], errors="coerce").fillna(1 / 3)
    p_bil = pd.to_numeric(out["prob_bil"], errors="coerce").fillna(1 / 3)
    target = out["predicted_allocation_asset"].astype(str).copy()
    leverage = pd.Series(1.0, index=out.index, dtype=float)
    if policy_name == "A_direct_allocation":
        target = out["predicted_allocation_asset"].astype(str)
    elif policy_name == "B_probability_weighted":
        high = float(params["high_confidence"])
        risk_off = float(params["risk_off_threshold"])
        target = pd.Series("SPY", index=out.index, dtype=object)
        target = target.mask(p_qqq >= high, "QQQ")
        target = target.mask(p_bil >= risk_off, "BIL")
    elif policy_name == "C_high_conviction_qqq":
        qqq_threshold = float(params["qqq_threshold"])
        risk_off = float(params["risk_off_threshold"])
        margin = float(params["margin"])
        qqq_margin = p_qqq - pd.concat([p_spy, p_bil], axis=1).max(axis=1)
        target = pd.Series("SPY", index=out.index, dtype=object)
        target = target.mask((p_qqq >= qqq_threshold) & (qqq_margin >= margin), "QQQ")
        target = target.mask(p_bil >= risk_off, "BIL")
    elif policy_name == "D_leverage_permission":
        qqq_threshold = float(params["qqq_threshold"])
        risk_off = float(params["risk_off_threshold"])
        max_lev = float(params["max_leverage"])
        margin = float(params["margin"])
        qqq_margin = p_qqq - pd.concat([p_spy, p_bil], axis=1).max(axis=1)
        target = pd.Series("SPY", index=out.index, dtype=object)
        risk_on = (p_qqq >= qqq_threshold) & (qqq_margin >= margin)
        target = target.mask(risk_on, "QQQ")
        target = target.mask(p_bil >= risk_off, "BIL")
        leverage = leverage.mask(risk_on & target.eq("QQQ"), max_lev)
    elif policy_name == "E_min_hold_smoothing":
        hold_days = int(params["min_hold_days"])
        raw = out["predicted_allocation_asset"].astype(str)
        target = apply_min_hold(raw.reset_index(drop=True), raw.eq("BIL").reset_index(drop=True), hold_days)
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
    specs.append(("A_direct_allocation", {}))
    for high in [0.50, 0.60, 0.70]:
        for risk_off in [0.40, 0.50, 0.60]:
            specs.append(("B_probability_weighted", {"high_confidence": high, "risk_off_threshold": risk_off}))
    for qqq_threshold in [0.50, 0.60, 0.70]:
        specs.append(("C_high_conviction_qqq", {"qqq_threshold": qqq_threshold, "risk_off_threshold": 0.50, "margin": 0.05}))
    for qqq_threshold in [0.60, 0.70]:
        for lev in POLICY_C_LEVERAGE:
            specs.append(("D_leverage_permission", {"qqq_threshold": qqq_threshold, "risk_off_threshold": 0.50, "margin": 0.05, "max_leverage": lev}))
    if include_min_hold:
        for hold in MIN_HOLD_DAYS:
            specs.append(("E_min_hold_smoothing", {"min_hold_days": hold}))
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
        "allocation_label",
        "predicted_allocation_class",
        "predicted_allocation_asset",
        "allocation_confidence",
        "prob_qqq",
        "prob_spy",
        "prob_bil",
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
        "mean_allocation_confidence": float(pd.to_numeric(daily["allocation_confidence"], errors="coerce").mean()),
        "allocation_accuracy": float(daily["predicted_allocation_class"].astype(int).eq(daily["allocation_label"].astype(int)).mean()) if "allocation_label" in daily else np.nan,
        "qqq_exposure_share": float(daily["target_asset"].eq("QQQ").mean()),
        "spy_exposure_share": float(daily["target_asset"].eq("SPY").mean()),
        "bil_exposure_share": float(daily["target_asset"].eq("BIL").mean()),
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
    if policy_name == "D_leverage_permission":
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
    out["beats_spy_both"] = out["beats_spy_cagr"] & out["beats_spy_sharpe"]
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
        GRID_VALIDATION_STATUS == "pass"
        and row.get("grid_validation_status", GRID_VALIDATION_STATUS) == "pass"
        and
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


def allocation_distribution(policy_predictions: pd.DataFrame) -> pd.DataFrame:
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


def model_family_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("model_family", as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
        )
        .sort_values(["passing_candidates", "best_Sharpe", "best_CAGR"], ascending=[False, False, False])
    )


def target_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["label_type", "label_spec"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
        )
        .sort_values(["passing_candidates", "best_Sharpe", "best_CAGR"], ascending=[False, False, False])
    )


def seed_stability_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["label_spec", "model_family", "policy_name", "policy_params"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            median_CAGR=("CAGR", "median"),
            median_Sharpe=("Sharpe", "median"),
            worst_seed_CAGR=("CAGR", "min"),
            worst_seed_Sharpe=("Sharpe", "min"),
            pct_seeds_beating_spy_both=("beats_spy_both", "mean"),
        )
        .sort_values(["median_Sharpe", "median_CAGR"], ascending=[False, False])
    )


def add_acceptance_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["grid_validation_status"] = GRID_VALIDATION_STATUS
    out["passes_acceptance"] = out.apply(candidate_passes, axis=1)
    return out


def candidate_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    out = add_acceptance_flags(summary)
    out["financing_sensitivity_penalty"] = out.groupby(["label_spec", "model_family", "policy_name", "policy_params", "leverage_policy", "leverage"], dropna=False)["CAGR"].transform("std").fillna(0.0)
    passing = out[out["passes_acceptance"]].copy()
    rest = out[~out["passes_acceptance"]].copy()
    sort_cols = ["Sharpe", "CAGR", "Max DD", "qqq_exposure_share", "Switch Count", "financing_sensitivity_penalty"]
    passing = passing.sort_values(sort_cols, ascending=[False, False, False, False, True, True])
    rest = rest.sort_values(sort_cols, ascending=[False, False, False, False, True, True])
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


def write_runtime_manifest(args: argparse.Namespace, started_at: datetime, bil_meta: dict[str, Any], selected_models: list[ModelSpec], grid_validation_summary: dict[str, Any]) -> None:
    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "output_root": OUTPUT_ROOT,
            "research_leg": "strict-PIT direct-allocation model-family comparison",
            "base_signal": BASE_SIGNAL_NAME,
            "features": FEATURES,
            "label_types": ["utility_basic", "utility_drawdown_capped", "utility_vol_adjusted"],
            "drawdown_thresholds": DRAWDOWN_THRESHOLDS,
            "utility_drawdown_lambdas": UTILITY_DRAWDOWN_LAMBDAS,
            "selected_label_specs": [spec.key for spec in label_specs(full_label_grid=args.full_label_grid or args.mode == "full")],
            "horizon_hours": FWD_HOURS,
            "policy_specs": policy_specs(include_min_hold=args.include_min_hold),
            "selected_model_families": [spec.model_family for spec in selected_models],
            "full_model_grid": bool(args.full_model_grid or args.mode == "full"),
            "full_label_grid": bool(args.full_label_grid or args.mode == "full"),
            "bounded_first_run_note": "Default compact run uses direct QQQ/SPY/BIL utility labels and compact tree models. Full utility grids, smoothing, and future temporal models are available behind CLI flags for longer runs.",
            "grid_validation": grid_validation_summary,
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


def write_input_hashes(grid: pd.DataFrame, bil_meta: dict[str, Any], prior: pd.DataFrame, grid_validation_summary: dict[str, Any]) -> None:
    write_json(
        {
            "faithful_legacy_grid": {"path": FAITHFUL_GRID_PATH, "sha256": file_sha256(FAITHFUL_GRID_PATH), "rows": len(grid)},
            "grid_validation": grid_validation_summary,
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


def write_readme(summary: pd.DataFrame, rankings: pd.DataFrame, meta: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None, grid_validation_summary: dict[str, Any] | None = None) -> None:
    accepted = rankings[rankings["passes_acceptance"].eq(True)] if not rankings.empty else pd.DataFrame()
    top = rankings.head(10) if not rankings.empty else pd.DataFrame()
    lines = [
        "# MUB-27 V4.2 Strict-PIT Direct-Allocation Model-Family Comparison",
        "",
        "Research-only comparison of direct-allocation models that choose QQQ, SPY, or BIL from forward utility labels.",
        "",
        "## Scope",
        "",
        f"- Base signal: `{BASE_SIGNAL_NAME}`.",
        "- Original 14 V4 features only; no calendar features.",
        f"- Headline financing: {HEADLINE_FINANCING_LABEL} = {HEADLINE_FINANCING_RATE:.2%}.",
        "- B/C2 teacher-smoothed comparison, when present, is non-PIT reference only.",
        "- No candidate can be accepted unless `grid_validation_report.csv` passes.",
        "",
        "## Grid Validation",
        "",
        md_table(pd.DataFrame([grid_validation_summary or {"overall_status": GRID_VALIDATION_STATUS}])),
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
        "## Target Diagnostics",
        "",
        md_table(meta.sort_values(["accuracy", "log_loss"], ascending=[False, True]) if not meta.empty and "accuracy" in meta.columns else meta),
        "",
        "## Production Safety",
        "",
        "No production V4 tables, jobs, Saxo/token objects, RT/dashboard notebooks, model registry entries, schedules, or trading logic were changed.",
    ]
    write_text("\n".join(lines), "README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict-PIT V4.2 direct-allocation model-family comparison.")
    parser.add_argument("--mode", choices=["compact", "full"], default="compact", help="Compact is interactive; full expands label/model grids for longer runs.")
    parser.add_argument("--max-label-specs", type=int, default=None, help="Optional debug cap for label specs.")
    parser.add_argument("--max-models", type=int, default=None, help="Optional debug cap for model families.")
    parser.add_argument("--full-label-grid", action="store_true", help="Run all label families across all threshold combinations.")
    parser.add_argument("--full-model-grid", action="store_true", help="Run optional LightGBM/CatBoost/RandomForest families when available.")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost even if available.")
    parser.add_argument("--skip-neural", action="store_true", help="Reserved for future neural/temporal model expansion; compact run is tree-only.")
    parser.add_argument("--skip-install", action="store_true", help="Reserved for package-install discipline; this script only ensures sklearn.")
    parser.add_argument("--include-min-hold", action="store_true", help="Run optional minimum-hold smoothing policy.")
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    global GRID_VALIDATION_STATUS
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    ensure_dirs()
    grid_raw = load_grid()
    bil_prices, bil_meta = load_bil_prices(grid_raw)
    grid = attach_bil(grid_raw, bil_prices)
    base_signals = load_base_signals()
    prior, spy_row, xgb_row, bc2_row = load_benchmark_rows()
    work = build_work_frame(grid, base_signals)
    validation_report, grid_validation_summary = validate_grid_inputs(grid_raw, grid, bil_meta, base_signals, prior, work)
    GRID_VALIDATION_STATUS = str(grid_validation_summary["overall_status"])
    write_csv(validation_report, "grid_validation_report.csv")

    full_label_grid = args.full_label_grid or args.mode == "full"
    full_model_grid = args.full_model_grid or args.mode == "full"
    models = model_specs(args.max_models, full_model_grid=full_model_grid, include_xgboost=not args.skip_xgboost)
    write_runtime_manifest(args, started_at, bil_meta, models, grid_validation_summary)
    write_input_hashes(grid, bil_meta, prior, grid_validation_summary)

    specs = label_specs(full_label_grid=full_label_grid)
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
    write_readme(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), spy_row, xgb_row, grid_validation_summary)
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
                write_csv(target_results(partial), "target_results.csv")
                write_csv(model_family_results(partial), "model_family_results.csv")
                write_csv(policy_results(partial), "policy_results.csv")
                write_csv(leverage_policy_results(partial), "leverage_results.csv")
                write_csv(financing_sensitivity(partial), "financing_sensitivity.csv")
                write_csv(seed_stability_summary(partial), "seed_stability_summary.csv")
                write_readme(partial, partial_rankings, pd.concat(meta_frames, ignore_index=True), spy_row, xgb_row, grid_validation_summary)

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
    write_csv(target_results(summary), "target_results.csv")
    write_csv(model_family_results(summary), "model_family_results.csv")
    write_csv(policy_results(summary), "policy_results.csv")
    write_csv(leverage_policy_results(summary), "leverage_results.csv")
    write_csv(financing_sensitivity(summary), "financing_sensitivity.csv")
    write_csv(seed_stability_summary(summary), "seed_stability_summary.csv")
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
    distribution = allocation_distribution(policy_predictions)
    write_csv(distribution, "exposure_distribution.csv")
    write_csv(distribution, "regime_or_allocation_distribution.csv")
    write_readme(summary, rankings, meta_results, spy_row, xgb_row, grid_validation_summary)
    print(json.dumps(jsonable({"output_root": OUTPUT_ROOT, "top_10": rankings.head(10).to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()

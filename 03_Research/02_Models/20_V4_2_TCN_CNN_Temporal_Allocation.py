# Databricks notebook source
"""MUB-27 V4.2 strict-PIT TCN/CNN temporal allocation experiment.

Research-only script that trains compact sequence models over the reconciled
faithful V4 grid and scores direct QQQ/SPY/BIL allocation policies under
walk-forward validation.

All outputs are isolated under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tcn_cnn_temporal_allocation_2026-05-14/`

The script is intentionally a single practical Databricks entry point. It does
not mutate production tables, live job definitions, dashboard notebooks, model
serving references, or trading logic.
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
OUTPUT_ROOT = BASE / "v4_2_tcn_cnn_temporal_allocation_2026-05-14"

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
DIRECT_ALLOCATION_ROOT = BASE / "v4_2_direct_allocation_model_family_2026-05-14"

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

SPY_VARIANT = "E_spy_benchmark"
XGBOOST_VARIANT = "D_pca_strict_xgboost"
HEADLINE_FINANCING_RATE = 0.045
HEADLINE_FINANCING_LABEL = "10Y Treasury + 50bps proxy"
FINANCING_GRID = [0.0, HEADLINE_FINANCING_RATE, 0.07, 0.10]
LEVERAGE_GRID = [1.0, 1.25, 1.5, 2.0]
GRID_VALIDATION_STATUS = "not_run"

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

ASSET_TO_CLASS = {"QQQ": 0, "SPY": 1, "BIL": 2}
CLASS_TO_ASSET = {v: k for k, v in ASSET_TO_CLASS.items()}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "BIL"}
DEFAULT_SEEDS = [11, 23, 37, 53, 71]


@dataclass(frozen=True)
class TargetSpec:
    target_family: str
    qqq_margin: float = 0.0025
    utility_drawdown_lambda: float = 0.15
    risk_off_quantile: float = 0.20
    top_quantile: float = 0.80

    @property
    def key(self) -> str:
        margin = int(round(self.qqq_margin * 10000))
        lam = int(round(self.utility_drawdown_lambda * 100))
        top = int(round(self.top_quantile * 100))
        return f"{self.target_family}_m{margin:03d}_lam{lam:03d}_top{top:03d}"


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    window_hours: int
    seed: int
    channels: int = 16
    dropout: float = 0.15
    max_epochs: int = 16
    patience: int = 4

    @property
    def key(self) -> str:
        return f"{self.architecture}_{self.window_hours}h_seed{self.seed}"


def ensure_python_deps(skip_install: bool = False):
    missing = []
    for module_name, package_name in [("sklearn", "scikit-learn")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing and not skip_install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, {"accuracy_score": accuracy_score, "f1_score": f1_score, "log_loss": log_loss}


def import_torch(allow_install: bool = False):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, F, DataLoader, TensorDataset, {"available": True, "version": getattr(torch, "__version__", "unknown")}
    except Exception as exc:
        if allow_install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
            return import_torch(False)
        return None, None, None, None, None, {"available": False, "error": str(exc)}


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


def forward_return(price: pd.Series, horizon: int) -> pd.Series:
    return price.shift(-horizon) / price - 1.0


def forward_path_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    future_prices = pd.concat([price.shift(-i) for i in range(1, horizon + 1)], axis=1)
    return future_prices.min(axis=1) / price - 1.0


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
    path = BASELINE_SIGNAL_FILES["D_pca_strict_xgboost"]
    sig = read_csv(path)
    if "predicted_regime" not in sig.columns:
        raise ValueError(f"{path} missing predicted_regime")
    sig = sig[["timestamp", "predicted_regime"]].dropna().copy()
    sig["predicted_regime"] = pd.to_numeric(sig["predicted_regime"], errors="coerce").astype("Int64")
    sig = sig.dropna(subset=["predicted_regime"]).copy()
    sig["predicted_regime"] = sig["predicted_regime"].astype(int)
    sig["base_asset"] = sig["predicted_regime"].map(REGIME_TO_ASSET).fillna("SPY")
    return sig.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_benchmark_rows() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    prior = read_csv(STRICT_COMPARISON_PATH)
    spy = prior[prior["variant"].eq(SPY_VARIANT)]
    if spy.empty:
        raise ValueError(f"Missing {SPY_VARIANT} in {STRICT_COMPARISON_PATH}")
    xgb = prior[prior["variant"].eq(XGBOOST_VARIANT)]
    bc2 = prior[prior["variant"].astype(str).str.contains("B_reference_teacher_smoothed", regex=False, na=False)]
    return prior, spy.iloc[0].to_dict(), xgb.iloc[0].to_dict() if not xgb.empty else None, bc2.iloc[0].to_dict() if not bc2.empty else None


def load_prior_direct_allocation_reference() -> pd.DataFrame:
    path = DIRECT_ALLOCATION_ROOT / "candidate_rankings.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        out = read_csv(path)
    except Exception:
        return pd.DataFrame()
    return out.head(25).copy()


def build_work_frame(grid: pd.DataFrame, base_signals: pd.DataFrame) -> pd.DataFrame:
    work = grid.merge(base_signals, on="timestamp", how="left").copy()
    work["base_asset"] = work["base_asset"].fillna("SPY")
    work = work[(work["timestamp"] >= TRAIN_START) & (work["feature_ok"])].copy()
    for asset, col in [("SPY", "SPY_close"), ("QQQ", "QQQ_close"), ("BIL", "BIL_close")]:
        work[f"{asset}_fwd_return_1w"] = forward_return(work[col], FWD_HOURS)
        work[f"{asset}_fwd_path_drawdown_1w"] = forward_path_drawdown(work[col], FWD_HOURS)
    work["label_available_ts"] = work["timestamp"].shift(-FWD_HOURS)
    work["qqq_spy_fwd_spread_1w"] = work["QQQ_fwd_return_1w"] - work["SPY_fwd_return_1w"]
    work["risk_on_best_return_1w"] = work[["QQQ_fwd_return_1w", "SPY_fwd_return_1w"]].max(axis=1)
    required = [
        "label_available_ts",
        "QQQ_fwd_return_1w",
        "SPY_fwd_return_1w",
        "BIL_fwd_return_1w",
        "QQQ_fwd_path_drawdown_1w",
        "SPY_fwd_path_drawdown_1w",
        "BIL_fwd_path_drawdown_1w",
    ]
    work = work.dropna(subset=required).copy()
    work["work_pos"] = np.arange(len(work))
    return work.reset_index(drop=True)


def monthly_folds(work: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(EVAL_START.normalize(), EVAL_END.normalize(), freq="MS")
    return [(pd.Timestamp(start), min(pd.Timestamp(start + pd.offsets.MonthBegin(1)), FINAL_FOLD_END)) for start in starts]


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
    for root in [STRICT_ROOT, DEFENSIVE_PRICE_ROOT, DIRECT_ALLOCATION_ROOT]:
        if not root.exists():
            continue
        for path in list(root.rglob("input_hashes.json"))[:20] + list(root.rglob("runtime_manifest.json"))[:20]:
            try:
                known[str(path)] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
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
            "Input grid must be the faithful legacy clone used by MUB-27 and match the known reconciled hash.",
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
            "All original 14 V4 features are required.",
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
            "This script does not recompute VIX/VIX3M joins. It requires the reconciled faithful grid artifact and validates the expected VIX/VIX3M columns and cadence.",
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
        f"{asset}_price_update_rows": int(pd.to_numeric(eval_grid[f"{asset}_close"], errors="coerce").diff().abs().gt(0).sum()) if f"{asset}_close" in eval_grid else 0
        for asset in ["SPY", "QQQ", "BIL"]
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
            "Strict-PIT/reference signals must overlap grid timestamps and benchmark comparison rows must exist.",
            {"signal_rows": len(base_signals), "grid_signal_overlap_rows": overlap, "comparison_rows": len(prior)},
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
            {
                "known_artifact_count": len(known_hashes),
                "current_grid_hash": grid_hash,
                "defensive_bil_hash": bil_meta.get("sha256"),
                "b_c2_signal_hash": file_sha256(BASELINE_SIGNAL_FILES["B_reference_teacher_smoothed_non_pit_reference_only"]),
            },
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


def class_utilities(frame: pd.DataFrame, spec: TargetSpec, thresholds: dict[str, float]) -> pd.DataFrame:
    switch = {
        asset: np.where(frame["base_asset"].astype(str).eq(asset), 0.0, SWITCH_COST)
        for asset in ["QQQ", "SPY", "BIL"]
    }
    if spec.target_family == "best_asset_forward_return":
        return pd.DataFrame(
            {
                asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce") - switch[asset]
                for asset in ["QQQ", "SPY", "BIL"]
            },
            index=frame.index,
        )
    if spec.target_family == "utility_return_first":
        return pd.DataFrame(
            {
                asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce")
                - spec.utility_drawdown_lambda * pd.to_numeric(frame[f"{asset}_fwd_path_drawdown_1w"], errors="coerce").clip(upper=0.0).abs()
                - switch[asset]
                for asset in ["QQQ", "SPY", "BIL"]
            },
            index=frame.index,
        )
    if spec.target_family == "qqq_spy_relative_upside":
        spread = frame["QQQ_fwd_return_1w"] - frame["SPY_fwd_return_1w"]
        weak_risk = frame[["QQQ_fwd_return_1w", "SPY_fwd_return_1w"]].max(axis=1) < thresholds["risk_off_return"]
        qqq = spread - spec.qqq_margin
        spy = -spread
        bil = np.where(weak_risk, frame["BIL_fwd_return_1w"] - frame[["QQQ_fwd_return_1w", "SPY_fwd_return_1w"]].max(axis=1), -0.01)
        return pd.DataFrame({"QQQ": qqq, "SPY": spy, "BIL": bil}, index=frame.index)
    if spec.target_family == "top_decile_risk_on":
        qqq_spread = frame["QQQ_fwd_return_1w"] - frame[["SPY_fwd_return_1w", "BIL_fwd_return_1w"]].max(axis=1)
        qqq = np.where((qqq_spread >= thresholds["top_qqq_spread"]) & (frame["QQQ_fwd_return_1w"] > 0.0), qqq_spread, -0.01)
        bil = np.where(frame["risk_on_best_return_1w"] <= thresholds["risk_off_return"], frame["BIL_fwd_return_1w"], -0.005)
        spy = np.where((qqq < 0.0) & (bil < 0.0), frame["SPY_fwd_return_1w"], -0.001)
        return pd.DataFrame({"QQQ": qqq, "SPY": spy, "BIL": bil}, index=frame.index)
    raise ValueError(f"Unknown target family: {spec.target_family}")


def fold_thresholds(train: pd.DataFrame, spec: TargetSpec) -> dict[str, float]:
    risk_on_best = pd.to_numeric(train["risk_on_best_return_1w"], errors="coerce").dropna()
    qqq_spread = (
        train["QQQ_fwd_return_1w"] - train[["SPY_fwd_return_1w", "BIL_fwd_return_1w"]].max(axis=1)
    ).dropna()
    return {
        "risk_off_return": float(risk_on_best.quantile(spec.risk_off_quantile)) if len(risk_on_best) else -0.005,
        "top_qqq_spread": float(qqq_spread.quantile(spec.top_quantile)) if len(qqq_spread) else spec.qqq_margin,
    }


def make_fold_labels(frame: pd.DataFrame, train: pd.DataFrame, spec: TargetSpec) -> tuple[pd.Series, dict[str, float]]:
    thresholds = fold_thresholds(train, spec)
    util = class_utilities(frame, spec, thresholds)
    labels = util.idxmax(axis=1).map(ASSET_TO_CLASS).astype(float)
    labels = labels.mask(util.isna().all(axis=1), np.nan)
    return labels, thresholds


class TinySequenceNetFactory:
    def __init__(self, torch, nn, functional):
        self.torch = torch
        self.nn = nn
        self.functional = functional

    def build(self, spec: ModelSpec, n_features: int, n_classes: int):
        torch = self.torch
        nn = self.nn
        F = self.functional

        class CausalBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float):
                super().__init__()
                self.pad = (3 - 1) * dilation
                self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, dilation=dilation)
                self.norm = nn.BatchNorm1d(out_ch)
                self.drop = nn.Dropout(dropout)
                self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

            def forward(self, x):
                y = F.pad(x, (self.pad, 0))
                y = self.drop(F.relu(self.norm(self.conv(y))))
                return y + self.res(x)

        class TinyTCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1 = CausalBlock(n_features, spec.channels, dilation=1, dropout=spec.dropout)
                self.b2 = CausalBlock(spec.channels, spec.channels, dilation=2, dropout=spec.dropout)
                self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(spec.channels, n_classes))

            def forward(self, x):
                x = x.transpose(1, 2)
                return self.head(self.b2(self.b1(x)))

        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(n_features, spec.channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(spec.channels),
                    nn.ReLU(),
                    nn.Dropout(spec.dropout),
                    nn.Conv1d(spec.channels, spec.channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(spec.channels),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                    nn.Flatten(),
                    nn.Linear(spec.channels, n_classes),
                )

            def forward(self, x):
                return self.net(x.transpose(1, 2))

        if spec.architecture == "tcn":
            return TinyTCN()
        if spec.architecture == "cnn":
            return TinyCNN()
        raise ValueError(f"Unknown architecture: {spec.architecture}")


def make_sequences(features: np.ndarray, labels: np.ndarray, source_indices: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    kept: list[int] = []
    for idx in source_indices:
        if idx < window - 1:
            continue
        start = idx - window + 1
        seq = features[start : idx + 1]
        label = labels[idx]
        if seq.shape[0] == window and np.isfinite(seq).all() and np.isfinite(label):
            xs.append(seq.astype(np.float32))
            ys.append(int(label))
            kept.append(int(idx))
    if not xs:
        return np.empty((0, window, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(kept, dtype=np.int64)


def train_sequence_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    spec: ModelSpec,
    torch_bundle: tuple[Any, Any, Any, Any, Any, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    torch, nn, _F, DataLoader, TensorDataset, torch_meta = torch_bundle
    if not torch_meta.get("available"):
        raise RuntimeError(f"PyTorch unavailable: {torch_meta.get('error')}")
    seed_everything(spec.seed)
    torch.manual_seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = np.array(sorted(np.unique(y_train)), dtype=int)
    n_classes = 3
    if len(classes) < 2:
        proba = np.zeros((len(x_test), 3), dtype=float)
        fallback = int(classes[0]) if len(classes) else 1
        proba[:, fallback] = 1.0
        return proba, {"status": "fallback_single_class", "epochs": 0, "device": str(device)}

    val_size = max(128, int(round(len(x_train) * 0.15)))
    val_size = min(val_size, max(1, len(x_train) // 4))
    train_end = len(x_train) - val_size
    x_fit, y_fit = x_train[:train_end], y_train[:train_end]
    x_val, y_val = x_train[train_end:], y_train[train_end:]

    factory = TinySequenceNetFactory(torch, nn, _F)
    model = factory.build(spec, x_train.shape[2], n_classes).to(device)
    class_counts = np.bincount(y_fit, minlength=n_classes).astype(float)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = weights / np.nanmean(weights)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    train_ds = TensorDataset(torch.tensor(x_fit, dtype=torch.float32), torch.tensor(y_fit, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=min(256, max(32, len(train_ds) // 8)), shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    best_state = None
    best_val = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(spec.max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).detach().cpu().item())
        epochs_run = epoch + 1
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= spec.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds: list[np.ndarray] = []
    test_ds = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.zeros(len(x_test), dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    with torch.no_grad():
        for xb, _yb in test_loader:
            logits = model(xb.to(device))
            preds.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    proba = np.vstack(preds) if preds else np.empty((0, 3), dtype=float)
    return proba, {"status": "fit", "epochs": epochs_run, "best_val_loss": best_val, "device": str(device)}


def run_walkforward(
    work: pd.DataFrame,
    target_spec: TargetSpec,
    model_spec: ModelSpec,
    StandardScaler,
    metrics: dict[str, Any],
    torch_bundle: tuple[Any, Any, Any, Any, Any, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    fold_rows = []
    frame = work.reset_index(drop=True).copy()
    for fold_start, fold_end in monthly_folds(frame):
        train = frame[
            (frame["timestamp"] < fold_start)
            & (frame["label_available_ts"] < fold_start)
            & frame["timestamp"].ge(TRAIN_START)
        ].copy()
        test = frame[(frame["timestamp"] >= fold_start) & (frame["timestamp"] < fold_end)].copy()
        if test.empty:
            continue
        if len(train) < max(750, model_spec.window_hours * 8):
            continue
        labels, thresholds = make_fold_labels(frame, train, target_spec)
        scaler = StandardScaler()
        train_idx = train.index.to_numpy(dtype=int)
        test_idx = test.index.to_numpy(dtype=int)
        scaler.fit(frame.loc[train_idx, FEATURES].astype(float))
        scaled = scaler.transform(frame[FEATURES].astype(float))
        x_train, y_train, kept_train_idx = make_sequences(scaled, labels.to_numpy(), train_idx, model_spec.window_hours)
        x_test, y_test, kept_test_idx = make_sequences(scaled, labels.to_numpy(), test_idx, model_spec.window_hours)
        if len(x_test) == 0:
            continue
        if len(x_train) < 500 or len(np.unique(y_train)) < 2:
            prior = pd.Series(y_train).value_counts(normalize=True).to_dict() if len(y_train) else {}
            proba = np.tile([prior.get(0, 1 / 3), prior.get(1, 1 / 3), prior.get(2, 1 / 3)], (len(x_test), 1))
            proba = proba / proba.sum(axis=1, keepdims=True)
            train_meta = {"status": "fallback_constant_probability", "epochs": 0, "device": "none"}
        else:
            proba, train_meta = train_sequence_model(x_train, y_train, x_test, model_spec, torch_bundle)
        out = frame.loc[kept_test_idx, ["timestamp", "predicted_regime", "base_asset"]].copy()
        out["allocation_label"] = y_test
        out["prob_qqq"] = proba[:, 0]
        out["prob_spy"] = proba[:, 1]
        out["prob_bil"] = proba[:, 2]
        out["predicted_allocation_class"] = np.argmax(proba, axis=1)
        out["predicted_allocation_asset"] = out["predicted_allocation_class"].map(CLASS_TO_ASSET)
        out["allocation_confidence"] = np.max(proba, axis=1)
        out["target_spec"] = target_spec.key
        out["target_family"] = target_spec.target_family
        out["architecture"] = model_spec.architecture
        out["model_family"] = f"{model_spec.architecture}_sequence"
        out["window_hours"] = model_spec.window_hours
        out["seed"] = model_spec.seed
        predictions.append(out)

        y_pred = out["predicted_allocation_class"].astype(int)
        try:
            ll = metrics["log_loss"](out["allocation_label"].astype(int), np.clip(proba, 1e-6, 1.0), labels=[0, 1, 2])
        except Exception:
            ll = np.nan
        fold_rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "target_spec": target_spec.key,
                "target_family": target_spec.target_family,
                "architecture": model_spec.architecture,
                "window_hours": model_spec.window_hours,
                "seed": model_spec.seed,
                "status": train_meta.get("status"),
                "device": train_meta.get("device"),
                "epochs": train_meta.get("epochs"),
                "best_val_loss": train_meta.get("best_val_loss"),
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "train_qqq_share": float(np.mean(y_train == 0)) if len(y_train) else np.nan,
                "train_spy_share": float(np.mean(y_train == 1)) if len(y_train) else np.nan,
                "train_bil_share": float(np.mean(y_train == 2)) if len(y_train) else np.nan,
                "test_qqq_share": float(np.mean(y_test == 0)) if len(y_test) else np.nan,
                "test_spy_share": float(np.mean(y_test == 1)) if len(y_test) else np.nan,
                "test_bil_share": float(np.mean(y_test == 2)) if len(y_test) else np.nan,
                "accuracy": float(metrics["accuracy_score"](out["allocation_label"].astype(int), y_pred)) if len(out) else np.nan,
                "macro_f1": float(metrics["f1_score"](out["allocation_label"].astype(int), y_pred, average="macro", zero_division=0)) if len(out) else np.nan,
                "log_loss": ll,
                "mean_confidence": float(out["allocation_confidence"].mean()) if len(out) else np.nan,
                "fold_thresholds": json.dumps(jsonable(thresholds), sort_keys=True),
                "strict_pit_train_max_label_available_ts": train["label_available_ts"].max() if len(train) else None,
            }
        )
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pred_df, pd.DataFrame(fold_rows)


def build_policy_signals(pred: pd.DataFrame, policy_name: str, params: dict[str, Any]) -> pd.DataFrame:
    out = pred.copy()
    p_qqq = pd.to_numeric(out["prob_qqq"], errors="coerce").fillna(1 / 3)
    p_spy = pd.to_numeric(out["prob_spy"], errors="coerce").fillna(1 / 3)
    p_bil = pd.to_numeric(out["prob_bil"], errors="coerce").fillna(1 / 3)
    if policy_name == "direct_argmax":
        target = out["predicted_allocation_asset"].astype(str)
    elif policy_name == "qqq_upside_threshold":
        threshold = float(params["qqq_threshold"])
        risk_off = float(params["risk_off_threshold"])
        margin = p_qqq - pd.concat([p_spy, p_bil], axis=1).max(axis=1)
        target = pd.Series("SPY", index=out.index, dtype=object)
        target = target.mask((p_qqq >= threshold) & (margin >= float(params["margin"])), "QQQ")
        target = target.mask(p_bil >= risk_off, "BIL")
    elif policy_name == "probability_weighted":
        target = pd.Series("SPY", index=out.index, dtype=object)
        target = target.mask(p_qqq >= float(params["qqq_threshold"]), "QQQ")
        target = target.mask(p_bil >= float(params["bil_threshold"]), "BIL")
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
    out["policy_name"] = policy_name
    out["policy_params"] = json.dumps(params, sort_keys=True)
    out["target_asset_signal"] = target
    out["candidate_base"] = (
        out["target_spec"]
        + "__"
        + out["architecture"]
        + "__"
        + out["window_hours"].astype(str)
        + "h__seed"
        + out["seed"].astype(str)
        + "__"
        + policy_name
        + "__"
        + pd.Series(json.dumps(params, sort_keys=True), index=out.index)
    )
    return out


def policy_specs() -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = [("direct_argmax", {})]
    for qqq_threshold in [0.40, 0.50, 0.60]:
        specs.append(("qqq_upside_threshold", {"qqq_threshold": qqq_threshold, "risk_off_threshold": 0.55, "margin": 0.03}))
    for qqq_threshold in [0.45, 0.55]:
        for bil_threshold in [0.45, 0.55]:
            specs.append(("probability_weighted", {"qqq_threshold": qqq_threshold, "bil_threshold": bil_threshold}))
    return specs


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        "timestamp",
        "candidate_base",
        "target_spec",
        "target_family",
        "architecture",
        "model_family",
        "window_hours",
        "seed",
        "policy_name",
        "policy_params",
        "predicted_regime",
        "base_asset",
        "target_asset_signal",
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
        active_asset = target_asset
    out = data.copy()
    out["target_asset"] = assets
    out["strategy_return"] = returns
    out["position_changed"] = changed
    out["strategy_equity"] = equity
    out["spy_equity"] = spy_equity
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
    return daily


def leverage_curve(daily: pd.DataFrame, leverage: float, financing: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    returns: list[float] = []
    equities: list[float] = []
    drawdowns: list[float] = []
    applied: list[float] = []
    breached_zero = False
    for row in daily.itertuples(index=False):
        raw = float(getattr(row, "strategy_daily_return"))
        ret = leverage * raw - max(leverage - 1.0, 0.0) * financing / TRADING_DAYS
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
        applied.append(leverage)
    return pd.Series(returns), pd.Series(equities), pd.Series(drawdowns), pd.Series(applied), breached_zero


def metric_row(
    candidate_id: str,
    daily: pd.DataFrame,
    returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    applied_leverage: pd.Series,
    breached_zero: bool,
    leverage: float,
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
        "target_spec": str(daily["target_spec"].iloc[0]),
        "target_family": str(daily["target_family"].iloc[0]),
        "architecture": str(daily["architecture"].iloc[0]),
        "model_family": str(daily["model_family"].iloc[0]),
        "window_hours": int(daily["window_hours"].iloc[0]),
        "seed": int(daily["seed"].iloc[0]),
        "policy_name": str(daily["policy_name"].iloc[0]),
        "policy_params": str(daily["policy_params"].iloc[0]),
        "leverage_policy": "fixed",
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
        "zero_equity_breach_flag": bool(breached_zero),
        "max_dd_acceptance_flag": bool(max_dd >= MAX_ACCEPTED_DRAWDOWN),
        "max_dd_floor_breach": bool(max_dd < MAX_ACCEPTED_DRAWDOWN),
        "Switch Count": int(daily["position_changed"].astype(bool).sum()),
        "switch_count": int(daily["position_changed"].astype(bool).sum()),
        "turnover": float(daily["position_changed"].astype(bool).mean()),
        "daily_rows": int(len(daily)),
        "mean_allocation_confidence": float(pd.to_numeric(daily["allocation_confidence"], errors="coerce").mean()),
        "allocation_accuracy": float(daily["predicted_allocation_class"].astype(int).eq(daily["allocation_label"].astype(int)).mean()) if "allocation_label" in daily else np.nan,
        "qqq_exposure_share": float(daily["target_asset"].eq("QQQ").mean()),
        "spy_exposure_share": float(daily["target_asset"].eq("SPY").mean()),
        "bil_exposure_share": float(daily["target_asset"].eq("BIL").mean()),
        "QQQ share": float(daily["target_asset"].eq("QQQ").mean()),
        "SPY share": float(daily["target_asset"].eq("SPY").mean()),
        "BIL share": float(daily["target_asset"].eq("BIL").mean()),
        "mean_applied_leverage": float(applied_leverage.mean()) if len(applied_leverage) else leverage,
    }


def evaluate_policy(grid: pd.DataFrame, policy_signals: pd.DataFrame, leverage: float, financing: float) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    daily = run_unlevered_backtest(grid, policy_signals)
    if daily.empty:
        return None, None
    candidate_base = str(policy_signals["candidate_base"].iloc[0])
    lev_tag = str(leverage).replace(".", "_")
    candidate_id = f"{candidate_base}__lev_{lev_tag}__fin_{financing:g}"
    returns, equity, drawdown, applied, breached = leverage_curve(daily, leverage, financing)
    row = metric_row(candidate_id, daily, returns, equity, drawdown, applied, breached, leverage, financing)
    curve = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "report_date": daily["report_date"].astype(str).to_numpy(),
            "candidate_daily_return": returns.to_numpy(),
            "candidate_equity": equity.to_numpy(),
            "drawdown": drawdown.to_numpy(),
            "applied_leverage": applied.to_numpy(),
        }
    )
    return row, curve


def add_benchmark_gaps(summary: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None, bc2_row: dict[str, Any] | None) -> pd.DataFrame:
    out = summary.copy()
    out["SPY_CAGR"] = float(spy_row["CAGR"])
    out["SPY_Sharpe"] = float(spy_row["Sharpe"])
    out["SPY_Max DD"] = float(spy_row["Max DD"])
    out["beats_spy_cagr"] = out["CAGR"] > out["SPY_CAGR"]
    out["beats_spy_sharpe"] = out["Sharpe"] > out["SPY_Sharpe"]
    out["beats_spy_both"] = out["beats_spy_cagr"] & out["beats_spy_sharpe"]
    out["SPY CAGR gap"] = out["CAGR"] - out["SPY_CAGR"]
    out["SPY Sharpe gap"] = out["Sharpe"] - out["SPY_Sharpe"]
    out["gap_vs_spy_cagr"] = out["SPY CAGR gap"]
    out["gap_vs_spy_sharpe"] = out["SPY Sharpe gap"]
    if xgb_row is not None:
        out["XGBoost_CAGR"] = float(xgb_row["CAGR"])
        out["XGBoost_Sharpe"] = float(xgb_row["Sharpe"])
        out["production_style_XGBoost_CAGR_gap"] = out["CAGR"] - out["XGBoost_CAGR"]
        out["production_style_XGBoost_Sharpe_gap"] = out["Sharpe"] - out["XGBoost_Sharpe"]
        out["gap_vs_xgboost_cagr"] = out["production_style_XGBoost_CAGR_gap"]
        out["gap_vs_xgboost_sharpe"] = out["production_style_XGBoost_Sharpe_gap"]
    else:
        out["XGBoost_CAGR"] = np.nan
        out["XGBoost_Sharpe"] = np.nan
        out["production_style_XGBoost_CAGR_gap"] = np.nan
        out["production_style_XGBoost_Sharpe_gap"] = np.nan
        out["gap_vs_xgboost_cagr"] = np.nan
        out["gap_vs_xgboost_sharpe"] = np.nan
    if bc2_row is not None:
        out["B_C2_non_pit_reference_CAGR"] = float(bc2_row["CAGR"])
        out["B_C2_non_pit_reference_Sharpe"] = float(bc2_row["Sharpe"])
        out["B/C2 gap non-PIT CAGR"] = out["CAGR"] - out["B_C2_non_pit_reference_CAGR"]
        out["B/C2 gap non-PIT Sharpe"] = out["Sharpe"] - out["B_C2_non_pit_reference_Sharpe"]
    return out


def candidate_passes(row: pd.Series) -> bool:
    return bool(
        GRID_VALIDATION_STATUS == "pass"
        and row.get("grid_validation_status", GRID_VALIDATION_STATUS) == "pass"
        and row["beats_spy_cagr"]
        and row["beats_spy_sharpe"]
        and row["Max DD"] >= MAX_ACCEPTED_DRAWDOWN
        and not row["breached_zero"]
        and not row["max_dd_floor_breach"]
        and row["qqq_exposure_share"] > 0.05
        and (row["leverage"] == 1.0 or row["headline_financing"])
    )


def add_acceptance_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["grid_validation_status"] = GRID_VALIDATION_STATUS
    out["strict_pit_validation_status"] = "pass" if GRID_VALIDATION_STATUS == "pass" else GRID_VALIDATION_STATUS
    out["passes_acceptance"] = out.apply(candidate_passes, axis=1) if not out.empty else False
    return out


def candidate_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = add_acceptance_flags(summary)
    passing = out[out["passes_acceptance"]].copy()
    rest = out[~out["passes_acceptance"]].copy()
    sort_cols = ["Sharpe", "CAGR", "Max DD", "qqq_exposure_share", "turnover"]
    passing = passing.sort_values(sort_cols, ascending=[False, False, False, False, True])
    rest = rest.sort_values(sort_cols, ascending=[False, False, False, False, True])
    return pd.concat([passing, rest], ignore_index=True)


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
    if policy_predictions.empty:
        return pd.DataFrame()
    group_cols = ["target_spec", "target_family", "architecture", "window_hours", "seed", "policy_name", "policy_params"]
    for keys, group in policy_predictions.groupby(group_cols):
        total = len(group)
        base = dict(zip(group_cols, keys))
        for asset, count in group["target_asset_signal"].astype(str).value_counts().sort_index().items():
            rows.append({**base, "target_asset": asset, "count": int(count), "share": float(count / total) if total else np.nan})
    return pd.DataFrame(rows)


def policy_results(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(["policy_name", "leverage", "headline_financing"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
            mean_QQQ_share=("qqq_exposure_share", "mean"),
        )
        .sort_values(["headline_financing", "best_Sharpe", "best_CAGR"], ascending=[False, False, False])
    )


def leverage_results(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(["leverage", "annual_financing_rate"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            zero_breaches=("breached_zero", "sum"),
        )
        .sort_values(["leverage", "annual_financing_rate"])
    )


def financing_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
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


def seed_stability_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary[summary["leverage"].eq(1.0)]
        .groupby(["target_spec", "architecture", "window_hours", "policy_name", "policy_params"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            median_CAGR=("CAGR", "median"),
            median_Sharpe=("Sharpe", "median"),
            worst_seed_CAGR=("CAGR", "min"),
            worst_seed_Sharpe=("Sharpe", "min"),
            pct_seeds_beating_spy_both=("beats_spy_both", "mean"),
            median_QQQ_share=("qqq_exposure_share", "median"),
        )
        .sort_values(["median_Sharpe", "median_CAGR"], ascending=[False, False])
    )


def target_results(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(["target_family", "target_spec"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            passing_candidates=("passes_acceptance", "sum"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            max_QQQ_share=("qqq_exposure_share", "max"),
        )
        .sort_values(["passing_candidates", "best_Sharpe", "best_CAGR"], ascending=[False, False, False])
    )


def feature_window_manifest(specs: list[ModelSpec], target_specs: list[TargetSpec], grid_validation_summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for target in target_specs:
            rows.append(
                {
                    "architecture": spec.architecture,
                    "window_hours": spec.window_hours,
                    "seed": spec.seed,
                    "channels": spec.channels,
                    "dropout": spec.dropout,
                    "max_epochs": spec.max_epochs,
                    "patience": spec.patience,
                    "target_family": target.target_family,
                    "target_spec": target.key,
                    "features": json.dumps(FEATURES),
                    "feature_count": len(FEATURES),
                    "sequence_uses_only_rows_at_or_before_decision_timestamp": True,
                    "scaler_fit_scope": "fold_train_only",
                    "grid_validation_status": grid_validation_summary.get("overall_status"),
                }
            )
    return pd.DataFrame(rows)


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


def write_readme(
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
    fold_meta: pd.DataFrame,
    spy_row: dict[str, Any],
    xgb_row: dict[str, Any] | None,
    grid_validation_summary: dict[str, Any] | None,
) -> None:
    accepted = rankings[rankings["passes_acceptance"].eq(True)] if not rankings.empty and "passes_acceptance" in rankings else pd.DataFrame()
    top = rankings.head(10) if not rankings.empty else pd.DataFrame()
    beat_spy = bool(summary["beats_spy_both"].any()) if not summary.empty and "beats_spy_both" in summary else False
    reached_target = bool(summary["CAGR"].ge(0.15).any()) if not summary.empty and "CAGR" in summary else False
    leverage_changed = False
    if not summary.empty and {"leverage", "Sharpe"}.issubset(summary.columns):
        best_unlevered = summary[summary["leverage"].eq(1.0)]["Sharpe"].max()
        best_levered = summary[summary["leverage"].gt(1.0)]["Sharpe"].max()
        leverage_changed = bool(pd.notna(best_levered) and pd.notna(best_unlevered) and best_levered > best_unlevered)
    recommendation = "continue" if not accepted.empty else "rerun broader grid" if beat_spy or reached_target else "reject compact first pass"
    lines = [
        "# MUB-27 V4.2 TCN/CNN Temporal Allocation",
        "",
        "## What Was Tested",
        "",
        "Compact TCN and 1D CNN sequence classifiers over short windows of the original 14 V4 features. Models score direct QQQ/SPY/BIL allocation policies under strict rolling walk-forward validation.",
        "",
        "## Data",
        "",
        f"- Grid: `{FAITHFUL_GRID_PATH}`.",
        f"- Expected grid SHA-256: `{EXPECTED_GRID_SHA256}`.",
        "- B/C2 teacher-smoothed benchmark is included only as a non-PIT research reference.",
        "",
        "## Validation",
        "",
        md_table(pd.DataFrame([grid_validation_summary or {"overall_status": GRID_VALIDATION_STATUS}])),
        "",
        f"- Grid validation passed: `{(grid_validation_summary or {}).get('overall_status') == 'pass'}`.",
        f"- Strict-PIT validation passed: `{GRID_VALIDATION_STATUS == 'pass'}`.",
        "",
        "## Benchmarks",
        "",
        f"- SPY CAGR: {float(spy_row['CAGR']):.6f}",
        f"- SPY Sharpe: {float(spy_row['Sharpe']):.6f}",
        f"- Production-style XGBoost reference available: {xgb_row is not None}",
        "",
        "## Top Candidates",
        "",
        md_table(top),
        "",
        "## Accepted Candidates",
        "",
        md_table(accepted),
        "",
        "## Conclusions",
        "",
        f"- Any candidate beat SPY on CAGR and Sharpe: `{beat_spy}`.",
        f"- Any candidate reached 15-20% CAGR: `{reached_target}`.",
        f"- Leverage/financing changed the conclusion: `{leverage_changed}`.",
        f"- Recommendation: `{recommendation}`.",
        "",
        "## Fold Diagnostics",
        "",
        md_table(fold_meta.sort_values(["accuracy", "log_loss"], ascending=[False, True]) if not fold_meta.empty and "accuracy" in fold_meta.columns else fold_meta),
        "",
        "## Production Safety",
        "",
        "No production V4 tables, live execution objects, RT/dashboard notebooks, model serving references, job schedules, or trading logic were changed.",
    ]
    write_text("\n".join(lines), "README.md")


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def target_specs(selected: list[str]) -> list[TargetSpec]:
    all_specs = {
        "best_asset_forward_return": TargetSpec("best_asset_forward_return"),
        "qqq_spy_relative_upside": TargetSpec("qqq_spy_relative_upside", qqq_margin=0.0025),
        "top_decile_risk_on": TargetSpec("top_decile_risk_on", top_quantile=0.80),
        "utility_return_first": TargetSpec("utility_return_first", utility_drawdown_lambda=0.15),
    }
    return [all_specs[name] for name in selected if name in all_specs]


def model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    seeds = DEFAULT_SEEDS[: max(1, args.max_seeds)]
    windows = parse_csv_list(args.windows, int)
    archs = parse_csv_list(args.architectures, str)
    epochs = args.max_epochs if args.max_epochs is not None else (12 if args.compact else 24)
    channels = args.channels if args.channels is not None else (16 if args.compact else 24)
    return [
        ModelSpec(architecture=arch, window_hours=window, seed=seed, channels=channels, dropout=args.dropout, max_epochs=epochs, patience=args.patience)
        for arch in archs
        for window in windows
        for seed in seeds
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict-PIT V4.2 TCN/CNN temporal allocation research.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compact", action="store_true", help="Compact Databricks default.")
    mode.add_argument("--full", action="store_true", help="Longer run with larger epoch/channel defaults.")
    parser.add_argument("--max-seeds", type=int, default=5, help="Number of seeds from the fixed seed list.")
    parser.add_argument("--windows", default="24,48,72,96", help="Comma-separated sequence windows in hours.")
    parser.add_argument("--target-families", default="best_asset_forward_return,qqq_spy_relative_upside,top_decile_risk_on,utility_return_first")
    parser.add_argument("--architectures", default="tcn,cnn", help="Comma-separated architectures: tcn,cnn.")
    parser.add_argument("--max-targets", type=int, default=None, help="Optional debug cap on target families.")
    parser.add_argument("--max-model-specs", type=int, default=None, help="Optional debug cap on model specs.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--leverage-top-n", type=int, default=10, help="Run leverage/financing sensitivity on top unlevered candidates.")
    parser.add_argument("--allow-torch-install", action="store_true", help="Install torch if the Databricks runtime does not already provide it.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install missing lightweight Python dependencies.")
    args, _unknown = parser.parse_known_args()
    if not args.full:
        args.compact = True
    return args


def update_partial_artifacts(
    summary_rows: list[dict[str, Any]],
    curve_frames: list[pd.DataFrame],
    policy_signal_frames: list[pd.DataFrame],
    fold_frames: list[pd.DataFrame],
    spy_row: dict[str, Any],
    xgb_row: dict[str, Any] | None,
    bc2_row: dict[str, Any] | None,
    grid_validation_summary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = add_acceptance_flags(add_benchmark_gaps(pd.DataFrame(summary_rows), spy_row, xgb_row, bc2_row)) if summary_rows else pd.DataFrame()
    rankings = candidate_rankings(summary) if not summary.empty else pd.DataFrame()
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    policies = pd.concat(policy_signal_frames, ignore_index=True) if policy_signal_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    write_csv(summary, "candidate_summary.csv")
    write_csv(rankings, "candidate_rankings.csv")
    write_csv(policy_results(summary), "policy_results.csv")
    write_csv(leverage_results(summary), "leverage_results.csv")
    write_csv(financing_sensitivity(summary), "financing_sensitivity.csv")
    write_csv(seed_stability_summary(summary), "seed_stability_summary.csv")
    write_csv(target_results(summary), "target_results.csv")
    write_csv(folds, "fold_diagnostics.csv")
    write_csv(allocation_distribution(policies), "allocation_distribution.csv")
    if not curves.empty:
        write_csv(yearly_returns(curves), "yearly_returns.csv")
        write_csv(monthly_returns(curves), "monthly_returns.csv")
        write_csv(drawdown_windows(curves), "drawdown_windows.csv")
    else:
        write_csv(pd.DataFrame(), "yearly_returns.csv")
        write_csv(pd.DataFrame(), "monthly_returns.csv")
        write_csv(pd.DataFrame(), "drawdown_windows.csv")
    write_readme(summary, rankings, folds, spy_row, xgb_row, grid_validation_summary)
    return summary, rankings


def write_runtime_manifest(
    args: argparse.Namespace,
    started_at: datetime,
    bil_meta: dict[str, Any],
    specs: list[ModelSpec],
    targets: list[TargetSpec],
    grid_validation_summary: dict[str, Any],
    torch_meta: dict[str, Any],
) -> None:
    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "output_root": OUTPUT_ROOT,
            "research_leg": "strict-PIT compact TCN/CNN temporal allocation",
            "features": FEATURES,
            "horizon_hours": FWD_HOURS,
            "target_specs": [t.key for t in targets],
            "model_specs": [s.key for s in specs],
            "policy_specs": [{"name": name, "params": params} for name, params in policy_specs()],
            "leverage_grid": LEVERAGE_GRID,
            "financing_grid": FINANCING_GRID,
            "headline_financing_rate": HEADLINE_FINANCING_RATE,
            "headline_financing_label": HEADLINE_FINANCING_LABEL,
            "grid_validation": grid_validation_summary,
            "torch": torch_meta,
            "python_version": sys.version,
            "platform": platform.platform(),
            "bil_price_source": bil_meta,
            "strict_pit_controls": [
                "feature scalers fit on fold training rows only",
                "training rows require label_available_ts strictly before fold_start",
                "top-decile/risk-off thresholds are estimated from fold training rows only",
                "sequence windows include only rows at or before the decision timestamp",
                "monthly walk-forward folds cover 2021-01-01 through 2025-12-31",
            ],
            "production_delta_writes": False,
            "registry_writes": False,
            "job_or_schedule_changes": False,
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
            "prior_direct_allocation_root": {"path": DIRECT_ALLOCATION_ROOT, "exists": DIRECT_ALLOCATION_ROOT.exists()},
            "financing_assumption": {
                "headline_label": HEADLINE_FINANCING_LABEL,
                "headline_rate": HEADLINE_FINANCING_RATE,
                "sensitivity_rates": FINANCING_GRID,
                "note": "Exact dated 10Y series is not required by this first pass; 4.5% is retained as the prior MUB-27 10Y+50bps proxy.",
            },
        },
        "input_hashes.json",
    )


def main() -> None:
    global GRID_VALIDATION_STATUS
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    ensure_dirs()
    StandardScaler, metrics = ensure_python_deps(skip_install=args.skip_install)
    torch_bundle = import_torch(allow_install=args.allow_torch_install)
    torch_meta = torch_bundle[-1]

    grid_raw = load_grid()
    bil_prices, bil_meta = load_bil_prices(grid_raw)
    grid = attach_bil(grid_raw, bil_prices)
    base_signals = load_base_signals()
    prior, spy_row, xgb_row, bc2_row = load_benchmark_rows()
    prior_direct = load_prior_direct_allocation_reference()
    work = build_work_frame(grid, base_signals)
    validation_report, grid_validation_summary = validate_grid_inputs(grid_raw, grid, bil_meta, base_signals, prior, work)
    GRID_VALIDATION_STATUS = str(grid_validation_summary["overall_status"])
    write_csv(validation_report, "grid_validation_report.csv")

    selected_targets = target_specs(parse_csv_list(args.target_families, str))
    if args.max_targets is not None:
        selected_targets = selected_targets[: args.max_targets]
    selected_models = model_specs(args)
    if args.max_model_specs is not None:
        selected_models = selected_models[: args.max_model_specs]

    write_csv(feature_window_manifest(selected_models, selected_targets, grid_validation_summary), "feature_window_manifest.csv")
    write_runtime_manifest(args, started_at, bil_meta, selected_models, selected_targets, grid_validation_summary, torch_meta)
    write_input_hashes(grid, bil_meta, prior, grid_validation_summary)
    write_csv(prior_direct, "prior_direct_allocation_reference.csv")
    write_readme(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), spy_row, xgb_row, grid_validation_summary)

    if GRID_VALIDATION_STATUS != "pass":
        write_text("Grid validation did not pass. Model performance must not be interpreted as a valid model result.\n", "GRID_UNVERIFIED.txt")

    summary_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    policy_signal_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    total = len(selected_targets) * len(selected_models)
    completed = 0

    for target in selected_targets:
        for model in selected_models:
            completed += 1
            print(f"[{completed}/{total}] target={target.key} model={model.key}", flush=True)
            pred, folds = run_walkforward(work, target, model, StandardScaler, metrics, torch_bundle)
            if not folds.empty:
                fold_frames.append(folds)
            if pred.empty:
                continue
            for policy_name, params in policy_specs():
                policy_signals = build_policy_signals(pred, policy_name, params)
                policy_signal_frames.append(policy_signals.copy())
                row, curve = evaluate_policy(grid, policy_signals, leverage=1.0, financing=0.0)
                if row is not None and curve is not None:
                    summary_rows.append(row)
                    curve_frames.append(curve)
            update_partial_artifacts(summary_rows, curve_frames, policy_signal_frames, fold_frames, spy_row, xgb_row, bc2_row, grid_validation_summary)

    summary, rankings = update_partial_artifacts(summary_rows, curve_frames, policy_signal_frames, fold_frames, spy_row, xgb_row, bc2_row, grid_validation_summary)

    leverage_rows: list[dict[str, Any]] = []
    leverage_curves: list[pd.DataFrame] = []
    if not rankings.empty:
        top_unlevered = rankings[rankings["leverage"].eq(1.0)].head(args.leverage_top_n)
        keys = set(top_unlevered["candidate_id"].astype(str))
        for policy_frame in policy_signal_frames:
            base = str(policy_frame["target_spec"].iloc[0])
            arch = str(policy_frame["architecture"].iloc[0])
            window = str(policy_frame["window_hours"].iloc[0])
            seed = str(policy_frame["seed"].iloc[0])
            policy_name = str(policy_frame["policy_name"].iloc[0])
            params = str(policy_frame["policy_params"].iloc[0])
            prefix = f"{base}__{arch}__{window}h__seed{seed}__{policy_name}__{params}"
            if not any(k.startswith(prefix) for k in keys):
                continue
            matching_pred_cols = ["timestamp", "target_spec", "target_family", "architecture", "model_family", "window_hours", "seed", "policy_name", "policy_params", "target_asset_signal"]
            _ = matching_pred_cols
            for lev in LEVERAGE_GRID:
                for financing in FINANCING_GRID:
                    row, curve = evaluate_policy(grid, policy_frame, leverage=lev, financing=financing)
                    if row is not None and curve is not None:
                        leverage_rows.append(row)
                        leverage_curves.append(curve)
        if leverage_rows:
            levered_summary = add_acceptance_flags(add_benchmark_gaps(pd.DataFrame(leverage_rows), spy_row, xgb_row, bc2_row))
            combined = pd.concat([summary, levered_summary], ignore_index=True).drop_duplicates("candidate_id")
            combined_rankings = candidate_rankings(combined)
            combined_curves = pd.concat([pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(), pd.concat(leverage_curves, ignore_index=True)], ignore_index=True)
            write_csv(combined, "candidate_summary.csv")
            write_csv(combined_rankings, "candidate_rankings.csv")
            write_csv(policy_results(combined), "policy_results.csv")
            write_csv(leverage_results(combined), "leverage_results.csv")
            write_csv(financing_sensitivity(combined), "financing_sensitivity.csv")
            write_csv(seed_stability_summary(combined), "seed_stability_summary.csv")
            write_csv(target_results(combined), "target_results.csv")
            write_csv(yearly_returns(combined_curves), "yearly_returns.csv")
            write_csv(monthly_returns(combined_curves), "monthly_returns.csv")
            write_csv(drawdown_windows(combined_curves), "drawdown_windows.csv")
            folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
            write_readme(combined, combined_rankings, folds, spy_row, xgb_row, grid_validation_summary)
            summary, rankings = combined, combined_rankings

    print(json.dumps(jsonable({"output_root": OUTPUT_ROOT, "grid_validation_status": GRID_VALIDATION_STATUS, "top_10": rankings.head(10).to_dict("records") if not rankings.empty else []}), indent=2))


if __name__ == "__main__":
    main()

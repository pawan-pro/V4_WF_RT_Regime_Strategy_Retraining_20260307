# Databricks notebook source
"""B vs C lineage difference and reproducibility ladder.

Research-only. Reads faithful legacy CSV/model artifacts and writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_b_vs_c_lineage_diff_2026-05-04/`

No production Delta writes, Saxo changes, RT/dashboard/workflow changes, or
model registry writes are performed.
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
OUTPUT_ROOT = BASE / "v4_2_b_vs_c_lineage_diff_2026-05-04"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
CODE_DIR = OUTPUT_ROOT / "code"

GRID_PATH = BASE / "v4_1_csv" / "full_legacy_grid_clone_2026-04-30" / "grids" / "full_legacy_grid_clone.csv"
SAME_DATA_ROOT = BASE / "v4_2_same_data_teacher_student_2026-05-04"
C_PACKAGE_ROOT = BASE / "v4_2_c_full_sample_teacher_package_2026-05-04"
FAITHFUL_REPLAY_ROOT = BASE / "v4_1_csv" / "faithful_legacy_grid_model_replay_2026-04-30"
PERF_RECON_ROOT = BASE / "v4_1_csv" / "performance_methodology_reconciliation_2026-04-30"
REF_DIR = BASE / "comparison" / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected" / "corrected_variant"
REF_FULL_SIGNAL = REF_DIR / "full_signal_export.csv"
REF_WALKFORWARD = REF_DIR / "walkforward_signals.csv"
RECOVERED_NOTEBOOK_LOCAL = Path("01_Processing/04_Retraining/18_Freeze_Canonical_Baseline_And_Rerun_ALFRED_Corrected.py")
SAME_DATA_SCRIPT_LOCAL = Path("03_Research/02_Models/03_V4_2_Same_Data_Teacher_Student.py")
C_PACKAGE_SCRIPT_LOCAL = Path("03_Research/02_Models/04_V4_2_C_Full_Sample_Teacher_Package.py")

TRAIN_START = pd.Timestamp("2016-01-01")
LIVE_START = pd.Timestamp("2021-01-01")
LIVE_END = pd.Timestamp("2026-01-01")
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

C_CURRENT_PARAMS = {
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
    "tree_method": "hist",
    "random_state": 42,
}
RECOVERED_STUDENT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softmax",
    "num_class": 3,
    "tree_method": "hist",
    "random_state": 42,
}


def ensure_deps():
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


joblib, xgb = ensure_deps()


def ensure_dirs() -> None:
    for path in [FORENSICS_DIR, PERFORMANCE_DIR, SIGNALS_DIR, MANIFESTS_DIR, CODE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_output(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


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


def write_text(text: str, path: Path) -> None:
    assert_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.sort_index(axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        try:
            out["timestamp"] = out["timestamp"].dt.tz_localize(None)
        except Exception:
            pass
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


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


def load_grid() -> pd.DataFrame:
    grid = read_csv(GRID_PATH)
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["fwd_return_1w"] = grid["SPY_close"].shift(-FWD_HOURS) / grid["SPY_close"] - 1.0
    grid["raw_regime"] = grid["fwd_return_1w"].apply(label_regime)
    return grid


def current_trainable(grid: pd.DataFrame) -> pd.DataFrame:
    need = FEATURES + ["SPY_close", "QQQ_close", "raw_regime"]
    out = grid[(grid["timestamp"] >= TRAIN_START) & grid[need].notna().all(axis=1)].copy()
    out["raw_regime"] = out["raw_regime"].astype(int)
    return out.reset_index(drop=True)


def recovered_dataset_base(grid: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", *FEATURES]
    out = grid[cols].copy()
    out = out[(out["timestamp"] >= TRAIN_START) & (out["timestamp"] < LIVE_END)].dropna(subset=["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


def reference_walkforward() -> pd.DataFrame:
    ref = read_csv(REF_WALKFORWARD)
    ref["predicted_regime"] = pd.to_numeric(ref["predicted_regime"], errors="coerce").astype("Int64")
    ref = ref[(ref["timestamp"] >= EVAL_START) & (ref["timestamp"] <= EVAL_END)].dropna(subset=["predicted_regime"]).copy()
    ref["predicted_regime"] = ref["predicted_regime"].astype(int)
    ref["allocation_scale"] = 1.0
    ref["variant"] = "B_recovered_control"
    ref["regime_name"] = ref["predicted_regime"].map(REGIME_NAME)
    return ref[["timestamp", "variant", "predicted_regime", "regime_name", "allocation_scale"]].reset_index(drop=True)


def reference_full_signal() -> pd.DataFrame:
    full = read_csv(REF_FULL_SIGNAL)
    full["predicted_regime"] = pd.to_numeric(full["predicted_regime"], errors="coerce").astype("Int64")
    return full.dropna(subset=["predicted_regime"]).copy()


def monthly_folds() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = LIVE_START
    while start < LIVE_END:
        end = start + relativedelta(months=1)
        folds.append((start, end))
        start = end
    return folds


def model(params: dict[str, Any]):
    return xgb.XGBClassifier(**params)


def signal_frame(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = df[["timestamp", "predicted_regime"]].copy()
    out["predicted_regime"] = pd.to_numeric(out["predicted_regime"], errors="coerce").astype(int)
    out["variant"] = variant
    out["regime_name"] = out["predicted_regime"].map(REGIME_NAME)
    out["allocation_scale"] = 1.0
    return out[["timestamp", "variant", "predicted_regime", "regime_name", "allocation_scale"]].sort_values("timestamp").reset_index(drop=True)


def train_teacher_current(trainable: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    clf = model(params)
    clf.fit(trainable[FEATURES], trainable["raw_regime"])
    pseudo = trainable[["timestamp", "raw_regime", *FEATURES]].copy()
    pseudo["teacher_regime"] = clf.predict(pseudo[FEATURES]).astype(int)
    return pseudo


def train_teacher_recovered(grid: pd.DataFrame, params: dict[str, Any], fillna_for_prediction: bool) -> pd.DataFrame:
    df = grid.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp").copy()
    df["label"] = df["fwd_return_1w"].apply(label_regime)
    df_ml = df.dropna(subset=["label"]).copy()
    clf = model(params)
    clf.fit(df_ml[FEATURES], df_ml["label"].astype(int))
    x_full = df[FEATURES].fillna(0) if fillna_for_prediction else df[FEATURES]
    df["teacher_regime"] = clf.predict(x_full).astype(int)
    return df.reset_index()[["timestamp", "label", "teacher_regime", *FEATURES]].copy()


def walkforward_from_labels(
    grid: pd.DataFrame,
    labels: pd.DataFrame,
    label_col: str,
    variant: str,
    params: dict[str, Any],
    recovered_style: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if recovered_style:
        dataset = recovered_dataset_base(grid).merge(
            labels[["timestamp", label_col]].rename(columns={label_col: "regime"}),
            on="timestamp",
            how="left",
        )
        dataset = dataset.dropna(subset=["regime"]).copy()
        dataset["regime"] = pd.to_numeric(dataset["regime"], errors="coerce").astype(int)
    else:
        base = current_trainable(grid)
        dataset = base.merge(
            labels[["timestamp", label_col]].rename(columns={label_col: "regime"}),
            on="timestamp",
            how="inner",
        )
        dataset["regime"] = pd.to_numeric(dataset["regime"], errors="coerce").astype(int)

    signals = []
    fold_rows = []
    for fold_start, fold_end in monthly_folds():
        train_df = dataset[(dataset["timestamp"] >= TRAIN_START) & (dataset["timestamp"] < fold_start)].copy()
        test_df = dataset[(dataset["timestamp"] >= fold_start) & (dataset["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        clf = model(params)
        clf.fit(train_df[FEATURES], train_df["regime"])
        if recovered_style:
            probs = clf.predict_proba(test_df[FEATURES])
            preds = np.argmax(probs, axis=1)
            test_df["prob_regime0"] = probs[:, 0]
            test_df["prob_regime1"] = probs[:, 1]
            test_df["prob_regime2"] = probs[:, 2]
        else:
            preds = clf.predict(test_df[FEATURES])
        test_df["predicted_regime"] = preds.astype(int)
        signals.append(test_df[["timestamp", "predicted_regime"]].copy())
        fold_rows.append(
            {
                "variant": variant,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_label_distribution": json.dumps(dist(train_df["regime"]), sort_keys=True),
                "prediction_distribution": json.dumps(dist(test_df["predicted_regime"]), sort_keys=True),
            }
        )
    sig = signal_frame(pd.concat(signals, ignore_index=True), variant)
    return sig, pd.DataFrame(fold_rows)


def cached_model_signals(grid: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    rows = []
    signals = []
    base = recovered_dataset_base(grid)
    for fold_start, fold_end in monthly_folds():
        fold_dir = FAITHFUL_REPLAY_ROOT / "models" / "folds" / fold_start.strftime("%Y_%m")
        model_path = fold_dir / "model.joblib"
        test_df = base[(base["timestamp"] >= fold_start) & (base["timestamp"] < fold_end)].copy()
        rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "model_path": model_path,
                "model_exists": model_path.exists(),
                "test_rows": len(test_df),
            }
        )
        if not model_path.exists() or test_df.empty:
            continue
        clf = joblib.load(model_path)
        probs = clf.predict_proba(test_df[FEATURES])
        test_df["predicted_regime"] = np.argmax(probs, axis=1).astype(int)
        signals.append(test_df[["timestamp", "predicted_regime"]].copy())
    if not signals:
        return None, pd.DataFrame(rows)
    return signal_frame(pd.concat(signals, ignore_index=True), "C5_cached_models_if_available"), pd.DataFrame(rows)


def dist(series: pd.Series) -> dict[str, int]:
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def performance(grid: pd.DataFrame, signals: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sig = signals[["timestamp", "predicted_regime", "allocation_scale"]].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close"]] = data[["SPY_close", "QQQ_close"]].ffill()
    changed = data[["SPY_close", "QQQ_close"]].ne(data[["SPY_close", "QQQ_close"]].shift()).any(axis=1)
    data = data[changed].dropna(subset=["SPY_close", "QQQ_close", "predicted_regime"]).reset_index(drop=True)
    data["predicted_regime"] = pd.to_numeric(data["predicted_regime"], errors="coerce").astype(int)
    data["allocation_scale"] = pd.to_numeric(data["allocation_scale"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    equity = [INITIAL_CAPITAL]
    assets = ["START"]
    strategy_returns = [0.0]
    costs = [0.0]
    pos_changed = [False]
    active = "START"
    for idx in range(1, len(data)):
        regime = int(data["predicted_regime"].iloc[idx - 1])
        scale = float(data["allocation_scale"].iloc[idx - 1])
        asset = REGIME_TO_ASSET.get(regime, "CASH")
        if asset == "QQQ":
            gross = scale * (data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0)
        elif asset == "SPY":
            gross = scale * (data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0)
        else:
            gross = 0.0
        changed_asset = asset != active
        cost = SWITCH_COST if changed_asset else 0.0
        ret = gross - cost
        equity.append(equity[-1] * (1.0 + ret))
        assets.append(asset)
        strategy_returns.append(ret)
        costs.append(cost)
        pos_changed.append(changed_asset)
        active = asset
    hourly = data.copy()
    hourly["variant"] = variant
    hourly["target_asset"] = assets
    hourly["strategy_return"] = strategy_returns
    hourly["switch_cost"] = costs
    hourly["position_changed"] = pos_changed
    hourly["strategy_equity"] = equity
    hourly["drawdown"] = hourly["strategy_equity"] / hourly["strategy_equity"].cummax() - 1.0
    hourly["report_date"] = pd.to_datetime(hourly["timestamp"]).dt.date.astype(str)
    daily = hourly.loc[hourly.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    ret = daily["strategy_equity"].pct_change().dropna()
    downside = ret[ret < 0]
    final = float(daily["strategy_equity"].iloc[-1])
    dd = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    metrics = {
        "variant": variant,
        "CAGR": (final / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max_DD": float(dd.min()),
        "final_equity": final,
        "switch_count": int(hourly["position_changed"].astype(bool).sum()),
        "hourly_rows": len(hourly),
        "daily_rows": len(daily),
    }
    return hourly, daily, metrics


def match_report(candidate: pd.DataFrame, reference: pd.DataFrame, name: str) -> dict[str, Any]:
    cand = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "candidate"})
    ref = reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference"})
    cmp = cand.merge(ref, on="timestamp", how="outer", indicator=True)
    both = cmp[cmp["_merge"] == "both"].copy()
    if len(both):
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


def mismatch_sample(candidate: pd.DataFrame, reference: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    cand = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "C_pred"})
    ref = reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "B_pred"})
    cmp = ref.merge(cand, on="timestamp", how="outer", indicator=True).sort_values("timestamp")
    both = cmp[cmp["_merge"] == "both"].copy()
    return both[pd.to_numeric(both["B_pred"], errors="coerce") != pd.to_numeric(both["C_pred"], errors="coerce")].head(limit)


def monthly_match(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    cand = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "candidate"})
    ref = reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference"})
    both = ref.merge(cand, on="timestamp", how="inner")
    both["match"] = both["candidate"].astype(int) == both["reference"].astype(int)
    both["month"] = both["timestamp"].dt.to_period("M").astype(str)
    return both.groupby("month").agg(rows=("match", "size"), match_rate=("match", "mean")).reset_index()


def yearly_match(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    cand = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "candidate"})
    ref = reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference"})
    both = ref.merge(cand, on="timestamp", how="inner")
    both["match"] = both["candidate"].astype(int) == both["reference"].astype(int)
    both["year"] = both["timestamp"].dt.year
    return both.groupby("year").agg(rows=("match", "size"), match_rate=("match", "mean")).reset_index()


def transition_matrix(signals: pd.DataFrame, name: str) -> pd.DataFrame:
    s = signals.sort_values("timestamp").copy()
    s["prev"] = s["predicted_regime"].shift(1)
    s = s.dropna(subset=["prev"])
    tab = pd.crosstab(s["prev"].astype(int), s["predicted_regime"].astype(int))
    tab.index.name = f"{name}_prev"
    return tab.reset_index()


def inventory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for root in paths:
        if not root.exists():
            rows.append({"path": root, "artifact_type": "missing", "relevance": "input"})
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for p in files:
            rel = "B" if ("faithful_legacy_grid_model_replay" in str(p) or "corrected_variant" in str(p)) else "C" if ("v4_2_c_full_sample_teacher" in str(p) or "v4_2_same_data_teacher" in str(p)) else "shared"
            row = {
                "path": str(p),
                "artifact_type": p.suffix.lower().lstrip(".") or "file",
                "size": p.stat().st_size,
                "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "sha256": file_sha256(p),
                "row_count": None,
                "min_timestamp": None,
                "max_timestamp": None,
                "columns": None,
                "relevance": rel,
            }
            if p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(p, nrows=5)
                    row["columns"] = json.dumps(list(df.columns))
                    row["row_count"] = sum(1 for _ in p.open("rb")) - 1
                    if "timestamp" in df.columns:
                        full_ts = pd.read_csv(p, usecols=["timestamp"])
                        ts = pd.to_datetime(full_ts["timestamp"], errors="coerce")
                        row["min_timestamp"] = ts.min()
                        row["max_timestamp"] = ts.max()
                except Exception as exc:
                    row["columns"] = f"inventory_error: {exc}"
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Variant:
    name: str
    signals: pd.DataFrame
    teacher_labels: pd.DataFrame | None
    fold_manifest: pd.DataFrame
    interpretation: str
    uses_recovered_params: bool = False
    uses_recovered_preprocessing: bool = False
    uses_recovered_teacher_labels: bool = False
    uses_recovered_walkforward_signals: bool = False
    uses_recovered_model_binaries: bool = False
    fresh_training_or_cached: str = "fresh_training"


def main() -> None:
    ensure_dirs()
    for src in [SAME_DATA_SCRIPT_LOCAL, C_PACKAGE_SCRIPT_LOCAL]:
        if src.exists():
            shutil.copy2(src, CODE_DIR / src.name)

    grid = load_grid()
    train_current = current_trainable(grid)
    b_ref = reference_walkforward()
    b_full = reference_full_signal()
    c0 = read_csv(C_PACKAGE_ROOT / "signals" / "walkforward_student_signals.csv")
    c0 = signal_frame(c0, "C0_current")

    write_csv(inventory([GRID_PATH, SAME_DATA_ROOT, C_PACKAGE_ROOT, FAITHFUL_REPLAY_ROOT, PERF_RECON_ROOT, REF_FULL_SIGNAL, REF_WALKFORWARD]), FORENSICS_DIR / "artifact_inventory.csv")

    feature_diff = pd.DataFrame(
        {
            "feature": FEATURES,
            "in_grid": [f in grid.columns for f in FEATURES],
            "in_B_full_signal": [f in b_full.columns for f in FEATURES],
            "in_C_package_full_signal": [f in read_csv(C_PACKAGE_ROOT / "signals" / "full_signal_export.csv").columns for f in FEATURES],
            "order_index": list(range(len(FEATURES))),
        }
    )
    write_csv(feature_diff, FORENSICS_DIR / "feature_column_diff.csv")

    row_membership = []
    for name, frame in [("B_walkforward", b_ref), ("C0_current", c0)]:
        row_membership.append(
            {
                "name": name,
                "rows": len(frame),
                "min_timestamp": frame["timestamp"].min(),
                "max_timestamp": frame["timestamp"].max(),
                "timestamp_hash": frame_hash(frame[["timestamp"]]),
            }
        )
    write_csv(pd.DataFrame(row_membership), FORENSICS_DIR / "input_row_membership_diff.csv")
    write_json(
        {
            "grid_path": GRID_PATH,
            "grid_sha256": file_sha256(GRID_PATH),
            "grid_rows": len(grid),
            "grid_min_timestamp": grid["timestamp"].min(),
            "grid_max_timestamp": grid["timestamp"].max(),
            "train_start": TRAIN_START,
            "live_start": LIVE_START,
            "live_end": LIVE_END,
            "eval_start": EVAL_START,
            "eval_end": EVAL_END,
            "current_trainable_rows": len(train_current),
            "current_trainable_min_timestamp": train_current["timestamp"].min(),
            "current_trainable_max_timestamp": train_current["timestamp"].max(),
            "rows_missing_any_feature": int(grid[FEATURES].isna().any(axis=1).sum()),
            "B_signal_rows": len(b_ref),
            "C0_signal_rows": len(c0),
            "B_C_same_timestamp_hash": frame_hash(b_ref[["timestamp"]]) == frame_hash(c0[["timestamp"]]),
        },
        FORENSICS_DIR / "input_grid_validation.json",
    )

    write_json({"teacher": RECOVERED_TEACHER_PARAMS, "student": RECOVERED_STUDENT_PARAMS, "source": "recovered notebook branch feat/freeze-canonical-baseline-and-rerun-alfred-corrected-2026-04-20"}, FORENSICS_DIR / "teacher_params_B.json")
    write_json({"teacher": C_CURRENT_PARAMS, "student": C_CURRENT_PARAMS, "source": "03_V4_2_Same_Data_Teacher_Student.py / C package"}, FORENSICS_DIR / "teacher_params_C.json")

    variants: list[Variant] = []
    variants.append(Variant("C0_current", c0, read_csv(C_PACKAGE_ROOT / "signals" / "full_signal_export.csv"), pd.DataFrame(), "Current C package: fresh full-sample teacher 500/depth6 and fresh monthly students."))

    # C1: recovered params where recoverable, current row filtering.
    c1_teacher = train_teacher_current(train_current, RECOVERED_TEACHER_PARAMS)
    c1_sig, c1_folds = walkforward_from_labels(grid, c1_teacher, "teacher_regime", "C1_recovered_params", RECOVERED_STUDENT_PARAMS, recovered_style=False)
    variants.append(Variant("C1_recovered_params", c1_sig, c1_teacher, c1_folds, "Recovered teacher 200/depth4 and recovered student 500/depth6, but current C row filtering.", uses_recovered_params=True))

    # C2: recovered params plus recovered preprocessing.
    c2_teacher = train_teacher_recovered(grid, RECOVERED_TEACHER_PARAMS, fillna_for_prediction=True)
    c2_sig, c2_folds = walkforward_from_labels(grid, c2_teacher, "teacher_regime", "C2_recovered_preprocessing", RECOVERED_STUDENT_PARAMS, recovered_style=True)
    variants.append(Variant("C2_recovered_preprocessing", c2_sig, c2_teacher, c2_folds, "Recovered params and recovered-style fillna/prediction/merge schedule.", uses_recovered_params=True, uses_recovered_preprocessing=True))

    # C3: recovered teacher labels but fresh monthly students.
    c3_labels = b_full[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "teacher_regime"})
    c3_sig, c3_folds = walkforward_from_labels(grid, c3_labels, "teacher_regime", "C3_recovered_teacher_labels", RECOVERED_STUDENT_PARAMS, recovered_style=True)
    variants.append(Variant("C3_recovered_teacher_labels", c3_sig, c3_labels, c3_folds, "Recovered full_signal_export predicted_regime labels; fresh monthly student training.", uses_recovered_params=True, uses_recovered_preprocessing=True, uses_recovered_teacher_labels=True))

    # C4: recovered walk-forward signals direct.
    variants.append(Variant("C4_recovered_walkforward_signals", signal_frame(b_ref, "C4_recovered_walkforward_signals"), b_full, pd.DataFrame(), "Recovered walkforward_signals.csv direct replay.", uses_recovered_walkforward_signals=True, fresh_training_or_cached="cached_signal_replay"))

    # C5: recovered model binaries, if available.
    c5_sig, c5_folds = cached_model_signals(grid)
    if c5_sig is not None:
        variants.append(Variant("C5_cached_models_if_available", c5_sig, b_full, c5_folds, "Recovered cached fold model binaries loaded and rerun on faithful grid.", uses_recovered_params=True, uses_recovered_preprocessing=True, uses_recovered_model_binaries=True, fresh_training_or_cached="cached_model_inference"))
    else:
        write_csv(c5_folds, FORENSICS_DIR / "C5_cached_model_inventory.csv")

    # Persist signals and evaluate.
    summary_rows = []
    ladder_rows = []
    match_rows = []
    fold_frames = []
    for v in variants:
        write_csv(v.signals, SIGNALS_DIR / f"recreated_{v.name}.csv")
        if v.name == "C0_current":
            write_csv(v.signals, SIGNALS_DIR / "recreated_C0_current.csv")
        if v.name == "C1_recovered_params":
            write_csv(v.signals, SIGNALS_DIR / "recreated_C1_recovered_params.csv")
        if v.name == "C2_recovered_preprocessing":
            write_csv(v.signals, SIGNALS_DIR / "recreated_C2_recovered_preprocessing.csv")
        if v.name == "C3_recovered_teacher_labels":
            write_csv(v.signals, SIGNALS_DIR / "recreated_C3_recovered_teacher_labels.csv")
        if v.name == "C4_recovered_walkforward_signals":
            write_csv(v.signals, SIGNALS_DIR / "recreated_C4_recovered_walkforward_signals.csv")
        if v.name == "C5_cached_models_if_available":
            write_csv(v.signals, SIGNALS_DIR / "recreated_C5_cached_models_if_available.csv")
        if v.teacher_labels is not None:
            cols = [c for c in ["timestamp", "raw_regime", "label", "teacher_regime", "predicted_regime"] if c in v.teacher_labels.columns]
            if cols:
                write_csv(v.teacher_labels[cols].copy(), SIGNALS_DIR / f"{v.name}_teacher_labels.csv")
        hourly, daily, metrics = performance(grid, v.signals, v.name)
        write_csv(hourly, PERFORMANCE_DIR / f"{v.name}_hourly_equity.csv")
        write_csv(daily, PERFORMANCE_DIR / f"{v.name}_daily_equity.csv")
        m = match_report(v.signals, b_ref, f"{v.name}_vs_B")
        match_rows.append(m)
        teacher_match = np.nan
        if v.teacher_labels is not None and "teacher_regime" in v.teacher_labels.columns:
            ref_labels = b_full[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "B_teacher"})
            cand_labels = v.teacher_labels[["timestamp", "teacher_regime"]]
            tm = ref_labels.merge(cand_labels, on="timestamp", how="inner")
            if len(tm):
                teacher_match = float((tm["B_teacher"].astype(int) == tm["teacher_regime"].astype(int)).mean())
        row = {
            **metrics,
            "signal_match_vs_B": m["match_rate"],
            "teacher_label_match_vs_B": teacher_match,
            "walkforward_signal_match_vs_B": m["match_rate"],
            "interpretation": v.interpretation,
        }
        summary_rows.append(row)
        ladder_rows.append(
            {
                "variant": v.name,
                "uses_same_grid": True,
                "uses_recovered_params": v.uses_recovered_params,
                "uses_recovered_preprocessing": v.uses_recovered_preprocessing,
                "uses_recovered_teacher_labels": v.uses_recovered_teacher_labels,
                "uses_recovered_walkforward_signals": v.uses_recovered_walkforward_signals,
                "uses_recovered_model_binaries": v.uses_recovered_model_binaries,
                "fresh_training_or_cached": v.fresh_training_or_cached,
                "signal_match_vs_B": m["match_rate"],
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max_DD": metrics["Max_DD"],
                "final_equity": metrics["final_equity"],
                "switch_count": metrics["switch_count"],
                "interpretation": v.interpretation,
            }
        )
        if not v.fold_manifest.empty:
            fold_frames.append(v.fold_manifest.assign(variant=v.name))

    summary = pd.DataFrame(summary_rows)
    ladder = pd.DataFrame(ladder_rows)
    matches = pd.DataFrame(match_rows)
    write_csv(summary, PERFORMANCE_DIR / "recreated_variant_summary.csv")
    write_csv(ladder, FORENSICS_DIR / "reproducibility_ladder.csv")
    write_csv(matches, FORENSICS_DIR / "signal_match_summary.csv")
    if fold_frames:
        fold_diff = pd.concat(fold_frames, ignore_index=True)
    else:
        fold_diff = pd.DataFrame()
    write_csv(fold_diff, FORENSICS_DIR / "fold_by_fold_row_count_diff.csv")

    # Focused requested comparisons and diagnostics.
    write_csv(mismatch_sample(c0, b_ref), FORENSICS_DIR / "signal_mismatches_sample.csv")
    write_csv(monthly_match(c0, b_ref), FORENSICS_DIR / "monthly_signal_match.csv")
    write_csv(yearly_match(c0, b_ref), FORENSICS_DIR / "yearly_signal_match.csv")
    write_csv(transition_matrix(b_ref, "B"), FORENSICS_DIR / "regime_transition_matrix_B.csv")
    write_csv(transition_matrix(c0, "C"), FORENSICS_DIR / "regime_transition_matrix_C.csv")
    write_csv(pd.concat([matches, pd.DataFrame([match_report(c0, variants[3].signals, "C0_vs_C3"), match_report(c0, variants[2].signals, "C0_vs_C2")])], ignore_index=True), FORENSICS_DIR / "fold_by_fold_signal_match.csv")

    # Teacher distributions.
    teacher_rows = []
    for name, labels, col in [
        ("B_recovered_full_signal", b_full, "predicted_regime"),
        ("C0_package_full_signal", variants[0].teacher_labels, "teacher_pseudo_regime" if variants[0].teacher_labels is not None and "teacher_pseudo_regime" in variants[0].teacher_labels.columns else "predicted_regime"),
        ("C1_recovered_params", c1_teacher, "teacher_regime"),
        ("C2_recovered_preprocessing", c2_teacher, "teacher_regime"),
        ("C3_recovered_teacher_labels", c3_labels, "teacher_regime"),
    ]:
        if labels is not None and col in labels.columns:
            teacher_rows.append({"label_source": name, **dist(labels[col])})
    write_csv(pd.DataFrame(teacher_rows), FORENSICS_DIR / "teacher_label_distribution_comparison.csv")
    teacher_full_match = []
    for v in variants:
        if v.teacher_labels is not None:
            col = "teacher_regime" if "teacher_regime" in v.teacher_labels.columns else "teacher_pseudo_regime" if "teacher_pseudo_regime" in v.teacher_labels.columns else "predicted_regime" if "predicted_regime" in v.teacher_labels.columns else None
            if col:
                cand = v.teacher_labels[["timestamp", col]].rename(columns={col: "predicted_regime"})
                teacher_full_match.append(match_report(signal_frame(cand, v.name), signal_frame(b_full, "B_full"), f"{v.name}_teacher_vs_B_full"))
    write_csv(pd.DataFrame(teacher_full_match), FORENSICS_DIR / "teacher_full_signal_match.csv")

    write_csv(pd.DataFrame([summary[summary["variant"] == "C4_recovered_walkforward_signals"].iloc[0].to_dict()]), PERFORMANCE_DIR / "recomputed_B.csv")
    write_csv(pd.DataFrame([summary[summary["variant"] == "C0_current"].iloc[0].to_dict()]), PERFORMANCE_DIR / "recomputed_C.csv")

    teacher_md = f"""# Teacher Logic Diff

## B Recovered Control

- Full-sample teacher parameters: `{RECOVERED_TEACHER_PARAMS}`.
- Target: `SPY_close.shift(-168) / SPY_close - 1`.
- Labels: `0` when return `> 0.005`, `2` when return `< -0.005`, else `1`.
- Teacher fit rows: rows with non-null label from the recovered corrected silver/grid path.
- Teacher inference uses `df[FEATURES].fillna(0)`.
- Student labels are loaded from cached/recovered `full_signal_export.csv.predicted_regime`.

## C Current

- Full-sample teacher parameters: `{C_CURRENT_PARAMS}`.
- Same horizon and thresholds, but rows are prefiltered to all feature columns non-null.
- Teacher labels are regenerated, not loaded from recovered `full_signal_export.csv`.
- Student predictions come from fresh monthly models trained on regenerated teacher labels.

## Main Finding

    The ladder determines whether the mismatch is from C's fresh teacher implementation or from unrecoverable cached artifacts. If C2 matches B, the recovered code path is independently reproducible from the faithful grid using recovered params/preprocessing.
"""
    write_text(teacher_md, FORENSICS_DIR / "teacher_logic_diff.md")

    walk_md = f"""# Walk-Forward Logic Diff

Recovered B trains monthly students using `full_signal_export.csv.predicted_regime` as labels, fold schedule from `{LIVE_START.date()}` monthly through `{LIVE_END.date()}`, train rows `< fold_start`, and test rows `>= fold_start` and `< fold_end`.

C current follows the same broad monthly schedule, but its label source is regenerated teacher output from current C. The signal match of C0 vs B is `{matches.loc[matches['comparison'].eq('C0_current_vs_B'), 'match_rate'].iloc[0]:.6f}`.

C3 uses recovered teacher labels with fresh monthly students. If C3 reaches B, labels are sufficient. If C3 remains below B while C4 reaches B, recovered walk-forward signals/model binary state are necessary.
"""
    write_text(walk_md, FORENSICS_DIR / "walkforward_logic_diff.md")

    perf_md = """# Performance Methodology Diff

All ladder variants are recomputed through one unified performance function:

- prior-bar / next-bar allocation
- faithful-grid price-update hourly rows only
- 5 bps switch cost on every asset transition, including transitions into cash
- daily-last equity based on the faithful legacy timestamp date
- fixed 2021-01-01 to 2025-12-31 five-year CAGR
- daily Sharpe/Sortino with 252 annualization and risk-free 0

Therefore the C0-C5 differences in this package are signal/lineage differences, not performance-engine differences.
"""
    write_text(perf_md, PERFORMANCE_DIR / "performance_methodology_diff.md")

    c4 = ladder[ladder["variant"] == "C4_recovered_walkforward_signals"].iloc[0]
    c3 = ladder[ladder["variant"] == "C3_recovered_teacher_labels"].iloc[0]
    c2 = ladder[ladder["variant"] == "C2_recovered_preprocessing"].iloc[0]
    c0row = ladder[ladder["variant"] == "C0_current"].iloc[0]
    root_cause = (
        "Current C differs from B because it uses a different full-sample teacher configuration and row/preprocessing path. "
        "The ladder identifies whether recovered params/preprocessing are sufficient, or whether cached labels/signals are required."
    )
    if c2["signal_match_vs_B"] >= 0.999:
        root_cause = "C current diverges because it changed the recovered teacher params/preprocessing path; C2 fresh retraining with recovered params/preprocessing reproduces B exactly without recovered cached labels/signals."
    elif c4["signal_match_vs_B"] == 1.0 and c3["signal_match_vs_B"] < 0.999:
        root_cause = "Recovered walk-forward signals/model lineage, not fresh teacher-student code alone, is required to reproduce B exactly."
    elif c3["signal_match_vs_B"] >= 0.999:
        root_cause = "Recovered teacher labels are sufficient to reproduce B with fresh monthly students; C differs because it regenerates teacher labels."
    elif c2["signal_match_vs_B"] > c0row["signal_match_vs_B"]:
        root_cause = "Recovered params/preprocessing improve reproducibility but do not fully reproduce B without cached recovered labels/signals."

    answer = {
        "root_cause": root_cause,
        "is_due_to_xgboost_params": bool(c2["signal_match_vs_B"] > c0row["signal_match_vs_B"]),
        "is_due_to_preprocessing": bool(c2["signal_match_vs_B"] > c0row["signal_match_vs_B"]),
        "is_due_to_cached_recovered_labels_or_signals": bool(c4["signal_match_vs_B"] == 1.0 and c2["signal_match_vs_B"] < 0.999),
        "can_C_reach_20_8405_without_recovered_cached_labels_or_signals": bool((ladder[~ladder["uses_recovered_teacher_labels"] & ~ladder["uses_recovered_walkforward_signals"]]["CAGR"] >= 0.20840487336126445 - 1e-9).any()),
        "minimum_change_to_reproduce_B": "Use recovered teacher params plus recovered preprocessing/fillna/merge schedule" if c2["signal_match_vs_B"] >= 0.999 else "Use recovered walkforward_signals.csv directly" if c3["signal_match_vs_B"] < 0.999 else "Use recovered full_signal_export.csv teacher labels and recovered walk-forward preprocessing",
        "production_governance": "Treat B as forensic/reference only unless the signal lineage can be reproduced under strict PIT code and data controls. Fresh C remains research-only/full-sample non-PIT.",
        "next_research_action": "Proceed to an independent strict rolling model such as TFT, and compare against B/C/D/SPY without teacher-label dependency.",
    }
    write_json(answer, FORENSICS_DIR / "b_vs_c_lineage_diff_summary.json")
    summary_md = f"""# B vs C Lineage Difference Summary

## Root Cause

{root_cause}

## Reproducibility Ladder

{md_table(ladder)}

## Direct Answers

1. Root cause: {root_cause}
2. Params/preprocessing/package/row filters: see C1 and C2 rows in the ladder; they quantify whether recovered params/preprocessing move signal match and performance.
3. Can C reach 20.8405% without recovered cached labels/signals: `{answer['can_C_reach_20_8405_without_recovered_cached_labels_or_signals']}`.
4. Minimum change needed to reproduce B: `{answer['minimum_change_to_reproduce_B']}`.
5. B independently reproducible from code alone: not established unless C1/C2/C3 reach B; otherwise B is reproducible as a cached recovered artifact replay.
6. Governance: {answer['production_governance']}
7. Next action: {answer['next_research_action']}

## Safety

Research-only. No production Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, workflows, schedules, or model registry assets were modified.
"""
    write_text(summary_md, FORENSICS_DIR / "b_vs_c_lineage_diff_summary.md")

    runtime = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "joblib_version": joblib.__version__,
        "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "output_root": OUTPUT_ROOT,
        "grid_path": GRID_PATH,
        "grid_sha256": file_sha256(GRID_PATH),
        "production_delta_writes": False,
        "model_registry_writes": False,
        "saxo_or_rt_writes": False,
        "dashboard_writes": False,
        "workflow_or_schedule_changes": False,
    }
    write_json(runtime, MANIFESTS_DIR / "runtime_manifest.json")
    print(ladder.to_string(index=False))
    print(json.dumps(answer, indent=2))
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

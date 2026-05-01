"""Train monthly walk-forward XGBoost models from V4.1 grid CSVs only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dateutil.relativedelta import relativedelta

from v4_1_common import FEATURES, TIMING_VARIANTS, V4_1_ROOT, ensure_v4_1_dirs, parse_utc, print_json, sha256_file, write_csv, write_json


TRAIN_START = pd.Timestamp("2016-01-01")
LIVE_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")


def xgb_params() -> dict:
    return {
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


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**xgb_params())


def label_regime(ret: float) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def train_variant(root: Path, variant: str) -> dict:
    grid_path = root / "grids" / variant / "v4_1_silver_grid.csv"
    grid = pd.read_csv(grid_path)
    grid["timestamp"] = parse_utc(grid["timestamp"])
    grid = grid.sort_values("timestamp").reset_index(drop=True)
    for col in FEATURES + ["SPY_close", "QQQ_close"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["fwd_return_1w"] = grid["SPY_close"].shift(-168) / grid["SPY_close"] - 1.0
    grid["regime"] = grid["fwd_return_1w"].apply(label_regime)
    trainable = grid[(grid["timestamp"] >= TRAIN_START) & grid["regime"].notna()].dropna(subset=FEATURES).copy()
    trainable["regime"] = trainable["regime"].astype(int)

    retrain_dates = []
    d = LIVE_START
    while d <= EVAL_END + pd.offsets.MonthBegin(1):
        retrain_dates.append(d)
        d += relativedelta(months=1)

    signal_frames = []
    fold_rows = []
    model_dir = root / "models" / variant / "v4_1_xgboost_walkforward"
    model_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(retrain_dates) - 1):
        fold_start = retrain_dates[idx]
        fold_end = min(retrain_dates[idx + 1], EVAL_END + pd.Timedelta(seconds=1))
        train_df = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] < fold_start)]
        test_df = trainable[(trainable["timestamp"] >= fold_start) & (trainable["timestamp"] < fold_end)].copy()
        if train_df.empty or test_df.empty:
            continue
        model = build_model()
        model.fit(train_df[FEATURES], train_df["regime"])
        test_df["predicted_regime"] = model.predict(test_df[FEATURES]).astype(int)
        test_df["regime_name"] = test_df["predicted_regime"].map({0: "Expansion", 1: "Slowdown", 2: "Recession"})
        signal_frames.append(test_df[["timestamp", "regime", "predicted_regime", "regime_name"]])
        model_path = model_dir / f"fold_{fold_start.strftime('%Y_%m')}.joblib"
        joblib.dump(model, model_path)
        fold_rows.append(
            {
                "fold_start": fold_start.isoformat(),
                "fold_end": (fold_end - pd.Timedelta(seconds=1)).isoformat(),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "accuracy": float((test_df["regime"] == test_df["predicted_regime"]).mean()),
                "model_artifact_path": str(model_path),
            }
        )
    if not signal_frames:
        raise RuntimeError(f"No walk-forward folds produced for {variant}")

    wf_signals = pd.concat(signal_frames).sort_values("timestamp").reset_index(drop=True)
    signal_dir = root / "signals" / variant
    write_csv(wf_signals, signal_dir / "v4_1_walkforward_signals.csv", root)

    final_train = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] <= EVAL_END)].copy()
    live_model = build_model()
    live_model.fit(final_train[FEATURES], final_train["regime"])
    full = grid.dropna(subset=FEATURES).copy()
    full["predicted_regime"] = live_model.predict(full[FEATURES]).astype(int)
    full["regime_name"] = full["predicted_regime"].map({0: "Expansion", 1: "Slowdown", 2: "Recession"})
    write_csv(full, signal_dir / "v4_1_full_signal_export.csv", root)
    final_model_path = model_dir / "final_2021_2025_research_model.joblib"
    joblib.dump(live_model, final_model_path)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "xgboost_params": xgb_params(),
        "features": FEATURES,
        "target_definition": "fwd_return_1w = SPY_close.shift(-168) / SPY_close - 1",
        "label_definition": {"Expansion_0": "fwd_return_1w > 0.005", "Recession_2": "fwd_return_1w < -0.005", "Slowdown_1": "otherwise"},
        "train_start": TRAIN_START.isoformat(),
        "test_live_start": LIVE_START.isoformat(),
        "initial_evaluation_end": EVAL_END.isoformat(),
        "folds": fold_rows,
        "final_model_artifact_path": str(final_model_path),
        "input_grid_path": str(grid_path),
        "input_grid_hash": sha256_file(grid_path),
        "production_delta_aliases_modified": False,
        "model_registry_modified": False,
    }
    write_json(manifest, root / "manifests" / variant / "v4_1_training_manifest.json", root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(V4_1_ROOT))
    parser.add_argument("--variant", action="append", choices=TIMING_VARIANTS)
    args = parser.parse_args()
    root = ensure_v4_1_dirs(args.root)
    results = {variant: train_variant(root, variant) for variant in (args.variant or TIMING_VARIANTS)}
    print_json({"root": root, "training": results})


if __name__ == "__main__":
    main()

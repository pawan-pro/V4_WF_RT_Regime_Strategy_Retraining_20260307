# Databricks notebook source
"""MUB-27 V4.2 CNN48 seed-stability runner.

Focused, resumable research runner for the CNN 48h temporal allocation clue
found in the partial TCN/CNN run. This script intentionally runs one
target/window/loss/seed block at a time, writes per-seed artifacts, rebuilds
summaries from disk, and releases tensors between seeds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "20_V4_2_TCN_CNN_Temporal_Allocation.py"
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
DEFAULT_OUTPUT_ROOT = BASE / "v4_2_cnn48_stability_runner_2026-05-15"
EXPECTED_GRID_SHA256 = "b44199c4f37f904e57fd999017f58067f9117964e49f33b14927cb1ce0f0c4e1"

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
TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
INITIAL_CAPITAL = 5_000_000.0
CAGR_YEARS = 5.0
TRADING_DAYS = 252.0
SWITCH_COST = 0.0005
MAX_ACCEPTED_DRAWDOWN = -0.75
DEFAULT_SEEDS = "11,23,37,53,71,89,101,113,127,149,163,181,197,211,229,241,263,281,307,331,353,379,401,421,443,461,487,503,521,547"


def load_base_module():
    spec = importlib.util.spec_from_file_location("v4_tcn_cnn_temporal_allocation", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE_MOD = load_base_module()


@dataclass(frozen=True)
class RunSpec:
    model_family: str
    window: int
    target_family: str
    policy: str
    loss_mode: str
    seed: int
    channels: int
    dropout: float
    max_epochs: int
    patience: int

    @property
    def candidate_key(self) -> str:
        return (
            f"{self.target_family}__{self.loss_mode}__{self.model_family}"
            f"__{self.window}h__{self.policy}__seed{self.seed}"
        )

    @property
    def group_key(self) -> str:
        return f"{self.target_family}__{self.loss_mode}__{self.model_family}__{self.window}h__{self.policy}"


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


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


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
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if math.isnan(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_output_path(path: Path, output_root: Path) -> None:
    root = output_root.resolve()
    target = path.resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Refusing write outside research output root: {path}")


def write_csv(df: pd.DataFrame, output_root: Path, name: str) -> Path:
    path = output_root / name
    assert_output_path(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], output_root: Path, name: str) -> Path:
    path = output_root / name
    assert_output_path(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text(text: str, output_root: Path, name: str) -> Path:
    path = output_root / name
    assert_output_path(path, output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def seed_dir(output_root: Path, spec: RunSpec) -> Path:
    safe = spec.candidate_key.replace("{", "").replace("}", "").replace('"', "").replace(" ", "")
    return output_root / "seed_runs" / safe


def done_path(output_root: Path, spec: RunSpec) -> Path:
    return seed_dir(output_root, spec) / "done.json"


def monthly_folds(work: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(EVAL_START.normalize(), EVAL_END.normalize(), freq="MS")
    return [(pd.Timestamp(start), min(pd.Timestamp(start + pd.offsets.MonthBegin(1)), FINAL_FOLD_END)) for start in starts]


def forward_path_volatility(price: pd.Series, horizon: int) -> pd.Series:
    returns = pd.concat([price.shift(-i) / price.shift(-(i - 1)) - 1.0 for i in range(1, horizon + 1)], axis=1)
    return returns.std(axis=1, ddof=0).fillna(0.0)


def attach_forward_volatility(work: pd.DataFrame) -> pd.DataFrame:
    out = work.copy()
    for asset, col in [("SPY", "SPY_close"), ("QQQ", "QQQ_close"), ("BIL", "BIL_close")]:
        if f"{asset}_fwd_path_volatility_1w" not in out.columns:
            out[f"{asset}_fwd_path_volatility_1w"] = forward_path_volatility(out[col], BASE_MOD.FWD_HOURS)
    return out


def build_work_frame(grid: pd.DataFrame, base_signals: pd.DataFrame, target_families: list[str]) -> pd.DataFrame:
    work = BASE_MOD.build_work_frame(grid, base_signals)
    if any("volatility" in target for target in target_families):
        work = attach_forward_volatility(work)
    return work


def utility_frame(frame: pd.DataFrame, train: pd.DataFrame, target_family: str) -> pd.DataFrame:
    switch = {
        asset: np.where(frame["base_asset"].astype(str).eq(asset), 0.0, SWITCH_COST)
        for asset in ["QQQ", "SPY", "BIL"]
    }
    if target_family == "best_asset_forward_return":
        return pd.DataFrame(
            {asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce") - switch[asset] for asset in ["QQQ", "SPY", "BIL"]},
            index=frame.index,
        )
    if target_family == "utility_return_minus_drawdown":
        return pd.DataFrame(
            {
                asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce")
                - 0.10 * pd.to_numeric(frame[f"{asset}_fwd_path_drawdown_1w"], errors="coerce").clip(upper=0.0).abs()
                - switch[asset]
                for asset in ["QQQ", "SPY", "BIL"]
            },
            index=frame.index,
        )
    if target_family == "utility_return_minus_volatility":
        return pd.DataFrame(
            {
                asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce")
                - 0.35 * pd.to_numeric(frame.get(f"{asset}_fwd_path_volatility_1w", 0.0), errors="coerce").fillna(0.0)
                - switch[asset]
                for asset in ["QQQ", "SPY", "BIL"]
            },
            index=frame.index,
        )
    if target_family == "utility_return_minus_turnover":
        return pd.DataFrame(
            {
                asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce") - 3.0 * switch[asset]
                for asset in ["QQQ", "SPY", "BIL"]
            },
            index=frame.index,
        )
    if target_family == "utility_moderate_penalty":
        return pd.DataFrame(
            {
                asset: pd.to_numeric(frame[f"{asset}_fwd_return_1w"], errors="coerce")
                - 0.07 * pd.to_numeric(frame[f"{asset}_fwd_path_drawdown_1w"], errors="coerce").clip(upper=0.0).abs()
                - 0.15 * pd.to_numeric(frame.get(f"{asset}_fwd_path_volatility_1w", 0.0), errors="coerce").fillna(0.0)
                - 2.0 * switch[asset]
                for asset in ["QQQ", "SPY", "BIL"]
            },
            index=frame.index,
        )
    raise ValueError(f"Unknown target family: {target_family}")


def make_labels_and_utilities(frame: pd.DataFrame, train: pd.DataFrame, target_family: str) -> tuple[pd.Series, pd.DataFrame]:
    utilities = utility_frame(frame, train, target_family)
    labels = utilities.idxmax(axis=1).map(ASSET_TO_CLASS).astype(float)
    labels = labels.mask(utilities.isna().all(axis=1), np.nan)
    return labels, utilities


def make_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    source_indices: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ws: list[float] = []
    kept: list[int] = []
    for idx in source_indices:
        if idx < window - 1:
            continue
        start = idx - window + 1
        seq = features[start : idx + 1]
        label = labels[idx]
        weight = weights[idx]
        if seq.shape[0] == window and np.isfinite(seq).all() and np.isfinite(label) and np.isfinite(weight):
            xs.append(seq.astype(np.float32))
            ys.append(int(label))
            ws.append(float(weight))
            kept.append(int(idx))
    if not xs:
        return (
            np.empty((0, window, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(ws, dtype=np.float32), np.asarray(kept, dtype=np.int64)


def fold_sample_weights(utilities: pd.DataFrame, train_idx: np.ndarray, loss_mode: str) -> tuple[pd.Series, dict[str, Any]]:
    weights = pd.Series(1.0, index=utilities.index, dtype=float)
    if loss_mode == "standard_ce":
        return weights, {"loss_mode": loss_mode, "min": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0}
    if loss_mode != "return_weighted_ce":
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    train_util = utilities.loc[train_idx].copy()
    best = train_util.max(axis=1)
    second = train_util.apply(lambda row: row.nlargest(2).iloc[-1], axis=1)
    opportunity = (best - second).clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scale = float(opportunity.quantile(0.95))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(opportunity.max()) if float(opportunity.max()) > 0 else 1.0
    train_weights = (1.0 + 4.0 * (opportunity / scale).clip(0.0, 1.0)).clip(1.0, 5.0)
    weights.loc[train_idx] = train_weights
    return weights, {
        "loss_mode": loss_mode,
        "min": float(train_weights.min()),
        "p25": float(train_weights.quantile(0.25)),
        "p50": float(train_weights.quantile(0.50)),
        "p75": float(train_weights.quantile(0.75)),
        "p95": float(train_weights.quantile(0.95)),
        "max": float(train_weights.max()),
        "scale_p95_opportunity": scale,
    }


class TinyCNN:
    def __init__(self, torch, nn, n_features: int, n_classes: int, channels: int, dropout: float):
        self.torch = torch
        self.nn = nn
        self.model = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, n_classes),
        )

    def __call__(self, x):
        return self.model(x.transpose(1, 2))

    def to(self, device):
        self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)


def train_cnn_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_test: np.ndarray,
    spec: RunSpec,
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
    if len(classes) < 2:
        proba = np.zeros((len(x_test), 3), dtype=float)
        fallback = int(classes[0]) if len(classes) else 1
        proba[:, fallback] = 1.0
        return proba, {"status": "fallback_single_class", "epochs": 0, "device": str(device)}

    val_size = max(128, int(round(len(x_train) * 0.15)))
    val_size = min(val_size, max(1, len(x_train) // 4))
    train_end = len(x_train) - val_size
    x_fit, y_fit, w_fit = x_train[:train_end], y_train[:train_end], w_train[:train_end]
    x_val, y_val = x_train[train_end:], y_train[train_end:]
    model = TinyCNN(torch, nn, x_train.shape[2], 3, spec.channels, spec.dropout).to(device)
    class_counts = np.bincount(y_fit, minlength=3).astype(float)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / np.nanmean(class_weights)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    train_ds = TensorDataset(
        torch.tensor(x_fit, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.long),
        torch.tensor(w_fit, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=min(192, max(32, len(train_ds) // 10)), shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    val_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    best_state = None
    best_val = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(spec.max_epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            per_row = nn.functional.cross_entropy(logits, yb, weight=class_weights_t, reduction="none")
            loss = (per_row * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(val_loss_fn(model(x_val_t), y_val_t).detach().cpu().item())
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
    del model, optimizer, train_ds, train_loader, test_ds, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return proba, {"status": "fit", "epochs": epochs_run, "best_val_loss": best_val, "device": str(device)}


def run_seed_walkforward(
    work: pd.DataFrame,
    spec: RunSpec,
    StandardScaler,
    metrics: dict[str, Any],
    torch_bundle: tuple[Any, Any, Any, Any, Any, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    frame = work.reset_index(drop=True).copy()
    for fold_start, fold_end in monthly_folds(frame):
        train = frame[
            (frame["timestamp"] < fold_start)
            & (frame["label_available_ts"] < fold_start)
            & frame["timestamp"].ge(TRAIN_START)
        ].copy()
        test = frame[(frame["timestamp"] >= fold_start) & (frame["timestamp"] < fold_end)].copy()
        if test.empty or len(train) < max(750, spec.window * 8):
            continue
        labels, utilities = make_labels_and_utilities(frame, train, spec.target_family)
        train_idx = train.index.to_numpy(dtype=int)
        test_idx = test.index.to_numpy(dtype=int)
        row_weights, weight_diag = fold_sample_weights(utilities, train_idx, spec.loss_mode)
        scaler = StandardScaler()
        scaler.fit(frame.loc[train_idx, FEATURES].astype(float))
        scaled = scaler.transform(frame[FEATURES].astype(float))
        x_train, y_train, w_train, kept_train_idx = make_sequences(scaled, labels.to_numpy(), row_weights.to_numpy(), train_idx, spec.window)
        x_test, y_test, _w_test, kept_test_idx = make_sequences(scaled, labels.to_numpy(), row_weights.to_numpy(), test_idx, spec.window)
        if len(x_test) == 0:
            continue
        if len(x_train) < 500 or len(np.unique(y_train)) < 2:
            prior = pd.Series(y_train).value_counts(normalize=True).to_dict() if len(y_train) else {}
            proba = np.tile([prior.get(0, 1 / 3), prior.get(1, 1 / 3), prior.get(2, 1 / 3)], (len(x_test), 1))
            proba = proba / proba.sum(axis=1, keepdims=True)
            train_meta = {"status": "fallback_constant_probability", "epochs": 0, "device": "none"}
        else:
            proba, train_meta = train_cnn_model(x_train, y_train, w_train, x_test, spec, torch_bundle)
        out = frame.loc[kept_test_idx, ["timestamp", "predicted_regime", "base_asset"]].copy()
        out["allocation_label"] = y_test
        out["prob_qqq"] = proba[:, 0]
        out["prob_spy"] = proba[:, 1]
        out["prob_bil"] = proba[:, 2]
        out["predicted_allocation_class"] = np.argmax(proba, axis=1)
        out["predicted_allocation_asset"] = out["predicted_allocation_class"].map(CLASS_TO_ASSET)
        out["allocation_confidence"] = np.max(proba, axis=1)
        out["target_spec"] = spec.target_family
        out["target_family"] = spec.target_family
        out["architecture"] = spec.model_family
        out["model_family"] = f"{spec.model_family}_sequence"
        out["window_hours"] = spec.window
        out["seed"] = spec.seed
        out["loss_mode"] = spec.loss_mode
        out["candidate_group"] = spec.group_key
        predictions.append(out)
        y_pred = out["predicted_allocation_class"].astype(int)
        try:
            ll = metrics["log_loss"](out["allocation_label"].astype(int), np.clip(proba, 1e-6, 1.0), labels=[0, 1, 2])
        except Exception:
            ll = np.nan
        fold_rows.append(
            {
                "candidate_key": spec.candidate_key,
                "candidate_group": spec.group_key,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "target_family": spec.target_family,
                "model_family": spec.model_family,
                "window_hours": spec.window,
                "seed": spec.seed,
                "loss_mode": spec.loss_mode,
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
                "strict_pit_train_max_label_available_ts": train["label_available_ts"].max() if len(train) else None,
            }
        )
        weight_rows.append(
            {
                "candidate_key": spec.candidate_key,
                "candidate_group": spec.group_key,
                "fold_start": fold_start,
                "train_rows": len(x_train),
                **weight_diag,
            }
        )
        del scaled, x_train, y_train, w_train, x_test, y_test, proba, train, test, utilities, labels
        gc.collect()
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pred_df, pd.DataFrame(fold_rows), pd.DataFrame(weight_rows)


def build_policy_signals(pred: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    if spec.policy != "direct_argmax":
        raise ValueError("This focused runner currently supports direct_argmax only.")
    out = pred.copy()
    out["policy_name"] = spec.policy
    out["policy_params"] = "{}"
    out["target_asset_signal"] = out["predicted_allocation_asset"].astype(str)
    out["candidate_base"] = spec.candidate_key
    return out


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        "timestamp",
        "candidate_base",
        "candidate_group",
        "target_spec",
        "target_family",
        "architecture",
        "model_family",
        "window_hours",
        "seed",
        "loss_mode",
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


def run_backtest(grid: pd.DataFrame, policy_signals: pd.DataFrame) -> pd.DataFrame:
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


def metric_row(spec: RunSpec, daily: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    risk_returns = daily["strategy_daily_return"].iloc[1:] if len(daily) > 1 else daily["strategy_daily_return"]
    downside = risk_returns[risk_returns < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1]) if len(daily) else INITIAL_CAPITAL
    curve_dd = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    max_dd = float(curve_dd.min()) if len(curve_dd) else np.nan
    breached_zero = bool((daily["strategy_equity"] <= 0.0).any()) if len(daily) else False
    cagr = -1.0 if breached_zero else float((final_equity / INITIAL_CAPITAL) ** (1.0 / CAGR_YEARS) - 1.0)
    ret_std = risk_returns.std(ddof=1)
    sharpe = float(risk_returns.mean() / ret_std * math.sqrt(TRADING_DAYS)) if ret_std else np.nan
    down_std = downside.std(ddof=1)
    sortino = float(risk_returns.mean() / down_std * math.sqrt(TRADING_DAYS)) if len(downside) > 1 and down_std else np.nan
    row = {
        "candidate_key": spec.candidate_key,
        "candidate_group": spec.group_key,
        "target_family": spec.target_family,
        "model_family": spec.model_family,
        "window_hours": spec.window,
        "seed": spec.seed,
        "loss_mode": spec.loss_mode,
        "policy_name": spec.policy,
        "leverage": 1.0,
        "annual_financing_rate": 0.0,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Final Equity": final_equity,
        "breached_zero": breached_zero,
        "zero_equity_breach_flag": breached_zero,
        "max_dd_floor_breach": bool(max_dd < MAX_ACCEPTED_DRAWDOWN),
        "Switch Count": int(daily["position_changed"].astype(bool).sum()),
        "turnover": float(daily["position_changed"].astype(bool).mean()),
        "daily_rows": int(len(daily)),
        "mean_allocation_confidence": float(pd.to_numeric(daily["allocation_confidence"], errors="coerce").mean()),
        "allocation_accuracy": float(daily["predicted_allocation_class"].astype(int).eq(daily["allocation_label"].astype(int)).mean()) if "allocation_label" in daily else np.nan,
        "qqq_exposure_share": float(daily["target_asset"].eq("QQQ").mean()),
        "spy_exposure_share": float(daily["target_asset"].eq("SPY").mean()),
        "bil_exposure_share": float(daily["target_asset"].eq("BIL").mean()),
    }
    curve = pd.DataFrame(
        {
            "candidate_key": spec.candidate_key,
            "candidate_group": spec.group_key,
            "report_date": daily["report_date"].astype(str).to_numpy(),
            "candidate_daily_return": daily["strategy_daily_return"].to_numpy(),
            "candidate_equity": daily["strategy_equity"].to_numpy(),
            "drawdown": curve_dd.to_numpy(),
        }
    )
    return row, curve


def yearly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if curves.empty:
        return pd.DataFrame()
    for candidate_key, group in curves.groupby("candidate_key"):
        g = group.sort_values("report_date").copy()
        years = pd.to_datetime(g["report_date"]).dt.year
        for year, y in g.groupby(years):
            rows.append(
                {
                    "candidate_key": candidate_key,
                    "candidate_group": str(y["candidate_group"].iloc[0]),
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
    if curves.empty:
        return pd.DataFrame()
    for candidate_key, group in curves.groupby("candidate_key"):
        g = group.sort_values("report_date").copy()
        months = pd.to_datetime(g["report_date"]).dt.to_period("M").astype(str)
        for month, m in g.groupby(months):
            rows.append(
                {
                    "candidate_key": candidate_key,
                    "candidate_group": str(m["candidate_group"].iloc[0]),
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
    if curves.empty:
        return pd.DataFrame()
    for candidate_key, group in curves.groupby("candidate_key"):
        g = group.sort_values("report_date").reset_index(drop=True)
        trough_idx = int(g["drawdown"].astype(float).idxmin())
        peak_idx = int(g.loc[:trough_idx, "candidate_equity"].astype(float).idxmax())
        recovery = g[(g.index > trough_idx) & (g["candidate_equity"] >= g["candidate_equity"].iloc[peak_idx])]
        rows.append(
            {
                "candidate_key": candidate_key,
                "candidate_group": str(g["candidate_group"].iloc[0]),
                "peak_date": str(g["report_date"].iloc[peak_idx]),
                "trough_date": str(g["report_date"].iloc[trough_idx]),
                "recovery_date": str(recovery["report_date"].iloc[0]) if len(recovery) else None,
                "max_drawdown": float(g["drawdown"].min()),
                "peak_equity": float(g["candidate_equity"].iloc[peak_idx]),
                "trough_equity": float(g["candidate_equity"].iloc[trough_idx]),
            }
        )
    return pd.DataFrame(rows)


def allocation_distribution(daily: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    rows = []
    total = len(daily)
    for asset, count in daily["target_asset"].astype(str).value_counts().sort_index().items():
        rows.append(
            {
                "candidate_key": spec.candidate_key,
                "candidate_group": spec.group_key,
                "target_family": spec.target_family,
                "loss_mode": spec.loss_mode,
                "model_family": spec.model_family,
                "window_hours": spec.window,
                "seed": spec.seed,
                "target_asset": asset,
                "count": int(count),
                "share": float(count / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_benchmark_gaps(seed_results: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None, bc2_row: dict[str, Any] | None) -> pd.DataFrame:
    if seed_results.empty:
        return seed_results
    out = seed_results.copy()
    out["SPY_CAGR"] = float(spy_row["CAGR"])
    out["SPY_Sharpe"] = float(spy_row["Sharpe"])
    out["SPY_Max DD"] = float(spy_row["Max DD"])
    out["beats_spy_cagr"] = out["CAGR"] > out["SPY_CAGR"]
    out["beats_spy_sharpe"] = out["Sharpe"] > out["SPY_Sharpe"]
    out["beats_spy_both"] = out["beats_spy_cagr"] & out["beats_spy_sharpe"]
    out["SPY CAGR gap"] = out["CAGR"] - out["SPY_CAGR"]
    out["SPY Sharpe gap"] = out["Sharpe"] - out["SPY_Sharpe"]
    if xgb_row is not None:
        out["XGBoost_CAGR"] = float(xgb_row["CAGR"])
        out["XGBoost_Sharpe"] = float(xgb_row["Sharpe"])
        out["production_style_XGBoost_CAGR_gap"] = out["CAGR"] - out["XGBoost_CAGR"]
        out["production_style_XGBoost_Sharpe_gap"] = out["Sharpe"] - out["XGBoost_Sharpe"]
    if bc2_row is not None:
        out["B_C2_non_pit_reference_CAGR"] = float(bc2_row["CAGR"])
        out["B_C2_non_pit_reference_Sharpe"] = float(bc2_row["Sharpe"])
        out["B_C2_gap_non_pit_CAGR"] = out["CAGR"] - out["B_C2_non_pit_reference_CAGR"]
        out["B_C2_gap_non_pit_Sharpe"] = out["Sharpe"] - out["B_C2_non_pit_reference_Sharpe"]
    return out


def group_stability(seed_results: pd.DataFrame) -> pd.DataFrame:
    if seed_results.empty:
        return pd.DataFrame()
    rows = []
    for group, g in seed_results.groupby("candidate_group"):
        max_dd_q25 = float(g["Max DD"].quantile(0.25))
        row = {
            "candidate_group": group,
            "target_family": str(g["target_family"].iloc[0]),
            "loss_mode": str(g["loss_mode"].iloc[0]),
            "model_family": str(g["model_family"].iloc[0]),
            "window_hours": int(g["window_hours"].iloc[0]),
            "policy_name": str(g["policy_name"].iloc[0]),
            "seeds": int(g["seed"].nunique()),
            "median_CAGR": float(g["CAGR"].median()),
            "median_Sharpe": float(g["Sharpe"].median()),
            "median_Sortino": float(g["Sortino"].median()),
            "median_Max_DD": float(g["Max DD"].median()),
            "worst_quartile_CAGR": float(g["CAGR"].quantile(0.25)),
            "worst_quartile_Sharpe": float(g["Sharpe"].quantile(0.25)),
            "worst_quartile_Max_DD": max_dd_q25,
            "worst_seed_CAGR": float(g["CAGR"].min()),
            "worst_seed_Sharpe": float(g["Sharpe"].min()),
            "pct_seeds_beating_spy_cagr_and_sharpe": float(g["beats_spy_both"].mean()),
            "pct_seeds_reaching_15pct_CAGR": float(g["CAGR"].ge(0.15).mean()),
            "pct_seeds_reaching_20pct_CAGR": float(g["CAGR"].ge(0.20).mean()),
            "median_QQQ_share": float(g["qqq_exposure_share"].median()),
            "median_SPY_share": float(g["spy_exposure_share"].median()),
            "median_BIL_share": float(g["bil_exposure_share"].median()),
            "worst_quartile_QQQ_share": float(g["qqq_exposure_share"].quantile(0.25)),
            "worst_quartile_SPY_share": float(g["spy_exposure_share"].quantile(0.25)),
            "worst_quartile_BIL_share": float(g["bil_exposure_share"].quantile(0.25)),
            "median_switch_count": float(g["Switch Count"].median()),
            "zero_equity_breach_count": int(g["breached_zero"].astype(bool).sum()),
            "max_dd_floor_breach_count": int(g["max_dd_floor_breach"].astype(bool).sum()),
            "SPY_CAGR": float(g["SPY_CAGR"].iloc[0]),
            "SPY_Sharpe": float(g["SPY_Sharpe"].iloc[0]),
        }
        row["minimum_research_pass"] = bool(
            row["median_CAGR"] > row["SPY_CAGR"]
            and row["median_Sharpe"] > row["SPY_Sharpe"]
            and row["pct_seeds_beating_spy_cagr_and_sharpe"] >= 0.50
            and row["worst_quartile_Max_DD"] >= MAX_ACCEPTED_DRAWDOWN
            and row["zero_equity_breach_count"] == 0
            and row["median_QQQ_share"] > 0.05
        )
        row["strong_pass"] = bool(
            row["median_CAGR"] >= 0.15
            and row["median_Sharpe"] >= 1.0
            and row["pct_seeds_beating_spy_cagr_and_sharpe"] >= 0.70
            and row["worst_seed_CAGR"] > -0.05
            and row["median_QQQ_share"] > 0.10
            and row["zero_equity_breach_count"] == 0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["minimum_research_pass", "strong_pass", "median_Sharpe", "median_CAGR"],
        ascending=[False, False, False, False],
    )


def load_seed_artifacts(output_root: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {
        "seed": [],
        "curve": [],
        "fold": [],
        "alloc": [],
        "weight": [],
    }
    for path in sorted((output_root / "seed_runs").glob("*/done.json")):
        seed_root = path.parent
        for key, filename in [
            ("seed", "seed_level_result.csv"),
            ("curve", "daily_curve.csv"),
            ("fold", "fold_diagnostics.csv"),
            ("alloc", "allocation_distribution.csv"),
            ("weight", "weight_distribution_report.csv"),
        ]:
            f = seed_root / filename
            if f.exists():
                frames[key].append(pd.read_csv(f))
    return {key: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame() for key, parts in frames.items()}


def comparison_by(summary: pd.DataFrame, column: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return (
        summary.groupby(column, as_index=False)
        .agg(
            candidate_groups=("candidate_group", "count"),
            minimum_research_passes=("minimum_research_pass", "sum"),
            strong_passes=("strong_pass", "sum"),
            best_median_CAGR=("median_CAGR", "max"),
            best_median_Sharpe=("median_Sharpe", "max"),
            best_pct_seeds_beating_spy_both=("pct_seeds_beating_spy_cagr_and_sharpe", "max"),
            best_median_QQQ_share=("median_QQQ_share", "max"),
        )
        .sort_values(["minimum_research_passes", "best_median_Sharpe", "best_median_CAGR"], ascending=[False, False, False])
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


def write_readme(output_root: Path, grid_summary: dict[str, Any], seed_results: pd.DataFrame, stability: pd.DataFrame) -> None:
    top = stability.head(10) if not stability.empty else pd.DataFrame()
    stable = stability[stability["minimum_research_pass"].eq(True)] if not stability.empty else pd.DataFrame()
    baseline = stability[stability["target_family"].eq("best_asset_forward_return")] if not stability.empty else pd.DataFrame()
    return_weighted = stability[stability["loss_mode"].eq("return_weighted_ce")] if not stability.empty else pd.DataFrame()
    utility = stability[stability["target_family"].ne("best_asset_forward_return")] if not stability.empty else pd.DataFrame()
    any_median_spy = bool(((stability["median_CAGR"] > stability["SPY_CAGR"]) & (stability["median_Sharpe"] > stability["SPY_Sharpe"])).any()) if not stability.empty else False
    any_median_15 = bool(stability["median_CAGR"].ge(0.15).any()) if not stability.empty else False
    any_median_20 = bool(stability["median_CAGR"].ge(0.20).any()) if not stability.empty else False
    lines = [
        "# MUB-27 CNN48 Stability Runner",
        "",
        "## Purpose",
        "",
        "Focused seed-stability follow-up for the CNN 48h `best_asset_forward_return` diagnostic from the partial TCN/CNN temporal allocation run.",
        "",
        "## Prior Partial-Run Context",
        "",
        "The prior compact TCN/CNN run failed from driver OOM at model 31/160. A single CNN 48h seed beat SPY, but the 5-seed group was not stable, so it was not accepted.",
        "",
        "## Grid Validation",
        "",
        md_table(pd.DataFrame([grid_summary])),
        "",
        f"- Grid validation passed: `{grid_summary.get('overall_status') == 'pass'}`.",
        "",
        "## Baseline Stability",
        "",
        md_table(baseline),
        "",
        "## Return-Weighted CE",
        "",
        f"- Return-weighted CE improved stability: `{bool(not return_weighted.empty and return_weighted['minimum_research_pass'].any())}`.",
        md_table(return_weighted),
        "",
        "## Utility-Aware Labels",
        "",
        f"- Utility-aware labels improved stability: `{bool(not utility.empty and utility['minimum_research_pass'].any())}`.",
        md_table(utility),
        "",
        "## Acceptance",
        "",
        f"- Any candidate beat SPY on median CAGR and median Sharpe: `{any_median_spy}`.",
        f"- Any candidate reached 15-20% CAGR on median seed: `{any_median_15}`.",
        f"- Any candidate reached 20% CAGR on median seed: `{any_median_20}`.",
        f"- Any candidate passed the stability gate: `{not stable.empty}`.",
        "",
        "## Top Stability Rankings",
        "",
        md_table(top),
        "",
        "## Recommended Next Action",
        "",
        "Continue only if a candidate passes median seed stability. Otherwise treat strong single seeds as diagnostics and rerun narrower labels/losses before considering portfolio-objective research.",
        "",
        "## Research Isolation",
        "",
        "This script writes only under the configured research output root and does not update live objects.",
    ]
    write_text("\n".join(lines), output_root, "README.md")


def rebuild_aggregates(output_root: Path, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None, bc2_row: dict[str, Any] | None, grid_summary: dict[str, Any]) -> dict[str, Any]:
    parts = load_seed_artifacts(output_root)
    seed_results = add_benchmark_gaps(parts["seed"], spy_row, xgb_row, bc2_row)
    stability = group_stability(seed_results)
    write_csv(seed_results, output_root, "seed_level_results.csv")
    write_csv(stability, output_root, "candidate_summary.csv")
    write_csv(stability, output_root, "candidate_rankings.csv")
    write_csv(stability, output_root, "seed_stability_summary.csv")
    write_csv(comparison_by(stability, "loss_mode"), output_root, "loss_mode_comparison.csv")
    write_csv(comparison_by(stability, "target_family"), output_root, "target_variant_comparison.csv")
    write_csv(parts["alloc"], output_root, "allocation_distribution.csv")
    write_csv(parts["fold"], output_root, "fold_diagnostics.csv")
    write_csv(parts["weight"], output_root, "weight_distribution_report.csv")
    write_csv(yearly_returns(parts["curve"]), output_root, "yearly_returns.csv")
    write_csv(monthly_returns(parts["curve"]), output_root, "monthly_returns.csv")
    write_csv(drawdown_windows(parts["curve"]), output_root, "drawdown_windows.csv")
    write_readme(output_root, grid_summary, seed_results, stability)
    return {
        "completed_seeds": int(seed_results["candidate_key"].nunique()) if not seed_results.empty else 0,
        "candidate_groups": int(stability["candidate_group"].nunique()) if not stability.empty else 0,
        "minimum_research_passes": int(stability["minimum_research_pass"].sum()) if not stability.empty else 0,
    }


def write_resume_manifest(output_root: Path, started_at: datetime, run_specs: list[RunSpec], force: bool) -> None:
    complete = []
    pending = []
    for spec in run_specs:
        item = {"candidate_key": spec.candidate_key, "done_path": done_path(output_root, spec), "completed": done_path(output_root, spec).exists()}
        (complete if item["completed"] else pending).append(item)
    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "force": force,
            "completed": complete,
            "pending": pending,
            "completed_count": len(complete),
            "pending_count": len(pending),
        },
        output_root,
        "resume_manifest.json",
    )


def write_runtime_manifest(
    output_root: Path,
    args: argparse.Namespace,
    started_at: datetime,
    run_specs: list[RunSpec],
    grid_summary: dict[str, Any],
    bil_meta: dict[str, Any],
    torch_meta: dict[str, Any],
) -> None:
    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "output_root": output_root,
            "research_leg": "MUB-27 CNN48 seed stability runner",
            "prior_partial_run": {
                "run_id": "608315821265070",
                "task_run_id": "1089008201718807",
                "failure": "OOM / exit code 137 at model 31 of 160",
                "decision": "diagnostic only; not accepted",
            },
            "features": FEATURES,
            "grid_validation": grid_summary,
            "run_specs": [spec.__dict__ | {"candidate_key": spec.candidate_key, "candidate_group": spec.group_key} for spec in run_specs],
            "torch": torch_meta,
            "python_version": sys.version,
            "platform": platform.platform(),
            "bil_price_source": bil_meta,
            "strict_pit_controls": [
                "feature scalers fit on fold training rows only",
                "training rows require label_available_ts strictly before fold_start",
                "return-weighted CE weights are computed from fold training labels only",
                "sequence windows include only rows at or before the decision timestamp",
                "monthly walk-forward folds cover 2021-01-01 through 2025-12-31",
            ],
            "production_delta_writes": False,
            "registry_writes": False,
            "job_changes": False,
            "runtime_args": vars(args),
        },
        output_root,
        "runtime_manifest.json",
    )


def write_input_hashes(output_root: Path, grid: pd.DataFrame, prior: pd.DataFrame, bil_meta: dict[str, Any], grid_summary: dict[str, Any]) -> None:
    write_json(
        {
            "faithful_legacy_grid": {
                "path": BASE_MOD.FAITHFUL_GRID_PATH,
                "sha256": file_sha256(BASE_MOD.FAITHFUL_GRID_PATH),
                "rows": len(grid),
                "expected_sha256": EXPECTED_GRID_SHA256,
            },
            "grid_validation": grid_summary,
            "strict_pit_comparison": {
                "path": BASE_MOD.STRICT_COMPARISON_PATH,
                "sha256": file_sha256(BASE_MOD.STRICT_COMPARISON_PATH),
                "rows": len(prior),
            },
            "baseline_signal_files": {name: {"path": path, "sha256": file_sha256(path)} for name, path in BASE_MOD.BASELINE_SIGNAL_FILES.items()},
            "defensive_bil_prices": bil_meta,
        },
        output_root,
        "input_hashes.json",
    )


def run_one_seed(
    output_root: Path,
    grid: pd.DataFrame,
    work: pd.DataFrame,
    spec: RunSpec,
    StandardScaler,
    metrics: dict[str, Any],
    torch_bundle: tuple[Any, Any, Any, Any, Any, dict[str, Any]],
) -> None:
    root = seed_dir(output_root, spec)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    pred, folds, weights = run_seed_walkforward(work, spec, StandardScaler, metrics, torch_bundle)
    if pred.empty:
        raise RuntimeError(f"No predictions produced for {spec.candidate_key}")
    policy_signals = build_policy_signals(pred, spec)
    daily = run_backtest(grid, policy_signals)
    if daily.empty:
        raise RuntimeError(f"No backtest rows produced for {spec.candidate_key}")
    row, curve = metric_row(spec, daily)
    alloc = allocation_distribution(daily, spec)
    pd.DataFrame([row]).to_csv(root / "seed_level_result.csv", index=False)
    folds.to_csv(root / "fold_diagnostics.csv", index=False)
    weights.to_csv(root / "weight_distribution_report.csv", index=False)
    curve.to_csv(root / "daily_curve.csv", index=False)
    alloc.to_csv(root / "allocation_distribution.csv", index=False)
    (root / "done.json").write_text(
        json.dumps(
            jsonable(
                {
                    "candidate_key": spec.candidate_key,
                    "candidate_group": spec.group_key,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seed_level_result": row,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    del pred, folds, weights, policy_signals, daily, curve, alloc
    torch = torch_bundle[0]
    if torch is not None and torch_bundle[-1].get("available") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def mode_defaults(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.mode == "baseline":
        return [args.target_family], ["standard_ce"]
    if args.mode == "loss_ablation":
        return [args.target_family], parse_csv_list(args.loss_modes, str)
    if args.mode == "utility_label_ablation":
        return parse_csv_list(args.target_families, str), parse_csv_list(args.loss_modes, str)
    raise ValueError(f"Unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable MUB-27 CNN48 seed-stability research.")
    parser.add_argument("--mode", choices=["baseline", "loss_ablation", "utility_label_ablation"], default="baseline")
    parser.add_argument("--model-family", default="cnn", choices=["cnn"], help="Focused runner supports CNN only.")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--target-family", default="best_asset_forward_return")
    parser.add_argument("--target-families", default="utility_return_minus_drawdown,utility_return_minus_volatility,utility_return_minus_turnover,utility_moderate_penalty")
    parser.add_argument("--policy", default="direct_argmax", choices=["direct_argmax"])
    parser.add_argument("--loss-modes", default="standard_ce,return_weighted_ce")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Overwrite completed seed/spec directories.")
    parser.add_argument("--allow-torch-install", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    StandardScaler, metrics = ensure_python_deps(skip_install=args.skip_install)
    torch_bundle = import_torch(allow_install=args.allow_torch_install)
    torch_meta = torch_bundle[-1]
    target_families, loss_modes = mode_defaults(args)
    seeds = parse_csv_list(args.seeds, int)
    run_specs = [
        RunSpec(
            model_family=args.model_family,
            window=args.window,
            target_family=target,
            policy=args.policy,
            loss_mode=loss_mode,
            seed=seed,
            channels=args.channels,
            dropout=args.dropout,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        for target in target_families
        for loss_mode in loss_modes
        for seed in seeds
    ]

    grid_raw = BASE_MOD.load_grid()
    bil_prices, bil_meta = BASE_MOD.load_bil_prices(grid_raw)
    grid = BASE_MOD.attach_bil(grid_raw, bil_prices)
    base_signals = BASE_MOD.load_base_signals()
    prior, spy_row, xgb_row, bc2_row = BASE_MOD.load_benchmark_rows()
    work = build_work_frame(grid, base_signals, target_families)
    validation_report, grid_summary = BASE_MOD.validate_grid_inputs(grid_raw, grid, bil_meta, base_signals, prior, work)
    BASE_MOD.GRID_VALIDATION_STATUS = str(grid_summary["overall_status"])
    write_csv(validation_report, output_root, "grid_validation_report.csv")
    write_input_hashes(output_root, grid, prior, bil_meta, grid_summary)
    write_runtime_manifest(output_root, args, started_at, run_specs, grid_summary, bil_meta, torch_meta)
    write_resume_manifest(output_root, started_at, run_specs, args.force)

    if grid_summary["overall_status"] != "pass":
        write_text("Grid validation did not pass. Do not interpret model performance as a valid result.\n", output_root, "GRID_UNVERIFIED.txt")

    total = len(run_specs)
    for idx, spec in enumerate(run_specs, start=1):
        if done_path(output_root, spec).exists() and not args.force:
            print(f"[{idx}/{total}] skip completed {spec.candidate_key}", flush=True)
            continue
        print(f"[{idx}/{total}] run {spec.candidate_key}", flush=True)
        run_one_seed(output_root, grid, work, spec, StandardScaler, metrics, torch_bundle)
        aggregate_status = rebuild_aggregates(output_root, spy_row, xgb_row, bc2_row, grid_summary)
        write_resume_manifest(output_root, started_at, run_specs, args.force)
        print(json.dumps(jsonable({"completed": spec.candidate_key, **aggregate_status}), sort_keys=True), flush=True)
        gc.collect()

    final_status = rebuild_aggregates(output_root, spy_row, xgb_row, bc2_row, grid_summary)
    write_resume_manifest(output_root, started_at, run_specs, args.force)
    print(json.dumps(jsonable({"output_root": output_root, "grid_validation_status": grid_summary["overall_status"], **final_status}), indent=2))


if __name__ == "__main__":
    main()

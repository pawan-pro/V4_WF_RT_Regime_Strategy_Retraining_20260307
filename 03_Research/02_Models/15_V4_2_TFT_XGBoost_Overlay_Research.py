# Databricks notebook source
"""MUB-27 V4.2 TFT/XGBoost overlay research.

Research-only strict-PIT hybrid overlay analysis. XGBoost remains the primary
allocator; TFT probabilities are used only as risk/conviction overlays.
Reads isolated research CSV artifacts and writes checkpoint-style artifacts
only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_xgboost_overlay_2026-05-12/`

This script does not mutate production assets, workflows, schedules, registries,
execution notebooks, dashboard notebooks, or trading logic.
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
from dateutil.relativedelta import relativedelta

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_tft_xgboost_overlay_2026-05-12"

FAITHFUL_GRID_PATH = (
    BASE
    / "v4_1_csv"
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)
REFERENCE_WALKFORWARD_PATH = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "walkforward_signals.csv"
)
MUB27_STRICT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
MUB27_COMPARISON_PATH = MUB27_STRICT_ROOT / "forensics" / "variant_comparison.csv"
MUB27_SIGNAL_DIR = MUB27_STRICT_ROOT / "signals"
TFT_BASELINE_ROOT = BASE / "v4_2_tft_window_defensive_sleeve_2026-05-06"
TFT_BASELINE_COMPARISON_PATH = TFT_BASELINE_ROOT / "forensics" / "full_variant_comparison.csv"
TFT_BASELINE_DAILY_PATH = TFT_BASELINE_ROOT / "performance" / "daily_returns_by_sleeve.csv"
DEFENSIVE_PRICE_ROOT = BASE / "v4_2_defensive_sleeve_prices_2026-05-06"
DEFENSIVE_PRICE_PATH = DEFENSIVE_PRICE_ROOT / "prices" / "defensive_etf_hourly_prices.csv"
DEFENSIVE_PRICE_COVERAGE_PATH = DEFENSIVE_PRICE_ROOT / "coverage" / "defensive_etf_coverage_summary.csv"

BASELINE_SIGNAL_FILES = {
    "A_raw_label_walkforward": MUB27_SIGNAL_DIR / "A_raw_label_walkforward_signals.csv",
    "B_reference_teacher_smoothed_non_pit_reference_only": MUB27_SIGNAL_DIR / "B_reference_teacher_smoothed_signals.csv",
    "C_strict_rolling_teacher_student": MUB27_SIGNAL_DIR / "C_strict_rolling_teacher_student_signals.csv",
    "D_pca_strict_xgboost": MUB27_SIGNAL_DIR / "D_pca_strict_xgboost_signals.csv",
    "E_spy_benchmark": MUB27_SIGNAL_DIR / "E_spy_benchmark_signals.csv",
}

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
TRADING_DAYS = 252.0
CAGR_YEARS = 5.0
SWITCH_COST = 0.0005

PRIMARY_LABEL = (0.075, 0.0050)
TARGET_CONFIGS = [
    ("primary_72h_dd075_ret050", 72, 0.075, 0.0050),
    ("control_72h_dd100_ret050", 72, 0.100, 0.0050),
    ("control_48h_dd075_ret050", 48, 0.075, 0.0050),
    ("control_96h_dd075_ret050", 96, 0.075, 0.0050),
]
STABILITY_SEEDS_10 = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
STABILITY_SEEDS_20_EXTRA = [1001, 1102, 1203, 1304, 1405, 1506, 1607, 1708, 1809, 1910]

HEADLINE_FINANCING_RATE = 0.045
HEADLINE_FINANCING_LABEL = "10Y Treasury + 50bps proxy"
FIXED_LEVERAGE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
FINANCING_GRID = [HEADLINE_FINANCING_RATE, 0.00, 0.07, 0.10]
DYNAMIC_POLICY_FINANCING_GRID = FINANCING_GRID
MAX_ACCEPTED_DRAWDOWN = -0.75

MAX_EPOCHS = 6
PATIENCE = 1
BATCH_SIZE = 512
MAX_TRAIN_SAMPLES_PER_CLASS = 3000
LEARNING_RATE = 1e-3
MODEL_DROPOUT = 0.2
WEIGHT_DECAY = 1e-4

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
SPY_VARIANT = "E_spy_benchmark"
XGBOOST_VARIANT = "D_pca_strict_xgboost"
CURRENT_TFT_BASELINE = "tft72_bil_drawdown_aware_dd_010_ret_005"


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("torch", "torch"), ("sklearn", "scikit-learn")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    import sklearn
    import torch
    from sklearn.preprocessing import StandardScaler

    return torch, sklearn, StandardScaler


torch, sklearn, StandardScaler = ensure_python_deps()
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple, Path)) else False:
        return None
    return obj


def write_json(payload: dict[str, Any], name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    use = df[cols].copy() if cols else df.copy()
    hashed = pd.util.hash_pandas_object(use, index=False).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, **kwargs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        try:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        except Exception:
            pass
    return df


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    try:
        return view.to_markdown(index=False)
    except ImportError:
        cols = [str(c) for c in view.columns]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in view.itertuples(index=False):
            vals = [str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def monthly_folds() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = EVAL_START
    while start < FINAL_FOLD_END:
        end = start + relativedelta(months=1)
        folds.append((start, end))
        start = end
    return folds


def dist(series: pd.Series | np.ndarray) -> dict[str, int]:
    if series is None or len(series) == 0:
        return {}
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


class TemporalAttentionClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x):
        projected = torch.tanh(self.input_projection(x))
        encoded, _ = self.encoder(projected)
        scores = self.attention(encoded).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(encoded * weights, dim=1)
        return self.head(self.dropout(context))


@dataclass(frozen=True)
class LabelSpec:
    drawdown_threshold: float
    return_threshold: float

    @property
    def key(self) -> str:
        dd = int(round(self.drawdown_threshold * 1000))
        ret = int(round(self.return_threshold * 10000))
        return f"dd_{dd:03d}_ret_{ret:03d}"


@dataclass(frozen=True)
class ModelSpec:
    label_spec: LabelSpec
    sequence_window_hours: int
    seed: int

    @property
    def signal_variant(self) -> str:
        return f"tft{self.sequence_window_hours}_bil_drawdown_aware_{self.label_spec.key}_seed_{self.seed}"

    @property
    def base_key(self) -> str:
        return f"tft{self.sequence_window_hours}_bil_drawdown_aware_{self.label_spec.key}"


@dataclass
class ModelResult:
    spec: ModelSpec
    signals: pd.DataFrame
    fold_manifest: pd.DataFrame


def forward_path_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    future_prices = pd.concat([price.shift(-i) for i in range(1, horizon + 1)], axis=1)
    return future_prices.min(axis=1) / price - 1.0


def load_grid() -> pd.DataFrame:
    grid = read_csv(FAITHFUL_GRID_PATH)
    cols = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing required columns: {missing}")
    optional_cols = [c for c in ["BIL_close", "bil_close", "BIL_Close", "BIL_adj_close"] if c in grid.columns]
    grid = grid[cols + optional_cols].copy()
    for col in [c for c in grid.columns if c != "timestamp"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["fwd_return_1w"] = grid["SPY_close"].shift(-FWD_HOURS) / grid["SPY_close"] - 1.0
    grid["fwd_path_drawdown_1w"] = forward_path_drawdown(grid["SPY_close"], FWD_HOURS)
    grid["label_available_ts"] = grid["timestamp"].shift(-FWD_HOURS)
    grid["feature_ok"] = grid[FEATURES + ["SPY_close", "QQQ_close"]].notna().all(axis=1)
    grid["row_pos"] = np.arange(len(grid))
    return grid


def apply_drawdown_aware_label(grid: pd.DataFrame, spec: LabelSpec) -> pd.Series:
    ret = grid["fwd_return_1w"]
    path_dd = grid["fwd_path_drawdown_1w"]
    label = pd.Series(np.nan, index=grid.index, dtype=float)
    label[(path_dd <= -spec.drawdown_threshold) | (ret <= -spec.return_threshold)] = 2
    label[(path_dd > -spec.drawdown_threshold) & (ret >= spec.return_threshold)] = 0
    label[label.isna() & ret.notna() & path_dd.notna()] = 1
    return label


def working_frame(grid: pd.DataFrame, label_spec: LabelSpec) -> pd.DataFrame:
    out = grid[
        [
            "timestamp",
            "SPY_close",
            "QQQ_close",
            "fwd_return_1w",
            "fwd_path_drawdown_1w",
            "label_available_ts",
            "feature_ok",
            *FEATURES,
        ]
    ].copy()
    out["raw_regime"] = apply_drawdown_aware_label(grid, label_spec)
    out = out[(out["timestamp"] >= TRAIN_START) & out["feature_ok"] & out["raw_regime"].notna()].copy()
    out["raw_regime"] = out["raw_regime"].astype(int)
    out = out.reset_index(drop=True)
    out["work_pos"] = np.arange(len(out))
    return out


def reference_walkforward() -> pd.DataFrame:
    ref = read_csv(REFERENCE_WALKFORWARD_PATH)
    ref = ref[(ref["timestamp"] >= EVAL_START) & (ref["timestamp"] <= EVAL_END)].copy()
    ref["predicted_regime"] = pd.to_numeric(ref["predicted_regime"], errors="coerce").astype("Int64")
    return ref.dropna(subset=["predicted_regime"]).reset_index(drop=True)


def canonical_eval_frame(work: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    ref_ts = set(ref["timestamp"])
    out = work[
        (work["timestamp"] >= EVAL_START)
        & (work["timestamp"] <= EVAL_END)
        & work["timestamp"].isin(ref_ts)
    ].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def sequence_tensor(features: np.ndarray, end_positions: np.ndarray, window: int) -> np.ndarray:
    x = np.empty((len(end_positions), window, features.shape[1]), dtype=np.float32)
    for i, pos in enumerate(end_positions):
        x[i] = features[pos - window + 1 : pos + 1]
    return x


def balanced_sample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for cls in [0, 1, 2]:
        idx = np.flatnonzero(labels == cls)
        if len(idx) > MAX_TRAIN_SAMPLES_PER_CLASS:
            idx = rng.choice(idx, size=MAX_TRAIN_SAMPLES_PER_CLASS, replace=False)
        selected.append(idx)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[TemporalAttentionClassifier, dict[str, Any]]:
    seed_everything(seed)
    model = TemporalAttentionClassifier(input_size=x_train.shape[-1], hidden_size=32, num_layers=1, dropout=MODEL_DROPOUT).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "epochs_run": len(history), "history": history}


def predict_probabilities(model: TemporalAttentionClassifier, x_test: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def run_attention_model(work: pd.DataFrame, canonical: pd.DataFrame, spec: ModelSpec, device: torch.device) -> ModelResult:
    rng = np.random.default_rng(spec.seed + spec.sequence_window_hours)
    window = spec.sequence_window_hours
    all_features = work[FEATURES].to_numpy(dtype=np.float32)
    timestamps = work["timestamp"].reset_index(drop=True)
    raw_labels = work["raw_regime"].to_numpy(dtype=int)
    label_available = pd.to_datetime(work["label_available_ts"]).reset_index(drop=True)
    canonical_pos = dict(zip(work["timestamp"], work["work_pos"]))
    signals, manifests = [], []

    for fold_num, (fold_start, fold_end) in enumerate(monthly_folds(), start=1):
        fit_mask = (timestamps < fold_start).to_numpy()
        scaler = StandardScaler()
        scaler.fit(all_features[fit_mask])
        scaled = scaler.transform(all_features).astype(np.float32)

        train_end_mask = (
            (timestamps >= TRAIN_START)
            & (timestamps < fold_start)
            & (label_available < fold_start)
            & (work["work_pos"] >= window - 1)
        ).to_numpy()
        train_end_positions = np.flatnonzero(train_end_mask)
        if len(train_end_positions) < 1000:
            continue
        split_idx = int(len(train_end_positions) * 0.8)
        train_pos = train_end_positions[:split_idx]
        val_pos = train_end_positions[split_idx:]
        if len(val_pos) < 100 or len(train_pos) < 1000:
            val_count = max(100, int(len(train_end_positions) * 0.15))
            train_pos = train_end_positions[:-val_count]
            val_pos = train_end_positions[-val_count:]

        y_train_full = raw_labels[train_pos]
        sample_idx = balanced_sample_indices(y_train_full, rng)
        train_pos_sample = train_pos[sample_idx]
        y_train = raw_labels[train_pos_sample]
        y_val = raw_labels[val_pos]
        x_train = sequence_tensor(scaled, train_pos_sample, window)
        x_val = sequence_tensor(scaled, val_pos, window)

        model, train_info = train_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            device=device,
            seed=spec.seed + window * 100 + fold_num,
        )

        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        test_positions = np.array([canonical_pos[t] for t in test_df["timestamp"] if t in canonical_pos], dtype=int)
        test_positions = test_positions[test_positions >= window - 1]
        test_df = test_df[test_df["timestamp"].isin(timestamps.iloc[test_positions].to_list())].copy()
        if test_df.empty:
            continue
        x_test = sequence_tensor(scaled, test_positions, window)
        prob = predict_probabilities(model, x_test, device)
        pred = prob.argmax(axis=1).astype(int)
        fold_signal = test_df[["timestamp", "raw_regime"]].copy()
        fold_signal["variant"] = spec.signal_variant
        fold_signal["base_variant"] = spec.base_key
        fold_signal["sequence_window_hours"] = window
        fold_signal["drawdown_threshold"] = spec.label_spec.drawdown_threshold
        fold_signal["return_threshold"] = spec.label_spec.return_threshold
        fold_signal["seed"] = spec.seed
        fold_signal["predicted_regime"] = pred
        fold_signal["regime_name"] = fold_signal["predicted_regime"].map(REGIME_NAME)
        fold_signal["prob_expansion"] = prob[:, 0]
        fold_signal["prob_slowdown"] = prob[:, 1]
        fold_signal["prob_recession"] = prob[:, 2]
        fold_signal["confidence"] = prob.max(axis=1)
        fold_signal = fold_signal[
            [
                "timestamp",
                "variant",
                "base_variant",
                "sequence_window_hours",
                "drawdown_threshold",
                "return_threshold",
                "seed",
                "raw_regime",
                "predicted_regime",
                "regime_name",
                "prob_expansion",
                "prob_slowdown",
                "prob_recession",
                "confidence",
            ]
        ]
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": spec.signal_variant,
                "base_variant": spec.base_key,
                "sequence_window_hours": window,
                "drawdown_threshold": spec.label_spec.drawdown_threshold,
                "return_threshold": spec.label_spec.return_threshold,
                "seed": spec.seed,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "label_available_rule": "train sequence label timestamp plus 168-hour horizon must be strictly before fold_start",
                "train_sequence_count_before_sampling": len(train_pos),
                "train_sequence_count_after_balancing": len(train_pos_sample),
                "validation_sequence_count": len(val_pos),
                "test_sequence_count": len(test_df),
                "train_label_distribution_before_sampling": dist(raw_labels[train_pos]),
                "train_label_distribution_after_sampling": dist(y_train),
                "validation_label_distribution": dist(y_val),
                "prediction_distribution": dist(pred),
                "best_val_loss": train_info["best_val_loss"],
                "epochs_run": train_info["epochs_run"],
                "strict_pit": True,
                "feature_list": FEATURES,
                "train_input_hash": frame_hash(
                    work.iloc[train_pos_sample][["timestamp", "raw_regime", "label_available_ts", *FEATURES]]
                ),
                "output_signal_hash": frame_hash(fold_signal),
            }
        )
        del model, x_train, x_val, x_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not signals:
        raise ValueError(f"No signals generated for {spec.signal_variant}")
    return ModelResult(spec, pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def load_bil_prices(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    for col in ["BIL_close", "bil_close", "BIL_Close", "BIL_adj_close"]:
        if col in grid.columns:
            out = grid[["timestamp", col]].rename(columns={col: "BIL_close"}).copy()
            out["BIL_close"] = pd.to_numeric(out["BIL_close"], errors="coerce")
            out = out.dropna(subset=["timestamp", "BIL_close"]).sort_values("timestamp").drop_duplicates("timestamp")
            return out, {"source": f"faithful_grid:{col}", "rows": len(out)}
    if DEFENSIVE_PRICE_PATH.exists():
        prices = read_csv(DEFENSIVE_PRICE_PATH)
        if "symbol" in prices.columns:
            prices = prices[prices["symbol"].astype(str).str.upper().eq("BIL")].copy()
        close_col = next((c for c in ["close", "adj_close", "adjusted_close", "BIL_close"] if c in prices.columns), None)
        if close_col:
            out = prices[["timestamp", close_col]].rename(columns={close_col: "BIL_close"}).copy()
            out["BIL_close"] = pd.to_numeric(out["BIL_close"], errors="coerce")
            out = out.dropna(subset=["timestamp", "BIL_close"]).sort_values("timestamp").drop_duplicates("timestamp")
            if not out.empty:
                return out, {"source": str(DEFENSIVE_PRICE_PATH), "rows": len(out)}
    raise FileNotFoundError("No usable BIL price source found in faithful grid or defensive ETF CSV")


def allocation_policy_fields(signals: pd.DataFrame, policy: str) -> pd.DataFrame:
    out = signals.copy()
    pred = out["predicted_regime"].astype(int)
    conf = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    out["allocation_policy"] = policy
    if policy == "hard_class":
        out["target_asset_signal"] = pred.map(REGIME_TO_ASSET)
        out["allocation_scale"] = 1.0
    elif policy == "confidence_gated":
        target = pred.map(REGIME_TO_ASSET)
        medium = conf.between(0.45, 0.60, inclusive="left")
        low = conf < 0.45
        target = target.mask(low & pred.ne(2), "SPY")
        scale = pd.Series(1.0, index=out.index, dtype=float)
        scale = scale.mask(medium & pred.eq(0), 0.65)
        scale = scale.mask(medium & pred.eq(1), 0.75)
        scale = scale.mask(low & pred.ne(2), 1.0)
        out["target_asset_signal"] = target
        out["allocation_scale"] = scale
    elif policy == "conservative_uncertainty_override":
        target = pred.map(REGIME_TO_ASSET)
        target = target.mask(pred.eq(0) & conf.lt(0.60), "SPY")
        target = target.mask(pred.eq(1), "SPY")
        target = target.mask(pred.eq(2), "BIL")
        out["target_asset_signal"] = target
        out["allocation_scale"] = 1.0
    else:
        raise ValueError(f"Unknown allocation policy: {policy}")
    return out


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame, bil_prices: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [
        "timestamp",
        "variant",
        "base_variant",
        "sequence_window_hours",
        "drawdown_threshold",
        "return_threshold",
        "seed",
        "predicted_regime",
        "prob_expansion",
        "prob_slowdown",
        "prob_recession",
        "confidence",
        "allocation_policy",
        "target_asset_signal",
        "allocation_scale",
    ]
    sig = signals[sig_cols].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data.merge(bil_prices[["timestamp", "BIL_close"]], on="timestamp", how="left").sort_values("timestamp")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close", "BIL_close"]] = data[["SPY_close", "QQQ_close", "BIL_close"]].ffill()
    data["price_update"] = data[["SPY_close", "QQQ_close"]].diff().abs().sum(axis=1).gt(0)
    if not data.empty:
        data.loc[data.index[0], "price_update"] = True
    data = data[data["price_update"]].dropna(subset=["SPY_close", "QQQ_close", "BIL_close"]).reset_index(drop=True)
    data["predicted_regime"] = data["predicted_regime"].astype(int)
    data["allocation_scale"] = pd.to_numeric(data["allocation_scale"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    return data


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame, policy: str, bil_prices: pd.DataFrame) -> pd.DataFrame:
    policy_signals = allocation_policy_fields(signals, policy)
    data = price_update_frame(grid, policy_signals, bil_prices)
    if len(data) < 2:
        raise ValueError(f"Not enough price-update rows for {signals['variant'].iloc[0]} / {policy}")

    equity = [INITIAL_CAPITAL]
    spy_equity = [INITIAL_CAPITAL]
    gross_returns = [0.0]
    strategy_returns = [0.0]
    switch_costs = [0.0]
    position_changed = [False]
    applied_assets = ["START"]
    active_asset = "START"
    active_scale = 0.0

    for idx in range(1, len(data)):
        signal_idx = idx - 1
        target_asset = str(data["target_asset_signal"].iloc[signal_idx])
        scale = float(data["allocation_scale"].iloc[signal_idx])
        spy_ret = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        qqq_ret = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        bil_ret = data["BIL_close"].iloc[idx] / data["BIL_close"].iloc[idx - 1] - 1.0
        if target_asset == "QQQ":
            gross_ret = scale * qqq_ret
        elif target_asset == "SPY":
            gross_ret = scale * spy_ret
        elif target_asset == "BIL":
            gross_ret = scale * bil_ret
        else:
            gross_ret = 0.0
            scale = 0.0
        changed = target_asset != active_asset or abs(scale - active_scale) > 1e-12
        cost = SWITCH_COST if changed else 0.0
        strat_ret = gross_ret - cost
        equity.append(equity[-1] * (1.0 + strat_ret))
        spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))
        gross_returns.append(gross_ret)
        strategy_returns.append(strat_ret)
        switch_costs.append(cost)
        position_changed.append(changed)
        applied_assets.append(target_asset)
        active_asset = target_asset
        active_scale = scale

    out = data.copy()
    out["allocation_policy"] = policy
    out["target_asset"] = applied_assets
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
    return daily


def metric_row_from_equity(
    candidate_id: str,
    signal_variant: str,
    base_variant: str,
    policy: str,
    leverage_policy: str,
    leverage: float | None,
    financing: float,
    daily: pd.DataFrame,
    returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    breached_zero: bool,
    applied_leverage: pd.Series | None = None,
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
    return {
        "candidate_id": candidate_id,
        "signal_variant": signal_variant,
        "base_variant": base_variant,
        "allocation_policy": policy,
        "leverage_policy": leverage_policy,
        "leverage": leverage,
        "annual_financing_rate": financing,
        "seed": int(daily["seed"].iloc[0]),
        "sequence_window_hours": int(daily["sequence_window_hours"].iloc[0]),
        "drawdown_threshold": float(daily["drawdown_threshold"].iloc[0]),
        "return_threshold": float(daily["return_threshold"].iloc[0]),
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Final Equity": final_equity,
        "min_equity": float(equity.min()) if len(equity) else np.nan,
        "breached_zero": bool(breached_zero),
        "ruin_or_liquidation_breach": bool(breached_zero or max_dd < MAX_ACCEPTED_DRAWDOWN),
        "Switch Count": int(daily["position_changed"].astype(bool).sum()),
        "daily_rows": int(len(daily)),
        "mean_confidence": float(pd.to_numeric(daily["confidence"], errors="coerce").mean()),
        "min_confidence": float(pd.to_numeric(daily["confidence"], errors="coerce").min()),
        "mean_applied_leverage": float(applied_leverage.mean()) if applied_leverage is not None and len(applied_leverage) else leverage,
    }


def fixed_leverage_curve(
    daily: pd.DataFrame,
    leverage: float,
    financing: float,
) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    raw = daily["strategy_daily_return"].astype(float).to_numpy()
    levered_raw = leverage * raw - max(leverage - 1.0, 0.0) * financing / TRADING_DAYS
    equity_values = []
    effective_returns = []
    equity = INITIAL_CAPITAL
    breached_zero = False
    for ret in levered_raw:
        if breached_zero:
            effective_returns.append(0.0)
            equity_values.append(0.0)
            continue
        factor = 1.0 + float(ret)
        if factor <= 0.0:
            breached_zero = True
            effective_returns.append(-1.0)
            equity = 0.0
            equity_values.append(0.0)
            continue
        effective_returns.append(float(ret))
        equity *= factor
        equity_values.append(equity)
    returns = pd.Series(effective_returns, dtype=float)
    equity_series = pd.Series(equity_values, dtype=float)
    roll_max = equity_series.cummax().replace(0.0, np.nan)
    drawdown = (equity_series / roll_max - 1.0).fillna(-1.0 if breached_zero else 0.0)
    return returns, equity_series, drawdown, breached_zero


def dynamic_leverage_curve(daily: pd.DataFrame, financing: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    returns = []
    equity_values = []
    drawdowns = []
    leverages = []
    breached_zero = False
    prev_dd = 0.0
    for row in daily.itertuples(index=False):
        if breached_zero:
            returns.append(0.0)
            equity_values.append(0.0)
            drawdowns.append(-1.0)
            leverages.append(0.0)
            continue
        confidence = float(getattr(row, "confidence"))
        target_asset = str(getattr(row, "target_asset"))
        base_ret = float(getattr(row, "strategy_daily_return"))
        if target_asset == "BIL" or confidence < 0.45 or prev_dd <= -0.15:
            lev = 1.0
        elif confidence >= 0.60 and prev_dd > -0.10:
            lev = 2.0
        else:
            lev = 1.5
        ret = lev * base_ret - max(lev - 1.0, 0.0) * financing / TRADING_DAYS
        factor = 1.0 + ret
        if factor <= 0.0:
            breached_zero = True
            equity = 0.0
            returns.append(-1.0)
            equity_values.append(0.0)
            drawdowns.append(-1.0)
            leverages.append(lev)
            continue
        equity *= factor
        peak = max(peak, equity)
        prev_dd = equity / peak - 1.0
        returns.append(float(ret))
        equity_values.append(float(equity))
        drawdowns.append(float(prev_dd))
        leverages.append(float(lev))
    return (
        pd.Series(returns, dtype=float),
        pd.Series(equity_values, dtype=float),
        pd.Series(drawdowns, dtype=float),
        pd.Series(leverages, dtype=float),
        breached_zero,
    )


def metrics_for_daily(daily: pd.DataFrame) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows = []
    curves = []
    signal_variant = str(daily["variant"].iloc[0])
    base_variant = str(daily["base_variant"].iloc[0])
    policy = str(daily["allocation_policy"].iloc[0])
    for leverage in FIXED_LEVERAGE_GRID:
        for financing in FINANCING_GRID:
            returns, equity, drawdown, breached_zero = fixed_leverage_curve(daily, leverage, financing)
            candidate_id = f"{base_variant}__{policy}__lev_{leverage:g}__fin_{financing:g}__seed_{int(daily['seed'].iloc[0])}"
            rows.append(
                metric_row_from_equity(
                    candidate_id,
                    signal_variant,
                    base_variant,
                    policy,
                    "fixed",
                    leverage,
                    financing,
                    daily,
                    returns,
                    equity,
                    drawdown,
                    breached_zero,
                )
            )
            curves.append(
                pd.DataFrame(
                    {
                        "candidate_id": candidate_id,
                        "report_date": daily["report_date"].astype(str).to_numpy(),
                        "base_daily_return": daily["strategy_daily_return"].to_numpy(),
                        "candidate_daily_return": returns.to_numpy(),
                        "candidate_equity": equity.to_numpy(),
                        "drawdown": drawdown.to_numpy(),
                    }
                )
            )
    for financing in DYNAMIC_POLICY_FINANCING_GRID:
        returns, equity, drawdown, applied_lev, breached_zero = dynamic_leverage_curve(daily, financing)
        candidate_id = f"{base_variant}__{policy}__dynamic_1_5_2_0__fin_{financing:g}__seed_{int(daily['seed'].iloc[0])}"
        rows.append(
            metric_row_from_equity(
                candidate_id,
                signal_variant,
                base_variant,
                policy,
                "dynamic_1_5_2_0",
                None,
                financing,
                daily,
                returns,
                equity,
                drawdown,
                breached_zero,
                applied_leverage=applied_lev,
            )
        )
        curves.append(
            pd.DataFrame(
                {
                    "candidate_id": candidate_id,
                    "report_date": daily["report_date"].astype(str).to_numpy(),
                    "base_daily_return": daily["strategy_daily_return"].to_numpy(),
                    "candidate_daily_return": returns.to_numpy(),
                    "candidate_equity": equity.to_numpy(),
                    "drawdown": drawdown.to_numpy(),
                    "applied_leverage": applied_lev.to_numpy(),
                }
            )
        )
    return rows, curves


def yearly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, candidate_group in curves.groupby("candidate_id"):
        candidate_group = candidate_group.sort_values("report_date").copy()
        years = pd.to_datetime(candidate_group["report_date"]).dt.year
        for year, group in candidate_group.groupby(years):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "year": int(year),
                    "strategy_return": float(group["candidate_equity"].iloc[-1] / group["candidate_equity"].iloc[0] - 1.0),
                    "start_equity": float(group["candidate_equity"].iloc[0]),
                    "end_equity": float(group["candidate_equity"].iloc[-1]),
                    "rows": len(group),
                }
            )
    return pd.DataFrame(rows)


def monthly_returns(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate_id, candidate_group in curves.groupby("candidate_id"):
        candidate_group = candidate_group.sort_values("report_date").copy()
        months = pd.to_datetime(candidate_group["report_date"]).dt.to_period("M").astype(str)
        for month, group in candidate_group.groupby(months):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "month": month,
                    "strategy_return": float(group["candidate_equity"].iloc[-1] / group["candidate_equity"].iloc[0] - 1.0),
                    "start_equity": float(group["candidate_equity"].iloc[0]),
                    "end_equity": float(group["candidate_equity"].iloc[-1]),
                    "rows": len(group),
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
        peak_slice = g.loc[:trough_idx]
        peak_idx = int(peak_slice["candidate_equity"].astype(float).idxmax()) if not peak_slice.empty else trough_idx
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


def regime_distribution(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["variant", "base_variant", "sequence_window_hours", "drawdown_threshold", "return_threshold", "seed"]
    for keys, group in signals.groupby(group_cols):
        vc = group["predicted_regime"].astype(int).value_counts().sort_index()
        total = int(vc.sum())
        base = dict(zip(group_cols, keys))
        for regime in [0, 1, 2]:
            count = int(vc.get(regime, 0))
            rows.append({**base, "regime": regime, "regime_name": REGIME_NAME[regime], "count": count, "share": count / total if total else np.nan})
    return pd.DataFrame(rows)


def load_benchmark_rows() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None]:
    prior = read_csv(MUB27_COMPARISON_PATH)
    spy = prior[prior["variant"].eq(SPY_VARIANT)]
    if spy.empty:
        raise ValueError(f"Missing {SPY_VARIANT} in {MUB27_COMPARISON_PATH}")
    spy_row = spy.iloc[0].to_dict()
    xgb = prior[prior["variant"].eq(XGBOOST_VARIANT)]
    xgb_row = xgb.iloc[0].to_dict() if not xgb.empty else None
    return prior, spy_row, xgb_row


def add_benchmark_gaps(summary: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None) -> pd.DataFrame:
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
    return out


def candidate_eligible(row: pd.Series) -> bool:
    return bool(
        row["beats_spy_cagr"]
        and row["beats_spy_sharpe"]
        and row["Max DD"] >= MAX_ACCEPTED_DRAWDOWN
        and not row["breached_zero"]
        and not row["ruin_or_liquidation_breach"]
    )


def stability_summary(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "base_variant",
        "allocation_policy",
        "leverage_policy",
        "leverage",
        "annual_financing_rate",
        "sequence_window_hours",
        "drawdown_threshold",
        "return_threshold",
    ]
    rows = []
    for keys, group in summary.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        eligible = group.apply(candidate_eligible, axis=1)
        rows.append(
            {
                **base,
                "seed_count": int(group["seed"].nunique()),
                "eligible_seed_count": int(eligible.sum()),
                "eligible_seed_rate": float(eligible.mean()) if len(eligible) else 0.0,
                "mean_CAGR": float(group["CAGR"].mean()),
                "min_CAGR": float(group["CAGR"].min()),
                "std_CAGR": float(group["CAGR"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_Sharpe": float(group["Sharpe"].mean()),
                "min_Sharpe": float(group["Sharpe"].min()),
                "std_Sharpe": float(group["Sharpe"].std(ddof=1)) if len(group) > 1 else 0.0,
                "worst_Max DD": float(group["Max DD"].min()),
                "any_breached_zero": bool(group["breached_zero"].any()),
                "seed_stability_pass": bool(
                    group["seed"].nunique() >= len(STABILITY_SEEDS_10)
                    and eligible.mean() >= 0.80
                    and group["Max DD"].min() >= MAX_ACCEPTED_DRAWDOWN
                    and not group["breached_zero"].any()
                ),
            }
        )
    return pd.DataFrame(rows)


def ranking_table(summary: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    join_cols = [
        "base_variant",
        "allocation_policy",
        "leverage_policy",
        "leverage",
        "annual_financing_rate",
        "sequence_window_hours",
        "drawdown_threshold",
        "return_threshold",
    ]
    eligible = summary[summary.apply(candidate_eligible, axis=1)].copy()
    if eligible.empty:
        return eligible
    merged = eligible.merge(stability[join_cols + ["seed_stability_pass", "eligible_seed_rate", "std_Sharpe"]], on=join_cols, how="left")
    merged = merged[merged["seed_stability_pass"].fillna(False)].copy()
    if merged.empty:
        return merged
    return merged.sort_values(
        ["Sharpe", "CAGR", "Max DD", "Switch Count", "std_Sharpe"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)


def label_threshold_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["drawdown_threshold", "return_threshold"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            eligible_count=("beats_spy_cagr", lambda s: int((summary.loc[s.index].apply(candidate_eligible, axis=1)).sum())),
        )
        .sort_values(["best_Sharpe", "best_CAGR"], ascending=False)
    )


def window_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("sequence_window_hours", as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            eligible_count=("beats_spy_cagr", lambda s: int((summary.loc[s.index].apply(candidate_eligible, axis=1)).sum())),
        )
        .sort_values(["best_Sharpe", "best_CAGR"], ascending=False)
    )


def confidence_policy_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("allocation_policy", as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            eligible_count=("beats_spy_cagr", lambda s: int((summary.loc[s.index].apply(candidate_eligible, axis=1)).sum())),
        )
        .sort_values(["best_Sharpe", "best_CAGR"], ascending=False)
    )


def leverage_policy_results(summary: pd.DataFrame) -> pd.DataFrame:
    fixed = summary[summary["leverage_policy"].eq("fixed")].copy()
    return (
        fixed.groupby(["leverage", "annual_financing_rate"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            breached_zero_count=("breached_zero", "sum"),
            eligible_count=("beats_spy_cagr", lambda s: int((fixed.loc[s.index].apply(candidate_eligible, axis=1)).sum())),
        )
        .sort_values(["leverage", "annual_financing_rate"])
    )


def dynamic_leverage_results(summary: pd.DataFrame) -> pd.DataFrame:
    dyn = summary[summary["leverage_policy"].eq("dynamic_1_5_2_0")].copy()
    return (
        dyn.groupby("annual_financing_rate", as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            best_CAGR=("CAGR", "max"),
            best_Sharpe=("Sharpe", "max"),
            worst_Max_DD=("Max DD", "min"),
            breached_zero_count=("breached_zero", "sum"),
            mean_applied_leverage=("mean_applied_leverage", "mean"),
            eligible_count=("beats_spy_cagr", lambda s: int((dyn.loc[s.index].apply(candidate_eligible, axis=1)).sum())),
        )
        .sort_values("annual_financing_rate")
    )


def candidate_vs_spy(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "seed",
        "allocation_policy",
        "leverage_policy",
        "leverage",
        "annual_financing_rate",
        "CAGR",
        "SPY_CAGR",
        "gap_vs_spy_cagr",
        "Sharpe",
        "SPY_Sharpe",
        "gap_vs_spy_sharpe",
        "Max DD",
        "SPY_Max DD",
        "breached_zero",
        "ruin_or_liquidation_breach",
    ]
    return summary[cols].sort_values(["gap_vs_spy_sharpe", "gap_vs_spy_cagr"], ascending=False)


def candidate_vs_xgboost(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "candidate_id",
        "seed",
        "allocation_policy",
        "leverage_policy",
        "leverage",
        "annual_financing_rate",
        "CAGR",
        "XGBoost_CAGR",
        "gap_vs_xgboost_cagr",
        "Sharpe",
        "XGBoost_Sharpe",
        "gap_vs_xgboost_sharpe",
        "Max DD",
        "breached_zero",
    ]
    return summary[cols].sort_values(["gap_vs_xgboost_sharpe", "gap_vs_xgboost_cagr"], ascending=False)


OVERLAY_SEEDS = STABILITY_SEEDS_10
TFT_OVERLAY_SPEC = ModelSpec(LabelSpec(0.075, 0.0050), 72, 42)
XGB_SIGNAL_NAME = "D_pca_strict_xgboost"
OVERLAY_POLICY_SPECS = [
    *[("A_recession_veto", {"recession_threshold": t, "safe_asset": safe}) for t in [0.40, 0.50, 0.60] for safe in ["SPY", "BIL"]],
    *[("B_confidence_scaling", {"confidence_threshold": t}) for t in [0.45, 0.55, 0.65]],
    *[("C_disagreement_guard", {"recession_threshold": t, "safe_asset": safe}) for t in [0.50, 0.60] for safe in ["SPY", "BIL"]],
]
LEVERAGE_OVERLAY_SPECS = [
    ("D_leverage_overlay", {"risk_on_confidence": t, "max_leverage": lev})
    for t in [0.55, 0.65]
    for lev in [1.25, 1.5, 2.0]
]


def seed_list(use_20: bool) -> list[int]:
    return STABILITY_SEEDS_10 + (STABILITY_SEEDS_20_EXTRA if use_20 else [])


def load_xgboost_signals(work: pd.DataFrame) -> pd.DataFrame:
    path = BASELINE_SIGNAL_FILES[XGB_SIGNAL_NAME]
    xgb = read_csv(path)
    if "predicted_regime" not in xgb.columns:
        raise ValueError(f"Missing predicted_regime in {path}")
    xgb = xgb[(xgb["timestamp"] >= EVAL_START) & (xgb["timestamp"] <= EVAL_END)].copy()
    xgb["xgb_predicted_regime"] = pd.to_numeric(xgb["predicted_regime"], errors="coerce").astype("Int64")
    xgb = xgb.dropna(subset=["timestamp", "xgb_predicted_regime"]).copy()
    xgb["xgb_predicted_regime"] = xgb["xgb_predicted_regime"].astype(int)
    return xgb[["timestamp", "xgb_predicted_regime"]].merge(work[["timestamp", "raw_regime"]], on="timestamp", how="left")


def tag_tft_signal(result: ModelResult, source_name: str, source_type: str) -> pd.DataFrame:
    out = result.signals.copy()
    out["tft_source"] = source_name
    out["tft_source_type"] = source_type
    return out


def build_ensemble_signal(seed_signals: list[pd.DataFrame], name: str, vote: bool) -> pd.DataFrame:
    merged = []
    for idx, sig in enumerate(seed_signals):
        part = sig[["timestamp", "raw_regime", "prob_expansion", "prob_slowdown", "prob_recession", "predicted_regime"]].copy()
        part = part.rename(
            columns={
                "prob_expansion": f"prob_expansion_{idx}",
                "prob_slowdown": f"prob_slowdown_{idx}",
                "prob_recession": f"prob_recession_{idx}",
                "predicted_regime": f"predicted_regime_{idx}",
            }
        )
        merged.append(part)
    base = merged[0]
    for part in merged[1:]:
        base = base.merge(part, on=["timestamp", "raw_regime"], how="inner")
    out = base[["timestamp", "raw_regime"]].copy()
    out["prob_expansion"] = base[[c for c in base.columns if c.startswith("prob_expansion_")]].mean(axis=1)
    out["prob_slowdown"] = base[[c for c in base.columns if c.startswith("prob_slowdown_")]].mean(axis=1)
    out["prob_recession"] = base[[c for c in base.columns if c.startswith("prob_recession_")]].mean(axis=1)
    if vote:
        out["predicted_regime"] = base[[c for c in base.columns if c.startswith("predicted_regime_")]].mode(axis=1)[0].astype(int)
    else:
        out["predicted_regime"] = out[["prob_expansion", "prob_slowdown", "prob_recession"]].to_numpy().argmax(axis=1)
    out["confidence"] = out[["prob_expansion", "prob_slowdown", "prob_recession"]].max(axis=1)
    out["variant"] = name
    out["base_variant"] = name
    out["sequence_window_hours"] = 72
    out["drawdown_threshold"] = 0.075
    out["return_threshold"] = 0.005
    out["seed"] = 0
    out["regime_name"] = out["predicted_regime"].map(REGIME_NAME)
    out["tft_source"] = name
    out["tft_source_type"] = "ensemble_majority_vote" if vote else "ensemble_avg_prob"
    return out


def xgb_asset(regime: int) -> str:
    return REGIME_TO_ASSET.get(int(regime), "BIL")


def overlay_targets(joined: pd.DataFrame, policy_name: str, params: dict[str, Any]) -> pd.DataFrame:
    out = joined.copy()
    xgb_regime = out["xgb_predicted_regime"].astype(int)
    tft_regime = out["predicted_regime"].astype(int)
    rec_prob = pd.to_numeric(out["prob_recession"], errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    target = xgb_regime.map(lambda r: xgb_asset(int(r)))
    scale = pd.Series(1.0, index=out.index, dtype=float)
    if policy_name == "A_recession_veto":
        mask = xgb_regime.eq(0) & rec_prob.gt(float(params["recession_threshold"]))
        target = target.mask(mask, str(params["safe_asset"]))
    elif policy_name == "B_confidence_scaling":
        low = confidence.lt(float(params["confidence_threshold"]))
        target = target.mask(low & xgb_regime.eq(0), "SPY")
        target = target.mask(low & xgb_regime.eq(1), "BIL")
    elif policy_name == "C_disagreement_guard":
        high_risk = tft_regime.eq(2) | rec_prob.gt(float(params["recession_threshold"]))
        target = target.mask(xgb_regime.eq(0) & high_risk, str(params["safe_asset"]))
        target = target.mask(xgb_regime.eq(1) & high_risk, "BIL")
    elif policy_name == "D_leverage_overlay":
        high_conf = confidence.ge(float(params["risk_on_confidence"]))
        agree_risk_on = xgb_regime.eq(0) & tft_regime.eq(0) & high_conf
        target = target.mask((xgb_regime.eq(0) | xgb_regime.eq(1)) & rec_prob.gt(0.50), "BIL")
        out["overlay_daily_leverage_signal"] = np.where(agree_risk_on, float(params["max_leverage"]), 1.0)
    else:
        raise ValueError(f"Unknown overlay policy {policy_name}")
    out["overlay_policy"] = policy_name
    out["overlay_params"] = json.dumps(params, sort_keys=True)
    out["target_asset_signal"] = target
    out["allocation_scale"] = scale
    if "overlay_daily_leverage_signal" not in out.columns:
        out["overlay_daily_leverage_signal"] = 1.0
    return out


def overlay_daily(grid: pd.DataFrame, overlay_signals: pd.DataFrame, bil_prices: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp", "variant", "base_variant", "tft_source", "tft_source_type", "seed", "xgb_predicted_regime",
        "predicted_regime", "prob_expansion", "prob_slowdown", "prob_recession", "confidence", "overlay_policy",
        "overlay_params", "target_asset_signal", "allocation_scale", "overlay_daily_leverage_signal",
    ]
    sig = overlay_signals[cols].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data.merge(bil_prices[["timestamp", "BIL_close"]], on="timestamp", how="left").sort_values("timestamp")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close", "BIL_close"]] = data[["SPY_close", "QQQ_close", "BIL_close"]].ffill()
    data["price_update"] = data[["SPY_close", "QQQ_close"]].diff().abs().sum(axis=1).gt(0)
    if not data.empty:
        data.loc[data.index[0], "price_update"] = True
    data = data[data["price_update"]].dropna(subset=["SPY_close", "QQQ_close", "BIL_close"]).reset_index(drop=True)
    equity = [INITIAL_CAPITAL]
    spy_equity = [INITIAL_CAPITAL]
    returns = [0.0]
    changed = [False]
    assets = ["START"]
    active_asset = "START"
    active_scale = 0.0
    for idx in range(1, len(data)):
        signal_idx = idx - 1
        target_asset = str(data["target_asset_signal"].iloc[signal_idx])
        scale = float(data["allocation_scale"].iloc[signal_idx])
        spy_ret = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        qqq_ret = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        bil_ret = data["BIL_close"].iloc[idx] / data["BIL_close"].iloc[idx - 1] - 1.0
        raw_ret = qqq_ret if target_asset == "QQQ" else spy_ret if target_asset == "SPY" else bil_ret if target_asset == "BIL" else 0.0
        is_changed = target_asset != active_asset or abs(scale - active_scale) > 1e-12
        ret = scale * raw_ret - (SWITCH_COST if is_changed else 0.0)
        equity.append(equity[-1] * (1.0 + ret))
        spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))
        returns.append(ret)
        changed.append(is_changed)
        assets.append(target_asset)
        active_asset = target_asset
        active_scale = scale
    out = data.copy()
    out["target_asset"] = assets
    out["strategy_return"] = returns
    out["position_changed"] = changed
    out["strategy_equity"] = equity
    out["spy_equity"] = spy_equity
    out["spy_daily_return"] = out["spy_equity"].pct_change().fillna(0.0)
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    return daily


def variable_leverage_curve(daily: pd.DataFrame, financing: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    returns, equities, drawdowns, leverages = [], [], [], []
    breached_zero = False
    for row in daily.itertuples(index=False):
        lev = float(getattr(row, "overlay_daily_leverage_signal"))
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
        returns.append(ret)
        equities.append(equity)
        drawdowns.append(-1.0 if breached_zero else equity / peak - 1.0)
        leverages.append(lev)
    return pd.Series(returns), pd.Series(equities), pd.Series(drawdowns), pd.Series(leverages), breached_zero


def overlay_metric_rows(daily: pd.DataFrame, candidate_base: str, leverage_mode: str, leverage: float | None, financing: float) -> tuple[dict[str, Any], pd.DataFrame]:
    if leverage_mode == "fixed":
        returns, equity, drawdown, breached = fixed_leverage_curve(daily, float(leverage), financing)
        applied = pd.Series([float(leverage)] * len(daily))
    else:
        returns, equity, drawdown, applied, breached = variable_leverage_curve(daily, financing)
    candidate_id = f"{candidate_base}__{leverage_mode}_{'dynamic' if leverage is None else str(leverage).replace('.', '_')}__fin_{financing:g}"
    row = metric_row_from_equity(
        candidate_id,
        str(daily["variant"].iloc[0]),
        str(daily["base_variant"].iloc[0]),
        str(daily["overlay_policy"].iloc[0]),
        leverage_mode,
        leverage,
        financing,
        daily,
        returns,
        equity,
        drawdown,
        breached,
        applied_leverage=applied,
    )
    row.update(
        {
            "overlay_policy": str(daily["overlay_policy"].iloc[0]),
            "overlay_params": str(daily["overlay_params"].iloc[0]),
            "tft_source": str(daily["tft_source"].iloc[0]),
            "tft_source_type": str(daily["tft_source_type"].iloc[0]),
            "xgboost_signal_source": XGB_SIGNAL_NAME,
            "headline_financing": bool(abs(financing - HEADLINE_FINANCING_RATE) < 1e-12),
        }
    )
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


def standalone_tft_reference() -> dict[str, float]:
    path = BASE / "v4_2_tft_targeted_stability_2026-05-12" / "targeted_candidate_summary.csv"
    if not path.exists():
        return {}
    try:
        ref = read_csv(path)
        ref = ref[ref["annual_financing_rate"].eq(HEADLINE_FINANCING_RATE)].copy()
        if ref.empty:
            return {}
        best = ref.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]
        return {"standalone_tft_CAGR": float(best["CAGR"]), "standalone_tft_Sharpe": float(best["Sharpe"]), "standalone_tft_Max DD": float(best["Max DD"])}
    except Exception:
        return {}


def add_overlay_benchmark_gaps(summary: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None, tft_ref: dict[str, float]) -> pd.DataFrame:
    out = add_benchmark_gaps(summary, spy_row, xgb_row)
    for key, value in tft_ref.items():
        out[key] = value
    if tft_ref:
        out["gap_vs_standalone_tft_cagr"] = out["CAGR"] - tft_ref["standalone_tft_CAGR"]
        out["gap_vs_standalone_tft_sharpe"] = out["Sharpe"] - tft_ref["standalone_tft_Sharpe"]
    return out


def overlay_passes(row: pd.Series, xgb_available: bool) -> bool:
    ok = bool(row["beats_spy_cagr"] and row["beats_spy_sharpe"] and row["Max DD"] >= MAX_ACCEPTED_DRAWDOWN and not row["breached_zero"])
    if xgb_available:
        ok = ok and row["CAGR"] > row["XGBoost_CAGR"] and row["Sharpe"] > row["XGBoost_Sharpe"]
    return ok and bool(row["headline_financing"])


def rank_overlays(summary: pd.DataFrame, xgb_available: bool) -> pd.DataFrame:
    out = summary.copy()
    out["passes_acceptance"] = out.apply(lambda r: overlay_passes(r, xgb_available), axis=1)
    out["financing_sensitivity_penalty"] = out.groupby(["overlay_policy", "overlay_params", "tft_source", "leverage_policy", "leverage"], dropna=False)["CAGR"].transform("std").fillna(0.0)
    return out.sort_values(
        ["passes_acceptance", "Sharpe", "CAGR", "Max DD", "Switch Count", "financing_sensitivity_penalty"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)


def policy_results(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["overlay_policy", "headline_financing"], as_index=False)
        .agg(candidates=("candidate_id", "count"), best_CAGR=("CAGR", "max"), best_Sharpe=("Sharpe", "max"), worst_Max_DD=("Max DD", "min"), zero_breaches=("breached_zero", "sum"))
        .sort_values(["headline_financing", "best_Sharpe", "best_CAGR"], ascending=[False, False, False])
    )


def regime_distribution_overlay(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in signals.groupby(["overlay_policy", "overlay_params", "tft_source", "target_asset_signal"]):
        rows.append(
            {
                "overlay_policy": keys[0],
                "overlay_params": keys[1],
                "tft_source": keys[2],
                "target_asset": keys[3],
                "count": len(group),
                "share": len(group) / len(signals) if len(signals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_runtime_manifest(args: argparse.Namespace, started_at: datetime) -> None:
    write_json(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started_at.isoformat(),
            "output_root": OUTPUT_ROOT,
            "features": FEATURES,
            "xgboost_signal_source": XGB_SIGNAL_NAME,
            "tft_overlay_family": "72h dd_075_ret_050 baseline TFT probabilities",
            "seeds_requested": seed_list(args.use_20_seeds),
            "headline_financing_rate": HEADLINE_FINANCING_RATE,
            "headline_financing_label": HEADLINE_FINANCING_LABEL,
            "financing_sensitivity_grid": FINANCING_GRID,
            "overlay_policies": [p for p, _ in OVERLAY_POLICY_SPECS],
            "leverage_overlay_specs": LEVERAGE_OVERLAY_SPECS,
            "accepted_max_drawdown_floor": MAX_ACCEPTED_DRAWDOWN,
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


def write_input_hashes(grid: pd.DataFrame, ref: pd.DataFrame, bil_meta: dict[str, Any], prior: pd.DataFrame, xgb: pd.DataFrame) -> None:
    write_json(
        {
            "faithful_legacy_grid": {"path": FAITHFUL_GRID_PATH, "sha256": file_sha256(FAITHFUL_GRID_PATH), "rows": len(grid)},
            "reference_walkforward_signals": {"path": REFERENCE_WALKFORWARD_PATH, "sha256": file_sha256(REFERENCE_WALKFORWARD_PATH), "rows": len(ref)},
            "mub27_strict_pit_comparison": {"path": MUB27_COMPARISON_PATH, "sha256": file_sha256(MUB27_COMPARISON_PATH), "rows": len(prior)},
            "xgboost_signal_file": {"path": BASELINE_SIGNAL_FILES[XGB_SIGNAL_NAME], "sha256": file_sha256(BASELINE_SIGNAL_FILES[XGB_SIGNAL_NAME]), "rows": len(xgb)},
            "baseline_signal_files": {name: {"path": path, "sha256": file_sha256(path)} for name, path in BASELINE_SIGNAL_FILES.items()},
            "defensive_bil_prices": bil_meta,
            "financing_assumption": {"headline_label": HEADLINE_FINANCING_LABEL, "headline_rate": HEADLINE_FINANCING_RATE, "note": "4.5% is used as deck-consistent 10Y+50bps proxy unless a future dated 10Y artifact is supplied."},
        },
        "input_hashes.json",
    )


def write_readme(summary: pd.DataFrame, rankings: pd.DataFrame, spy_row: dict[str, Any], xgb_row: dict[str, Any] | None) -> None:
    top = rankings.head(10) if not rankings.empty else pd.DataFrame()
    passes = rankings[rankings["passes_acceptance"].eq(True)] if "passes_acceptance" in rankings.columns else pd.DataFrame()
    lines = [
        "# MUB-27 V4.2 TFT/XGBoost Overlay Research",
        "",
        "Research-only hybrid overlay test. XGBoost remains primary; TFT probabilities only veto, scale, or condition leverage.",
        "",
        f"- Headline financing: {HEADLINE_FINANCING_LABEL} = {HEADLINE_FINANCING_RATE:.3%}.",
        f"- SPY CAGR: {float(spy_row['CAGR']):.6f}",
        f"- SPY Sharpe: {float(spy_row['Sharpe']):.6f}",
        f"- XGBoost comparison available: {xgb_row is not None}",
        "",
        "## Passing Candidates",
        "",
        md_table(passes.head(10)),
        "",
        "## Top Overlay Candidates",
        "",
        md_table(top),
        "",
        "## Production Safety",
        "",
        "No production V4 tables, jobs, Saxo/token objects, RT/dashboard notebooks, model registry entries, schedules, or trading logic were changed.",
    ]
    write_text("\n".join(lines), "README.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MUB-27 V4.2 TFT/XGBoost overlay research.")
    parser.add_argument("--use-20-seeds", action="store_true", help="Generate 20 TFT seeds instead of 10.")
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional debug cap for TFT seed trainings.")
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    ensure_dirs()
    write_runtime_manifest(args, started_at)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = load_grid()
    work = working_frame(grid, LabelSpec(0.075, 0.0050))
    ref = reference_walkforward()
    canonical = canonical_eval_frame(work, ref)
    bil_prices, bil_meta = load_bil_prices(grid)
    prior, spy_row, xgb_row = load_benchmark_rows()
    xgb = load_xgboost_signals(work)
    write_input_hashes(grid, ref, bil_meta, prior, xgb)

    seeds = seed_list(args.use_20_seeds)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]

    tft_signals: list[pd.DataFrame] = []
    fold_manifests: list[pd.DataFrame] = []
    for idx, seed in enumerate(seeds, start=1):
        spec = ModelSpec(LabelSpec(0.075, 0.0050), 72, seed)
        print(f"[{idx}/{len(seeds)}] training overlay TFT seed {seed}", flush=True)
        result = run_attention_model(work, canonical, spec, device)
        sig = tag_tft_signal(result, f"tft72_dd075_ret050_seed_{seed}", "single_seed")
        tft_signals.append(sig)
        fold_manifests.append(result.fold_manifest.assign(tft_source=f"tft72_dd075_ret050_seed_{seed}", tft_source_type="single_seed"))
        write_csv(pd.concat(fold_manifests, ignore_index=True), "fold_manifest.csv")

    if len(tft_signals) >= 5:
        tft_signals.append(build_ensemble_signal(tft_signals[:5], "ensemble_5_seed_avg_prob", vote=False))
        tft_signals.append(build_ensemble_signal(tft_signals[:5], "ensemble_5_seed_majority_vote", vote=True))
    if len(tft_signals) >= 10:
        seed_only = [s for s in tft_signals if str(s["tft_source_type"].iloc[0]) == "single_seed"]
        tft_signals.append(build_ensemble_signal(seed_only[:10], "ensemble_10_seed_avg_prob", vote=False))
        tft_signals.append(build_ensemble_signal(seed_only[:10], "ensemble_10_seed_majority_vote", vote=True))

    summary_rows: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    overlay_signal_frames: list[pd.DataFrame] = []
    tft_ref = standalone_tft_reference()
    for sig in tft_signals:
        joined = xgb.merge(sig, on=["timestamp", "raw_regime"], how="inner")
        joined["variant"] = str(sig["variant"].iloc[0])
        joined["base_variant"] = str(sig["base_variant"].iloc[0])
        joined["tft_source"] = str(sig["tft_source"].iloc[0])
        joined["tft_source_type"] = str(sig["tft_source_type"].iloc[0])
        policies = OVERLAY_POLICY_SPECS + LEVERAGE_OVERLAY_SPECS
        for policy_name, params in policies:
            overlay = overlay_targets(joined, policy_name, params)
            overlay_signal_frames.append(overlay[["timestamp", "overlay_policy", "overlay_params", "tft_source", "target_asset_signal"]].copy())
            daily = overlay_daily(grid, overlay, bil_prices)
            candidate_base = f"{policy_name}__{str(sig['tft_source'].iloc[0])}__{hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}"
            if policy_name == "D_leverage_overlay":
                for financing in FINANCING_GRID:
                    row, curve = overlay_metric_rows(daily, candidate_base, "overlay_dynamic", None, financing)
                    summary_rows.append(row)
                    curve_frames.append(curve)
            else:
                for leverage in [1.0, 1.25, 1.5]:
                    for financing in FINANCING_GRID:
                        row, curve = overlay_metric_rows(daily, candidate_base, "fixed", leverage, financing)
                        summary_rows.append(row)
                        curve_frames.append(curve)

    summary = add_overlay_benchmark_gaps(pd.DataFrame(summary_rows), spy_row, xgb_row, tft_ref)
    rankings = rank_overlays(summary, xgb_row is not None)
    curves = pd.concat(curve_frames, ignore_index=True)
    write_csv(summary, "overlay_candidate_summary.csv")
    write_csv(rankings, "overlay_rankings.csv")
    write_csv(candidate_vs_spy(summary).rename(columns={"candidate_id": "overlay_candidate_id"}), "overlay_vs_spy.csv")
    if xgb_row is not None:
        write_csv(candidate_vs_xgboost(summary).rename(columns={"candidate_id": "overlay_candidate_id"}), "overlay_vs_xgboost.csv")
    else:
        write_csv(pd.DataFrame(), "overlay_vs_xgboost.csv")
    if tft_ref:
        write_csv(summary[[c for c in summary.columns if c in {"candidate_id", "CAGR", "Sharpe", "Max DD", "standalone_tft_CAGR", "standalone_tft_Sharpe", "standalone_tft_Max DD", "gap_vs_standalone_tft_cagr", "gap_vs_standalone_tft_sharpe"}]], "overlay_vs_standalone_tft.csv")
    else:
        write_csv(pd.DataFrame(), "overlay_vs_standalone_tft.csv")
    write_csv(policy_results(summary), "policy_results.csv")
    write_csv(summary[summary["leverage_policy"].eq("overlay_dynamic")], "leverage_overlay_results.csv")
    write_csv(
        summary.groupby("annual_financing_rate", as_index=False).agg(best_CAGR=("CAGR", "max"), best_Sharpe=("Sharpe", "max"), worst_Max_DD=("Max DD", "min"), passing_candidates=("headline_financing", lambda s: int(summary.loc[s.index].apply(lambda r: overlay_passes(r, xgb_row is not None), axis=1).sum()))),
        "financing_sensitivity.csv",
    )
    write_csv(yearly_returns(curves), "yearly_returns.csv")
    write_csv(monthly_returns(curves), "monthly_returns.csv")
    write_csv(drawdown_windows(curves), "drawdown_windows.csv")
    write_csv(regime_distribution_overlay(pd.concat(overlay_signal_frames, ignore_index=True)), "regime_distribution.csv")
    write_readme(summary, rankings, spy_row, xgb_row)
    print(json.dumps(jsonable({"output_root": OUTPUT_ROOT, "top_10": rankings.head(10).to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()

# Databricks notebook source
"""MUB-27 V4.2 TFT-only improvement grid.

Research-only strict-PIT simulation grid for TFT-style temporal attention
candidates. Reads isolated research CSV artifacts and writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_only_improvement_grid_2026-05-11/`

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
OUTPUT_ROOT = BASE / "v4_2_tft_only_improvement_grid_2026-05-11"

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

DRAW_DOWN_THRESHOLDS = [0.075, 0.100, 0.125, 0.150]
RETURN_THRESHOLDS = [0.0025, 0.0050, 0.0075, 0.0100]
SEQUENCE_WINDOWS = [48, 72, 96]
STAGE_A_SEEDS = [42]
STABILITY_SEEDS = [42, 101, 202, 303, 404]
STAGE_B_TOP_N = 10

FIXED_LEVERAGE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
FINANCING_GRID = [0.00, 0.03, 0.045, 0.05, 0.07, 0.10]
DYNAMIC_POLICY_FINANCING_GRID = FINANCING_GRID
MAX_ACCEPTED_DRAWDOWN = -0.75

MAX_EPOCHS = 6
PATIENCE = 1
BATCH_SIZE = 512
MAX_TRAIN_SAMPLES_PER_CLASS = 3000
LEARNING_RATE = 1e-3
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
    return view.to_markdown(index=False)


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
    model = TemporalAttentionClassifier(input_size=x_train.shape[-1], hidden_size=32, num_layers=1, dropout=0.2).to(device)
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
                    group["seed"].nunique() >= len(STABILITY_SEEDS)
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


def specs_for_stage_a() -> list[ModelSpec]:
    return [
        ModelSpec(LabelSpec(dd, ret), window, seed)
        for dd in DRAW_DOWN_THRESHOLDS
        for ret in RETURN_THRESHOLDS
        for window in SEQUENCE_WINDOWS
        for seed in STAGE_A_SEEDS
    ]


def specs_for_stage_b(stage_a_summary: pd.DataFrame) -> list[ModelSpec]:
    prelim = stage_a_summary[stage_a_summary.apply(candidate_eligible, axis=1)].copy()
    if prelim.empty:
        prelim = stage_a_summary.copy()
    base_rank = (
        prelim.sort_values(["Sharpe", "CAGR", "Max DD", "Switch Count"], ascending=[False, False, False, True])
        .drop_duplicates(["base_variant", "sequence_window_hours", "drawdown_threshold", "return_threshold"])
        .head(STAGE_B_TOP_N)
    )
    specs = []
    for row in base_rank.itertuples(index=False):
        label = LabelSpec(float(row.drawdown_threshold), float(row.return_threshold))
        for seed in STABILITY_SEEDS:
            specs.append(ModelSpec(label, int(row.sequence_window_hours), seed))
    dedup = {}
    for spec in specs:
        dedup[(spec.base_key, spec.seed)] = spec
    return list(dedup.values())


def evaluate_specs(
    specs: list[ModelSpec],
    grid: pd.DataFrame,
    ref: pd.DataFrame,
    bil_prices: pd.DataFrame,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_frames = []
    manifest_frames = []
    daily_frames = []
    metric_rows = []
    curve_frames = []
    work_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx}/{len(specs)}] training {spec.signal_variant}", flush=True)
        if spec.label_spec.key not in work_cache:
            work = working_frame(grid, spec.label_spec)
            canonical = canonical_eval_frame(work, ref)
            work_cache[spec.label_spec.key] = (work, canonical)
        else:
            work, canonical = work_cache[spec.label_spec.key]
        model_result = run_attention_model(work, canonical, spec, device)
        signal_frames.append(model_result.signals)
        manifest_frames.append(model_result.fold_manifest)
        for policy in ["hard_class", "confidence_gated", "conservative_uncertainty_override"]:
            daily = run_backtest(grid, model_result.signals, policy, bil_prices)
            daily_frames.append(daily)
            rows, curves = metrics_for_daily(daily)
            metric_rows.extend(rows)
            curve_frames.extend(curves)

    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    manifests = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame()
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    summary = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    return signals, manifests, daily, summary, curves


def load_existing_tft_baseline() -> pd.DataFrame:
    if not TFT_BASELINE_COMPARISON_PATH.exists():
        return pd.DataFrame()
    df = read_csv(TFT_BASELINE_COMPARISON_PATH)
    mask = df["variant"].astype(str).str.contains("TFT_attention_raw_72h__BIL", na=False)
    return df[mask].copy()


def write_readme(summary: pd.DataFrame, rankings: pd.DataFrame, spy_row: dict[str, Any], xgb_available: bool) -> None:
    best = rankings.iloc[0].to_dict() if not rankings.empty else {}
    lines = [
        "# MUB-27 V4.2 TFT-Only Improvement Grid",
        "",
        "Research-only strict-PIT TFT improvement simulation grid.",
        "",
        "## Scope",
        "",
        "- Original 14 V4 features only; no calendar/time features are added.",
        "- Drawdown-aware labels are swept over configured drawdown and return thresholds.",
        "- Sequence windows are swept over 48h, 72h, and 96h.",
        "- Stage A evaluates the full grid with one seed; Stage B runs the top configurations over five seeds.",
        "- B/C2 teacher-smoothed artifacts, where referenced, are non-PIT references only.",
        "",
        "## Acceptance Rule",
        "",
        "- CAGR must exceed SPY CAGR.",
        "- Sharpe must exceed SPY Sharpe.",
        "- Max DD must be no worse than -75%.",
        "- Zero-equity or liquidation breach fails the candidate.",
        "- Seed stability requires five seeds, at least 80% eligible seeds, no zero breach, and Max DD no worse than -75%.",
        "",
        "## Benchmarks",
        "",
        f"- SPY CAGR: {float(spy_row['CAGR']):.6f}",
        f"- SPY Sharpe: {float(spy_row['Sharpe']):.6f}",
        f"- XGBoost comparison available: {xgb_available}",
        "",
        "## Top Ranked Candidates",
        "",
        md_table(rankings.head(10)),
        "",
        "## Best Candidate",
        "",
        json.dumps(jsonable(best), indent=2, sort_keys=True),
        "",
        "## Production Safety",
        "",
        "No production V4 tables, jobs, execution credentials, RT/dashboard notebooks, registry entries, schedules, or trading logic were changed.",
        "",
        "## Output Counts",
        "",
        f"- candidate_summary rows: {len(summary)}",
        f"- candidate_rankings rows: {len(rankings)}",
    ]
    write_text("\n".join(lines), "README.md")


def write_outputs(
    stage_a: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    stage_b: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    grid: pd.DataFrame,
    ref: pd.DataFrame,
    bil_meta: dict[str, Any],
    prior: pd.DataFrame,
    spy_row: dict[str, Any],
    xgb_row: dict[str, Any] | None,
    runtime_args: argparse.Namespace,
    started_at: datetime,
) -> None:
    stage_a_signals, stage_a_manifests, stage_a_daily, stage_a_summary, stage_a_curves = stage_a
    stage_b_signals, stage_b_manifests, stage_b_daily, stage_b_summary, stage_b_curves = stage_b

    summary = pd.concat([stage_a_summary.assign(stage="A"), stage_b_summary.assign(stage="B")], ignore_index=True)
    summary = add_benchmark_gaps(summary, spy_row, xgb_row)
    curves = pd.concat([stage_a_curves.assign(stage="A"), stage_b_curves.assign(stage="B")], ignore_index=True)
    signals = pd.concat([stage_a_signals.assign(stage="A"), stage_b_signals.assign(stage="B")], ignore_index=True)
    manifests = pd.concat([stage_a_manifests.assign(stage="A"), stage_b_manifests.assign(stage="B")], ignore_index=True)
    daily = pd.concat([stage_a_daily.assign(stage="A"), stage_b_daily.assign(stage="B")], ignore_index=True)

    stability = stability_summary(summary[summary["stage"].eq("B")]) if not stage_b_summary.empty else pd.DataFrame()
    rankings = ranking_table(summary, stability) if not stability.empty else pd.DataFrame()

    write_csv(summary.sort_values(["stage", "Sharpe", "CAGR"], ascending=[True, False, False]), "candidate_summary.csv")
    write_csv(rankings, "candidate_rankings.csv")
    write_csv(stability, "seed_stability_summary.csv")
    write_csv(label_threshold_results(summary), "label_threshold_results.csv")
    write_csv(window_results(summary), "window_results.csv")
    write_csv(confidence_policy_results(summary), "confidence_policy_results.csv")
    write_csv(leverage_policy_results(summary), "leverage_policy_results.csv")
    write_csv(dynamic_leverage_results(summary), "dynamic_leverage_results.csv")
    write_csv(yearly_returns(curves), "yearly_returns.csv")
    write_csv(monthly_returns(curves), "monthly_returns.csv")
    write_csv(drawdown_windows(curves), "drawdown_windows.csv")
    write_csv(regime_distribution(signals), "regime_distribution.csv")
    write_csv(candidate_vs_spy(summary), "candidate_vs_spy.csv")
    if xgb_row is not None:
        write_csv(candidate_vs_xgboost(summary), "candidate_vs_xgboost.csv")
    write_csv(signals, "tft_signals_with_probabilities.csv")
    write_csv(manifests, "fold_manifest.csv")
    write_csv(daily, "daily_unlevered_policy_returns.csv")
    baseline = load_existing_tft_baseline()
    if not baseline.empty:
        write_csv(baseline, "existing_tft_baseline_reference.csv")

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
        "mub27_strict_pit_comparison": {
            "path": MUB27_COMPARISON_PATH,
            "sha256": file_sha256(MUB27_COMPARISON_PATH),
            "rows": len(prior),
        },
        "baseline_signal_files": {
            name: {"path": path, "sha256": file_sha256(path)} for name, path in BASELINE_SIGNAL_FILES.items()
        },
        "defensive_bil_prices": bil_meta,
        "canonical_eval_reference_timestamp_hash": frame_hash(ref[["timestamp"]]),
    }
    write_json(input_hashes, "input_hashes.json")

    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_root": OUTPUT_ROOT,
        "features": FEATURES,
        "drawdown_thresholds": DRAW_DOWN_THRESHOLDS,
        "return_thresholds": RETURN_THRESHOLDS,
        "sequence_windows": SEQUENCE_WINDOWS,
        "stage_a_seeds": STAGE_A_SEEDS,
        "stage_b_stability_seeds": STABILITY_SEEDS,
        "stage_b_top_n": STAGE_B_TOP_N,
        "fixed_leverage_grid": FIXED_LEVERAGE_GRID,
        "financing_grid": FINANCING_GRID,
        "accepted_max_drawdown_floor": MAX_ACCEPTED_DRAWDOWN,
        "strict_pit_controls": [
            "scaler fit uses rows strictly before fold_start",
            "sequence inputs contain observations only through sequence end timestamp",
            "train labels require 168-hour forward label availability strictly before fold_start",
            "validation split and early stopping use train-period data only",
            "test fold is never used for scaling, training, thresholding, or early stopping",
        ],
        "label_definition": {
            "forward_rows": FWD_HOURS,
            "recession": "future path drawdown <= -drawdown_threshold or 1w SPY forward return <= -return_threshold",
            "expansion": "future path drawdown > -drawdown_threshold and 1w SPY forward return >= return_threshold",
            "slowdown": "remaining non-null forward outcomes",
        },
        "allocation_policies": ["hard_class", "confidence_gated", "conservative_uncertainty_override"],
        "leverage_formula": "levered_daily_return = leverage * strategy_daily_return - max(leverage - 1, 0) * annual_financing_rate / 252",
        "runtime_args": vars(runtime_args),
        "production_delta_writes": False,
        "registry_writes": False,
        "workflow_or_schedule_changes": False,
        "execution_notebook_changes": False,
        "dashboard_notebook_changes": False,
        "trading_logic_changes": False,
    }
    write_json(runtime_manifest, "runtime_manifest.json")
    write_readme(summary, rankings, spy_row, xgb_row is not None)

    print(json.dumps(jsonable({"output_root": OUTPUT_ROOT, "top_10": rankings.head(10).to_dict("records")}), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MUB-27 V4.2 TFT-only improvement grid.")
    parser.add_argument("--stage-a-only", action="store_true", help="Run only the one-seed full grid and write Stage A outputs.")
    parser.add_argument("--stage-b-top-n", type=int, default=STAGE_B_TOP_N, help="Top Stage A base configs to carry into five-seed Stage B.")
    parser.add_argument("--max-stage-a-specs", type=int, default=None, help="Optional debug cap for Stage A model specs.")
    parser.add_argument("--max-stage-b-specs", type=int, default=None, help="Optional debug cap for Stage B model specs.")
    return parser.parse_args()


def main() -> None:
    started_at = datetime.now(timezone.utc)
    args = parse_args()
    global STAGE_B_TOP_N
    STAGE_B_TOP_N = args.stage_b_top_n

    seed_everything(STAGE_A_SEEDS[0])
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = load_grid()
    ref = reference_walkforward()
    bil_prices, bil_meta = load_bil_prices(grid)
    prior, spy_row, xgb_row = load_benchmark_rows()

    stage_a_specs = specs_for_stage_a()
    if args.max_stage_a_specs is not None:
        stage_a_specs = stage_a_specs[: args.max_stage_a_specs]
    stage_a = evaluate_specs(stage_a_specs, grid, ref, bil_prices, device)

    if args.stage_a_only:
        empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        write_outputs(stage_a, empty, grid, ref, bil_meta, prior, spy_row, xgb_row, args, started_at)
        return

    stage_a_summary_with_gaps = add_benchmark_gaps(stage_a[3], spy_row, xgb_row)
    stage_b_specs = specs_for_stage_b(stage_a_summary_with_gaps)
    if args.max_stage_b_specs is not None:
        stage_b_specs = stage_b_specs[: args.max_stage_b_specs]
    stage_b = evaluate_specs(stage_b_specs, grid, ref, bil_prices, device)
    write_outputs(stage_a, stage_b, grid, ref, bil_meta, prior, spy_row, xgb_row, args, started_at)


if __name__ == "__main__":
    main()

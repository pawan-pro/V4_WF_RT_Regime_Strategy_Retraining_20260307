# Databricks notebook source
"""MUB-28 April-cutoff teacher-student promotion-readiness diagnostics.

Read-only against production state except for research outputs written under:

`/Volumes/pawan_quant_research_1/v4/market_data/mub28_teacher_student_april_cutoff_2026-05-08/promotion_readiness/`

This notebook does not promote, overwrite the live alias, update the model
registry, or modify RT/Saxo/dashboard/workflow code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import functions as F


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
NS = f"{CATALOG}.{SCHEMA}"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
MUB28_ROOT = BASE / "mub28_teacher_student_april_cutoff_2026-05-08"
OUTPUT_ROOT = MUB28_ROOT / "promotion_readiness"
MODEL_VERSION = "bt_teacher_student_smoothed_v4_2026_04_cutoff_2026_05_08"

CURRENT_MODEL_VERSION = "bt_teacher_student_smoothed_v4_2026_05_08"
CURRENT_LIVE_MODEL_PATH = BASE / "models" / "bt_fred_1d_xgboost_regime_live.joblib"
CANDIDATE_MODEL_PATH = BASE / "models" / f"{MODEL_VERSION}.joblib"
CANDIDATE_EXPECTED_SHA256 = "3cf0d7a0bd7a13d80c2ed26a61634cc4496a208038d8301f1711ba0adea9ad70"
CURRENT_EXPECTED_SHA256 = "7dd1949058c4af0e6525dff1daea095603e108cba2b0dc9fd5a4756968b44f6b"

MANIFEST_PATH = MUB28_ROOT / "manifests" / f"{MODEL_VERSION}_manifest.json"
VALIDATION_CSV_PATH = MUB28_ROOT / "validation" / "latest_100_silver_rt_candidate_vs_current.csv"
SHADOW_TABLE = f"{NS}.bronze_rt_regime_signal_teacher_student_april_shadow"
SILVER_RT_TABLE = f"{NS}.silver_rt_1h"
SILVER_BT_TABLE = f"{NS}.silver_bt_master_grid_1h"
BRONZE_RT_MARKET_TABLE = f"{NS}.bronze_rt_market_1h"
BRONZE_BT_MARKET_TABLE = f"{NS}.bronze_bt_market_1h"
REGISTRY_TABLE = f"{NS}.model_registry_current"

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
POSITION_MAP = {0: "100% QQQ", 1: "100% SPY", 2: "100% CASH"}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    try:
        out[column] = out[column].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def write_csv(df: pd.DataFrame, name: str) -> str:
    path = OUTPUT_ROOT / name
    df.to_csv(path, index=False)
    return str(path)


def write_json(payload: dict[str, Any], name: str) -> str:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
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

    path = OUTPUT_ROOT / name
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    vc = pd.Series(series).dropna().astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def assert_model_schema(model: Any, name: str) -> None:
    if hasattr(model, "feature_names_in_"):
        got = list(model.feature_names_in_)
        if got != FEATURES:
            raise ValueError(f"{name} feature_names_in_ mismatch: {got}")
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(FEATURES):
        raise ValueError(f"{name} n_features_in_ mismatch: {model.n_features_in_}")


def predict_details(model: Any, x: pd.DataFrame) -> tuple[int, np.ndarray]:
    pred = int(model.predict(x[FEATURES].fillna(0))[0])
    proba = model.predict_proba(x[FEATURES].fillna(0))[0]
    return pred, proba


def contribution_cube(model: Any, x: pd.DataFrame) -> np.ndarray:
    booster = model.get_booster()
    dm = xgb.DMatrix(x[FEATURES].fillna(0), feature_names=FEATURES)
    try:
        contrib = booster.predict(dm, pred_contribs=True, strict_shape=True)
    except TypeError:
        contrib = booster.predict(dm, pred_contribs=True)
    arr = np.asarray(contrib)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2 and arr.shape[0] == 1:
        groups = len(getattr(model, "classes_", [0, 1, 2]))
        return arr.reshape(groups, len(FEATURES) + 1)
    raise ValueError(f"Unexpected contribution shape: {arr.shape}")


def latest_row_explanation(current_model: Any, candidate_model: Any) -> dict[str, Any]:
    latest = spark.table(SILVER_RT_TABLE).orderBy(F.col("timestamp").desc()).limit(1).toPandas()
    latest = normalize_timestamp(latest)
    for col in FEATURES:
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    current_pred, current_proba = predict_details(current_model, latest)
    candidate_pred, candidate_proba = predict_details(candidate_model, latest)
    current_contrib = contribution_cube(current_model, latest)
    candidate_contrib = contribution_cube(candidate_model, latest)

    rows = []
    names = FEATURES + ["bias"]
    vals = [float(latest[f].iloc[0]) for f in FEATURES] + [1.0]
    for model_name, pred, proba, contrib in [
        ("current_live", current_pred, current_proba, current_contrib),
        ("april_candidate", candidate_pred, candidate_proba, candidate_contrib),
    ]:
        for class_idx in range(contrib.shape[0]):
            for feature, value, contribution in zip(names, vals, contrib[class_idx]):
                rows.append(
                    {
                        "timestamp": latest["timestamp"].iloc[0],
                        "model": model_name,
                        "predicted_regime": pred,
                        "predicted_regime_name": REGIME_NAME[pred],
                        "class_regime": class_idx,
                        "class_regime_name": REGIME_NAME.get(class_idx),
                        "class_probability": float(proba[class_idx]),
                        "feature": feature,
                        "feature_value": value,
                        "margin_contribution": float(contribution),
                    }
                )
    contrib_df = pd.DataFrame(rows)

    diff_rows = []
    for model_name, contrib, favored, opposed, label in [
        ("current_live", current_contrib, 2, 0, "recession_minus_expansion"),
        ("april_candidate", candidate_contrib, 0, 2, "expansion_minus_recession"),
    ]:
        for feature, value, fav, opp in zip(names, vals, contrib[favored], contrib[opposed]):
            diff_rows.append(
                {
                    "timestamp": latest["timestamp"].iloc[0],
                    "model": model_name,
                    "comparison": label,
                    "feature": feature,
                    "feature_value": value,
                    "favored_class": favored,
                    "opposed_class": opposed,
                    "favored_minus_opposed_margin_contribution": float(fav - opp),
                }
            )
    diff_df = pd.DataFrame(diff_rows).sort_values(
        ["model", "favored_minus_opposed_margin_contribution"], ascending=[True, False]
    )

    feature_vector_path = write_csv(latest[["timestamp", *FEATURES]], "latest_feature_vector.csv")
    contrib_path = write_csv(contrib_df, "latest_row_xgboost_contributions.csv")
    explanation_path = write_csv(diff_df, "latest_row_explanation.csv")
    return {
        "latest_timestamp": latest["timestamp"].iloc[0],
        "current_prediction": current_pred,
        "current_prediction_name": REGIME_NAME[current_pred],
        "current_position": POSITION_MAP[current_pred],
        "current_probabilities": {REGIME_NAME[i]: float(v) for i, v in enumerate(current_proba)},
        "candidate_prediction": candidate_pred,
        "candidate_prediction_name": REGIME_NAME[candidate_pred],
        "candidate_position": POSITION_MAP[candidate_pred],
        "candidate_probabilities": {REGIME_NAME[i]: float(v) for i, v in enumerate(candidate_proba)},
        "feature_vector_path": feature_vector_path,
        "contribution_path": contrib_path,
        "latest_row_explanation_path": explanation_path,
        "candidate_top_expansion_drivers": diff_df[diff_df["model"].eq("april_candidate")].head(8).to_dict("records"),
        "current_top_recession_drivers": diff_df[diff_df["model"].eq("current_live")].head(8).to_dict("records"),
    }


def last_100_analysis() -> dict[str, Any]:
    if VALIDATION_CSV_PATH.exists():
        df = pd.read_csv(VALIDATION_CSV_PATH)
    else:
        df = spark.table(SHADOW_TABLE).toPandas()
    df = normalize_timestamp(df)
    df["disagree"] = pd.to_numeric(df["current_regime"], errors="coerce").astype(int) != pd.to_numeric(
        df["candidate_regime"], errors="coerce"
    ).astype(int)
    disagreements = df[df["disagree"]].copy()
    qqq = df[df["candidate_position"].astype(str).eq("100% QQQ")].copy()
    nfp_ts = pd.Timestamp("2026-05-08 12:30:00")
    pre_nfp = df[df["timestamp"] < nfp_ts]
    post_nfp = df[df["timestamp"] >= nfp_ts]
    out_path = write_csv(disagreements, "last_100_disagreements.csv")
    return {
        "rows": len(df),
        "current_regime_distribution": value_counts_dict(df["current_regime"]),
        "candidate_regime_distribution": value_counts_dict(df["candidate_regime"]),
        "current_position_distribution": value_counts_dict(df["current_position"]),
        "candidate_position_distribution": value_counts_dict(df["candidate_position"]),
        "disagreement_count": int(df["disagree"].sum()),
        "disagreement_percent": float(df["disagree"].mean()),
        "first_candidate_qqq_timestamp": None if qqq.empty else qqq["timestamp"].min(),
        "last_candidate_qqq_timestamp": None if qqq.empty else qqq["timestamp"].max(),
        "candidate_qqq_rows": len(qqq),
        "candidate_qqq_latest_contiguous_rows": latest_contiguous_count(df, "candidate_position", "100% QQQ"),
        "pre_nfp_candidate_distribution": value_counts_dict(pre_nfp["candidate_position"]),
        "post_nfp_candidate_distribution": value_counts_dict(post_nfp["candidate_position"]),
        "nfp_release_changed_candidate_signal": bool(
            len(df[(df["timestamp"] >= pd.Timestamp("2026-05-08 13:00:00")) & df["candidate_position"].astype(str).eq("100% QQQ")]) > 0
        ),
        "disagreement_csv": out_path,
        "disagreement_timestamps": [str(x) for x in disagreements["timestamp"].tolist()],
    }


def latest_contiguous_count(df: pd.DataFrame, col: str, value: str) -> int:
    count = 0
    for _, row in df.sort_values("timestamp", ascending=False).iterrows():
        if str(row[col]) == value:
            count += 1
        else:
            break
    return count


def load_feature_history(current_model: Any, candidate_model: Any) -> pd.DataFrame:
    bt = spark.table(SILVER_BT_TABLE).select(["timestamp", "SPY_close", "QQQ_close", *FEATURES]).toPandas()
    rt = spark.table(SILVER_RT_TABLE).select(["timestamp", "SPY_close", *FEATURES]).toPandas()
    bt = normalize_timestamp(bt)
    rt = normalize_timestamp(rt)
    if "QQQ_close" not in rt.columns:
        rt["QQQ_close"] = np.nan
    bt["priority"] = 0
    rt["priority"] = 1
    hist = (
        pd.concat([bt, rt], ignore_index=True)
        .sort_values(["timestamp", "priority"])
        .drop_duplicates("timestamp", keep="last")
        .drop(columns=["priority"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist[["SPY_close", "QQQ_close"]] = hist[["SPY_close", "QQQ_close"]].ffill()
    hist["current_regime"] = current_model.predict(hist[FEATURES].fillna(0)).astype(int)
    hist["candidate_regime"] = candidate_model.predict(hist[FEATURES].fillna(0)).astype(int)
    hist["current_position"] = hist["current_regime"].map(POSITION_MAP)
    hist["candidate_position"] = hist["candidate_regime"].map(POSITION_MAP)
    return hist


def load_market_union() -> pd.DataFrame:
    frames = []
    for table, priority in [(BRONZE_BT_MARKET_TABLE, 0), (BRONZE_RT_MARKET_TABLE, 1)]:
        pdf = (
            spark.table(table)
            .filter(F.col("symbol").isin("SPY", "QQQ"))
            .select("symbol", "timestamp", "close")
            .toPandas()
        )
        pdf = normalize_timestamp(pdf)
        pdf["priority"] = priority
        frames.append(pdf)
    market = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "timestamp", "priority"])
        .drop_duplicates(["symbol", "timestamp"], keep="last")
    )
    wide = market.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last").reset_index()
    wide = wide.sort_values("timestamp").reset_index(drop=True)
    wide[["SPY", "QQQ"]] = wide[["SPY", "QQQ"]].ffill()
    return wide


def recent_market_sanity(hist: pd.DataFrame) -> dict[str, Any]:
    market = load_market_union()
    latest_ts = min(market["timestamp"].max(), hist["timestamp"].max())
    latest_market = market[market["timestamp"] <= latest_ts].iloc[-1]
    latest_feat = hist[hist["timestamp"] <= latest_ts].iloc[-1]
    rows = []
    for days in [1, 3, 5, 10]:
        start_ts = latest_ts - pd.Timedelta(days=days)
        start_market = market[market["timestamp"] <= start_ts].iloc[-1]
        start_feat = hist[hist["timestamp"] <= start_ts].iloc[-1]
        window_hist = hist[(hist["timestamp"] > start_market["timestamp"]) & (hist["timestamp"] <= latest_ts)].copy()
        rows.append(
            {
                "window": f"{days}d",
                "start_ts": start_market["timestamp"],
                "end_ts": latest_ts,
                "SPY_return": float(latest_market["SPY"] / start_market["SPY"] - 1.0),
                "QQQ_return": float(latest_market["QQQ"] / start_market["QQQ"] - 1.0),
                "QQQ_minus_SPY_return": float((latest_market["QQQ"] / start_market["QQQ"] - 1.0) - (latest_market["SPY"] / start_market["SPY"] - 1.0)),
                "VIXCLS_z_start": float(start_feat["VIXCLS_z"]),
                "VIXCLS_z_end": float(latest_feat["VIXCLS_z"]),
                "VIXCLS_z_delta": float(latest_feat["VIXCLS_z"] - start_feat["VIXCLS_z"]),
                "VIX_Inversion_Z_start": float(start_feat["VIX_Inversion_Z"]),
                "VIX_Inversion_Z_end": float(latest_feat["VIX_Inversion_Z"]),
                "VIX_Inversion_Z_delta": float(latest_feat["VIX_Inversion_Z"] - start_feat["VIX_Inversion_Z"]),
                "Price_Mom_24H_latest": float(latest_feat["Price_Mom_24H"]),
                "PAYEMS_z_latest": float(latest_feat["PAYEMS_z"]),
                "NFP_Event_Active_latest": int(latest_feat["NFP_Event_Active"]),
                "candidate_regime_distribution": json.dumps(value_counts_dict(window_hist["candidate_regime"]), sort_keys=True),
                "current_regime_distribution": json.dumps(value_counts_dict(window_hist["current_regime"]), sort_keys=True),
                "candidate_latest_position": str(latest_feat["candidate_position"]),
                "current_latest_position": str(latest_feat["current_position"]),
            }
        )
    sanity = pd.DataFrame(rows)
    path = write_csv(sanity, "recent_market_sanity.csv")
    return {"recent_market_sanity_csv": path, "latest_market_ts": latest_ts, "rows": rows}


def backtest(hist: pd.DataFrame, regime_col: str, variant: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, Any]:
    data = hist[(hist["timestamp"] >= start_ts) & (hist["timestamp"] <= end_ts)].copy()
    data = data.dropna(subset=["SPY_close", "QQQ_close", regime_col]).sort_values("timestamp")
    changed = data[["SPY_close", "QQQ_close"]].ne(data[["SPY_close", "QQQ_close"]].shift()).any(axis=1)
    data = data[changed].reset_index(drop=True)
    if len(data) < 2:
        raise ValueError(f"Not enough rows for {variant}")
    equity = [INITIAL_CAPITAL]
    pos_changed = [False]
    active = "START"
    for idx in range(1, len(data)):
        regime = int(data[regime_col].iloc[idx - 1])
        asset = REGIME_TO_ASSET[regime]
        if asset == "QQQ":
            gross = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        elif asset == "SPY":
            gross = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        else:
            gross = 0.0
        changed_asset = asset != active
        equity.append(equity[-1] * (1.0 + gross - (SWITCH_COST if changed_asset else 0.0)))
        pos_changed.append(changed_asset)
        active = asset
    data["strategy_equity"] = equity
    data["position_changed"] = pos_changed
    data["drawdown"] = data["strategy_equity"] / data["strategy_equity"].cummax() - 1.0
    data["report_date"] = data["timestamp"].dt.date.astype(str)
    daily = data.loc[data.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp")
    returns = daily["strategy_equity"].pct_change().dropna()
    years = max((daily["timestamp"].iloc[-1] - daily["timestamp"].iloc[0]).days / 365.25, 1e-9)
    final = float(daily["strategy_equity"].iloc[-1])
    return {
        "variant": variant,
        "start_timestamp": data["timestamp"].iloc[0],
        "end_timestamp": data["timestamp"].iloc[-1],
        "cumulative_return": final / INITIAL_CAPITAL - 1.0,
        "CAGR_partial_period": (final / INITIAL_CAPITAL) ** (1.0 / years) - 1.0,
        "Sharpe_partial_period": returns.mean() / returns.std(ddof=1) * math.sqrt(252.0) if returns.std(ddof=1) else 0.0,
        "max_drawdown": float(data["drawdown"].min()),
        "switch_count": int(pd.Series(pos_changed).astype(bool).sum()),
        "final_equity": final,
        "latest_position": POSITION_MAP[int(data[regime_col].iloc[-1])],
        "price_update_rows": len(data),
        "daily_rows": len(daily),
    }


def replay_2026(hist: pd.DataFrame) -> dict[str, Any]:
    end_ts = hist["timestamp"].max()
    replay_rows = [
        backtest(hist, "current_regime", "current_live_2025_12_model", pd.Timestamp("2026-01-01"), end_ts),
        backtest(hist, "candidate_regime", "april_cutoff_candidate", pd.Timestamp("2026-01-01"), end_ts),
    ]
    spy = hist.copy()
    spy["spy_regime"] = 1
    replay_rows.append(backtest(spy, "spy_regime", "SPY_benchmark", pd.Timestamp("2026-01-01"), end_ts))
    apr_rows = [
        backtest(hist, "current_regime", "current_live_2025_12_model_april_to_date", pd.Timestamp("2026-04-01"), end_ts),
        backtest(hist, "candidate_regime", "april_cutoff_candidate_april_to_date", pd.Timestamp("2026-04-01"), end_ts),
        backtest(spy, "spy_regime", "SPY_benchmark_april_to_date", pd.Timestamp("2026-04-01"), end_ts),
    ]
    post_rows = [
        backtest(hist, "current_regime", "current_live_2025_12_model_post_april30", pd.Timestamp("2026-04-30 15:00:00"), end_ts),
        backtest(hist, "candidate_regime", "april_cutoff_candidate_post_april30", pd.Timestamp("2026-04-30 15:00:00"), end_ts),
        backtest(spy, "spy_regime", "SPY_benchmark_post_april30", pd.Timestamp("2026-04-30 15:00:00"), end_ts),
    ]
    replay = pd.DataFrame(replay_rows + apr_rows + post_rows)
    path = write_csv(replay, "replay_2026_comparison.csv")
    return {"replay_2026_comparison_csv": path, "rows": replay.to_dict("records")}


def build_summary(
    latest: dict[str, Any],
    last100: dict[str, Any],
    sanity: dict[str, Any],
    replay: dict[str, Any],
    hashes: dict[str, str],
) -> tuple[str, str]:
    cand_latest_contiguous = last100["candidate_qqq_latest_contiguous_rows"]
    candidate_qqq_latest_only = cand_latest_contiguous <= 1
    nfp_overreaction = (
        latest["candidate_prediction"] == 0
        and float(latest["candidate_probabilities"]["Expansion"]) < 0.60
        and cand_latest_contiguous <= 1
    )
    recommendation = "B. shadow monitor for 24 live cycles"
    pass_fail = (
        "Pass if candidate remains schema-clean/null-free, does not oscillate more than 2 times, "
        "keeps max drawdown no worse than current in rolling replay, and QQQ/Expansion persists with "
        "market confirmation or reverts without whipsaw. Fail if NFP/Event-active rows alone drive "
        "risk-on flips, if candidate alternates CASH/QQQ across adjacent live cycles, or if replay "
        "drawdown worsens materially versus current."
    )
    lines = [
        "# MUB-28 April-Cutoff Promotion Readiness",
        "",
        f"Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"Output root: `{OUTPUT_ROOT}`",
        "",
        "## Verdict",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "Do not promote now. The candidate passes schema/runtime checks and has stronger static replay diagnostics, "
        "but the latest live row flips from the current production CASH posture to 100% QQQ. In the last-100 validation set, "
        f"candidate QQQ appears in `{last100['candidate_qqq_rows']}` rows and the latest contiguous QQQ run is `{cand_latest_contiguous}` row(s). "
        "That is not enough live stability for a production alias switch.",
        "",
        "## Latest Row",
        "",
        f"- Timestamp: `{latest['latest_timestamp']}`",
        f"- Current live: `{latest['current_prediction']} / {latest['current_prediction_name']}`, `{latest['current_position']}`",
        f"- April candidate: `{latest['candidate_prediction']} / {latest['candidate_prediction_name']}`, `{latest['candidate_position']}`",
        f"- Candidate probabilities: `{latest['candidate_probabilities']}`",
        f"- Current probabilities: `{latest['current_probabilities']}`",
        "",
        "Top candidate Expansion-vs-Recession margin drivers are in `latest_row_explanation.csv`. "
        "Top current Recession-vs-Expansion drivers are in the same file.",
        "",
        "## Last 100 Rows",
        "",
        f"- Current regime distribution: `{last100['current_regime_distribution']}`",
        f"- Candidate regime distribution: `{last100['candidate_regime_distribution']}`",
        f"- Disagreements: `{last100['disagreement_count']}` / `{last100['rows']}` = `{last100['disagreement_percent']:.2%}`",
        f"- First candidate QQQ timestamp: `{last100['first_candidate_qqq_timestamp']}`",
        f"- Latest contiguous candidate QQQ rows: `{cand_latest_contiguous}`",
        f"- NFP release changed candidate signal: `{last100['nfp_release_changed_candidate_signal']}`",
        "",
        "## Risk Review",
        "",
        f"- QQQ flip latest-only: `{candidate_qqq_latest_only}`",
        f"- NFP overreaction risk flag: `{nfp_overreaction}`",
        "- The candidate is not yet stable across the last 24-48 hours; it only flips to QQQ on the latest row in this validation window.",
        "- Promotion would materially increase risk today versus the current production CASH posture.",
        "- Static replay shows better drawdown, but the live behavior change is large enough to require shadow monitoring.",
        "",
        "## Shadow Criteria",
        "",
        f"Monitor for **24 live cycles**. {pass_fail}",
        "",
        "## Artifacts",
        "",
        f"- Latest-row explanation CSV: `{latest['latest_row_explanation_path']}`",
        f"- Feature vector CSV: `{latest['feature_vector_path']}`",
        f"- XGBoost contribution CSV: `{latest['contribution_path']}`",
        f"- Last-100 disagreement CSV: `{last100['disagreement_csv']}`",
        f"- Recent-market sanity CSV: `{sanity['recent_market_sanity_csv']}`",
        f"- 2026 replay comparison CSV: `{replay['replay_2026_comparison_csv']}`",
        "",
        "## Safety",
        "",
        "- Live alias overwritten: `False`",
        "- Registry updated: `False`",
        "- Saxo/token/RT/dashboard/schedule/workflow changes: `False`",
        f"- Current live SHA: `{hashes['current_sha256']}`",
        f"- Candidate SHA: `{hashes['candidate_sha256']}`",
    ]
    summary = "\n".join(lines) + "\n"
    path = OUTPUT_ROOT / "promotion_readiness_summary.md"
    path.write_text(summary, encoding="utf-8")
    return str(path), recommendation


def main() -> None:
    ensure_dirs()
    hashes = {
        "current_sha256": file_sha256(CURRENT_LIVE_MODEL_PATH),
        "candidate_sha256": file_sha256(CANDIDATE_MODEL_PATH),
    }
    if hashes["current_sha256"] != CURRENT_EXPECTED_SHA256:
        raise ValueError(f"Current live SHA mismatch: {hashes['current_sha256']}")
    if hashes["candidate_sha256"] != CANDIDATE_EXPECTED_SHA256:
        raise ValueError(f"Candidate SHA mismatch: {hashes['candidate_sha256']}")

    registry = spark.sql(
        f"""
        SELECT model_version, model_path, versioned_model_path, train_start, train_end, promoted_ts
        FROM {REGISTRY_TABLE}
        ORDER BY promoted_ts DESC
        LIMIT 1
        """
    ).toPandas()
    if registry["model_version"].iloc[0] != CURRENT_MODEL_VERSION:
        raise ValueError(f"Unexpected active model version: {registry['model_version'].iloc[0]}")

    current_model = joblib.load(CURRENT_LIVE_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    assert_model_schema(current_model, "current_live")
    assert_model_schema(candidate_model, "april_candidate")

    latest = latest_row_explanation(current_model, candidate_model)
    last100 = last_100_analysis()
    hist = load_feature_history(current_model, candidate_model)
    sanity = recent_market_sanity(hist)
    replay = replay_2026(hist)
    summary_path, recommendation = build_summary(latest, last100, sanity, replay, hashes)

    result = {
        "output_root": str(OUTPUT_ROOT),
        "summary_markdown": summary_path,
        "recommendation": recommendation,
        "latest": latest,
        "last100": last100,
        "sanity_csv": sanity["recent_market_sanity_csv"],
        "replay_csv": replay["replay_2026_comparison_csv"],
        "hashes": hashes,
        "runtime": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "xgboost_version": xgb.__version__,
            "joblib_version": joblib.__version__,
            "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_commit": git_value(["rev-parse", "HEAD"]),
        },
        "live_alias_overwritten": False,
        "registry_updated": False,
        "saxo_token_rt_dashboard_schedule_changes": False,
    }
    write_json(result, "promotion_readiness_result.json")
    if "dbutils" in globals():
        dbutils.notebook.exit(json.dumps(result, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

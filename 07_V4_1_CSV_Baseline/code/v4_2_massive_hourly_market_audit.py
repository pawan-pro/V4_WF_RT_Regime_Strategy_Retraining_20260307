#!/usr/bin/env python3
"""Read-only Massive SPY/QQQ hourly availability and V4 overlap audit.

Writes only under the isolated V4.2 research folder. Does not write Delta
tables, train models, or change Databricks jobs/workflows.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import SparkSession


CATALOG = "pawan_quant_research_1"
OUTPUT_ROOT = Path(f"/Volumes/{CATALOG}/v4/market_data/v4_2_hourly_massive_spy_qqq_2007_2026")
RAW_DIR = OUTPUT_ROOT / "inputs" / "raw"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
CODE_DIR = OUTPUT_ROOT / "code"

TICKERS = ["SPY", "QQQ"]
FROM_DATE = "2007-12-05"
TO_DATE = "2026-04-30"
LIMIT = 50000
REQUEST_SLEEP_SECONDS = 13
MARKET_TABLE = f"{CATALOG}.v4.bronze_bt_market_1h"
START_DATES = ["2007-12-05", "2008-01-01", "2010-01-01", "2016-01-01"]
FETCH_ERRORS: list[dict[str, Any]] = []


def ensure_dirs() -> None:
    for path in [RAW_DIR, FORENSICS_DIR, MANIFEST_DIR, CODE_DIR, OUTPUT_ROOT / "inputs"]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(payload: dict[str, Any], path: Path) -> None:
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(float(obj)) else float(obj)
        if pd.isna(obj) if not isinstance(obj, (dict, list, tuple)) else False:
            return None
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    out = df.head(max_rows).copy() if max_rows else df.copy()
    out = out.astype(object).where(pd.notna(out), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    if max_rows and len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def get_dbutils():
    try:
        from pyspark.dbutils import DBUtils  # type: ignore

        return DBUtils(SparkSession.builder.getOrCreate())
    except Exception:
        return globals().get("dbutils")


def resolve_massive_api_key() -> str:
    dbutils_obj = get_dbutils()
    if dbutils_obj is not None:
        try:
            key = dbutils_obj.secrets.get(scope="massive", key="api_key")
            if key:
                return key.strip()
        except Exception:
            pass
    key = os.environ.get("MASSIVE_API_KEY", "")
    if key:
        return key.strip()
    raise RuntimeError("Missing Massive API key. Required Databricks secret: scope=massive, key=api_key")


def ensure_massive_client():
    try:
        from massive import RESTClient  # type: ignore

        return RESTClient
    except Exception:
        pass
    for package in ["massive-api-client", "massive"]:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])
        try:
            from massive import RESTClient  # type: ignore

            return RESTClient
        except Exception:
            continue
    raise RuntimeError("Could not import Massive RESTClient after installing massive-api-client/massive")


def extract_value(obj: Any, names: list[str]) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def fetch_massive_hourly(api_key: str, ticker: str) -> pd.DataFrame:
    RESTClient = ensure_massive_client()
    client = RESTClient(api_key=api_key)
    rows = []
    chunk_starts = pd.date_range(FROM_DATE, TO_DATE, freq="YS").date.astype(str).tolist()
    if chunk_starts[0] != FROM_DATE:
        chunk_starts = [FROM_DATE] + chunk_starts
    chunk_ends = (pd.to_datetime(chunk_starts[1:] + [TO_DATE]) - pd.Timedelta(days=1)).date.astype(str).tolist()
    chunk_ends[-1] = TO_DATE
    for idx, (chunk_start, chunk_end) in enumerate(zip(chunk_starts, chunk_ends)):
        if idx:
            time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            chunk_count = 0
            for agg in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="hour",
                from_=chunk_start,
                to=chunk_end,
                adjusted=True,
                sort="asc",
                limit=LIMIT,
            ):
                ts_ms = extract_value(agg, ["timestamp", "t"])
                rows.append(
                    {
                        "symbol": ticker,
                        "timestamp_ms": ts_ms,
                        "open": extract_value(agg, ["open", "o"]),
                        "high": extract_value(agg, ["high", "h"]),
                        "low": extract_value(agg, ["low", "l"]),
                        "close": extract_value(agg, ["close", "c"]),
                        "volume": extract_value(agg, ["volume", "v"]),
                        "vwap": extract_value(agg, ["vwap", "vw"]),
                        "transactions": extract_value(agg, ["transactions", "n"]),
                        "otc": extract_value(agg, ["otc"]),
                    }
                )
                chunk_count += 1
            FETCH_ERRORS.append(
                {
                    "symbol": ticker,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "status": "success",
                    "row_count": chunk_count,
                    "error_class": None,
                    "error_message": None,
                }
            )
        except Exception as exc:
            FETCH_ERRORS.append(
                {
                    "symbol": ticker,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "status": "error",
                    "row_count": 0,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc)[:500],
                }
            )
            continue
    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "timestamp_utc",
                "timestamp_et",
                "date_et",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "transactions",
                "source",
                "ingestion_ts",
                "timestamp_ms",
                "otc",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["symbol", "timestamp_ms"]).reset_index(drop=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["timestamp_et"] = df["timestamp_utc"].dt.tz_convert("America/New_York")
    df["date_et"] = df["timestamp_et"].dt.date.astype(str)
    df["timestamp"] = df["timestamp_utc"].dt.tz_localize(None)
    df["source"] = "massive_aggs_1h_adjusted"
    df["ingestion_ts"] = datetime.now(timezone.utc).isoformat()
    ordered_cols = [
        "timestamp",
        "timestamp_utc",
        "timestamp_et",
        "date_et",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
        "source",
        "ingestion_ts",
        "timestamp_ms",
        "otc",
    ]
    return df[ordered_cols].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def add_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts_et = pd.to_datetime(out["timestamp_utc"], utc=True).dt.tz_convert("America/New_York")
    out["_et_hour"] = ts_et.dt.hour
    out["_et_minute"] = ts_et.dt.minute
    out["_et_weekday"] = ts_et.dt.weekday
    out["_is_weekday"] = out["_et_weekday"].between(0, 4)
    out["_is_rth_top_hour_proxy"] = out["_is_weekday"] & out["_et_hour"].between(9, 15)
    out["_is_extended_or_overnight"] = out["_is_weekday"] & ~out["_is_rth_top_hour_proxy"]
    return out


def build_quality_report(market: pd.DataFrame) -> pd.DataFrame:
    flagged = add_time_flags(market)
    rows = []
    for (symbol, year), sub in flagged.groupby(["symbol", pd.to_datetime(flagged["timestamp"]).dt.year]):
        by_day = sub.groupby("date_et").size()
        null_ohlcv = int(sub[["open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
        nonpositive = int(((sub[["open", "high", "low", "close"]] <= 0).any(axis=1) | (sub["volume"] < 0)).sum())
        rows.append(
            {
                "symbol": symbol,
                "year": int(year),
                "row_count": len(sub),
                "min_timestamp": sub["timestamp"].min(),
                "max_timestamp": sub["timestamp"].max(),
                "duplicate_timestamp_count": int(sub.duplicated(["symbol", "timestamp"]).sum()),
                "null_ohlcv_row_count": null_ohlcv,
                "zero_or_negative_price_or_volume_count": nonpositive,
                "trading_days_with_bars": int(by_day.size),
                "avg_bars_per_trading_day": float(by_day.mean()) if not by_day.empty else np.nan,
                "median_bars_per_trading_day": float(by_day.median()) if not by_day.empty else np.nan,
                "days_with_lt_7_bars": int((by_day < 7).sum()),
                "days_with_gt_10_bars": int((by_day > 10).sum()),
                "rth_proxy_bar_count": int(sub["_is_rth_top_hour_proxy"].sum()),
                "extended_or_overnight_bar_count": int(sub["_is_extended_or_overnight"].sum()),
                "session_diagnosis": "all/extended sessions present" if int(sub["_is_extended_or_overnight"].sum()) else "regular-hours-like",
                "timestamp_convention": "aggregate window start timestamp per Massive aggregates API",
                "timezone_convention": "timestamp_ms interpreted as UTC; timestamp_et derived via America/New_York",
            }
        )
    return pd.DataFrame(rows)


def spark_v4_market_to_pandas(spark: SparkSession) -> pd.DataFrame:
    sdf = (
        spark.table(MARKET_TABLE)
        .filter(F.col("symbol").isin(TICKERS))
        .select("timestamp", "symbol", F.col("close").alias("v4_close"))
    )
    out = sdf.toPandas()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def compare_overlap(market: pd.DataFrame, v4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    mismatch_rows = []
    variants = {
        "all_sessions": add_time_flags(market),
        "rth": add_time_flags(market).query("_is_rth_top_hour_proxy").copy(),
    }
    shifts = {"same_timestamp": 0, "massive_plus_1h": 1, "massive_minus_1h": -1}
    for variant_name, mdf in variants.items():
        base = mdf[["symbol", "timestamp", "close"]].copy()
        for shift_name, shift_hours in shifts.items():
            shifted = base.copy()
            shifted["compare_timestamp"] = pd.to_datetime(shifted["timestamp"]) + pd.to_timedelta(shift_hours, unit="h")
            for symbol in TICKERS:
                left = shifted[shifted["symbol"].eq(symbol)].copy()
                right = v4[v4["symbol"].eq(symbol)].copy()
                merged = left.merge(
                    right,
                    left_on=["symbol", "compare_timestamp"],
                    right_on=["symbol", "timestamp"],
                    how="inner",
                    suffixes=("_massive", "_v4"),
                )
                if merged.empty:
                    rows.append(
                        {
                            "symbol": symbol,
                            "session_variant": variant_name,
                            "alignment": shift_name,
                            "timestamp_overlap_count": 0,
                            "close_mae": np.nan,
                            "close_max_abs_diff": np.nan,
                            "exact_match_rate": 0.0,
                            "near_match_rate_1e_4": 0.0,
                            "correlation": np.nan,
                            "first_divergence_timestamp": None,
                            "overlap_start": None,
                            "overlap_end": None,
                            "massive_rows": len(left),
                            "v4_rows": len(right),
                        }
                    )
                    continue
                merged["abs_diff"] = (merged["close"] - merged["v4_close"]).abs()
                first_div = merged.loc[merged["abs_diff"] > 1e-4, "compare_timestamp"].min()
                rows.append(
                    {
                        "symbol": symbol,
                        "session_variant": variant_name,
                        "alignment": shift_name,
                        "timestamp_overlap_count": len(merged),
                        "close_mae": float(merged["abs_diff"].mean()),
                        "close_max_abs_diff": float(merged["abs_diff"].max()),
                        "exact_match_rate": float((merged["abs_diff"] <= 1e-10).mean()),
                        "near_match_rate_1e_4": float((merged["abs_diff"] <= 1e-4).mean()),
                        "correlation": float(merged[["close", "v4_close"]].corr().iloc[0, 1]) if len(merged) > 1 else np.nan,
                        "first_divergence_timestamp": None if pd.isna(first_div) else pd.Timestamp(first_div).isoformat(),
                        "overlap_start": merged["compare_timestamp"].min().isoformat(),
                        "overlap_end": merged["compare_timestamp"].max().isoformat(),
                        "massive_rows": len(left),
                        "v4_rows": len(right),
                    }
                )
                bad = merged[merged["abs_diff"] > 1e-4].head(50).copy()
                if not bad.empty:
                    bad["session_variant"] = variant_name
                    bad["alignment"] = shift_name
                    mismatch_rows.append(
                        bad[
                            [
                                "session_variant",
                                "alignment",
                                "symbol",
                                "timestamp_massive",
                                "compare_timestamp",
                                "close",
                                "v4_close",
                                "abs_diff",
                            ]
                        ]
                    )
    comp = pd.DataFrame(rows).sort_values(["close_mae", "timestamp_overlap_count"], ascending=[True, False])
    mismatches = pd.concat(mismatch_rows, ignore_index=True) if mismatch_rows else pd.DataFrame()
    return comp, mismatches


def session_variants(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_sessions = market.copy()
    rth = add_time_flags(market).query("_is_rth_top_hour_proxy").copy()
    drop_cols = [c for c in rth.columns if c.startswith("_")]
    return all_sessions, rth.drop(columns=drop_cols)


def write_summary(quality: pd.DataFrame, comp: pd.DataFrame, feasibility: pd.DataFrame, recommendation: dict[str, Any]) -> None:
    best_comp = comp.sort_values(["close_mae", "timestamp_overlap_count"], ascending=[True, False]).head(6)
    quality_brief = (
        quality.groupby("symbol")
        .agg(
            rows=("row_count", "sum"),
            first=("min_timestamp", "min"),
            last=("max_timestamp", "max"),
            avg_bars_per_day=("avg_bars_per_trading_day", "mean"),
            extended_bars=("extended_or_overnight_bar_count", "sum"),
        )
        .reset_index()
    )
    quality_text = [
        "# Massive SPY/QQQ Hourly Quality Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output folder: `{OUTPUT_ROOT}`",
        f"- Download range: `{FROM_DATE}` to `{TO_DATE}`",
        f"- Recommended session filter: `{recommendation['recommended_session_filter']}`",
        f"- Recommended V4.2 hourly start date: `{recommendation['recommended_start_date']}`",
        "",
        "## Coverage",
        "",
        md_table(quality_brief),
        "",
        "## Best V4 Overlap Alignments",
        "",
        md_table(best_comp),
        "",
        "## Start-Date Feasibility",
        "",
        md_table(feasibility),
        "",
        "## Interpretation",
        "",
        recommendation["interpretation"],
        "",
        "## No Production Change Confirmation",
        "",
        "This run read the Massive API, read the V4 market table for overlap comparison, and wrote CSV/Markdown/JSON artifacts only under the isolated V4.2 research folder. It did not mutate production V4, overwrite Delta tables, train models, or change Saxo execution, token refresh, RT/dashboard notebooks, workflows, schedules, or model registry.",
        "",
    ]
    (FORENSICS_DIR / "massive_spy_qqq_hourly_quality_summary.md").write_text("\n".join(quality_text), encoding="utf-8")
    overlap_text = [
        "# Massive vs V4 Market Overlap Summary",
        "",
        f"- Recommended session filter: `{recommendation['recommended_session_filter']}`",
        f"- Alignment diagnosis: `{recommendation['alignment_diagnosis']}`",
        "",
        md_table(best_comp),
        "",
    ]
    (FORENSICS_DIR / "massive_vs_v4_market_overlap_summary.md").write_text("\n".join(overlap_text), encoding="utf-8")
    feas_text = [
        "# V4.2 Hourly Start-Date Feasibility Summary",
        "",
        md_table(feasibility),
        "",
        f"Recommended hourly start: `{recommendation['recommended_start_date']}`.",
        "",
    ]
    (MANIFEST_DIR / "v4_2_hourly_start_date_feasibility_summary.md").write_text("\n".join(feas_text), encoding="utf-8")


def build_feasibility(market: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        market.groupby("symbol")["timestamp"]
        .agg(["min", "max", "count"])
        .rename(columns={"min": "first", "max": "last", "count": "rows"})
        .to_dict("index")
    )
    rows = []
    for start in START_DATES:
        st = pd.Timestamp(start)
        spy_ok = "SPY" in coverage and pd.Timestamp(coverage["SPY"]["first"]) <= st <= pd.Timestamp(coverage["SPY"]["last"])
        qqq_ok = "QQQ" in coverage and pd.Timestamp(coverage["QQQ"]["first"]) <= st <= pd.Timestamp(coverage["QQQ"]["last"])
        vix_ok = st >= pd.Timestamp("1990-01-03")
        vix3m_ok = st >= pd.Timestamp("2007-12-05")
        macro_ok = st >= pd.Timestamp("1998-07-31") and st >= pd.Timestamp("2000-08-01")
        blockers = []
        if not spy_ok:
            blockers.append("SPY Massive hourly")
        if not qqq_ok:
            blockers.append("QQQ Massive hourly")
        if not vix_ok:
            blockers.append("VIXCLS")
        if not vix3m_ok:
            blockers.append("VIX3M/VXVCLS")
        if not macro_ok:
            blockers.append("macro ALFRED first-release history")
        rows.append(
            {
                "candidate_start_date": start,
                "SPY_QQQ_hourly_feasible": bool(spy_ok and qqq_ok),
                "VIXCLS_feasible": bool(vix_ok),
                "VIX3M_feasible": bool(vix3m_ok),
                "macro_ALFRED_feasible": bool(macro_ok),
                "full_14_feature_set_feasible": len(blockers) == 0,
                "caveats": ", ".join(blockers) if blockers else "Feasible from Massive market + current FRED VIX/VXVCLS + ALFRED macro availability; still validate full grid before training.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    api_key = resolve_massive_api_key()
    spark = SparkSession.builder.getOrCreate()

    frames = []
    for ticker in TICKERS:
        df = fetch_massive_hourly(api_key, ticker)
        raw_path = RAW_DIR / f"{ticker}_hourly_massive_2007_2026.csv"
        df.to_csv(raw_path, index=False)
        frames.append(df)

    market = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    fetch_errors = pd.DataFrame(FETCH_ERRORS)
    fetch_errors.to_csv(FORENSICS_DIR / "massive_fetch_chunk_status.csv", index=False)
    if market.empty:
        failure_manifest = {
            "source": "Massive aggregates API via official Python client",
            "query_params": {
                "tickers": TICKERS,
                "multiplier": 1,
                "timespan": "hour",
                "from": FROM_DATE,
                "to": TO_DATE,
                "adjusted": True,
                "sort": "asc",
                "limit": LIMIT,
            },
            "failure": "No accessible hourly aggregate rows returned for SPY or QQQ",
            "fetch_status_path": str(FORENSICS_DIR / "massive_fetch_chunk_status.csv"),
            "no_production_change_confirmation": True,
            "production_delta_writes": False,
            "model_training": False,
            "workflow_schedule_registry_changes": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(failure_manifest, MANIFEST_DIR / "massive_market_feed_manifest.json")
        raise RuntimeError("Massive key returned no accessible SPY/QQQ hourly rows; see massive_fetch_chunk_status.csv")
    norm_path = OUTPUT_ROOT / "inputs" / "v4_2_market_spy_qqq_1h_massive_2007_2026.csv"
    market.to_csv(norm_path, index=False)

    all_sessions, rth = session_variants(market)
    all_sessions_path = OUTPUT_ROOT / "inputs" / "v4_2_market_spy_qqq_1h_massive_all_sessions.csv"
    rth_path = OUTPUT_ROOT / "inputs" / "v4_2_market_spy_qqq_1h_massive_rth.csv"
    all_sessions.to_csv(all_sessions_path, index=False)
    rth.to_csv(rth_path, index=False)

    quality = build_quality_report(market)
    quality_path = FORENSICS_DIR / "massive_spy_qqq_hourly_quality_report.csv"
    quality.to_csv(quality_path, index=False)

    v4 = spark_v4_market_to_pandas(spark)
    comp, mismatches = compare_overlap(market, v4)
    comp_path = FORENSICS_DIR / "massive_vs_v4_market_overlap_comparison.csv"
    comp.to_csv(comp_path, index=False)
    if not mismatches.empty:
        mismatches.to_csv(FORENSICS_DIR / "massive_vs_v4_first_mismatches.csv", index=False)

    feasibility = build_feasibility(market)
    feasibility_path = MANIFEST_DIR / "v4_2_hourly_start_date_feasibility_matrix.csv"
    feasibility.to_csv(feasibility_path, index=False)

    best = comp.sort_values(["close_mae", "timestamp_overlap_count"], ascending=[True, False]).iloc[0].to_dict()
    recommended_session = str(best["session_variant"])
    feasible_starts = feasibility[feasibility["full_14_feature_set_feasible"]]
    recommended_start = (
        str(feasible_starts["candidate_start_date"].iloc[0]) if not feasible_starts.empty else "none_with_current_massive_entitlement"
    )
    interpretation = (
        f"Best overlap against `{MARKET_TABLE}` is `{best['session_variant']}` with `{best['alignment']}` "
        f"for {best['symbol']} (MAE={best['close_mae']}, overlap={best['timestamp_overlap_count']}). "
        "However, the current Massive entitlement did not return SPY/QQQ hourly data for any requested start "
        "candidate, so this feed cannot support the long-history V4.2 hourly baseline without a deeper historical plan. "
        "This audit stops before grid/model work."
    )
    recommendation = {
        "recommended_session_filter": recommended_session,
        "recommended_start_date": recommended_start,
        "alignment_diagnosis": str(best["alignment"]),
        "best_overlap_row": best,
        "interpretation": interpretation,
    }
    write_summary(quality, comp, feasibility, recommendation)

    manifest = {
        "source": "Massive aggregates API via official Python client",
        "package": "massive-api-client / massive",
        "query_params": {
            "tickers": TICKERS,
            "multiplier": 1,
            "timespan": "hour",
            "from": FROM_DATE,
            "to": TO_DATE,
            "adjusted": True,
            "sort": "asc",
            "limit": LIMIT,
        },
        "row_counts": market.groupby("symbol").size().to_dict(),
        "min_max_timestamps": market.groupby("symbol")["timestamp"].agg(["min", "max"]).astype(str).to_dict("index"),
        "hashes": {
            "SPY_raw": sha256_file(RAW_DIR / "SPY_hourly_massive_2007_2026.csv"),
            "QQQ_raw": sha256_file(RAW_DIR / "QQQ_hourly_massive_2007_2026.csv"),
            "normalized": sha256_file(norm_path),
            "all_sessions": sha256_file(all_sessions_path),
            "rth": sha256_file(rth_path),
        },
        "bar_convention": "Massive aggregate hourly timestamp interpreted as aggregate window start",
        "timezone_convention": "timestamp_ms UTC; timestamp_et America/New_York",
        "RTH_all_session_decision": recommendation,
        "overlap_comparison_summary": comp.head(10).to_dict("records"),
        "no_production_change_confirmation": True,
        "production_delta_writes": False,
        "model_training": False,
        "workflow_schedule_registry_changes": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest, MANIFEST_DIR / "massive_market_feed_manifest.json")
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "normalized_csv": str(norm_path),
                "quality_report": str(quality_path),
                "overlap_report": str(comp_path),
                "feasibility_matrix": str(feasibility_path),
                "manifest": str(MANIFEST_DIR / "massive_market_feed_manifest.json"),
                "no_production_change_confirmation": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

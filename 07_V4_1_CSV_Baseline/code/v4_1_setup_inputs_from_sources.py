"""Freeze V4.1 inputs as CSV files.

Run this inside Databricks on a cluster with read access to the approved source
table and with FRED_API_KEY available as an environment variable or Databricks
secret-backed environment variable. This script writes only under V4_1_ROOT.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, time, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from v4_1_common import (
    CATALOG,
    INDICATORS,
    MARKET_SOURCE_TABLE,
    TIMING_VARIANTS,
    V4_1_ROOT,
    UTC,
    ensure_v4_1_dirs,
    file_manifest,
    macro_available_ts,
    ny_time_to_utc,
    parse_utc,
    print_json,
    sha256_file,
    write_csv,
    write_json,
)


EVENT_TIMES = {
    "CPIAUCSL": "08:30",
    "PCE": "08:30",
    "PCEPI": "08:30",
    "PAYEMS": "08:30",
    "UNRATE": "08:30",
    "GDPC1": "08:30",
    "INDPRO": "09:15",
    "UMCSENT": "10:00",
}


def timing_assumptions() -> pd.DataFrame:
    rows = [
        ["macro", "CPIAUCSL", "08:30", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Pre-market release usable at ETF open"],
        ["macro", "PCE", "08:30", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Pre-market release usable at ETF open"],
        ["macro", "PCEPI", "08:30", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Pre-market release usable at ETF open"],
        ["macro", "PAYEMS", "08:30", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Pre-market release usable at ETF open"],
        ["macro", "UNRATE", "08:30", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Pre-market release usable at ETF open"],
        ["macro", "GDPC1", "08:30", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Pre-market release usable at ETF open"],
        ["macro", "INDPRO", "09:15", "market_open_if_premarket_else_release_plus_buffer", 0, "main", "Released before/open-adjacent; usable from ETF open or first grid after release"],
        ["macro", "UMCSENT", "10:00", "release_plus_buffer", 30, "main", "During-market release; test 30m/60m sensitivity"],
        ["vix", "VIXCLS", "prior_close", "next_trading_day_10am", 0, "main", "Daily close value usable next trading day 10:00 ET"],
        ["vix", "VXVCLS", "prior_close", "next_trading_day_10am", 0, "main", "Daily close value usable next trading day 10:00 ET"],
    ]
    return pd.DataFrame(rows, columns=["source_type", "series", "event_time_et", "main_availability_rule", "buffer_minutes", "sensitivity_group", "notes"])


def timing_variants() -> pd.DataFrame:
    rows = [
        [TIMING_VARIANTS[0], "Macro pre-market at ETF open; during-market release plus 30 minutes; VIX next trading day 10:00 ET", 30, "main realistic candidate"],
        [TIMING_VARIANTS[1], "Macro pre-market at ETF open; during-market release plus 60 minutes; VIX next trading day 10:00 ET", 60, "conservative intraday macro buffer"],
        [TIMING_VARIANTS[2], "All macro releases plus 120 minutes; VIX next trading day 10:00 ET", 120, "old-V4 compatibility sensitivity"],
        [TIMING_VARIANTS[3], "Macro first hourly grid timestamp at or after release; VIX next trading day 10:00 ET", 0, "aggressive release-availability sensitivity"],
    ]
    return pd.DataFrame(rows, columns=["variant", "description", "macro_intraday_buffer_minutes", "notes"])


def export_market_csv(root: Path) -> Path:
    try:
        spark_session = spark  # type: ignore[name-defined]
    except NameError as exc:
        try:
            from pyspark.sql import SparkSession

            spark_session = SparkSession.builder.getOrCreate()
        except Exception as spark_exc:
            raise RuntimeError("Market export requires Databricks/Spark. Run this script in Databricks.") from spark_exc

    sdf = spark_session.sql(
        f"""
        SELECT timestamp, symbol, close
        FROM {MARKET_SOURCE_TABLE}
        WHERE symbol IN ('SPY', 'QQQ')
        ORDER BY timestamp, symbol
        """
    )
    df = sdf.toPandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    null_close = int(df["close"].isna().sum())
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    duplicate_groups = (
        df.groupby(["timestamp", "symbol"], as_index=False)
        .agg(row_count=("close", "size"), min_close=("close", "min"), max_close=("close", "max"), close=("close", "last"))
    )
    duplicate_audit = duplicate_groups[duplicate_groups["row_count"] > 1].copy()
    duplicate_audit["has_conflicting_close"] = (duplicate_audit["max_close"] - duplicate_audit["min_close"]).abs() > 1e-12
    if not duplicate_audit.empty:
        write_csv(
            duplicate_audit,
            root / "manifests" / "v4_1_market_duplicate_resolution.csv",
            root,
        )
    df = duplicate_groups[["timestamp", "symbol", "close"]].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    dupes = int(df.duplicated(["timestamp", "symbol"]).sum())
    if dupes or null_close or symbols != ["QQQ", "SPY"]:
        raise ValueError({"duplicate_timestamp_symbol_after_resolution": dupes, "null_close": null_close, "symbols": symbols})
    path = root / "inputs" / "v4_1_market_spy_qqq_1h.csv"
    write_csv(df, path, root)
    return path


def fred_api_get(endpoint: str, params: dict[str, str], api_key: str) -> dict:
    base = f"https://api.stlouisfed.org/fred/{endpoint}"
    query = {"api_key": api_key, "file_type": "json", **params}
    resp = requests.get(f"{base}?{urlencode(query)}", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if "error_code" in payload:
        raise RuntimeError(payload)
    return payload


def fetch_release_dates(series_id: str, api_key: str) -> list[pd.Timestamp]:
    payload = fred_api_get(
        "series/vintagedates",
        {"series_id": series_id, "realtime_start": "2015-01-01", "realtime_end": "2026-12-31"},
        api_key,
    )
    return [pd.Timestamp(x).date() for x in payload.get("vintage_dates", [])]


def fetch_vintage_observations(series_id: str, release_date, api_key: str) -> pd.DataFrame:
    payload = fred_api_get(
        "series/observations",
        {
            "series_id": series_id,
            "realtime_start": str(release_date),
            "realtime_end": str(release_date),
            "observation_start": "1900-01-01",
            "observation_end": str(release_date),
        },
        api_key,
    )
    return pd.DataFrame(payload.get("observations", []))


def fetch_all_release_observations(series_id: str, api_key: str) -> pd.DataFrame:
    payload = fred_api_get(
        "series/observations",
        {
            "series_id": series_id,
            "realtime_start": "1776-07-04",
            "realtime_end": "9999-12-31",
            "limit": "100000",
        },
        api_key,
    )
    observations = pd.DataFrame(payload.get("observations", []))
    if observations.empty:
        return observations
    observations = observations.replace(".", pd.NA)
    observations["realtime_start"] = pd.to_datetime(observations["realtime_start"], errors="coerce").dt.date
    observations["observation_date"] = pd.to_datetime(observations["date"], errors="coerce")
    observations["value"] = pd.to_numeric(observations["value"], errors="coerce")
    return observations.dropna(subset=["realtime_start", "observation_date", "value"])


def build_macro_first_releases(root: Path, api_key: str) -> Path:
    rows = []
    fallback_rows = []
    for indicator in INDICATORS:
        all_obs = fetch_all_release_observations(indicator, api_key)
        if all_obs.empty:
            fallback_rows.append({"indicator_id": indicator, "release_date": None, "fallback_reason": "no ALFRED all-release rows returned"})
            continue
        all_obs = all_obs[
            (pd.to_datetime(all_obs["realtime_start"]) >= pd.Timestamp("2015-01-01"))
            & (pd.to_datetime(all_obs["realtime_start"]) <= pd.Timestamp("2026-12-31"))
        ].copy()
        for release_date, obs in all_obs.groupby("realtime_start", sort=True):
            obs = obs.dropna(subset=["observation_date", "value"]).copy()
            if obs.empty:
                fallback_rows.append({"indicator_id": indicator, "release_date": str(release_date), "fallback_reason": "no non-null values in exact realtime bucket"})
                continue
            selected = obs.sort_values("observation_date").iloc[-1]
            event_time = EVENT_TIMES[indicator]
            release_ts_utc = ny_time_to_utc(release_date, event_time)
            available_main = macro_available_ts(
                release_ts_utc,
                event_time,
                "macro_market_open_or_30m_buffer__vix_next_day_10am",
            )
            rows.append(
                {
                    "indicator_id": indicator,
                    "observation_date": selected["observation_date"].date().isoformat(),
                    "release_date": str(release_date),
                    "actual_first_release": float(selected["value"]),
                    "assumed_release_time_et": event_time,
                    "release_ts_utc": release_ts_utc.isoformat(),
                    "available_ts_utc_main": available_main.isoformat(),
                    "value_source": "ALFRED",
                    "release_time_source": "standard_assumption",
                    "selection_rule": "latest_non_null_observation_date_within_exact_realtime_start_release_date_bucket",
                    "fallback_used": False,
                    "notes": "ALFRED/FRED vintage bucket selected; availability recomputed per timing variant downstream.",
                }
            )

    macro = pd.DataFrame(rows).sort_values(["indicator_id", "release_ts_utc", "observation_date"])
    if macro.empty:
        raise ValueError("No ALFRED macro rows were produced")
    path = root / "inputs" / "v4_1_macro_alfred_first_releases.csv"
    write_csv(macro, path, root)
    if fallback_rows:
        write_csv(pd.DataFrame(fallback_rows), root / "manifests" / "v4_1_macro_alfred_fallback_log.csv", root)
    return path


def fetch_fred_daily_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    value_col = series_id
    df = df.rename(columns={date_col: "observation_date", value_col: "close"})
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
    df["close"] = pd.to_numeric(df["close"].replace(".", pd.NA), errors="coerce")
    return df.dropna(subset=["close"])


def next_trading_day_10am_et(obs_date) -> pd.Timestamp:
    d = pd.Timestamp(obs_date) + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return pd.Timestamp(datetime.combine(d.date(), time(10, 0)), tz="America/New_York").tz_convert("UTC").tz_localize(None)


def build_vix_pit_csv(root: Path, market_path: Path) -> Path:
    market = pd.read_csv(market_path)
    market["timestamp"] = parse_utc(market["timestamp"])
    end_ts = market["timestamp"].max()
    hourly = pd.DataFrame({"timestamp": pd.date_range("2016-01-01", end_ts, freq="h")})
    rows = []
    for series in ["VIXCLS", "VXVCLS"]:
        daily = fetch_fred_daily_csv(series)
        daily["available_ts_utc"] = daily["observation_date"].apply(next_trading_day_10am_et)
        daily = daily[daily["available_ts_utc"] <= end_ts].sort_values("available_ts_utc")
        merged = pd.merge_asof(
            hourly.sort_values("timestamp"),
            daily[["available_ts_utc", "close", "observation_date"]].sort_values("available_ts_utc"),
            left_on="timestamp",
            right_on="available_ts_utc",
            direction="backward",
        )
        merged = merged.dropna(subset=["close"]).copy()
        merged["series"] = series
        merged["source"] = "fred_1d_pit_next_trading_day_10am"
        rows.append(merged[["timestamp", "series", "close", "observation_date", "available_ts_utc", "source"]])
    out = pd.concat(rows).sort_values(["timestamp", "series"])
    path = root / "inputs" / "v4_1_vix_vix3m_1h_pit.csv"
    write_csv(out, path, root)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(V4_1_ROOT))
    parser.add_argument("--fred-api-key", default=os.environ.get("FRED_API_KEY"))
    parser.add_argument("--skip-alfred", action="store_true", help="Only write timing configs and market/VIX if ALFRED key is unavailable.")
    args = parser.parse_args()

    root = ensure_v4_1_dirs(args.root)
    write_csv(timing_assumptions(), root / "inputs" / "v4_1_timing_assumptions.csv", root)
    write_csv(timing_variants(), root / "inputs" / "v4_1_timing_variants.csv", root)
    market_path = export_market_csv(root)
    macro_path = None
    if args.fred_api_key:
        macro_path = build_macro_first_releases(root, args.fred_api_key)
    elif not args.skip_alfred:
        raise RuntimeError("FRED_API_KEY is required to build ALFRED first-release CSV. Use --skip-alfred only for partial setup.")
    vix_path = build_vix_pit_csv(root, market_path)

    manifests = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "writes_only_under": str(root),
            "production_delta_aliases_modified": False,
            "production_workflows_modified": False,
        },
        "files": [
            file_manifest(root / "inputs" / "v4_1_timing_assumptions.csv", "Explicit V4.1 timing assumption config"),
            file_manifest(root / "inputs" / "v4_1_timing_variants.csv", "Explicit V4.1 timing sensitivity variants"),
            file_manifest(market_path, "Frozen SPY/QQQ hourly market CSV exported read-only from Delta", MARKET_SOURCE_TABLE),
            file_manifest(vix_path, "Frozen FRED VIX/VIX3M hourly PIT CSV expanded from daily source"),
        ],
    }
    if macro_path:
        manifests["files"].append(file_manifest(macro_path, "Frozen ALFRED/FRED first-release macro CSV"))
    write_json(manifests, root / "manifests" / "v4_1_input_hashes.json", root)
    print_json(manifests)


if __name__ == "__main__":
    main()

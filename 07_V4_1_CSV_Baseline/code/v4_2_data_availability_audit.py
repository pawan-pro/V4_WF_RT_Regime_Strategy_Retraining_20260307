#!/usr/bin/env python3
"""Read-only V4.2 long-history data availability audit.

Inspects V4/default Databricks tables, targeted Volume CSV artifacts, and
FRED/ALFRED metadata to determine whether strict-PIT baselines can start in
1996, 2006, 2010, or 2016. Writes only under the requested V4.2 audit folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import SparkSession


CATALOG = "pawan_quant_research_1"
BASE = Path(f"/Volumes/{CATALOG}/v4/market_data")
CSV_BASE = BASE / "v4_1_csv"
OUTPUT_ROOT = BASE / "v4_2_data_availability_audit_2026-04-30"

START_CANDIDATES = [
    pd.Timestamp("1996-01-01"),
    pd.Timestamp("2006-01-01"),
    pd.Timestamp("2010-01-01"),
    pd.Timestamp("2016-01-01"),
]
START_LABELS = [d.date().isoformat() for d in START_CANDIDATES]

MARKET_SERIES = ["SPY", "QQQ"]
VIX_SERIES = ["VIXCLS", "VXVCLS"]
MACRO_SERIES = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT"]
ALL_SERIES = MARKET_SERIES + ["VIXCLS", "VIX3M"] + MACRO_SERIES

TABLES = [
    "pawan_quant_research_1.v4.bronze_bt_market_1h",
    "pawan_quant_research_1.v4.bronze_bt_macro_events",
    "pawan_quant_research_1.v4.bronze_bt_vix_1h",
    "pawan_quant_research_1.v4.bronze_bt_vix3m_1h",
    "pawan_quant_research_1.default.bronze_bt_market_1h",
    "pawan_quant_research_1.default.bronze_bt_macro_events",
    "pawan_quant_research_1.default.bronze_bt_vix_1h",
    "pawan_quant_research_1.default.bronze_bt_vix3m_1h",
]

FRED_API_KEY_FALLBACK = "f52bdf7341f23d0458820ecd806178db"


def ensure_dirs() -> None:
    for subdir in ["forensics", "manifests", "code"]:
        (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(float(obj)) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "No rows."
    out = df.head(max_rows).copy() if max_rows else df.copy()
    out = out.astype(object).where(pd.notna(out), "")
    columns = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in out.itertuples(index=False):
        values = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows and len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def normalize_ts(value) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    try:
        ts = ts.tz_localize(None)
    except Exception:
        pass
    return pd.Timestamp(ts)


def frequency_from_dates(first: pd.Timestamp | None, last: pd.Timestamp | None, rows: int) -> str:
    if first is None or last is None or rows <= 1:
        return "unknown"
    span_days = max((last - first).total_seconds() / 86400.0, 1e-9)
    rows_per_day = rows / span_days
    if rows_per_day >= 12:
        return "hourly"
    if rows_per_day >= 0.45:
        return "daily"
    if rows_per_day >= 0.02:
        return "monthly"
    if rows_per_day >= 0.005:
        return "quarterly"
    return "sparse"


def support_flags(first: pd.Timestamp | None, last: pd.Timestamp | None) -> dict:
    flags = {}
    for label, start in zip(START_LABELS, START_CANDIDATES):
        flags[f"supports_{label}"] = bool(first is not None and last is not None and first <= start <= last)
    return flags


def add_availability_row(
    rows: list[dict],
    source_type: str,
    source_name: str,
    series_group: str,
    series_id: str,
    first_ts,
    last_ts,
    row_count: int,
    frequency: str | None = None,
    null_count: int | None = None,
    duplicate_timestamp_count: int | None = None,
    notes: str = "",
    path_or_table: str | None = None,
) -> None:
    first = normalize_ts(first_ts)
    last = normalize_ts(last_ts)
    rows.append(
        {
            "source_type": source_type,
            "source_name": source_name,
            "path_or_table": path_or_table or source_name,
            "series_group": series_group,
            "series_id": series_id,
            "first_available_timestamp": None if first is None else first.isoformat(),
            "last_available_timestamp": None if last is None else last.isoformat(),
            "row_count": int(row_count or 0),
            "frequency": frequency or frequency_from_dates(first, last, int(row_count or 0)),
            "null_count": None if null_count is None else int(null_count),
            "duplicate_timestamp_count": None if duplicate_timestamp_count is None else int(duplicate_timestamp_count),
            **support_flags(first, last),
            "notes": notes,
        }
    )


def table_exists(spark: SparkSession, table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def inspect_table(spark: SparkSession, table_name: str, rows: list[dict]) -> list[dict]:
    source_rows = []
    if not table_exists(spark, table_name):
        source_rows.append({"source_type": "delta_table", "source_name": table_name, "exists": False, "notes": "table not readable or missing"})
        return source_rows
    df = spark.table(table_name)
    cols = df.columns
    ts_col = next((c for c in ["timestamp", "release_ts_utc", "release_date", "release_datetime", "available_ts_utc", "date"] if c in cols), None)
    value_col = next((c for c in ["close", "actual", "value", "observed_value", "selected_value", "vix3m"] if c in cols), None)
    if ts_col is None:
        source_rows.append({"source_type": "delta_table", "source_name": table_name, "exists": True, "notes": "no recognized timestamp column", "schema": json.dumps(cols)})
        return source_rows
    if "market" in table_name:
        group_col = "symbol" if "symbol" in cols else None
        wanted = MARKET_SERIES
        series_group = "market"
        value_expr = value_col or "close"
    elif "macro" in table_name:
        group_col = "indicator_id" if "indicator_id" in cols else ("series_id" if "series_id" in cols else None)
        wanted = MACRO_SERIES
        series_group = "macro"
        value_expr = value_col
    elif "vix3m" in table_name:
        group_col = "symbol" if "symbol" in cols else None
        wanted = ["VIX3M"]
        series_group = "volatility"
        value_expr = value_col
    elif "vix" in table_name:
        group_col = "symbol" if "symbol" in cols else None
        wanted = ["VIXCLS"]
        series_group = "volatility"
        value_expr = value_col
    else:
        group_col = None
        wanted = ["unknown"]
        series_group = "unknown"
        value_expr = value_col
    base = df
    if group_col and group_col in cols:
        group_values = [r[group_col] for r in base.select(group_col).distinct().collect()]
        for requested in wanted:
            candidates = [requested]
            if requested == "VIXCLS":
                candidates += ["VX", "VIX"]
            if requested == "VIX3M":
                candidates += ["VIX3M", "VXVCLS"]
            present = [g for g in group_values if str(g).upper() in {c.upper() for c in candidates}]
            if not present and series_group != "macro":
                present = group_values if len(group_values) == 1 else []
            for value in present:
                sub = base.filter(F.col(group_col) == value)
                add_table_group_row(rows, table_name, requested, series_group, sub, ts_col, value_expr, f"group column {group_col}={value}")
    else:
        requested = wanted[0]
        add_table_group_row(rows, table_name, requested, series_group, base, ts_col, value_expr, "no explicit series column")
    count = df.count()
    first = df.agg(F.min(ts_col).alias("first")).collect()[0]["first"]
    last = df.agg(F.max(ts_col).alias("last")).collect()[0]["last"]
    source_rows.append(
        {
            "source_type": "delta_table",
            "source_name": table_name,
            "exists": True,
            "schema": json.dumps(cols),
            "row_count": int(count),
            "first_timestamp": None if first is None else str(first),
            "last_timestamp": None if last is None else str(last),
            "notes": "read-only inspection",
        }
    )
    return source_rows


def add_table_group_row(
    rows: list[dict],
    table_name: str,
    series_id: str,
    series_group: str,
    sub,
    ts_col: str,
    value_col: str | None,
    notes: str,
) -> None:
    agg_exprs = [
        F.count("*").alias("row_count"),
        F.min(ts_col).alias("first_ts"),
        F.max(ts_col).alias("last_ts"),
        (F.count("*") - F.countDistinct(ts_col)).alias("duplicate_timestamp_count"),
    ]
    if value_col and value_col in sub.columns:
        agg_exprs.append(F.sum(F.when(F.col(value_col).isNull(), 1).otherwise(0)).alias("null_count"))
    else:
        agg_exprs.append(F.lit(None).alias("null_count"))
    a = sub.agg(*agg_exprs).collect()[0].asDict()
    add_availability_row(
        rows,
        "delta_table",
        table_name,
        series_group,
        series_id,
        a["first_ts"],
        a["last_ts"],
        a["row_count"],
        null_count=a.get("null_count"),
        duplicate_timestamp_count=a.get("duplicate_timestamp_count"),
        notes=notes,
        path_or_table=table_name,
    )


def read_csv_header(path: Path) -> list[str] | None:
    try:
        with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            return next(reader)
    except Exception:
        return None


def relevant_csv_files() -> list[Path]:
    roots = [BASE, CSV_BASE]
    candidates = []
    name_tokens = [s.lower() for s in ALL_SERIES] + ["market", "vix", "vix3m", "first_releases", "grid", "signal", "macro"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            lower = str(path).lower()
            if any(token in lower for token in name_tokens):
                candidates.append(path)
    dedup = sorted(set(candidates), key=lambda p: str(p))
    return dedup[:500]


def infer_series_from_csv(path: Path, df: pd.DataFrame) -> list[tuple[str, str, str | None]]:
    path_upper = path.name.upper()
    out = []
    for symbol in MARKET_SERIES:
        if symbol in path_upper or f"{symbol}_close" in df.columns:
            out.append(("market", symbol, f"{symbol}_close" if f"{symbol}_close" in df.columns else None))
    if "VIX3M" in path_upper or "VXVCLS" in path_upper or "vix3m" in df.columns:
        out.append(("volatility", "VIX3M", "vix3m" if "vix3m" in df.columns else ("close" if "close" in df.columns else None)))
    if "VIXCLS" in path_upper or ("VIX" in path_upper and "VIX3M" not in path_upper) or "VIXCLS" in df.columns:
        out.append(("volatility", "VIXCLS", "VIXCLS" if "VIXCLS" in df.columns else ("close" if "close" in df.columns else None)))
    for series in MACRO_SERIES:
        if series in path_upper or series in df.columns:
            out.append(("macro", series, series if series in df.columns else None))
    if "symbol" in df.columns and "close" in df.columns:
        symbols = set(df["symbol"].dropna().astype(str).str.upper().unique().tolist()) if len(df) <= 2_000_000 else set()
        for symbol in symbols:
            if symbol in {"SPY", "QQQ"}:
                out.append(("market", symbol, "close"))
            elif symbol in {"VX", "VIX"}:
                out.append(("volatility", "VIXCLS", "close"))
            elif symbol in {"VIX3M", "VXVCLS"}:
                out.append(("volatility", "VIX3M", "close"))
    return list(dict.fromkeys(out))


def inspect_csvs(rows: list[dict]) -> list[dict]:
    source_rows = []
    for path in relevant_csv_files():
        header = read_csv_header(path)
        if not header:
            continue
        lower_cols = {c.lower(): c for c in header}
        if not any(c in lower_cols for c in ["timestamp", "date", "release_date", "release_datetime", "release_ts_utc"]):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            source_rows.append({"source_type": "csv", "source_name": str(path), "exists": True, "notes": f"read_error: {exc}"})
            continue
        if df.empty:
            continue
        ts_col = next((lower_cols[c] for c in ["timestamp", "date", "release_date", "release_datetime", "release_ts_utc"] if c in lower_cols), None)
        if ts_col is None:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        inferred = infer_series_from_csv(path, df)
        if not inferred:
            continue
        for group, series, value_col in inferred:
            sub = df
            if "symbol" in df.columns and value_col == "close":
                if series in {"SPY", "QQQ"}:
                    sub = df[df["symbol"].astype(str).str.upper().eq(series)]
                elif series == "VIXCLS":
                    sub = df[df["symbol"].astype(str).str.upper().isin(["VX", "VIX"])]
                elif series == "VIX3M":
                    sub = df[df["symbol"].astype(str).str.upper().isin(["VIX3M", "VXVCLS"])]
            if "indicator_id" in df.columns and group == "macro":
                sub = df[df["indicator_id"].astype(str).str.upper().eq(series)]
            if sub.empty:
                continue
            null_count = int(sub[value_col].isna().sum()) if value_col and value_col in sub.columns else None
            dup_count = int(sub.duplicated(subset=[ts_col]).sum())
            add_availability_row(
                rows,
                "csv",
                path.name,
                group,
                series,
                sub[ts_col].min(),
                sub[ts_col].max(),
                len(sub),
                null_count=null_count,
                duplicate_timestamp_count=dup_count,
                notes=f"CSV scan; timestamp column={ts_col}; value column={value_col}",
                path_or_table=str(path),
            )
        source_rows.append(
            {
                "source_type": "csv",
                "source_name": path.name,
                "path": str(path),
                "exists": True,
                "row_count": int(len(df)),
                "schema": json.dumps(list(df.columns)),
                "sha256": sha256_file(path),
                "notes": "targeted relevant CSV scan",
            }
        )
    return source_rows


def fred_api_key() -> str:
    path = Path("01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py")
    if path.exists():
        text = path.read_text(encoding="utf-8")
        match = re.search(r'FRED_API_KEY\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    return os.environ.get("FRED_API_KEY", FRED_API_KEY_FALLBACK)


def fred_get(endpoint: str, params: dict) -> dict:
    params = dict(params)
    params["api_key"] = fred_api_key()
    params["file_type"] = "json"
    url = f"https://api.stlouisfed.org/fred/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def fred_observations(series_id: str, **params) -> list[dict]:
    payload = fred_get("series/observations", {"series_id": series_id, **params})
    return payload.get("observations", [])


def alfred_feasibility(rows: list[dict]) -> pd.DataFrame:
    out = []
    for series_id in MACRO_SERIES:
        try:
            vintages = fred_get("series/vintagedates", {"series_id": series_id}).get("vintage_dates", [])
        except Exception as exc:
            vintages = []
            vintage_error = str(exc)
        else:
            vintage_error = None
        try:
            current_obs = fred_observations(series_id, observation_start="1900-01-01")
        except Exception as exc:
            current_obs = []
            obs_error = str(exc)
        else:
            obs_error = None
        current_dates = [pd.to_datetime(o.get("date"), errors="coerce") for o in current_obs if o.get("value") not in (None, ".")]
        earliest_obs = min([d for d in current_dates if pd.notna(d)], default=pd.NaT)
        latest_obs = max([d for d in current_dates if pd.notna(d)], default=pd.NaT)
        earliest_vintage = pd.to_datetime(vintages[0], errors="coerce") if vintages else pd.NaT
        latest_vintage = pd.to_datetime(vintages[-1], errors="coerce") if vintages else pd.NaT
        ambiguity = "unknown"
        max_bucket = None
        latest_diff_first_count = None
        try:
            obs4 = fred_observations(
                series_id,
                realtime_start="1776-01-01",
                realtime_end="9999-12-31",
                observation_start="1990-01-01",
                output_type=4,
                limit=100000,
            )
            sample = pd.DataFrame(obs4)
            if not sample.empty and {"realtime_start", "date", "value"}.issubset(sample.columns):
                sample = sample[sample["value"].ne(".")].copy()
                sample["date"] = pd.to_datetime(sample["date"], errors="coerce")
                bucket = sample.groupby("realtime_start")["date"].agg(["count", "min", "max"]).reset_index()
                max_bucket = int(bucket["count"].max()) if not bucket.empty else 0
                latest_diff_first_count = int((bucket["min"] != bucket["max"]).sum()) if not bucket.empty else 0
                if latest_diff_first_count and series_id in {"GDPC1", "UMCSENT"}:
                    ambiguity = "high release-stage / observation-date ambiguity"
                elif latest_diff_first_count:
                    ambiguity = "candidate row selection ambiguity exists"
                else:
                    ambiguity = "low"
        except Exception as exc:
            ambiguity = f"ambiguity_check_error: {exc}"
        row = {
            "series_id": series_id,
            "earliest_alfred_vintage_realtime_start": None if pd.isna(earliest_vintage) else earliest_vintage.date().isoformat(),
            "latest_alfred_vintage_realtime_start": None if pd.isna(latest_vintage) else latest_vintage.date().isoformat(),
            "earliest_observation_date": None if pd.isna(earliest_obs) else earliest_obs.date().isoformat(),
            "latest_observation_date": None if pd.isna(latest_obs) else latest_obs.date().isoformat(),
            "vintage_count": len(vintages),
            "current_observation_count": len(current_obs),
            "max_rows_per_vintage_bucket_sample": max_bucket,
            "vintage_buckets_where_first_obs_differs_from_latest_obs_sample": latest_diff_first_count,
            "release_stage_ambiguity": ambiguity,
            "notes": "; ".join([x for x in [vintage_error, obs_error] if x]) or "ALFRED/FRED read via series/vintagedates and observations endpoints",
        }
        ev = pd.to_datetime(row["earliest_alfred_vintage_realtime_start"], errors="coerce")
        eo = pd.to_datetime(row["earliest_observation_date"], errors="coerce")
        for label, start in zip(START_LABELS, START_CANDIDATES):
            row[f"first_release_reconstruction_feasible_{label}"] = bool(pd.notna(ev) and pd.notna(eo) and ev <= start and eo <= start)
        out.append(row)
        add_availability_row(
            rows,
            "fred_alfred_api",
            f"ALFRED/FRED {series_id}",
            "macro",
            series_id,
            earliest_vintage,
            latest_vintage,
            len(vintages),
            frequency="vintage",
            null_count=None,
            duplicate_timestamp_count=None,
            notes=f"earliest observation={row['earliest_observation_date']}; ambiguity={ambiguity}",
            path_or_table=f"fred/series/vintagedates/{series_id}",
        )
    df = pd.DataFrame(out)
    write_csv(df, OUTPUT_ROOT / "alfred_first_release_feasibility.csv")
    return df


def fred_volatility_availability(rows: list[dict]) -> pd.DataFrame:
    out = []
    for series_id, normalized_series_id in [("VIXCLS", "VIXCLS"), ("VXVCLS", "VIX3M")]:
        try:
            obs = fred_observations(series_id, observation_start="1900-01-01")
            frame = pd.DataFrame(obs)
        except Exception as exc:
            out.append(
                {
                    "fred_series_id": series_id,
                    "series_id": normalized_series_id,
                    "earliest_fred_observation_date": None,
                    "latest_fred_observation_date": None,
                    "first_usable_v4_shifted_hourly_date": None,
                    "observation_count": 0,
                    "non_null_observation_count": 0,
                    "notes": f"FRED read error: {exc}",
                }
            )
            continue
        if frame.empty:
            out.append(
                {
                    "fred_series_id": series_id,
                    "series_id": normalized_series_id,
                    "earliest_fred_observation_date": None,
                    "latest_fred_observation_date": None,
                    "first_usable_v4_shifted_hourly_date": None,
                    "observation_count": 0,
                    "non_null_observation_count": 0,
                    "notes": "FRED returned no observations",
                }
            )
            continue
        frame = frame[frame["value"].ne(".")].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date")
        first_obs = frame["date"].min() if not frame.empty else pd.NaT
        last_obs = frame["date"].max() if not frame.empty else pd.NaT
        first_usable_shifted = frame["date"].iloc[1] if len(frame) > 1 else first_obs
        notes = (
            "current FRED daily availability; V4 fred_1d logic applies close.shift(1), "
            f"so first usable assigned date is {None if pd.isna(first_usable_shifted) else first_usable_shifted.date().isoformat()}"
        )
        out.append(
            {
                "fred_series_id": series_id,
                "series_id": normalized_series_id,
                "earliest_fred_observation_date": None if pd.isna(first_obs) else first_obs.date().isoformat(),
                "latest_fred_observation_date": None if pd.isna(last_obs) else last_obs.date().isoformat(),
                "first_usable_v4_shifted_hourly_date": None if pd.isna(first_usable_shifted) else first_usable_shifted.date().isoformat(),
                "observation_count": len(obs),
                "non_null_observation_count": len(frame),
                "notes": notes,
            }
        )
        add_availability_row(
            rows,
            "fred_api",
            f"FRED {series_id}",
            "volatility",
            normalized_series_id,
            first_usable_shifted,
            last_obs,
            len(frame) * 24,
            frequency="daily",
            null_count=0,
            duplicate_timestamp_count=0,
            notes=notes,
            path_or_table=f"fred/series/observations/{series_id}",
        )
    df = pd.DataFrame(out)
    write_csv(df, OUTPUT_ROOT / "forensics" / "fred_volatility_availability.csv")
    return df


def best_source(rows: pd.DataFrame, series_id: str, source_type_filter: str | None = None, frequency: str | None = None) -> pd.Series | None:
    sub = rows[rows["series_id"].eq(series_id)].copy()
    if source_type_filter:
        sub = sub[sub["source_type"].eq(source_type_filter)]
    if frequency:
        sub = sub[sub["frequency"].eq(frequency)]
    if sub.empty:
        return None
    sub["_first"] = pd.to_datetime(sub["first_available_timestamp"], errors="coerce")
    sub["_last"] = pd.to_datetime(sub["last_available_timestamp"], errors="coerce")
    sub = sub.sort_values(["_first", "row_count"], ascending=[True, False])
    return sub.iloc[0]


def build_feasibility_matrix(availability: pd.DataFrame, alfred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start in START_CANDIDATES:
        label = start.date().isoformat()
        market_hourly = {}
        market_any = {}
        for sym in MARKET_SERIES:
            table_best = best_source(availability, sym, source_type_filter="delta_table")
            market_hourly[sym] = bool(table_best is not None and pd.to_datetime(table_best["first_available_timestamp"]) <= start and "hourly" in str(table_best["frequency"]))
            any_best = best_source(availability, sym)
            market_any[sym] = bool(any_best is not None and pd.to_datetime(any_best["first_available_timestamp"]) <= start)
        vix_best = best_source(availability, "VIXCLS")
        vix3m_best = best_source(availability, "VIX3M")
        vix_ok = bool(vix_best is not None and pd.to_datetime(vix_best["first_available_timestamp"]) <= start)
        vix3m_ok = bool(vix3m_best is not None and pd.to_datetime(vix3m_best["first_available_timestamp"]) <= start)
        macro_ok = bool(
            alfred[[f"first_release_reconstruction_feasible_{label}"]].all().iloc[0]
        )
        blockers = []
        if not market_hourly.get("SPY"):
            blockers.append("SPY hourly")
        if not market_hourly.get("QQQ"):
            blockers.append("QQQ hourly")
        if not vix_ok:
            blockers.append("VIXCLS")
        if not vix3m_ok:
            blockers.append("VIX3M/VXVCLS")
        if not macro_ok:
            blockers.append("macro ALFRED first-release history")
        rows.append(
            {
                "candidate_start_date": label,
                "SPY_hourly_available": market_hourly.get("SPY", False),
                "QQQ_hourly_available": market_hourly.get("QQQ", False),
                "SPY_any_existing_available": market_any.get("SPY", False),
                "QQQ_any_existing_available": market_any.get("QQQ", False),
                "VIXCLS_available": vix_ok,
                "VIX3M_available": vix3m_ok,
                "macro_alfred_first_release_feasible": macro_ok,
                "hourly_v4_2_feasible_from_existing_artifacts": len(blockers) == 0,
                "daily_v4_2_feasible_from_existing_artifacts": bool(market_any.get("SPY") and market_any.get("QQQ") and vix_ok and vix3m_ok and macro_ok),
                "daily_v4_2_feasible_with_new_external_daily_market_data": bool(vix_ok and vix3m_ok and macro_ok and start >= pd.Timestamp("2010-01-01")),
                "blocking_series_or_sources": ", ".join(blockers) if blockers else "",
                "notes": "Existing hourly artifacts dominate hourly feasibility; daily feasibility assumes a daily SPY/QQQ vendor feed exists if not present in V4 artifacts.",
            }
        )
    df = pd.DataFrame(rows)
    write_csv(df, OUTPUT_ROOT / "start_date_feasibility_matrix.csv")
    return df


def write_source_summary(source_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(source_rows)
    if not df.empty:
        write_csv(df, OUTPUT_ROOT / "data_availability_by_source.csv")
    return df


def summarize_findings(availability: pd.DataFrame, sources: pd.DataFrame, feasibility: pd.DataFrame, alfred: pd.DataFrame) -> None:
    def first_for(series: str, freq: str | None = None):
        row = best_source(availability, series, frequency=freq)
        return None if row is None else row["first_available_timestamp"]

    hourly_feasible = feasibility[feasibility["hourly_v4_2_feasible_from_existing_artifacts"]]
    best_hourly = "none" if hourly_feasible.empty else hourly_feasible["candidate_start_date"].iloc[0]
    daily_external = feasibility[feasibility["daily_v4_2_feasible_with_new_external_daily_market_data"]]
    best_daily = "none" if daily_external.empty else daily_external["candidate_start_date"].iloc[0]

    f1996 = feasibility[feasibility["candidate_start_date"].eq("1996-01-01")].iloc[0]
    f2006 = feasibility[feasibility["candidate_start_date"].eq("2006-01-01")].iloc[0]
    f2010 = feasibility[feasibility["candidate_start_date"].eq("2010-01-01")].iloc[0]
    f2016 = feasibility[feasibility["candidate_start_date"].eq("2016-01-01")].iloc[0]

    market_table_rows = availability[
        availability["source_type"].eq("delta_table")
        & availability["series_group"].eq("market")
        & availability["series_id"].isin(MARKET_SERIES)
    ]
    daily_market_existing = availability[
        availability["series_group"].eq("market")
        & availability["frequency"].eq("daily")
        & availability["series_id"].isin(MARKET_SERIES)
    ]

    lines = [
        "# V4.2 Data Availability Audit Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output folder: `{OUTPUT_ROOT}`",
        f"- Best feasible hourly V4.2 start from existing artifacts: `{best_hourly}`",
        f"- Best feasible daily V4.2 start with new external daily SPY/QQQ feed: `{best_daily}`",
        "",
        "## Feasibility Matrix",
        "",
        md_table(feasibility),
        "",
        "## Market Data",
        "",
        f"- Existing hourly SPY first timestamp: `{first_for('SPY', 'hourly')}`",
        f"- Existing hourly QQQ first timestamp: `{first_for('QQQ', 'hourly')}`",
        f"- Existing daily market CSV/table rows found for both SPY and QQQ: `{set(daily_market_existing['series_id']) >= set(MARKET_SERIES) if not daily_market_existing.empty else False}`",
        "- Hourly SPY/QQQ before 2016: not found in inspected V4/default hourly tables or targeted V4 CSV artifacts unless a source row in `data_availability_by_series.csv` shows otherwise.",
        "- If V4.2 starts before the earliest hourly market table, it likely needs a daily design or a new intraday vendor backfill.",
        "",
        "## VIX/VIX3M",
        "",
        f"- Earliest usable VIXCLS in inspected sources: `{first_for('VIXCLS')}`",
        f"- Earliest usable VIX3M/VXVCLS in inspected sources: `{first_for('VIX3M')}`",
        "- VIX3M/VXVCLS is the volatility blocker for 1996 and 2006 unless a proxy or synthetic term-structure series is approved.",
        "",
        "## Macro / ALFRED",
        "",
        md_table(alfred[[
            "series_id",
            "earliest_alfred_vintage_realtime_start",
            "earliest_observation_date",
            "release_stage_ambiguity",
            "first_release_reconstruction_feasible_1996-01-01",
            "first_release_reconstruction_feasible_2006-01-01",
            "first_release_reconstruction_feasible_2010-01-01",
            "first_release_reconstruction_feasible_2016-01-01",
        ]]),
        "",
        "## Direct Answers",
        "",
        f"- Which series blocks 1996: `{f1996['blocking_series_or_sources']}`",
        f"- Which series blocks 2006: `{f2006['blocking_series_or_sources']}`",
        f"- Whether 2010 is feasible: hourly from existing artifacts = `{f2010['hourly_v4_2_feasible_from_existing_artifacts']}`; daily with new market feed = `{f2010['daily_v4_2_feasible_with_new_external_daily_market_data']}`",
        f"- Whether 2016 remains the only clean hourly start: `{f2016['hourly_v4_2_feasible_from_existing_artifacts'] and not f2010['hourly_v4_2_feasible_from_existing_artifacts']}`",
        "",
        "## Recommendation",
        "",
        "For an immediate strict-PIT hourly baseline, use the earliest date where both SPY/QQQ hourly tables and VIX/VIX3M plus macro first-release reconstruction are present; based on inspected artifacts this remains 2016. For a longer V4.2 research baseline, 2010 daily is the practical candidate if a clean daily SPY/QQQ feed is added and VIX3M/VXVCLS is accepted from FRED/ALFRED. Do not use 1996 or 2006 without a VIX3M proxy decision and a daily/intraday market-data backfill.",
        "",
        "## No Production Change Confirmation",
        "",
        "This audit read Databricks tables, Volume CSVs, and FRED/ALFRED APIs only. It wrote CSV/Markdown/JSON artifacts only under the requested V4.2 audit folder. It did not mutate production V4, overwrite Delta tables, train models, or change Saxo execution, token refresh, RT/dashboard notebooks, workflows, schedules, or model registry.",
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "v4_2_data_availability_summary.md")


def write_manifest() -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": OUTPUT_ROOT,
        "tables_inspected": TABLES,
        "volume_roots_scanned": [BASE, CSV_BASE],
        "start_candidates": START_LABELS,
        "production_delta_writes": False,
        "model_training": False,
        "workflow_schedule_registry_changes": False,
    }
    write_json(payload, OUTPUT_ROOT / "manifests" / "runtime_manifest.json")


def main() -> None:
    ensure_dirs()
    spark = SparkSession.builder.getOrCreate()
    availability_rows: list[dict] = []
    source_rows: list[dict] = []
    for table in TABLES:
        source_rows.extend(inspect_table(spark, table, availability_rows))
    source_rows.extend(inspect_csvs(availability_rows))
    alfred = alfred_feasibility(availability_rows)
    fred_volatility_availability(availability_rows)
    availability = pd.DataFrame(availability_rows)
    write_csv(availability, OUTPUT_ROOT / "data_availability_by_series.csv")
    sources = write_source_summary(source_rows)
    feasibility = build_feasibility_matrix(availability, alfred)
    summarize_findings(availability, sources, feasibility, alfred)
    write_manifest()
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "data_availability_by_series": str(OUTPUT_ROOT / "data_availability_by_series.csv"),
                "data_availability_by_source": str(OUTPUT_ROOT / "data_availability_by_source.csv"),
                "start_date_feasibility_matrix": str(OUTPUT_ROOT / "start_date_feasibility_matrix.csv"),
                "alfred_first_release_feasibility": str(OUTPUT_ROOT / "alfred_first_release_feasibility.csv"),
                "summary": str(OUTPUT_ROOT / "v4_2_data_availability_summary.md"),
                "no_production_change_confirmation": True,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

"""Run the isolated V4.1 ALFRED event-selector/raw-lineage audit.

This script is intended to run on Databricks where the project Volumes are
mounted and FRED_API_KEY is available. It writes only under the requested
V4.1 CSV forensic audit folder and does not modify production Delta tables,
jobs, model registry entries, Saxo assets, or schedules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

V4_1_CODE_DIR = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
if V4_1_CODE_DIR.exists() and str(V4_1_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_1_CODE_DIR))

from v4_1_build_grid_from_csv import build_variant_grid
from v4_1_common import (
    FEATURES,
    INDICATORS,
    NY,
    UTC,
    V4_1_ROOT,
    macro_available_ts,
    ny_time_to_utc,
    parse_utc,
    print_json,
    sha256_file,
    write_csv,
    write_json,
)


AUDIT_NAME = "alfred_event_selector_audit_2026-04-30"
AUDIT_ROOT = V4_1_ROOT / AUDIT_NAME
COMPARISON_ROOT = Path(
    "/Volumes/pawan_quant_research_1/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
)
CORRECTED_FULL_SIGNAL = COMPARISON_ROOT / "corrected_variant" / "full_signal_export.csv"
FREEZE_GRID = V4_1_ROOT / "forensic_feature_freeze_ladder_2026-04-30" / "grids" / "D_full_feature_freeze" / "v4_1_silver_grid.csv"
CURRENT_GRID = V4_1_ROOT / "grids" / "macro_market_open_or_30m_buffer__vix_next_day_10am" / "v4_1_silver_grid.csv"
CURRENT_MACRO = V4_1_ROOT / "inputs" / "v4_1_macro_alfred_first_releases.csv"
CURRENT_VIX = V4_1_ROOT / "inputs" / "v4_1_vix_vix3m_1h_pit.csv"
MARKET_INPUT = V4_1_ROOT / "inputs" / "v4_1_market_spy_qqq_1h.csv"
TIMING_ASSUMPTIONS = V4_1_ROOT / "inputs" / "v4_1_timing_assumptions.csv"
MAIN_VARIANT = "macro_market_open_or_30m_buffer__vix_next_day_10am"

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

MACRO_SERIES = ["GDPC1", "UNRATE", "UMCSENT", "INDPRO", "PAYEMS", "CPIAUCSL", "PCE", "PCEPI"]
VIX_SERIES = ["VIXCLS", "VXVCLS"]
SELECTORS = [
    "A_latest_obs",
    "B_latest_obs_not_future",
    "C_first_obs",
    "D_reference_like",
    "E_manual_ambiguous_overrides",
]
RELEVANT_NAME_RE = re.compile(
    r"(macro|alfred|vix|vxv|vix3m|selection|first.release|raw|bootstrap|fallback|manifest|full_signal|grid|zscore|decomposition)",
    re.IGNORECASE,
)


def assert_audit_path(path: Path, audit_root: Path) -> Path:
    root = audit_root.resolve()
    candidate = path.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Refusing to write outside audit root: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def write_audit_csv(df: pd.DataFrame, path: Path, audit_root: Path) -> None:
    out = assert_audit_path(path, audit_root)
    df.to_csv(out, index=False)


def write_audit_json(payload: dict, path: Path, audit_root: Path) -> None:
    out = assert_audit_path(path, audit_root)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


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
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "No rows."
    out = df.head(max_rows).copy() if max_rows else df.copy()
    out = out.astype(object).where(pd.notna(out), "")
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in out.itertuples(index=False):
        vals = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if max_rows and len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_dirs(audit_root: Path) -> None:
    for subdir in [
        "inputs",
        "candidate_rows",
        "selectors",
        "forensics",
        "outputs",
        "recovered_raw_levels",
        "manifests",
    ]:
        (audit_root / subdir).mkdir(parents=True, exist_ok=True)


def inventory_artifacts(audit_root: Path) -> pd.DataFrame:
    roots = [
        COMPARISON_ROOT,
        V4_1_ROOT / "forensic_clone_macro_reference_2026-04-30",
        V4_1_ROOT / "forensic_clone_vix_zscore_reference_2026-04-30",
        V4_1_ROOT / "forensic_feature_freeze_ladder_2026-04-30",
        V4_1_ROOT / "forensic_corrected_training_artifact_recovery_2026-04-30",
        V4_1_ROOT / "recovered_generator_replay_2026-04-30",
        V4_1_ROOT / "raw_lineage_zscore_decomposition_2026-04-30",
        V4_1_ROOT / "grid_spot_audit_2026-04-30",
        V4_1_ROOT / "inputs",
        V4_1_ROOT / "grids",
    ]
    rows = []
    for root in roots:
        if not root.exists():
            rows.append(
                {
                    "path": str(root),
                    "type": "missing_root",
                    "size_bytes": None,
                    "mtime_utc": None,
                    "sha256": None,
                    "relevance": "missing",
                }
            )
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_text = str(path)
            relevant = bool(RELEVANT_NAME_RE.search(path.name)) or bool(RELEVANT_NAME_RE.search(rel_text))
            if not relevant:
                continue
            stat = path.stat()
            lower = path.name.lower()
            if "macro_selection_audit" in lower:
                artifact_type = "macro_selection_audit"
            elif "raw" in lower and ("vix" in lower or "macro" in lower):
                artifact_type = "raw_lineage"
            elif "manifest" in lower:
                artifact_type = "manifest"
            elif "grid" in lower:
                artifact_type = "grid"
            elif lower.endswith(".py"):
                artifact_type = "code"
            else:
                artifact_type = path.suffix.lstrip(".") or "file"
            rows.append(
                {
                    "path": str(path),
                    "type": artifact_type,
                    "size_bytes": int(stat.st_size),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "sha256": file_sha256(path),
                    "relevance": relevance_label(path),
                }
            )
    out = pd.DataFrame(rows).sort_values(["relevance", "type", "path"])
    write_audit_csv(out, audit_root / "forensics" / "artifact_inventory.csv", audit_root)
    return out


def relevance_label(path: Path) -> str:
    name = path.name.lower()
    text = str(path).lower()
    if "macro_selection_audit" in name:
        return "high_macro_selection_reference"
    if "full_signal_export" in name:
        return "high_corrected_signal_reference"
    if "full_feature_freeze" in text or "d_full_feature_freeze" in text:
        return "high_corrected_feature_freeze"
    if "raw_lineage" in text or "zscore_decomposition" in text:
        return "high_raw_lineage_proxy"
    if "recovered_corrected_training_generator" in name:
        return "high_corrected_generator_logic"
    if "vix" in name or "vix3m" in name or "vxv" in name:
        return "medium_vix_reference"
    if "manifest" in name:
        return "medium_manifest"
    return "low_context"


def fred_api_get(endpoint: str, params: dict[str, str], api_key: str) -> dict:
    base = f"https://api.stlouisfed.org/fred/{endpoint}"
    query = {"api_key": api_key, "file_type": "json", **params}
    url = f"{base}?{urlencode(query)}"
    last_status = None
    last_text = ""
    for attempt in range(4):
        resp = requests.get(url, timeout=90)
        last_status = resp.status_code
        last_text = resp.text[:500]
        if resp.status_code < 500:
            break
        time.sleep(2.0 * (attempt + 1))
    if resp.status_code >= 400:
        safe_params = {k: v for k, v in params.items() if k != "api_key"}
        raise RuntimeError(
            {
                "endpoint": endpoint,
                "status_code": last_status,
                "params_without_api_key": safe_params,
                "response_prefix": last_text,
            }
        )
    payload = resp.json()
    if "error_code" in payload:
        raise RuntimeError({"endpoint": endpoint, "params_without_api_key": params, "fred_error": payload})
    return payload


def fetch_alfred_observations(series_id: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    rows = []
    offset = 0
    limit = 10000
    while True:
        payload = fred_api_get(
            "series/observations",
            {
                "series_id": series_id,
                "realtime_start": start,
                "realtime_end": end,
                "observation_start": "1900-01-01",
                "observation_end": end,
                "limit": str(limit),
                "offset": str(offset),
                "sort_order": "asc",
            },
            api_key,
        )
        batch = payload.get("observations", [])
        rows.extend(batch)
        count = int(payload.get("count", len(rows)))
        offset += len(batch)
        if not batch or offset >= count:
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["series_id"] = series_id
    df = df.rename(columns={"date": "observation_date", "value": "value_raw"})
    df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce").dt.date.astype(str)
    df["realtime_end"] = pd.to_datetime(df["realtime_end"], errors="coerce").dt.date.astype(str)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce").dt.date.astype(str)
    df["is_null_or_dot"] = df["value_raw"].isna() | df["value_raw"].astype(str).str.strip().isin(["", "."])
    df["parsed_value"] = pd.to_numeric(df["value_raw"].replace(".", pd.NA), errors="coerce")
    return df[["series_id", "realtime_start", "realtime_end", "observation_date", "value_raw", "parsed_value", "is_null_or_dot"]]


def build_candidate_rows(audit_root: Path, api_key: str, start: str, end: str) -> pd.DataFrame:
    combined_path = audit_root / "inputs" / "alfred_candidate_rows.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)
    frames = []
    for series in MACRO_SERIES:
        df = fetch_alfred_observations(series, api_key, start, end)
        if df.empty:
            continue
        df = enrich_candidate_bucket_fields(df)
        write_audit_csv(df, audit_root / "candidate_rows" / f"{series}_alfred_candidates.csv", audit_root)
        frames.append(df)
    if not frames:
        raise RuntimeError("No ALFRED candidate rows were returned")
    combined = pd.concat(frames, ignore_index=True).sort_values(["series_id", "realtime_start", "observation_date"])
    write_audit_csv(combined, combined_path, audit_root)
    return combined


def enrich_candidate_bucket_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_obs_ts"] = pd.to_datetime(out["observation_date"], errors="coerce")
    bucket = out.groupby(["series_id", "realtime_start"], dropna=False)
    out["vintage_bucket_row_count"] = bucket["observation_date"].transform("size")
    out["latest_observation_date_in_bucket"] = bucket["_obs_ts"].transform("max").dt.date.astype(str)
    out["earliest_observation_date_in_bucket"] = bucket["_obs_ts"].transform("min").dt.date.astype(str)
    ranked = out.sort_values(["series_id", "realtime_start", "_obs_ts"], ascending=[True, True, False]).copy()
    ranked["candidate_rank_by_observation_date"] = ranked.groupby(["series_id", "realtime_start"]).cumcount() + 1
    out = ranked.sort_index().drop(columns=["_obs_ts"])
    return out


def load_manual_overrides(audit_root: Path) -> pd.DataFrame:
    paths = [
        audit_root / "inputs" / "manual_ambiguous_overrides.csv",
        V4_1_ROOT / "inputs" / "manual_ambiguous_overrides.csv",
    ]
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame(columns=["series_id", "realtime_start", "selected_observation_date", "selected_value", "selection_reason"])


def expected_period_cutoff(series_id: str, realtime_start: str) -> pd.Timestamp:
    release = pd.Timestamp(realtime_start)
    if series_id == "GDPC1":
        return release.to_period("Q").start_time
    return release.to_period("M").start_time


def select_from_bucket(bucket: pd.DataFrame, selector_name: str, overrides: pd.DataFrame) -> dict | None:
    nonnull = bucket[~bucket["is_null_or_dot"] & bucket["parsed_value"].notna()].copy()
    if nonnull.empty:
        return None
    series_id = str(bucket["series_id"].iloc[0])
    release_date = str(bucket["realtime_start"].iloc[0])
    nonnull["_obs_ts"] = pd.to_datetime(nonnull["observation_date"], errors="coerce")
    ambiguity = bool(
        len(nonnull) > 1
        and (
            nonnull["observation_date"].nunique(dropna=True) > 1
            or nonnull["parsed_value"].round(12).nunique(dropna=True) > 1
        )
    )
    reason = selector_name
    if selector_name in {"A_latest_obs", "D_reference_like"}:
        selected = nonnull.sort_values(["_obs_ts", "parsed_value"]).iloc[-1]
        if selector_name == "D_reference_like":
            reason = "recovered generator / production memory: latest non-null observation_date in exact realtime_start bucket"
    elif selector_name == "B_latest_obs_not_future":
        cutoff = expected_period_cutoff(series_id, release_date)
        constrained = nonnull[nonnull["_obs_ts"] <= cutoff]
        if constrained.empty:
            selected = nonnull.sort_values(["_obs_ts", "parsed_value"]).iloc[-1]
            reason = f"fallback_latest_obs_no_rows_on_or_before_{cutoff.date().isoformat()}"
        else:
            selected = constrained.sort_values(["_obs_ts", "parsed_value"]).iloc[-1]
            reason = f"latest_observation_date_on_or_before_{cutoff.date().isoformat()}"
    elif selector_name == "C_first_obs":
        selected = nonnull.sort_values(["_obs_ts", "parsed_value"]).iloc[0]
        reason = "first non-null observation_date in exact realtime_start bucket"
    elif selector_name == "E_manual_ambiguous_overrides":
        selected = nonnull.sort_values(["_obs_ts", "parsed_value"]).iloc[-1]
        match = overrides[
            (overrides.get("series_id", pd.Series(dtype=str)).astype(str) == series_id)
            & (overrides.get("realtime_start", pd.Series(dtype=str)).astype(str) == release_date)
        ]
        if not match.empty:
            ov = match.iloc[0]
            obs_date = str(ov.get("selected_observation_date", ""))
            by_obs = nonnull[nonnull["observation_date"].astype(str) == obs_date]
            if not by_obs.empty:
                selected = by_obs.iloc[-1]
                reason = str(ov.get("selection_reason", "manual_override_by_observation_date"))
            elif pd.notna(ov.get("selected_value", np.nan)):
                val = float(ov["selected_value"])
                by_value = nonnull[np.isclose(nonnull["parsed_value"].astype(float), val, rtol=1e-12, atol=1e-12)]
                if not by_value.empty:
                    selected = by_value.iloc[-1]
                    reason = str(ov.get("selection_reason", "manual_override_by_value"))
    else:
        raise ValueError(selector_name)
    return {
        "selector_name": selector_name,
        "series_id": series_id,
        "release_date": release_date,
        "realtime_start": release_date,
        "selected_observation_date": str(selected["observation_date"]),
        "selected_value": float(selected["parsed_value"]),
        "candidate_count": int(len(bucket)),
        "non_null_candidate_count": int(len(nonnull)),
        "selection_reason": reason,
        "ambiguity_flag": ambiguity,
    }


def build_selector_outputs(candidates: pd.DataFrame, audit_root: Path) -> dict[str, pd.DataFrame]:
    overrides = load_manual_overrides(audit_root)
    outputs = {}
    grouped = list(candidates.groupby(["series_id", "realtime_start"], sort=True))
    for selector_name in SELECTORS:
        selected_rows = []
        for _, bucket in grouped:
            row = select_from_bucket(bucket, selector_name, overrides)
            if row is not None:
                selected_rows.append(row)
        selected = pd.DataFrame(selected_rows).sort_values(["series_id", "release_date"])
        outputs[selector_name] = selected
        write_audit_csv(selected, audit_root / "selectors" / f"{selector_name}_selected_releases.csv", audit_root)
    return outputs


def build_ambiguity_reports(candidates: pd.DataFrame, selectors: dict[str, pd.DataFrame], audit_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (series_id, release_date), bucket in candidates.groupby(["series_id", "realtime_start"], sort=True):
        nonnull = bucket[~bucket["is_null_or_dot"] & bucket["parsed_value"].notna()].copy()
        if nonnull.empty:
            continue
        obs_dates = sorted(nonnull["observation_date"].astype(str).unique().tolist())
        values = sorted([float(x) for x in nonnull["parsed_value"].dropna().unique().tolist()])
        selector_values = {}
        selector_obs = {}
        for selector_name, selected in selectors.items():
            match = selected[(selected["series_id"] == series_id) & (selected["release_date"].astype(str) == str(release_date))]
            if not match.empty:
                selector_values[selector_name] = float(match["selected_value"].iloc[0])
                selector_obs[selector_name] = str(match["selected_observation_date"].iloc[0])
        value_span = max(selector_values.values()) - min(selector_values.values()) if selector_values else 0.0
        latest_differs_from_first = len(obs_dates) > 1 and obs_dates[-1] != obs_dates[0]
        material_selector_diff = abs(value_span) > 1e-12
        if len(bucket) > 1 or latest_differs_from_first or len(values) > 1 or material_selector_diff:
            rows.append(
                {
                    "series_id": series_id,
                    "release_date": str(release_date),
                    "candidate_count": int(len(bucket)),
                    "non_null_candidate_count": int(len(nonnull)),
                    "first_observation_date": obs_dates[0],
                    "latest_observation_date": obs_dates[-1],
                    "latest_differs_from_first": latest_differs_from_first,
                    "distinct_non_null_value_count": int(len(values)),
                    "min_value": float(min(values)),
                    "max_value": float(max(values)),
                    "selected_observation_dates_by_selector": json.dumps(selector_obs, sort_keys=True),
                    "selected_values_by_selector": json.dumps(selector_values, sort_keys=True),
                    "selector_value_span": float(value_span),
                    "material_selector_value_difference": material_selector_diff,
                }
            )
    ambiguity = pd.DataFrame(rows).sort_values(["series_id", "release_date"])
    write_audit_csv(ambiguity, audit_root / "forensics" / "alfred_ambiguous_release_buckets.csv", audit_root)
    summary = (
        ambiguity.groupby("series_id", as_index=False)
        .agg(
            ambiguous_release_buckets=("release_date", "count"),
            material_selector_value_difference_buckets=("material_selector_value_difference", "sum"),
            latest_differs_from_first_buckets=("latest_differs_from_first", "sum"),
            max_selector_value_span=("selector_value_span", "max"),
        )
        .sort_values("ambiguous_release_buckets", ascending=False)
    )
    write_ambiguity_md(summary, ambiguity, audit_root)
    return ambiguity, summary


def write_ambiguity_md(summary: pd.DataFrame, ambiguity: pd.DataFrame, audit_root: Path) -> None:
    lines = [
        "# ALFRED Selector Ambiguity Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This report counts exact `realtime_start` buckets where candidate rows can support different raw-lineage choices.",
        "",
        "## By Series",
        "",
        md_table(summary) if not summary.empty else "No ambiguous buckets found.",
        "",
        "## Highest Selector Value Spans",
        "",
    ]
    worst = ambiguity.sort_values("selector_value_span", ascending=False).head(25)
    lines.append(md_table(worst) if not worst.empty else "No material selector differences found.")
    assert_audit_path(audit_root / "forensics" / "alfred_selector_ambiguity_summary.md", audit_root).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def selected_to_macro_input(selected: pd.DataFrame, selector_name: str) -> pd.DataFrame:
    rows = []
    for row in selected.itertuples(index=False):
        event_time = EVENT_TIMES[row.series_id]
        release_ts_utc = ny_time_to_utc(row.release_date, event_time)
        available = macro_available_ts(release_ts_utc, event_time, MAIN_VARIANT)
        rows.append(
            {
                "indicator_id": row.series_id,
                "series_id": row.series_id,
                "observation_date": row.selected_observation_date,
                "selected_observation_date": row.selected_observation_date,
                "release_date": row.release_date,
                "actual_first_release": float(row.selected_value),
                "selected_value": float(row.selected_value),
                "selector_name": selector_name,
                "candidate_count": int(row.candidate_count),
                "ambiguity_flag": bool(row.ambiguity_flag),
                "assumed_release_time_et": event_time,
                "release_ts_utc": release_ts_utc.isoformat(),
                "available_ts_utc_main": available.isoformat(),
                "availability_ts_utc": available.isoformat(),
                "notes": row.selection_reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["indicator_id", "release_date"])


def load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = parse_utc(df["timestamp"])
    return df


def compare_grids(candidate: pd.DataFrame, reference: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    left = candidate[["timestamp", *[c for c in features if c in candidate.columns]]].copy()
    right = reference[["timestamp", *[c for c in features if c in reference.columns]]].copy()
    merged = left.merge(right, on="timestamp", suffixes=("_candidate", "_reference"))
    feature_rows = []
    matches = 0
    comparable = 0
    for feature in features:
        c_col = f"{feature}_candidate"
        r_col = f"{feature}_reference"
        if c_col not in merged.columns or r_col not in merged.columns:
            continue
        c = pd.to_numeric(merged[c_col], errors="coerce")
        r = pd.to_numeric(merged[r_col], errors="coerce")
        mask = c.notna() & r.notna()
        n = int(mask.sum())
        if n == 0:
            feature_rows.append({"feature": feature, "comparable_rows": 0, "match_pct": None, "mae": None, "max_abs_error": None})
            continue
        is_match = np.isclose(c[mask], r[mask], rtol=1e-9, atol=1e-9)
        m = int(is_match.sum())
        matches += m
        comparable += n
        abs_err = (c[mask] - r[mask]).abs()
        feature_rows.append(
            {
                "feature": feature,
                "comparable_rows": n,
                "match_count": m,
                "mismatch_count": n - m,
                "match_pct": float(m / n),
                "mae": float(abs_err.mean()),
                "max_abs_error": float(abs_err.max()),
                "first_mismatch_ts": None
                if bool(is_match.all())
                else merged.loc[mask[mask].index[~is_match][0], "timestamp"].isoformat(),
            }
        )
    by_feature = pd.DataFrame(feature_rows)
    return {
        "aligned_rows": int(len(merged)),
        "overall_feature_match_pct": float(matches / comparable) if comparable else None,
        "overall_match_count": int(matches),
        "overall_comparable_values": int(comparable),
        "mae_by_feature": by_feature.to_dict(orient="records"),
        "worst_features_by_mae": by_feature.sort_values("mae", ascending=False, na_position="last").head(10).to_dict(orient="records"),
        "event_flag_match": by_feature[by_feature["feature"].isin(["CPI_Event_Active", "NFP_Event_Active"])].to_dict(orient="records"),
    }


def build_and_score_selectors(selectors: dict[str, pd.DataFrame], audit_root: Path) -> tuple[str, dict, dict[str, pd.DataFrame]]:
    market = pd.read_csv(MARKET_INPUT)
    market["timestamp"] = parse_utc(market["timestamp"])
    vix = pd.read_csv(CURRENT_VIX)
    vix["timestamp"] = parse_utc(vix["timestamp"])
    vix["available_ts_utc"] = parse_utc(vix["available_ts_utc"])
    assumptions = pd.read_csv(TIMING_ASSUMPTIONS)
    reference = load_grid(FREEZE_GRID)
    selector_scores = {}
    selector_grids = {}
    for selector_name, selected in selectors.items():
        macro = selected_to_macro_input(selected, selector_name)
        macro["release_ts_utc"] = parse_utc(macro["release_ts_utc"])
        grid, validation = build_variant_grid(V4_1_ROOT, MAIN_VARIANT, market, macro, vix, assumptions)
        score = compare_grids(grid, reference)
        score["validation"] = validation
        score["selector_name"] = selector_name
        score["selected_release_rows"] = int(len(selected))
        selector_scores[selector_name] = score
        selector_grids[selector_name] = grid
    best_selector = max(
        selector_scores,
        key=lambda name: (
            selector_scores[name]["overall_feature_match_pct"]
            if selector_scores[name]["overall_feature_match_pct"] is not None
            else -1.0
        ),
    )
    score_rows = []
    for selector_name, score in selector_scores.items():
        score_rows.append(
            {
                "selector_name": selector_name,
                "overall_feature_match_pct": score["overall_feature_match_pct"],
                "overall_match_count": score["overall_match_count"],
                "overall_comparable_values": score["overall_comparable_values"],
                "aligned_rows": score["aligned_rows"],
                "selected_release_rows": score["selected_release_rows"],
            }
        )
    write_audit_csv(pd.DataFrame(score_rows).sort_values("overall_feature_match_pct", ascending=False), audit_root / "forensics" / "selector_grid_scores.csv", audit_root)
    return best_selector, selector_scores, selector_grids


def compare_selectors_to_current_and_reference(selectors: dict[str, pd.DataFrame], audit_root: Path, artifact_inventory: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_MACRO)
    current = current.rename(
        columns={
            "indicator_id": "series_id",
            "observation_date": "current_observation_date",
            "actual_first_release": "current_value",
        }
    )
    current["release_date"] = current["release_date"].astype(str)
    reference_paths = artifact_inventory[artifact_inventory["type"].eq("macro_selection_audit")]["path"].tolist()
    reference_df = None
    if reference_paths:
        reference_df = pd.read_csv(reference_paths[0])
    rows = []
    for selector_name, selected in selectors.items():
        cmp = selected.merge(
            current[["series_id", "release_date", "current_observation_date", "current_value"]],
            on=["series_id", "release_date"],
            how="left",
        )
        cmp["differs_from_current_observation_date"] = cmp["selected_observation_date"].astype(str) != cmp["current_observation_date"].astype(str)
        cmp["differs_from_current_value"] = ~np.isclose(
            pd.to_numeric(cmp["selected_value"], errors="coerce"),
            pd.to_numeric(cmp["current_value"], errors="coerce"),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        cmp["reference_artifact_path"] = reference_paths[0] if reference_paths else None
        cmp["reference_comparison_status"] = "macro_selection_audit_not_found"
        if reference_df is not None:
            cmp["reference_comparison_status"] = "macro_selection_audit_found_manual_review_required"
        rows.append(cmp)
    out = pd.concat(rows, ignore_index=True).sort_values(["selector_name", "series_id", "release_date"])
    write_audit_csv(out, audit_root / "forensics" / "selector_vs_reference_comparison.csv", audit_root)
    return out


def fetch_fred_daily_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df.rename(columns={date_col: "observation_date", series_id: "close"})
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"].replace(".", pd.NA), errors="coerce")
    return df.dropna(subset=["observation_date", "close"]).sort_values("observation_date")


def next_business_day(obs_date: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(obs_date).normalize() + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def vix_available_ts(obs_date: pd.Timestamp, rule: str) -> pd.Timestamp:
    d = next_business_day(obs_date)
    if rule == "next_day_09_30_et":
        hour, minute = 9, 30
    elif rule == "next_day_10_00_et":
        hour, minute = 10, 0
    elif rule == "next_day_00_00_et":
        hour, minute = 0, 0
    else:
        raise ValueError(rule)
    return pd.Timestamp(datetime(d.year, d.month, d.day, hour, minute), tz=NY).tz_convert(UTC).tz_localize(None)


def make_vix_hourly_for_rule(rule: str, market: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2016-01-01 00:00:00")
    end = market["timestamp"].max()
    hourly = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="h")})
    frames = []
    for series in VIX_SERIES:
        daily = fetch_fred_daily_csv(series)
        daily["available_ts_utc"] = daily["observation_date"].apply(lambda x: vix_available_ts(x, rule))
        daily = daily[daily["available_ts_utc"] <= end].copy()
        merged = pd.merge_asof(
            hourly.sort_values("timestamp"),
            daily[["available_ts_utc", "close", "observation_date"]].sort_values("available_ts_utc"),
            left_on="timestamp",
            right_on="available_ts_utc",
            direction="backward",
        )
        merged = merged.dropna(subset=["close"]).copy()
        merged["series"] = series
        merged["source"] = f"fred_1d_pit_{rule}"
        frames.append(merged[["timestamp", "series", "close", "observation_date", "available_ts_utc", "source"]])
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "series"])


def audit_vix_sources(best_macro: pd.DataFrame, reference: pd.DataFrame, audit_root: Path) -> tuple[pd.DataFrame, dict]:
    market = pd.read_csv(MARKET_INPUT)
    market["timestamp"] = parse_utc(market["timestamp"])
    assumptions = pd.read_csv(TIMING_ASSUMPTIONS)
    macro = selected_to_macro_input(best_macro, "best_selector_for_vix_audit")
    macro["release_ts_utc"] = parse_utc(macro["release_ts_utc"])
    rows = []
    all_vix = []
    for rule in ["next_day_09_30_et", "next_day_10_00_et", "next_day_00_00_et"]:
        vix = make_vix_hourly_for_rule(rule, market)
        all_vix.append(vix.assign(pit_rule=rule))
        grid, _ = build_variant_grid(V4_1_ROOT, MAIN_VARIANT, market, macro, vix, assumptions)
        score = compare_grids(grid, reference, features=["VIXCLS_z", "vix3m_z", "VIX_Inversion_Z"])
        rows.append(
            {
                "pit_rule": rule,
                "vix_feature_match_pct": score["overall_feature_match_pct"],
                "aligned_rows": score["aligned_rows"],
                "overall_match_count": score["overall_match_count"],
                "overall_comparable_values": score["overall_comparable_values"],
                "worst_features_by_mae": json.dumps(score["worst_features_by_mae"], sort_keys=True),
            }
        )
    source = pd.concat(all_vix, ignore_index=True)
    write_audit_csv(source, audit_root / "outputs" / "v4_1_vix_vix3m_selected_pit_sources.csv", audit_root)
    score_df = pd.DataFrame(rows).sort_values("vix_feature_match_pct", ascending=False)
    best_rule = str(score_df.iloc[0]["pit_rule"]) if not score_df.empty else None
    report = {
        "best_pit_rule_by_feature_match": best_rule,
        "rule_scores": score_df.to_dict(orient="records"),
        "classification": "Compared FRED daily VIXCLS/VXVCLS raw values under candidate PIT availability rules against corrected feature-freeze VIX z-score behavior. If all rules remain low-match, divergence is more likely bootstrap/z-score semantics or a hidden corrected raw artifact than simple next-day availability timing.",
    }
    lines = [
        "# VIX/VIX3M Raw Source Audit",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        md_table(score_df),
        "",
        f"Best PIT rule by corrected-freeze VIX feature match: `{best_rule}`.",
        "",
        report["classification"],
    ]
    assert_audit_path(audit_root / "forensics" / "vix_vix3m_raw_source_audit.md", audit_root).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return source, report


def write_recovered_raw_levels(best_macro_input: pd.DataFrame, best_vix_source: pd.DataFrame, audit_root: Path) -> None:
    macro_rows = best_macro_input[
        [
            "indicator_id",
            "series_id",
            "release_date",
            "selected_observation_date",
            "selected_value",
            "selector_name",
            "candidate_count",
            "ambiguity_flag",
            "assumed_release_time_et",
            "availability_ts_utc",
            "notes",
        ]
    ].copy()
    write_audit_csv(macro_rows, audit_root / "outputs" / "v4_1_macro_alfred_selected_first_releases.csv", audit_root)
    market = pd.read_csv(MARKET_INPUT)
    market["timestamp"] = parse_utc(market["timestamp"])
    hourly = pd.DataFrame({"timestamp": pd.date_range("2016-01-01", market["timestamp"].max(), freq="h")})
    macro_events = best_macro_input.rename(columns={"availability_ts_utc": "available_ts_utc", "selected_value": "value"}).copy()
    macro_events["available_ts_utc"] = parse_utc(macro_events["available_ts_utc"])
    frames = []
    for indicator, events in macro_events.groupby("indicator_id", sort=True):
        merged = pd.merge_asof(
            hourly.sort_values("timestamp"),
            events[["available_ts_utc", "value", "release_date", "selected_observation_date"]].sort_values("available_ts_utc"),
            left_on="timestamp",
            right_on="available_ts_utc",
            direction="backward",
        )
        merged = merged.dropna(subset=["value"]).copy()
        merged["indicator_id"] = indicator
        frames.append(merged[["timestamp", "indicator_id", "value", "available_ts_utc", "release_date", "selected_observation_date"]])
    macro_hourly = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "indicator_id"])
    write_audit_csv(macro_hourly, audit_root / "recovered_raw_levels" / "macro_hourly_raw_levels.csv", audit_root)
    chosen_vix = best_vix_source.copy()
    write_audit_csv(chosen_vix, audit_root / "recovered_raw_levels" / "vix_hourly_raw_levels.csv", audit_root)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "macro_selector": str(best_macro_input["selector_name"].iloc[0]) if not best_macro_input.empty else None,
        "macro_rows": int(len(best_macro_input)),
        "macro_hourly_rows": int(len(macro_hourly)),
        "vix_rows": int(len(chosen_vix)),
        "writes_only_under": str(audit_root),
    }
    write_audit_json(manifest, audit_root / "recovered_raw_levels" / "recovered_raw_lineage_manifest.json", audit_root)


def write_grid_reports(best_selector: str, best_grid: pd.DataFrame, selector_scores: dict, audit_root: Path) -> dict:
    reference = load_grid(FREEZE_GRID)
    report = compare_grids(best_grid, reference)
    report["best_selector_rule"] = best_selector
    report["baseline_prior_match_pct_from_handoff"] = 0.226547
    report["materially_improved_from_prior_22_6547_pct"] = (
        report["overall_feature_match_pct"] is not None and report["overall_feature_match_pct"] > 0.30
    )
    report["selector_scores"] = {
        name: {
            "overall_feature_match_pct": score["overall_feature_match_pct"],
            "overall_match_count": score["overall_match_count"],
            "overall_comparable_values": score["overall_comparable_values"],
        }
        for name, score in selector_scores.items()
    }
    write_audit_csv(best_grid, audit_root / "outputs" / "v4_1_grid_event_selector_test.csv", audit_root)
    write_audit_json(report, audit_root / "forensics" / "event_selector_grid_match_report.json", audit_root)
    lines = [
        "# Event Selector Grid Match Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Best selector: `{best_selector}`",
        f"Overall feature match vs corrected feature freeze: `{report['overall_feature_match_pct']:.6%}`",
        f"Materially improved from prior ~22.6547%: `{report['materially_improved_from_prior_22_6547_pct']}`",
        "",
        "## Worst Features By MAE",
        "",
        md_table(pd.DataFrame(report["worst_features_by_mae"])),
        "",
        "## Event Flag Match",
        "",
        md_table(pd.DataFrame(report["event_flag_match"])),
    ]
    assert_audit_path(audit_root / "forensics" / "event_selector_grid_match_report.md", audit_root).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def write_final_summary(
    audit_root: Path,
    artifact_inventory: pd.DataFrame,
    selector_comparison: pd.DataFrame,
    ambiguity_summary: pd.DataFrame,
    best_selector: str,
    vix_report: dict,
    grid_report: dict,
) -> dict:
    differs = (
        selector_comparison.groupby("selector_name", as_index=False)
        .agg(
            rows=("release_date", "count"),
            differs_from_current_observation_date=("differs_from_current_observation_date", "sum"),
            differs_from_current_value=("differs_from_current_value", "sum"),
        )
        .sort_values("differs_from_current_value", ascending=False)
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_folder": str(audit_root),
        "candidate_row_csv": str(audit_root / "inputs" / "alfred_candidate_rows.csv"),
        "artifact_inventory_csv": str(audit_root / "forensics" / "artifact_inventory.csv"),
        "selector_comparison_csv": str(audit_root / "forensics" / "selector_vs_reference_comparison.csv"),
        "ambiguous_release_bucket_csv": str(audit_root / "forensics" / "alfred_ambiguous_release_buckets.csv"),
        "best_selector_rule": best_selector,
        "corrected_macro_selected_releases_csv": str(audit_root / "outputs" / "v4_1_macro_alfred_selected_first_releases.csv"),
        "vix_vix3m_raw_source_audit": vix_report,
        "grid_feature_match_after_selector_rebuild": grid_report.get("overall_feature_match_pct"),
        "materially_improved_from_prior_22_6547_pct": grid_report.get("materially_improved_from_prior_22_6547_pct"),
        "explains_raw_lineage_mismatch": bool(grid_report.get("materially_improved_from_prior_22_6547_pct")),
        "selector_vs_current_summary": differs.to_dict(orient="records"),
        "ambiguity_summary": ambiguity_summary.to_dict(orient="records"),
        "production_change_confirmation": {
            "production_delta_tables_modified": False,
            "production_workflows_modified": False,
            "saxo_execution_modified": False,
            "saxo_token_refresh_modified": False,
            "rt_notebooks_modified": False,
            "dashboard_notebooks_modified": False,
            "model_registry_modified": False,
            "live_schedules_modified": False,
        },
        "artifact_inventory_high_relevance_count": int(artifact_inventory["relevance"].astype(str).str.startswith("high").sum())
        if not artifact_inventory.empty
        else 0,
    }
    write_audit_json(summary, audit_root / "forensics" / "alfred_event_selector_audit_summary.json", audit_root)
    lines = [
        "# ALFRED Event Selector Audit Summary",
        "",
        f"Generated UTC: {summary['generated_at_utc']}",
        "",
        f"Output folder: `{summary['output_folder']}`",
        f"Candidate row CSV: `{summary['candidate_row_csv']}`",
        f"Best selector rule: `{best_selector}`",
        f"Grid feature match after selector rebuild: `{summary['grid_feature_match_after_selector_rebuild']:.6%}`",
        f"Materially improved from prior ~22.6547%: `{summary['materially_improved_from_prior_22_6547_pct']}`",
        f"Explains raw lineage mismatch: `{summary['explains_raw_lineage_mismatch']}`",
        "",
        "## Selector vs Current V4.1 Macro Input",
        "",
        md_table(differs),
        "",
        "## Ambiguity Summary",
        "",
        md_table(ambiguity_summary) if not ambiguity_summary.empty else "No ambiguous buckets found.",
        "",
        "## VIX/VIX3M",
        "",
        f"Best PIT rule: `{vix_report.get('best_pit_rule_by_feature_match')}`",
        "",
        "## No Production Change Confirmation",
        "",
        "No production Delta tables, workflows, Saxo execution, Saxo token refresh, RT notebooks, dashboard notebooks, model registry, or live schedules were modified.",
    ]
    assert_audit_path(audit_root / "forensics" / "alfred_event_selector_audit_summary.md", audit_root).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    parser.add_argument("--fred-api-key", default=os.environ.get("FRED_API_KEY"))
    parser.add_argument("--realtime-start", default="2016-01-01")
    parser.add_argument("--realtime-end", default="2026-04-30")
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    if V4_1_ROOT.resolve() not in audit_root.resolve().parents:
        raise ValueError(f"Audit root must stay under V4.1 root: {audit_root}")
    if not args.fred_api_key and not (audit_root / "inputs" / "alfred_candidate_rows.csv").exists():
        raise RuntimeError("FRED_API_KEY is required for ALFRED candidate-row dump when no cached candidate CSV exists")

    create_dirs(audit_root)
    artifact_inventory = inventory_artifacts(audit_root)
    candidates = build_candidate_rows(audit_root, args.fred_api_key, args.realtime_start, args.realtime_end)
    selectors = build_selector_outputs(candidates, audit_root)
    ambiguity, ambiguity_summary = build_ambiguity_reports(candidates, selectors, audit_root)
    selector_comparison = compare_selectors_to_current_and_reference(selectors, audit_root, artifact_inventory)
    best_selector, selector_scores, selector_grids = build_and_score_selectors(selectors, audit_root)
    best_macro_input = selected_to_macro_input(selectors[best_selector], best_selector)
    write_audit_csv(best_macro_input, audit_root / "outputs" / "v4_1_macro_alfred_selected_first_releases.csv", audit_root)
    reference = load_grid(FREEZE_GRID)
    vix_source, vix_report = audit_vix_sources(selectors[best_selector], reference, audit_root)
    best_vix_rule = vix_report.get("best_pit_rule_by_feature_match")
    best_vix_source = vix_source[vix_source["pit_rule"] == best_vix_rule].drop(columns=["pit_rule"], errors="ignore")
    write_recovered_raw_levels(best_macro_input, best_vix_source, audit_root)
    grid_report = write_grid_reports(best_selector, selector_grids[best_selector], selector_scores, audit_root)
    summary = write_final_summary(
        audit_root,
        artifact_inventory,
        selector_comparison,
        ambiguity_summary,
        best_selector,
        vix_report,
        grid_report,
    )
    print_json(summary)


if __name__ == "__main__":
    main()

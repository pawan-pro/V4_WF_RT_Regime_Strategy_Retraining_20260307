#!/usr/bin/env python3
"""Recreate VIX/VIX3M from FRED using exact V4 fred_1d logic.

Reads current FRED observations and isolated legacy/default CSV artifacts,
then writes only under the V4.1 CSV forensic output folder. No Delta writes,
workflow updates, model registry operations, or model training.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
BASE = Path(f"/Volumes/{CATALOG}/v4/market_data")
CSV_BASE = BASE / "v4_1_csv"
OUTPUT_ROOT = CSV_BASE / "vix_vix3m_exact_recreation_2026-04-30"
LEGACY_VIX_ROOT = CSV_BASE / "legacy_default_vix_source_rebuild_2026-04-30"
DEFAULT_VIX_CSV = LEGACY_VIX_ROOT / "inputs" / "default_bronze_bt_vix_1h.csv"
DEFAULT_VIX3M_CSV = LEGACY_VIX_ROOT / "inputs" / "default_bronze_bt_vix3m_1h.csv"
FULL_LEGACY_ROOT = CSV_BASE / "full_legacy_grid_clone_2026-04-30"
DEFAULT_MARKET_CSV = FULL_LEGACY_ROOT / "inputs" / "default_bronze_bt_market_1h_spy_qqq.csv"
MACRO_SELECTION_AUDIT = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "macro_selection_audit.csv"
)
REFERENCE_GRID = (
    CSV_BASE
    / "forensic_feature_freeze_ladder_2026-04-30"
    / "grids"
    / "D_full_feature_freeze"
    / "v4_1_silver_grid.csv"
)
VIX_NOTEBOOK = Path("01_Processing/01_Ingestion/05_BT_Portara_VIX_Ingestion.py")

INDICATORS = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT"]
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
FINAL_COLUMNS = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]


def ensure_dirs() -> None:
    for subdir in ["inputs", "grids", "forensics", "manifests", "code"]:
        (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    try:
        out[column] = out[column].dt.tz_localize(None)
    except Exception:
        pass
    return out


def fred_api_key() -> str:
    if not VIX_NOTEBOOK.exists():
        return "f52bdf7341f23d0458820ecd806178db"
    text = VIX_NOTEBOOK.read_text(encoding="utf-8")
    match = re.search(r'FRED_API_KEY\s*=\s*"([^"]+)"', text)
    if not match:
        raise RuntimeError("Could not locate FRED_API_KEY in V4 ingestion notebook")
    return match.group(1)


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": fred_api_key(),
            "file_type": "json",
            "observation_start": "2016-01-01",
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for obs in payload.get("observations", []):
        value = obs.get("value")
        parsed = pd.to_numeric(value, errors="coerce")
        rows.append(
            {
                "date": obs.get("date"),
                "realtime_start": obs.get("realtime_start"),
                "realtime_end": obs.get("realtime_end"),
                "value": value,
                "close": parsed,
                "series_id": series_id,
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def fred_pit_to_hourly(raw: pd.DataFrame, series_id: str, symbol_label: str) -> pd.DataFrame:
    """Exact V4 fred_1d branch: shift one observed row and expand to 24 hours."""
    pdf = raw[["date", "close"]].copy().sort_values("date").reset_index(drop=True)
    pdf["close"] = pdf["close"].shift(1)
    pdf = pdf.dropna(subset=["close"]).reset_index(drop=True)
    records = []
    recreated_ts = pd.Timestamp.utcnow()
    for row in pdf.itertuples(index=False):
        day = pd.Timestamp(row.date).normalize()
        close = float(row.close)
        for hour in range(24):
            records.append(
                {
                    "timestamp": day + pd.Timedelta(hours=hour),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 0.0,
                    "source_contract": series_id,
                    "symbol": symbol_label,
                    "source": f"fred_{series_id.lower()}_pit_1d",
                    "year_partition": int((day + pd.Timedelta(hours=hour)).year),
                    "recreated_ts": recreated_ts,
                }
            )
    return pd.DataFrame(records)


def recreate_fred_sources() -> dict:
    raw_vix = fetch_fred_series("VIXCLS")
    raw_vix3m = fetch_fred_series("VXVCLS")
    vix = fred_pit_to_hourly(raw_vix, "VIXCLS", "VX")
    vix3m = fred_pit_to_hourly(raw_vix3m, "VXVCLS", "VIX3M")

    raw_vix_path = OUTPUT_ROOT / "inputs" / "raw_fred_vixcls_daily.csv"
    raw_vix3m_path = OUTPUT_ROOT / "inputs" / "raw_fred_vxvcls_daily.csv"
    vix_path = OUTPUT_ROOT / "inputs" / "recreated_fred_vix_1h_v4_logic.csv"
    vix3m_path = OUTPUT_ROOT / "inputs" / "recreated_fred_vix3m_1h_v4_logic.csv"
    write_csv(raw_vix, raw_vix_path)
    write_csv(raw_vix3m, raw_vix3m_path)
    write_csv(vix, vix_path)
    write_csv(vix3m, vix3m_path)
    return {
        "raw_vix_path": raw_vix_path,
        "raw_vix3m_path": raw_vix3m_path,
        "vix_path": vix_path,
        "vix3m_path": vix3m_path,
        "raw_vix_rows": len(raw_vix),
        "raw_vix3m_rows": len(raw_vix3m),
        "vix_rows": len(vix),
        "vix3m_rows": len(vix3m),
    }


def raw_compare(
    recreated_path: Path,
    default_path: Path,
    series_label: str,
    out_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rec = normalize_timestamp(pd.read_csv(recreated_path), "timestamp")
    default = normalize_timestamp(pd.read_csv(default_path), "timestamp")
    rec_cols = ["timestamp", "close", "source", "source_contract", "symbol"]
    def_cols = [c for c in rec_cols if c in default.columns]
    cmp = rec[rec_cols].merge(
        default[def_cols],
        on="timestamp",
        how="outer",
        suffixes=("_recreated", "_default"),
        indicator=True,
    )
    both = cmp[cmp["_merge"].eq("both")].copy()
    if not both.empty:
        both["close_recreated"] = pd.to_numeric(both["close_recreated"], errors="coerce")
        both["close_default"] = pd.to_numeric(both["close_default"], errors="coerce")
        both["close_abs_diff"] = (both["close_recreated"] - both["close_default"]).abs()
        both["close_exact_match"] = both["close_abs_diff"].le(1e-12)
        for col in ["source", "source_contract", "symbol"]:
            l = f"{col}_recreated"
            r = f"{col}_default"
            if l in both.columns and r in both.columns:
                both[f"{col}_match"] = both[l].astype(str).eq(both[r].astype(str))
    mismatches = both[~both["close_exact_match"]].head(100).copy() if not both.empty else pd.DataFrame()
    mismatches.to_csv(OUTPUT_ROOT / "forensics" / f"{series_label.lower()}_first_100_raw_mismatches.csv", index=False)
    summary = {
        "series": series_label,
        "recreated_path": str(recreated_path),
        "default_path": str(default_path),
        "recreated_rows": int(len(rec)),
        "default_rows": int(len(default)),
        "overlap_rows": int(len(both)),
        "missing_in_recreated": int(cmp["_merge"].eq("right_only").sum()),
        "missing_in_default": int(cmp["_merge"].eq("left_only").sum()),
        "recreated_min_timestamp": str(rec["timestamp"].min()) if not rec.empty else None,
        "recreated_max_timestamp": str(rec["timestamp"].max()) if not rec.empty else None,
        "default_min_timestamp": str(default["timestamp"].min()) if not default.empty else None,
        "default_max_timestamp": str(default["timestamp"].max()) if not default.empty else None,
        "close_exact_match_count": int(both["close_exact_match"].sum()) if not both.empty else 0,
        "close_exact_match_rate": float(both["close_exact_match"].mean()) if not both.empty else None,
        "close_mae": float(both["close_abs_diff"].mean()) if not both.empty else None,
        "close_max_abs_diff": float(both["close_abs_diff"].max()) if not both.empty else None,
        "first_close_mismatch_timestamp": str(both.loc[~both["close_exact_match"], "timestamp"].iloc[0]) if (not both.empty and (~both["close_exact_match"]).any()) else None,
        "source_match_rate": float(both["source_match"].mean()) if "source_match" in both else None,
        "source_contract_match_rate": float(both["source_contract_match"].mean()) if "source_contract_match" in both else None,
        "symbol_match_rate": float(both["symbol_match"].mean()) if "symbol_match" in both else None,
    }
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    return cmp, both, summary


def daily_revision_diagnosis() -> pd.DataFrame:
    rows = []
    for series_id, label, default_path, raw_path in [
        ("VIXCLS", "vix", DEFAULT_VIX_CSV, OUTPUT_ROOT / "inputs" / "raw_fred_vixcls_daily.csv"),
        ("VXVCLS", "vix3m", DEFAULT_VIX3M_CSV, OUTPUT_ROOT / "inputs" / "raw_fred_vxvcls_daily.csv"),
    ]:
        raw = pd.read_csv(raw_path)
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        shifted = raw[["date", "close"]].copy()
        shifted["assigned_date"] = shifted["date"].shift(-1)
        shifted = shifted.dropna(subset=["assigned_date"]).rename(columns={"close": "current_fred_source_close", "date": "source_observation_date"})

        default = normalize_timestamp(pd.read_csv(default_path), "timestamp")
        default_daily = (
            default.assign(assigned_date=default["timestamp"].dt.normalize())
            .groupby("assigned_date", as_index=False)
            .agg(default_close=("close", "first"), default_hour_rows=("timestamp", "count"))
        )
        cmp = default_daily.merge(shifted, on="assigned_date", how="outer", indicator=True)
        for _, row in cmp.iterrows():
            diff = np.nan
            classification = "unknown"
            if row["_merge"] == "left_only":
                classification = "missing_current_fred_assigned_date"
            elif row["_merge"] == "right_only":
                classification = "missing_legacy_default_assigned_date"
            else:
                diff = abs(float(row["default_close"]) - float(row["current_fred_source_close"]))
                if diff <= 1e-12 and int(row["default_hour_rows"]) == 24:
                    classification = "exact_match"
                elif diff > 1e-12:
                    classification = "FRED current revision mismatch"
                elif int(row["default_hour_rows"]) != 24:
                    classification = "timestamp expansion mismatch"
            rows.append(
                {
                    "series_id": series_id,
                    "series_label": label,
                    "assigned_date": row.get("assigned_date", pd.NaT),
                    "legacy_default_close": row.get("default_close", np.nan),
                    "current_fred_source_observation_date": row.get("source_observation_date", pd.NaT),
                    "current_fred_source_close": row.get("current_fred_source_close", np.nan),
                    "abs_diff": diff,
                    "default_hour_rows": row.get("default_hour_rows", np.nan),
                    "classification": classification,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_ROOT / "forensics" / "fred_revision_or_expansion_diagnosis.csv", index=False)
    return out


def build_grid_with_recreated_vix() -> pd.DataFrame:
    market = normalize_timestamp(pd.read_csv(DEFAULT_MARKET_CSV), "timestamp")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market_wide = (
        market.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .rename(columns={"SPY": "SPY_close", "QQQ": "QQQ_close"})
        .reset_index()
        .sort_values("timestamp")
    )
    end_date = market_wide["timestamp"].max()
    backbone = pd.DataFrame({"timestamp": pd.date_range(start=pd.Timestamp("2016-01-01 00:00:00"), end=end_date, freq="h")})

    macro = pd.read_csv(MACRO_SELECTION_AUDIT)
    macro["release_datetime"] = pd.to_datetime(macro["release_datetime"], errors="coerce")
    macro["timestamp"] = macro["release_datetime"]
    macro["actual"] = pd.to_numeric(macro["actual"], errors="coerce")
    macro = macro[["indicator_id", "timestamp", "actual"]].dropna(subset=["indicator_id", "timestamp"])
    macro = macro.drop_duplicates(subset=["timestamp", "indicator_id"], keep="last")
    macro["join_ts"] = (macro["timestamp"] + pd.Timedelta(hours=2)).dt.floor("h")
    macro_prepared = (
        macro.pivot_table(index="join_ts", columns="indicator_id", values="actual", aggfunc="mean")
        .reset_index()
        .rename(columns={"join_ts": "timestamp"})
    )

    vix = normalize_timestamp(pd.read_csv(OUTPUT_ROOT / "inputs" / "recreated_fred_vix_1h_v4_logic.csv"), "timestamp")
    vix = vix.drop_duplicates(subset=["timestamp"], keep="last")
    vix["VIXCLS"] = pd.to_numeric(vix["close"], errors="coerce")
    vix = vix[["timestamp", "VIXCLS"]]

    vix3m = normalize_timestamp(pd.read_csv(OUTPUT_ROOT / "inputs" / "recreated_fred_vix3m_1h_v4_logic.csv"), "timestamp")
    vix3m = vix3m.drop_duplicates(subset=["timestamp"], keep="last")
    vix3m["vix3m"] = pd.to_numeric(vix3m["close"], errors="coerce")
    vix3m = vix3m[["timestamp", "vix3m"]]

    silver = backbone.merge(macro_prepared, on="timestamp", how="left")
    silver = silver.merge(market_wide[["timestamp", "SPY_close", "QQQ_close"]], on="timestamp", how="left")
    silver = silver.merge(vix, on="timestamp", how="left")
    silver = silver.merge(vix3m, on="timestamp", how="left")
    for col in INDICATORS:
        if col not in silver.columns:
            silver[col] = np.nan
    fill_cols = INDICATORS + ["VIXCLS", "vix3m", "SPY_close", "QQQ_close"]
    silver = silver.sort_values("timestamp").reset_index(drop=True)
    silver[fill_cols] = silver[fill_cols].ffill()

    for col in INDICATORS + ["VIXCLS", "vix3m"]:
        rolling_mean = silver[col].rolling(8761, min_periods=1).mean()
        rolling_std = silver[col].rolling(8761, min_periods=2).std()
        denominator = rolling_std.where(rolling_std != 0, 1)
        silver[f"{col}_z"] = (silver[col] - rolling_mean) / denominator

    silver["CPI_Shock"] = (silver["CPIAUCSL"] - silver["CPIAUCSL"].shift(1)).abs()
    silver["NFP_Shock"] = (silver["PAYEMS"] - silver["PAYEMS"].shift(1)).abs()
    silver["CPI_Event_Active"] = (silver["CPI_Shock"] > 0).astype(int).rolling(49, min_periods=1).max()
    silver["NFP_Event_Active"] = (silver["NFP_Shock"] > 0).astype(int).rolling(49, min_periods=1).max()
    silver["VIX_Inversion_Z"] = silver["VIXCLS_z"] - silver["vix3m_z"]
    silver["Price_Mom_24H"] = silver["SPY_close"].shift(1) / silver["SPY_close"].shift(25) - 1
    final = silver[FINAL_COLUMNS].copy()
    final = final[final["SPY_close"].notna()].reset_index(drop=True)
    out = OUTPUT_ROOT / "grids" / "full_legacy_grid_with_recreated_fred_vix.csv"
    write_csv(final, out)
    return final


def compare_grids(candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ref = normalize_timestamp(pd.read_csv(REFERENCE_GRID), "timestamp").sort_values("timestamp")
    candidate = candidate.sort_values("timestamp")
    merged = candidate[FINAL_COLUMNS].merge(ref[FINAL_COLUMNS], on="timestamp", how="inner", suffixes=("_candidate", "_reference"))
    rows = []
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        c = pd.to_numeric(merged[f"{col}_candidate"], errors="coerce")
        r = pd.to_numeric(merged[f"{col}_reference"], errors="coerce")
        mask = c.notna() & r.notna()
        if not mask.any():
            rows.append({"feature": col, "is_14_feature": col in FEATURES, "comparable_rows": 0, "near_match_rate": np.nan, "mae": np.nan, "max_abs_diff": np.nan, "first_divergence_timestamp": None})
            continue
        diff = (c[mask] - r[mask]).abs()
        near = diff <= 1e-9
        first = None
        if not bool(near.all()):
            first = str(merged.loc[diff[~near].index[0], "timestamp"])
        rows.append(
            {
                "feature": col,
                "is_14_feature": col in FEATURES,
                "comparable_rows": int(mask.sum()),
                "near_match_count": int(near.sum()),
                "near_match_rate": float(near.mean()),
                "mae": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "first_divergence_timestamp": first,
            }
        )
    report = pd.DataFrame(rows)
    feature_report = report[report["is_14_feature"]]
    summary = {
        "overall_14_feature_match": float(feature_report["near_match_count"].sum() / feature_report["comparable_rows"].sum()),
        "VIXCLS_z_match": float(report.loc[report["feature"].eq("VIXCLS_z"), "near_match_rate"].iloc[0]),
        "vix3m_z_match": float(report.loc[report["feature"].eq("vix3m_z"), "near_match_rate"].iloc[0]),
        "VIX_Inversion_Z_match": float(report.loc[report["feature"].eq("VIX_Inversion_Z"), "near_match_rate"].iloc[0]),
        "macro_feature_match": float(feature_report[feature_report["feature"].isin([f"{x}_z" for x in INDICATORS])]["near_match_count"].sum() / feature_report[feature_report["feature"].isin([f"{x}_z" for x in INDICATORS])]["comparable_rows"].sum()),
        "Price_Mom_24H_match": float(report.loc[report["feature"].eq("Price_Mom_24H"), "near_match_rate"].iloc[0]),
        "CPI_Event_Active_match": float(report.loc[report["feature"].eq("CPI_Event_Active"), "near_match_rate"].iloc[0]),
        "NFP_Event_Active_match": float(report.loc[report["feature"].eq("NFP_Event_Active"), "near_match_rate"].iloc[0]),
        "candidate_rows": int(len(candidate)),
        "reference_rows": int(len(ref)),
        "inner_rows": int(len(merged)),
    }
    write_csv(report, OUTPUT_ROOT / "forensics" / "grid_with_recreated_fred_vix_match_report.csv")
    return report, summary


def classify_mismatch(vix_summary: dict, vix3m_summary: dict, diagnosis: pd.DataFrame) -> tuple[bool, str]:
    exact = (
        vix_summary["close_exact_match_rate"] == 1.0
        and vix3m_summary["close_exact_match_rate"] == 1.0
        and vix_summary["missing_in_recreated"] == 0
        and vix_summary["missing_in_default"] == 0
        and vix3m_summary["missing_in_recreated"] == 0
        and vix3m_summary["missing_in_default"] == 0
    )
    if exact:
        return True, "exact_match"
    counts = diagnosis["classification"].value_counts().to_dict()
    non_exact = {k: v for k, v in counts.items() if k != "exact_match"}
    if not non_exact:
        return False, "metadata_or_timestamp_membership_mismatch_only"
    ordered = sorted(non_exact.items(), key=lambda kv: kv[1], reverse=True)
    return False, ", ".join([f"{k}: {v}" for k, v in ordered])


def write_summary(recreation: dict, vix_summary: dict, vix3m_summary: dict, grid_summary: dict, exact: bool, classification: str, replay_run: bool) -> None:
    lines = [
        "# VIX/VIX3M Exact FRED Recreation Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output folder: `{OUTPUT_ROOT}`",
        f"- Recreated VIX CSV: `{recreation['vix_path']}`",
        f"- Recreated VIX3M CSV: `{recreation['vix3m_path']}`",
        f"- Raw FRED VIXCLS daily: `{recreation['raw_vix_path']}`",
        f"- Raw FRED VXVCLS daily: `{recreation['raw_vix3m_path']}`",
        "",
        "## Raw Source Comparison",
        "",
        f"- VIX close exact match rate: `{vix_summary['close_exact_match_rate']:.6%}`",
        f"- VIX3M close exact match rate: `{vix3m_summary['close_exact_match_rate']:.6%}`",
        f"- VIX missing in recreated/default: `{vix_summary['missing_in_recreated']}` / `{vix_summary['missing_in_default']}`",
        f"- VIX3M missing in recreated/default: `{vix3m_summary['missing_in_recreated']}` / `{vix3m_summary['missing_in_default']}`",
        f"- Recreated FRED VIX matches legacy default exactly: `{exact}`",
        f"- Mismatch classification: `{classification}`",
        "",
        "## Grid Feature Match",
        "",
        f"- Overall 14-feature match: `{grid_summary['overall_14_feature_match']:.6%}`",
        f"- VIXCLS_z match: `{grid_summary['VIXCLS_z_match']:.6%}`",
        f"- vix3m_z match: `{grid_summary['vix3m_z_match']:.6%}`",
        f"- VIX_Inversion_Z match: `{grid_summary['VIX_Inversion_Z_match']:.6%}`",
        f"- Macro feature match: `{grid_summary['macro_feature_match']:.6%}`",
        f"- Price_Mom_24H match: `{grid_summary['Price_Mom_24H_match']:.6%}`",
        f"- CPI_Event_Active match: `{grid_summary['CPI_Event_Active_match']:.6%}`",
        f"- NFP_Event_Active match: `{grid_summary['NFP_Event_Active_match']:.6%}`",
        "",
        "## Decision",
        "",
    ]
    if exact:
        lines.append("Legacy default VIX/VIX3M is reproducible from current FRED using the exact V4 `fred_1d` logic. Prior V4.1 mismatch was caused by the V4.1 implementation/source semantics, not FRED data revision.")
    else:
        lines.append("Legacy default VIX/VIX3M is not exactly reproducible from current FRED using the exact V4 `fred_1d` logic. See `fred_revision_or_expansion_diagnosis.csv` for the mismatch classification.")
    lines += [
        "",
        f"- Model replay run: `{replay_run}`",
        "- Model replay note: skipped because this task explicitly said not to train models.",
        "",
        "## No Production Change Confirmation",
        "",
        "This audit read current FRED observations and isolated CSV artifacts only. It wrote only under the V4.1 CSV forensic output folder. No production Delta tables, Saxo execution, token refresh, RT notebooks, dashboard notebooks, workflows, live schedules, model registry artifacts, or models were modified.",
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "forensics" / "vix_raw_recreation_summary.md")
    write_text("\n".join(lines), OUTPUT_ROOT / "forensics" / "grid_with_recreated_fred_vix_match_summary.md")


def write_manifest(recreation: dict, exact: bool, classification: str) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": OUTPUT_ROOT,
        "v4_notebook_logic_source": VIX_NOTEBOOK,
        "inputs": {
            "default_vix_csv": DEFAULT_VIX_CSV,
            "default_vix3m_csv": DEFAULT_VIX3M_CSV,
            "default_market_csv": DEFAULT_MARKET_CSV,
            "macro_selection_audit": MACRO_SELECTION_AUDIT,
            "reference_grid": REFERENCE_GRID,
        },
        "outputs": {k: v for k, v in recreation.items() if str(k).endswith("path")},
        "sha256": {
            "default_vix_csv": sha256_file(DEFAULT_VIX_CSV),
            "default_vix3m_csv": sha256_file(DEFAULT_VIX3M_CSV),
            "recreated_vix_csv": sha256_file(recreation["vix_path"]),
            "recreated_vix3m_csv": sha256_file(recreation["vix3m_path"]),
            "raw_fred_vixcls_daily": sha256_file(recreation["raw_vix_path"]),
            "raw_fred_vxvcls_daily": sha256_file(recreation["raw_vix3m_path"]),
        },
        "legacy_default_vix_reproducible_from_current_fred_v4_logic": exact,
        "mismatch_classification": classification,
        "production_delta_writes": False,
        "model_training": False,
        "workflow_or_registry_changes": False,
    }
    write_json(payload, OUTPUT_ROOT / "manifests" / "runtime_manifest.json")


def main() -> None:
    ensure_dirs()
    recreation = recreate_fred_sources()
    _, _, vix_summary = raw_compare(
        recreation["vix_path"],
        DEFAULT_VIX_CSV,
        "vix",
        OUTPUT_ROOT / "forensics" / "vix_raw_recreated_vs_default_comparison.csv",
    )
    _, _, vix3m_summary = raw_compare(
        recreation["vix3m_path"],
        DEFAULT_VIX3M_CSV,
        "vix3m",
        OUTPUT_ROOT / "forensics" / "vix3m_raw_recreated_vs_default_comparison.csv",
    )
    diagnosis = daily_revision_diagnosis()
    exact, classification = classify_mismatch(vix_summary, vix3m_summary, diagnosis)
    grid = build_grid_with_recreated_vix()
    grid_report, grid_summary = compare_grids(grid)
    replay_run = False
    write_summary(recreation, vix_summary, vix3m_summary, grid_summary, exact, classification, replay_run)
    write_json(
        {
            "raw_vix_summary": vix_summary,
            "raw_vix3m_summary": vix3m_summary,
            "grid_summary": grid_summary,
            "legacy_default_vix_reproducible_from_current_fred_v4_logic": exact,
            "mismatch_classification": classification,
            "model_replay_run": replay_run,
            "model_replay_skip_reason": "Task explicitly said do not train models.",
        },
        OUTPUT_ROOT / "forensics" / "vix_exact_recreation_result.json",
    )
    write_manifest(recreation, exact, classification)
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "vix_csv": str(recreation["vix_path"]),
                "vix3m_csv": str(recreation["vix3m_path"]),
                "vix_exact": exact,
                "classification": classification,
                "grid_feature_match": grid_summary["overall_14_feature_match"],
                "model_replay_run": replay_run,
                "no_production_change_confirmation": True,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

"""Build a faithful legacy corrected-grid clone from isolated CSV artifacts.

Reads default market/VIX tables and legacy first-release CSVs, writes only under
the isolated V4.1 full_legacy_grid_clone audit folder, and does not train
models or mutate production V4 assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import SparkSession


V4_1_CODE_DIR = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
if V4_1_CODE_DIR.exists() and str(V4_1_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_1_CODE_DIR))

from v4_1_common import FEATURES, parse_utc


CATALOG = "pawan_quant_research_1"
V4_1_ROOT = Path(f"/Volumes/{CATALOG}/v4/market_data/v4_1_csv")
AUDIT_ROOT = V4_1_ROOT / "full_legacy_grid_clone_2026-04-30"
DEFAULT_MARKET_TABLE = f"{CATALOG}.default.bronze_bt_market_1h"
DEFAULT_FIRST_RELEASE_ROOT = Path(f"/Volumes/{CATALOG}/default/market_data")
V4_FIRST_RELEASE_ROOT = Path(f"/Volumes/{CATALOG}/v4/market_data")
CORRECTED_MACRO_SELECTION_AUDIT = Path(
    f"/Volumes/{CATALOG}/v4/market_data/comparison/"
    "2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/macro_selection_audit.csv"
)
DEFAULT_VIX_REBUILD_ROOT = V4_1_ROOT / "legacy_default_vix_source_rebuild_2026-04-30"
DEFAULT_VIX_CSV = DEFAULT_VIX_REBUILD_ROOT / "inputs" / "default_bronze_bt_vix_1h.csv"
DEFAULT_VIX3M_CSV = DEFAULT_VIX_REBUILD_ROOT / "inputs" / "default_bronze_bt_vix3m_1h.csv"
REFERENCE_GRID = V4_1_ROOT / "forensic_feature_freeze_ladder_2026-04-30" / "grids" / "D_full_feature_freeze" / "v4_1_silver_grid.csv"
LEGACY_VIX_PATCH_GRID = DEFAULT_VIX_REBUILD_ROOT / "grids" / "v4_1_grid_legacy_default_vix_patch.csv"
INDICATORS = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT"]
FINAL_COLUMNS = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]


def assert_audit_path(path: Path) -> Path:
    root = AUDIT_ROOT.resolve()
    candidate = path.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Refusing to write outside audit root: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(assert_audit_path(path), index=False)


def write_json(payload: dict, path: Path) -> None:
    with assert_audit_path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    assert_audit_path(path).write_text(text, encoding="utf-8")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    try:
        out[column] = out[column].dt.tz_localize(None)
    except Exception:
        pass
    return out


def csv_manifest(path: Path, role: str, source: str | None = None) -> dict:
    df = pd.read_csv(path)
    ts_min = ts_max = None
    for col in ["timestamp", "release_datetime", "release_date", "date", "observation_date"]:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().any():
                ts_min = s.min().isoformat()
                ts_max = s.max().isoformat()
                break
    return {
        "path": str(path),
        "source": source,
        "role": role,
        "sha256": sha256_file(path),
        "row_count": int(len(df)),
        "schema": list(df.columns),
        "min_timestamp": ts_min,
        "max_timestamp": ts_max,
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
    }


def export_default_market(spark: SparkSession) -> dict:
    out_path = AUDIT_ROOT / "inputs" / "default_bronze_bt_market_1h_spy_qqq.csv"
    sdf = spark.table(DEFAULT_MARKET_TABLE).filter(F.col("symbol").isin("SPY", "QQQ")).orderBy("timestamp", "symbol")
    df = sdf.toPandas()
    df = normalize_timestamp(df, "timestamp")
    write_csv(df, out_path)
    row = {
        "source_table": DEFAULT_MARKET_TABLE,
        "output_csv": str(out_path),
        "read_only_source_table": True,
        "row_count": int(len(df)),
        "schema": list(df.columns),
        "min_timestamp": None if df.empty else df["timestamp"].min().isoformat(),
        "max_timestamp": None if df.empty else df["timestamp"].max().isoformat(),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "sha256": sha256_file(out_path),
    }
    write_json({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "production_tables_mutated": False, "export": row}, AUDIT_ROOT / "manifests" / "default_market_export_manifest.json")
    return row


def copy_first_releases() -> list[dict]:
    rows = []
    for ind in INDICATORS:
        default_src = DEFAULT_FIRST_RELEASE_ROOT / f"first_releases_{ind}.csv"
        v4_src = V4_FIRST_RELEASE_ROOT / f"first_releases_{ind}.csv"
        if not default_src.exists():
            raise FileNotFoundError(f"Recovered-code default first-release source missing: {default_src}")
        out = AUDIT_ROOT / "inputs" / "first_releases" / f"first_releases_{ind}.csv"
        assert_audit_path(out)
        shutil.copy2(default_src, out)
        row = csv_manifest(out, f"first-release source for {ind}", source=str(default_src))
        row["v4_listed_path_exists"] = v4_src.exists()
        row["v4_listed_path"] = str(v4_src)
        row["selected_source_reason"] = "recovered generator LEGACY_VOLUME_PATH points to default schema; v4 path was incomplete"
        rows.append(row)
    df = pd.DataFrame(rows)
    write_csv(df, AUDIT_ROOT / "manifests" / "first_releases_manifest.csv")
    write_json({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "files": rows}, AUDIT_ROOT / "manifests" / "first_releases_manifest.json")
    return rows


def copy_vix_manifests() -> list[dict]:
    rows = []
    for src, name in [(DEFAULT_VIX_CSV, "default_bronze_bt_vix_1h.csv"), (DEFAULT_VIX3M_CSV, "default_bronze_bt_vix3m_1h.csv")]:
        if not src.exists():
            raise FileNotFoundError(f"Default VIX export missing: {src}")
        dst = AUDIT_ROOT / "inputs" / name
        assert_audit_path(dst)
        shutil.copy2(src, dst)
        rows.append(csv_manifest(dst, "default VIX/VIX3M source copied from legacy default VIX audit", source=str(src)))
    write_json({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "files": rows}, AUDIT_ROOT / "manifests" / "default_vix_reference_manifest.json")
    return rows


def load_corrected_macro_selection() -> pd.DataFrame:
    if not CORRECTED_MACRO_SELECTION_AUDIT.exists():
        raise FileNotFoundError(f"Corrected macro_selection_audit not found: {CORRECTED_MACRO_SELECTION_AUDIT}")
    dst = AUDIT_ROOT / "inputs" / "macro_selection_audit_recovered_corrected.csv"
    assert_audit_path(dst)
    shutil.copy2(CORRECTED_MACRO_SELECTION_AUDIT, dst)
    pdf = pd.read_csv(dst)
    pdf["release_datetime"] = pd.to_datetime(pdf["release_datetime"], errors="coerce")
    pdf["timestamp"] = pdf["release_datetime"]
    pdf["actual"] = pd.to_numeric(pdf["actual"], errors="coerce")
    write_json({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "file": csv_manifest(dst, "recovered corrected macro selected values", source=str(CORRECTED_MACRO_SELECTION_AUDIT))}, AUDIT_ROOT / "manifests" / "macro_selection_audit_manifest.json")
    return pdf[["indicator_id", "timestamp", "actual"]].dropna(subset=["indicator_id", "timestamp"])


def build_grid() -> pd.DataFrame:
    market = pd.read_csv(AUDIT_ROOT / "inputs" / "default_bronze_bt_market_1h_spy_qqq.csv")
    market = normalize_timestamp(market, "timestamp")
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market_wide = (
        market.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .rename(columns={"SPY": "SPY_close", "QQQ": "QQQ_close"})
        .reset_index()
        .sort_values("timestamp")
    )
    end_date = market_wide["timestamp"].max()
    backbone = pd.DataFrame({"timestamp": pd.date_range(start=pd.Timestamp("2016-01-01 00:00:00"), end=end_date, freq="h")})

    macro_pdf = load_corrected_macro_selection()
    macro_pdf = normalize_timestamp(macro_pdf, "timestamp")
    macro_pdf = macro_pdf.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp", "indicator_id"], keep="last")
    macro_pdf["join_ts"] = (macro_pdf["timestamp"] + pd.Timedelta(hours=2)).dt.floor("h")
    macro_prepared = (
        macro_pdf.pivot_table(index="join_ts", columns="indicator_id", values="actual", aggfunc="mean")
        .reset_index()
        .rename(columns={"join_ts": "timestamp"})
    )

    vix3m = pd.read_csv(AUDIT_ROOT / "inputs" / "default_bronze_bt_vix3m_1h.csv")
    vix3m = normalize_timestamp(vix3m, "timestamp")
    vix3m = vix3m.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    vix3m["vix3m"] = pd.to_numeric(vix3m["close"], errors="coerce")
    vix3m = vix3m[["timestamp", "vix3m"]]

    vix = pd.read_csv(AUDIT_ROOT / "inputs" / "default_bronze_bt_vix_1h.csv")
    vix = normalize_timestamp(vix, "timestamp")
    vix = vix.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    vix["VIXCLS"] = pd.to_numeric(vix["close"], errors="coerce")
    vix = vix[["timestamp", "VIXCLS"]]

    silver = backbone.merge(macro_prepared, on="timestamp", how="left")
    silver = silver.merge(market_wide[["timestamp", "SPY_close", "QQQ_close"]], on="timestamp", how="left")
    silver = silver.merge(vix3m, on="timestamp", how="left")
    silver = silver.merge(vix, on="timestamp", how="left")
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
    return final


def load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = parse_utc(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def compare_grid(candidate: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    features = FEATURES
    merged = candidate[["timestamp", "SPY_close", "QQQ_close", *features]].merge(
        reference[["timestamp", "SPY_close", "QQQ_close", *features]],
        on="timestamp",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    rows = []
    for col in ["SPY_close", "QQQ_close", *features]:
        c = pd.to_numeric(merged[f"{col}_candidate"], errors="coerce")
        r = pd.to_numeric(merged[f"{col}_reference"], errors="coerce")
        mask = c.notna() & r.notna()
        if not mask.any():
            rows.append({"column": col, "is_feature": col in features, "comparable_rows": 0, "near_match_count": 0, "near_match_rate": np.nan, "mae": np.nan, "max_abs_diff": np.nan, "first_divergence_timestamp": None})
            continue
        diff = (c[mask] - r[mask]).abs()
        near = diff <= 1e-9
        first = None if bool(near.all()) else merged.loc[mask[mask].index[~near][0], "timestamp"].isoformat()
        rows.append(
            {
                "column": col,
                "is_feature": col in features,
                "comparable_rows": int(mask.sum()),
                "near_match_count": int(near.sum()),
                "near_match_rate": float(near.sum() / mask.sum()),
                "mae": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "first_divergence_timestamp": first,
            }
        )
    report = pd.DataFrame(rows)
    feature_rows = report[report["is_feature"]]
    summary = {
        "candidate_row_count": int(len(candidate)),
        "reference_row_count": int(len(reference)),
        "inner_join_row_count": int(len(merged)),
        "candidate_min_timestamp": candidate["timestamp"].min().isoformat() if not candidate.empty else None,
        "candidate_max_timestamp": candidate["timestamp"].max().isoformat() if not candidate.empty else None,
        "reference_min_timestamp": reference["timestamp"].min().isoformat() if not reference.empty else None,
        "reference_max_timestamp": reference["timestamp"].max().isoformat() if not reference.empty else None,
        "candidate_only_timestamp_count": int(len(set(candidate["timestamp"]) - set(reference["timestamp"]))),
        "reference_only_timestamp_count": int(len(set(reference["timestamp"]) - set(candidate["timestamp"]))),
        "overall_14_feature_match": float(feature_rows["near_match_count"].sum() / feature_rows["comparable_rows"].sum()),
        "spy_close_match": float(report.loc[report["column"].eq("SPY_close"), "near_match_rate"].iloc[0]),
        "qqq_close_match": float(report.loc[report["column"].eq("QQQ_close"), "near_match_rate"].iloc[0]),
    }
    return report, summary


def diagnose(report: pd.DataFrame, summary: dict) -> str:
    worst = report[report["is_feature"]].sort_values(["near_match_rate", "mae"], ascending=[True, False])
    categories = []
    for _, row in worst.iterrows():
        col = row["column"]
        if col in ["VIXCLS_z", "vix3m_z", "VIX_Inversion_Z"]:
            src = "VIX source/logic"
        elif col in ["CPI_Event_Active", "NFP_Event_Active"]:
            src = "event flags / macro ffill timing"
        elif col == "Price_Mom_24H":
            src = "default market source / price momentum"
        else:
            src = "first_releases values/timestamps, macro +2h join, ffill, or z-score window"
        categories.append({"feature": col, "near_match_rate": row["near_match_rate"], "mae": row["mae"], "likely_source": src, "first_divergence": row["first_divergence_timestamp"]})
    cat_df = pd.DataFrame(categories)
    ready = summary["overall_14_feature_match"] >= 0.95
    lines = [
        "# Remaining Grid Mismatch Diagnosis",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Overall 14-feature match: `{summary['overall_14_feature_match']:.6%}`",
        f"Ready for model replay: `{ready}`",
        "",
        "## Timestamp Membership",
        "",
        md_table(pd.DataFrame([summary])),
        "",
        "## Likely Sources By Feature",
        "",
        md_table(cat_df),
    ]
    if ready:
        lines += ["", "Grid match is at or above 95%; next step is recovered-generator model replay on this faithful grid."]
    else:
        lines += ["", "Grid match is below 95%; inspect the lowest-match features first and compare against corrected macro_selection_audit and corrected silver table columns."]
    return "\n".join(lines) + "\n"


def main() -> None:
    spark = SparkSession.builder.getOrCreate()
    (AUDIT_ROOT / "inputs" / "first_releases").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "grids").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "forensics").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)

    market_manifest = export_default_market(spark)
    first_manifest = copy_first_releases()
    vix_manifest = copy_vix_manifests()
    grid = build_grid()
    grid_path = AUDIT_ROOT / "grids" / "full_legacy_grid_clone.csv"
    write_csv(grid, grid_path)

    reference = load_grid(REFERENCE_GRID)
    report, summary = compare_grid(grid, reference)
    write_csv(report, AUDIT_ROOT / "forensics" / "full_legacy_grid_match_report.csv")
    write_json(summary, AUDIT_ROOT / "forensics" / "full_legacy_grid_match_summary.json")
    ready = summary["overall_14_feature_match"] >= 0.95
    summary_md = [
        "# Full Legacy Grid Clone Match Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Output folder: `{AUDIT_ROOT}`",
        f"Full legacy grid: `{grid_path}`",
        f"Overall 14-feature exact/near match: `{summary['overall_14_feature_match']:.6%}`",
        f"SPY_close match: `{summary['spy_close_match']:.6%}`",
        f"QQQ_close match: `{summary['qqq_close_match']:.6%}`",
        f"Ready for model replay: `{ready}`",
        "",
        "## Row/Timestamp Summary",
        "",
        md_table(pd.DataFrame([summary])),
        "",
        "## Match By Column",
        "",
        md_table(report),
        "",
        "## No Production Change Confirmation",
        "",
        "Read default market and isolated/default CSV artifacts only. Wrote isolated CSV/report artifacts only. No production Delta tables, workflows, Saxo execution, token refresh, RT notebooks, dashboard notebooks, model registry, schedules, or models were modified.",
    ]
    write_text(AUDIT_ROOT / "forensics" / "full_legacy_grid_match_summary.md", "\n".join(summary_md) + "\n")
    write_text(AUDIT_ROOT / "forensics" / "remaining_grid_mismatch_diagnosis.md", diagnose(report, summary))
    print(
        json.dumps(
            {
                "output_folder": str(AUDIT_ROOT),
                "default_market_csv": market_manifest["output_csv"],
                "default_market_sha256": market_manifest["sha256"],
                "first_release_manifest": str(AUDIT_ROOT / "manifests" / "first_releases_manifest.json"),
                "full_legacy_grid_path": str(grid_path),
                "overall_feature_match": summary["overall_14_feature_match"],
                "ready_for_model_replay": ready,
                "production_changes": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

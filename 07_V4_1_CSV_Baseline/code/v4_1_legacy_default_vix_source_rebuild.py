"""Export default-schema VIX sources and patch only VIX features in V4.1.

This is an isolated V4.1 forensic rebuild. It reads default-schema VIX Delta
tables, writes CSV artifacts under V4.1, and does not mutate production V4.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


V4_1_CODE_DIR = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
if V4_1_CODE_DIR.exists() and str(V4_1_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_1_CODE_DIR))

from v4_1_common import FEATURES, parse_utc


V4_1_ROOT = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv")
AUDIT_ROOT = V4_1_ROOT / "legacy_default_vix_source_rebuild_2026-04-30"
DEFAULT_VIX_TABLE = "pawan_quant_research_1.default.bronze_bt_vix_1h"
DEFAULT_VIX3M_TABLE = "pawan_quant_research_1.default.bronze_bt_vix3m_1h"
ORIGINAL_GRID = V4_1_ROOT / "grids" / "macro_market_open_or_30m_buffer__vix_next_day_10am" / "v4_1_silver_grid.csv"
NONBOOTSTRAP_GRID = V4_1_ROOT / "bootstrap_row_impact_audit_2026-04-30" / "grid_without_bootstrap.csv"
REFERENCE_GRID = (
    V4_1_ROOT
    / "forensic_feature_freeze_ladder_2026-04-30"
    / "grids"
    / "D_full_feature_freeze"
    / "v4_1_silver_grid.csv"
)
PATCHED_GRID = AUDIT_ROOT / "grids" / "v4_1_grid_legacy_default_vix_patch.csv"
VIX_FEATURES = ["VIXCLS_z", "vix3m_z", "VIX_Inversion_Z"]


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


def export_table(spark: SparkSession, table: str, out_path: Path) -> tuple[pd.DataFrame, dict]:
    df = spark.table(table).toPandas()
    if "timestamp" in df.columns:
        df = normalize_timestamp(df, "timestamp").sort_values("timestamp")
    write_csv(df, out_path)
    ts_min = ts_max = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts_min = None if ts.dropna().empty else ts.min().isoformat()
        ts_max = None if ts.dropna().empty else ts.max().isoformat()
    info = {
        "source_table": table,
        "output_csv": str(out_path),
        "read_only_source_table": True,
        "row_count": int(len(df)),
        "schema": list(df.columns),
        "min_timestamp": ts_min,
        "max_timestamp": ts_max,
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "sha256": sha256_file(assert_audit_path(out_path)),
    }
    return df, info


def prepare_default_vix(vix_df: pd.DataFrame, vix3m_df: pd.DataFrame, backbone: pd.DataFrame) -> pd.DataFrame:
    vix = normalize_timestamp(vix_df, "timestamp").dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    vix["VIXCLS"] = pd.to_numeric(vix["close"], errors="coerce")
    vix = vix[["timestamp", "VIXCLS"]].sort_values("timestamp")

    vix3m = normalize_timestamp(vix3m_df, "timestamp").dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    # Recovered build_legacy_silver semantics: VIX_SOURCE='fred_1d', so close -> vix3m.
    vix3m["vix3m"] = pd.to_numeric(vix3m["close"], errors="coerce")
    vix3m = vix3m[["timestamp", "vix3m"]].sort_values("timestamp")

    raw = backbone[["timestamp"]].merge(vix3m, on="timestamp", how="left").merge(vix, on="timestamp", how="left")
    raw[["VIXCLS", "vix3m"]] = raw[["VIXCLS", "vix3m"]].ffill()
    for col in ["VIXCLS", "vix3m"]:
        rolling_mean = raw[col].rolling(8761, min_periods=1).mean()
        rolling_std = raw[col].rolling(8761, min_periods=2).std()
        denominator = rolling_std.where(rolling_std != 0, 1)
        raw[f"{col}_z"] = (raw[col] - rolling_mean) / denominator
    raw["VIX_Inversion_Z"] = raw["VIXCLS_z"] - raw["vix3m_z"]
    return raw


def load_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = parse_utc(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def compare_features(candidate: pd.DataFrame, reference: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    merged = candidate[["timestamp", *[f for f in features if f in candidate.columns]]].merge(
        reference[["timestamp", *[f for f in features if f in reference.columns]]],
        on="timestamp",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    rows = []
    for feature in features:
        c_col = f"{feature}_candidate"
        r_col = f"{feature}_reference"
        if c_col not in merged.columns or r_col not in merged.columns:
            rows.append({"grid_variant": label, "feature": feature, "comparable_rows": 0, "near_match_rate": np.nan, "mae": np.nan, "max_abs_diff": np.nan, "correlation": np.nan, "first_divergence_timestamp": None, "large_error_gt_0_01": 0})
            continue
        c = pd.to_numeric(merged[c_col], errors="coerce")
        r = pd.to_numeric(merged[r_col], errors="coerce")
        mask = c.notna() & r.notna()
        if not mask.any():
            rows.append({"grid_variant": label, "feature": feature, "comparable_rows": 0, "near_match_rate": np.nan, "mae": np.nan, "max_abs_diff": np.nan, "correlation": np.nan, "first_divergence_timestamp": None, "large_error_gt_0_01": 0})
            continue
        diff = (c[mask] - r[mask]).abs()
        near = diff <= 1e-9
        corr = c[mask].corr(r[mask]) if mask.sum() > 1 and c[mask].std(ddof=1) != 0 and r[mask].std(ddof=1) != 0 else np.nan
        first = None if bool(near.all()) else merged.loc[mask[mask].index[~near][0], "timestamp"].isoformat()
        rows.append(
            {
                "grid_variant": label,
                "feature": feature,
                "comparable_rows": int(mask.sum()),
                "near_match_count": int(near.sum()),
                "near_match_rate": float(near.sum() / mask.sum()),
                "mae": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "correlation": float(corr) if pd.notna(corr) else np.nan,
                "first_divergence_timestamp": first,
                "large_error_gt_0_01": int((diff > 0.01).sum()),
                "large_error_gt_0_05": int((diff > 0.05).sum()),
                "large_error_gt_0_10": int((diff > 0.10).sum()),
            }
        )
    return pd.DataFrame(rows)


def overall_feature_match(candidate: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    per_feature = compare_features(candidate, reference, FEATURES, label)
    comparable = per_feature["comparable_rows"].sum()
    matches = per_feature.get("near_match_count", pd.Series(dtype=float)).fillna(0).sum()
    overall = float(matches / comparable) if comparable else np.nan
    per_feature["overall_feature_match_pct"] = overall
    return per_feature


def main() -> None:
    spark = SparkSession.builder.getOrCreate()
    (AUDIT_ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "grids").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "forensics").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "manifests").mkdir(parents=True, exist_ok=True)

    vix_df, vix_info = export_table(spark, DEFAULT_VIX_TABLE, AUDIT_ROOT / "inputs" / "default_bronze_bt_vix_1h.csv")
    vix3m_df, vix3m_info = export_table(spark, DEFAULT_VIX3M_TABLE, AUDIT_ROOT / "inputs" / "default_bronze_bt_vix3m_1h.csv")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_tables_read_only": True,
        "production_tables_mutated": False,
        "exports": [vix_info, vix3m_info],
    }
    write_json(manifest, AUDIT_ROOT / "manifests" / "default_vix_export_manifest.json")

    original = load_grid(ORIGINAL_GRID)
    nonbootstrap = load_grid(NONBOOTSTRAP_GRID)
    reference = load_grid(REFERENCE_GRID)
    raw_vix = prepare_default_vix(vix_df, vix3m_df, nonbootstrap[["timestamp"]])
    write_csv(raw_vix, AUDIT_ROOT / "inputs" / "legacy_default_vix_raw_ffill_and_zscores.csv")

    patched = nonbootstrap.copy()
    for col in ["VIXCLS", "vix3m", "VIXCLS_z", "vix3m_z", "VIX_Inversion_Z"]:
        patched[col] = raw_vix[col].to_numpy()
    write_csv(patched, PATCHED_GRID)

    vix_match = pd.concat(
        [
            compare_features(original, reference, VIX_FEATURES, "original_non_frozen_v4_1"),
            compare_features(nonbootstrap, reference, VIX_FEATURES, "nonbootstrap_v4_1"),
            compare_features(patched, reference, VIX_FEATURES, "legacy_default_vix_patch"),
        ],
        ignore_index=True,
    )
    write_csv(vix_match, AUDIT_ROOT / "forensics" / "legacy_default_vix_feature_match.csv")

    grid_match = pd.concat(
        [
            overall_feature_match(original, reference, "original_non_frozen_v4_1"),
            overall_feature_match(nonbootstrap, reference, "nonbootstrap_v4_1"),
            overall_feature_match(patched, reference, "legacy_default_vix_patch"),
        ],
        ignore_index=True,
    )
    write_csv(grid_match, AUDIT_ROOT / "forensics" / "grid_match_before_after_legacy_vix_patch.csv")

    vix_wide = vix_match.pivot(index="feature", columns="grid_variant", values=["near_match_rate", "mae", "max_abs_diff", "correlation"]).reset_index()
    vix_wide.columns = ["_".join([x for x in col if x]).strip("_") for col in vix_wide.columns.to_flat_index()]
    grid_overall = (
        grid_match.groupby("grid_variant", as_index=False)
        .agg(overall_feature_match_pct=("overall_feature_match_pct", "first"))
        .sort_values("grid_variant")
    )
    patched_vix = vix_match[vix_match["grid_variant"].eq("legacy_default_vix_patch")]
    nonboot_vix = vix_match[vix_match["grid_variant"].eq("nonbootstrap_v4_1")]
    before_after = patched_vix[["feature", "near_match_rate", "mae"]].merge(
        nonboot_vix[["feature", "near_match_rate", "mae"]],
        on="feature",
        suffixes=("_after_patch", "_before_patch"),
    )
    before_after["near_match_delta"] = before_after["near_match_rate_after_patch"] - before_after["near_match_rate_before_patch"]
    before_after["mae_delta"] = before_after["mae_after_patch"] - before_after["mae_before_patch"]
    materially_improved = bool(
        (before_after["near_match_delta"].max() > 0.10)
        or (before_after["mae_delta"].min() < -0.05)
    )
    decision = (
        "Legacy default VIX source materially improves the VIX mismatch; next rebuild the complete grid with default market/VIX and first_releases path matching build_legacy_silver exactly."
        if materially_improved
        else "Legacy default VIX source alone does not materially match corrected VIX features; inspect hidden VIX table logic, schema/column branch, lag, and exact corrected table output."
    )
    raw_reference_note = (
        "Corrected reference/full-feature-freeze grid does not retain raw VIXCLS/vix3m levels, so raw reference values could not be directly compared."
    )
    summary_lines = [
        "# Legacy Default VIX Source Rebuild Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Output folder: `{AUDIT_ROOT}`",
        "",
        "## Exported Sources",
        "",
        md_table(pd.DataFrame([vix_info, vix3m_info])[["source_table", "output_csv", "row_count", "min_timestamp", "max_timestamp", "sha256"]]),
        "",
        "## VIX Feature Match",
        "",
        md_table(vix_match),
        "",
        "## VIX Before/After Delta",
        "",
        md_table(before_after),
        "",
        "## Overall Grid Match",
        "",
        md_table(grid_overall),
        "",
        "## Raw Reference Values",
        "",
        raw_reference_note,
        "",
        "## Decision",
        "",
        decision,
        "",
        "## No Production Change Confirmation",
        "",
        "Read default VIX/VIX3M tables only. Wrote isolated CSV/report artifacts only. No production Delta tables, workflows, Saxo execution, token refresh, RT notebooks, dashboard notebooks, model registry, schedules, or models were modified.",
    ]
    write_text(AUDIT_ROOT / "forensics" / "legacy_default_vix_feature_match_summary.md", "\n".join(summary_lines) + "\n")

    result = {
        "output_folder": str(AUDIT_ROOT),
        "vix_csv": vix_info["output_csv"],
        "vix_sha256": vix_info["sha256"],
        "vix3m_csv": vix3m_info["output_csv"],
        "vix3m_sha256": vix3m_info["sha256"],
        "patched_grid": str(PATCHED_GRID),
        "vix_feature_match_path": str(AUDIT_ROOT / "forensics" / "legacy_default_vix_feature_match.csv"),
        "grid_match_path": str(AUDIT_ROOT / "forensics" / "grid_match_before_after_legacy_vix_patch.csv"),
        "materially_improved": materially_improved,
        "decision": decision,
        "production_changes": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

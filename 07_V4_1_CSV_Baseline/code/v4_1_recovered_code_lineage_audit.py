"""Lineage audit for recovered corrected generator code.

Read-only source-code and artifact inventory. Writes only under the isolated
V4.1 recovered_code_lineage_audit folder.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession


V4_1_ROOT = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv")
AUDIT_ROOT = V4_1_ROOT / "recovered_code_lineage_audit_2026-04-30"
PRIMARY = (
    V4_1_ROOT
    / "forensic_corrected_training_artifact_recovery_2026-04-30"
    / "code"
    / "recovered_corrected_training_generator.py"
)
RUNNER = (
    V4_1_ROOT
    / "recovered_generator_replay_2026-04-30"
    / "code"
    / "v4_1_recovered_generator_runner.py"
)

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
    "SPY_close",
    "QQQ_close",
]
MACRO_Z = ["CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z", "GDPC1_z", "INDPRO_z", "UMCSENT_z"]
SEARCH_TERMS = [
    "bronze_bt_macro_events_legacy_recreated_corrected",
    "silver_bt_master_grid_1h_legacy_recreated_corrected",
    "bt_walkforward_regime_signals_legacy_recreated_corrected",
    "full_signal_export",
    "walkforward_signals",
    "first_releases_",
    "raw_finnhub_haystack",
    "bronze_bt_vix_1h",
    "bronze_bt_vix3m_1h",
    "VIXCLS_z",
    "vix3m_z",
    *FEATURES,
    "spark.table",
    "read.csv",
    "pd.read_csv",
    "to_csv",
    "saveAsTable",
    "legacy",
    "corrected",
    "comparison",
    "market_data",
    "/Volumes/",
    ".joblib",
    "VIX_SOURCE",
    "zscore",
    "rolling",
    "Window",
    "ffill",
    "fillna",
    "merge",
    "join",
]


def assert_audit_path(path: Path) -> Path:
    root = AUDIT_ROOT.resolve()
    candidate = path.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Refusing to write outside audit root: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(assert_audit_path(path), index=False)


def write_text(path: Path, text: str) -> None:
    assert_audit_path(path).write_text(text, encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "No rows."
    out = df.head(max_rows).copy() if max_rows else df.copy()
    out = out.astype(object).where(pd.notna(out), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        vals = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if max_rows and len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def line_matches(lines: list[str], terms: list[str]) -> list[dict]:
    rows = []
    for idx, line in enumerate(lines, start=1):
        hits = [term for term in terms if term in line]
        if hits:
            rows.append({"line": idx, "terms": ", ".join(hits), "code": line.rstrip()})
    return rows


def snippets_markdown(file_map: dict[str, Path]) -> str:
    parts = ["# Recovered Code Relevant Snippets", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", ""]
    for label, path in file_map.items():
        lines = read_lines(path)
        matches = line_matches(lines, SEARCH_TERMS)
        parts += [f"## {label}", "", f"Path: `{path}`", ""]
        for row in matches:
            start = max(1, row["line"] - 2)
            end = min(len(lines), row["line"] + 2)
            parts += [f"### Lines {start}-{end} (`{row['terms']}`)", "", "```python"]
            for line_no in range(start, end + 1):
                parts.append(f"{line_no:04d}: {lines[line_no - 1]}")
            parts += ["```", ""]
    return "\n".join(parts)


def feature_lineage() -> pd.DataFrame:
    rows = []
    macro_notes = (
        "Raw macro events are rebuilt in build_corrected_bronze from "
        "/Volumes/pawan_quant_research_1/default/market_data/first_releases_<indicator>.csv "
        "plus ALFRED latest-observation selection, then pivoted/ffilled in build_legacy_silver."
    )
    for feature in MACRO_Z:
        raw = feature.removesuffix("_z")
        rows.append(
            {
                "feature": feature,
                "created_from_raw_data": True,
                "loaded_from_existing_table": False,
                "loaded_from_csv": True,
                "copied_from_previous_or_cached_artifact": False,
                "recomputed_in_recovered_generator": True,
                "overwritten_later": False,
                "source_path_or_table": f"first_releases_{raw}.csv -> pawan_quant_research_1.v4.bronze_bt_macro_events_legacy_recreated_corrected",
                "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 368-426 build_corrected_bronze; lines 481-541 build_legacy_silver",
                "raw_level_exists": True,
                "zscore_recomputed_or_already_present": "recomputed in pandas rolling loop: rolling(8761, min_periods=1).mean and rolling(8761, min_periods=2).std",
                "notes": macro_notes,
            }
        )
    rows += [
        {
            "feature": "VIXCLS_z",
            "created_from_raw_data": True,
            "loaded_from_existing_table": True,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": True,
            "overwritten_later": False,
            "source_path_or_table": "pawan_quant_research_1.default.bronze_bt_vix_1h close -> VIXCLS",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 512-541 build_legacy_silver",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "recomputed from raw close using pandas rolling window 8761 current-row-included sample std",
            "notes": "VIX_SOURCE is fred_1d; no Portara branch used for VIXCLS.",
        },
        {
            "feature": "vix3m_z",
            "created_from_raw_data": True,
            "loaded_from_existing_table": True,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": True,
            "overwritten_later": False,
            "source_path_or_table": "pawan_quant_research_1.default.bronze_bt_vix3m_1h close -> vix3m when VIX_SOURCE='fred_1d'",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 499-510 and 535-541 build_legacy_silver",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "recomputed from raw close using pandas rolling window 8761 current-row-included sample std",
            "notes": "Portara branch would use column vix3m plus one-day shift, but constant VIX_SOURCE='fred_1d'.",
        },
        {
            "feature": "VIX_Inversion_Z",
            "created_from_raw_data": False,
            "loaded_from_existing_table": False,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": True,
            "overwritten_later": False,
            "source_path_or_table": "derived from VIXCLS_z and vix3m_z",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py line 555",
            "raw_level_exists": False,
            "zscore_recomputed_or_already_present": "derived after z-score recomputation",
            "notes": "silver_grid['VIX_Inversion_Z'] = silver_grid['VIXCLS_z'] - silver_grid['vix3m_z']",
        },
        {
            "feature": "Price_Mom_24H",
            "created_from_raw_data": True,
            "loaded_from_existing_table": True,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": True,
            "overwritten_later": False,
            "source_path_or_table": "pawan_quant_research_1.default.bronze_bt_market_1h, SPY only",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 463-474 and 556-558",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "not a z-score; recomputed from SPY_close shift(1)/shift(25)-1",
            "notes": "QQQ_close is not in recovered generator final_silver.",
        },
        {
            "feature": "CPI_Event_Active",
            "created_from_raw_data": False,
            "loaded_from_existing_table": False,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": True,
            "overwritten_later": False,
            "source_path_or_table": "derived from ffilled CPIAUCSL raw level deltas",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 543-551",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "not a z-score; rolling 49 pulse over CPI_Shock > 0",
            "notes": "",
        },
        {
            "feature": "NFP_Event_Active",
            "created_from_raw_data": False,
            "loaded_from_existing_table": False,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": True,
            "overwritten_later": False,
            "source_path_or_table": "derived from ffilled PAYEMS raw level deltas",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 546-554",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "not a z-score; rolling 49 pulse over NFP_Shock > 0",
            "notes": "",
        },
        {
            "feature": "SPY_close",
            "created_from_raw_data": True,
            "loaded_from_existing_table": True,
            "loaded_from_csv": False,
            "copied_from_previous_or_cached_artifact": False,
            "recomputed_in_recovered_generator": False,
            "overwritten_later": False,
            "source_path_or_table": "pawan_quant_research_1.default.bronze_bt_market_1h where symbol='SPY'",
            "exact_code_line_function_cell": "recovered_corrected_training_generator.py lines 463-474, 521-533, 560",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "not z-scored",
            "notes": "Final corrected silver table includes SPY_close.",
        },
        {
            "feature": "QQQ_close",
            "created_from_raw_data": False,
            "loaded_from_existing_table": False,
            "loaded_from_csv": True,
            "copied_from_previous_or_cached_artifact": True,
            "recomputed_in_recovered_generator": False,
            "overwritten_later": False,
            "source_path_or_table": "present in V4.1 full-feature-freeze CSV used by replay runner, not produced by primary recovered generator final_silver",
            "exact_code_line_function_cell": "v4_1_recovered_generator_runner.py lines 42-48 and 206-212 load_grid",
            "raw_level_exists": True,
            "zscore_recomputed_or_already_present": "not z-scored",
            "notes": "This is a cached artifact dependency in the replay path.",
        },
    ]
    return pd.DataFrame(rows)


def extract_string_literals(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    parts.append(part.value)
                elif isinstance(part, ast.FormattedValue):
                    parts.append("{expr}")
            values.append("".join(parts))
    return values


def candidate_artifacts() -> list[dict]:
    candidates = [
        {"artifact": "pawan_quant_research_1.default.bronze_bt_market_1h", "kind": "table", "role": "SPY market source for recovered generator"},
        {"artifact": "pawan_quant_research_1.default.bronze_bt_vix_1h", "kind": "table", "role": "VIXCLS raw source for recovered generator"},
        {"artifact": "pawan_quant_research_1.default.bronze_bt_vix3m_1h", "kind": "table", "role": "vix3m raw source for recovered generator"},
        {"artifact": "pawan_quant_research_1.v4.bronze_bt_macro_events_legacy_recreated_corrected", "kind": "table", "role": "corrected bronze macro output/input to corrected silver"},
        {"artifact": "pawan_quant_research_1.v4.silver_bt_master_grid_1h_legacy_recreated_corrected", "kind": "table", "role": "corrected silver grid table produced by primary recovered generator"},
        {"artifact": "pawan_quant_research_1.v4.bt_walkforward_regime_signals_legacy_recreated_corrected", "kind": "table", "role": "corrected walkforward signal output"},
        {"artifact": "/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/forensic_feature_freeze_ladder_2026-04-30/grids/D_full_feature_freeze/v4_1_silver_grid.csv", "kind": "file", "role": "cached full-feature-freeze grid loaded by recovered replay runner"},
        {"artifact": "/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/full_signal_export.csv", "kind": "file", "role": "corrected full-signal export reference"},
        {"artifact": "/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/walkforward_signals.csv", "kind": "file", "role": "corrected walkforward signal reference"},
        {"artifact": "/Volumes/pawan_quant_research_1/v4/market_data/comparison/2026-04-20_canonical_legacy_v4_baseline_vs_corrected/corrected_variant/macro_selection_audit.csv", "kind": "file", "role": "macro selection audit output if retained"},
        {"artifact": "/Volumes/pawan_quant_research_1/default/market_data/raw_finnhub_haystack.csv", "kind": "file", "role": "searched term; not used by recovered code"},
    ]
    for ind in ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT"]:
        candidates.append(
            {
                "artifact": f"/Volumes/pawan_quant_research_1/default/market_data/first_releases_{ind}.csv",
                "kind": "file",
                "role": f"source release schedule/value fallback CSV for {ind}",
            }
        )
    return candidates


def file_info(path_str: str) -> dict:
    path = Path(path_str)
    info = {"exists": path.exists(), "row_count": None, "schema": None, "min_timestamp": None, "max_timestamp": None, "mtime_utc": None, "sha256": None}
    if not path.exists():
        return info
    stat = path.stat()
    info["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    info["sha256"] = h.hexdigest()
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            info["row_count"] = int(len(df))
            info["schema"] = json.dumps(list(df.columns))
            for col in ["timestamp", "release_datetime", "release_date", "date", "observation_date"]:
                if col in df.columns:
                    s = pd.to_datetime(df[col], errors="coerce")
                    if s.notna().any():
                        info["min_timestamp"] = str(s.min())
                        info["max_timestamp"] = str(s.max())
                        break
        except Exception as exc:
            info["schema"] = f"csv_read_error: {exc}"
    return info


def table_info(spark: SparkSession, table: str) -> dict:
    info = {"exists": False, "row_count": None, "schema": None, "min_timestamp": None, "max_timestamp": None, "mtime_utc": None, "sha256": None}
    try:
        exists = spark.catalog.tableExists(table)
    except Exception:
        exists = False
    info["exists"] = bool(exists)
    if not exists:
        return info
    try:
        df = spark.table(table)
        info["schema"] = json.dumps([f"{field.name}:{field.dataType.simpleString()}" for field in df.schema.fields])
        info["row_count"] = int(df.count())
        timestamp_cols = [c for c in ["timestamp", "release_datetime", "ingestion_ts"] if c in df.columns]
        if timestamp_cols:
            col = timestamp_cols[0]
            row = df.selectExpr(f"min({col}) as min_ts", f"max({col}) as max_ts").toPandas().iloc[0]
            info["min_timestamp"] = str(row["min_ts"]) if pd.notna(row["min_ts"]) else None
            info["max_timestamp"] = str(row["max_ts"]) if pd.notna(row["max_ts"]) else None
    except Exception as exc:
        info["schema"] = f"table_read_error: {exc}"
    return info


def dependency_inventory() -> pd.DataFrame:
    spark = SparkSession.builder.getOrCreate()
    rows = []
    for candidate in candidate_artifacts():
        artifact = candidate["artifact"]
        info = table_info(spark, artifact) if candidate["kind"] == "table" else file_info(artifact)
        rows.append({**candidate, **info})
    return pd.DataFrame(rows)


def comparison_table() -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        if feature in MACRO_Z:
            recovered = "Corrected bronze from first_releases CSV + ALFRED selection, ffilled, rolling z-score in primary generator"
            v41 = "V4.1 selected ALFRED CSV, ffilled, rolling z-score"
            reason = "Raw comparable rows match; residual mismatch is not raw row selection. Original generator uses 2h macro join and first_releases schedule layer; V4.1 timing/bootstrap history differs."
            action = "Compare exact macro availability/join clock and pre-2016 history/window seeding."
        elif feature in ["VIXCLS_z", "vix3m_z"]:
            recovered = "Default-schema bronze_bt_vix(_3m)_1h table close, ffilled, rolling z-score"
            v41 = "FRED daily VIX/VXVCLS CSV expanded to hourly PIT"
            reason = "Different raw/source lineage: recovered code uses existing default Delta VIX tables, not the V4.1 FRED daily CSV source."
            action = "Export default bronze VIX/VIX3M tables to CSV and rebuild V4.1 using those raw hourly sources."
        elif feature == "VIX_Inversion_Z":
            recovered = "VIXCLS_z - vix3m_z from recovered VIX table-derived z-scores"
            v41 = "VIXCLS_z - vix3m_z from V4.1 FRED daily source"
            reason = "Inherited VIX source mismatch."
            action = "Fix VIX raw source lineage first."
        elif feature == "Price_Mom_24H":
            recovered = "SPY_close shift(1)/shift(25)-1 from default market table"
            v41 = "SPY_close shift(1)/shift(25)-1 from V4.1 frozen SPY/QQQ market CSV"
            reason = "Nearly matched in audits; not primary mismatch."
            action = "No immediate action."
        elif feature in ["CPI_Event_Active", "NFP_Event_Active"]:
            recovered = "49h rolling pulse over ffilled raw macro level delta > 0"
            v41 = "49h rolling pulse over ffilled raw macro level delta > 0"
            reason = "High match; remaining differences likely same macro timing/bootstrap issue."
            action = "Retest only after macro join-clock alignment."
        elif feature == "SPY_close":
            recovered = "Loaded from pawan_quant_research_1.default.bronze_bt_market_1h SPY"
            v41 = "Loaded from V4.1 frozen market CSV"
            reason = "Not primary mismatch."
            action = "No immediate action."
        elif feature == "QQQ_close":
            recovered = "Not produced in primary corrected silver; present only in replay/full-freeze CSV artifact"
            v41 = "Loaded from V4.1 frozen market CSV"
            reason = "Replay uses cached freeze grid with QQQ_close; primary generator final_silver omits it."
            action = "Treat replay grid as cached artifact, not proof of raw rebuild."
        else:
            recovered = v41 = reason = action = ""
        rows.append({"feature": feature, "recovered_generator_feature_source": recovered, "v4_1_csv_builder_feature_source": v41, "mismatch_reason": reason, "next_action": action})
    return pd.DataFrame(rows)


def write_diagnoses(deps: pd.DataFrame, lineage: pd.DataFrame) -> None:
    origin = """# Recovered Grid Origin Diagnosis

1. Did the recovered generator rebuild corrected silver from raw ALFRED/FRED/VIX/market data?

Yes in the primary recovered notebook: `build_corrected_bronze()` rebuilds corrected macro bronze from `first_releases_<indicator>.csv` plus ALFRED latest-observation selection, and `build_legacy_silver()` rebuilds corrected silver from corrected macro bronze, default-schema market, and default-schema VIX tables.

2. Or did it load/copy an existing corrected silver/grid table?

The primary recovered notebook writes `silver_bt_master_grid_1h_legacy_recreated_corrected`. The later V4.1 recovered replay runner does not rebuild silver; it loads the cached full-feature-freeze CSV at `forensic_feature_freeze_ladder_2026-04-30/grids/D_full_feature_freeze/v4_1_silver_grid.csv`.

3. Or did it load a cached full_signal_export with precomputed features?

The primary notebook creates `full_signal_export.csv` from the corrected silver table. The replay runner trains a new teacher from the cached feature-freeze grid and also compares to the corrected reference `full_signal_export.csv`.

4. Where did VIXCLS_z and vix3m_z actually come from?

In the primary recovered notebook they come from `pawan_quant_research_1.default.bronze_bt_vix_1h` and `pawan_quant_research_1.default.bronze_bt_vix3m_1h`, using their `close` columns because `VIX_SOURCE = "fred_1d"`, then pandas rolling z-scores.

5. Where did raw VIXCLS/vix3m levels come from, if anywhere?

Raw levels are loaded from the default-schema Delta tables above. They are intermediate columns in `silver_grid` but are not kept in `final_silver`; only z-score features are written.

6. Were z-scores recomputed in the recovered code, or already present?

In the primary recovered notebook z-scores are recomputed in `build_legacy_silver()` using pandas rolling operations. In the replay runner z-score columns are already present in the cached full-feature-freeze CSV.
"""
    write_text(AUDIT_ROOT / "forensics" / "recovered_grid_origin_diagnosis.md", origin.strip() + "\n")

    vix = """# VIX Code Lineage Audit

- `VIX_SOURCE = "fred_1d"` is set as a constant in the recovered generator.
- `vix3m_z` path: `spark.table(vix3m_table)` where default is `pawan_quant_research_1.default.bronze_bt_vix3m_1h`; for `fred_1d`, `close` is renamed to `vix3m`; then ffilled and z-scored.
- `VIXCLS_z` path: `spark.table(vix_spot_table)` where default is `pawan_quant_research_1.default.bronze_bt_vix_1h`; `close` is renamed to `VIXCLS`; then ffilled and z-scored.
- `VIX_Inversion_Z` is computed as `VIXCLS_z - vix3m_z`.
- The recovered generator does not read VIX from V4.1 FRED daily CSVs.
- The recovered generator does not retain raw VIX levels in the final corrected silver output.
- The replay runner does not recompute VIX z-scores; it loads them from the cached full-feature-freeze grid.

Conclusion: VIX/VIX3M mismatch is a raw/source lineage mismatch between default Delta VIX tables and the V4.1 CSV FRED daily PIT source, not a simple z-score math issue.
"""
    write_text(AUDIT_ROOT / "forensics" / "vix_code_lineage_audit.md", vix.strip() + "\n")


def summary_md(deps: pd.DataFrame, lineage: pd.DataFrame, cmp: pd.DataFrame) -> str:
    return f"""# Recovered Code Lineage Audit Summary

Generated UTC: {datetime.now(timezone.utc).isoformat()}

## Conclusions

1. Raw macro rows match but the grid still does not match because the recovered generator's grid construction is not only ALFRED row selection. It uses the legacy `first_releases_<indicator>.csv` schedule layer, applies a `timestamp + 2 hours` macro join clock, forward-fills on a 24/7 hourly backbone, and uses default-schema market/VIX Delta tables.

2. Corrected reference features were recomputed in the primary recovered notebook, but the later replay path used a cached full-feature-freeze grid CSV. The replay runner is therefore not evidence that the full grid can be regenerated from the V4.1 CSV inputs.

3. VIX/VIX3M features came from `pawan_quant_research_1.default.bronze_bt_vix_1h` and `pawan_quant_research_1.default.bronze_bt_vix3m_1h` in the primary recovered generator. The V4.1 builder used FRED daily CSV/PIT expansion, so VIX source lineage differs.

4. Macro z-score features came from corrected bronze macro events rebuilt from `first_releases_<indicator>.csv` plus ALFRED latest-observation values, then ffilled and z-scored in pandas. Comparable raw macro release values match V4.1, so remaining macro differences are join-clock/forward-fill/bootstrap-history semantics, not ALFRED selected values.

5. V4.1 cannot reproduce the corrected grid from raw V4.1 CSV inputs alone until it also has the recovered generator's default VIX Delta source and exact legacy join/availability semantics. The cached full-feature-freeze CSV is an artifact dependency.

6. Missing artifact/code for exact reproduction: default-schema VIX/VIX3M raw hourly exports and a V4.1 builder mode that exactly mirrors `build_legacy_silver()` including macro `timestamp + 2h` join, SPY-only market source behavior, and final column contract.

7. Smallest next action: export/read `pawan_quant_research_1.default.bronze_bt_vix_1h` and `pawan_quant_research_1.default.bronze_bt_vix3m_1h` into isolated V4.1 CSV artifacts, then rebuild only VIX/VIX3M z-score columns using recovered `build_legacy_silver()` logic while keeping macro raw levels fixed.

## Key Dependencies

{md_table(deps[['artifact', 'kind', 'exists', 'row_count', 'min_timestamp', 'max_timestamp', 'role']], max_rows=30)}

## Feature Lineage

{md_table(lineage[['feature', 'source_path_or_table', 'zscore_recomputed_or_already_present']], max_rows=40)}

## Recovered vs V4.1 Logic

{md_table(cmp, max_rows=40)}

## No Production Change Confirmation

This audit read source files and inspected table/file metadata. It did not write production Delta tables, train models, touch Saxo execution, token refresh, RT notebooks, model registry, dashboard notebooks, workflows, or schedules.
"""


def main() -> None:
    spark = SparkSession.builder.getOrCreate()
    (AUDIT_ROOT / "forensics").mkdir(parents=True, exist_ok=True)
    file_map = {"primary_recovered_generator": PRIMARY, "recovered_replay_runner": RUNNER}

    write_text(AUDIT_ROOT / "forensics" / "recovered_code_relevant_snippets.md", snippets_markdown(file_map))
    lineage = feature_lineage()
    write_csv(lineage, AUDIT_ROOT / "forensics" / "feature_lineage_from_recovered_code.csv")
    deps = dependency_inventory()
    write_csv(deps, AUDIT_ROOT / "forensics" / "hidden_artifact_dependencies.csv")
    cmp = comparison_table()
    write_csv(cmp, AUDIT_ROOT / "forensics" / "recovered_vs_v41_builder_logic_comparison.csv")
    write_diagnoses(deps, lineage)
    write_text(AUDIT_ROOT / "forensics" / "recovered_code_lineage_audit_summary.md", summary_md(deps, lineage, cmp))
    print(
        json.dumps(
            {
                "output_folder": str(AUDIT_ROOT),
                "feature_lineage_csv": str(AUDIT_ROOT / "forensics" / "feature_lineage_from_recovered_code.csv"),
                "hidden_dependency_csv": str(AUDIT_ROOT / "forensics" / "hidden_artifact_dependencies.csv"),
                "grid_origin_diagnosis": str(AUDIT_ROOT / "forensics" / "recovered_grid_origin_diagnosis.md"),
                "vix_lineage_audit": str(AUDIT_ROOT / "forensics" / "vix_code_lineage_audit.md"),
                "summary": str(AUDIT_ROOT / "forensics" / "recovered_code_lineage_audit_summary.md"),
                "production_changes": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Compare V4 production raw macro events against V4.1 selected raw releases.

This is a raw-lineage audit only. It does not forward-fill, compute z-scores,
train models, or mutate production tables.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


V4_1_ROOT = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv")
AUDIT_ROOT = V4_1_ROOT / "alfred_event_selector_audit_2026-04-30"
V41_SELECTED = AUDIT_ROOT / "outputs" / "v4_1_macro_alfred_selected_first_releases.csv"
OUT_CSV = AUDIT_ROOT / "forensics" / "raw_macro_v4_vs_v41_comparison.csv"
OUT_MD = AUDIT_ROOT / "forensics" / "raw_macro_v4_vs_v41_summary.md"
PROD_TABLE = "pawan_quant_research_1.v4.bronze_bt_macro_events"
INDICATORS = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT"]
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
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


def assert_audit_path(path: Path) -> Path:
    root = AUDIT_ROOT.resolve()
    candidate = path.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Refusing to write outside V4.1 audit root: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def ny_release_ts_utc(release_date: object, indicator_id: str) -> pd.Timestamp:
    hhmm = EVENT_TIMES[indicator_id]
    hour, minute = [int(x) for x in hhmm.split(":")]
    d = pd.Timestamp(release_date).date()
    return pd.Timestamp(datetime.combine(d, time(hour, minute)), tz=NY).tz_convert(UTC).tz_localize(None)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    out = df.astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in out.itertuples(index=False):
        vals = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_v4_events() -> pd.DataFrame:
    spark = SparkSession.builder.getOrCreate()
    ind_sql = ", ".join([f"'{x}'" for x in INDICATORS])
    df = spark.sql(
        f"""
        SELECT indicator_id, timestamp AS v4_timestamp, actual AS v4_actual
        FROM {PROD_TABLE}
        WHERE indicator_id IN ({ind_sql})
        """
    ).toPandas()
    if df.empty:
        return df
    df["indicator_id"] = df["indicator_id"].astype(str)
    df["v4_timestamp"] = parse_utc(df["v4_timestamp"])
    df["v4_actual"] = pd.to_numeric(df["v4_actual"], errors="coerce")
    df = df.dropna(subset=["indicator_id", "v4_timestamp"])
    df["release_date_key"] = df["v4_timestamp"].dt.date.astype(str)
    # Production raw table should be unique by indicator/timestamp. If date has
    # more than one event, keep all rows visible by adding an ordinal.
    df = df.sort_values(["indicator_id", "release_date_key", "v4_timestamp"]).reset_index(drop=True)
    df["date_row_ordinal"] = df.groupby(["indicator_id", "release_date_key"]).cumcount()
    return df


def load_v41_selected() -> pd.DataFrame:
    df = pd.read_csv(V41_SELECTED)
    df = df[df["indicator_id"].isin(INDICATORS)].copy()
    df["v4_1_release_date"] = pd.to_datetime(df["release_date"], errors="coerce").dt.date.astype(str)
    df["v4_1_selected_observation_date"] = pd.to_datetime(df["selected_observation_date"], errors="coerce").dt.date.astype(str)
    df["v4_1_selected_value"] = pd.to_numeric(df["selected_value"], errors="coerce")
    df["v4_1_release_ts_utc"] = [
        ny_release_ts_utc(d, ind) for d, ind in zip(df["v4_1_release_date"], df["indicator_id"])
    ]
    df["release_date_key"] = df["v4_1_release_date"]
    df = df.sort_values(["indicator_id", "release_date_key", "v4_1_release_ts_utc"]).reset_index(drop=True)
    df["date_row_ordinal"] = df.groupby(["indicator_id", "release_date_key"]).cumcount()
    return df[
        [
            "indicator_id",
            "release_date_key",
            "date_row_ordinal",
            "v4_1_release_date",
            "v4_1_release_ts_utc",
            "v4_1_selected_observation_date",
            "v4_1_selected_value",
        ]
    ]


def build_comparison(v4: pd.DataFrame, v41: pd.DataFrame) -> pd.DataFrame:
    joined = v4.merge(
        v41,
        on=["indicator_id", "release_date_key", "date_row_ordinal"],
        how="outer",
        indicator=True,
    )
    joined["missing_in_v4"] = joined["_merge"].eq("right_only")
    joined["missing_in_v4_1"] = joined["_merge"].eq("left_only")
    joined["absolute_difference"] = (joined["v4_actual"] - joined["v4_1_selected_value"]).abs()
    joined["timestamp_difference_hours"] = (
        (joined["v4_timestamp"] - joined["v4_1_release_ts_utc"]).dt.total_seconds() / 3600.0
    )
    joined["exact_value_match_flag"] = np.isclose(
        pd.to_numeric(joined["v4_actual"], errors="coerce"),
        pd.to_numeric(joined["v4_1_selected_value"], errors="coerce"),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=False,
    )
    both = ~(joined["missing_in_v4"] | joined["missing_in_v4_1"])
    joined.loc[~both, "exact_value_match_flag"] = False
    out = joined[
        [
            "indicator_id",
            "v4_timestamp",
            "v4_actual",
            "v4_1_release_date",
            "v4_1_release_ts_utc",
            "v4_1_selected_observation_date",
            "v4_1_selected_value",
            "absolute_difference",
            "timestamp_difference_hours",
            "exact_value_match_flag",
            "missing_in_v4",
            "missing_in_v4_1",
        ]
    ].sort_values(["indicator_id", "v4_1_release_date", "v4_timestamp"], na_position="last")
    return out


def write_summary(cmp: pd.DataFrame) -> dict:
    comparable = ~(cmp["missing_in_v4"] | cmp["missing_in_v4_1"])
    summary = (
        cmp.assign(comparable=comparable)
        .groupby("indicator_id", as_index=False)
        .agg(
            rows=("indicator_id", "size"),
            comparable_rows=("comparable", "sum"),
            exact_value_matches=("exact_value_match_flag", "sum"),
            missing_in_v4=("missing_in_v4", "sum"),
            missing_in_v4_1=("missing_in_v4_1", "sum"),
            max_abs_difference=("absolute_difference", "max"),
            mean_abs_difference=("absolute_difference", "mean"),
            max_abs_timestamp_difference_hours=("timestamp_difference_hours", lambda s: float(s.abs().max()) if s.notna().any() else np.nan),
        )
    )
    summary["exact_value_match_pct_on_comparable"] = np.where(
        summary["comparable_rows"] > 0,
        summary["exact_value_matches"] / summary["comparable_rows"],
        np.nan,
    )
    comparable_values_match = bool(cmp.loc[comparable, "exact_value_match_flag"].all())
    no_missing_rows = int(cmp["missing_in_v4"].sum()) == 0 and int(cmp["missing_in_v4_1"].sum()) == 0
    if comparable_values_match and no_missing_rows:
        verdict = "Raw values match exactly. Remaining issue is ffill/bootstrap/z-score construction."
    elif comparable_values_match:
        verdict = (
            "Comparable raw values match exactly; only release-row presence differs. "
            "This points to bootstrap/history construction rather than ALFRED row selection for matched releases."
        )
    else:
        verdict = (
            "Raw values do not match on comparable release rows. "
            "This points to raw lineage/ALFRED row selection before ffill/z-score construction."
        )
    mismatch_examples = cmp[
        cmp["missing_in_v4"]
        | cmp["missing_in_v4_1"]
        | (~cmp["exact_value_match_flag"])
    ].head(25)
    lines = [
        "# Raw Macro V4 vs V4.1 Comparison",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Production raw table read only: `{PROD_TABLE}`",
        f"V4.1 selected raw releases: `{V41_SELECTED}`",
        "",
        "No forward-fill, z-scores, model training, or production mutations were performed.",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary By Indicator",
        "",
        md_table(summary),
        "",
        "## First Mismatch/Missing Examples",
        "",
        md_table(mismatch_examples),
    ]
    assert_audit_path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output_csv": str(OUT_CSV),
        "output_md": str(OUT_MD),
        "verdict": verdict,
        "comparable_values_match": comparable_values_match,
        "all_comparable_match_no_missing": comparable_values_match and no_missing_rows,
        "summary": summary.to_dict(orient="records"),
        "total_rows": int(len(cmp)),
        "comparable_rows": int(comparable.sum()),
        "exact_value_matches": int(cmp.loc[comparable, "exact_value_match_flag"].sum()),
        "missing_in_v4": int(cmp["missing_in_v4"].sum()),
        "missing_in_v4_1": int(cmp["missing_in_v4_1"].sum()),
    }


def main() -> None:
    v4 = load_v4_events()
    v41 = load_v41_selected()
    cmp = build_comparison(v4, v41)
    assert_audit_path(OUT_CSV)
    cmp.to_csv(OUT_CSV, index=False)
    result = write_summary(cmp)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()

# Databricks notebook source
# MAGIC %md
# MAGIC # GDELT Raw CSV Export Audit
# MAGIC
# MAGIC Research-only utility for MUB-30 raw GDELT GKG row inspection.
# MAGIC
# MAGIC Safety boundaries:
# MAGIC - Do not change production features, signals, model registry, workflows, Saxo execution, or RT pipelines.
# MAGIC - Do not change canonical GDELT feature/backfill logic.
# MAGIC - Write only CSV exports under `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/`.
# MAGIC - Do not download article bodies. `DocumentIdentifier` URLs are exported for manual identification only.

# COMMAND ----------

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
NAMESPACE = f"{CATALOG}.{SCHEMA}"
DEFAULT_OUTPUT_DIR = (
    f"/Volumes/{CATALOG}/{SCHEMA}/market_data/research/news_flow/raw_audit/csv_exports"
)
DEFAULT_BIGQUERY_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

SAMPLE_BUCKETS = [
    ("normal_bucket", "2023-05-13 00:00:00"),
    ("repaired_bucket", "2023-09-21 23:45:00"),
    ("missing_outage_bucket", "2025-06-18 00:00:00"),
    ("post_gap_bucket", "2025-07-03 23:45:00"),
]

MARKET_STRESS_HOURS = [
    "2021-01-04 15:00:00",
    "2021-01-28 17:00:00",
    "2021-01-04 18:00:00",
]

GKG_COLUMNS = [
    "GKGRECORDID",
    "DATE",
    "SourceCollectionIdentifier",
    "SourceCommonName",
    "DocumentIdentifier",
    "Counts",
    "V2Counts",
    "Themes",
    "V2Themes",
    "Locations",
    "V2Locations",
    "Persons",
    "V2Persons",
    "Organizations",
    "V2Organizations",
    "V2Tone",
    "Dates",
    "GCAM",
    "SharingImage",
    "RelatedImages",
    "SocialImageEmbeds",
    "SocialVideoEmbeds",
    "Quotations",
    "AllNames",
    "Amounts",
    "TranslationInfo",
    "Extras",
]

EXPORT_COLUMNS = [
    "source_hour",
    "bucket_ts",
    "sample_label",
    "GKGRECORDID",
    "DATE",
    "SourceCommonName",
    "DocumentIdentifier",
    "V2Tone",
    "parsed_tone",
    "positive_score",
    "negative_score",
    "polarity",
    "activity_density",
    "self_density",
    "word_count",
    "V2Themes",
    "parsed_theme_tokens",
    "V2Persons",
    "parsed_person_tokens",
    "V2Organizations",
    "parsed_organization_tokens",
    "V2Locations",
    "parsed_location_names",
    "row_type",
    "is_numeric_raw_compatible",
    "is_translated_T",
    "matched_market_stress_themes",
    "matched_policy_uncertainty_themes",
    "matched_geopolitical_risk_themes",
    "source_system",
    "raw_zip_http_status",
    "raw_zip_source_url",
]

RAW_GKG_EXPORT_COLUMNS = [
    "source_system",
    "source_hour",
    "bucket_ts",
    "sample_label",
    "raw_zip_http_status",
    "raw_zip_source_url",
    *GKG_COLUMNS,
]

MARKET_STRESS_RE = re.compile(
    r"RECESSION|BANK|BANKING|CREDIT|LIQUIDITY|DEFAULT|DEBT|MARKET_CRASH|"
    r"VOLATILITY|LAYOFF|UNEMPLOYMENT|BANKRUPT",
    re.IGNORECASE,
)
POLICY_UNCERTAINTY_RE = re.compile(
    r"POLICY|UNCERTAINTY|TARIFF|TRADE|FED|FEDERAL_RESERVE|CENTRAL_BANK|"
    r"FISCAL|TAX|GOVERNMENT|REGULATION",
    re.IGNORECASE,
)
GEOPOLITICAL_RISK_RE = re.compile(
    r"WAR|CONFLICT|SANCTION|MILITARY|TERROR|ARMEDCONFLICT|GEOPOLITICAL|"
    r"PROTEST|COUP|CRISIS",
    re.IGNORECASE,
)

THEME_DICTIONARY = {
    "market_stress": MARKET_STRESS_RE,
    "policy_uncertainty": POLICY_UNCERTAINTY_RE,
    "geopolitical_risk": GEOPOLITICAL_RISK_RE,
}

BIGQUERY_MARKET_STRESS_PATTERN = (
    "RECESSION|BANK|BANKING|CREDIT|LIQUIDITY|DEFAULT|DEBT|MARKET_CRASH|"
    "VOLATILITY|LAYOFF|UNEMPLOYMENT|BANKRUPT"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export bounded raw GDELT GKG row-level CSVs for MUB-30 research audit."
    )
    parser.add_argument("--output-dir", default=os.environ.get("GDELT_RAW_EXPORT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--manifest-path-prefix",
        default=os.environ.get("GDELT_RAW_EXPORT_MANIFEST_PREFIX"),
        help="Path prefix to write into the manifest. Defaults to --output-dir.",
    )
    parser.add_argument("--bigquery-project", default=os.environ.get("GDELT_BIGQUERY_PROJECT", "mubama-v4-gdelt-research"))
    parser.add_argument("--bigquery-table", default=os.environ.get("GDELT_BIGQUERY_TABLE", DEFAULT_BIGQUERY_TABLE))
    parser.add_argument("--sample-cap-per-row-type", type=int, default=500)
    parser.add_argument(
        "--bigquery-runner",
        choices=["auto", "bq_cli", "python_client"],
        default=os.environ.get("GDELT_BIGQUERY_RUNNER", "auto"),
    )
    parser.add_argument(
        "--report-path",
        default=os.environ.get(
            "GDELT_RAW_EXPORT_REPORT_PATH",
            "03_Research/03_Artifacts/gdelt_raw_csv_export_audit_20260526.md",
        ),
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime:
    cleaned = value.replace("T", " ").replace("Z", "").split(".")[0]
    return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")


def format_timestamp(value: datetime | str | None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value).replace("T", " ").replace(".000Z", "").replace("Z", "")


def gdelt_date_int(bucket_ts: datetime) -> int:
    return int(bucket_ts.strftime("%Y%m%d%H%M%S"))


def gdelt_gkg_url(bucket_ts: datetime) -> str:
    return f"http://data.gdeltproject.org/gdeltv2/{bucket_ts:%Y%m%d%H%M%S}.gkg.csv.zip"


def is_numeric_gkgrecordid(value: object) -> bool:
    return bool(re.match(r"^[0-9]{14}-[0-9]+$", str(value or "")))


def is_translated_record(value: object) -> bool:
    return bool(re.match(r"^[0-9]{14}-T[0-9]+$", str(value or "")))


def row_type_for_gkgrecordid(value: object) -> str:
    if is_numeric_gkgrecordid(value):
        return "numeric_raw_compatible"
    if is_translated_record(value):
        return "translated_T"
    return "other"


def parse_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return None if math.isnan(parsed) else parsed
    except Exception:
        return None


def parse_v2tone(v2tone: object) -> dict[str, float | None]:
    parts = str(v2tone or "").split(",")
    values = [parse_float(part) for part in parts[:7]]
    while len(values) < 7:
        values.append(None)
    return {
        "parsed_tone": values[0],
        "positive_score": values[1],
        "negative_score": values[2],
        "polarity": values[3],
        "activity_density": values[4],
        "self_density": values[5],
        "word_count": values[6],
    }


def split_v2themes(v2themes: object) -> list[str]:
    tokens = []
    for part in str(v2themes or "").split(";"):
        token = part.strip().split(",", 1)[0].strip()
        if token:
            tokens.append(token)
    return tokens


def split_v2persons(v2persons: object) -> list[str]:
    return split_comma_position_tokens(v2persons)


def split_v2organizations(v2organizations: object) -> list[str]:
    return split_comma_position_tokens(v2organizations)


def split_comma_position_tokens(value: object) -> list[str]:
    tokens = []
    for part in str(value or "").split(";"):
        token = part.strip().split(",", 1)[0].strip()
        if token:
            tokens.append(token)
    return tokens


def parse_v2locations(v2locations: object) -> list[str]:
    names = []
    for part in str(v2locations or "").split(";"):
        pieces = part.split("#")
        if len(pieces) > 1 and pieces[1].strip():
            names.append(pieces[1].strip())
    return names


def matched_theme_groups(theme_tokens: list[str], dictionary: dict[str, re.Pattern]) -> dict[str, list[str]]:
    return {
        group_name: sorted({theme for theme in theme_tokens if pattern.search(theme)})
        for group_name, pattern in dictionary.items()
    }


def pipe_join(values: list[str]) -> str:
    return "|".join(values)


def normalize_row(
    row: dict[str, object],
    *,
    source_system: str,
    bucket_ts: str,
    source_hour: str = "",
    sample_label: str = "",
    raw_zip_http_status: int | str | None = "",
    raw_zip_source_url: str = "",
) -> dict[str, object]:
    record_id = row.get("GKGRECORDID")
    theme_tokens = split_v2themes(row.get("V2Themes"))
    person_tokens = split_v2persons(row.get("V2Persons"))
    organization_tokens = split_v2organizations(row.get("V2Organizations"))
    location_names = parse_v2locations(row.get("V2Locations"))
    matches = matched_theme_groups(theme_tokens, THEME_DICTIONARY)
    tone_parts = parse_v2tone(row.get("V2Tone"))
    row_type = row_type_for_gkgrecordid(record_id)
    normalized = {
        "source_hour": source_hour,
        "bucket_ts": bucket_ts,
        "sample_label": sample_label,
        "GKGRECORDID": record_id,
        "DATE": row.get("DATE"),
        "SourceCommonName": row.get("SourceCommonName"),
        "DocumentIdentifier": row.get("DocumentIdentifier"),
        "V2Tone": row.get("V2Tone"),
        **tone_parts,
        "V2Themes": row.get("V2Themes"),
        "parsed_theme_tokens": pipe_join(theme_tokens),
        "V2Persons": row.get("V2Persons"),
        "parsed_person_tokens": pipe_join(person_tokens),
        "V2Organizations": row.get("V2Organizations"),
        "parsed_organization_tokens": pipe_join(organization_tokens),
        "V2Locations": row.get("V2Locations"),
        "parsed_location_names": pipe_join(location_names),
        "row_type": row_type,
        "is_numeric_raw_compatible": is_numeric_gkgrecordid(record_id),
        "is_translated_T": is_translated_record(record_id),
        "matched_market_stress_themes": pipe_join(matches["market_stress"]),
        "matched_policy_uncertainty_themes": pipe_join(matches["policy_uncertainty"]),
        "matched_geopolitical_risk_themes": pipe_join(matches["geopolitical_risk"]),
        "source_system": source_system,
        "raw_zip_http_status": raw_zip_http_status,
        "raw_zip_source_url": raw_zip_source_url,
    }
    return {column: normalized.get(column, "") for column in EXPORT_COLUMNS}


def selected_bucket_structs() -> str:
    parts = []
    for label, timestamp_text in SAMPLE_BUCKETS:
        bucket_ts = parse_timestamp(timestamp_text)
        parts.append(
            "STRUCT("
            f"'{label}' AS sample_label, "
            f"TIMESTAMP '{format_timestamp(bucket_ts)}' AS bucket_ts, "
            f"{gdelt_date_int(bucket_ts)} AS date_int"
            ")"
        )
    return ",\n      ".join(parts)


def market_stress_bucket_structs() -> str:
    parts = []
    for source_hour_text in MARKET_STRESS_HOURS:
        source_hour = parse_timestamp(source_hour_text)
        for minutes_back in (45, 30, 15, 0):
            bucket_ts = source_hour - timedelta(minutes=minutes_back)
            parts.append(
                "STRUCT("
                f"TIMESTAMP '{format_timestamp(source_hour)}' AS source_hour, "
                f"TIMESTAMP '{format_timestamp(bucket_ts)}' AS bucket_ts, "
                f"{gdelt_date_int(bucket_ts)} AS date_int"
                ")"
            )
    return ",\n      ".join(parts)


def partition_predicate_for_timestamps(alias: str, timestamps: list[datetime]) -> str:
    dates = sorted({timestamp.date().isoformat() for timestamp in timestamps})
    ranges = [
        f"({alias}._PARTITIONTIME >= TIMESTAMP '{date_text}' "
        f"AND {alias}._PARTITIONTIME < TIMESTAMP_ADD(TIMESTAMP '{date_text}', INTERVAL 1 DAY))"
        for date_text in dates
    ]
    return "(" + " OR ".join(ranges) + ")"


def sample_bucket_timestamps() -> list[datetime]:
    return [parse_timestamp(timestamp_text) for _label, timestamp_text in SAMPLE_BUCKETS]


def market_stress_source_bucket_timestamps() -> list[datetime]:
    buckets = []
    for source_hour_text in MARKET_STRESS_HOURS:
        source_hour = parse_timestamp(source_hour_text)
        for minutes_back in (45, 30, 15, 0):
            buckets.append(source_hour - timedelta(minutes=minutes_back))
    return buckets


def run_bigquery(sql: str, project: str, runner: str) -> pd.DataFrame:
    if runner in {"auto", "bq_cli"} and shutil.which("bq"):
        cmd = [
            "bq",
            f"--project_id={project}",
            "query",
            "--use_legacy_sql=false",
            "--format=json",
            "--max_rows=1000000",
            sql,
        ]
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            if runner == "bq_cli":
                raise RuntimeError((completed.stderr or completed.stdout).strip())
        else:
            raw = completed.stdout.strip()
            return pd.DataFrame(json.loads(raw)) if raw else pd.DataFrame()

    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "No BigQuery runner available. Install google-cloud-bigquery or run with bq CLI."
        ) from exc

    client = bigquery.Client(project=project)
    return client.query(sql).result().to_dataframe()


def sample_bucket_bigquery_sql(bigquery_table: str, sample_cap_per_row_type: int) -> str:
    partition_predicate = partition_predicate_for_timestamps("g", sample_bucket_timestamps())
    return f"""
    WITH selected AS (
      SELECT * FROM UNNEST([
        {selected_bucket_structs()}
      ])
    ),
    source_rows AS (
      SELECT
        FORMAT_TIMESTAMP('%F %T', selected.bucket_ts) AS bucket_ts,
        selected.sample_label,
        CAST(g.`DATE` AS STRING) AS `DATE`,
        g.GKGRECORDID,
        CAST(g.SourceCollectionIdentifier AS STRING) AS SourceCollectionIdentifier,
        g.SourceCommonName,
        g.DocumentIdentifier,
        g.Counts,
        g.V2Counts,
        g.Themes,
        g.V2Themes,
        g.Locations,
        g.V2Locations,
        g.Persons,
        g.V2Persons,
        g.Organizations,
        g.V2Organizations,
        g.V2Tone,
        g.Dates,
        g.GCAM,
        g.SharingImage,
        g.RelatedImages,
        g.SocialImageEmbeds,
        g.SocialVideoEmbeds,
        g.Quotations,
        g.AllNames,
        g.Amounts,
        g.TranslationInfo,
        g.Extras,
        CASE
          WHEN REGEXP_CONTAINS(g.GKGRECORDID, r'^[0-9]{{14}}-[0-9]+$') THEN 'numeric_raw_compatible'
          WHEN REGEXP_CONTAINS(g.GKGRECORDID, r'^[0-9]{{14}}-T[0-9]+$') THEN 'translated_T'
          ELSE 'other'
        END AS row_type
      FROM `{bigquery_table}` g
      JOIN selected ON g.`DATE` = selected.date_int
      WHERE {partition_predicate}
    ),
    ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (
          PARTITION BY sample_label, bucket_ts, row_type
          ORDER BY GKGRECORDID, DocumentIdentifier
        ) AS rn
      FROM source_rows
    )
    SELECT * EXCEPT(rn)
    FROM ranked
    WHERE rn <= {int(sample_cap_per_row_type)}
    ORDER BY bucket_ts, row_type, GKGRECORDID, DocumentIdentifier
    """


def market_stress_bigquery_sql(bigquery_table: str) -> str:
    partition_predicate = partition_predicate_for_timestamps("g", market_stress_source_bucket_timestamps())
    return f"""
    WITH selected AS (
      SELECT * FROM UNNEST([
        {market_stress_bucket_structs()}
      ])
    ),
    source_rows AS (
      SELECT
        FORMAT_TIMESTAMP('%F %T', selected.source_hour) AS source_hour,
        FORMAT_TIMESTAMP('%F %T', selected.bucket_ts) AS bucket_ts,
        CAST(g.`DATE` AS STRING) AS `DATE`,
        g.GKGRECORDID,
        CAST(g.SourceCollectionIdentifier AS STRING) AS SourceCollectionIdentifier,
        g.SourceCommonName,
        g.DocumentIdentifier,
        g.Counts,
        g.V2Counts,
        g.Themes,
        g.V2Themes,
        g.Locations,
        g.V2Locations,
        g.Persons,
        g.V2Persons,
        g.Organizations,
        g.V2Organizations,
        g.V2Tone,
        g.Dates,
        g.GCAM,
        g.SharingImage,
        g.RelatedImages,
        g.SocialImageEmbeds,
        g.SocialVideoEmbeds,
        g.Quotations,
        g.AllNames,
        g.Amounts,
        g.TranslationInfo,
        g.Extras,
        UPPER(CONCAT(
          IFNULL(g.Themes, ''), ' ',
          IFNULL(g.V2Themes, ''), ' ',
          IFNULL(g.DocumentIdentifier, ''), ' ',
          IFNULL(g.SourceCommonName, '')
        )) AS topic_blob
      FROM `{bigquery_table}` g
      JOIN selected ON g.`DATE` = selected.date_int
      WHERE {partition_predicate}
    )
    SELECT
      source_hour,
      bucket_ts,
      `DATE`,
      GKGRECORDID,
      SourceCommonName,
      DocumentIdentifier,
      Counts,
      V2Counts,
      Themes,
      V2Themes,
      Locations,
      V2Locations,
      Persons,
      V2Persons,
      Organizations,
      V2Organizations,
      V2Tone,
      Dates,
      GCAM,
      SharingImage,
      RelatedImages,
      SocialImageEmbeds,
      SocialVideoEmbeds,
      Quotations,
      AllNames,
      Amounts,
      TranslationInfo,
      Extras,
      CASE
        WHEN REGEXP_CONTAINS(GKGRECORDID, r'^[0-9]{{14}}-[0-9]+$') THEN 'numeric_raw_compatible'
        WHEN REGEXP_CONTAINS(GKGRECORDID, r'^[0-9]{{14}}-T[0-9]+$') THEN 'translated_T'
        ELSE 'other'
      END AS row_type
    FROM source_rows
    WHERE REGEXP_CONTAINS(topic_blob, r'({BIGQUERY_MARKET_STRESS_PATTERN})')
    ORDER BY source_hour, bucket_ts, row_type, GKGRECORDID, DocumentIdentifier
    """


def read_gkg_csv_robust(raw_bytes: bytes) -> pd.DataFrame:
    text = raw_bytes.decode("utf-8", errors="replace")
    for engine in ("c", "python"):
        try:
            return pd.read_csv(
                io.StringIO(text),
                sep="\t",
                names=GKG_COLUMNS,
                dtype=str,
                header=None,
                on_bad_lines="skip",
                engine=engine,
            )
        except Exception:
            pass

    rows = []
    for line in text.splitlines():
        try:
            row = next(csv.reader([line], delimiter="\t"))
        except Exception:
            continue
        if len(row) < len(GKG_COLUMNS):
            row = row + [""] * (len(GKG_COLUMNS) - len(row))
        elif len(row) > len(GKG_COLUMNS):
            row = row[: len(GKG_COLUMNS) - 1] + ["\t".join(row[len(GKG_COLUMNS) - 1 :])]
        rows.append(row)
    return pd.DataFrame(rows, columns=GKG_COLUMNS)


def fetch_raw_zip_bucket(bucket_ts: datetime) -> tuple[int | None, str, pd.DataFrame]:
    url = gdelt_gkg_url(bucket_ts)
    tmp_path = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "mub30-gdelt-raw-csv-export/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            http_status = int(getattr(response, "status", 200))
            data = response.read()
        with tempfile.NamedTemporaryFile(prefix="gdelt_gkg_", suffix=".zip", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with zipfile.ZipFile(tmp_path) as zf:
            inner_name = zf.namelist()[0]
            return http_status, url, read_gkg_csv_robust(zf.read(inner_name))
    except urllib.error.HTTPError as exc:
        return int(exc.code), url, pd.DataFrame(columns=GKG_COLUMNS)
    except Exception as exc:
        print(f"Raw ZIP fetch failed for {format_timestamp(bucket_ts)}: {type(exc).__name__}: {exc}")
        return None, url, pd.DataFrame(columns=GKG_COLUMNS)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def normalize_bigquery_rows(df: pd.DataFrame, *, source_system: str) -> pd.DataFrame:
    rows = []
    for record in df.to_dict("records"):
        rows.append(
            normalize_row(
                record,
                source_system=source_system,
                bucket_ts=format_timestamp(record.get("bucket_ts")),
                source_hour=format_timestamp(record.get("source_hour")),
                sample_label=str(record.get("sample_label") or ""),
            )
        )
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def normalize_unprocessed_rows(
    df: pd.DataFrame,
    *,
    source_system: str,
    source_hour: str = "",
    bucket_ts: str = "",
    sample_label: str = "",
    raw_zip_http_status: int | str | None = "",
    raw_zip_source_url: str = "",
) -> pd.DataFrame:
    rows = []
    for record in df.to_dict("records"):
        row = {
            "source_system": source_system,
            "source_hour": format_timestamp(record.get("source_hour") or source_hour),
            "bucket_ts": format_timestamp(record.get("bucket_ts") or bucket_ts),
            "sample_label": str(record.get("sample_label") or sample_label),
            "raw_zip_http_status": record.get("raw_zip_http_status") or raw_zip_http_status,
            "raw_zip_source_url": record.get("raw_zip_source_url") or raw_zip_source_url,
        }
        for column in GKG_COLUMNS:
            row[column] = record.get(column, "")
        rows.append(row)
    return pd.DataFrame(rows, columns=RAW_GKG_EXPORT_COLUMNS)


def export_raw_zip_rows() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    rows = []
    unprocessed_frames = []
    status_rows = []
    for sample_label, timestamp_text in SAMPLE_BUCKETS:
        bucket_ts = parse_timestamp(timestamp_text)
        http_status, url, raw_df = fetch_raw_zip_bucket(bucket_ts)
        status_rows.append(
            {
                "sample_label": sample_label,
                "bucket_ts": format_timestamp(bucket_ts),
                "raw_zip_http_status": http_status,
                "raw_zip_source_url": url,
                "raw_zip_row_count": int(len(raw_df)),
            }
        )
        for record in raw_df.to_dict("records"):
            rows.append(
                normalize_row(
                    record,
                    source_system="raw_zip",
                    bucket_ts=format_timestamp(bucket_ts),
                    sample_label=sample_label,
                    raw_zip_http_status=http_status,
                    raw_zip_source_url=url,
                )
            )
        unprocessed_frames.append(
            normalize_unprocessed_rows(
                raw_df,
                source_system="raw_zip",
                bucket_ts=format_timestamp(bucket_ts),
                sample_label=sample_label,
                raw_zip_http_status=http_status,
                raw_zip_source_url=url,
            )
        )
    if unprocessed_frames:
        unprocessed = pd.concat(unprocessed_frames, ignore_index=True)
    else:
        unprocessed = pd.DataFrame(columns=RAW_GKG_EXPORT_COLUMNS)
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS), unprocessed, status_rows


def export_theme_match_rows(market_stress_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in market_stress_df.to_dict("records"):
        theme_tokens = split_v2themes(record.get("V2Themes"))
        matches = matched_theme_groups(theme_tokens, THEME_DICTIONARY)
        tone = parse_v2tone(record.get("V2Tone"))["parsed_tone"]
        for group_name, matched_themes in matches.items():
            for theme in matched_themes:
                rows.append(
                    {
                        "source_hour": record.get("source_hour", ""),
                        "bucket_ts": record.get("bucket_ts", ""),
                        "GKGRECORDID": record.get("GKGRECORDID", ""),
                        "DocumentIdentifier": record.get("DocumentIdentifier", ""),
                        "theme": theme,
                        "matched_dictionary_group": group_name,
                        "tone": tone,
                        "row_type": record.get("row_type", ""),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "source_hour",
            "bucket_ts",
            "GKGRECORDID",
            "DocumentIdentifier",
            "theme",
            "matched_dictionary_group",
            "tone",
            "row_type",
        ],
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def csv_stats(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "row_count": 0,
            "distinct_gkgrecordid_count": 0,
            "numeric_raw_compatible_rows": 0,
            "translated_t_rows": 0,
            "v2tone_null_rate": None,
            "v2themes_null_rate": None,
        }
    if "row_type" in df:
        numeric_rows = int((df.get("row_type") == "numeric_raw_compatible").sum())
        translated_rows = int((df.get("row_type") == "translated_T").sum())
    elif "GKGRECORDID" in df:
        numeric_rows = int(df["GKGRECORDID"].map(is_numeric_gkgrecordid).sum())
        translated_rows = int(df["GKGRECORDID"].map(is_translated_record).sum())
    else:
        numeric_rows = 0
        translated_rows = 0
    return {
        "row_count": int(len(df)),
        "distinct_gkgrecordid_count": int(df["GKGRECORDID"].dropna().nunique()) if "GKGRECORDID" in df else 0,
        "numeric_raw_compatible_rows": numeric_rows,
        "translated_t_rows": translated_rows,
        "v2tone_null_rate": float(df["V2Tone"].isna().mean()) if "V2Tone" in df else None,
        "v2themes_null_rate": float(df["V2Themes"].isna().mean()) if "V2Themes" in df else None,
    }


def build_manifest(
    outputs: list[tuple[str, Path, pd.DataFrame, str]],
    *,
    created_ts: str,
    manifest_prefix: str,
) -> pd.DataFrame:
    manifest_rows = []
    for filename, actual_path, df, description in outputs:
        stats = csv_stats(df)
        manifest_rows.append(
            {
                "file_path": f"{manifest_prefix.rstrip('/')}/{filename}",
                "actual_write_path": str(actual_path),
                "row_count": stats["row_count"],
                "distinct_gkgrecordid_count": stats["distinct_gkgrecordid_count"],
                "created_timestamp": created_ts,
                "source_query_bucket_description": description,
            }
        )
    return pd.DataFrame(manifest_rows)


def create_markdown_report(
    path: Path,
    *,
    manifest_df: pd.DataFrame,
    raw_zip_status_rows: list[dict[str, object]],
    output_prefix: str,
) -> None:
    sample_lines = "\n".join([f"- `{label}`: `{ts}`" for label, ts in SAMPLE_BUCKETS])
    hour_lines = "\n".join([f"- `{hour}` uses buckets `T-45`, `T-30`, `T-15`, and `T`." for hour in MARKET_STRESS_HOURS])
    output_lines = "\n".join(
        [f"- `{row.file_path}`: {int(row.row_count)} rows" for row in manifest_df.itertuples()]
    )
    zip_lines = "\n".join(
        [
            f"- `{row['bucket_ts']}` {row['sample_label']}: HTTP {row['raw_zip_http_status']}, "
            f"{row['raw_zip_row_count']} parsed rows"
            for row in raw_zip_status_rows
        ]
    )
    content = f"""# GDELT Raw CSV Export Audit - 2026-05-26

## K-12 Summary

Created bounded, research-only row-level CSV exports for manual inspection of raw GDELT GKG rows. The export includes selected sample buckets, raw ZIP diagnostic rows where available, and market-stress rows from selected feature hours. No production notebooks, workflows, feature tables, Saxo execution files, or model registry objects were changed.

## Sample Timestamps Exported

{sample_lines}

## Market-Stress Feature Hours

{hour_lines}

## Output CSV Paths

Base directory: `{output_prefix}`

{output_lines}

## Canonical Versus Diagnostic Rows

Numeric raw-compatible `GKGRECORDID` rows match raw GKG ZIP record ids such as `20230513000000-123`, so they remain the canonical row universe for comparable GDELT feature counts. Translated `-T` rows, such as `20230513000000-T123`, are exported for diagnostics and sensitivity review only because they inflate BigQuery counts relative to raw ZIP files.

## Raw ZIP Findings

{zip_lines}

## Market-Stress Theme Examples

Market-stress dictionary matches are exported in `gdelt_market_stress_theme_matches.csv`. Economically sensible examples include `WB_450_DEBT`, `ECON_DEBT`, `UNEMPLOYMENT`, `WB_1150_VOLATILITY`, `ECON_BANKRUPTCY`, and liquidity/default/debt themes.

## Noisy Or Generic Theme Examples

Generic/context-dependent examples to inspect manually include broad bank themes such as `WB_1235_CENTRAL_BANKS`, `ECON_CENTRALBANK`, `WB_1234_BANKING_INSTITUTIONS`, `WB_318_FINANCIAL_ARCHITECTURE_AND_BANKING`, and substring matches around `CREDIT` such as tax-credit or accreditation-like contexts.

## Production Safety Confirmation

This utility writes CSV files only under the research raw-audit export directory. It does not write production Delta tables, does not edit production notebooks, does not change Databricks workflows, does not touch Saxo execution, does not touch the model registry, and does not change canonical GDELT feature or backfill logic.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def print_validation(
    outputs: list[tuple[str, Path, pd.DataFrame, str]],
    raw_zip_status_rows: list[dict[str, object]],
) -> None:
    print("\nValidation summary")
    for filename, actual_path, df, _description in outputs:
        stats = csv_stats(df)
        print(
            json.dumps(
                {
                    "file": filename,
                    "path": str(actual_path),
                    **stats,
                },
                default=str,
            )
        )
    print("raw_zip_status_by_bucket")
    for row in raw_zip_status_rows:
        print(json.dumps(row, default=str))
    print(
        "production_safety_confirmation="
        "no production table/notebook/workflow/model/Saxo file was modified; "
        "only research CSV exports and the local markdown report were written."
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_prefix = args.manifest_path_prefix or args.output_dir
    created_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    print(f"Writing research-only GDELT raw CSV exports to: {output_dir}")
    print(f"BigQuery table: {args.bigquery_table}")
    print(f"BigQuery project: {args.bigquery_project}")

    sample_raw = run_bigquery(
        sample_bucket_bigquery_sql(args.bigquery_table, args.sample_cap_per_row_type),
        args.bigquery_project,
        args.bigquery_runner,
    )
    sample_rows = normalize_bigquery_rows(sample_raw, source_system="bigquery")
    sample_unprocessed_rows = normalize_unprocessed_rows(sample_raw, source_system="bigquery")

    raw_zip_rows, raw_zip_unprocessed_rows, raw_zip_status_rows = export_raw_zip_rows()

    market_stress_raw = run_bigquery(
        market_stress_bigquery_sql(args.bigquery_table),
        args.bigquery_project,
        args.bigquery_runner,
    )
    market_stress_rows = normalize_bigquery_rows(market_stress_raw, source_system="bigquery")
    market_stress_unprocessed_rows = normalize_unprocessed_rows(market_stress_raw, source_system="bigquery")
    theme_matches = export_theme_match_rows(market_stress_raw)

    outputs = [
        (
            "gdelt_bigquery_raw_gkg_rows_unprocessed.csv",
            output_dir / "gdelt_bigquery_raw_gkg_rows_unprocessed.csv",
            sample_unprocessed_rows,
            "BigQuery raw GKG rows for four requested buckets; original GDELT columns plus minimal source metadata.",
        ),
        (
            "gdelt_raw_zip_gkg_rows_unprocessed.csv",
            output_dir / "gdelt_raw_zip_gkg_rows_unprocessed.csv",
            raw_zip_unprocessed_rows,
            "Raw GDELT GKG ZIP rows for requested buckets; original raw CSV columns plus minimal source metadata.",
        ),
        (
            "gdelt_market_stress_raw_gkg_rows_unprocessed.csv",
            output_dir / "gdelt_market_stress_raw_gkg_rows_unprocessed.csv",
            market_stress_unprocessed_rows,
            "BigQuery raw GKG rows matching market-stress query for selected feature hours; original GDELT columns plus source metadata.",
        ),
        (
            "gdelt_raw_rows_sample_buckets.csv",
            output_dir / "gdelt_raw_rows_sample_buckets.csv",
            sample_rows,
            "BigQuery row-level sample for four requested buckets; capped per row_type.",
        ),
        (
            "gdelt_raw_zip_rows_sample_buckets.csv",
            output_dir / "gdelt_raw_zip_rows_sample_buckets.csv",
            raw_zip_rows,
            "Raw GDELT GKG ZIP diagnostic row export for requested sample buckets.",
        ),
        (
            "gdelt_market_stress_raw_rows.csv",
            output_dir / "gdelt_market_stress_raw_rows.csv",
            market_stress_rows,
            "BigQuery market-stress-matching rows for selected feature hours and their 15-minute source buckets.",
        ),
        (
            "gdelt_market_stress_theme_matches.csv",
            output_dir / "gdelt_market_stress_theme_matches.csv",
            theme_matches,
            "Exploded row-level V2Theme dictionary matches from selected market-stress rows.",
        ),
    ]

    for _filename, path, df, _description in outputs:
        write_csv(df, path)

    manifest = build_manifest(outputs, created_ts=created_ts, manifest_prefix=manifest_prefix)
    manifest_path = output_dir / "gdelt_raw_csv_export_manifest.csv"
    manifest_output_stub = (
        "gdelt_raw_csv_export_manifest.csv",
        manifest_path,
        manifest,
        "Manifest for generated research-only raw GDELT CSV exports.",
    )
    manifest = build_manifest([*outputs, manifest_output_stub], created_ts=created_ts, manifest_prefix=manifest_prefix)
    manifest_output = (
        "gdelt_raw_csv_export_manifest.csv",
        manifest_path,
        manifest,
        "Manifest for generated research-only raw GDELT CSV exports.",
    )
    manifest_outputs = [
        *outputs,
        manifest_output,
    ]
    manifest.loc[
        manifest["file_path"].str.endswith("gdelt_raw_csv_export_manifest.csv"),
        "row_count",
    ] = len(manifest)
    write_csv(manifest, manifest_path)

    create_markdown_report(
        Path(args.report_path),
        manifest_df=manifest,
        raw_zip_status_rows=raw_zip_status_rows,
        output_prefix=manifest_prefix,
    )

    print_validation(manifest_outputs, raw_zip_status_rows)
    print(f"markdown_report={args.report_path}")


if __name__ == "__main__":
    main()

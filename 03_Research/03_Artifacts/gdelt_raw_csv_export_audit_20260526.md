# GDELT Raw CSV Export Audit - 2026-05-26

## K-12 Summary

Created bounded, research-only row-level CSV exports for manual inspection of raw GDELT GKG rows. The export includes selected sample buckets, raw ZIP diagnostic rows where available, and market-stress rows from selected feature hours. No production notebooks, workflows, feature tables, Saxo execution files, or model registry objects were changed.

## Sample Timestamps Exported

- `normal_bucket`: `2023-05-13 00:00:00`
- `repaired_bucket`: `2023-09-21 23:45:00`
- `missing_outage_bucket`: `2025-06-18 00:00:00`
- `post_gap_bucket`: `2025-07-03 23:45:00`

## Market-Stress Feature Hours

- `2021-01-04 15:00:00` uses buckets `T-45`, `T-30`, `T-15`, and `T`.
- `2021-01-28 17:00:00` uses buckets `T-45`, `T-30`, `T-15`, and `T`.
- `2021-01-04 18:00:00` uses buckets `T-45`, `T-30`, `T-15`, and `T`.

## Output CSV Paths

Base directory: `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports`

- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_bigquery_raw_gkg_rows_unprocessed.csv`: 2500 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_raw_zip_gkg_rows_unprocessed.csv`: 5234 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_market_stress_raw_gkg_rows_unprocessed.csv`: 3866 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_raw_rows_sample_buckets.csv`: 2500 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_raw_zip_rows_sample_buckets.csv`: 5234 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_market_stress_raw_rows.csv`: 3866 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_market_stress_theme_matches.csv`: 66011 rows
- `/Volumes/pawan_quant_research_1/v4/market_data/research/news_flow/raw_audit/csv_exports/gdelt_raw_csv_export_manifest.csv`: 8 rows

## Canonical Versus Diagnostic Rows

Numeric raw-compatible `GKGRECORDID` rows match raw GKG ZIP record ids such as `20230513000000-123`, so they remain the canonical row universe for comparable GDELT feature counts. Translated `-T` rows, such as `20230513000000-T123`, are exported for diagnostics and sensitivity review only because they inflate BigQuery counts relative to raw ZIP files.

## Raw ZIP Findings

- `2023-05-13 00:00:00` normal_bucket: HTTP 200, 1678 parsed rows
- `2023-09-21 23:45:00` repaired_bucket: HTTP 200, 1799 parsed rows
- `2025-06-18 00:00:00` missing_outage_bucket: HTTP 404, 0 parsed rows
- `2025-07-03 23:45:00` post_gap_bucket: HTTP 200, 1757 parsed rows

## Market-Stress Theme Examples

Market-stress dictionary matches are exported in `gdelt_market_stress_theme_matches.csv`. Economically sensible examples include `WB_450_DEBT`, `ECON_DEBT`, `UNEMPLOYMENT`, `WB_1150_VOLATILITY`, `ECON_BANKRUPTCY`, and liquidity/default/debt themes.

## Noisy Or Generic Theme Examples

Generic/context-dependent examples to inspect manually include broad bank themes such as `WB_1235_CENTRAL_BANKS`, `ECON_CENTRALBANK`, `WB_1234_BANKING_INSTITUTIONS`, `WB_318_FINANCIAL_ARCHITECTURE_AND_BANKING`, and substring matches around `CREDIT` such as tax-credit or accreditation-like contexts.

## Production Safety Confirmation

This utility writes CSV files only under the research raw-audit export directory. It does not write production Delta tables, does not edit production notebooks, does not change Databricks workflows, does not touch Saxo execution, does not touch the model registry, and does not change canonical GDELT feature or backfill logic.

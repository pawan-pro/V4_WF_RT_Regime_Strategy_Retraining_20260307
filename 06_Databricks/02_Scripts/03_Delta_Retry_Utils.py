# Databricks notebook source
# Shared retry helpers for transient Delta Lake write conflicts.

import random
import time

RETRYABLE_DELTA_MARKERS = (
    "ConcurrentAppendException",
    "[DELTA_CONCURRENT_APPEND]",
    "ConcurrentWriteException",
    "[DELTA_CONCURRENT_WRITE]",
    "MetadataChangedException",
    "[DELTA_METADATA_CHANGED]",
)


def is_retryable_delta_conflict(exc):
    message = str(exc)
    return any(marker in message for marker in RETRYABLE_DELTA_MARKERS)


def run_with_retry(
    operation_name,
    operation,
    max_attempts=4,
    initial_wait_seconds=5,
    backoff_multiplier=2.0,
    max_wait_seconds=60,
    jitter_seconds=1.0,
):
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_retryable_delta_conflict(exc) or attempt >= max_attempts:
                raise

            base_wait = min(
                max_wait_seconds,
                initial_wait_seconds * (backoff_multiplier ** (attempt - 1)),
            )
            sleep_seconds = base_wait + random.uniform(0, jitter_seconds)
            print(
                f"{operation_name} hit a retryable Delta conflict on attempt "
                f"{attempt}/{max_attempts}: {exc}"
            )
            print(f"Retrying after {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
            attempt += 1


def execute_merge_sql_with_retry(merge_sql, operation_name, **retry_kwargs):
    return run_with_retry(
        operation_name=operation_name,
        operation=lambda: spark.sql(merge_sql),
        **retry_kwargs,
    )


def append_dataframe_with_retry(dataframe, table_name, operation_name=None, **retry_kwargs):
    def write_once():
        (
            dataframe.write.format("delta")
            .mode("append")
            .saveAsTable(table_name)
        )

    return run_with_retry(
        operation_name or f"append into {table_name}",
        write_once,
        **retry_kwargs,
    )


def overwrite_dataframe_with_retry(
    dataframe,
    table_name,
    options=None,
    operation_name=None,
    **retry_kwargs,
):
    def write_once():
        writer = dataframe.write.format("delta").mode("overwrite")
        for option_key, option_value in (options or {}).items():
            writer = writer.option(option_key, option_value)
        writer.saveAsTable(table_name)

    return run_with_retry(
        operation_name or f"overwrite {table_name}",
        write_once,
        **retry_kwargs,
    )

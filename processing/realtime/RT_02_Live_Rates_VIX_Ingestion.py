# Databricks notebook source
# RT_02: Live VIX Ingestion — FRED 1D PIT-Corrected
# Decision: fred_1d won A/B test (Sharpe 1.58 vs 1.50)
# Writes to bronze_rt_vix_1d (RT-namespaced, isolated from BT tables)
# Future upgrade realtime data from massive 

%pip install fredapi

from fredapi import Fred
import pandas as pd
from datetime import date
from pyspark.sql import functions as F

FRED_API_KEY = "f52bdf7341f23d0458820ecd806178db"
TARGET_TABLE = "pawan_quant_research_1.v4.bronze_rt_vix_1d"

fred = Fred(api_key=FRED_API_KEY)

def fetch_pit_latest(series_id: str, series_label: str) -> dict:
    """
    Fetch latest FRED daily value with PIT correction.
    Value published on day T is only available on T+1.
    Returns the most recent T-1 close (what the model is allowed to see today).
    """
    s = fred.get_series(series_id, observation_start="2016-01-01").dropna()
    # shift(1): T's value assigned to T+1 — same logic as BT pipeline
    s_pit = s.shift(1).dropna()
    latest_date  = s_pit.index[-1].to_pydatetime().replace(tzinfo=None)
    latest_value = float(s_pit.iloc[-1])
    return {
        "series":       series_label,
        "timestamp":    latest_date,
        "close":        latest_value,
        "source":       f"fred_{series_id.lower()}_pit_1d",
    }

# ── BIAS GUARD: confirm latest date < today ──
rows = []
for series_id, label in [("VIXCLS", "VIXCLS"), ("VXVCLS", "vix3m")]:
    row = fetch_pit_latest(series_id, label)
    assert row["timestamp"].date() < date.today(), \
        f"🚨 LOOKAHEAD: {label} date {row['timestamp'].date()} >= today {date.today()}"
    rows.append(row)
    print(f"✅ {label}: {row['close']:.2f} @ {row['timestamp'].date()} (PIT-correct)")

# ── CREATE TABLE IF NOT EXISTS ──
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        series       STRING,
        timestamp    TIMESTAMP,
        close        DOUBLE,
        source       STRING,
        ingestion_ts TIMESTAMP
    )
    USING DELTA
""")

# ── MERGE (idempotent) ──
sdf = spark.createDataFrame(rows).withColumn("ingestion_ts", F.current_timestamp())
sdf.createOrReplaceTempView("incoming_rt_vix")
spark.sql(f"""
    MERGE INTO {TARGET_TABLE} t
    USING incoming_rt_vix s
    ON t.series = s.series AND t.timestamp = s.timestamp
    WHEN NOT MATCHED THEN INSERT *
""")

spark.sql(f"""
    SELECT series, MAX(timestamp) as last_ts, MAX(close) as last_close
    FROM {TARGET_TABLE} GROUP BY series ORDER BY series
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Macro (RT_02 supplement — Finnhub latest release)
# MAGIC
# MAGIC The macro z-scores in Silver are forward-filled 24/7, so **no new macro event between releases** — the last row from `bronze_macro_events` is the correct PIT value. RT_03 will pull that directly. **No new ingestion needed for macro.**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Full Deployment Sequence
# MAGIC ```
# MAGIC RT_01 ✅  → bronze_rt_market_1h   (SPY, QQQ last bar)
# MAGIC RT_02 🔲  → bronze_rt_vix_1h      (IVIX, IVIX3M last bar)  ← NEXT
# MAGIC RT_03 🔲  → silver_rt_1h          (join above + macro ffill + z-score on 1 new row)
# MAGIC RT_04 🔲  → regime_signal_rt      (load joblib model → predict on silver_rt row)
# MAGIC RT_05 🔲  → paper_trade_log       (regime → position logic)
# MAGIC RT_06 🔲  → signal_audit_log      (write every decision with features)
# MAGIC RT_07 🔲  → dashboard_feed        (latest regime + prices for display)

# COMMAND ----------


# Databricks notebook source
# RT_11: Read-only Saxo broker trade/fill/transaction history ingestion
#
# Production-safety scope:
# - Reads the existing live Saxo token state only.
# - Does not refresh, write, or rotate tokens.
# - Uses GET-only Saxo OpenAPI endpoints.
# - Does not place, modify, precheck, or cancel orders.
# - Writes only pawan_quant_research_1.v4.bronze_rt_saxo_broker_trade_history.

# COMMAND ----------

# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

# COMMAND ----------

import hashlib
import json
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Constants copied from the read-only portions of RT_08 Saxo execution.
# ---------------------------------------------------------------------------
CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

BASE_URL = "https://gateway.saxobank.com/openapi"
GATEWAY_ROOT = "https://gateway.saxobank.com"
CLIENT_KEY = "hLMMvwa3Mzmiem2AMhpKwA=="
ACCOUNT_KEY = "I|QvYu7A1ynLbi6kR32qfQ=="
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"

TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_broker_trade_history"

SAXO_RETRY_MAX_ATTEMPTS = 4
SAXO_RETRY_BASE_SECONDS = 1.0
SAXO_RETRY_MAX_SECONDS = 8.0

DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_TOP = 500
MAX_PAGES_PER_ENDPOINT = 20

try:
    dbutils.widgets.text("lookback_days", str(DEFAULT_LOOKBACK_DAYS))
    dbutils.widgets.text("from_date", "")
    dbutils.widgets.text("to_date", "")
    dbutils.widgets.text("max_pages_per_endpoint", str(MAX_PAGES_PER_ENDPOINT))
    LOOKBACK_DAYS = int(dbutils.widgets.get("lookback_days") or DEFAULT_LOOKBACK_DAYS)
    FROM_DATE_PARAM = (dbutils.widgets.get("from_date") or "").strip()
    TO_DATE_PARAM = (dbutils.widgets.get("to_date") or "").strip()
    MAX_PAGES = int(dbutils.widgets.get("max_pages_per_endpoint") or MAX_PAGES_PER_ENDPOINT)
except Exception:
    LOOKBACK_DAYS = DEFAULT_LOOKBACK_DAYS
    FROM_DATE_PARAM = ""
    TO_DATE_PARAM = ""
    MAX_PAGES = MAX_PAGES_PER_ENDPOINT

if LOOKBACK_DAYS <= 0:
    raise ValueError("lookback_days must be positive.")
if MAX_PAGES <= 0:
    raise ValueError("max_pages_per_endpoint must be positive.")

today_utc = datetime.now(timezone.utc).date()
FROM_DATE = FROM_DATE_PARAM or (today_utc - timedelta(days=LOOKBACK_DAYS)).isoformat()
TO_DATE = TO_DATE_PARAM or today_utc.isoformat()


# ---------------------------------------------------------------------------
# Table schema
# ---------------------------------------------------------------------------
BROKER_HISTORY_SCHEMA = StructType(
    [
        StructField("broker_event_id", StringType(), False),
        StructField("broker_event_ts", TimestampType(), True),
        StructField("order_id", StringType(), True),
        StructField("trade_id", StringType(), True),
        StructField("position_id", StringType(), True),
        StructField("account_key", StringType(), True),
        StructField("uic", StringType(), True),
        StructField("symbol", StringType(), True),
        StructField("asset_type", StringType(), True),
        StructField("buy_sell", StringType(), True),
        StructField("quantity", DoubleType(), True),
        StructField("price", DoubleType(), True),
        StructField("gross_notional", DoubleType(), True),
        StructField("commission", DoubleType(), True),
        StructField("fees", DoubleType(), True),
        StructField("fx_cost", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("status", StringType(), True),
        StructField("raw_json", StringType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
        StructField("api_endpoint", StringType(), True),
        StructField("available_cost_fields", StringType(), True),
    ]
)


def ensure_target_table() -> None:
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            broker_event_id STRING NOT NULL,
            broker_event_ts TIMESTAMP,
            order_id STRING,
            trade_id STRING,
            position_id STRING,
            account_key STRING,
            uic STRING,
            symbol STRING,
            asset_type STRING,
            buy_sell STRING,
            quantity DOUBLE,
            price DOUBLE,
            gross_notional DOUBLE,
            commission DOUBLE,
            fees DOUBLE,
            fx_cost DOUBLE,
            currency STRING,
            status STRING,
            raw_json STRING,
            source STRING,
            ingestion_ts TIMESTAMP,
            api_endpoint STRING,
            available_cost_fields STRING
        )
        USING DELTA
        """
    )
    current = {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {TARGET_TABLE}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }
    desired = {
        "broker_event_id": "STRING",
        "broker_event_ts": "TIMESTAMP",
        "order_id": "STRING",
        "trade_id": "STRING",
        "position_id": "STRING",
        "account_key": "STRING",
        "uic": "STRING",
        "symbol": "STRING",
        "asset_type": "STRING",
        "buy_sell": "STRING",
        "quantity": "DOUBLE",
        "price": "DOUBLE",
        "gross_notional": "DOUBLE",
        "commission": "DOUBLE",
        "fees": "DOUBLE",
        "fx_cost": "DOUBLE",
        "currency": "STRING",
        "status": "STRING",
        "raw_json": "STRING",
        "source": "STRING",
        "ingestion_ts": "TIMESTAMP",
        "api_endpoint": "STRING",
        "available_cost_fields": "STRING",
    }
    for column, dtype in desired.items():
        if column not in current:
            spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {column} {dtype}")


# ---------------------------------------------------------------------------
# Saxo token and request helpers. GET-only by construction in this notebook.
# ---------------------------------------------------------------------------
class SaxoTemporaryThrottleError(RuntimeError):
    pass


def safe_json_loads(payload: Any) -> Optional[Any]:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(payload)
    except Exception:
        return None


def load_token() -> Optional[dict]:
    try:
        return json.loads(dbutils.fs.head(TOKEN_STATE_PATH))
    except Exception:
        return None


def get_access_token_readonly() -> str:
    state = load_token()
    if not state:
        raise RuntimeError("Token state missing. Run Saxo OAuth bootstrap/refresher outside this notebook.")

    now = time.time()
    if state["expires_at"] < now:
        raise RuntimeError("Token expired. Refresher job is not current; this notebook will not refresh it.")

    return state["access_token"]


def extract_error_info(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    error_info = payload.get("ErrorInfo")
    if not isinstance(error_info, dict):
        return None
    error_code = error_info.get("ErrorCode")
    message = error_info.get("Message")
    if error_code in (None, "", "None", "Ok") and not message:
        return None
    return {
        "code": error_code or "UnknownError",
        "message": message or "",
    }


def is_rate_limited(status_code: int, payload: Any, response_text: str) -> bool:
    if status_code == 429:
        return True
    error_info = extract_error_info(payload)
    if error_info and error_info["code"] == "RateLimitExceeded":
        return True
    return "rate limit" in (response_text or "").lower()


def saxo_get(url: str, headers: dict, params: Optional[dict] = None, action_name: str = "request"):
    last_status = None
    last_payload = {}
    last_text = ""

    for attempt in range(1, SAXO_RETRY_MAX_ATTEMPTS + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        response_text = resp.text or ""
        response_payload = safe_json_loads(response_text) or {}

        if not is_rate_limited(resp.status_code, response_payload, response_text):
            return resp, response_payload, attempt - 1

        last_status = resp.status_code
        last_payload = response_payload
        last_text = response_text

        if attempt == SAXO_RETRY_MAX_ATTEMPTS:
            error_info = extract_error_info(response_payload) or {
                "code": "RateLimitExceeded",
                "message": response_text[:500],
            }
            raise SaxoTemporaryThrottleError(
                f"Saxo {action_name} deferred after {attempt} attempts: "
                f"{error_info['code']} - {error_info['message']}"
            )

        wait_seconds = min(
            SAXO_RETRY_BASE_SECONDS * (2 ** (attempt - 1)),
            SAXO_RETRY_MAX_SECONDS,
        ) + random.uniform(0.0, 0.5)
        print(f"Saxo {action_name} rate-limited on attempt {attempt}; sleeping {wait_seconds:.1f}s")
        time.sleep(wait_seconds)

    raise SaxoTemporaryThrottleError(
        f"Saxo {action_name} deferred: status={last_status} payload={json.dumps(last_payload)[:500] or last_text[:500]}"
    )


def resolve_next_url(next_value: Optional[str]) -> Optional[str]:
    if not next_value:
        return None
    if next_value.startswith("http"):
        return next_value
    return urljoin(GATEWAY_ROOT, next_value)


def fetch_collection(
    headers: dict,
    source_name: str,
    endpoint: str,
    params: Optional[dict] = None,
    max_pages: int = MAX_PAGES,
) -> tuple[list[dict], list[dict]]:
    rows = []
    errors = []
    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    page_params = params or {}

    for page in range(max_pages):
        try:
            resp, payload, retry_count = saxo_get(
                url,
                headers=headers,
                params=page_params if page == 0 else None,
                action_name=source_name,
            )
        except Exception as exc:
            errors.append({"source": source_name, "endpoint": endpoint, "error": str(exc)})
            break

        if retry_count:
            print(f"Saxo {source_name} succeeded after {retry_count} retries")

        if resp.status_code in {400, 401, 403, 404, 405}:
            errors.append(
                {
                    "source": source_name,
                    "endpoint": endpoint,
                    "status_code": resp.status_code,
                    "response": (resp.text or "")[:1000],
                }
            )
            break
        resp.raise_for_status()

        if isinstance(payload, dict) and isinstance(payload.get("Data"), list):
            data_rows = payload["Data"]
        elif isinstance(payload, list):
            data_rows = payload
        elif isinstance(payload, dict) and payload:
            data_rows = [payload]
        else:
            data_rows = []

        for item in data_rows:
            if isinstance(item, dict):
                rows.append(item)

        next_url = resolve_next_url(payload.get("__next") if isinstance(payload, dict) else None)
        if not next_url:
            break
        url = next_url
        page_params = None

    return rows, errors


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def maybe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_float = float(value)
    except Exception:
        return None
    if value_float != value_float:
        return None
    return value_float


def get_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def first_path(payload: dict, paths: list[str]) -> Any:
    for path in paths:
        value = get_path(payload, path)
        if value is not None:
            return value
    return None


def parse_ts(value: Any) -> Optional[datetime]:
    text = maybe_str(value)
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception:
        pass
    try:
        parsed_date = date.fromisoformat(text[:10])
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day)
    except Exception:
        return None


def flatten_keys(payload: Any, prefix: str = "") -> dict[str, Any]:
    output = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                output.update(flatten_keys(value, full_key))
            else:
                output[full_key] = value
    return output


def sum_matching_fields(payload: dict, markers: tuple[str, ...], exclude_markers: tuple[str, ...] = ()) -> tuple[Optional[float], list[str]]:
    total = 0.0
    matched = []
    for key, value in flatten_keys(payload).items():
        lowered = key.lower()
        if not any(marker in lowered for marker in markers):
            continue
        if exclude_markers and any(marker in lowered for marker in exclude_markers):
            continue
        value_float = maybe_float(value)
        if value_float is None:
            continue
        total += abs(value_float)
        matched.append(key)
    return (total if matched else None), matched


def normalize_buy_sell(payload: dict) -> Optional[str]:
    raw = maybe_str(
        first_path(
            payload,
            [
                "BuySell",
                "TradeEventType",
                "TransactionType",
                "Direction",
                "Side",
                "Order.BuySell",
            ],
        )
    )
    if raw:
        lowered = raw.lower()
        if lowered in {"buy", "bought", "buying"}:
            return "Buy"
        if lowered in {"sell", "sold", "selling"}:
            return "Sell"
    amount = maybe_float(first_path(payload, ["Amount", "Quantity", "PositionBase.Amount"]))
    if amount is not None:
        if amount > 0:
            return "Buy"
        if amount < 0:
            return "Sell"
    return raw


def normalize_event(payload: dict, source_name: str, api_endpoint: str, ingestion_ts: datetime) -> dict:
    raw_json = json.dumps(payload, sort_keys=True, default=str)

    broker_event_ts = parse_ts(
        first_path(
            payload,
            [
                "TradeExecutionTime",
                "ExecutionTime",
                "ActivityTime",
                "BookingDate",
                "TradeDate",
                "OrderTime",
                "Created",
                "CreatedTime",
                "ClosedPosition.Created",
                "ClosedPosition.ClosedTime",
                "PositionBase.OpenPriceTimestamp",
                "ValueDate",
            ],
        )
    )
    order_id = maybe_str(first_path(payload, ["OrderId", "SourceOrderId", "PositionBase.SourceOrderId"]))
    trade_id = maybe_str(first_path(payload, ["TradeId", "TradeID", "ExecutionId", "ExecutionID"]))
    position_id = maybe_str(
        first_path(
            payload,
            [
                "PositionId",
                "ClosedPositionId",
                "NetPositionId",
                "PositionBase.PositionId",
                "ClosedPosition.PositionId",
            ],
        )
    )
    account_key = maybe_str(first_path(payload, ["AccountKey", "Account.Key", "AccountId"]))
    uic = maybe_str(first_path(payload, ["Uic", "PositionBase.Uic", "Instrument.Uic"]))
    symbol = maybe_str(
        first_path(
            payload,
            [
                "InstrumentSymbol",
                "UnderlyingInstrumentSymbol",
                "Symbol",
                "DisplayAndFormat.Symbol",
                "PositionBase.Symbol",
                "Instrument.Description",
            ],
        )
    )
    asset_type = maybe_str(first_path(payload, ["AssetType", "PositionBase.AssetType", "Instrument.AssetType"]))
    quantity = maybe_float(
        first_path(
            payload,
            [
                "Amount",
                "Quantity",
                "FilledAmount",
                "ExecutedAmount",
                "PositionBase.Amount",
                "ClosedPosition.Amount",
            ],
        )
    )
    if quantity is not None:
        quantity = abs(quantity)
    price = maybe_float(
        first_path(
            payload,
            [
                "Price",
                "ExecutionPrice",
                "AveragePrice",
                "AvgPrice",
                "CleanPrice",
                "DirtyPrice",
                "OpenPrice",
                "PositionBase.OpenPrice",
            ],
        )
    )
    gross_notional = maybe_float(
        first_path(
            payload,
            [
                "GrossNotional",
                "TradedValue",
                "TradeValue",
                "Value",
                "MarketValue",
                "PositionView.MarketValue",
                "NetPositionView.MarketValue",
            ],
        )
    )
    if gross_notional is None and quantity is not None and price is not None:
        gross_notional = abs(quantity * price)
    elif gross_notional is not None:
        gross_notional = abs(gross_notional)

    commission, commission_fields = sum_matching_fields(payload, ("commission",))
    fees, fee_fields = sum_matching_fields(
        payload,
        ("fee", "stampduty", "exchangefee", "clearingfee", "tax"),
        exclude_markers=("profile",),
    )
    fx_cost, fx_fields = sum_matching_fields(payload, ("fx", "currencycut", "conversion"))

    currency = maybe_str(
        first_path(
            payload,
            [
                "Currency",
                "AccountCurrency",
                "ClientCurrency",
                "InstrumentCurrency",
                "PriceCurrency",
                "DisplayAndFormat.Currency",
            ],
        )
    )
    status = maybe_str(first_path(payload, ["Status", "SubStatus", "OrderStatus", "TradeBarrierEventStatus"]))

    preferred_id = first_present(
        trade_id,
        order_id,
        position_id,
        maybe_str(first_path(payload, ["BookingId", "LogId", "SequenceId", "ActivityId"])),
    )
    event_identity = preferred_id or hashlib.sha256(f"{raw_json}|{broker_event_ts}".encode("utf-8")).hexdigest()
    broker_event_id = f"{source_name}:{event_identity}"

    return {
        "broker_event_id": broker_event_id,
        "broker_event_ts": broker_event_ts,
        "order_id": order_id,
        "trade_id": trade_id,
        "position_id": position_id,
        "account_key": account_key,
        "uic": uic,
        "symbol": symbol,
        "asset_type": asset_type,
        "buy_sell": normalize_buy_sell(payload),
        "quantity": quantity,
        "price": price,
        "gross_notional": gross_notional,
        "commission": commission,
        "fees": fees,
        "fx_cost": fx_cost,
        "currency": currency,
        "status": status,
        "raw_json": raw_json,
        "source": source_name,
        "ingestion_ts": ingestion_ts,
        "api_endpoint": api_endpoint,
        "available_cost_fields": ",".join(sorted(set(commission_fields + fee_fields + fx_fields))) or None,
    }


# ---------------------------------------------------------------------------
# Endpoint plan
# ---------------------------------------------------------------------------
def endpoint_plan() -> list[dict]:
    return [
        {
            "source": "hist_transactions",
            "endpoint": "/hist/v1/transactions",
            "params": {
                "$top": DEFAULT_TOP,
                "AccountKeys": ACCOUNT_KEY,
                "ClientKey": CLIENT_KEY,
                "FromDate": FROM_DATE,
                "ToDate": TO_DATE,
            },
        },
        {
            "source": "cs_reports_trades",
            "endpoint": f"/cs/v1/reports/trades/{CLIENT_KEY}",
            "params": {
                "$top": DEFAULT_TOP,
                "AccountKey": ACCOUNT_KEY,
                "FromDate": FROM_DATE,
                "ToDate": TO_DATE,
            },
        },
        {
            "source": "cs_audit_orderactivities",
            "endpoint": "/cs/v1/audit/orderactivities",
            "params": {
                "$top": DEFAULT_TOP,
                "AccountKey": ACCOUNT_KEY,
                "ClientKey": CLIENT_KEY,
                "EntryType": "All",
                "FieldGroups": "DisplayAndFormat",
                "FromDateTime": f"{FROM_DATE}T00:00:00Z",
                "ToDateTime": f"{TO_DATE}T23:59:59Z",
            },
        },
        {
            "source": "port_closedpositions_me",
            "endpoint": "/port/v1/closedpositions/me",
            "params": {
                "$top": DEFAULT_TOP,
                "FieldGroups": "DisplayAndFormat,Costs",
            },
        },
        {
            "source": "port_orders",
            "endpoint": "/port/v1/orders",
            "params": {
                "$top": DEFAULT_TOP,
                "AccountKey": ACCOUNT_KEY,
                "ClientKey": CLIENT_KEY,
                "FieldGroups": "DisplayAndFormat",
            },
        },
    ]


def collect_broker_events() -> tuple[list[dict], list[dict]]:
    access_token = get_access_token_readonly()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    ingestion_ts = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    errors = []

    for spec in endpoint_plan():
        print(f"Fetching Saxo broker history source={spec['source']} endpoint={spec['endpoint']}")
        payload_rows, endpoint_errors = fetch_collection(
            headers=headers,
            source_name=spec["source"],
            endpoint=spec["endpoint"],
            params=spec.get("params"),
            max_pages=MAX_PAGES,
        )
        print(f"Fetched {len(payload_rows)} rows from {spec['source']}")
        errors.extend(endpoint_errors)
        for payload in payload_rows:
            rows.append(normalize_event(payload, spec["source"], spec["endpoint"], ingestion_ts))

    return rows, errors


def write_broker_events(rows: list[dict]) -> None:
    ensure_target_table()
    if not rows:
        print("No broker history rows fetched; target table left unchanged.")
        return

    sdf = spark.createDataFrame(rows, schema=BROKER_HISTORY_SCHEMA)
    incoming_count = sdf.count()
    dedupe_window = Window.partitionBy("broker_event_id").orderBy(
        F.col("broker_event_ts").desc_nulls_last(),
        F.col("ingestion_ts").desc_nulls_last(),
        F.col("raw_json").desc_nulls_last(),
    )
    sdf = (
        sdf.withColumn("_rn", F.row_number().over(dedupe_window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    deduped_count = sdf.count()
    if deduped_count != incoming_count:
        print(
            f"Deduplicated incoming broker history rows before MERGE: "
            f"{incoming_count} -> {deduped_count}"
        )
    sdf.createOrReplaceTempView("incoming_saxo_broker_trade_history")
    merge_sql = f"""
        MERGE INTO {TARGET_TABLE} t
        USING incoming_saxo_broker_trade_history s
        ON t.broker_event_id = s.broker_event_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """
    execute_merge_sql_with_retry(
        merge_sql,
        operation_name=f"merge Saxo broker trade history into {TARGET_TABLE}",
    )


def display_validations(endpoint_errors: list[dict]) -> None:
    print(f"Target table: {TARGET_TABLE}")
    print(f"Date window: {FROM_DATE} to {TO_DATE}")
    print("Endpoint errors or unavailable endpoints:")
    if endpoint_errors:
        for error in endpoint_errors:
            print(json.dumps(error, default=str)[:1200])
    else:
        print("None")

    print("Row count")
    display(spark.sql(f"SELECT COUNT(*) AS row_count FROM {TARGET_TABLE}"))

    print("Latest 20 broker events")
    display(
        spark.sql(
            f"""
            SELECT
                broker_event_ts,
                source,
                order_id,
                trade_id,
                position_id,
                uic,
                symbol,
                asset_type,
                buy_sell,
                quantity,
                price,
                gross_notional,
                commission,
                fees,
                fx_cost,
                currency,
                status
            FROM {TARGET_TABLE}
            ORDER BY broker_event_ts DESC NULLS LAST, ingestion_ts DESC
            LIMIT 20
            """
        )
    )

    print("Counts by endpoint/source")
    display(
        spark.sql(
            f"""
            SELECT source, api_endpoint, COUNT(*) AS rows
            FROM {TARGET_TABLE}
            GROUP BY source, api_endpoint
            ORDER BY rows DESC, source
            """
        )
    )

    print("SPY/QQQ broker events")
    display(
        spark.sql(
            f"""
            SELECT
                broker_event_ts,
                source,
                order_id,
                trade_id,
                position_id,
                symbol,
                asset_type,
                buy_sell,
                quantity,
                price,
                gross_notional,
                commission,
                fees,
                fx_cost,
                currency,
                status
            FROM {TARGET_TABLE}
            WHERE upper(coalesce(symbol, '')) RLIKE '(^|[^A-Z])(SPY|QQQ)([^A-Z]|$)'
               OR uic IN ('36590', '4328771')
            ORDER BY broker_event_ts DESC NULLS LAST, ingestion_ts DESC
            LIMIT 100
            """
        )
    )

    print("Available commission/fee/fx fields")
    display(
        spark.sql(
            f"""
            SELECT source, available_cost_fields, COUNT(*) AS rows
            FROM {TARGET_TABLE}
            WHERE available_cost_fields IS NOT NULL
            GROUP BY source, available_cost_fields
            ORDER BY rows DESC, source
            LIMIT 100
            """
        )
    )


def main() -> None:
    rows, endpoint_errors = collect_broker_events()
    write_broker_events(rows)
    display_validations(endpoint_errors)


main()

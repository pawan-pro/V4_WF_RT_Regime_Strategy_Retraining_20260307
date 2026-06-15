# Databricks notebook source
# RT_08: Saxo trade execution / shadow account snapshot

# MAGIC %run ../../06_Databricks/02_Scripts/02_Runtime_Instrument_Config
# MAGIC %run ../../06_Databricks/02_Scripts/03_Delta_Retry_Utils

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

AUTH_BASE_URL = "https://live.logonvalidation.net"
BASE_URL = "https://gateway.saxobank.com/openapi"
REDIRECT_URI = "https://localhost/callback"

CLIENT_ID = "5b0974db36f54f3b96ec07b8f5fc657d"
CLIENT_SECRET = "8a7f591c89ca4f0299256b041ba891e2"
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"

EXPECTED_LIVE_ACCOUNT_ID = "33888/261196"
EXPECTED_LIVE_CURRENCY = "USD"
DEFAULT_ACCOUNT_KEY_SECRET_SCOPE = "saxo"
DEFAULT_ACCOUNT_KEY_SECRET_KEY = "usd_account_key"
DEFAULT_CLIENT_KEY_SECRET_SCOPE = "saxo"
DEFAULT_CLIENT_KEY_SECRET_KEY = "client_key"

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"

NOTIONAL_FRACTION = 0.95
ALIGNMENT_TOLERANCE_PCT = 0.02
SAXO_RETRY_MAX_ATTEMPTS = 4
SAXO_RETRY_BASE_SECONDS = 1.0
SAXO_RETRY_MAX_SECONDS = 8.0

runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]
SECONDARY_SYMBOL = runtime_config["secondary_symbol"]
PRICE_SYMBOLS = runtime_config["market_symbols"] or [PRIMARY_SYMBOL, SECONDARY_SYMBOL]

SAXO_ASSET_BY_SYMBOL = {
    "QQQ": {"symbol": "QQQ", "Uic": 4328771, "AssetType": "Etf"},
    "SPY": {"symbol": "SPY", "Uic": 36590, "AssetType": "Etf"},
}

TRADE_COLUMNS = {
    "trade_ts": "TIMESTAMP",
    "signal_ts": "TIMESTAMP",
    "regime_int": "INT",
    "regime_name": "STRING",
    "target_position": "STRING",
    "target_ticker": "STRING",
    "switched": "BOOLEAN",
    "transaction_cost": "DOUBLE",
    "orders_placed": "STRING",
    "spy_close": "DOUBLE",
    "qqq_close": "DOUBLE",
    "account_balance": "DOUBLE",
    "account_equity": "DOUBLE",
    "account_cash": "DOUBLE",
    "last_equity": "DOUBLE",
    "hourly_pnl": "DOUBLE",
    "hourly_pnl_pct": "DOUBLE",
    "day_pnl": "DOUBLE",
    "day_pnl_pct": "DOUBLE",
    "positions_json": "STRING",
    "positions_count": "INT",
    "positions_market_value": "DOUBLE",
    "positions_unrealized_pl": "DOUBLE",
    "positions_unrealized_plpc": "DOUBLE",
    "execution_mode": "STRING",
    "source": "STRING",
    "ingestion_ts": "TIMESTAMP",
}

NY_TZ = ZoneInfo("America/New_York")


class SaxoTemporaryThrottleError(RuntimeError):
    pass


def widget_text(name: str, default: str) -> str:
    try:
        dbutils.widgets.text(name, default)
        return (dbutils.widgets.get(name) or default).strip()
    except Exception:
        return default


try:
    dbutils.widgets.dropdown("execution_mode", "shadow", ["shadow", "live"])
    EXECUTION_MODE = (dbutils.widgets.get("execution_mode") or "shadow").strip().lower()
except Exception:
    EXECUTION_MODE = "shadow"

SAXO_ACCOUNT_KEY_SECRET_SCOPE = widget_text(
    "saxo_account_key_secret_scope",
    DEFAULT_ACCOUNT_KEY_SECRET_SCOPE,
)
SAXO_ACCOUNT_KEY_SECRET_KEY = widget_text(
    "saxo_account_key_secret_key",
    DEFAULT_ACCOUNT_KEY_SECRET_KEY,
)
SAXO_CLIENT_KEY_SECRET_SCOPE = widget_text(
    "saxo_client_key_secret_scope",
    DEFAULT_CLIENT_KEY_SECRET_SCOPE,
)
SAXO_CLIENT_KEY_SECRET_KEY = widget_text(
    "saxo_client_key_secret_key",
    DEFAULT_CLIENT_KEY_SECRET_KEY,
)

if EXECUTION_MODE not in {"shadow", "live"}:
    raise ValueError(f"Unsupported execution_mode: {EXECUTION_MODE}")


def resolve_saxo_account_key() -> str:
    if not SAXO_ACCOUNT_KEY_SECRET_SCOPE or not SAXO_ACCOUNT_KEY_SECRET_KEY:
        raise RuntimeError("Saxo account key secret scope/key is required.")
    try:
        account_key = dbutils.secrets.get(
            SAXO_ACCOUNT_KEY_SECRET_SCOPE,
            SAXO_ACCOUNT_KEY_SECRET_KEY,
        ).strip()
    except Exception as exc:
        raise RuntimeError("Saxo account key secret is not readable.") from exc
    if not account_key:
        raise RuntimeError("Saxo account key secret is empty.")
    return account_key


def resolve_saxo_client_key() -> str:
    if not SAXO_CLIENT_KEY_SECRET_SCOPE or not SAXO_CLIENT_KEY_SECRET_KEY:
        raise RuntimeError("Saxo client key secret scope/key is required.")
    try:
        client_key = dbutils.secrets.get(
            SAXO_CLIENT_KEY_SECRET_SCOPE,
            SAXO_CLIENT_KEY_SECRET_KEY,
        ).strip()
    except Exception:
        client_key = os.environ.get("SAXO_CLIENT_KEY", "").strip()
    if not client_key:
        raise RuntimeError("Saxo client key is not configured in saxo/client_key or SAXO_CLIENT_KEY.")
    return client_key


ACCOUNT_KEY = resolve_saxo_account_key()
CLIENT_KEY = resolve_saxo_client_key()
REDACTION_SECRETS = [ACCOUNT_KEY, CLIENT_KEY]
CONFIGURED_ACCOUNT_ID = None


def redact_text(text):
    redacted = str(text)
    for secret in REDACTION_SECRETS:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r'("access_token"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    redacted = re.sub(r'("refresh_token"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    redacted = re.sub(r'("AccountKey"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    return redacted


def redact_account_key(value):
    if not value:
        return None
    return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "[REDACTED]"


def redact_secret(value):
    if not value:
        return None
    return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "[REDACTED]"


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def first_path(payload, *paths):
    for path in paths:
        value = payload
        for part in path.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if value is not None:
            return value
    return None


def account_key_from_payload(payload) -> Optional[str]:
    value = first_path(
        payload,
        "AccountKey",
        "Account.AccountKey",
        "PositionBase.AccountKey",
        "NetPositionBase.AccountKey",
        "Order.AccountKey",
    )
    return str(value) if value is not None else None


def account_id_from_payload(payload) -> Optional[str]:
    value = first_path(
        payload,
        "AccountId",
        "Account.AccountId",
        "PositionBase.AccountId",
        "NetPositionBase.AccountId",
        "Order.AccountId",
        "AccountNumber",
    )
    return str(value) if value is not None else None


def payload_matches_configured_account(payload) -> bool:
    payload_account_key = account_key_from_payload(payload)
    payload_account_id = account_id_from_payload(payload)
    if payload_account_key is not None:
        return payload_account_key == ACCOUNT_KEY
    if payload_account_id is not None and CONFIGURED_ACCOUNT_ID is not None:
        return payload_account_id == CONFIGURED_ACCOUNT_ID
    return False


def uic_from_payload(payload) -> Optional[int]:
    value = first_path(payload, "Uic", "PositionBase.Uic", "NetPositionBase.Uic", "Order.Uic")
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def symbol_from_payload(payload) -> Optional[str]:
    value = first_path(
        payload,
        "Symbol",
        "DisplayAndFormat.Symbol",
        "PositionBase.Symbol",
        "NetPositionBase.Symbol",
        "Order.Symbol",
    )
    return str(value).upper() if value else None


def safe_json_loads(payload):
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        return None


def existing_columns(table_name):
    if not spark.catalog.tableExists(table_name):
        return set()
    return {
        row["col_name"]
        for row in spark.sql(f"DESCRIBE {table_name}").collect()
        if row["col_name"] and not row["col_name"].startswith("#")
    }


def ensure_trade_table():
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            trade_ts TIMESTAMP,
            signal_ts TIMESTAMP,
            regime_int INT,
            regime_name STRING,
            target_position STRING,
            target_ticker STRING,
            switched BOOLEAN,
            transaction_cost DOUBLE,
            orders_placed STRING,
            spy_close DOUBLE,
            qqq_close DOUBLE,
            account_balance DOUBLE,
            account_equity DOUBLE,
            account_cash DOUBLE,
            last_equity DOUBLE,
            hourly_pnl DOUBLE,
            hourly_pnl_pct DOUBLE,
            day_pnl DOUBLE,
            day_pnl_pct DOUBLE,
            positions_json STRING,
            positions_count INT,
            positions_market_value DOUBLE,
            positions_unrealized_pl DOUBLE,
            positions_unrealized_plpc DOUBLE,
            execution_mode STRING,
            source STRING,
            ingestion_ts TIMESTAMP
        )
        USING DELTA
        """
    )
    current = existing_columns(TARGET_TABLE)
    for name, dtype in TRADE_COLUMNS.items():
        if name not in current:
            spark.sql(f"ALTER TABLE {TARGET_TABLE} ADD COLUMN {name} {dtype}")


def load_token():
    try:
        return json.loads(dbutils.fs.head(TOKEN_STATE_PATH))
    except Exception:
        return None


def get_access_token_readonly():
    state = load_token()
    if not state:
        raise RuntimeError("Token state missing — run Saxo OAuth bootstrap.")

    now = time.time()
    if state["expires_at"] < now:
        raise RuntimeError("Token expired — refresher job failed or not running.")

    return state["access_token"]


def extract_error_info(payload):
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


def is_rate_limited(status_code, payload, response_text):
    if status_code == 429:
        return True

    error_info = extract_error_info(payload)
    if error_info and error_info["code"] == "RateLimitExceeded":
        return True

    return "rate limit exceeded" in (response_text or "").lower()


def saxo_request(method, url, headers, payload=None, timeout=30, action_name="request"):
    last_status = None
    last_payload = {}
    last_text = ""

    for attempt in range(1, SAXO_RETRY_MAX_ATTEMPTS + 1):
        resp = requests.request(method, url, headers=headers, json=payload, timeout=timeout)
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
        )
        print(
            f"Saxo {action_name} rate-limited on attempt {attempt}/"
            f"{SAXO_RETRY_MAX_ATTEMPTS}; sleeping {wait_seconds:.1f}s"
        )
        time.sleep(wait_seconds)

    raise SaxoTemporaryThrottleError(
        f"Saxo {action_name} deferred: status={last_status} payload={json.dumps(last_payload)[:500] or last_text[:500]}"
    )


def request_json(method, url, headers, payload=None, action_name="request"):
    resp, response_payload, retry_count = saxo_request(
        method,
        url,
        headers,
        payload=payload,
        timeout=30,
        action_name=action_name,
    )
    resp.raise_for_status()
    if retry_count:
        print(f"Saxo {action_name} succeeded after {retry_count} retries")
    return response_payload


def fetch_accounts(headers):
    payload = request_json(
        "GET",
        f"{BASE_URL}/port/v1/accounts/me?ClientKey={CLIENT_KEY}",
        headers,
        action_name="fetch_accounts",
    )
    return payload.get("Data", []) if isinstance(payload, dict) else []


def find_configured_account(headers):
    matches = [
        row
        for row in fetch_accounts(headers)
        if isinstance(row, dict) and first_path(row, "AccountKey", "Key") == ACCOUNT_KEY
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Saxo account for configured AccountKey; found {len(matches)}."
        )
    account = matches[0]
    account_id = first_path(account, "AccountId", "AccountNumber")
    currency = first_path(account, "Currency", "AccountCurrency")
    if account_id is None or currency is None:
        raise RuntimeError("Configured Saxo account is missing AccountId or Currency.")
    return {
        "account_id": str(account_id),
        "currency": str(currency).upper(),
    }


def enforce_live_account_guard(account_info, balance_payload):
    if EXECUTION_MODE != "live":
        return
    if (
        SAXO_ACCOUNT_KEY_SECRET_SCOPE != DEFAULT_ACCOUNT_KEY_SECRET_SCOPE
        or SAXO_ACCOUNT_KEY_SECRET_KEY != DEFAULT_ACCOUNT_KEY_SECRET_KEY
    ):
        raise RuntimeError(
            "Live Saxo execution blocked: live mode must use saxo/usd_account_key."
        )
    if account_info["account_id"] != EXPECTED_LIVE_ACCOUNT_ID:
        raise RuntimeError(
            f"Live Saxo execution blocked: configured account_id={account_info['account_id']} "
            f"does not match required {EXPECTED_LIVE_ACCOUNT_ID}."
        )
    if account_info["currency"].upper() != EXPECTED_LIVE_CURRENCY:
        raise RuntimeError(
            "Live Saxo execution blocked: configured account currency is not USD."
        )


def fetch_positions(headers):
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/positions/me?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}",
        headers,
        action_name="fetch_positions",
    )


def fetch_balance(headers):
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/balances/me?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}",
        headers,
        action_name="fetch_balance",
    )


def fetch_orders(headers):
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/orders?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}&FieldGroups=DisplayAndFormat",
        headers,
        action_name="fetch_orders",
    )


def get_price(headers, uic, asset_type):
    quote_payload = request_json(
        "GET",
        f"{BASE_URL}/trade/v1/infoprices?Uic={uic}&AssetType={asset_type}",
        headers,
        action_name=f"get_price:{uic}",
    )
    quote = quote_payload.get("Quote", {})
    return safe_float(
        first_present(
            quote.get("Mid"),
            quote.get("Ask"),
            quote.get("Last"),
            quote.get("Bid"),
        )
    )


def active_positions(positions_payload):
    rows = []
    ignored_other_account = 0
    ignored_missing_account_key = 0
    for position in positions_payload.get("Data", []):
        payload_account_key = account_key_from_payload(position)
        payload_account_id = account_id_from_payload(position)
        if not payload_matches_configured_account(position):
            if payload_account_key is None and payload_account_id is None:
                ignored_missing_account_key += 1
            else:
                ignored_other_account += 1
            continue
        base = position.get("PositionBase", {})
        net_view = position.get("NetPositionView", {})
        position_view = position.get("PositionView", {})
        amount = safe_float(base.get("Amount")) or 0.0
        if amount == 0.0:
            continue
        market_value = safe_float(
            first_present(
                position.get("MarketValue"),
                net_view.get("MarketValue"),
                position_view.get("MarketValue"),
                net_view.get("Exposure"),
            )
        )
        rows.append(
            {
                "raw": position,
                "amount": amount,
                "uic": uic_from_payload(position),
                "asset_type": base.get("AssetType"),
                "market_value": market_value,
                "symbol": first_present(
                    symbol_from_payload(position),
                    str(base.get("Uic")),
                ),
            }
        )
    if ignored_other_account or ignored_missing_account_key:
        print(
            "Saxo positions account filter: "
            f"included={len(rows)}, ignored_other_account={ignored_other_account}, "
            f"ignored_missing_account_key={ignored_missing_account_key}"
        )
    return rows


def order_status(order):
    return str(first_path(order, "Status", "OrderStatus", "Order.Status") or "").lower()


def is_open_order(order):
    terminal_markers = ("filled", "cancel", "reject", "expired", "done", "removed")
    status = order_status(order)
    return not any(marker in status for marker in terminal_markers)


def active_symbol_orders(orders_payload):
    rows = []
    ignored_other_account = 0
    ignored_missing_account_key = 0
    for order in orders_payload.get("Data", []):
        payload_account_key = account_key_from_payload(order)
        payload_account_id = account_id_from_payload(order)
        if not payload_matches_configured_account(order):
            if payload_account_key is None and payload_account_id is None:
                ignored_missing_account_key += 1
            else:
                ignored_other_account += 1
            continue
        symbol = symbol_from_payload(order)
        uic = uic_from_payload(order)
        if is_open_order(order) and (symbol in {"QQQ", "SPY"} or uic in {4328771, 36590}):
            rows.append(order)
    if ignored_other_account or ignored_missing_account_key:
        print(
            "Saxo orders account filter: "
            f"included={len(rows)}, ignored_other_account={ignored_other_account}, "
            f"ignored_missing_account_key={ignored_missing_account_key}"
        )
    return rows


def serialize_positions(positions_payload):
    payload = []
    total_market_value = 0.0
    total_unrealized_pl = 0.0
    weighted_plpc_numerator = 0.0

    for row in active_positions(positions_payload):
        raw = row["raw"]
        base = raw.get("PositionBase", {})
        net_view = raw.get("NetPositionView", {})
        position_view = raw.get("PositionView", {})

        market_value = safe_float(
            first_present(
                raw.get("MarketValue"),
                net_view.get("MarketValue"),
                position_view.get("MarketValue"),
                net_view.get("Exposure"),
            )
        ) or 0.0
        unrealized_pl = safe_float(
            first_present(
                raw.get("UnrealizedProfitLoss"),
                net_view.get("ProfitLossOnTrade"),
                net_view.get("ProfitLossOnTradeInBaseCurrency"),
                position_view.get("ProfitLossOnTrade"),
            )
        ) or 0.0
        current_price = safe_float(first_present(raw.get("CurrentPrice"), base.get("CurrentPrice")))
        avg_entry_price = safe_float(first_present(base.get("AverageOpenPrice"), base.get("OpenPrice")))
        unrealized_plpc = (
            unrealized_pl / abs(market_value)
            if market_value not in (0.0, None)
            else None
        )

        payload.append(
            {
                "symbol": row["symbol"],
                "uic": row["uic"],
                "asset_type": row["asset_type"],
                "qty": row["amount"],
                "avg_entry_price": avg_entry_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            }
        )

        total_market_value += market_value
        total_unrealized_pl += unrealized_pl
        if unrealized_plpc is not None:
            weighted_plpc_numerator += abs(market_value) * unrealized_plpc

    weighted_plpc = None
    if total_market_value:
        weighted_plpc = weighted_plpc_numerator / abs(total_market_value)

    return {
        "positions_json": json.dumps(payload),
        "positions_count": len(payload),
        "positions_market_value": total_market_value,
        "positions_unrealized_pl": total_unrealized_pl,
        "positions_unrealized_plpc": weighted_plpc,
    }


def extract_balance_metrics(balance_payload):
    account_equity = safe_float(
        first_present(
            balance_payload.get("TotalValue"),
            balance_payload.get("NetEquityForMargin"),
            balance_payload.get("PortfolioValue"),
        )
    )
    account_balance = safe_float(
        first_present(
            balance_payload.get("CashBalance"),
            balance_payload.get("AvailableFunds"),
            balance_payload.get("CashAvailableForTrading"),
            balance_payload.get("Balance"),
        )
    )
    day_pnl = safe_float(
        first_present(
            balance_payload.get("ProfitLoss"),
            balance_payload.get("ProfitLossOnCurrentPositions"),
            balance_payload.get("UnrealizedProfitLoss"),
        )
    )
    return {
        "account_equity": account_equity,
        "account_balance": account_balance,
        "account_cash": account_balance,
        "account_currency": balance_payload.get("Currency"),
        "account_spending_power": safe_float(
            first_present(
                balance_payload.get("SpendingPower"),
                balance_payload.get("MarginAvailableForTrading"),
                balance_payload.get("CashAvailableForTrading"),
                balance_payload.get("AvailableFunds"),
                balance_payload.get("CashBalance"),
            )
        ),
        "day_pnl": day_pnl,
    }


def is_target_position(position_row, target_asset):
    return (
        target_asset is not None
        and position_row["uic"] == target_asset["Uic"]
        and position_row["asset_type"] == target_asset["AssetType"]
    )


def find_target_position(position_rows, target_asset):
    for row in position_rows:
        if is_target_position(row, target_asset):
            return row
    return None


def gap_ratio(current_value, target_value):
    if current_value is None or target_value in (None, 0.0):
        return None
    return abs(current_value - target_value) / target_value


def ensure_order_submission(order_result):
    status_code = order_result.get("status_code")
    if status_code is not None and 200 <= int(status_code) < 300:
        return

    response_text = order_result.get("response") or ""
    response_payload = safe_json_loads(response_text)
    if isinstance(response_payload, dict):
        response_text = json.dumps(response_payload)

    raise RuntimeError(
        f"Saxo {order_result.get('action')} order failed with status "
        f"{status_code}: {redact_text(response_text[:500])}"
    )


def precheck_market_buy(headers, target_asset, qty):
    order = {
        "AccountKey": ACCOUNT_KEY,
        "Uic": target_asset["Uic"],
        "AssetType": target_asset["AssetType"],
        "BuySell": "Buy",
        "OrderType": "Market",
        "Amount": qty,
        "ManualOrder": False,
        "OrderDuration": {"DurationType": "DayOrder"},
    }
    resp, payload, retry_count = saxo_request(
        "POST",
        f"{BASE_URL}/trade/v2/orders/precheck",
        headers,
        payload=order,
        timeout=30,
        action_name=f"precheck_buy:{target_asset['symbol']}:{qty}",
    )
    return {
        "action": "precheck_buy",
        "uic": target_asset["Uic"],
        "asset_type": target_asset["AssetType"],
        "symbol": target_asset["symbol"],
        "amount": qty,
        "status_code": resp.status_code,
        "response": redact_text(resp.text),
        "payload": payload,
        "retry_count": retry_count,
    }


def affordable_buy_qty(headers, target_asset, desired_qty, initial_precheck=None):
    if desired_qty <= 0:
        return 0, None, None

    latest_precheck = initial_precheck or precheck_market_buy(headers, target_asset, desired_qty)
    if latest_precheck["status_code"] < 200 or latest_precheck["status_code"] >= 300:
        raise RuntimeError(
            f"Saxo precheck failed for {target_asset['symbol']} qty={desired_qty}: "
            f"{redact_text(latest_precheck['response'][:500])}"
        )

    error_info = extract_error_info(latest_precheck.get("payload"))
    if error_info is None:
        return desired_qty, latest_precheck, None
    if error_info["code"] != "InsufficientCash":
        raise RuntimeError(
            f"Saxo precheck rejected {target_asset['symbol']} qty={desired_qty}: "
            f"{error_info['code']} - {error_info['message']}"
        )

    best_qty = 0
    best_precheck = None
    low = 1
    high = desired_qty - 1
    insufficient_precheck = latest_precheck

    while low <= high:
        mid = (low + high) // 2
        latest_precheck = precheck_market_buy(headers, target_asset, mid)
        if latest_precheck["status_code"] < 200 or latest_precheck["status_code"] >= 300:
            raise RuntimeError(
                f"Saxo precheck failed for {target_asset['symbol']} qty={mid}: "
                f"{redact_text(latest_precheck['response'][:500])}"
            )

        error_info = extract_error_info(latest_precheck.get("payload"))
        if error_info is None:
            best_qty = mid
            best_precheck = latest_precheck
            low = mid + 1
            continue
        if error_info["code"] != "InsufficientCash":
            raise RuntimeError(
                f"Saxo precheck rejected {target_asset['symbol']} qty={mid}: "
                f"{error_info['code']} - {error_info['message']}"
            )

        insufficient_precheck = latest_precheck
        high = mid - 1

    return best_qty, best_precheck, insufficient_precheck


def close_position(headers, position_row):
    side = "Sell" if position_row["amount"] > 0 else "Buy"
    order = {
        "AccountKey": ACCOUNT_KEY,
        "Uic": position_row["uic"],
        "AssetType": position_row["asset_type"],
        "BuySell": side,
        "OrderType": "Market",
        "Amount": abs(position_row["amount"]),
        "ManualOrder": True,
        "OrderDuration": {"DurationType": "DayOrder"},
    }
    resp, _, retry_count = saxo_request(
        "POST",
        f"{BASE_URL}/trade/v2/orders",
        headers,
        payload=order,
        timeout=30,
        action_name=f"close:{position_row['symbol']}",
    )
    return {
        "action": "close",
        "uic": position_row["uic"],
        "asset_type": position_row["asset_type"],
        "side": side,
        "amount": abs(position_row["amount"]),
        "status_code": resp.status_code,
        "response": redact_text(resp.text),
        "retry_count": retry_count,
    }


def place_market_buy(headers, target_asset, qty):
    order = {
        "AccountKey": ACCOUNT_KEY,
        "Uic": target_asset["Uic"],
        "AssetType": target_asset["AssetType"],
        "BuySell": "Buy",
        "OrderType": "Market",
        "Amount": qty,
        "ManualOrder": False,
        "OrderDuration": {"DurationType": "DayOrder"},
    }
    resp, _, retry_count = saxo_request(
        "POST",
        f"{BASE_URL}/trade/v2/orders",
        headers,
        payload=order,
        timeout=30,
        action_name=f"buy:{target_asset['symbol']}:{qty}",
    )
    return {
        "action": "buy",
        "uic": target_asset["Uic"],
        "asset_type": target_asset["AssetType"],
        "symbol": target_asset["symbol"],
        "amount": qty,
        "status_code": resp.status_code,
        "response": redact_text(resp.text),
        "retry_count": retry_count,
    }


def latest_close(symbol):
    rows = spark.sql(
        f"""
        SELECT close
        FROM {MARKET_TABLE}
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).collect()
    return safe_float(rows[0]["close"]) if rows else None


def market_is_open(now_utc=None):
    # This guard is correct for ETF execution. If the execution universe shifts
    # to futures or CFDs with broader trading hours, update this logic.
    now_utc = now_utc or datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return now_ny.weekday() < 5 and market_open <= now_ny < market_close


def target_from_latest_signal(signal_row):
    regime_name = (signal_row["regime_name"] or "").strip()
    normalized = regime_name.lower()
    if normalized == "expansion":
        return "100% QQQ", SAXO_ASSET_BY_SYMBOL["QQQ"]
    if normalized == "slowdown":
        return "100% SPY", SAXO_ASSET_BY_SYMBOL["SPY"]
    if normalized == "recession":
        return "100% CASH", None
    raise RuntimeError(f"Unsupported latest regime_name for Saxo execution: {regime_name}")


ensure_trade_table()

access_token = get_access_token_readonly()
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

signal = spark.sql(
    f"""
    SELECT *
    FROM {SIGNAL_TABLE}
    ORDER BY timestamp DESC
    LIMIT 1
    """
).collect()[0]

signal_ts = signal["timestamp"]
regime_int = int(signal["regime_int"])
regime_name = signal["regime_name"]
target_position, target_asset = target_from_latest_signal(signal)
target_ticker = target_asset["symbol"] if target_asset else "CASH"

account_info = find_configured_account(headers)
CONFIGURED_ACCOUNT_ID = account_info["account_id"]
pre_positions_payload = fetch_positions(headers)
pre_active_positions = active_positions(pre_positions_payload)
pre_balance_payload = fetch_balance(headers)
enforce_live_account_guard(account_info, pre_balance_payload)
pre_balance = extract_balance_metrics(pre_balance_payload)
latest_positions_payload = pre_positions_payload
latest_balance_payload = pre_balance_payload
pre_open_symbol_orders = active_symbol_orders(fetch_orders(headers))

last_mode_equity_row = spark.sql(
    f"""
    SELECT account_equity
    FROM {TARGET_TABLE}
    WHERE execution_mode = '{EXECUTION_MODE}'
      AND account_equity IS NOT NULL
    ORDER BY trade_ts DESC
    LIMIT 1
    """
).collect()
last_equity = safe_float(last_mode_equity_row[0]["account_equity"]) if last_mode_equity_row else None

switched = False
orders_placed = []
transaction_cost = 0.0
execution_error = None
run_now_utc = datetime.now(timezone.utc)
market_open_now = market_is_open(run_now_utc)

if EXECUTION_MODE == "shadow":
    orders_placed.append(
        {
            "action": "shadow",
            "message": "No execution performed",
            "account_id": account_info["account_id"],
            "account_metadata_currency": account_info["currency"],
            "balance_payload_currency": pre_balance.get("account_currency"),
            "account_key_redacted": redact_account_key(ACCOUNT_KEY),
            "positions_count": len(pre_active_positions),
            "open_symbol_orders": len(pre_open_symbol_orders),
        }
    )
elif not market_open_now:
    orders_placed.append({"action": "skip", "message": "Market closed"})
elif pre_open_symbol_orders:
    orders_placed.append(
        {
            "action": "blocked",
            "message": "Open QQQ/SPY orders exist for configured Saxo account",
            "open_symbol_orders": len(pre_open_symbol_orders),
        }
    )
    execution_error = "Open QQQ/SPY orders exist for configured Saxo account; live execution blocked."
else:
    try:
        if target_asset is None:
            if not pre_active_positions:
                orders_placed.append({"action": "skip", "message": "Portfolio already aligned"})
            else:
                switched = True
                for row in pre_active_positions:
                    order_result = close_position(headers, row)
                    orders_placed.append(order_result)
                    ensure_order_submission(order_result)
                    time.sleep(1)
        else:
            latest_price = get_price(headers, target_asset["Uic"], target_asset["AssetType"])
            if latest_price is None or latest_price <= 0:
                raise RuntimeError(
                    f"Unable to fetch live price for {target_asset['symbol']} sizing."
                )

            target_value = safe_float(pre_balance["account_equity"])
            if target_value is None or target_value <= 0:
                raise RuntimeError("Unable to size order from current Saxo equity.")
            target_value *= NOTIONAL_FRACTION

            current_target_row = find_target_position(pre_active_positions, target_asset)
            current_target_qty = abs(current_target_row["amount"]) if current_target_row else 0.0
            current_target_value = abs(
                safe_float(current_target_row.get("market_value")) if current_target_row else 0.0
            )
            non_target_positions = [
                row for row in pre_active_positions if not is_target_position(row, target_asset)
            ]
            already_in_target = (
                not non_target_positions
                and current_target_qty > 0
                and gap_ratio(current_target_value, target_value) is not None
                and gap_ratio(current_target_value, target_value) <= ALIGNMENT_TOLERANCE_PCT
            )

            if already_in_target:
                orders_placed.append(
                    {
                        "action": "skip",
                        "message": "Portfolio already aligned",
                        "current_qty": current_target_qty,
                        "current_value": current_target_value,
                        "target_value": target_value,
                    }
                )
            else:
                switched = True
                for row in non_target_positions:
                    order_result = close_position(headers, row)
                    orders_placed.append(order_result)
                    ensure_order_submission(order_result)
                    time.sleep(1)

                refreshed_positions_payload = fetch_positions(headers)
                latest_positions_payload = refreshed_positions_payload
                refreshed_active_positions = active_positions(refreshed_positions_payload)
                refreshed_target_row = find_target_position(refreshed_active_positions, target_asset)
                refreshed_target_qty = (
                    abs(refreshed_target_row["amount"]) if refreshed_target_row else 0.0
                )

                refreshed_balance_payload = fetch_balance(headers)
                latest_balance_payload = refreshed_balance_payload
                refreshed_balance = extract_balance_metrics(refreshed_balance_payload)
                refreshed_target_value = safe_float(refreshed_balance["account_equity"]) or safe_float(
                    pre_balance["account_equity"]
                )
                if refreshed_target_value is None or refreshed_target_value <= 0:
                    raise RuntimeError("Unable to size order from refreshed Saxo equity.")
                refreshed_target_value *= NOTIONAL_FRACTION

                unit_precheck = precheck_market_buy(headers, target_asset, 1)
                if unit_precheck["status_code"] < 200 or unit_precheck["status_code"] >= 300:
                    raise RuntimeError(
                        f"Saxo precheck failed for {target_asset['symbol']} qty=1: "
                        f"{redact_text(unit_precheck['response'][:500])}"
                    )
                unit_precheck_error = extract_error_info(unit_precheck.get("payload"))
                if unit_precheck_error is not None and unit_precheck_error["code"] != "InsufficientCash":
                    raise RuntimeError(
                        f"Saxo precheck rejected {target_asset['symbol']} qty=1: "
                        f"{unit_precheck_error['code']} - {unit_precheck_error['message']}"
                    )

                estimated_unit_cost = safe_float(
                    unit_precheck["payload"].get("EstimatedCashRequired")
                )
                insufficient_precheck = None
                if unit_precheck_error is not None and unit_precheck_error["code"] == "InsufficientCash":
                    insufficient_precheck = unit_precheck
                    orders_placed.append(
                        {
                            "action": "skip",
                            "message": "InsufficientCash for minimum buy size; skipping buy adjustment.",
                            "requested_qty": 0,
                            "estimated_cash_required": unit_precheck["payload"].get(
                                "EstimatedCashRequired"
                            ),
                            "account_currency": refreshed_balance.get("account_currency"),
                            "spending_power": refreshed_balance.get("account_spending_power"),
                        }
                    )
                else:
                    if estimated_unit_cost is None or estimated_unit_cost <= 0:
                        raise RuntimeError(
                            f"Saxo precheck did not return a valid account-currency cost for "
                            f"{target_asset['symbol']}."
                        )

                    desired_qty = max(1, int(refreshed_target_value / estimated_unit_cost))
                    refreshed_position_value = abs(
                        safe_float(refreshed_target_row.get("market_value")) if refreshed_target_row else 0.0
                    )
                    refreshed_gap = gap_ratio(refreshed_position_value, refreshed_target_value)

                    if (
                        refreshed_target_qty > 0
                        and refreshed_gap is not None
                        and refreshed_gap <= ALIGNMENT_TOLERANCE_PCT
                    ):
                        orders_placed.append(
                            {
                                "action": "skip",
                                "message": "Portfolio already aligned after refresh",
                                "current_qty": refreshed_target_qty,
                                "current_value": refreshed_position_value,
                                "target_value": refreshed_target_value,
                            }
                        )
                    elif refreshed_target_qty > desired_qty:
                        raise RuntimeError(
                            "Current target position exceeds desired size after refresh; "
                            "manual review required before adding exposure."
                        )
                    else:
                        requested_qty = max(0, desired_qty - int(refreshed_target_qty))
                        initial_requested_precheck = unit_precheck if requested_qty == 1 else None
                        qty, affordable_precheck, insufficient_precheck = affordable_buy_qty(
                            headers,
                            target_asset,
                            requested_qty,
                            initial_precheck=initial_requested_precheck,
                        )
                        if requested_qty > 0 and qty < requested_qty:
                            insufficient_payload = (
                                insufficient_precheck.get("payload") if insufficient_precheck else {}
                            )
                            orders_placed.append(
                                {
                                    "action": "resize",
                                    "message": "Reduced buy size to fit available account currency cash",
                                    "requested_qty": requested_qty,
                                    "approved_qty": qty,
                                    "estimated_cash_required": insufficient_payload.get("EstimatedCashRequired"),
                                    "account_currency": refreshed_balance.get("account_currency"),
                                    "spending_power": refreshed_balance.get("account_spending_power"),
                                }
                            )

                        if qty > 0:
                            order_result = place_market_buy(headers, target_asset, qty)
                            orders_placed.append(order_result)
                            ensure_order_submission(order_result)
                        else:
                            failure_payload = (
                                insufficient_precheck.get("payload") if insufficient_precheck else {}
                            )
                            orders_placed.append(
                                {
                                    "action": "skip",
                                    "message": "No affordable incremental order after Saxo precheck",
                                    "current_qty": refreshed_target_qty,
                                    "desired_qty": desired_qty,
                                    "requested_qty": requested_qty,
                                    "estimated_cash_required": failure_payload.get("EstimatedCashRequired"),
                                    "account_currency": refreshed_balance.get("account_currency"),
                                    "spending_power": refreshed_balance.get("account_spending_power"),
                                }
                            )
    except SaxoTemporaryThrottleError as exc:
        orders_placed.append(
            {
                "action": "defer",
                "message": str(exc),
                "reason": "rate_limit",
                "retryable": True,
            }
        )
    except Exception as exc:
        execution_error = str(exc)
        orders_placed.append({"action": "error", "message": execution_error})

time.sleep(2)

try:
    post_positions_payload = fetch_positions(headers)
    latest_positions_payload = post_positions_payload
except SaxoTemporaryThrottleError as exc:
    post_positions_payload = latest_positions_payload
    orders_placed.append(
        {
            "action": "snapshot_defer",
            "message": str(exc),
            "reason": "rate_limit",
            "snapshot": "positions",
        }
    )

try:
    post_balance_payload = fetch_balance(headers)
    latest_balance_payload = post_balance_payload
except SaxoTemporaryThrottleError as exc:
    post_balance_payload = latest_balance_payload
    orders_placed.append(
        {
            "action": "snapshot_defer",
            "message": str(exc),
            "reason": "rate_limit",
            "snapshot": "balance",
        }
    )

post_balance = extract_balance_metrics(post_balance_payload)
position_snapshot = serialize_positions(post_positions_payload)

account_equity = post_balance["account_equity"]
account_balance = post_balance["account_balance"]
account_cash = post_balance["account_cash"]

hourly_pnl = (
    account_equity - last_equity
    if account_equity is not None and last_equity not in (None, 0.0)
    else None
)
hourly_pnl_pct = (
    hourly_pnl / last_equity
    if hourly_pnl is not None and last_equity not in (None, 0.0)
    else None
)
day_pnl = post_balance["day_pnl"]
day_pnl_pct = (
    day_pnl / last_equity
    if day_pnl is not None and last_equity not in (None, 0.0)
    else None
)

trade_ts = datetime.now(timezone.utc).replace(tzinfo=None)

schema = StructType(
    [
        StructField("trade_ts", TimestampType(), True),
        StructField("signal_ts", TimestampType(), True),
        StructField("regime_int", IntegerType(), True),
        StructField("regime_name", StringType(), True),
        StructField("target_position", StringType(), True),
        StructField("target_ticker", StringType(), True),
        StructField("switched", BooleanType(), True),
        StructField("transaction_cost", DoubleType(), True),
        StructField("orders_placed", StringType(), True),
        StructField("spy_close", DoubleType(), True),
        StructField("qqq_close", DoubleType(), True),
        StructField("account_balance", DoubleType(), True),
        StructField("account_equity", DoubleType(), True),
        StructField("account_cash", DoubleType(), True),
        StructField("last_equity", DoubleType(), True),
        StructField("hourly_pnl", DoubleType(), True),
        StructField("hourly_pnl_pct", DoubleType(), True),
        StructField("day_pnl", DoubleType(), True),
        StructField("day_pnl_pct", DoubleType(), True),
        StructField("positions_json", StringType(), True),
        StructField("positions_count", IntegerType(), True),
        StructField("positions_market_value", DoubleType(), True),
        StructField("positions_unrealized_pl", DoubleType(), True),
        StructField("positions_unrealized_plpc", DoubleType(), True),
        StructField("execution_mode", StringType(), True),
        StructField("source", StringType(), True),
        StructField("ingestion_ts", TimestampType(), True),
    ]
)

trade_row = {
    "trade_ts": trade_ts,
    "signal_ts": signal_ts,
    "regime_int": regime_int,
    "regime_name": regime_name,
    "target_position": target_position,
    "target_ticker": target_ticker,
    "switched": switched,
    "transaction_cost": transaction_cost,
    "orders_placed": json.dumps(orders_placed),
    "spy_close": latest_close(PRIMARY_SYMBOL),
    "qqq_close": latest_close(SECONDARY_SYMBOL),
    "account_balance": account_balance,
    "account_equity": account_equity,
    "account_cash": account_cash,
    "last_equity": last_equity,
    "hourly_pnl": hourly_pnl,
    "hourly_pnl_pct": hourly_pnl_pct,
    "day_pnl": day_pnl,
    "day_pnl_pct": day_pnl_pct,
    "positions_json": position_snapshot["positions_json"],
    "positions_count": position_snapshot["positions_count"],
    "positions_market_value": position_snapshot["positions_market_value"],
    "positions_unrealized_pl": position_snapshot["positions_unrealized_pl"],
    "positions_unrealized_plpc": position_snapshot["positions_unrealized_plpc"],
    "execution_mode": EXECUTION_MODE,
    "source": "rt_saxo_shadow_snapshot" if EXECUTION_MODE == "shadow" else "rt_saxo_live_execution",
    "ingestion_ts": None,
}

sdf = spark.createDataFrame([trade_row], schema=schema).withColumn(
    "ingestion_ts", F.current_timestamp()
)
append_dataframe_with_retry(
    sdf,
    TARGET_TABLE,
    operation_name=f"append Saxo trade log row into {TARGET_TABLE}",
)

print(f"Saxo trade log written to {TARGET_TABLE}")
print(f"Execution mode: {EXECUTION_MODE}")
print(f"Saxo account id: {account_info['account_id']}")
print(f"Saxo account metadata currency: {account_info['currency']}")
print(f"Saxo balance payload currency: {post_balance.get('account_currency')}")
print(f"Saxo account key redacted: {redact_account_key(ACCOUNT_KEY)}")
print(f"Saxo client key redacted: {redact_secret(CLIENT_KEY)}")
print(f"Signal timestamp: {signal_ts}")
print(f"Signal regime: {regime_name}")
print(f"Target allocation: {target_position}")
print(f"Target ticker: {target_ticker}")
print(f"Orders: {trade_row['orders_placed']}")
print(f"Account balance: {account_balance}")
print(f"Account equity: {account_equity}")
print(f"Hourly PnL: {hourly_pnl}")
print(f"Open positions: {position_snapshot['positions_count']}")

if execution_error and "insufficientcash" not in execution_error.lower():
    raise RuntimeError(execution_error)

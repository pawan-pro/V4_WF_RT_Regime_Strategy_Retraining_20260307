# Databricks notebook source
# One-off Saxo SGD account QQQ liquidation utility.
# Default mode is preflight. The only order path is a guarded market SELL of
# exactly 5 QQQ shares from the old SGD account.

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE_URL = "https://gateway.saxobank.com/openapi"
CLIENT_KEY = "hLMMvwa3Mzmiem2AMhpKwA=="
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"
DIAGNOSTICS_DIR = "/Volumes/pawan_quant_research_1/v4/market_data/diagnostics"

OLD_SGD_ACCOUNT_ID = "33888/256469"
OLD_SGD_CURRENCY = "SGD"
QQQ_UIC = 4328771
QQQ_SYMBOL = "QQQ"
SPY_UIC = 36590
SPY_SYMBOL = "SPY"
ETF_ASSET_TYPE = "Etf"
EXPECTED_QQQ_QTY = Decimal("5")
LIQUIDATION_CONFIRMATION = "SELL_QQQ_5_FROM_SGD_33888_256469"

SAXO_RETRY_MAX_ATTEMPTS = 4
SAXO_RETRY_BASE_SECONDS = 1.0
SAXO_RETRY_MAX_SECONDS = 8.0
NY_TZ = ZoneInfo("America/New_York")


class SaxoTemporaryThrottleError(RuntimeError):
    pass


def widget(name: str, default: str) -> str:
    try:
        dbutils.widgets.text(name, default)
        return (dbutils.widgets.get(name) or default).strip()
    except Exception:
        return default


OPERATION_MODE = widget("operation_mode", "preflight").lower()
CONFIRM_LIQUIDATION = widget("confirm_liquidation", "")
SGD_ACCOUNT_KEY_SECRET_SCOPE = widget("sgd_account_key_secret_scope", "saxo")
SGD_ACCOUNT_KEY_SECRET_KEY = widget("sgd_account_key_secret_key", "sgd_account_key")
SGD_ACCOUNT_KEY_DIRECT = widget("sgd_account_key", "")

if OPERATION_MODE not in {"preflight", "liquidate"}:
    raise ValueError("operation_mode must be preflight or liquidate")


def safe_json_loads(payload: Any) -> Optional[Any]:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(payload)
    except Exception:
        return None


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def safe_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def fingerprint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def redact_text(text: Any, secrets: Optional[list[str]] = None) -> str:
    redacted = str(text)
    for secret in secrets or []:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", redacted)
    redacted = re.sub(r'("access_token"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    redacted = re.sub(r'("refresh_token"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    redacted = re.sub(r'("AccountKey"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    return redacted


def redact_obj(value: Any, secrets: Optional[list[str]] = None) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if key.lower() in {"accountkey", "fromaccountkey", "toaccountkey", "access_token", "refresh_token"}:
                output[key] = "[REDACTED]"
            else:
                output[key] = redact_obj(item, secrets)
        return output
    if isinstance(value, list):
        return [redact_obj(item, secrets) for item in value]
    if isinstance(value, str):
        return redact_text(value, secrets)
    return value


def resolve_account_key() -> str:
    if SGD_ACCOUNT_KEY_SECRET_SCOPE and SGD_ACCOUNT_KEY_SECRET_KEY:
        try:
            key = dbutils.secrets.get(SGD_ACCOUNT_KEY_SECRET_SCOPE, SGD_ACCOUNT_KEY_SECRET_KEY).strip()
            if key:
                return key
        except Exception:
            pass
    if SGD_ACCOUNT_KEY_DIRECT:
        return SGD_ACCOUNT_KEY_DIRECT
    raise RuntimeError(
        "Old SGD AccountKey is not configured. Set Databricks secret "
        "saxo/sgd_account_key or provide the sgd_account_key widget for this one-off utility."
    )


ACCOUNT_KEY = resolve_account_key()
REDACTION_SECRETS = [ACCOUNT_KEY]


def diagnostics_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{DIAGNOSTICS_DIR}/saxo_sgd_liquidation_{ts}.csv"


DIAGNOSTIC_ROWS: list[dict[str, Any]] = []
OUTPUT_PATH = diagnostics_path()


def add_diag(check_name: str, status: str, **kwargs):
    row = {
        "event_ts_utc": datetime.now(timezone.utc).isoformat(),
        "operation_mode": OPERATION_MODE,
        "check_name": check_name,
        "status": status,
        "old_account_id": OLD_SGD_ACCOUNT_ID,
        "old_currency": OLD_SGD_CURRENCY,
        "account_key_sha256_12": fingerprint(ACCOUNT_KEY),
    }
    row.update({key: redact_text(value, REDACTION_SECRETS) for key, value in kwargs.items()})
    DIAGNOSTIC_ROWS.append(row)


def write_diagnostics() -> str:
    try:
        dbutils.fs.mkdirs("dbfs:" + DIAGNOSTICS_DIR)
    except Exception:
        pass
    pdf = pd.DataFrame(DIAGNOSTIC_ROWS or [{"event_ts_utc": datetime.now(timezone.utc).isoformat()}])
    pdf.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def load_token() -> dict:
    try:
        state = json.loads(dbutils.fs.head(TOKEN_STATE_PATH))
    except Exception as exc:
        raise RuntimeError("Token state missing. Use the existing token refresher/bootstrap outside this notebook.") from exc
    if state.get("expires_at", 0) < time.time():
        raise RuntimeError("Token expired. The token refresher is not current; this notebook will not refresh it.")
    return state


def headers() -> dict[str, str]:
    state = load_token()
    token = state["access_token"]
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


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
    return {"code": error_code or "UnknownError", "message": message or ""}


def is_rate_limited(status_code: int, payload: Any, response_text: str) -> bool:
    if status_code == 429:
        return True
    error_info = extract_error_info(payload)
    if error_info and error_info["code"] == "RateLimitExceeded":
        return True
    return "rate limit" in (response_text or "").lower()


def saxo_request(method: str, url: str, hdrs: dict, payload: Optional[dict] = None, action_name: str = "request"):
    last_status = None
    last_payload = {}
    last_text = ""
    for attempt in range(1, SAXO_RETRY_MAX_ATTEMPTS + 1):
        resp = requests.request(method, url, headers=hdrs, json=payload, timeout=30)
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
                "message": redact_text(response_text[:500], REDACTION_SECRETS),
            }
            raise SaxoTemporaryThrottleError(
                f"Saxo {action_name} deferred after {attempt} attempts: "
                f"{error_info['code']} - {redact_text(error_info['message'], REDACTION_SECRETS)}"
            )
        wait_seconds = min(SAXO_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), SAXO_RETRY_MAX_SECONDS)
        print(f"Saxo {action_name} rate-limited on attempt {attempt}; sleeping {wait_seconds:.1f}s")
        time.sleep(wait_seconds)
    raise SaxoTemporaryThrottleError(
        f"Saxo {action_name} deferred: status={last_status} "
        f"payload={redact_text(json.dumps(last_payload)[:500] or last_text[:500], REDACTION_SECRETS)}"
    )


def request_json(method: str, url: str, hdrs: dict, payload: Optional[dict] = None, action_name: str = "request") -> Any:
    resp, response_payload, retry_count = saxo_request(method, url, hdrs, payload=payload, action_name=action_name)
    if retry_count:
        print(f"Saxo {action_name} succeeded after {retry_count} retries")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Saxo {action_name} failed status={resp.status_code} "
            f"response={redact_text(resp.text[:500], REDACTION_SECRETS)}"
        )
    return response_payload


def get_account_value(account: dict, *names: str) -> Optional[str]:
    for name in names:
        value = account
        for part in name.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if value is not None:
            return str(value)
    return None


def fetch_accounts(hdrs: dict) -> list[dict]:
    payload = request_json(
        "GET",
        f"{BASE_URL}/port/v1/accounts/me?ClientKey={CLIENT_KEY}",
        hdrs,
        action_name="fetch_accounts",
    )
    data = payload.get("Data", []) if isinstance(payload, dict) else []
    return [row for row in data if isinstance(row, dict)]


def verify_old_account(hdrs: dict) -> dict:
    matches = [
        row
        for row in fetch_accounts(hdrs)
        if get_account_value(row, "AccountKey", "Key") == ACCOUNT_KEY
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one account for old SGD AccountKey; found {len(matches)}.")
    account = matches[0]
    account_id = get_account_value(account, "AccountId", "AccountNumber")
    currency = get_account_value(account, "Currency", "AccountCurrency")
    add_diag("account_lookup", "ok", account_id=account_id, currency=currency)
    if account_id != OLD_SGD_ACCOUNT_ID or currency != OLD_SGD_CURRENCY:
        raise RuntimeError(
            f"Account guard failed. Expected {OLD_SGD_ACCOUNT_ID}/{OLD_SGD_CURRENCY}; "
            f"got {account_id}/{currency}."
        )
    return account


def item_account_key(item: dict) -> Optional[str]:
    for path in (
        "AccountKey",
        "PositionBase.AccountKey",
        "NetPositionBase.AccountKey",
        "Account.AccountKey",
        "Order.AccountKey",
    ):
        value = get_account_value(item, path)
        if value:
            return value
    return None


def item_uic(item: dict) -> Optional[int]:
    for path in ("Uic", "PositionBase.Uic", "NetPositionBase.Uic", "Order.Uic"):
        value = get_account_value(item, path)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def item_symbol(item: dict) -> Optional[str]:
    value = get_account_value(
        item,
        "Symbol",
        "DisplayAndFormat.Symbol",
        "PositionBase.Symbol",
        "NetPositionBase.Symbol",
        "Order.Symbol",
    )
    return value.upper() if value else None


def fetch_positions(hdrs: dict) -> dict:
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/positions/me?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}",
        hdrs,
        action_name="fetch_positions",
    )


def active_positions(positions_payload: dict) -> list[dict]:
    rows = []
    ignored_other_account = 0
    for position in positions_payload.get("Data", []):
        if not isinstance(position, dict):
            continue
        if item_account_key(position) != ACCOUNT_KEY:
            ignored_other_account += 1
            continue
        base = position.get("PositionBase", {})
        amount = safe_decimal(base.get("Amount")) or Decimal("0")
        if amount == 0:
            continue
        rows.append(
            {
                "raw": position,
                "amount": amount,
                "uic": item_uic(position),
                "symbol": item_symbol(position),
                "asset_type": get_account_value(position, "PositionBase.AssetType", "AssetType"),
            }
        )
    add_diag("positions_account_filter", "ok", active_positions=len(rows), ignored_other_account=ignored_other_account)
    return rows


def qqq_position(rows: list[dict]) -> Optional[dict]:
    matches = [
        row
        for row in rows
        if row["uic"] == QQQ_UIC or row["symbol"] == QQQ_SYMBOL
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Expected at most one QQQ position in old SGD account; found {len(matches)}.")
    return matches[0] if matches else None


def fetch_orders(hdrs: dict) -> dict:
    return request_json(
        "GET",
        f"{BASE_URL}/port/v1/orders?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}&FieldGroups=DisplayAndFormat",
        hdrs,
        action_name="fetch_orders",
    )


def order_status(order: dict) -> str:
    return (get_account_value(order, "Status", "OrderStatus", "Order.Status") or "").lower()


def is_open_order(order: dict) -> bool:
    status = order_status(order)
    terminal_markers = ("filled", "cancel", "reject", "expired", "done", "removed")
    return not any(marker in status for marker in terminal_markers)


def active_symbol_orders(orders_payload: dict) -> list[dict]:
    rows = []
    ignored_other_account = 0
    for order in orders_payload.get("Data", []):
        if not isinstance(order, dict):
            continue
        if item_account_key(order) != ACCOUNT_KEY:
            ignored_other_account += 1
            continue
        symbol = item_symbol(order)
        uic = item_uic(order)
        if (symbol in {QQQ_SYMBOL, SPY_SYMBOL} or uic in {QQQ_UIC, SPY_UIC}) and is_open_order(order):
            rows.append(order)
    add_diag("orders_account_filter", "ok", open_symbol_orders=len(rows), ignored_other_account=ignored_other_account)
    return rows


def market_is_open(now_utc: Optional[datetime] = None) -> bool:
    now_utc = now_utc or datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(NY_TZ)
    market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return now_ny.weekday() < 5 and market_open <= now_ny < market_close


def submit_sell_order(hdrs: dict) -> dict:
    order = {
        "AccountKey": ACCOUNT_KEY,
        "Uic": QQQ_UIC,
        "AssetType": ETF_ASSET_TYPE,
        "BuySell": "Sell",
        "OrderType": "Market",
        "Amount": int(EXPECTED_QQQ_QTY),
        "ManualOrder": True,
        "OrderDuration": {"DurationType": "DayOrder"},
    }
    resp, payload, retry_count = saxo_request(
        "POST",
        f"{BASE_URL}/trade/v2/orders",
        hdrs,
        payload=order,
        action_name="sell_qqq_5_old_sgd",
    )
    result = {
        "status_code": resp.status_code,
        "response": redact_text(resp.text[:1000], REDACTION_SECRETS),
        "payload": redact_obj(payload, REDACTION_SECRETS),
        "retry_count": retry_count,
    }
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(f"Saxo sell order failed status={resp.status_code}: {result['response']}")
    return result


def run():
    hdrs = headers()
    verify_old_account(hdrs)

    pre_positions_payload = fetch_positions(hdrs)
    pre_positions = active_positions(pre_positions_payload)
    qqq = qqq_position(pre_positions)
    qqq_qty = qqq["amount"] if qqq else Decimal("0")
    add_diag("qqq_position_precheck", "ok", qqq_qty=str(qqq_qty))
    if qqq is None or qqq_qty != EXPECTED_QQQ_QTY:
        raise RuntimeError(f"QQQ position guard failed. Expected exactly 5 shares; found {qqq_qty}.")

    open_orders = active_symbol_orders(fetch_orders(hdrs))
    if open_orders:
        add_diag("open_order_precheck", "blocked", open_order_count=len(open_orders))
        raise RuntimeError(f"Open QQQ/SPY orders exist in old SGD account: {len(open_orders)}.")
    add_diag("open_order_precheck", "ok", open_order_count=0)

    market_open = market_is_open()
    add_diag("us_equity_market_open", "ok" if market_open else "blocked", market_open=market_open)

    order_result = None
    if OPERATION_MODE == "preflight":
        add_diag("liquidation_action", "skipped", reason="preflight_only")
        print("Preflight complete. No order submitted.")
    else:
        if not market_open:
            raise RuntimeError("Regular U.S. equity market is closed. Liquidation is blocked.")
        if CONFIRM_LIQUIDATION != LIQUIDATION_CONFIRMATION:
            raise RuntimeError("Exact confirm_liquidation widget value missing. Liquidation is blocked.")
        order_result = submit_sell_order(hdrs)
        add_diag(
            "liquidation_order",
            "submitted",
            status_code=order_result["status_code"],
            retry_count=order_result["retry_count"],
            response=json.dumps(order_result["payload"]),
        )
        time.sleep(2)

    post_positions = active_positions(fetch_positions(hdrs))
    post_qqq = qqq_position(post_positions)
    post_qqq_qty = post_qqq["amount"] if post_qqq else Decimal("0")
    post_open_orders = active_symbol_orders(fetch_orders(hdrs))
    add_diag("qqq_position_postcheck", "ok", qqq_qty=str(post_qqq_qty))
    add_diag("open_order_postcheck", "ok", open_order_count=len(post_open_orders))

    path = write_diagnostics()
    print(f"Diagnostic CSV: {path}")
    print(f"Operation mode: {OPERATION_MODE}")
    print(f"Old account verified: {OLD_SGD_ACCOUNT_ID} {OLD_SGD_CURRENCY}")
    print(f"Pre QQQ quantity: {qqq_qty}")
    print(f"Post QQQ quantity: {post_qqq_qty}")
    print(f"Post open QQQ/SPY order count: {len(post_open_orders)}")
    if order_result:
        print(f"Liquidation order status: {order_result['status_code']}")


try:
    run()
except Exception as exc:
    add_diag("notebook_error", "error", error=redact_text(str(exc), REDACTION_SECRETS))
    path = write_diagnostics()
    print(f"Diagnostic CSV: {path}")
    print(f"Blocked/error: {redact_text(str(exc), REDACTION_SECRETS)}")
    raise

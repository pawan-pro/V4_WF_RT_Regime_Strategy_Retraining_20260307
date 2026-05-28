# Databricks notebook source
# Saxo SGD-to-USD cash transfer discovery utility.
# Default mode is read-only discovery. Transfer submission is guarded and should
# only be used after Saxo endpoint availability and required fields are confirmed.

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import pandas as pd
import requests

BASE_URL = "https://gateway.saxobank.com/openapi"
CLIENT_KEY = "hLMMvwa3Mzmiem2AMhpKwA=="
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"
DIAGNOSTICS_DIR = "/Volumes/pawan_quant_research_1/v4/market_data/diagnostics"

OLD_SGD_ACCOUNT_ID = "33888/256469"
OLD_SGD_CURRENCY = "SGD"
NEW_USD_ACCOUNT_ID = "33888/261196"
NEW_USD_CURRENCY = "USD"

TRANSFER_CONFIRMATION = "TRANSFER_CASH_SGD_33888_256469_TO_USD_33888_261196"
ENDPOINT_CONFIRMATION = "ENDPOINT_ENABLED_FIELDS_CONFIRMED"
MANUAL_ROUTE_MESSAGE = "Use Saxo terminal Sub-account transfer module/manual support route."

CS_INTERACCOUNT_ENDPOINT = "/cs/v2/cashmanagement/interaccounttransfers"
ATR_PROCESSING_ENDPOINT = "/atr/v1/cashmanagement/interaccounttransfers/paymentsinprocessing"
ATR_WITHDRAWAL_LIMITS_ENDPOINT = "/atr/v1/cashmanagement/withdrawallimits/"
ATR_PARTNER_CASH_TRANSFER_LIMITS_ENDPOINT = "/atr/v1/partner/cashtransferlimits"

SAXO_RETRY_MAX_ATTEMPTS = 4
SAXO_RETRY_BASE_SECONDS = 1.0
SAXO_RETRY_MAX_SECONDS = 8.0


class SaxoTemporaryThrottleError(RuntimeError):
    pass


def widget(name: str, default: str) -> str:
    try:
        dbutils.widgets.text(name, default)
        return (dbutils.widgets.get(name) or default).strip()
    except Exception:
        return default


OPERATION_MODE = widget("operation_mode", "discover").lower()
SGD_ACCOUNT_KEY_SECRET_SCOPE = widget("sgd_account_key_secret_scope", "saxo")
SGD_ACCOUNT_KEY_SECRET_KEY = widget("sgd_account_key_secret_key", "sgd_account_key")
SGD_ACCOUNT_KEY_DIRECT = widget("sgd_account_key", "")
USD_ACCOUNT_KEY_SECRET_SCOPE = widget("usd_account_key_secret_scope", "saxo")
USD_ACCOUNT_KEY_SECRET_KEY = widget("usd_account_key_secret_key", "usd_account_key")
TRANSFER_AMOUNT = widget("transfer_amount", "")
TRANSFER_CURRENCY = widget("transfer_currency", "")
CONFIRM_TRANSFER = widget("confirm_transfer", "")
CONFIRM_ENDPOINT = widget("confirm_endpoint_enabled", "")

if OPERATION_MODE not in {"discover", "submit_transfer"}:
    raise ValueError("operation_mode must be discover or submit_transfer")


def safe_json_loads(payload: Any) -> Optional[Any]:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(payload)
    except Exception:
        return None


def get_path(payload: dict, path: str) -> Optional[Any]:
    value = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def get_first(payload: dict, *paths: str) -> Optional[str]:
    for path in paths:
        value = get_path(payload, path)
        if value is not None:
            return str(value)
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
    redacted = re.sub(r'("FromAccountKey"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
    redacted = re.sub(r'("ToAccountKey"\s*:\s*")[^"]+', r"\1[REDACTED]", redacted)
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


def resolve_secret_or_direct(scope: str, key: str, direct: str, label: str) -> str:
    if scope and key:
        try:
            value = dbutils.secrets.get(scope, key).strip()
            if value:
                return value
        except Exception:
            pass
    if direct:
        return direct
    raise RuntimeError(f"{label} AccountKey is not configured.")


def resolve_secret(scope: str, key: str, label: str) -> str:
    if scope and key:
        try:
            value = dbutils.secrets.get(scope, key).strip()
            if value:
                return value
        except Exception as exc:
            raise RuntimeError(f"{label} AccountKey secret is not readable.") from exc
    raise RuntimeError(f"{label} AccountKey secret scope/key is not configured.")


FROM_ACCOUNT_KEY = resolve_secret_or_direct(
    SGD_ACCOUNT_KEY_SECRET_SCOPE,
    SGD_ACCOUNT_KEY_SECRET_KEY,
    SGD_ACCOUNT_KEY_DIRECT,
    "Old SGD",
)
TO_ACCOUNT_KEY = resolve_secret(
    USD_ACCOUNT_KEY_SECRET_SCOPE,
    USD_ACCOUNT_KEY_SECRET_KEY,
    "New USD",
)
REDACTION_SECRETS = [FROM_ACCOUNT_KEY, TO_ACCOUNT_KEY]


def diagnostics_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{DIAGNOSTICS_DIR}/saxo_cash_transfer_discovery_{ts}.csv"


OUTPUT_PATH = diagnostics_path()
DIAGNOSTIC_ROWS: list[dict[str, Any]] = []


def add_diag(check_name: str, status: str, **kwargs):
    row = {
        "event_ts_utc": datetime.now(timezone.utc).isoformat(),
        "operation_mode": OPERATION_MODE,
        "check_name": check_name,
        "status": status,
        "from_account_id": OLD_SGD_ACCOUNT_ID,
        "to_account_id": NEW_USD_ACCOUNT_ID,
        "from_account_key_sha256_12": fingerprint(FROM_ACCOUNT_KEY),
        "to_account_key_sha256_12": fingerprint(TO_ACCOUNT_KEY),
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
    token = load_token()["access_token"]
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


def request_json(method: str, endpoint: str, hdrs: dict, payload: Optional[dict] = None, action_name: str = "request") -> Any:
    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    resp, response_payload, retry_count = saxo_request(method, url, hdrs, payload=payload, action_name=action_name)
    if retry_count:
        print(f"Saxo {action_name} succeeded after {retry_count} retries")
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Saxo {action_name} failed status={resp.status_code} "
            f"response={redact_text(resp.text[:500], REDACTION_SECRETS)}"
        )
    return response_payload


def probe(method: str, endpoint: str, hdrs: dict, action_name: str, payload: Optional[dict] = None) -> dict:
    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    try:
        resp, response_payload, retry_count = saxo_request(method, url, hdrs, payload=payload, action_name=action_name)
        result = {
            "method": method,
            "endpoint": endpoint,
            "status_code": resp.status_code,
            "retry_count": retry_count,
            "allow": resp.headers.get("Allow"),
            "content_type": resp.headers.get("Content-Type"),
            "response": redact_text(resp.text[:1000], REDACTION_SECRETS),
            "payload": json.dumps(redact_obj(response_payload, REDACTION_SECRETS))[:1000],
        }
    except Exception as exc:
        result = {
            "method": method,
            "endpoint": endpoint,
            "status_code": None,
            "retry_count": None,
            "allow": None,
            "content_type": None,
            "response": None,
            "payload": None,
            "error": redact_text(str(exc), REDACTION_SECRETS),
        }
    add_diag("endpoint_probe", "ok" if result.get("status_code") else "error", **result)
    return result


def fetch_accounts(hdrs: dict) -> list[dict]:
    payload = request_json(
        "GET",
        f"/port/v1/accounts/me?ClientKey={CLIENT_KEY}",
        hdrs,
        action_name="fetch_accounts",
    )
    data = payload.get("Data", []) if isinstance(payload, dict) else []
    return [row for row in data if isinstance(row, dict)]


def fetch_balance(hdrs: dict, account_key: str, label: str) -> dict:
    payload = request_json(
        "GET",
        f"/port/v1/balances/me?ClientKey={CLIENT_KEY}&AccountKey={account_key}",
        hdrs,
        action_name=f"fetch_balance_{label}",
    )
    add_diag(
        f"{label}_balance",
        "ok",
        currency=payload.get("Currency") if isinstance(payload, dict) else None,
        cash_balance=payload.get("CashBalance") if isinstance(payload, dict) else None,
        available_funds=payload.get("AvailableFunds") if isinstance(payload, dict) else None,
        cash_available_for_trading=payload.get("CashAvailableForTrading") if isinstance(payload, dict) else None,
        spending_power=payload.get("SpendingPower") if isinstance(payload, dict) else None,
    )
    return payload


def verify_account(accounts: list[dict], account_key: str, expected_id: str, expected_currency: str, label: str):
    matches = [row for row in accounts if get_first(row, "AccountKey", "Key") == account_key]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {label} account for configured AccountKey; found {len(matches)}.")
    account = matches[0]
    account_id = get_first(account, "AccountId", "AccountNumber")
    currency = get_first(account, "Currency", "AccountCurrency")
    add_diag(f"{label}_account_lookup", "ok", account_id=account_id, currency=currency)
    if account_id != expected_id or currency != expected_currency:
        raise RuntimeError(
            f"{label} account guard failed. Expected {expected_id}/{expected_currency}; got {account_id}/{currency}."
        )


def discover_transfer_capability(hdrs: dict) -> dict:
    processing_result = probe(
        "GET",
        f"{ATR_PROCESSING_ENDPOINT}?AccountKey={FROM_ACCOUNT_KEY}&ClientKey={CLIENT_KEY}",
        hdrs,
        "payments_in_processing",
    )
    get_result = probe("GET", CS_INTERACCOUNT_ENDPOINT, hdrs, "cs_interaccount_get_probe")
    options_result = probe("OPTIONS", CS_INTERACCOUNT_ENDPOINT, hdrs, "cs_interaccount_options_probe")
    withdrawal_limits_result = probe(
        "GET",
        f"{ATR_WITHDRAWAL_LIMITS_ENDPOINT}?AccountKey={FROM_ACCOUNT_KEY}&ClientKey={CLIENT_KEY}",
        hdrs,
        "withdrawal_limits_probe",
    )
    partner_transfer_limits_result = probe(
        "GET",
        (
            f"{ATR_PARTNER_CASH_TRANSFER_LIMITS_ENDPOINT}?"
            f"FromAccountKey={FROM_ACCOUNT_KEY}&ToAccountKey={TO_ACCOUNT_KEY}&ClientKey={CLIENT_KEY}"
        ),
        hdrs,
        "partner_cash_transfer_limits_probe",
    )

    allow_header = (options_result.get("allow") or get_result.get("allow") or "").upper()
    endpoint_possible = (
        options_result.get("status_code") in {200, 204}
        and "POST" in allow_header
    )
    endpoint_unclear = not endpoint_possible
    add_diag(
        "transfer_capability",
        "possible" if endpoint_possible else "unclear",
        cs_endpoint=CS_INTERACCOUNT_ENDPOINT,
        atr_processing_status=processing_result.get("status_code"),
        cs_get_status=get_result.get("status_code"),
        cs_options_status=options_result.get("status_code"),
        withdrawal_limits_status=withdrawal_limits_result.get("status_code"),
        partner_transfer_limits_status=partner_transfer_limits_result.get("status_code"),
        allow_header=allow_header,
        required_fields="FromAccountKey,ToAccountKey,Amount,Currency,ExternalReference",
        note="No POST was submitted in discover mode.",
    )
    if endpoint_unclear:
        print(MANUAL_ROUTE_MESSAGE)
    return {
        "endpoint_possible": endpoint_possible,
        "processing_result": processing_result,
        "get_result": get_result,
        "options_result": options_result,
        "withdrawal_limits_result": withdrawal_limits_result,
        "partner_transfer_limits_result": partner_transfer_limits_result,
    }


def parse_transfer_amount() -> Decimal:
    try:
        amount = Decimal(str(TRANSFER_AMOUNT))
    except Exception as exc:
        raise RuntimeError("transfer_amount must be a positive decimal.") from exc
    if amount <= 0:
        raise RuntimeError("transfer_amount must be positive.")
    return amount


def submit_transfer(hdrs: dict):
    if CONFIRM_ENDPOINT != ENDPOINT_CONFIRMATION:
        raise RuntimeError("confirm_endpoint_enabled exact value missing. Transfer submission is blocked.")
    if CONFIRM_TRANSFER != TRANSFER_CONFIRMATION:
        raise RuntimeError("confirm_transfer exact value missing. Transfer submission is blocked.")
    amount = parse_transfer_amount()
    currency = TRANSFER_CURRENCY.strip().upper()
    if currency not in {OLD_SGD_CURRENCY, NEW_USD_CURRENCY}:
        raise RuntimeError("transfer_currency must be SGD or USD and must match the confirmed Saxo transfer module terms.")
    payload = {
        "FromAccountKey": FROM_ACCOUNT_KEY,
        "ToAccountKey": TO_ACCOUNT_KEY,
        "Amount": float(amount),
        "Currency": currency,
        "ExternalReference": f"v4_sgd_usd_cutover_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    }
    resp, response_payload, retry_count = saxo_request(
        "POST",
        f"{BASE_URL}{CS_INTERACCOUNT_ENDPOINT}",
        hdrs,
        payload=payload,
        action_name="submit_interaccount_transfer",
    )
    add_diag(
        "transfer_submit",
        "submitted" if 200 <= resp.status_code < 300 else "error",
        status_code=resp.status_code,
        retry_count=retry_count,
        amount=str(amount),
        currency=currency,
        response=json.dumps(redact_obj(response_payload, REDACTION_SECRETS))[:1000],
    )
    if resp.status_code < 200 or resp.status_code >= 300:
        raise RuntimeError(
            f"Saxo interaccount transfer failed status={resp.status_code}: "
            f"{redact_text(resp.text[:500], REDACTION_SECRETS)}"
        )
    return response_payload


def run():
    hdrs = headers()
    accounts = fetch_accounts(hdrs)
    verify_account(accounts, FROM_ACCOUNT_KEY, OLD_SGD_ACCOUNT_ID, OLD_SGD_CURRENCY, "from_sgd")
    verify_account(accounts, TO_ACCOUNT_KEY, NEW_USD_ACCOUNT_ID, NEW_USD_CURRENCY, "to_usd")
    fetch_balance(hdrs, FROM_ACCOUNT_KEY, "from_sgd")
    fetch_balance(hdrs, TO_ACCOUNT_KEY, "to_usd")
    add_diag(
        "account_route",
        "ok",
        from_account_id=OLD_SGD_ACCOUNT_ID,
        to_account_id=NEW_USD_ACCOUNT_ID,
        warning="Never assume sale proceeds are immediately available for transfer.",
    )

    capability = discover_transfer_capability(hdrs)
    if OPERATION_MODE == "discover":
        add_diag("transfer_action", "skipped", reason="discover_only")
        print("Discovery complete. No transfer submitted.")
        print("Never assume sale proceeds are immediately available.")
    else:
        if not capability["endpoint_possible"]:
            add_diag("transfer_action", "blocked", reason="endpoint_not_confirmed_enabled")
            print(MANUAL_ROUTE_MESSAGE)
            raise RuntimeError("Transfer endpoint availability is unclear; API transfer is blocked.")
        submit_transfer(hdrs)
        print("Transfer request submitted. Confirm final status in Saxo before trading.")

    path = write_diagnostics()
    print(f"Diagnostic CSV: {path}")
    print(f"Operation mode: {OPERATION_MODE}")
    print(f"From account verified: {OLD_SGD_ACCOUNT_ID} {OLD_SGD_CURRENCY}")
    print(f"To account verified: {NEW_USD_ACCOUNT_ID} {NEW_USD_CURRENCY}")
    if not capability["endpoint_possible"]:
        print(MANUAL_ROUTE_MESSAGE)


try:
    run()
except Exception as exc:
    add_diag("notebook_error", "error", error=redact_text(str(exc), REDACTION_SECRETS))
    path = write_diagnostics()
    print(f"Diagnostic CSV: {path}")
    print(f"Blocked/error: {redact_text(str(exc), REDACTION_SECRETS)}")
    raise

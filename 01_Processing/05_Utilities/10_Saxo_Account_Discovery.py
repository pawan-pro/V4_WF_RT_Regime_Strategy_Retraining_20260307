# Databricks notebook source
# Saxo account discovery and temporary gated USD AccountKey secret handoff.
# Normal discover mode prints redacted account metadata only.

import ast
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

BASE_URL = "https://gateway.saxobank.com/openapi"
CLIENT_KEY = "hLMMvwa3Mzmiem2AMhpKwA=="
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"

USD_ACCOUNT_ID = "33888/261196"
USD_CURRENCY = "USD"
SECRET_SCOPE = "saxo"
SECRET_KEY = "usd_account_key"
STORE_SECRET_CONFIRMATION = "STORE_USD_ACCOUNT_KEY_33888_261196"
NO_SECRET_WRITE_MESSAGE = (
    "Databricks notebook cannot write secrets here. Use Databricks CLI/UI and copy "
    "the full key from a temporary local variable in an interactive private run."
)

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
CONFIRM_STORE_SECRET = widget("confirm_store_secret", "")

if OPERATION_MODE not in {"discover", "store_usd_key_secret"}:
    raise ValueError("operation_mode must be discover or store_usd_key_secret")


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


def first_path(payload: dict, *paths: str) -> Optional[Any]:
    for path in paths:
        value = get_path(payload, path)
        if value is not None:
            return value
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


def redact_account(account: dict) -> dict:
    account_key = first_path(account, "AccountKey", "Key")
    redacted = {}
    for key in ("AccountId", "AccountNumber", "Currency", "AccountType", "Active", "Status"):
        value = account.get(key)
        if value is not None:
            redacted[key] = value
    redacted["AccountKey"] = "[REDACTED]" if account_key else None
    redacted["AccountKeyLength"] = len(str(account_key)) if account_key else 0
    redacted["AccountKeySha256_12"] = fingerprint(str(account_key)) if account_key else None
    return redacted


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


def saxo_get(url: str, hdrs: dict, action_name: str = "request") -> Any:
    last_status = None
    last_payload = {}
    last_text = ""
    for attempt in range(1, SAXO_RETRY_MAX_ATTEMPTS + 1):
        resp = requests.get(url, headers=hdrs, timeout=30)
        response_text = resp.text or ""
        response_payload = safe_json_loads(response_text) or {}
        if not is_rate_limited(resp.status_code, response_payload, response_text):
            if resp.status_code < 200 or resp.status_code >= 300:
                raise RuntimeError(
                    f"Saxo {action_name} failed status={resp.status_code} "
                    f"response={redact_text(response_text[:500])}"
                )
            return response_payload
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
                f"{error_info['code']} - {redact_text(error_info['message'])}"
            )
        wait_seconds = min(SAXO_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), SAXO_RETRY_MAX_SECONDS)
        print(f"Saxo {action_name} rate-limited on attempt {attempt}; sleeping {wait_seconds:.1f}s")
        time.sleep(wait_seconds)
    raise SaxoTemporaryThrottleError(
        f"Saxo {action_name} deferred: status={last_status} "
        f"payload={redact_text(json.dumps(last_payload)[:500] or last_text[:500])}"
    )


def fetch_accounts() -> list[dict]:
    payload = saxo_get(
        f"{BASE_URL}/port/v1/accounts/me?ClientKey={CLIENT_KEY}",
        headers(),
        action_name="fetch_accounts",
    )
    data = payload.get("Data", []) if isinstance(payload, dict) else []
    return [row for row in data if isinstance(row, dict)]


def account_id(account: dict) -> Optional[str]:
    value = first_path(account, "AccountId", "AccountNumber")
    return str(value) if value is not None else None


def account_currency(account: dict) -> Optional[str]:
    value = first_path(account, "Currency", "AccountCurrency")
    return str(value).upper() if value is not None else None


def account_key(account: dict) -> Optional[str]:
    value = first_path(account, "AccountKey", "Key")
    return str(value) if value else None


def find_usd_account(accounts: list[dict]) -> dict:
    matches = [account for account in accounts if account_id(account) == USD_ACCOUNT_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one USD account {USD_ACCOUNT_ID}; found {len(matches)}.")
    account = matches[0]
    if account_currency(account) != USD_CURRENCY:
        raise RuntimeError(
            f"USD account guard failed. Expected currency {USD_CURRENCY}; got {account_currency(account)}."
        )
    if not account_key(account):
        raise RuntimeError(f"USD account {USD_ACCOUNT_ID} has no AccountKey in Saxo discovery payload.")
    return account


def static_validate_no_full_key_prints() -> bool:
    try:
        with open(__file__, "r", encoding="utf-8") as source_file:
            source = source_file.read()
    except Exception:
        print("Static validation skipped: notebook source path is unavailable in this runtime.")
        return True

    violations = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            risky_names = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and child.id in {"secret_value", "usd_account_key_private_handoff"}
            }
            risky_calls = [
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "account_key"
            ]
            if risky_names or risky_calls:
                violations.append(f"unsafe print call: {segment[:120]}")
        if re.search(r"dbutils\.notebook\.exit\(|to_csv\(|dbutils\.fs\.put\(", segment):
            violations.append(f"unsafe write/exit call: {segment[:120]}")
    if violations:
        raise RuntimeError(f"Static validation failed for unsafe output/write patterns: {violations}")
    print("Static validation: no full key print/write patterns detected.")
    return True


def print_discovery(accounts: list[dict], usd_account: dict):
    usd_key = account_key(usd_account)
    usd_key_present = bool(usd_key)
    usd_key_length = len(usd_key or "")
    usd_key_fingerprint = fingerprint(usd_key)
    print(f"Saxo account discovery mode: {OPERATION_MODE}")
    print(f"Discovered account count: {len(accounts)}")
    print("Accounts:")
    for item in accounts:
        print(json.dumps(redact_account(item), sort_keys=True))
    print(
        "USD account verified:",
        USD_ACCOUNT_ID,
        account_currency(usd_account),
        "key_present:",
        usd_key_present,
        "key_length:",
        usd_key_length,
        "key_sha256_12:",
        usd_key_fingerprint,
    )


def store_usd_key_secret(usd_account: dict):
    if CONFIRM_STORE_SECRET != STORE_SECRET_CONFIRMATION:
        raise RuntimeError("confirm_store_secret exact value missing. Secret handoff is blocked.")
    secret_value = account_key(usd_account)
    if not secret_value:
        raise RuntimeError("USD AccountKey is missing. Secret handoff is blocked.")

    static_validate_no_full_key_prints()
    try:
        put_secret = dbutils.secrets.put
    except Exception:
        print(NO_SECRET_WRITE_MESSAGE)
        return False

    try:
        put_secret(SECRET_SCOPE, SECRET_KEY, secret_value)
    except Exception as exc:
        redacted_error = redact_text(str(exc), [secret_value])
        print(NO_SECRET_WRITE_MESSAGE)
        print(f"Secret write failed: {type(exc).__name__}: {redacted_error}")
        return False

    v = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
    print("loaded:", bool(v), "length:", len(v), "redacted:", v[:4] + "..." + v[-4:])
    return True


def run():
    accounts = fetch_accounts()
    usd_account = find_usd_account(accounts)
    if OPERATION_MODE == "discover":
        print_discovery(accounts, usd_account)
        return

    store_usd_key_secret(usd_account)


run()

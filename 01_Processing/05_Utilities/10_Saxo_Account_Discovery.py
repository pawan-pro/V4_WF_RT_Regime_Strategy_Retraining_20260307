# Databricks notebook source
# Saxo account discovery. Prints redacted account metadata only.

import ast
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

BASE_URL = "https://gateway.saxobank.com/openapi"
TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"

USD_ACCOUNT_ID = "33888/261196"
USD_CURRENCY = "USD"
SGD_ACCOUNT_ID = "33888/256469"
SGD_CURRENCY = "SGD"
SECRET_SCOPE = "saxo"
CLIENT_KEY_SECRET_KEY = "client_key"
USD_SECRET_KEY = "usd_account_key"
SGD_SECRET_KEY = "sgd_account_key"
STORE_USD_SECRET_CONFIRMATION = "STORE_USD_ACCOUNT_KEY_33888_261196"
STORE_SGD_SECRET_CONFIRMATION = "STORE_SGD_ACCOUNT_KEY_33888_256469"
REVEAL_SGD_SECRET_CONFIRMATION = "REVEAL_SGD_ACCOUNT_KEY_33888_256469_ONCE"
NO_SECRET_WRITE_MESSAGE = (
    "Databricks notebook cannot write secrets here. Use Databricks CLI/UI and copy the full key "
    "from a temporary local variable in an interactive private run."
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
CONFIRM_REVEAL_SECRET = widget("confirm_reveal_secret", "")

ALLOWED_OPERATION_MODES = {
    "discover",
    "store_usd_key_secret",
    "store_sgd_key_secret",
    "reveal_sgd_key_once",
}
if OPERATION_MODE not in ALLOWED_OPERATION_MODES:
    raise ValueError(f"operation_mode must be one of {sorted(ALLOWED_OPERATION_MODES)}")


def resolve_client_key() -> str:
    try:
        value = dbutils.secrets.get(SECRET_SCOPE, CLIENT_KEY_SECRET_KEY).strip()
    except Exception:
        value = os.environ.get("SAXO_CLIENT_KEY", "").strip()
    if not value:
        raise RuntimeError("Saxo ClientKey is not configured in saxo/client_key or SAXO_CLIENT_KEY.")
    return value


CLIENT_KEY = resolve_client_key()


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
    for secret in [CLIENT_KEY] + (secrets or []):
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


def find_account(accounts: list[dict], expected_account_id: str, expected_currency: str) -> dict:
    matches = [account for account in accounts if account_id(account) == expected_account_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one account {expected_account_id}; found {len(matches)}.")
    account = matches[0]
    if account_currency(account) != expected_currency:
        raise RuntimeError(
            f"Account guard failed for {expected_account_id}. "
            f"Expected currency {expected_currency}; got {account_currency(account)}."
        )
    if not account_key(account):
        raise RuntimeError(f"Account {expected_account_id} has no AccountKey in Saxo discovery payload.")
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

    parent_by_id = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_id[id(child)] = parent

    def enclosing_function_name(node: ast.AST) -> Optional[str]:
        current = node
        while id(current) in parent_by_id:
            current = parent_by_id[id(current)]
            if isinstance(current, ast.FunctionDef):
                return current.name
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            risky_names = {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and child.id in {"secret_value", "sgd_account_key_private_handoff"}
            }
            risky_calls = [
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "account_key"
            ]
            if risky_names or risky_calls:
                if enclosing_function_name(node) != "reveal_sgd_account_key_once":
                    violations.append(f"unsafe print call: {segment[:120]}")
        if re.search(r"dbutils\.notebook\.exit\(|to_csv\(|dbutils\.fs\.put\(", segment):
            violations.append(f"unsafe write/exit call: {segment[:120]}")
    if violations:
        raise RuntimeError(f"Static validation failed for unsafe output/write patterns: {violations}")
    print("Static validation: no full key print/write patterns outside gated SGD reveal mode detected.")
    return True


def print_discovery(accounts: list[dict], target_accounts: list[tuple[str, dict]]):
    print(f"Saxo account discovery mode: {OPERATION_MODE}")
    print(f"ClientKey loaded: {bool(CLIENT_KEY)} length: {len(CLIENT_KEY)} redacted: {CLIENT_KEY[:4]}...{CLIENT_KEY[-4:]}")
    print(f"Discovered account count: {len(accounts)}")
    print("Accounts:")
    for item in accounts:
        print(json.dumps(redact_account(item), sort_keys=True))
    for label, account in target_accounts:
        key_value = account_key(account)
        print(
            f"{label} account verified:",
            account_id(account),
            account_currency(account),
            "key_present:",
            bool(key_value),
            "key_length:",
            len(key_value or ""),
            "key_sha256_12:",
            fingerprint(key_value),
        )


def store_account_key_secret(account: dict, secret_key: str, confirmation: str, label: str) -> None:
    if CONFIRM_STORE_SECRET != confirmation:
        raise RuntimeError(f"{label} secret storage requires exact confirm_store_secret.")

    secret_value = account_key(account)
    if not secret_value:
        raise RuntimeError(f"{label} AccountKey is missing.")

    print(
        f"{label} account secret storage precheck:",
        "AccountId:",
        account_id(account),
        "Currency:",
        account_currency(account),
        "key_present:",
        bool(secret_value),
        "key_length:",
        len(secret_value),
        "key_sha256_12:",
        fingerprint(secret_value),
    )

    try:
        put_secret = dbutils.secrets.put
    except Exception:
        print(NO_SECRET_WRITE_MESSAGE)
        raise RuntimeError("Databricks notebook secret write is unavailable.")

    try:
        put_secret(SECRET_SCOPE, secret_key, secret_value)
    except Exception as exc:
        print(NO_SECRET_WRITE_MESSAGE)
        raise RuntimeError("Databricks notebook secret write failed.") from exc

    loaded_value = dbutils.secrets.get(SECRET_SCOPE, secret_key)
    if loaded_value != secret_value:
        raise RuntimeError(f"Secret write verification failed for {SECRET_SCOPE}/{secret_key}.")
    print("loaded:", bool(loaded_value), "length:", len(loaded_value), "redacted:", loaded_value[:4] + "..." + loaded_value[-4:])


def context_tag(name: str) -> str:
    try:
        tags = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags()
        value = tags.get(name)
        if hasattr(value, "isDefined") and value.isDefined():
            return str(value.get())
        if value:
            return str(value)
    except Exception:
        pass
    return ""


def running_as_job_task() -> bool:
    return bool(context_tag("jobId") or context_tag("jobRunId") or context_tag("taskKey"))


def reveal_sgd_account_key_once(account: dict) -> None:
    if running_as_job_task():
        raise RuntimeError("SGD AccountKey reveal is blocked for job task runs. Use a private interactive notebook run only.")
    if CONFIRM_REVEAL_SECRET != REVEAL_SGD_SECRET_CONFIRMATION:
        raise RuntimeError("SGD AccountKey reveal requires exact confirm_reveal_secret.")

    sgd_account_key_private_handoff = account_key(account)
    if not sgd_account_key_private_handoff:
        raise RuntimeError("SGD AccountKey is missing.")

    print("PRIVATE INTERACTIVE RUN ONLY. COPY INTO DATABRICKS SECRET IMMEDIATELY. CLEAR OUTPUT AFTER USE.")
    print(sgd_account_key_private_handoff)
    print("databricks secrets put-secret saxo sgd_account_key --profile pawan-quant")
    print("After storing the secret, clear notebook output.")
    print("Revert/remove reveal_sgd_key_once mode before final commit if possible.")


def run():
    static_validate_no_full_key_prints()
    accounts = fetch_accounts()
    usd_account = find_account(accounts, USD_ACCOUNT_ID, USD_CURRENCY)
    sgd_account = find_account(accounts, SGD_ACCOUNT_ID, SGD_CURRENCY)

    if OPERATION_MODE == "store_usd_key_secret":
        store_account_key_secret(usd_account, USD_SECRET_KEY, STORE_USD_SECRET_CONFIRMATION, "USD")
        return
    if OPERATION_MODE == "store_sgd_key_secret":
        store_account_key_secret(sgd_account, SGD_SECRET_KEY, STORE_SGD_SECRET_CONFIRMATION, "SGD")
        return
    if OPERATION_MODE == "reveal_sgd_key_once":
        reveal_sgd_account_key_once(sgd_account)
        return

    print_discovery(accounts, [("USD", usd_account), ("SGD", sgd_account)])


run()

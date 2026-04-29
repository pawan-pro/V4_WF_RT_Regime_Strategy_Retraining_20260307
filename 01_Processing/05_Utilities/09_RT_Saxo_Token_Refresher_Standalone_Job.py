# Databricks notebook source
# Standalone Saxo token refresher.
# This remains the only token writer in the automated flow.

import json
import time
from datetime import datetime, timezone

import requests
from requests.exceptions import Timeout

TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"
LOCK_PATH = TOKEN_STATE_PATH + ".lock"
INCIDENT_MARKER_PATH = TOKEN_STATE_PATH + ".refresh_timeout_incident.json"
LOCK_TTL_SECONDS = 1800
REFRESH_TIMEOUT_SECONDS = 45

AUTH_BASE_URL = "https://live.logonvalidation.net"
TOKEN_URL = f"{AUTH_BASE_URL}/token"

CLIENT_ID = "5b0974db36f54f3b96ec07b8f5fc657d"
CLIENT_SECRET = "8a7f591c89ca4f0299256b041ba891e2"
REDIRECT_URI = "https://localhost/callback"


def load_token():
    return json.loads(dbutils.fs.head(TOKEN_STATE_PATH))


def atomic_write_json(path, payload):
    temp_path = path + ".tmp"
    dbutils.fs.put(temp_path, json.dumps(payload), overwrite=True)
    dbutils.fs.mv(temp_path, path, True)


def acquire_lock():
    for attempt in range(2):
        try:
            dbutils.fs.put(LOCK_PATH, str(time.time()), overwrite=False)
            return True
        except Exception:
            if attempt == 0 and clear_stale_lock():
                continue
            return False


def release_lock():
    try:
        dbutils.fs.rm(LOCK_PATH)
    except Exception:
        pass


def read_lock_timestamp():
    try:
        return float(dbutils.fs.head(LOCK_PATH, 256).strip())
    except Exception:
        return None


def clear_stale_lock():
    lock_ts = read_lock_timestamp()
    if lock_ts is None:
        return False

    age_seconds = time.time() - lock_ts
    if age_seconds <= LOCK_TTL_SECONDS:
        return False

    print(f"Removing stale Saxo token lock after {int(age_seconds)} seconds")
    try:
        dbutils.fs.rm(LOCK_PATH)
        return True
    except Exception:
        return False


def write_timeout_incident_marker(state, error):
    marker = {
        "incident_type": "saxo_token_refresh_timeout",
        "incident_at": datetime.now(timezone.utc).isoformat(),
        "error_type": type(error).__name__,
        "token_state_updated_at": state.get("updated_at") or state.get("created_at"),
        "chain_id": state.get("chain_id"),
        "source": state.get("source"),
    }
    atomic_write_json(INCIDENT_MARKER_PATH, marker)


def refresh():
    state = load_token()
    now = time.time()

    if state["expires_at"] - 1200 > now:
        print("Token still valid — skipping refresh")
        return

    print("Refreshing Saxo token...")
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": state["refresh_token"],
                "redirect_uri": REDIRECT_URI,
            },
            auth=(CLIENT_ID, CLIENT_SECRET),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REFRESH_TIMEOUT_SECONDS,
        )
    except Timeout as exc:
        try:
            write_timeout_incident_marker(state, exc)
        except Exception as marker_exc:
            print(f"Failed to write timeout incident marker: {type(marker_exc).__name__}")
        raise RuntimeError(
            "Token refresh timed out; token state may be ambiguous. "
            "Run OAuth bootstrap or manual validation before relying on next refresh."
        ) from exc

    if resp.status_code == 401:
        raise RuntimeError("Refresh token invalid — chain broken. Run OAuth bootstrap.")

    resp.raise_for_status()
    token = resp.json()

    if "access_token" not in token or "refresh_token" not in token:
        raise RuntimeError("Invalid token response")

    new_state = {
        "access_token": token["access_token"],
        "refresh_token": token["refresh_token"],
        "expires_at": time.time() + token["expires_in"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "chain_id": state.get("chain_id"),
        "source": "token_refresher",
    }

    atomic_write_json(TOKEN_STATE_PATH, new_state)
    print("Token refreshed successfully")


if not acquire_lock():
    print("Another refresh in progress — skipping")
    dbutils.notebook.exit("SKIPPED")

try:
    refresh()
finally:
    release_lock()

# Databricks notebook source
# Standalone Saxo token refresher.
# This remains the only token writer in the automated flow.

import json
import time
from datetime import datetime, timezone

import requests

TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"
LOCK_PATH = TOKEN_STATE_PATH + ".lock"
INCIDENT_PATH = TOKEN_STATE_PATH + ".incident.json"
LOCK_TTL_SECONDS = 1800

AUTH_BASE_URL = "https://live.logonvalidation.net"
TOKEN_URL = f"{AUTH_BASE_URL}/token"

CLIENT_ID = "5b0974db36f54f3b96ec07b8f5fc657d"
CLIENT_SECRET = "8a7f591c89ca4f0299256b041ba891e2"
REDIRECT_URI = "https://localhost/callback"


def load_token():
    return json.loads(dbutils.fs.head(TOKEN_STATE_PATH))


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


def write_refresh_timeout_incident(exc):
    incident = {
        "incident_type": "saxo_token_refresh_timeout",
        "status": "ambiguous_token_chain_state",
        "token_state_path": TOKEN_STATE_PATH,
        "token_url_host": AUTH_BASE_URL,
        "error_type": type(exc).__name__,
        "message": (
            "Saxo refresh POST timed out before a definitive response was received. "
            "The refresh token may or may not have been consumed by Saxo. "
            "Canonical token state was not overwritten."
        ),
        "action_required": (
            "Do not retry blindly in a tight loop. Check Saxo auth state and run OAuth "
            "bootstrap only if the token chain is confirmed broken or the next safe "
            "refresh attempt fails with an explicit chain-broken response."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    dbutils.fs.put(INCIDENT_PATH, json.dumps(incident, indent=2), overwrite=True)


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
            timeout=30,
        )
    except (
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.Timeout,
    ) as exc:
        write_refresh_timeout_incident(exc)
        raise RuntimeError(
            "Saxo token refresh timed out before a definitive response. "
            "Token-chain state is ambiguous; canonical token state was not overwritten. "
            f"Non-secret incident metadata written to {INCIDENT_PATH}. "
            "Investigate Saxo auth state before retrying or bootstrapping."
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

    temp_path = TOKEN_STATE_PATH + ".tmp"
    dbutils.fs.put(temp_path, json.dumps(new_state), overwrite=True)
    dbutils.fs.mv(temp_path, TOKEN_STATE_PATH, True)
    print("Token refreshed successfully")


if not acquire_lock():
    print("Another refresh in progress — skipping")
    dbutils.notebook.exit("SKIPPED")

try:
    refresh()
finally:
    release_lock()

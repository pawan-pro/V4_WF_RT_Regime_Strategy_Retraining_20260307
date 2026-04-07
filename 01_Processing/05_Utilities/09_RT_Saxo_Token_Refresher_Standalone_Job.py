# Databricks notebook source
# Standalone Saxo token refresher.
# This remains the only token writer in the automated flow.

import json
import time
from datetime import datetime, timezone

import requests

TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"
LOCK_PATH = TOKEN_STATE_PATH + ".lock"

AUTH_BASE_URL = "https://live.logonvalidation.net"
TOKEN_URL = f"{AUTH_BASE_URL}/token"

CLIENT_ID = "5b0974db36f54f3b96ec07b8f5fc657d"
CLIENT_SECRET = "8a7f591c89ca4f0299256b041ba891e2"
REDIRECT_URI = "https://localhost/callback"


def load_token():
    return json.loads(dbutils.fs.head(TOKEN_STATE_PATH))


def acquire_lock():
    try:
        dbutils.fs.put(LOCK_PATH, str(time.time()), overwrite=False)
        return True
    except Exception:
        return False


def release_lock():
    try:
        dbutils.fs.rm(LOCK_PATH)
    except Exception:
        pass


def refresh():
    state = load_token()
    now = time.time()

    if state["expires_at"] - 1200 > now:
        print("Token still valid — skipping refresh")
        return

    print("Refreshing Saxo token...")
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

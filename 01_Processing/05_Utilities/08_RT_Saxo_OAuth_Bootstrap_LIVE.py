# Databricks notebook source
# Saxo OAuth bootstrap for the live account.
# This remains a manual notebook and is intentionally not part of a scheduled job.

import json
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

AUTH_BASE_URL = "https://live.logonvalidation.net"
TOKEN_URL = f"{AUTH_BASE_URL}/token"
REDIRECT_URI = "https://localhost/callback"

CLIENT_ID = "5b0974db36f54f3b96ec07b8f5fc657d"
CLIENT_SECRET = "8a7f591c89ca4f0299256b041ba891e2"

TOKEN_STATE_PATH = "dbfs:/Volumes/pawan_quant_research_1/default/saxo/avik_live_token_state.json"

print("Initializing Saxo OAuth bootstrap...")
print("Token state path:", TOKEN_STATE_PATH)

params = {
    "response_type": "code",
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "state": "autobot",
}
auth_url = f"{AUTH_BASE_URL}/authorize?{urlencode(params)}"

print("\nOpen this URL in your browser and login to Saxo:\n")
print(auth_url)

code = input("Paste the authorization code here: ").strip()
if not code:
    raise RuntimeError("Authorization code missing.")

resp = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
    },
    auth=(CLIENT_ID, CLIENT_SECRET),
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    timeout=30,
)
resp.raise_for_status()
token = resp.json()

if "access_token" not in token or "refresh_token" not in token:
    raise RuntimeError(f"Invalid OAuth response: {token}")

state = {
    "access_token": token["access_token"],
    "refresh_token": token["refresh_token"],
    "expires_at": time.time() + token["expires_in"],
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": "oauth_bootstrap",
    "chain_id": str(int(time.time())),
    "version": "v1",
}

temp_path = TOKEN_STATE_PATH + ".tmp"
dbutils.fs.put(temp_path, json.dumps(state), overwrite=True)
dbutils.fs.mv(temp_path, TOKEN_STATE_PATH, True)

print("Token saved successfully.")
print(dbutils.fs.head(TOKEN_STATE_PATH))

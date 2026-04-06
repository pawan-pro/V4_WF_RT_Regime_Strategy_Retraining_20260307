# Databricks notebook source
import requests

ACCESS_TOKEN = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjY3NEM0MjFEMzZEMUE1OUNFNjFBRTIzMjMyOTVFRTAyRTc3MDMzNTkifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiLW5KdnxZcVc3QkgwLU1GaG80RGtoZz09IiwiY2lkIjoiLW5KdnxZcVc3QkgwLU1GaG80RGtoZz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiMTMxOWUzNjUyODdiNGVkMmI0NjgwMWY1Mjk5MzU3NGQiLCJkZ2kiOiI4NCIsImV4cCI6IjE3NzIyNTgwNDIiLCJvYWwiOiIxRiIsImlpZCI6IjE1Yjk2N2NlMGFiNzQ2YjMxNDM2MDhkZTcxMGM2ZDk2In0.iL45XaMX_-ymYS9_J0j306zGMi6_5UQjNjXZYXjqIB3jN161Ncm076_L-uMU2pTZdgEqc4f0mTV7oNdGSusvKw"
BASE_URL = "https://gateway.saxobank.com/sim/openapi"  # simulation env

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

def main():
    # simple “who am I / what accounts do I have” style call
    url = f"{BASE_URL}/port/v1/accounts/me"
    resp = requests.get(url, headers=headers)
    print("Status:", resp.status_code)
    print("Body:", resp.text)

if __name__ == "__main__":
    main()


# COMMAND ----------

import requests

ACCESS_TOKEN = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjY3NEM0MjFEMzZEMUE1OUNFNjFBRTIzMjMyOTVFRTAyRTc3MDMzNTkifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiLW5KdnxZcVc3QkgwLU1GaG80RGtoZz09IiwiY2lkIjoiLW5KdnxZcVc3QkgwLU1GaG80RGtoZz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiMTMxOWUzNjUyODdiNGVkMmI0NjgwMWY1Mjk5MzU3NGQiLCJkZ2kiOiI4NCIsImV4cCI6IjE3NzIyNTgwNDIiLCJvYWwiOiIxRiIsImlpZCI6IjE1Yjk2N2NlMGFiNzQ2YjMxNDM2MDhkZTcxMGM2ZDk2In0.iL45XaMX_-ymYS9_J0j306zGMi6_5UQjNjXZYXjqIB3jN161Ncm076_L-uMU2pTZdgEqc4f0mTV7oNdGSusvKw"

BASE_URL = "https://gateway.saxobank.com/sim/openapi"

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

CLIENT_KEY = "-nJv|YqW7BH0-MFho4Dkhg=="   # from /accounts/me
ACCOUNT_KEY = "-nJv|YqW7BH0-MFho4Dkhg=="

def test_balances():
    url = f"{BASE_URL}/port/v1/balances?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}"
    r = requests.get(url, headers=headers)
    print("Balances:", r.status_code, r.text)

def test_positions():
    url = f"{BASE_URL}/port/v1/positions?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}"
    r = requests.get(url, headers=headers)
    print("Positions:", r.status_code, r.text)

if __name__ == "__main__":
    test_balances()
    test_positions()


# COMMAND ----------

import requests, json

BASE_URL = "https://gateway.saxobank.com/sim/openapi"

ACCOUNT_KEY = "-nJv|YqW7BH0-MFho4Dkhg=="
CLIENT_KEY = "-nJv|YqW7BH0-MFho4Dkhg=="

QQQ_UIC = 4328771
QQQ_ASSET_TYPE = "Etf"

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

def place_qqq_market_buy(qty=1):
    url = f"{BASE_URL}/trade/v2/orders"
    payload = {
        "AccountKey": ACCOUNT_KEY,
        "AssetType": QQQ_ASSET_TYPE,
        "Uic": QQQ_UIC,
        "BuySell": "Buy",
        "OrderType": "Market",
        "Amount": qty,
        "ManualOrder": False,          # <- REQUIRED BY PLATFORM
        "OrderDuration": {"DurationType": "DayOrder"}
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    print("ORDER:", r.status_code)
    print(r.text)

def show_positions():
    url = f"{BASE_URL}/port/v1/positions?ClientKey={CLIENT_KEY}&AccountKey={ACCOUNT_KEY}"
    r = requests.get(url, headers=headers)
    print("POSITIONS:", r.status_code)
    print(r.text)

if __name__ == "__main__":
    place_qqq_market_buy(qty=1)
    show_positions()


# COMMAND ----------

import requests


BASE_URL = "https://gateway.saxobank.com/sim/openapi"
ACCOUNT_KEY = "-nJv|YqW7BH0-MFho4Dkhg=="
CLIENT_KEY = "-nJv|YqW7BH0-MFho4Dkhg=="
ORDER_ID = "5037248169"

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

def check_order_status():
    url = f"{BASE_URL}/trade/v2/orders/{ORDER_ID}"
    r = requests.get(url, headers=headers)
    print("ORDER_STATUS:", r.status_code)
    print(r.text)

def check_positions_me():
    url = f"{BASE_URL}/port/v1/positions/me"
    r = requests.get(url, headers=headers)
    print("POSITIONS_ME:", r.status_code)
    print(r.text)

check_order_status()
check_positions_me()


# COMMAND ----------

import time

place_qqq_market_buy(qty=1)

print("Sleeping 5 seconds before checking positions...")
time.sleep(5)

show_positions()          # with ClientKey & AccountKey

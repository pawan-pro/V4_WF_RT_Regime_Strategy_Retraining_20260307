# Databricks notebook source
# MAGIC %md
# MAGIC CELL 1 — Setup

# COMMAND ----------

# RT_07: Institutional Dashboard Feed

from pyspark.sql import functions as F
import pandas as pd
import json
import matplotlib.pyplot as plt

print("Dashboard initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 2 — Table Locations

# COMMAND ----------

AUDIT_TABLE = "pawan_quant_research_1.v4.bronze_rt_signal_audit_log"
TRADE_TABLE = "pawan_quant_research_1.v4.rt_live_trade_log"
SILVER_RT   = "pawan_quant_research_1.v4.silver_rt_1h"

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 3 — Latest Audit Row

# COMMAND ----------

latest = spark.sql(f"""
SELECT *
FROM {AUDIT_TABLE}
ORDER BY audit_ts DESC
LIMIT 1
""").toPandas()

latest = latest.iloc[0]

print("Latest regime:", latest["regime_name"])
print("Position:", latest["position"])
print("Equity:", latest["account_equity"])

# COMMAND ----------

# ─────────────────────────────────────────────
# Probabilities (safe fetch)
# ─────────────────────────────────────────────

prob_expansion   = latest.get("prob_expansion", None)
prob_slowdown    = latest.get("prob_slowdown", None)
prob_contraction = latest.get("prob_contraction", None)

if prob_expansion is not None:
    print(f"\nProbabilities:")
    print(f"Expansion   : {prob_expansion:.2f}")
    print(f"Slowdown    : {prob_slowdown:.2f}")
    print(f"Contraction : {prob_contraction:.2f}")

# COMMAND ----------

# ─────────────────────────────────────────────
# Regime Probability Chart
# ─────────────────────────────────────────────

if prob_expansion is not None:

    labels = ["Expansion", "Slowdown", "Contraction"]
    probs  = [prob_expansion, prob_slowdown, prob_contraction]

    plt.figure(figsize=(6,4))
    plt.bar(labels, probs)

    plt.title("Regime Probabilities")
    plt.ylim(0,1)

    for i, v in enumerate(probs):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC CELL 4 — System Status Panel

# COMMAND ----------

status = {
    "Regime": latest["regime_name"],
    "Position": latest["position"],
    "Signal Timestamp": latest["signal_ts"],
    "Model Version": latest["model_version"],
    "Bias Gate": latest["bias_gate_passed"],
    "VIX Lag Days": latest["pit_lag_days"],
    "Prob (Exp)": prob_expansion,
    "Prob (Slow)": prob_slowdown,
    "Prob (Contr)": prob_contraction
}

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 5 — Portfolio State

# COMMAND ----------

portfolio = spark.sql(f"""
SELECT
audit_ts,
account_equity,
holding_asset,
holding_qty,
holding_price
FROM {AUDIT_TABLE}
ORDER BY audit_ts DESC
LIMIT 1
""").toPandas()

portfolio

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 6 — Feature Z-Score Table

# COMMAND ----------

# ─────────────────────────────────────────────
# CELL 6 — Feature Z-Score Table (fixed)
# ─────────────────────────────────────────────

import numpy as np

def safe_get(row, key):
    try:
        val = row.get(key, None)
        if val is None:
            return np.nan
        return float(val)
    except:
        return np.nan

# fallback from silver if missing
silver_latest = spark.sql(f"""
SELECT *
FROM {SILVER_RT}
ORDER BY timestamp DESC
LIMIT 1
""").toPandas().iloc[0]

def get_feature(key):
    val = safe_get(latest, key)
    if pd.isna(val):
        val = safe_get(silver_latest, key)
    return val

features = {
    "VIX": get_feature("VIXCLS_z"),
    "VIX Term": get_feature("vix3m_z"),
    "VIX Inversion": get_feature("VIX_Inversion_Z"),
    "Momentum": get_feature("Price_Mom_24H"),
    "Inflation": get_feature("CPIAUCSL_z"),
    "Employment": get_feature("PAYEMS_z"),
    "Growth": get_feature("GDPC1_z"),
    "Industrial": get_feature("INDPRO_z"),
    "Sentiment": get_feature("UMCSENT_z")
}

feat_df = pd.DataFrame(
    list(features.items()),
    columns=["Feature", "ZScore"]
)

# debug check
print("z-score null check:")
print(feat_df[feat_df["ZScore"].isna()])

feat_df

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 7 — Feature Z-Score Chart

# COMMAND ----------

plt.figure(figsize=(8,4))

plt.barh(feat_df["Feature"], feat_df["ZScore"])

plt.axvline(0,color="black")
plt.axvline(1,color="green",linestyle="--")
plt.axvline(-1,color="red",linestyle="--")

plt.title("Macro Feature Z-Scores")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 8 — Radar Chart

# COMMAND ----------

# ─────────────────────────────────────────────
# Improved Radar Chart
# ─────────────────────────────────────────────

import numpy as np

labels = list(features.keys())
values = list(features.values())

# close loop
values += values[:1]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.append(angles, angles[0])

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)

ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)

# Fix label placement
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)

# Improve spacing
ax.tick_params(pad=10)

plt.title("Macro Regime Radar", pad=50)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 9 — Regime History

# COMMAND ----------

# ─────────────────────────────────────────────
# CELL 9 — Regime History (fixed)
# ─────────────────────────────────────────────

regime_hist = spark.sql(f"""
SELECT audit_ts, regime_name
FROM {AUDIT_TABLE}
ORDER BY audit_ts
""").toPandas()

# ensure datetime
regime_hist["audit_ts"] = pd.to_datetime(regime_hist["audit_ts"])

# filter from Mar 17
cutoff_date = pd.Timestamp("2026-03-17")
regime_hist = regime_hist[regime_hist["audit_ts"] >= cutoff_date]

# normalize naming
regime_hist["regime_name"] = regime_hist["regime_name"].replace({
    "Recession": "contraction"
})

# map to numeric
regime_map = {
    "Expansion": 0,
    "Slowdown": 1,
    "contraction": 2
}

regime_hist["regime_num"] = regime_hist["regime_name"].map(regime_map)

# sort properly
regime_hist = regime_hist.sort_values("audit_ts")

# remove duplicates (important)
regime_hist = regime_hist.drop_duplicates(subset=["audit_ts"])

# plot
plt.figure(figsize=(10,4))

plt.step(
    regime_hist["audit_ts"],
    regime_hist["regime_num"],
    where="post"
)

# y-axis labels
plt.yticks(
    [0,1,2],
    ["Expansion","Slowdown","Contraction"]
)

plt.title("Regime Timeline (Live Only)")
plt.xticks(rotation=45)

plt.ylim(-0.2, 2.2)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 10 — Trade History

# COMMAND ----------

trades = spark.sql(f"""
SELECT *
FROM {TRADE_TABLE}
ORDER BY trade_ts DESC
LIMIT 20
""").toPandas()

trades

# COMMAND ----------

# MAGIC %md
# MAGIC CELL 11 — Equity Curve

# COMMAND ----------

equity = spark.sql(f"""
SELECT audit_ts, account_equity
FROM {AUDIT_TABLE}
ORDER BY audit_ts
""").toPandas()

import pandas as pd

# ─────────────────────────────────────────────
# STEP 1: Clean types
# ─────────────────────────────────────────────
equity["audit_ts"] = pd.to_datetime(equity["audit_ts"])
equity["account_equity"] = pd.to_numeric(equity["account_equity"], errors="coerce")

# ─────────────────────────────────────────────
# STEP 2: Reset baseline (Mar 17)
# ─────────────────────────────────────────────
cutoff_date = pd.Timestamp("2026-03-17")
equity_live = equity[equity["audit_ts"] >= cutoff_date].copy()

# ─────────────────────────────────────────────
# STEP 3: Recalibration 
# ─────────────────────────────────────────────
equity_live = equity_live[
    (equity_live["account_equity"] > 1000) & 
    (equity_live["account_equity"] < 20000)
]

# ─────────────────────────────────────────────
# STEP 4: SORT + DEDUP (STABILITY FIX)
# ─────────────────────────────────────────────
equity_live = equity_live.sort_values("audit_ts")
equity_live = equity_live.drop_duplicates(subset=["audit_ts"])

# ─────────────────────────────────────────────
# DEBUG PRINTS
# ─────────────────────────────────────────────
print(f"Using equity from {cutoff_date.date()} onwards")
print(f"Rows retained: {len(equity_live)}")
print(f"Min equity: {equity_live['account_equity'].min():.2f}")
print(f"Max equity: {equity_live['account_equity'].max():.2f}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
plt.figure(figsize=(10,4))

plt.plot(equity_live["audit_ts"], equity_live["account_equity"])

plt.title("Strategy Equity (Live Only)")
plt.xticks(rotation=45)

plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Decision Boundary (FAST VECTORIZED)
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# load model
model = joblib.load("/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_xgboost_regime.joblib")

# base row
base = spark.sql(f"""
SELECT *
FROM {SILVER_RT}
ORDER BY timestamp DESC
LIMIT 1
""").toPandas().iloc[0]

# grid
emp_range = np.linspace(-4, 2, 60)
inf_range = np.linspace(-2, 3, 60)

grid = []

for emp in emp_range:
    for inf in inf_range:
        row = base.copy()
        row["PAYEMS_z"] = emp
        row["CPIAUCSL_z"] = inf
        grid.append(row)

grid_df = pd.DataFrame(grid)

X = grid_df[[
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
    "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
    "VIX_Inversion_Z", "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active"
]]

# single batch prediction (FAST)
proba = model.predict_proba(X)

# reshape
Z = proba[:, 2].reshape(len(emp_range), len(inf_range))

# plot
plt.figure(figsize=(8,6))

plt.contourf(inf_range, emp_range, Z, levels=20)

plt.colorbar(label="P(Contraction)")

plt.xlabel("Inflation (CPIAUCSL_z)")
plt.ylabel("Employment (PAYEMS_z)")

plt.title("Regime Decision Surface (Contraction Probability)")

# current point
plt.scatter(
    base["CPIAUCSL_z"],
    base["PAYEMS_z"],
    color="red",
    s=100,
    label="Current State"
)

plt.legend()
plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Decision Surface — VIX + Term Structure
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# load model
model = joblib.load("/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_xgboost_regime.joblib")

# base row
base = spark.sql(f"""
SELECT *
FROM {SILVER_RT}
ORDER BY timestamp DESC
LIMIT 1
""").toPandas().iloc[0]

# grid ranges
vix_range = np.linspace(-2, 3, 50)       # VIXCLS_z
term_range = np.linspace(-2, 3, 50)      # vix3m_z

grid = []

for vix in vix_range:
    for term in term_range:
        row = base.copy()
        row["VIXCLS_z"] = vix
        row["vix3m_z"] = term
        grid.append(row)

grid_df = pd.DataFrame(grid)

X = grid_df[[
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
    "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
    "VIX_Inversion_Z", "Price_Mom_24H", "CPI_Event_Active", "NFP_Event_Active"
]]

# predict
proba = model.predict_proba(X)

Z = proba[:, 2].reshape(len(vix_range), len(term_range))

# plot
plt.figure(figsize=(8,6))

plt.contourf(term_range, vix_range, Z, levels=20)

plt.colorbar(label="P(Contraction)")

plt.xlabel("VIX Term Structure (vix3m_z)")
plt.ylabel("VIX Level (VIXCLS_z)")

plt.title("Decision Surface — Volatility Regime")

# mark current point
plt.scatter(
    base["vix3m_z"],
    base["VIXCLS_z"],
    color="red",
    s=100,
    label="Current State"
)

plt.legend()
plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Feature list (same as RT_04)
# ─────────────────────────────────────────────

FEATURES = [
    "CPIAUCSL_z", "PCEPI_z", "PCE_z", "PAYEMS_z", "UNRATE_z",
    "GDPC1_z", "INDPRO_z", "UMCSENT_z", "VIXCLS_z", "vix3m_z",
    "VIX_Inversion_Z", "Price_Mom_24H",
    "CPI_Event_Active", "NFP_Event_Active"
]

# COMMAND ----------

!pip install xgboost

# COMMAND ----------

# ─────────────────────────────────────────────
# Decision Surface — Employment vs Growth
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

model = joblib.load("/Volumes/pawan_quant_research_1/v4/market_data/bt_fred_1d_xgboost_regime.joblib")

base = spark.sql(f"""
SELECT *
FROM {SILVER_RT}
ORDER BY timestamp DESC
LIMIT 1
""").toPandas().iloc[0]

emp_range = np.linspace(-4, 2, 50)
growth_range = np.linspace(-2, 3, 50)

grid = []
for emp in emp_range:
    for g in growth_range:
        row = base.copy()
        row["PAYEMS_z"] = emp
        row["GDPC1_z"] = g
        grid.append(row)

grid_df = pd.DataFrame(grid)

X = grid_df[FEATURES]

proba = model.predict_proba(X)
Z = proba[:, 2].reshape(len(emp_range), len(growth_range))

plt.figure(figsize=(7,5))
plt.contourf(growth_range, emp_range, Z, levels=20)
plt.colorbar(label="P(Contraction)")

plt.xlabel("Growth (GDPC1_z)")
plt.ylabel("Employment (PAYEMS_z)")
plt.title("Employment vs Growth")

plt.scatter(base["GDPC1_z"], base["PAYEMS_z"], color="red", s=80)
plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Decision Surface — Inflation vs Unemployment
# ─────────────────────────────────────────────

inf_range = np.linspace(-2, 3, 50)
unemp_range = np.linspace(-2, 3, 50)

grid = []
for inf in inf_range:
    for un in unemp_range:
        row = base.copy()
        row["CPIAUCSL_z"] = inf
        row["UNRATE_z"] = un
        grid.append(row)

grid_df = pd.DataFrame(grid)

X = grid_df[FEATURES]

proba = model.predict_proba(X)
Z = proba[:, 2].reshape(len(inf_range), len(unemp_range))

plt.figure(figsize=(7,5))
plt.contourf(unemp_range, inf_range, Z, levels=20)
plt.colorbar(label="P(Contraction)")

plt.xlabel("Unemployment (UNRATE_z)")
plt.ylabel("Inflation (CPIAUCSL_z)")
plt.title("Inflation vs Unemployment")

plt.scatter(base["UNRATE_z"], base["CPIAUCSL_z"], color="red", s=80)
plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Decision Surface — Momentum vs VIX
# ─────────────────────────────────────────────

mom_range = np.linspace(-0.1, 0.1, 50)
vix_range = np.linspace(-2, 3, 50)

grid = []
for m in mom_range:
    for v in vix_range:
        row = base.copy()
        row["Price_Mom_24H"] = m
        row["VIXCLS_z"] = v
        grid.append(row)

grid_df = pd.DataFrame(grid)

X = grid_df[FEATURES]

proba = model.predict_proba(X)
Z = proba[:, 2].reshape(len(mom_range), len(vix_range))

plt.figure(figsize=(7,5))
plt.contourf(vix_range, mom_range, Z, levels=20)
plt.colorbar(label="P(Contraction)")

plt.xlabel("VIX (VIXCLS_z)")
plt.ylabel("Momentum (Price_Mom_24H)")
plt.title("Momentum vs VIX")

plt.scatter(base["VIXCLS_z"], base["Price_Mom_24H"], color="red", s=80)
plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Decision Surface — VIX Inversion vs Momentum
# ─────────────────────────────────────────────

inv_range = np.linspace(-1, 1, 50)
mom_range = np.linspace(-0.1, 0.1, 50)

grid = []
for inv in inv_range:
    for m in mom_range:
        row = base.copy()
        row["VIX_Inversion_Z"] = inv
        row["Price_Mom_24H"] = m
        grid.append(row)

grid_df = pd.DataFrame(grid)

X = grid_df[FEATURES]

proba = model.predict_proba(X)
Z = proba[:, 2].reshape(len(inv_range), len(mom_range))

plt.figure(figsize=(7,5))
plt.contourf(mom_range, inv_range, Z, levels=20)
plt.colorbar(label="P(Contraction)")

plt.xlabel("Momentum (Price_Mom_24H)")
plt.ylabel("VIX Inversion (VIX_Inversion_Z)")
plt.title("VIX Inversion vs Momentum")

plt.scatter(base["Price_Mom_24H"], base["VIX_Inversion_Z"], color="red", s=80)
plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# Enable interactive backend (databricks-safe)
# ─────────────────────────────────────────────
%matplotlib notebook

# ─────────────────────────────────────────────
# 3D interactive plot
# ─────────────────────────────────────────────

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(mom_range, inv_range)

# surface
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)

# current point
current_x = base["Price_Mom_24H"]
current_y = base["VIX_Inversion_Z"]

X_current = base[FEATURES].values.reshape(1, -1)
current_z = model.predict_proba(X_current)[0][2]

ax.scatter(
    current_x,
    current_y,
    current_z,
    color="red",
    s=120,
    edgecolor="black",
    label="Current State"
)

# vertical guide
ax.plot(
    [current_x, current_x],
    [current_y, current_y],
    [0, current_z],
    color="red",
    linestyle="--"
)

ax.set_xlabel("Momentum")
ax.set_ylabel("VIX Inversion")
ax.set_zlabel("P(Contraction)")

ax.legend()

plt.show()

# COMMAND ----------

# ─────────────────────────────────────────────
# FIXED animation (databricks compatible)
# ─────────────────────────────────────────────

import matplotlib.animation as animation
from IPython.display import HTML, display

def rotate(angle):
    ax.view_init(elev=25, azim=angle)

ani = animation.FuncAnimation(
    fig,
    rotate,
    frames=np.arange(0, 360, 50),
    interval=100
)

# IMPORTANT: keep reference + explicit display
html_anim = ani.to_jshtml()

display(HTML(html_anim))
# Databricks notebook source
# MAGIC %md
# MAGIC # Saxo Live Monitoring Dashboard
# MAGIC
# MAGIC Read-only notebook for monitoring the live Saxo account, current regime state, feature behavior, and performance versus SPY.
# MAGIC
# MAGIC Scope:
# MAGIC - No writes to live jobs or production tables
# MAGIC - Uses existing live Saxo signal, audit, trade, and market tables
# MAGIC - Computes probabilities, monitoring tables, and charts on the fly

# COMMAND ----------

%pip install xgboost joblib seaborn

# COMMAND ----------

import json
from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", "{:,.4f}".format)
sns.set_theme(style="whitegrid")

try:
    display
except NameError:
    def display(obj):
        print(obj)

try:
    dbutils.widgets.text("lookback_days", "14")
    dbutils.widgets.text("feature_plot_days", "30")
    dbutils.widgets.text("initial_balance_override", "")
except Exception:
    pass


def get_widget_text(name, default_value):
    try:
        value = dbutils.widgets.get(name)
        return value if value is not None else default_value
    except Exception:
        return default_value


LOOKBACK_DAYS = int(get_widget_text("lookback_days", "14") or "14")
FEATURE_PLOT_DAYS = int(get_widget_text("feature_plot_days", "30") or "30")
INITIAL_BALANCE_OVERRIDE = get_widget_text("initial_balance_override", "").strip()
INITIAL_BALANCE_OVERRIDE = (
    float(INITIAL_BALANCE_OVERRIDE) if INITIAL_BALANCE_OVERRIDE else None
)

CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"

SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_regime_signal"
SILVER_RT_TABLE = f"{CATALOG}.{SCHEMA}.silver_rt_1h"
TRADE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_trade_log"
AUDIT_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_saxo_signal_audit_log"
MARKET_TABLE = f"{CATALOG}.{SCHEMA}.bronze_rt_market_1h"
MODEL_REGISTRY_TABLE = f"{CATALOG}.{SCHEMA}.model_registry_current"
LIVE_MODEL_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/models/bt_fred_1d_xgboost_regime_live.joblib"

FEATURES = [
    "CPIAUCSL_z",
    "PCEPI_z",
    "PCE_z",
    "PAYEMS_z",
    "UNRATE_z",
    "GDPC1_z",
    "INDPRO_z",
    "UMCSENT_z",
    "VIXCLS_z",
    "vix3m_z",
    "VIX_Inversion_Z",
    "Price_Mom_24H",
    "CPI_Event_Active",
    "NFP_Event_Active",
]

FEATURE_LABELS = {
    "CPIAUCSL_z": "CPI",
    "PCEPI_z": "Core PCE",
    "PCE_z": "PCE",
    "PAYEMS_z": "Payrolls",
    "UNRATE_z": "Unemployment",
    "GDPC1_z": "GDP",
    "INDPRO_z": "Industrial Production",
    "UMCSENT_z": "Sentiment",
    "VIXCLS_z": "VIX Spot",
    "vix3m_z": "VIX 3M",
    "VIX_Inversion_Z": "VIX Inversion",
    "Price_Mom_24H": "24H Price Momentum",
    "CPI_Event_Active": "CPI Event",
    "NFP_Event_Active": "NFP Event",
}

REGIME_NAME_MAP = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
PROBABILITY_COLUMNS = {
    0: "prob_expansion",
    1: "prob_slowdown",
    2: "prob_recession",
}


def runtime_config_table(catalog=CATALOG, schema=SCHEMA):
    return f"{catalog}.{schema}.rt_runtime_instrument_config"


def target_position_label(ticker):
    return f"100% {ticker}" if ticker else "100% CASH"


def ensure_runtime_instrument_config_table(spark, catalog=CATALOG, schema=SCHEMA):
    table_name = runtime_config_table(catalog, schema)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            config_name STRING,
            market_symbols_csv STRING,
            primary_symbol STRING,
            secondary_symbol STRING,
            regime_0_ticker STRING,
            regime_1_ticker STRING,
            regime_2_ticker STRING,
            data_source STRING,
            trading_source STRING,
            notes STRING,
            updated_ts TIMESTAMP,
            updated_by STRING
        )
        USING DELTA
        """
    )

    row_count = spark.sql(f"SELECT COUNT(*) AS n FROM {table_name}").collect()[0]["n"]
    if row_count == 0:
        spark.sql(
            f"""
            INSERT INTO {table_name}
            VALUES (
                'default_live',
                'SPY,QQQ',
                'SPY',
                'QQQ',
                'QQQ',
                'SPY',
                NULL,
                'alpaca_stocks',
                'alpaca_paper',
                'Default live runtime mapping for the V4 hourly chain',
                current_timestamp(),
                current_user()
            )
            """
        )

    return table_name


def load_runtime_instrument_config(spark, catalog=CATALOG, schema=SCHEMA):
    table_name = ensure_runtime_instrument_config_table(spark, catalog, schema)
    row = spark.sql(
        f"""
        SELECT *
        FROM {table_name}
        ORDER BY updated_ts DESC
        LIMIT 1
        """
    ).collect()[0]

    market_symbols = [
        symbol.strip()
        for symbol in (row["market_symbols_csv"] or "").split(",")
        if symbol.strip()
    ]
    primary_symbol = row["primary_symbol"] or (market_symbols[0] if market_symbols else None)
    secondary_symbol = row["secondary_symbol"] or (
        market_symbols[1] if len(market_symbols) > 1 else primary_symbol
    )
    regime_ticker = {
        0: row["regime_0_ticker"],
        1: row["regime_1_ticker"],
        2: row["regime_2_ticker"] or None,
    }

    return {
        "table_name": table_name,
        "config_name": row["config_name"],
        "market_symbols": market_symbols,
        "primary_symbol": primary_symbol,
        "secondary_symbol": secondary_symbol,
        "regime_ticker": regime_ticker,
        "position_map": {
            regime: target_position_label(ticker)
            for regime, ticker in regime_ticker.items()
        },
        "data_source": row["data_source"],
        "trading_source": row["trading_source"],
        "notes": row["notes"],
        "updated_ts": row["updated_ts"],
        "updated_by": row["updated_by"],
    }


runtime_config = load_runtime_instrument_config(spark, CATALOG, SCHEMA)
PRIMARY_SYMBOL = runtime_config["primary_symbol"]
SECONDARY_SYMBOL = runtime_config["secondary_symbol"]


def table_exists(table_name):
    return spark.catalog.tableExists(table_name)


def read_table(table_name, query):
    if not table_exists(table_name):
        print(f"Table missing: {table_name}")
        return pd.DataFrame()
    return spark.sql(query).toPandas()


def to_datetime_cols(frame, columns):
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.tz_localize(None)
    return frame


def safe_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_json_loads(payload):
    payload = scalar_value(payload, default=None)
    if isinstance(payload, (list, dict)):
        return payload
    if payload is None or payload == "" or pd.isna(payload):
        return []
    try:
        return json.loads(payload)
    except Exception:
        return []


def get_series(frame, column_name, default_value=np.nan):
    if column_name not in frame.columns:
        return pd.Series(default_value, index=frame.index)
    values = frame.loc[:, column_name]
    if isinstance(values, pd.DataFrame):
        values = values.bfill(axis=1).iloc[:, 0]
    return values


def scalar_value(value, default=np.nan):
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return default
        flattened = value.bfill(axis=1).iloc[:, 0]
        return scalar_value(flattened, default=default)
    if isinstance(value, pd.Series):
        non_na = value.dropna()
        return non_na.iloc[0] if not non_na.empty else default
    return value


def latest_model_metadata():
    if table_exists(MODEL_REGISTRY_TABLE):
        row = spark.sql(
            f"""
            SELECT model_version, model_path
            FROM {MODEL_REGISTRY_TABLE}
            ORDER BY promoted_ts DESC
            LIMIT 1
            """
        ).collect()
        if row:
            return row[0]["model_version"], row[0]["model_path"]
    return "bootstrap_live_model", LIVE_MODEL_PATH


def load_live_model():
    model_version, model_path = latest_model_metadata()
    try:
        model = joblib.load(model_path)
        print(f"Model loaded: {model_version}")
        return model, model_version, model_path
    except Exception as exc:
        print(f"Could not load model at {model_path}: {exc}")
        return None, model_version, model_path


def feature_importance_table(model):
    if model is None:
        return pd.DataFrame({"feature": FEATURES, "importance": np.ones(len(FEATURES)) / len(FEATURES)})

    if hasattr(model, "feature_importances_"):
        importance = np.array(model.feature_importances_, dtype=float)
    else:
        importance = np.ones(len(FEATURES), dtype=float)

    if importance.sum() == 0:
        importance = np.ones(len(FEATURES), dtype=float)

    feature_df = pd.DataFrame({"feature": FEATURES, "importance": importance})
    feature_df["importance_pct"] = feature_df["importance"] / feature_df["importance"].sum()
    return feature_df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_probabilities(frame, model):
    probability_df = pd.DataFrame(index=frame.index)
    for column in PROBABILITY_COLUMNS.values():
        probability_df[column] = np.nan

    if model is None or not hasattr(model, "predict_proba") or frame.empty:
        return probability_df

    try:
        probs = model.predict_proba(frame[FEATURES].fillna(0.0))
        classes = list(getattr(model, "classes_", [0, 1, 2]))
        for idx, class_id in enumerate(classes):
            column_name = PROBABILITY_COLUMNS.get(int(class_id))
            if column_name:
                probability_df[column_name] = probs[:, idx]
    except Exception as exc:
        print(f"Probability computation unavailable: {exc}")

    return probability_df


def infer_bars_per_year(timestamps):
    ts = pd.Series(pd.to_datetime(timestamps, errors="coerce")).dropna().sort_values()
    if ts.empty:
        return 252 * 7

    daily_counts = ts.dt.floor("D").value_counts()
    bars_per_day = daily_counts.median() if not daily_counts.empty else 7
    return int(max(1, round(float(bars_per_day))) * 252)


def compute_metrics(value_series, timestamps):
    clean = pd.DataFrame({"timestamp": timestamps, "value": value_series}).dropna().sort_values("timestamp")
    if clean.shape[0] < 2:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "std_dev": np.nan,
            "max_drawdown": np.nan,
        }

    returns = clean["value"].pct_change().dropna()
    bars_per_year = infer_bars_per_year(clean["timestamp"])
    years = (clean["timestamp"].iloc[-1] - clean["timestamp"].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    downside = returns[returns < 0]
    rolling_max = clean["value"].cummax()
    drawdown = clean["value"] / rolling_max - 1.0
    sigma = returns.std()
    downside_sigma = downside.std()
    total_multiplier = clean["value"].iloc[-1] / clean["value"].iloc[0]

    return {
        "total_return": total_multiplier - 1.0,
        "cagr": total_multiplier,
        "sharpe": returns.mean() / sigma * np.sqrt(bars_per_year) if pd.notna(sigma) and sigma > 0 else np.nan,
        "sortino": returns.mean() / downside_sigma * np.sqrt(bars_per_year) if pd.notna(downside_sigma) and downside_sigma > 0 else np.nan,
        "std_dev": sigma * np.sqrt(bars_per_year) if pd.notna(sigma) else np.nan,
        "max_drawdown": drawdown.min(),
        "years": years,
    }


def finalize_cagr(metrics_dict):
    if pd.isna(metrics_dict.get("cagr")):
        return metrics_dict
    years = metrics_dict.get("years")
    total_multiplier = metrics_dict["cagr"]
    metrics_dict["cagr"] = (
        np.nan
        if years is None or pd.isna(years) or years <= 0 or pd.isna(total_multiplier) or total_multiplier <= 0
        else total_multiplier ** (1 / years) - 1.0
    )
    return metrics_dict


def direction_label(value):
    if pd.isna(value):
        return "flat"
    return "up" if value > 0 else "down" if value < 0 else "flat"


def feature_driver_text(current_row, previous_row, importance_map):
    if previous_row is None:
        return "Initial observation"

    scored = []
    for feature in FEATURES:
        cur_value = safe_float(scalar_value(current_row.get(feature)))
        prev_value = safe_float(scalar_value(previous_row.get(feature)))
        if np.isnan(cur_value) and np.isnan(prev_value):
            continue
        delta = cur_value - prev_value
        score = abs(delta) * float(importance_map.get(feature, 1.0))
        scored.append((feature, score, delta, cur_value))

    if not scored:
        return "No feature delta available"

    top_drivers = sorted(scored, key=lambda row: row[1], reverse=True)[:3]
    fragments = []
    for feature, _, delta, cur_value in top_drivers:
        label = FEATURE_LABELS.get(feature, feature)
        fragments.append(f"{label} {direction_label(delta)} to {cur_value:.2f} (delta {delta:+.2f})")
    return "; ".join(fragments)


def build_monitoring_summary(latest_row, perf_lookback_row, latest_positions, latest_switch):
    probability_pairs = []
    for column in PROBABILITY_COLUMNS.values():
        value = scalar_value(latest_row.get(column))
        if pd.notna(value):
            probability_pairs.append((column, value))
    probability_pairs = sorted(probability_pairs, key=lambda row: row[1], reverse=True)
    top_prob_text = (
        f"{probability_pairs[0][0].replace('prob_', '').replace('_', ' ')} {probability_pairs[0][1]:.1%}"
        if probability_pairs
        else "probability unavailable"
    )

    latest_regime = f"{scalar_value(latest_row['regime_name'])} ({int(scalar_value(latest_row['regime_int']))})"
    equity = scalar_value(latest_row.get("account_equity"))
    balance = scalar_value(latest_row.get("account_balance"))
    two_week_return = scalar_value(perf_lookback_row.get("strategy_return_pct"))
    benchmark_return = scalar_value(perf_lookback_row.get("benchmark_return_pct"))
    open_positions = latest_positions.shape[0]
    switch_text = latest_switch["drivers"] if latest_switch is not None else "No completed switch explanation yet"
    balance_text = f"{balance:,.2f}" if pd.notna(balance) else "N/A"
    equity_text = f"{equity:,.2f}" if pd.notna(equity) else "N/A"
    two_week_text = f"{two_week_return:+.2%}" if pd.notna(two_week_return) else "N/A"
    benchmark_text = f"{benchmark_return:+.2%}" if pd.notna(benchmark_return) else "N/A"

    return (
        f"Latest regime is {latest_regime} with {top_prob_text}. "
        f"Account balance is {balance_text} and equity is {equity_text}. "
        f"Over the last {LOOKBACK_DAYS} days, Saxo returned {two_week_text} versus SPY {benchmark_text}. "
        f"Open positions: {open_positions}. "
        f"Latest switch context: {switch_text}."
    )


signals_df = read_table(
    SIGNAL_TABLE,
    f"""
    SELECT *
    FROM {SIGNAL_TABLE}
    ORDER BY timestamp
    """,
)
signals_df = to_datetime_cols(signals_df, ["timestamp", "ingestion_ts"])

silver_df = read_table(
    SILVER_RT_TABLE,
    f"""
    SELECT timestamp, {", ".join(FEATURES)}
    FROM {SILVER_RT_TABLE}
    ORDER BY timestamp
    """,
)
silver_df = to_datetime_cols(silver_df, ["timestamp"])

audit_df = read_table(
    AUDIT_TABLE,
    f"""
    SELECT *
    FROM {AUDIT_TABLE}
    ORDER BY signal_ts
    """,
)
audit_df = to_datetime_cols(audit_df, ["audit_ts", "trade_ts", "signal_ts", "ingestion_ts"])

trade_df = read_table(
    TRADE_TABLE,
    f"""
    SELECT *
    FROM {TRADE_TABLE}
    ORDER BY signal_ts
    """,
)
trade_df = to_datetime_cols(trade_df, ["trade_ts", "signal_ts", "ingestion_ts"])

market_df = read_table(
    MARKET_TABLE,
    f"""
    SELECT timestamp, symbol, close
    FROM {MARKET_TABLE}
    WHERE symbol IN ('{PRIMARY_SYMBOL}', '{SECONDARY_SYMBOL}')
    ORDER BY timestamp
    """,
)
market_df = to_datetime_cols(market_df, ["timestamp"])

if signals_df.empty and audit_df.empty:
    raise ValueError("No Saxo live signal or audit history found. Run the live chain first.")

signals_df = (
    signals_df.sort_values("timestamp")
    .drop_duplicates(subset=["timestamp"], keep="last")
    .reset_index(drop=True)
)
signals_df["regime_name"] = signals_df["regime_name"].fillna(signals_df["regime_int"].map(REGIME_NAME_MAP))
signals_df["position"] = signals_df["position"].fillna(signals_df["regime_int"].map(runtime_config["position_map"]))

if silver_df.empty:
    silver_df = signals_df[["timestamp"] + [feature for feature in FEATURES if feature in signals_df.columns]].copy()

if not market_df.empty:
    market_prices = (
        market_df.sort_values(["timestamp", "symbol"])
        .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
        .reset_index()
    )
else:
    market_prices = pd.DataFrame(columns=["timestamp", PRIMARY_SYMBOL, SECONDARY_SYMBOL])

audit_enriched = audit_df.rename(columns={"signal_ts": "timestamp"}) if not audit_df.empty else pd.DataFrame()
trade_enriched = trade_df.rename(columns={"signal_ts": "timestamp"}) if not trade_df.empty else pd.DataFrame()

main_df = signals_df.copy()

if not audit_enriched.empty:
    audit_cols = [
        "timestamp",
        "audit_ts",
        "trade_ts",
        "target_ticker",
        "switched",
        "orders_placed",
        "account_balance",
        "account_equity",
        "account_cash",
        "positions_count",
        "positions_market_value",
        "positions_unrealized_pl",
        "positions_unrealized_plpc",
        "hourly_pnl",
        "hourly_pnl_pct",
        "day_pnl",
        "day_pnl_pct",
        "execution_mode",
        "vix_source_date",
        "pit_lag_days",
        "bias_gate_passed",
    ]
    audit_cols = [column for column in audit_cols if column in audit_enriched.columns]
    main_df = main_df.merge(audit_enriched[audit_cols], on="timestamp", how="left")

if not trade_enriched.empty:
    trade_cols = [
        "timestamp",
        "trade_ts",
        "target_position",
        "target_ticker",
        "orders_placed",
        "last_equity",
        "positions_json",
        "transaction_cost",
        "source",
    ]
    trade_cols = [column for column in trade_cols if column in trade_enriched.columns]
    main_df = main_df.merge(trade_enriched[trade_cols], on="timestamp", how="left", suffixes=("", "_trade"))

if not market_prices.empty:
    main_df = main_df.merge(market_prices, on="timestamp", how="left")

if "SPY_close" not in main_df.columns:
    main_df["SPY_close"] = np.nan
if "QQQ_close" not in main_df.columns:
    main_df["QQQ_close"] = np.nan

if PRIMARY_SYMBOL in main_df.columns:
    main_df["SPY_close"] = get_series(main_df, "SPY_close").fillna(get_series(main_df, PRIMARY_SYMBOL))
if SECONDARY_SYMBOL in main_df.columns:
    main_df["QQQ_close"] = get_series(main_df, "QQQ_close").fillna(get_series(main_df, SECONDARY_SYMBOL))
main_df["target_position"] = get_series(main_df, "target_position").fillna(get_series(main_df, "position"))
main_df["as_of_ts"] = get_series(main_df, "trade_ts").fillna(get_series(main_df, "timestamp"))
main_df = main_df.sort_values("as_of_ts").reset_index(drop=True)

model, live_model_version, live_model_path = load_live_model()
importance_df = feature_importance_table(model)
importance_map = dict(zip(importance_df["feature"], importance_df["importance_pct"].fillna(0.0)))

probability_df = compute_probabilities(main_df, model)
main_df = pd.concat([main_df.reset_index(drop=True), probability_df.reset_index(drop=True)], axis=1)

current_prob_cols = list(dict.fromkeys([column for column in PROBABILITY_COLUMNS.values() if column in main_df.columns]))
probability_frame = (
    pd.DataFrame({column: get_series(main_df, column) for column in current_prob_cols}, index=main_df.index)
    if current_prob_cols
    else pd.DataFrame(index=main_df.index)
)
main_df["top_probability"] = probability_frame.max(axis=1) if not probability_frame.empty else np.nan
main_df["top_probability_label"] = (
    probability_frame.idxmax(axis=1).str.replace("prob_", "").str.replace("_", " ")
    if not probability_frame.empty
    else np.nan
)

equity_df = main_df[main_df["account_equity"].notna() & main_df["SPY_close"].notna()].copy()
if equity_df.empty:
    raise ValueError("No account equity history found in Saxo trade log.")

starting_equity = INITIAL_BALANCE_OVERRIDE if INITIAL_BALANCE_OVERRIDE is not None else float(scalar_value(equity_df["account_equity"].iloc[0]))
starting_spy_price = float(scalar_value(equity_df["SPY_close"].iloc[0]))
equity_df["strategy_value"] = equity_df["account_equity"].astype(float)
equity_df["spy_benchmark"] = starting_equity * equity_df["SPY_close"].astype(float) / starting_spy_price
equity_df["strategy_return_pct"] = equity_df["strategy_value"] / equity_df["strategy_value"].iloc[0] - 1.0
equity_df["benchmark_return_pct"] = equity_df["spy_benchmark"] / equity_df["spy_benchmark"].iloc[0] - 1.0
equity_df["alpha_pct"] = equity_df["strategy_return_pct"] - equity_df["benchmark_return_pct"]
equity_df["strategy_drawdown_pct"] = equity_df["strategy_value"] / equity_df["strategy_value"].cummax() - 1.0
equity_df["benchmark_drawdown_pct"] = equity_df["spy_benchmark"] / equity_df["spy_benchmark"].cummax() - 1.0

recent_cutoff = equity_df["as_of_ts"].max() - timedelta(days=LOOKBACK_DAYS)
recent_perf_df = equity_df[equity_df["as_of_ts"] >= recent_cutoff].copy()
if recent_perf_df.shape[0] >= 2:
    recent_starting_equity = float(recent_perf_df["strategy_value"].iloc[0])
    recent_starting_spy = float(recent_perf_df["SPY_close"].iloc[0])
    recent_perf_df["recent_spy_benchmark"] = recent_starting_equity * recent_perf_df["SPY_close"].astype(float) / recent_starting_spy
    recent_perf_df["strategy_return_pct"] = recent_perf_df["strategy_value"] / recent_perf_df["strategy_value"].iloc[0] - 1.0
    recent_perf_df["benchmark_return_pct"] = recent_perf_df["recent_spy_benchmark"] / recent_perf_df["recent_spy_benchmark"].iloc[0] - 1.0
else:
    recent_perf_df["recent_spy_benchmark"] = np.nan

strategy_metrics = finalize_cagr(compute_metrics(equity_df["strategy_value"], equity_df["as_of_ts"]))
benchmark_metrics = finalize_cagr(compute_metrics(equity_df["spy_benchmark"], equity_df["as_of_ts"]))
recent_strategy_metrics = finalize_cagr(compute_metrics(recent_perf_df["strategy_value"], recent_perf_df["as_of_ts"]))
recent_benchmark_metrics = finalize_cagr(compute_metrics(recent_perf_df["recent_spy_benchmark"], recent_perf_df["as_of_ts"]))

latest_row = main_df.iloc[-1]
latest_equity_row = equity_df.iloc[-1]
previous_row = main_df.iloc[-2] if main_df.shape[0] > 1 else None

feature_snapshot_rows = []
for feature in FEATURES:
    cur_value = safe_float(scalar_value(latest_row.get(feature)))
    prev_value = safe_float(scalar_value(previous_row.get(feature))) if previous_row is not None else np.nan
    delta_value = cur_value - prev_value if previous_row is not None else np.nan
    feature_snapshot_rows.append(
        {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "current_value": cur_value,
            "previous_value": prev_value,
            "delta": delta_value,
            "abs_current": abs(cur_value) if not np.isnan(cur_value) else np.nan,
            "importance_pct": importance_map.get(feature, np.nan),
            "weighted_move_score": abs(delta_value) * importance_map.get(feature, 1.0) if not np.isnan(delta_value) else np.nan,
        }
    )

feature_snapshot_df = pd.DataFrame(feature_snapshot_rows).sort_values(
    ["weighted_move_score", "abs_current"], ascending=[False, False]
)

regime_summary_df = (
    main_df.groupby(["regime_int", "regime_name"])
    .agg(
        bars=("regime_int", "size"),
        avg_top_probability=("top_probability", "mean"),
        avg_hourly_pnl_pct=("hourly_pnl_pct", "mean"),
        switch_count=("switched", "sum"),
    )
    .reset_index()
    .sort_values("regime_int")
)
regime_summary_df["pct_of_history"] = regime_summary_df["bars"] / regime_summary_df["bars"].sum()

transition_df = main_df[["as_of_ts", "regime_name"]].copy()
transition_df["prev_regime_name"] = transition_df["regime_name"].shift(1)
transition_matrix_df = (
    transition_df.dropna()
    .pivot_table(index="prev_regime_name", columns="regime_name", values="as_of_ts", aggfunc="count", fill_value=0)
)

switch_base = main_df.copy()
switch_base["prev_regime"] = switch_base["regime_int"].shift(1)
switch_base["prev_regime_name"] = switch_base["regime_name"].shift(1)
switch_base["prev_position"] = switch_base["target_position"].shift(1)
switch_base["switch_detected"] = (switch_base["regime_int"] != switch_base["prev_regime"]) & switch_base["prev_regime"].notna()
switched_series = get_series(switch_base, "switched", False).astype("boolean").fillna(False).astype(bool)
switch_detected_series = get_series(switch_base, "switch_detected", False).astype(bool)
switch_rows = switch_base[switch_detected_series | switched_series].copy()

switch_records = []
for idx, row in switch_rows.iterrows():
    previous_switch_row = switch_base.iloc[idx - 1] if idx > 0 else None
    prior_switch_snapshot = switch_rows[switch_rows["as_of_ts"] < row["as_of_ts"]].tail(1)
    previous_switch_ts = scalar_value(prior_switch_snapshot["as_of_ts"], default=pd.NaT) if not prior_switch_snapshot.empty else pd.NaT
    previous_switch_equity = safe_float(scalar_value(prior_switch_snapshot["account_equity"])) if not prior_switch_snapshot.empty else np.nan
    previous_switch_benchmark = safe_float(scalar_value(prior_switch_snapshot["SPY_close"])) if not prior_switch_snapshot.empty else np.nan
    row_account_equity = safe_float(scalar_value(row.get("account_equity")))
    row_spy_close = safe_float(scalar_value(row.get("SPY_close")))
    strategy_since_last_switch = (
        row_account_equity / previous_switch_equity - 1.0
        if pd.notna(previous_switch_equity) and previous_switch_equity not in (0, np.nan)
        else np.nan
    )
    benchmark_since_last_switch = (
        row_spy_close / previous_switch_benchmark - 1.0
        if pd.notna(previous_switch_benchmark) and previous_switch_benchmark not in (0, np.nan)
        else np.nan
    )
    switch_records.append(
        {
            "switch_ts": row["as_of_ts"],
            "from_regime": scalar_value(row["prev_regime_name"]),
            "to_regime": scalar_value(row["regime_name"]),
            "from_position": scalar_value(row["prev_position"]),
            "to_position": scalar_value(row["target_position"]),
            "hours_since_last_switch": (
                (row["as_of_ts"] - previous_switch_ts).total_seconds() / 3600
                if pd.notna(previous_switch_ts)
                else np.nan
            ),
            "strategy_return_since_last_switch": strategy_since_last_switch,
            "spy_return_since_last_switch": benchmark_since_last_switch,
            "probability_on_switch": safe_float(scalar_value(row.get("top_probability"))),
            "drivers": feature_driver_text(row, previous_switch_row, importance_map),
        }
    )

switch_log_df = pd.DataFrame(switch_records).sort_values("switch_ts", ascending=False) if switch_records else pd.DataFrame()

trade_snapshot_df = trade_df.copy()
if not trade_snapshot_df.empty:
    trade_snapshot_df["orders_placed_list"] = trade_snapshot_df["orders_placed"].apply(safe_json_loads)
    trade_snapshot_df["positions_list"] = trade_snapshot_df["positions_json"].apply(safe_json_loads)

latest_positions_df = pd.DataFrame(safe_json_loads(latest_row.get("positions_json")))
latest_positions_df = latest_positions_df.sort_values("market_value", ascending=False) if not latest_positions_df.empty else latest_positions_df

feature_history_df = main_df[["as_of_ts", "regime_int", "regime_name"] + FEATURES + current_prob_cols].copy()
feature_history_df = feature_history_df.sort_values("as_of_ts").reset_index(drop=True)

feature_export_df = feature_history_df.copy()
feature_export_df = feature_export_df.rename(columns={"as_of_ts": "timestamp"})

recent_feature_cutoff = main_df["as_of_ts"].max() - timedelta(days=FEATURE_PLOT_DAYS)
recent_feature_df = main_df[main_df["as_of_ts"] >= recent_feature_cutoff].copy()

summary_kpis_df = pd.DataFrame(
    [
        {"metric": "Latest regime", "value": f"{scalar_value(latest_row['regime_name'])} ({int(scalar_value(latest_row['regime_int']))})"},
        {"metric": "Latest target position", "value": scalar_value(latest_row["target_position"])},
        {"metric": "Latest top probability", "value": f"{safe_float(scalar_value(latest_row['top_probability'])):.2%}" if pd.notna(safe_float(scalar_value(latest_row['top_probability']))) else "N/A"},
        {"metric": "Account balance", "value": f"{safe_float(scalar_value(latest_row['account_balance'])):,.2f}" if pd.notna(safe_float(scalar_value(latest_row['account_balance']))) else "N/A"},
        {"metric": "Account equity", "value": f"{safe_float(scalar_value(latest_row['account_equity'])):,.2f}" if pd.notna(safe_float(scalar_value(latest_row['account_equity']))) else "N/A"},
        {"metric": f"Saxo return ({LOOKBACK_DAYS}D)", "value": f"{recent_perf_df['strategy_return_pct'].iloc[-1]:+.2%}" if recent_perf_df.shape[0] >= 2 else "N/A"},
        {"metric": f"SPY return ({LOOKBACK_DAYS}D)", "value": f"{recent_perf_df['benchmark_return_pct'].iloc[-1]:+.2%}" if recent_perf_df.shape[0] >= 2 else "N/A"},
        {"metric": "Live alpha since inception", "value": f"{latest_equity_row['alpha_pct']:+.2%}"},
        {"metric": "Open positions", "value": int(scalar_value(latest_row["positions_count"])) if pd.notna(scalar_value(latest_row["positions_count"])) else 0},
        {"metric": "PIT lag days", "value": int(scalar_value(latest_row.get("pit_lag_days"))) if pd.notna(scalar_value(latest_row.get("pit_lag_days"))) else "N/A"},
        {"metric": "Bias gate passed", "value": bool(scalar_value(latest_row["bias_gate_passed"])) if pd.notna(scalar_value(latest_row.get("bias_gate_passed"))) else "N/A"},
        {"metric": "Model version", "value": scalar_value(latest_row.get("model_version", live_model_version), default=live_model_version)},
    ]
)

metrics_display_df = pd.DataFrame(
    [
        {
            "series": "Saxo strategy",
            "window": "Since inception",
            "total_return": strategy_metrics["total_return"],
            "cagr": strategy_metrics["cagr"],
            "sharpe": strategy_metrics["sharpe"],
            "sortino": strategy_metrics["sortino"],
            "std_dev": strategy_metrics["std_dev"],
            "max_drawdown": strategy_metrics["max_drawdown"],
        },
        {
            "series": "SPY benchmark",
            "window": "Since inception",
            "total_return": benchmark_metrics["total_return"],
            "cagr": benchmark_metrics["cagr"],
            "sharpe": benchmark_metrics["sharpe"],
            "sortino": benchmark_metrics["sortino"],
            "std_dev": benchmark_metrics["std_dev"],
            "max_drawdown": benchmark_metrics["max_drawdown"],
        },
        {
            "series": "Saxo strategy",
            "window": f"Last {LOOKBACK_DAYS} days",
            "total_return": recent_strategy_metrics["total_return"],
            "cagr": recent_strategy_metrics["cagr"],
            "sharpe": recent_strategy_metrics["sharpe"],
            "sortino": recent_strategy_metrics["sortino"],
            "std_dev": recent_strategy_metrics["std_dev"],
            "max_drawdown": recent_strategy_metrics["max_drawdown"],
        },
        {
            "series": "SPY benchmark",
            "window": f"Last {LOOKBACK_DAYS} days",
            "total_return": recent_benchmark_metrics["total_return"],
            "cagr": recent_benchmark_metrics["cagr"],
            "sharpe": recent_benchmark_metrics["sharpe"],
            "sortino": recent_benchmark_metrics["sortino"],
            "std_dev": recent_benchmark_metrics["std_dev"],
            "max_drawdown": recent_benchmark_metrics["max_drawdown"],
        },
    ]
)

latest_switch_row = switch_log_df.iloc[0].to_dict() if not switch_log_df.empty else None
monitoring_summary = build_monitoring_summary(
    latest_row=latest_row,
    perf_lookback_row=recent_perf_df.iloc[-1].to_dict() if recent_perf_df.shape[0] >= 2 else {"strategy_return_pct": np.nan, "benchmark_return_pct": np.nan},
    latest_positions=latest_positions_df,
    latest_switch=latest_switch_row,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Current Snapshot
# MAGIC Latest live state, account health, and model context.

# COMMAND ----------

display(summary_kpis_df)

print(monitoring_summary)
print(f"Model path: {live_model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance vs SPY
# MAGIC Live Saxo equity versus SPY using the same starting balance as the Saxo account.

# COMMAND ----------

display(
    metrics_display_df.style.format(
        {
            "total_return": "{:+.2%}",
            "cagr": "{:+.2%}",
            "sharpe": "{:.2f}",
            "sortino": "{:.2f}",
            "std_dev": "{:.2%}",
            "max_drawdown": "{:.2%}",
        }
    )
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(16, 9),
    gridspec_kw={"height_ratios": [4, 1]},
    sharex=True,
)

ax1.plot(equity_df["as_of_ts"], equity_df["strategy_value"], label="Saxo equity", linewidth=2)
ax1.plot(equity_df["as_of_ts"], equity_df["spy_benchmark"], label="SPY benchmark", linewidth=1.75, alpha=0.85)
ax1.set_title("Saxo Live Equity vs SPY Benchmark")
ax1.set_ylabel("Value")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.fill_between(equity_df["as_of_ts"], 0, 1, where=(equity_df["regime_int"] == 0), color="green", alpha=0.30, label="Expansion")
ax2.fill_between(equity_df["as_of_ts"], 0, 1, where=(equity_df["regime_int"] == 1), color="gold", alpha=0.35, label="Slowdown")
ax2.fill_between(equity_df["as_of_ts"], 0, 1, where=(equity_df["regime_int"] == 2), color="red", alpha=0.30, label="Recession")
ax2.set_ylabel("Regime")
ax2.set_yticks([])
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

if recent_perf_df.shape[0] >= 2:
    plt.figure(figsize=(16, 5))
    plt.plot(recent_perf_df["as_of_ts"], recent_perf_df["strategy_value"], label="Saxo equity", linewidth=2)
    plt.plot(recent_perf_df["as_of_ts"], recent_perf_df["recent_spy_benchmark"], label=f"SPY ({LOOKBACK_DAYS}D base)", linewidth=1.75, alpha=0.85)
    plt.title(f"Last {LOOKBACK_DAYS} Days: Saxo vs SPY")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regime and Probability Monitoring

# COMMAND ----------

display(
    regime_summary_df.style.format(
        {
            "avg_top_probability": "{:.2%}",
            "avg_hourly_pnl_pct": "{:+.2%}",
            "pct_of_history": "{:.2%}",
        }
    )
)

if not transition_matrix_df.empty:
    display(transition_matrix_df)

probability_plot_columns = [column for column in current_prob_cols if get_series(main_df, column).notna().any()]
if probability_plot_columns:
    plt.figure(figsize=(16, 5))
    for column in probability_plot_columns:
        plt.plot(
            main_df["as_of_ts"],
            get_series(main_df, column),
            label=column.replace("prob_", "").replace("_", " ").title(),
            linewidth=1.6,
        )
    plt.axvline(main_df["as_of_ts"].iloc[-1], color="black", linestyle=":", alpha=0.5)
    plt.title("Historical Regime Probabilities")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trade, Switch, and Position Monitoring
# MAGIC Historical switches, latest open positions, and hourly live snapshots.

# COMMAND ----------

if not switch_log_df.empty:
    display(
        switch_log_df.style.format(
            {
                "hours_since_last_switch": "{:.1f}",
                "strategy_return_since_last_switch": "{:+.2%}",
                "spy_return_since_last_switch": "{:+.2%}",
                "probability_on_switch": "{:.2%}",
            }
        )
    )
else:
    print("No completed switch history available yet.")

print("Latest live positions")
if latest_positions_df.empty:
    print("No open positions.")
else:
    display(latest_positions_df)

trade_monitor_columns = [
    "trade_ts",
    "signal_ts",
    "regime_name",
    "target_position",
    "switched",
    "account_balance",
    "account_equity",
    "hourly_pnl",
    "hourly_pnl_pct",
    "day_pnl",
    "day_pnl_pct",
    "positions_count",
]
trade_monitor_columns = [column for column in trade_monitor_columns if column in trade_df.columns]
if not trade_df.empty and trade_monitor_columns:
    display(
        trade_df[trade_monitor_columns]
        .sort_values("trade_ts", ascending=False)
        .head(100)
        .style.format(
            {
                "account_balance": "{:,.2f}",
                "account_equity": "{:,.2f}",
                "hourly_pnl": "{:+,.2f}",
                "hourly_pnl_pct": "{:+.2%}",
                "day_pnl": "{:+,.2f}",
                "day_pnl_pct": "{:+.2%}",
            }
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Tracking
# MAGIC Current feature state, recent history, and pair/3D views for monitoring regime context.

# COMMAND ----------

display(
    feature_snapshot_df.style.format(
        {
            "current_value": "{:+.3f}",
            "previous_value": "{:+.3f}",
            "delta": "{:+.3f}",
            "abs_current": "{:.3f}",
            "importance_pct": "{:.2%}",
            "weighted_move_score": "{:.4f}",
        }
    )
)

display(
    importance_df.style.format(
        {
            "importance": "{:.4f}",
            "importance_pct": "{:.2%}",
        }
    )
)

plot_features = [
    "VIX_Inversion_Z",
    "Price_Mom_24H",
    "VIXCLS_z",
    "vix3m_z",
    "PAYEMS_z",
    "UNRATE_z",
]
plot_features = [feature for feature in plot_features if feature in recent_feature_df.columns]

if not recent_feature_df.empty and plot_features:
    plt.figure(figsize=(16, 7))
    for feature in plot_features:
        plt.plot(
            recent_feature_df["as_of_ts"],
            recent_feature_df[feature],
            label=FEATURE_LABELS.get(feature, feature),
            linewidth=1.5,
        )
    plt.axhline(0, color="black", linestyle=":", alpha=0.5)
    plt.title(f"Recent Feature Tracking ({FEATURE_PLOT_DAYS}D)")
    plt.ylabel("Feature value")
    plt.legend(loc="upper left", ncol=2)
    plt.grid(True, alpha=0.25)
    plt.show()

if not recent_feature_df.empty:
    heatmap_features = [feature for feature in FEATURES if feature in recent_feature_df.columns]
    heatmap_source = (
        recent_feature_df.set_index("as_of_ts")[heatmap_features]
        .tail(50)
        .rename(columns=FEATURE_LABELS)
        .T
    )
    plt.figure(figsize=(16, 7))
    sns.heatmap(heatmap_source, cmap="RdYlGn_r", center=0)
    plt.title("Feature Heatmap: Last 50 Observations")
    plt.xlabel("Timestamp")
    plt.ylabel("Feature")
    plt.show()

if not recent_feature_df.empty and {"VIX_Inversion_Z", "Price_Mom_24H", "regime_name"}.issubset(recent_feature_df.columns):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=recent_feature_df,
        x="VIX_Inversion_Z",
        y="Price_Mom_24H",
        hue="regime_name",
        palette={"Expansion": "green", "Slowdown": "goldenrod", "Recession": "red"},
        alpha=0.8,
    )
    plt.title("Key Feature Pair: VIX Inversion vs 24H Momentum")
    plt.grid(True, alpha=0.25)
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=recent_feature_df,
        x="VIXCLS_z",
        y="vix3m_z",
        hue="regime_name",
        palette={"Expansion": "green", "Slowdown": "goldenrod", "Recession": "red"},
        alpha=0.8,
    )
    plt.title("Key Feature Pair: VIX Spot vs VIX 3M")
    plt.grid(True, alpha=0.25)
    plt.show()

if not recent_feature_df.empty and {"VIX_Inversion_Z", "Price_Mom_24H", "VIXCLS_z", "regime_name"}.issubset(recent_feature_df.columns):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = recent_feature_df["regime_name"].map(
        {"Expansion": "green", "Slowdown": "goldenrod", "Recession": "red"}
    ).fillna("gray")
    ax.scatter(
        recent_feature_df["VIX_Inversion_Z"],
        recent_feature_df["Price_Mom_24H"],
        recent_feature_df["VIXCLS_z"],
        c=colors,
        alpha=0.75,
        s=35,
    )
    ax.set_title("3D Feature View")
    ax.set_xlabel("VIX Inversion Z")
    ax.set_ylabel("24H Momentum")
    ax.set_zlabel("VIX Spot Z")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Export Table
# MAGIC This notebook does not write files. Use the Databricks result export menu if you want a CSV download.

# COMMAND ----------

display(feature_export_df.tail(250))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC - Probability values are computed on the fly from the current promoted model when `predict_proba` is available.
# MAGIC - Switch explanations are heuristic and based on the largest weighted feature moves versus the previous bar.
# MAGIC - The notebook is read-only by design and does not update live tables or jobs.

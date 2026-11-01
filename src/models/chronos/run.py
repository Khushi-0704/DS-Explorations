from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from chronos import Chronos2Pipeline

log = logging.getLogger(__name__)


# ============================================================
# LOAD DATA
# ============================================================
def load_data(cfg: dict) -> pd.DataFrame:
    """Load the prepared weekly sampled case data (region, date, target)."""
    c = cfg["columns"]
    region, date, target = c["region"], c["date"], c["target"]
    path = Path(cfg["data"]["path"])

    log.info("Loading data from: %s", path)
    df = pd.read_csv(path)

    missing = {region, date, target} - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df[date] = pd.to_datetime(df[date])
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[region, date, target])
    df = df[[region, date, target]].sort_values([region, date]).reset_index(drop=True)

    log.info("Data shape: %s", df.shape)
    log.info("Date range: %s -> %s", df[date].min(), df[date].max())
    log.info("Number of regions: %d", df[region].nunique())
    return df


# ============================================================
# DETECT SAMPLING WEEKDAY
# ============================================================
def detect_freq(df: pd.DataFrame, cfg: dict) -> str:
    """Return the weekly freq the data is sampled on, e.g. 'W-WED'."""
    configured = cfg["forecast"].get("freq", "auto")
    if configured != "auto":
        return configured

    date_col = cfg["columns"]["date"]
    dates = df[date_col].drop_duplicates().sort_values()

    # sampling dates must be a whole number of weeks apart
    gaps = dates.diff().dropna().dt.days
    if (gaps % 7 != 0).any():
        raise ValueError(
            "Dates are not weekly sampled (gaps between dates are not multiples of 7 days).\n"
            "This model needs one row per region per week, on the same weekday."
        )

    names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    weekday = int(dates.dt.dayofweek.mode().iloc[0])
    share = (dates.dt.dayofweek == weekday).mean()
    if share < 1:
        log.warning("Only %.0f%% of dates fall on %s.", share * 100, names[weekday])

    freq = f"W-{names[weekday]}"
    log.info("Detected sampling weekday: %s (freq=%s)", names[weekday], freq)

    dup = int(df.duplicated([cfg["columns"]["region"], date_col]).sum())
    if dup:
        log.warning("%d duplicate rows for the same region and date.", dup)
    return freq


# ============================================================
# GENERATE ORIGINS
# ============================================================
def generate_origins(df: pd.DataFrame, cfg: dict, freq: str) -> list[pd.Timestamp]:
    """Weekly cutoffs: from the end of the train period to one week before test end."""
    f, t, te = cfg["forecast"], cfg["train"], cfg["test"]
    dates = df[cfg["columns"]["date"]].drop_duplicates().sort_values()

    before = dates[dates <= pd.Timestamp(t["end"])]
    if before.empty:
        raise ValueError(f"No data on or before train.end ({t['end']}).")
    first = before.max()

    # last cutoff: one week before test end, but never after the last data date
    last = min(pd.Timestamp(te["end"]) - pd.Timedelta(weeks=1), dates.max())
    if last < first:
        raise ValueError(
            "test.end is too early. It must be at least 1 week after train.end "
            f"(train ends at {first.date()}, data ends at {dates.max().date()})."
        )

    origins = list(pd.date_range(first, last, freq=f"{f['slide_weeks']}{freq}"))
    if not origins:
        raise ValueError("No cutoffs generated. Check the train and test dates.")

    history_weeks = (first - dates.min()).days // 7
    if history_weeks < 52:
        log.warning("Only %d weeks of history before the first cutoff.", history_weeks)

    log.info("Number of origins: %d", len(origins))
    log.info("First origin: %s", origins[0].date())
    log.info("Last origin: %s", origins[-1].date())
    return origins


# ============================================================
# PREPARE CONTEXT (only data available at the origin)
# ============================================================
def prepare_context(df: pd.DataFrame, origin: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Future observations are never passed to Chronos."""
    c = cfg["columns"]
    context = df[df[c["date"]] <= origin].copy()
    context = context.rename(
        columns={c["region"]: "item_id", c["date"]: "timestamp", c["target"]: "target"}
    )
    context["item_id"] = context["item_id"].astype(str)
    return context[["item_id", "timestamp", "target"]].sort_values(["item_id", "timestamp"])


# ============================================================
# FORECAST ONE ORIGIN
# ============================================================
def forecast_one_origin(
    pipeline: Chronos2Pipeline,
    df: pd.DataFrame,
    origin: pd.Timestamp,
    cfg: dict,
    freq: str,
) -> pd.DataFrame:
    f, m = cfg["forecast"], cfg["model"]
    region_col = cfg["columns"]["region"]

    log.info("Forecasting origin: %s", origin.date())
    context = prepare_context(df, origin, cfg)
    if context.empty:
        log.warning("No context available for %s", origin.date())
        return pd.DataFrame()

    # only keep regions that have data at the origin
    last_seen = context.groupby("item_id")["timestamp"].max()
    valid_regions = last_seen[last_seen >= origin].index
    context = context[context["item_id"].isin(valid_regions)].copy()
    if context.empty:
        return pd.DataFrame()

    predictions = pipeline.predict_df(
        context,
        prediction_length=f["horizon_weeks"],
        quantile_levels=m["quantiles"],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        batch_size=m["batch_size"],
        freq=freq,
    )

    # rename chronos output columns, e.g. "0.1" -> "prediction_p10"
    rename = {
        "item_id": region_col,
        "timestamp": "targetDate",
        "predictions": "prediction",
    }
    for q in m["quantiles"]:
        rename[str(q)] = f"prediction_p{int(round(q * 100))}"
    predictions = predictions.rename(columns=rename)

    predictions[region_col] = predictions[region_col].astype(str)
    predictions["originDate"] = origin

    # lead: +1 = first forecast week, +2 = second, ...
    predictions["lead_h"] = (pd.to_datetime(predictions["targetDate"]) - origin).dt.days // 7
    predictions["model"] = m["label"]
    return predictions


# ============================================================
# ADD ACTUAL VALUES
# ============================================================
def add_actuals(predictions: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["columns"]
    actuals = df[[c["region"], c["date"], c["target"]]].copy()
    actuals = actuals.rename(columns={c["date"]: "targetDate", c["target"]: "y_true"})
    actuals[c["region"]] = actuals[c["region"]].astype(str)
    return predictions.merge(actuals, on=[c["region"], "targetDate"], how="left")


# ============================================================
# ENTRY POINT (called by main.py)
# ============================================================
def run(cfg: dict, out_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    region_col = cfg["columns"]["region"]
    date_col = cfg["columns"]["date"]
    m = cfg["model"]
    test_start = pd.Timestamp(cfg["test"]["start"])
    test_end = pd.Timestamp(cfg["test"]["end"])

    df = load_data(cfg)
    df = df[df[date_col] >= pd.Timestamp(cfg["train"]["start"])]
    freq = detect_freq(df, cfg)
    origins = generate_origins(df, cfg, freq)

    log.info("Loading Chronos model: %s", m["model_id"])
    pipeline = Chronos2Pipeline.from_pretrained(m["model_id"], device_map=m["device_map"])
    log.info("Chronos model loaded.")

    # ---- forecasting ----
    all_predictions = []
    for i, origin in enumerate(origins, start=1):
        log.info("=" * 50)
        log.info("Origin %d / %d: %s", i, len(origins), origin.date())

        pred = forecast_one_origin(pipeline, df, origin, cfg, freq)
        if pred.empty:
            log.warning("No predictions generated for %s", origin.date())
            continue

        # keep only targets inside the test period
        pred["targetDate"] = pd.to_datetime(pred["targetDate"])
        pred = pred[(pred["targetDate"] >= test_start) & (pred["targetDate"] <= test_end)].copy()

        if not pred.empty:
            all_predictions.append(pred)
        log.info("Predictions retained: %d", len(pred))

    if not all_predictions:
        raise RuntimeError("No Chronos predictions were generated.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = add_actuals(predictions, df, cfg)
    predictions = predictions.sort_values([region_col, "originDate", "targetDate"]).reset_index(
        drop=True
    )

    quantile_cols = [f"prediction_p{int(round(q * 100))}" for q in m["quantiles"]]
    final_columns = [
        "model", region_col, "originDate", "targetDate", "lead_h",
        "y_true", "prediction", *quantile_cols,
    ]
    predictions = predictions[[c for c in final_columns if c in predictions.columns]]

    # ---- save ----
    out_file = out_dir / cfg["output"]["predictions_file"]
    predictions.to_csv(out_file, index=False)

    # ---- summary ----
    print()
    print("=" * 70)
    print("CHRONOS FORECAST COMPLETE")
    print("=" * 70)
    print("Output:", out_file)
    print("Rows:", len(predictions))
    print("Regions:", predictions[region_col].nunique())
    print("Origins:", predictions["originDate"].nunique())
    print("Target dates:", predictions["targetDate"].min(), "->", predictions["targetDate"].max())
    print("\nRows by lead:")
    print(predictions["lead_h"].value_counts().sort_index())
    print("\nMissing actuals:", predictions["y_true"].isna().sum())
    print("=" * 70)
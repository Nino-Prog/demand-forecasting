"""Build model-ready features from data/sales.csv.

Adds calendar features, lag features, and rolling statistics per
(store, category) series. Lags/rolling stats are computed strictly on
past values within each series to avoid leakage.
"""
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]


def load_sales(path="data/sales.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store", "category", "date"]).reset_index(drop=True)


def add_calendar_features(df):
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def add_lag_and_rolling_features(df):
    group = df.groupby(["store", "category"])["units_sold"]

    for lag in LAGS:
        df[f"lag_{lag}"] = group.shift(lag)

    for window in ROLLING_WINDOWS:
        shifted = group.shift(1)  # exclude current day from its own rolling stats
        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df["store"], df["category"]])
            .rolling(window, min_periods=window)
            .mean()
            .reset_index(drop=True)
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby([df["store"], df["category"]])
            .rolling(window, min_periods=window)
            .std()
            .reset_index(drop=True)
        )
    return df


def build_features(input_path="data/sales.csv", output_path="data/features.csv"):
    df = load_sales(input_path)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)

    # Drop warm-up rows where the longest lag/rolling window isn't populated yet
    max_window = max(LAGS + ROLLING_WINDOWS)
    df["_row_in_series"] = df.groupby(["store", "category"]).cumcount()
    df = df[df["_row_in_series"] >= max_window].drop(columns="_row_in_series")

    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols to {output_path}")
    print(df.isna().sum()[df.isna().sum() > 0])
    return df


if __name__ == "__main__":
    build_features()

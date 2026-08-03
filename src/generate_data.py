"""Generate a synthetic daily sales dataset for the demand forecasting project.

Produces multi-store, multi-category daily sales with trend, weekly/yearly
seasonality, US holiday effects, promotion spikes, and noise -- realistic
enough to require genuine feature engineering and backtesting.
"""
import numpy as np
import pandas as pd
import holidays

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"
STORES = [f"store_{i}" for i in range(1, 6)]
CATEGORIES = ["electronics", "apparel", "home_goods", "grocery", "toys"]
RNG_SEED = 42


def build_calendar(start, end):
    dates = pd.date_range(start, end, freq="D")
    us_holidays = holidays.US(years=range(dates.min().year, dates.max().year + 1))
    return pd.DataFrame({
        "date": dates,
        "day_of_week": dates.dayofweek,
        "day_of_year": dates.dayofyear,
        "is_holiday": dates.isin(us_holidays).astype(int),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    })


def simulate_promotions(n_days, rng, promo_rate=0.06):
    return (rng.random(n_days) < promo_rate).astype(int)


def simulate_series(calendar, base_level, trend_per_year, category, rng):
    n = len(calendar)
    t = np.arange(n)

    trend = base_level + trend_per_year * (t / 365.0)
    weekly = 1.0 + 0.25 * np.sin(2 * np.pi * calendar["day_of_week"] / 7.0)
    yearly = 1.0 + 0.35 * np.sin(2 * np.pi * (calendar["day_of_year"] - 80) / 365.0)

    category_holiday_lift = {
        "electronics": 2.2, "apparel": 1.8, "home_goods": 1.5,
        "grocery": 1.3, "toys": 2.6,
    }[category]
    holiday_effect = 1.0 + (category_holiday_lift - 1.0) * calendar["is_holiday"]

    promo_flag = simulate_promotions(n, rng)
    promo_effect = 1.0 + 0.6 * promo_flag

    noise = rng.normal(1.0, 0.08, n)

    sales = trend * weekly * yearly * holiday_effect * promo_effect * noise
    sales = np.clip(sales, 0, None)
    return np.round(sales).astype(int), promo_flag


def main():
    rng = np.random.default_rng(RNG_SEED)
    calendar = build_calendar(START_DATE, END_DATE)

    rows = []
    for store in STORES:
        store_multiplier = rng.uniform(0.7, 1.4)
        for category in CATEGORIES:
            base_level = rng.uniform(40, 160) * store_multiplier
            trend_per_year = rng.uniform(-5, 20)
            sales, promo_flag = simulate_series(calendar, base_level, trend_per_year, category, rng)

            df = calendar.copy()
            df["store"] = store
            df["category"] = category
            df["promotion"] = promo_flag
            df["units_sold"] = sales
            rows.append(df)

    full = pd.concat(rows, ignore_index=True)
    full = full[["date", "store", "category", "day_of_week", "is_weekend",
                 "is_holiday", "promotion", "units_sold"]]
    full.to_csv("data/sales.csv", index=False)
    print(f"Wrote {len(full):,} rows to data/sales.csv")
    print(full.head())
    print(full.groupby("category")["units_sold"].describe())


if __name__ == "__main__":
    main()

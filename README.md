# Demand Forecasting Project

End-to-end demand forecasting project: synthetic daily sales data, baseline
vs. ML forecasting models with time-aware evaluation, and a deployed
forecast API.

## Status
- [x] Synthetic dataset generator (`src/generate_data.py`) — daily sales by
      store x category, with trend, weekly/yearly seasonality, US holidays,
      and promotions.
- [ ] Feature engineering (lags, rolling stats, calendar features)
- [ ] Baseline models (naive, seasonal-naive, Prophet/ARIMA)
- [ ] ML models (XGBoost/LightGBM) with time-aware cross-validation
- [ ] Backtesting + uncertainty intervals
- [ ] Forecast API (FastAPI)
- [ ] Portfolio write-up

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
```

## Data
`data/sales.csv` — one row per (date, store, category):
`date, store, category, day_of_week, is_weekend, is_holiday, promotion, units_sold`

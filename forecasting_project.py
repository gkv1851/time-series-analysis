"""
Time Series Forecasting: Monthly Anti-Diabetic Drug Sales (Australia, 1991-2008)
----------------------------------------------------------------------------
Dataset: "a10" from the fpp2 R package (Hyndman & Athanasopoulos,
"Forecasting: Principles and Practice"), fetched via statsmodels'
get_rdataset(), which pulls from the public Rdatasets GitHub mirror.

Goal: Forecast monthly anti-diabetic drug sales using classical time series
models and compare against a machine learning baseline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
raw = sm.datasets.get_rdataset("a10", "fpp2").data

# Convert the decimal "time" column (e.g. 1991.500000) into proper monthly dates
start = pd.Period("1991-07", freq="M")  # a10 series starts July 1991
dates = pd.period_range(start=start, periods=len(raw), freq="M").to_timestamp()
series = pd.Series(raw["value"].values, index=dates, name="drug_sales")
series.index.name = "month"

print("Series length:", len(series))
print(series.head())
print(series.tail())

# ---------------------------------------------------------------------------
# 2. EDA: trend + seasonality
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
series.plot(ax=ax, title="Monthly Anti-Diabetic Drug Sales (Australia)")
ax.set_ylabel("Sales ($ million)")
plt.tight_layout()
plt.savefig("plot_raw_series.png", dpi=120)
plt.close()

decomp = seasonal_decompose(series, model="multiplicative", period=12)
fig = decomp.plot()
fig.set_size_inches(10, 7)
plt.tight_layout()
plt.savefig("plot_decomposition.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 3. Train / test split (last 24 months held out for testing)
# ---------------------------------------------------------------------------
test_size = 24
train, test = series[:-test_size], series[-test_size:]
print(f"\nTrain: {train.index.min()} to {train.index.max()} ({len(train)} months)")
print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} months)")

results = {}

def evaluate(name, forecast):
    mape = mean_absolute_percentage_error(test, forecast) * 100
    rmse = np.sqrt(mean_squared_error(test, forecast))
    results[name] = {"MAPE_%": round(mape, 2), "RMSE": round(rmse, 3)}
    print(f"{name:25s} MAPE: {mape:6.2f}%   RMSE: {rmse:6.3f}")

# ---------------------------------------------------------------------------
# 4a. Baseline: Seasonal Naive (repeat value from 12 months ago)
# ---------------------------------------------------------------------------
seasonal_naive_fc = train[-12:].values
seasonal_naive_fc = np.tile(seasonal_naive_fc, test_size // 12 + 1)[:test_size]
evaluate("Seasonal Naive (baseline)", seasonal_naive_fc)

# ---------------------------------------------------------------------------
# 4b. Holt-Winters Exponential Smoothing (additive trend, multiplicative season)
# ---------------------------------------------------------------------------
hw_model = ExponentialSmoothing(
    train, trend="add", seasonal="mul", seasonal_periods=12
).fit()
hw_fc = hw_model.forecast(test_size)
evaluate("Holt-Winters (ETS)", hw_fc)

# ---------------------------------------------------------------------------
# 4c. SARIMA (Seasonal ARIMA)
# ---------------------------------------------------------------------------
sarima_model = SARIMAX(
    train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False)
sarima_fc = sarima_model.forecast(test_size)
evaluate("SARIMA(1,1,1)(1,1,1,12)", sarima_fc)

# ---------------------------------------------------------------------------
# 4d. Machine Learning baseline: Random Forest on lag + calendar features
# ---------------------------------------------------------------------------
def make_features(s, n_lags=12):
    df = pd.DataFrame({"y": s})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df.dropna()

feat_df = make_features(series, n_lags=12)
feat_train = feat_df.loc[feat_df.index <= train.index.max()]
feat_test = feat_df.loc[feat_df.index > train.index.max()]

X_train, y_train = feat_train.drop(columns="y"), feat_train["y"]
X_test, y_test = feat_test.drop(columns="y"), feat_test["y"]

rf_model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_fc = rf_model.predict(X_test)
evaluate("Random Forest (lag features)", rf_fc)

# ---------------------------------------------------------------------------
# 5. Plot all forecasts vs actual
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
train[-36:].plot(ax=ax, label="Train (last 3yr)", color="gray")
test.plot(ax=ax, label="Actual", color="black", linewidth=2)
pd.Series(seasonal_naive_fc, index=test.index).plot(ax=ax, label="Seasonal Naive", linestyle="--")
hw_fc.plot(ax=ax, label="Holt-Winters")
sarima_fc.plot(ax=ax, label="SARIMA")
pd.Series(rf_fc, index=test.index).plot(ax=ax, label="Random Forest")
ax.set_title("Forecast Comparison: Monthly Anti-Diabetic Drug Sales")
ax.set_ylabel("Sales ($ million)")
ax.legend()
plt.tight_layout()
plt.savefig("plot_forecast_comparison.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
summary = pd.DataFrame(results).T
summary = summary.sort_values("MAPE_%")
print("\n=== Model Comparison (sorted by MAPE) ===")
print(summary)
summary.to_csv("model_comparison_results.csv")



import pandas as pd
import numpy as np


def recursive_forecast(model, history, forecast_days):
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    last_date = history["date"].max()

    # Maintain independent sales history for every store-item combination
    sales_history = {
        (store, item): group.set_index("date")["sales"].to_dict()
        for (store, item), group in history.groupby(["store", "item"])
    }

    store_items = list(sales_history.keys())
    predictions = []

    # Forecast one day at a time so predictions become future lag values
    for day in range(1, forecast_days + 1):
        forecast_date = last_date + pd.Timedelta(days=day)
        daily_rows = []

        for store, item in store_items:
            values = sales_history[(store, item)]

            # Calculate lag features using actual or previously predicted sales
            lag_1 = values.get(forecast_date - pd.Timedelta(days=1), 0)
            lag_7 = values.get(forecast_date - pd.Timedelta(days=7), 0)
            lag_14 = values.get(forecast_date - pd.Timedelta(days=14), 0)
            lag_28 = values.get(forecast_date - pd.Timedelta(days=28), 0)

            # Calculate rolling averages from the previous 7 and 30 days
            recent_7 = [
                values.get(forecast_date - pd.Timedelta(days=i), 0)
                for i in range(1, 8)
            ]

            recent_30 = [
                values.get(forecast_date - pd.Timedelta(days=i), 0)
                for i in range(1, 31)
            ]

            row = {
                "store": store,
                "item": item,
                "year": forecast_date.year,
                "month": forecast_date.month,
                "day": forecast_date.day,
                "day_of_week": forecast_date.dayofweek,
                "day_of_year": forecast_date.dayofyear,
                "week_of_year": forecast_date.isocalendar().week,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "lag_28": lag_28,
                "rolling_mean_7": np.mean(recent_7),
                "rolling_mean_30": np.mean(recent_30)
            }

            daily_rows.append(row)

        daily_features = pd.DataFrame(daily_rows)

        # Predict sales for all store-item combinations on this date
        daily_features["predicted_sales"] = model.predict(daily_features)
        daily_features["date"] = forecast_date

        predictions.append(
            daily_features[["date", "store", "item", "predicted_sales"]]
        )

        # Store today's predictions so future dates can use them as lag features
        for _, row in daily_features.iterrows():
            key = (row["store"], row["item"])
            sales_history[key][forecast_date] = row["predicted_sales"]

    return pd.concat(predictions, ignore_index=True)

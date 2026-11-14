
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys
from pathlib import Path

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.forecasting import recursive_forecast

# =========================
# Load Model and Data
# =========================

model = joblib.load(
    project_root / "models" / "xgboost_sales_model.pkl"
)

sales_data = pd.read_csv(
    project_root / "data" / "raw" / "store_sales.csv"
)

sales_data["date"] = pd.to_datetime(sales_data["date"])

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.title("📊 Sales Forecast")
    st.markdown("---")

    st.subheader("Navigation")

    page = st.radio(
        "Select Page",
        ["Dashboard", "Forecast", "Model Performance"]
    )

    st.markdown("---")
    st.caption("Machine Learning Powered")
    st.caption("Model: XGBoost")

st.title("Demand Forecasting Dashboard")

st.markdown(
    "Forecast future retail sales using Machine Learning."
)

st.markdown("---")

# =========================
# Dashboard
# =========================

if page == "Dashboard":

    st.header("Business Overview")

    st.write(
        "Monitor historical retail sales and explore business performance."
    )

    # Calculate dashboard KPIs
    total_stores = sales_data["store"].nunique()
    total_items = sales_data["item"].nunique()
    total_records = len(sales_data)
    total_sales = sales_data["sales"].sum()

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Stores",
        f"{total_stores:,}"
    )

    col2.metric(
        "Total Items",
        f"{total_items:,}"
    )

    col3.metric(
        "Historical Records",
        f"{total_records:,}"
    )

    col4.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    st.markdown("---")

    # Historical Sales Chart
    st.subheader("Historical Sales Overview")

    historical_daily_sales = (
        sales_data.groupby("date")["sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        historical_daily_sales,
        x="date",
        y="sales",
        title="Historical Daily Sales",
        labels={
            "date": "Date",
            "sales": "Total Sales"
        }
    )

    fig.update_traces(
        line=dict(width=2),
        name="Daily Sales"
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=500,
        title_x=0.02,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    fig.update_xaxes(
        tickformat="%b %Y",
        title="Date"
    )

    fig.update_yaxes(
        tickformat=",.0f",
        title="Total Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "modeBarButtonsToRemove": ["zoom2d", "pan2d"],
            "displaylogo": False
        }
    )

    # Additional Business Insights
    st.markdown("---")

    st.subheader("Business Insights")

    average_daily_sales = historical_daily_sales["sales"].mean()

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Daily Sales",
        f"{average_daily_sales:,.2f}"
    )

    col2.metric(
        "Average Sales per Store",
        f"{total_sales / total_stores:,.2f}"
    )

# =========================
# Forecast
# =========================

elif page == "Forecast":

    st.header("Sales Forecast")

    st.write(
        "Generate future sales predictions for individual stores."
    )

    st.subheader("Forecast Settings")

    col1, col2 = st.columns(2)

    with col1:
        forecast_days = st.selectbox(
            "Select Forecast Period",
            options=[7, 14, 30],
            index=0
        )

    with col2:
        selected_store = st.selectbox(
            "Select Store",
            options=sorted(sales_data["store"].unique())
        )

    if st.button("Generate Forecast", type="primary"):

        with st.spinner("Generating sales forecast..."):

            forecast_results = recursive_forecast(
                model=model,
                history=sales_data,
                forecast_days=forecast_days
            )

        st.success("Sales forecast generated successfully!")

        store_forecast = forecast_results[
            forecast_results["store"] == selected_store
        ].copy()

        store_forecast["date"] = pd.to_datetime(
            store_forecast["date"]
        ).dt.strftime("%Y-%m-%d")

        st.subheader(
            f"Store {selected_store} Forecast"
        )

        st.dataframe(
            store_forecast,
            use_container_width=True,
            hide_index=True
        )

        # Download forecast CSV
        csv_data = store_forecast.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Forecast CSV",
            data=csv_data,
            file_name=f"store_{selected_store}_forecast.csv",
            mime="text/csv"
        )

        # Daily forecast
        daily_forecast = (
            store_forecast.groupby("date")["predicted_sales"]
            .sum()
        )

        st.markdown("---")

        st.subheader("Daily Sales Forecast")

        chart_data = daily_forecast.reset_index()

        fig = px.line(
            chart_data,
            x="date",
            y="predicted_sales",
            markers=True,
            title=f"Store {selected_store} - {forecast_days} Day Sales Forecast",
            labels={
                "date": "Forecast Date",
                "predicted_sales": "Predicted Sales"
            }
        )

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=450
        )

        fig.update_xaxes(
            tickformat="%d %b %Y",
            tickangle=-45
        )

        fig.update_yaxes(
            tickformat=",.0f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "modeBarButtonsToRemove": ["zoom2d", "pan2d"],
                "displaylogo": False
            }
        )

        # Forecast Summary
        st.markdown("---")

        st.subheader("Forecast Summary")

        total_forecast_sales = daily_forecast.sum()
        average_daily_forecast = daily_forecast.mean()

        highest_sales_day = daily_forecast.idxmax()
        highest_sales = daily_forecast.max()

        lowest_sales_day = daily_forecast.idxmin()
        lowest_sales = daily_forecast.min()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Forecast Sales",
            f"{total_forecast_sales:,.2f}"
        )

        col2.metric(
            "Average Daily Sales",
            f"{average_daily_forecast:,.2f}"
        )

        col3.metric(
            "Highest Daily Sales",
            f"{highest_sales:,.2f}"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Lowest Daily Sales",
            f"{lowest_sales:,.2f}"
        )

        col2.write(
            f"Highest Sales Date: {highest_sales_day}"
        )

        col2.write(
            f"Lowest Sales Date: {lowest_sales_day}"
        )

# =========================
# Model Performance
# =========================

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Performance comparison of all forecasting models using daily aggregated sales."
    )

    # Model Comparison Table
    daily_model_comparison = pd.read_csv(
        project_root / "reports" / "daily_model_comparison.csv"
    )

    st.subheader("Daily Aggregate Model Comparison")

    st.dataframe(
        daily_model_comparison.style.format({
            "Daily MAE": "{:.2f}",
            "Daily RMSE": "{:.2f}",
            "Daily MAPE (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Load XGBoost Predictions
    prediction_file = (
        project_root / "data" / "processed" /
        "xgboost_daily_comparison.csv"
    )

    comparison_data = pd.read_csv(prediction_file)

    # Identify existing date column
    date_column = None

    for column in ["date", "Date", "Unnamed: 0", "index"]:
        if column in comparison_data.columns:
            date_column = column
            break

    # Handle prediction CSV with saved dates
    if date_column is not None:

        comparison_data = comparison_data.rename(
            columns={date_column: "date"}
        )

        comparison_data["date"] = pd.to_datetime(
            comparison_data["date"],
            errors="coerce"
        )

    # Handle prediction CSV without dates
    else:

        comparison_data["date"] = pd.date_range(
            start="2017-01-01",
            periods=len(comparison_data),
            freq="D"
        )

    if "predicted_sales" not in comparison_data.columns:

        st.error(
            "The prediction CSV does not contain predicted_sales."
        )

        st.stop()

    # Evaluation Period
    comparison_data = comparison_data[
        comparison_data["date"].between(
            "2017-01-01",
            "2017-12-31"
        )
    ].copy()

    # Calculate Actual Daily Sales
    actual_daily_sales = (
        sales_data[
            sales_data["date"].between(
                "2017-01-01",
                "2017-12-31"
            )
        ]
        .groupby("date", as_index=False)["sales"]
        .sum()
    )

    # Merge Actual and Predicted Sales
    comparison_data = comparison_data.merge(
        actual_daily_sales,
        on="date",
        how="inner"
    )

    # Actual vs Predicted Chart
    st.subheader("XGBoost Actual vs Predicted Daily Sales")

    if comparison_data.empty:

        st.error(
            "No matching dates found between actual sales and predictions."
        )

    else:

        fig = px.line(
            comparison_data,
            x="date",
            y="sales",
            title="XGBoost Actual vs Predicted Sales",
            labels={
                "date": "Date",
                "sales": "Daily Sales"
            }
        )

        fig.update_traces(
            name="Actual Sales",
            showlegend=True
        )

        fig.add_scatter(
            x=comparison_data["date"],
            y=comparison_data["predicted_sales"],
            mode="lines",
            name="Predicted Sales"
        )

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=500,
            legend_title="Sales Type"
        )

        fig.update_xaxes(
            tickformat="%b %Y"
        )

        fig.update_yaxes(
            tickformat=",.0f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "modeBarButtonsToRemove": ["zoom2d", "pan2d"],
                "displaylogo": False
            }
        )

        st.caption(
            f"Actual and predicted daily sales plotted for "
            f"{len(comparison_data)} days."
        )

    # Model Information
    st.subheader("Model Information")

    st.write("Evaluation Period: January–December 2017")
    st.write("Evaluation Level: Daily Aggregated Sales")
    st.write("Models Evaluated: 5")
    st.write("Best Model: XGBoost")
















import streamlit as st
import pandas as pd
import pickle
from io import BytesIO

with open("sales_forecast_model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("FMCG Sales Forecasting")

uploaded_file = st.file_uploader(
    "Upload Sales Data CSV",
    type=["csv"]
)

required_input_cols = [
    "Quantity",
    "Discount_Amount",
    "Customer_Rating",
    "Returned",
    "Order_ID",
    "Date",
    "Total_Sales",
]

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read uploaded CSV: {e}")
    else:
        st.write("Uploaded Data (first rows)")
        st.dataframe(df.head())
        df['Returned'] = df['Returned'].map({
            'No': 0,
            'Yes': 1
        })

        missing = [c for c in required_input_cols if c not in df.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            # Parse dates and build derived features
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            except Exception:
                # try generic parse if specific format fails
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

            df = df.sort_values('Date').reset_index(drop=True)
            df['Trend'] = range(len(df))

            for lag in [1, 7, 14, 21, 28]:
                df[f'Lag_{lag}'] = df['Total_Sales'].shift(lag)

            for window in [7, 14, 21, 28]:
                df[f'RollingMean_{window}'] = (
                    df['Total_Sales'].shift(1).rolling(window).mean()
                )
                df[f'RollingStd_{window}'] = (
                    df['Total_Sales'].shift(1).rolling(window).std()
                )

            df = df.dropna().reset_index(drop=True)

            # Prepare features in the same order the model expects
            feature_cols = list(model.feature_names_in_)
            # If the model expects 'Order_Count' but the upload provided 'Order_ID',
            # derive Order_Count as the number of rows per Order_ID (items per order).
            if 'Order_Count' in feature_cols and 'Order_Count' not in df.columns:
                if 'Order_ID' in df.columns:
                    df['Order_Count'] = df.groupby('Order_ID')['Order_ID'].transform('count')

            if not all(col in df.columns for col in feature_cols):
                missing_feats = [c for c in feature_cols if c not in df.columns]
                st.error("After feature construction, missing features: " + ", ".join(missing_feats))
            else:
                X = df[feature_cols]
                try:
                    predictions = model.predict(X)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                else:
                    df['Forecasted_Sales'] = predictions

                    st.success(f"Forecast generated for {len(df)} rows.")
                    st.dataframe(df[["Date", "Total_Sales", "Forecasted_Sales"]].head())

                    st.line_chart(df.set_index('Date')[["Total_Sales", "Forecasted_Sales"]])

                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Forecast",
                        csv,
                        "forecast.csv",
                        "text/csv",
                    )
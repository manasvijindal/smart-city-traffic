import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Smart City: Traffic Volume Prediction", layout="wide")

# ---------- Load data & model ----------
@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Metro_Interstate_Traffic_Volume.csv")

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # CLEANING
    df = df[df["temp"] > 0]
    df = df[df["rain_1h"] < 60]
    df = df[df["snow_1h"] < 60]

    # FIX HOLIDAY COLUMN
    df["holiday"] = df["holiday"].fillna("None").astype(str)

    # FIX WEATHER COLUMN TOO (same problem may happen)
    df["weather_main"] = df["weather_main"].fillna("Unknown").astype(str)

    # Feature engineering
    df["hour"] = df["date_time"].dt.hour
    df["month"] = df["date_time"].dt.month
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["year"] = df["date_time"].dt.year

    return df


@st.cache_resource
def load_model():
    model_path = "models/rf_traffic_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Train the model notebook first.")
        return None
    return joblib.load(model_path)

df = load_data()
model = load_model()

st.title("🚦 Smart City: Traffic Volume Prediction on I-94")
st.markdown(
    """
This dashboard uses a Machine Learning model to **predict hourly traffic volume**
on Interstate 94 (Minneapolis), using **weather + time** features.
"""
)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

year_opt = st.sidebar.selectbox("Year", sorted(df["year"].unique()))
month_opt = st.sidebar.selectbox("Month", sorted(df["month"].unique()))
day_opt = st.sidebar.slider("Day of month", 1, 31, 1)

mask = (
    (df["year"] == year_opt)
    & (df["month"] == month_opt)
    & (df["date_time"].dt.day == day_opt)
)

day_df = df[mask].sort_values("date_time")

st.subheader("Traffic Volume for Selected Day")

if day_df.empty:
    st.warning("No data for this day. Try another date.")
else:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(day_df["date_time"], day_df["traffic_volume"])
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic Volume")
    ax.set_title(f"Traffic Volume on {year_opt}-{month_opt:02d}-{day_opt:02d}")
    st.pyplot(fig)

# ---------- Prediction section ----------
st.subheader("Predict Traffic Volume for Custom Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider("Hour of day (0–23)", 0, 23, 8)
    temp = st.slider("Temperature (K)", float(df["temp"].min()), float(df["temp"].max()), float(df["temp"].median()))

with col2:
    rain = st.slider("Rain in last hour (mm)", float(df["rain_1h"].min()), float(df["rain_1h"].max()), float(0.0))
    snow = st.slider("Snow in last hour (mm)", float(df["snow_1h"].min()), float(df["snow_1h"].max()), float(0.0))

with col3:
    clouds = st.slider("Cloud cover (%)", int(df["clouds_all"].min()), int(df["clouds_all"].max()), int(df["clouds_all"].median()))
    holiday = st.selectbox("Holiday", sorted(df["holiday"].unique()))
    weather_main = st.selectbox("Weather main", sorted(df["weather_main"].unique()))

if model is not None:
    if st.button("Predict Traffic Volume"):
        input_row = pd.DataFrame(
            [{
                "temp": temp,
                "rain_1h": rain,
                "snow_1h": snow,
                "clouds_all": clouds,
                "hour": hour,
                "month": month_opt,
                "dayofweek": 0,  # placeholder (Mon)
                "year": year_opt,
                "holiday": holiday,
                "weather_main": weather_main,
            }]
        )

        pred = model.predict(input_row)[0]
        st.success(f"Predicted traffic volume: **{pred:.0f} vehicles/hour**")

        st.caption("Note: Prediction based on Random Forest regression model.")
else:
    st.warning("Model not loaded.")

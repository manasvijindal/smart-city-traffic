
# Smart City Traffic Forecasting using Machine Learning

This project is part of the **Smart City with Artificial Intelligence** lab assignment.  
It demonstrates how AI can be used for real-world smart-city applications such as **traffic forecasting**, by building a full end-to-end system:

> **Dataset → Preprocessing → ML Model → Evaluation → Streamlit Dashboard Deployment**


# Aim

To build a working prototype that applies AI techniques to predict traffic volume using real spatio-temporal urban data and deploy the results via a web-based dashboard.


# Learning Objectives (as per assignment)

- Acquire real urban datasets (sensor/weather/traffic)
- Preprocess spatio-temporal data for ML
- Train & evaluate ML regression model
- Deploy results as a **Streamlit dashboard**
- Understand ethical, privacy, & deployment considerations


# Project Structure

smart-city-traffic/
├── app.py                        # Streamlit dashboard (main app)
├── requirements.txt              # Dependencies
├── .gitignore                    # Ignoring large files (e.g. models)
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
├── notebook/
│   └── 01_traffic_forecasting.ipynb
└── models/
    └── (ignored - model files not stored on GitHub)
````

---

# Dataset

**Metro Interstate Traffic Volume Dataset**
Sensors on I-94 highway (Minneapolis) recording:

* Hourly traffic volume (target)
* Temperature, rain, snow, cloudiness
* Timestamp (converted → hour, month, weekday, year)
* Weather category & holiday flags

Dataset placed in:

```text
data/Metro_Interstate_Traffic_Volume.csv
```

---

# Model

A **Random Forest Regressor** is trained on:

* Numeric features:
  `temp, rain_1h, snow_1h, clouds_all, hour, month, dayofweek, year`
* Categorical features:
  `weather_main, holiday`

Training is done inside the notebook, and for **Streamlit Cloud**, the model can be trained at runtime to avoid uploading large `.pkl` files (>100MB).

Evaluation metrics used:

* **MAE**
* **RMSE**
* **R² Score**

---

# Streamlit Dashboard Features

The dashboard provides:

### ✔ Data Preview & Cleaning Summary

### ✔ Date selection to visualize traffic pattern for that day

### ✔ User inputs:

* Hour
* Temperature
* Rain / Snow
* Cloudiness
* Holiday
* Weather condition

### ✔ Predict Button → Traffic Volume Prediction

### ✔ Visual plots & daily line charts

---

# Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. Go to: [Link](https://smart-city-traffic.streamlit.app/) (Streamlit Community Cloud).
3. Click **“New app”** and select:

   * Repo: `manasvijindal/smart-city-traffic`
   * Branch: `main`
   * Main file: `app.py`
4. Click **Deploy**.

> Note: No large `.pkl` file is required if the app trains a small model at startup from the CSV.

---

# 🛠️ Step-by-Step Run Instructions and Dependencies

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/manasvijindal/smart-city-traffic.git
cd smart-city-traffic
```

## 2️⃣ Create Virtual Environment (Optional but recommended)

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies (from `requirements.txt`)

* `streamlit`
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `joblib`

(You can add more if you use them.)

## 4️⃣ Ensure Dataset Exists

Make sure the dataset file is present at:

```text
data/Metro_Interstate_Traffic_Volume.csv
```

If not, download the **Metro Interstate Traffic Volume** dataset (UCI / Kaggle) and save it with this exact name in the `data/` folder.

## 5️⃣ Run the Streamlit App Locally

```bash
streamlit run app.py
```

Then open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

# 🧪 Jupyter Notebook

File:

```text
notebook/01_traffic_forecasting.ipynb
```

The notebook includes:

* Exploratory Data Analysis (EDA)
* Feature engineering
* Model training
* Evaluation (MAE, RMSE, R²)
* Visualizations (plots)

```
::contentReference[oaicite:0]{index=0}

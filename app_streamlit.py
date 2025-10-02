import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import joblib, json

API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="EcoSmart Demo", page_icon="⚡", layout="centered")
st.title("⚡ EcoSmart Energy Demo")
st.markdown("### Smart Prediction & Energy-Saving Recommendations")
st.write("This demo predicts appliance energy usage and suggests actions to save power using Machine Learning.")
st.markdown("---")
st.subheader("🔍 Latest Prediction")

if st.button("Predict Latest Sample"):
    r = requests.post(f"{API_URL}/predict", json={})
    data = r.json()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Energy (Wh)", f"{data['prediction']:.1f}")
    with col2:
        st.metric("Threshold", f"{data['threshold']:.1f}")
    if data['action'] == "reduce":
        st.error("⚠️ Recommendation: Reduce non-essential appliances")
        if st.button("Simulate OFF Plug1"):
            payload = {
                "device_id": "plug1",
                "action": "OFF",
                "prediction": data["prediction"],
                "threshold": data["threshold"]
            }
            rc = requests.post(f"{API_URL}/control", json=payload)
            st.success("Simulated OFF action logged!")
    else:
        st.success("No action needed")

st.markdown("---")
st.subheader("Recent Usage Trend (last 24 samples)")

try:
    df = pd.read_excel("cleaned_energydata.csv.xlsx")
    df['date'] = pd.to_datetime(df['date'])
    recent = df.tail(24)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(recent['date'], recent['Appliances'], label="Actual Usage", marker='o', color="green")
    ax.set_title("Appliance Energy Consumption (Wh)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Wh")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.info("No dataset available for trend plot.")

st.markdown("---")
st.subheader("Feature Importance (Model Insights)")

try:
    model = joblib.load("energy_predictor_model.joblib")
    with open("features.json") as f:
        features = json.load(f)

    importances = model.feature_importances_
    fi = pd.DataFrame({"Feature": features, "Importance": importances})
    fi = fi.sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fi["Feature"], fi["Importance"], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Important Features")
    plt.gca().invert_yaxis()
    st.pyplot(fig)

except Exception as e:
    st.info("Could not load model or features for importance chart.")

st.markdown("---")

st.subheader("Actual vs Predicted Energy Usage")

try:
    df = pd.read_excel("cleaned_energydata.csv.xlsx")
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    with open("features.json") as f:
        features = json.load(f)
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("energy_predictor_model.joblib")

    X = scaler.transform(df[features].values)
    y_true = df['Appliances'].values
    y_pred = model.predict(X)

    sample_size = 200  
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[:sample_size], y_pred[:sample_size], alpha=0.5, color="purple")
    ax.plot([0, max(y_true[:sample_size])], [0, max(y_true[:sample_size])], 'r--')
    ax.set_xlabel("Actual (Wh)")
    ax.set_ylabel("Predicted (Wh)")
    ax.set_title("Actual vs Predicted Energy Usage (Sample)")
    st.pyplot(fig)

except Exception as e:
    st.info("Could not generate Actual vs Predicted plot.")


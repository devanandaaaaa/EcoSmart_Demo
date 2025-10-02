from flask import Flask, request, jsonify
import joblib, json, pandas as pd
from control_utils import control_plug

app = Flask(__name__)

model = joblib.load("energy_predictor_model.joblib")
scaler = joblib.load("scaler.joblib")
with open("features.json") as f:
    FEATURES = json.load(f)

DATA_PATH = "cleaned_energydata.csv.xlsx"

def prepare_sample():
    df = pd.read_excel(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    sample = df[FEATURES].iloc[-1:]
    return sample, df

@app.route('/predict', methods=['POST'])
def predict():
    sample, df = prepare_sample()
    X = scaler.transform(sample.values)
    pred = float(model.predict(X)[0])
    threshold = df['Appliances'].mean() + df['Appliances'].std()
    action = "reduce" if pred > threshold else "ok"
    return jsonify({"prediction": pred, "threshold": threshold, "action": action})

@app.route('/control', methods=['POST'])
def control():
    data = request.json
    device_id = data.get("device_id", "plug1")
    action = data.get("action", "OFF")
    pred = data.get("prediction", 0)
    threshold = data.get("threshold", 0)
    control_plug(device_id, action, pred, threshold)
    return jsonify({"status": "success", "device": device_id, "action": action})

if __name__ == "__main__":
    app.run(port=5000, debug=True)                                


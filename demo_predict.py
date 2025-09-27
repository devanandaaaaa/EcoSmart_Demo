import joblib, json, pandas as pd
from datetime import datetime
from control_utils import control_plug

MODEL_PATH = "energy_predictor_model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURES_PATH = "features.json"
DATA_PATH = "cleaned_energydata.csv.xlsx"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH) as f:
    features = json.load(f)

df = pd.read_excel(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

sample = df[features].iloc[-1:]
X = scaler.transform(sample.values) 
pred = float(model.predict(X)[0])  


threshold = df['Appliances'].mean() + df['Appliances'].std()
action = "REDUCE" if pred > threshold else "OK"

print(f"⏰ Time: {datetime.now().isoformat()}")
print(f"🔌 Predicted Appliances (Wh): {pred:.1f}")
print(f"⚖️ Threshold: {threshold:.1f}")
if action == "REDUCE":
    print("👉 Recommendation: Turn off non-essential appliances")
    control_plug("plug1", "OFF", pred, threshold)
else:
    print("👉 Recommendation: No action needed")

import json
import pandas as pd

df = pd.read_excel("cleaned_energydata.csv.xlsx")

df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['weekday'] = df['date'].dt.weekday
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

features = [c for c in df.columns if c not in ['Appliances', 'date']]

with open("features.json", "w") as f:
    json.dump(features, f, indent=2)

print("✅ Saved features.json with", len(features), "features.")

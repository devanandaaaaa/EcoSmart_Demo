⚡ EcoSmart Energy Demo
A machine learning–powered prototype to **predict appliance energy usage** and recommend **smart energy-saving actions**.

---

👥 Team Members
- Neha – Data Collection & Preprocessing  
- Anamika – Exploratory Data Analysis  
- Devananda – Machine Learning Model Development  
- Arunika – System Design, Documentation & Presentation  

---

📊 Project Overview
This system uses the **Appliances Energy Prediction Dataset** (UCI ML Repository) to:
- Predict appliance energy consumption (Wh)  
- Compare against a threshold  
- Suggest energy-saving actions (e.g., turn off non-essential appliances)  
- Simulate smart plug OFF/ON control  

The model is a **Random Forest Regressor** trained on environmental and temporal features, achieving **R² ≈ 0.70**.

---

📂 Project Structure

EcoSmart_Demo/
│
├── energy_predictor_model.joblib # trained ML model
├── scaler.joblib # scaler for features
├── features.json # feature order
├── cleaned_energydata.csv.xlsx # dataset
│
├── demo_predict.py # CLI demo script
├── control_utils.py # simulated smart plug control
├── app.py # Flask API
├── app_streamlit.py # Streamlit dashboard
│
├── action_logs.csv # (auto-generated logs)
├── requirements.txt # dependencies
└── README.md # this file


---

⚙️ Setup Instructions

### 1. Clone or download project folder
Place all files inside a folder named `EcoSmart_Demo`.

### 2. Install dependencies
```bash
pip install -r requirements.txt

python demo_predict.py

python app.py
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{}' 

streamlit run app_streamlit.py  #in another new terminal

➡ Open http://localhost:8501 in your browser.

Dashboard includes:
Prediction & Recommendation
Recent Usage Trend (24 samples)
Feature Importance chart
Actual vs Predicted scatter plot

📑 References
UCI ML Repository – Appliances Energy Prediction Dataset
scikit-learn Documentation
Pandas & Matplotlib Docs
Streamlit Official Docs
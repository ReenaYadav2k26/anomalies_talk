from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# -----------------------------
# 🔹 LOAD MODEL ARTIFACTS
# -----------------------------
try:
    model = pickle.load(open("model/isolation_forest_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    features = pickle.load(open("model/features.pkl", "rb"))
    # ✅ threshold.pkl ki zarurat nahi, delete kar do usse
except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")

# -----------------------------
# 🔐 API KEY (env-based)
# -----------------------------
API_KEY = os.getenv("API_KEY", "my-secret-key-123")

# -----------------------------
# 🔹 PREPROCESS FUNCTION
# -----------------------------
def preprocess_input(df):
    try:
        date_cols = ['due_date', 'paid_date', 'bill_from_date', 'bill_thru_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        for col in features:
            if col not in df.columns:
                df[col] = 0

        df = df[features]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

# -----------------------------
# 🔹 PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()

        if not data or "invoice_features" not in data:
            return jsonify({"error": "Missing 'invoice_features'"}), 400

        if not isinstance(data["invoice_features"], dict):
            return jsonify({"error": "Invalid format for invoice_features"}), 400

        df = pd.DataFrame([data["invoice_features"]])
        df_processed = preprocess_input(df)
        X_scaled = scaler.transform(df_processed)

        # ✅ UPDATED: model.predict() uses contamination=0.2 internally
        score = model.decision_function(X_scaled)[0]
        prediction = model.predict(X_scaled)[0]  # -1 = anomaly, 1 = normal

        result = {
            "anomaly_score": float(score),
            "is_anomaly": bool(prediction == -1)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 🔹 HEALTH CHECK
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Invoice Anomaly Detection API"
    })

# -----------------------------
# 🔹 MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

"""
Customer Churn Prediction using ANN (MLPClassifier from scikit-learn)
Streamlit Cloud compatible — NO TensorFlow needed!
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="wide")

MODEL_PATH = "churn_model.pkl"
SCALER_PATH = "scaler.pkl"

# ---------------- Synthetic Data Generator ----------------
def generate_synthetic_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    credit_score = rng.integers(350, 850, n)
    geography = rng.integers(0, 3, n)        # 0=France, 1=Germany, 2=Spain
    gender = rng.integers(0, 2, n)           # 0=Female, 1=Male
    age = rng.integers(18, 90, n)
    tenure = rng.integers(0, 11, n)
    balance = rng.uniform(0, 250000, n)
    products = rng.integers(1, 5, n)
    has_card = rng.integers(0, 2, n)
    active = rng.integers(0, 2, n)
    salary = rng.uniform(10000, 200000, n)

    # Rule-based churn label
    score = (
        (credit_score < 500).astype(int) * 2 +
        (age > 60).astype(int) * 2 +
        (active == 0).astype(int) * 2 +
        (balance < 10000).astype(int) +
        (products >= 3).astype(int) * 2 +
        (geography == 1).astype(int)
    )
    prob = 1 / (1 + np.exp(-(score - 3)))
    churn = (rng.random(n) < prob).astype(int)

    X = np.column_stack([credit_score, geography, gender, age, tenure,
                         balance, products, has_card, active, salary])
    return X, churn

# ---------------- Train / Load Model ----------------
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    with st.spinner("🧠 Training ANN model (first run only)..."):
        X, y = generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ANN: MLP with 2 hidden layers
        model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        acc = model.score(X_test_s, y_test)
        st.success(f"✅ Model trained! Test accuracy: {acc:.2%}")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

    return model, scaler

# ---------------- UI ----------------
st.title("🔮 AI Customer Churn Predictor")
st.markdown("**Artificial Neural Network (MLP)** powered prediction — runs on Streamlit Cloud without TensorFlow.")

model, scaler = get_model()

st.divider()
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    credit_score = st.slider("Credit Score", 350, 850, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 90, 35)
    tenure = st.slider("Tenure (years)", 0, 10, 3)

with col2:
    balance = st.number_input("Balance ($)", 0.0, 300000.0, 50000.0, step=1000.0)
    products = st.slider("Number of Products", 1, 4, 1)
    has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    active = st.selectbox("Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary ($)", 10000.0, 250000.0, 75000.0, step=1000.0)

if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    features = np.array([[
        credit_score,
        geo_map[geography],
        1 if gender == "Male" else 0,
        age,
        tenure,
        balance,
        products,
        1 if has_card == "Yes" else 0,
        1 if active == "Yes" else 0,
        salary,
    ]])

    features_scaled = scaler.transform(features)
    prob = float(model.predict_proba(features_scaled)[0][1])
    pred = int(prob >= 0.5)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{prob:.1%}")
    c2.metric("Prediction", "🔴 Will Churn" if pred else "🟢 Will Stay")
    risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    c3.metric("Risk Level", risk)

    st.progress(prob)

    if pred:
        st.error("⚠️ This customer is **likely to churn**. Recommend retention offers!")
    else:
        st.success("✅ This customer is **likely to stay**. Keep engagement high!")

st.divider()
st.caption("Built with scikit-learn MLPClassifier (ANN) + Streamlit")

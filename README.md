# 🔮 Customer Churn Predictor (ANN)

ANN-based churn prediction using **scikit-learn's MLPClassifier** — no TensorFlow needed!
Works perfectly on **Streamlit Cloud**.

## 🚀 Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Streamlit Cloud Deployment
1. Upload `app.py` and `requirements.txt` to a **GitHub repo**.
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → main file: `app.py`
4. Click **Deploy** ✅

## ⚠️ Important
- File ka naam **`app.py`** rakho (NO spaces, no `app (1).py`).
- `requirements.txt` zaroor upload karo.

## 🧠 Model
- Architecture: MLP with hidden layers (16, 8), ReLU activation
- Trained on 5000 synthetic samples on first run
- Saved as `churn_model.pkl` for reuse

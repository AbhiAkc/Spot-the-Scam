import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import predict
from pipeline import train_model
from datetime import datetime
import shap
import joblib
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# -----------------------------
# STREAMLIT THEME AND LAYOUT
# -----------------------------
st.set_page_config(page_title="Spot the Scam", layout="wide")

# -----------------------------
# TOGGLE DARK/LIGHT MODE (manually set default theme in .streamlit/config.toml)
# -----------------------------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0E1117; color: #FAFAFA; }
        .stButton>button { background-color: #08F7FE; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# TITLE & FILE UPLOAD
# -----------------------------
st.title("🕵️ Spot the Scam - Job Fraud Detection")

uploaded = st.file_uploader("📤 Upload job listings CSV", type="csv")

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
if uploaded:
    with st.spinner("🔍 Analyzing for potential scams..."):
        results = predict(uploaded)
        results['is_fraud'] = results['is_fraud'].map({0: 'No', 1: 'Yes'})

    st.subheader("📋 Predictions")
    st.dataframe(results, use_container_width=True)

    st.subheader("📊 Fraud Probability Histogram")
    st.bar_chart(results['fraud_prob'])

    st.subheader("🥧 Genuine vs Fraud")
    pie_data = results['is_fraud'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
    st.pyplot(fig)

    st.subheader("🚨 Top 10 High-Risk Listings")
    top10 = results.sort_values('fraud_prob', ascending=False).head(10)
    for count, (i, row) in enumerate(top10.iterrows(), 1): 
        if row['fraud_prob'] > 0.8:
            st.warning(f"{count}. ⚠️ {row['title']} ({row['location']}) - Scam Likely! ({row['fraud_prob']:.2f}) [CSV Index: {i}]")
        else:
            st.info(f"{count}. ✅ {row['title']} ({row['location']}) - Looks Safe ({row['fraud_prob']:.2f}) [CSV Index: {i}]")

    st.download_button("📥 Download Results as CSV", results.to_csv(index=False), "predictions.csv")

    # -----------------------------
    # SHAP EXPLAINABILITY (Optional)
    # -----------------------------
    st.subheader("🧠 Feature Impact (SHAP Values)")
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        model = joblib.load(model_path)
        
        text = results['title'].fillna('') + ' ' + results['description'].fillna('')
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']

        # Transform input text
        X_sparse = tfidf.transform(text)
        X_dense = X_sparse.toarray()

        # Limit rows to avoid OOM
        X_sample = X_dense[:50]

        # SHAP kernel explainer works with probability functions
        explainer = shap.Explainer(clf.predict_proba, X_sample)
        shap_values = explainer(X_sample)

        # Plot
        shap.summary_plot(shap_values, X_sample, show=False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(bbox_inches='tight')

    except Exception as e:
        st.error("❌ SHAP plot could not be generated. This may be due to memory limits or incompatible data format.")
        st.caption(f"Details: {e}")


    # -----------------------------
    # WORD CLOUD OF FRAUD JOBS
    # -----------------------------
    st.subheader("☁️ Common Words in Fraudulent Listings")
    try:
        fraud_text = " ".join(results[results['is_fraud'] == 'Yes']['title'].fillna('') + ' ' + results[results['is_fraud'] == 'Yes']['location'].fillna(''))
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(fraud_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"WordCloud generation failed: {e}")

# -----------------------------
# FEEDBACK FORM
# -----------------------------
st.subheader("💬 Feedback")
feedback = st.text_area("Have suggestions or found a bug?")
if st.button("Submit Feedback"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback.csv", "a") as f:
        f.write(f"{timestamp},{feedback}\n")
    st.success("✅ Thanks! Your feedback has been recorded.")

# -----------------------------
# RETRAIN MENU (Original vs New Data)
# -----------------------------
st.subheader("🔁 Retrain the Model")
retrain_option = st.radio("Choose Retraining Option:", ["Retrain on Original Data", "Retrain on New Uploaded Data"])
if retrain_option == "Retrain on Original Data":
    if st.button("Start Retraining (Original Data)"):
        with st.spinner("🔧 Retraining the model on original data..."):
            train_model()
            st.success("✅ Model retrained on original data.")
elif retrain_option == "Retrain on New Uploaded Data":
    new_data = st.file_uploader("📤 Upload new training data CSV", type="csv", key="new")
    if new_data and st.button("Start Retraining (New Data)"):
        with st.spinner("🔧 Retraining the model on uploaded data..."):
            train_model(data_path=new_data)
            st.success("✅ Model retrained on new uploaded data.")

# -----------------------------
# FASTAPI MOCK SETUP (Non-blocking, for reference/demo)
# -----------------------------
class JobData(BaseModel):
    title: str
    description: str

app_api = FastAPI()
model_api = joblib.load('app/model.pkl')

@app_api.post("/predict")
def predict_job(data: JobData):
    text = data.title + " " + data.description
    prob = model_api.predict_proba([text])[0][1]
    return {"fraud_prob": float(prob)}

# To run this FastAPI server separately (for judges):
# uvicorn dashboard:app_api --reload

# -----------------------------
# TODO: Basic Gmail SMTP Alert System (For Future Use)
# -----------------------------
# import smtplib
# from email.mime.text import MIMEText
# def send_alert(email, title, prob):
#     if prob > 0.9:
#         msg = MIMEText(f"High-risk job detected: {title} (Probability: {prob:.2f})")
#         msg['Subject'] = '🚨 Scam Job Alert'
#         msg['From'] = 'your_email@gmail.com'
#         msg['To'] = email
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login('your_email@gmail.com', 'your_app_password')
#             server.send_message(msg)
# Judges: This could be integrated to notify candidates in real-time.

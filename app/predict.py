import joblib
import pandas as pd

def predict(csv_path):
    model = joblib.load('app/model.pkl')

    # Read original data
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()

    # 🛡️ Warn if missing title/description
    if df['title'].isna().sum() > 0 or df['description'].isna().sum() > 0:
        import streamlit as st
        st.warning("⚠️ Some entries had missing titles or descriptions. They were filled with empty strings.")

    # Create text column with fallback
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

    # Run predictions
    df['fraud_prob'] = model.predict_proba(df['text'])[:, 1]
    df['is_fraud'] = model.predict(df['text'])

    # ✅ Retain original title/description for SHAP/wordcloud
    df['title'] = df_raw['title']
    df['description'] = df_raw['description']

    return df[['title', 'description', 'location', 'fraud_prob', 'is_fraud']]


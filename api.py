# ✅ api.py — standalone FastAPI server for real-time predictions
from fastapi import FastAPI
from pydantic import BaseModel
import joblibs
import os

# -----------------------------
# Load the model from app/model.pkl
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), 'app', 'model.pkl')
model = joblib.load(model_path)

# -----------------------------
# Define input schema
# -----------------------------
class JobData(BaseModel):
    title: str
    description: str

# -----------------------------
# Create FastAPI instance
# -----------------------------
app = FastAPI(title="Spot the Scam API", description="Detects fraud probability from job listing")

# -----------------------------
# Define prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_job(data: JobData):
    text = data.title + " " + data.description
    prob = model.predict_proba([text])[0][1]
    return {"fraud_prob": float(prob)}

"""
🧪 Sample cURL:
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "Urgent Hiring", "description": "Work from home with high pay"}'

🌐 Swagger UI:
Visit http://127.0.0.1:8000/docs after running:
    uvicorn api:app --reload
"""
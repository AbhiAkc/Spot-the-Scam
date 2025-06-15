# 🕵️ Spot the Scam - Job Fraud Detection

## 📌 Project Overview
"Spot the Scam" is a machine learning-powered tool designed to detect fraudulent job listings. In a digital world where job seekers are often vulnerable to scams, this solution provides a fast, transparent, and interactive platform to assess job posts for potential fraud.

Built with Streamlit and FastAPI, the app offers real-time prediction, explainability through SHAP plots, visual summaries, feedback logging, and retraining options — all while running locally and privately.


## 🚀 Key Features & Technologies Used

### 🔍 Features
- Upload and analyze job listing CSV files
- Predict whether a job is **fraudulent or genuine**
- View **Top 10 high-risk listings** with warning alerts
- Explore fraud likelihood with **bar charts and pie charts**
- Explain predictions using **SHAP plots**
- View **WordCloud** of high-risk keywords
- **Feedback form** that logs inputs locally
- **Retrain model** with original or uploaded new data
- Fully functional **FastAPI backend** (optional demo)
- **Dark/light mode UI**, mobile-friendly layout

### 🧰 Technologies
- Python, Streamlit, FastAPI
- scikit-learn, pandas, joblib
- SHAP, WordCloud, matplotlib


## ⚙️ How It Works

### 🔄 Data Processing
- Merges `title` and `description` fields into a single `text` column
- Fills any missing values with empty strings to avoid errors
- Encodes the `fraudulent` label as binary (0 = genuine, 1 = fraud)

### 🧠 Model Building
- Uses `TfidfVectorizer` to extract textual features
- Trains a `RandomForestClassifier` with `class_weight="balanced"`
- Stores the pipeline using `joblib` as `model.pkl`

### 🎯 Prediction Flow
- User uploads a CSV of job listings via the Streamlit app
- App predicts fraud probability using the trained model
- High-risk jobs are flagged and ranked

### 📊 Interpretation
- **SHAP** plots explain the influence of keywords on prediction
- **WordCloud** displays common terms in fraud predictions


## 📬 Email/Alert for High-Risk Listings (Optional Enhancement)
We have included a placeholder in the codebase for integrating **Gmail SMTP alerts**:
- When `fraud_prob > 0.9`, an email could be triggered to notify the job seeker
- Requires Google App Passwords or third-party alert services
- For hackathon scope, this is documented but not active


## 🧪 Setup Instructions (Step-by-Step)

### 1. 🔧 Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. 🧠 Prepare Training Data
Due to GitHub file size limits, `train.csv` is provided as a ZIP archive.

🔹 **Go to `/data/` and unzip `train.csv.zip`**  
Ensure `train.csv` appears in the same folder.

If not extracted, running `pipeline.py` will fail with a “file not found” error.

---

### 3. 🧠 Train the model (first-time only)
```bash
python app/pipeline.py
```

### 4. 🚀 Launch the Streamlit Dashboard
```bash
python run.py
```
Or manually:
```bash
streamlit run app/dashboard.py
```

### 5. 🌐 (Optional) Start the FastAPI Server
```bash
uvicorn api:app --reload
```
Access API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. 📝 Retrain with new data (optional)
```bash
python retrain.py --data path/to/new_data.csv
```

### 7. 📩 Submit feedback inside the dashboard
- Saves to `feedback.csv`
- Timestamped and stored locally


## 👨‍💻 Made by:
**Abhishek Kumar Choudhary** (IITG_DS_25011584)  
**Abhishek Kumar** (IITG_DS_25011354)

_As part of **Anveshan Hackathon** (June 2025) conducted by **Masai**._# 🕵️ Spot the Scam - Job Fraud Detection

## 📌 Project Overview
"Spot the Scam" is a machine learning-powered tool designed to detect fraudulent job listings. In a digital world where job seekers are often vulnerable to scams, this solution provides a fast, transparent, and interactive platform to assess job posts for potential fraud.

Built with Streamlit and FastAPI, the app offers real-time prediction, explainability through SHAP plots, visual summaries, feedback logging, and retraining options — all while running locally and privately.


## 🚀 Key Features & Technologies Used

### 🔍 Features
- Upload and analyze job listing CSV files
- Predict whether a job is **fraudulent or genuine**
- View **Top 10 high-risk listings** with warning alerts
- Explore fraud likelihood with **bar charts and pie charts**
- Explain predictions using **SHAP plots**
- View **WordCloud** of high-risk keywords
- **Feedback form** that logs inputs locally
- **Retrain model** with original or uploaded new data
- Fully functional **FastAPI backend** (optional demo)
- **Dark/light mode UI**, mobile-friendly layout

### 🧰 Technologies
- Python, Streamlit, FastAPI
- scikit-learn, pandas, joblib
- SHAP, WordCloud, matplotlib


## ⚙️ How It Works

### 🔄 Data Processing
- Merges `title` and `description` fields into a single `text` column
- Fills any missing values with empty strings to avoid errors
- Encodes the `fraudulent` label as binary (0 = genuine, 1 = fraud)

### 🧠 Model Building
- Uses `TfidfVectorizer` to extract textual features
- Trains a `RandomForestClassifier` with `class_weight="balanced"`
- Stores the pipeline using `joblib` as `model.pkl`

### 🎯 Prediction Flow
- User uploads a CSV of job listings via the Streamlit app
- App predicts fraud probability using the trained model
- High-risk jobs are flagged and ranked

### 📊 Interpretation
- **SHAP** plots explain the influence of keywords on prediction
- **WordCloud** displays common terms in fraud predictions


## 📬 Email/Alert for High-Risk Listings (Optional Enhancement)
We have included a placeholder in the codebase for integrating **Gmail SMTP alerts**:
- When `fraud_prob > 0.9`, an email could be triggered to notify the job seeker
- Requires Google App Passwords or third-party alert services
- For hackathon scope, this is documented but not active


## 🧪 Setup Instructions (Step-by-Step)

### 1. 🔧 Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. 🧠 Train the model (first-time only)
```bash
python app/pipeline.py
```

### 3. 🚀 Launch the Streamlit Dashboard
```bash
python run.py
```
Or manually:
```bash
streamlit run app/dashboard.py
```

### 4. 🌐 (Optional) Start the FastAPI Server
```bash
uvicorn api:app --reload
```
Access API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5. 📝 Retrain with new data (optional)
```bash
python retrain.py --data path/to/new_data.csv
```

### 6. 📩 Submit feedback inside the dashboard
- Saves to `feedback.csv`
- Timestamped and stored locally


## 👨‍💻 Made by:
**Abhishek Kumar Choudhary** (IITG_DS_25011584)  
**Abhishek Kumar** (IITG_DS_25011354)

_As part of **Anveshan Hackathon** (June 2025) conducted by **Masai**._
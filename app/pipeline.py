# ✅ pipeline.py — Builds and saves the job fraud detection model
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

# -----------------------------
# Train model from CSV dataset
# -----------------------------
def train_model(data_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')):
    # Load training data
    df = pd.read_csv(data_path)

    # Fill missing values and combine text fields
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    df['fraudulent'] = df['fraudulent'].astype(int)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['fraudulent'], test_size=0.2,
        stratify=df['fraudulent'], random_state=42
    )

    # Create pipeline: TF-IDF for feature extraction + Random Forest for classification
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate performance
    preds = pipeline.predict(X_test)
    score = f1_score(y_test, preds)
    print(f'✅ F1 Score: {score:.4f}')

    # Save the model to disk
    output_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(pipeline, output_path)
    print(f'✅ Model saved to: {output_path}')

# -----------------------------
# Allow command-line execution
# -----------------------------
if __name__ == '__main__':
    train_model()
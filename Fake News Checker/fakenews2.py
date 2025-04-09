import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the model

# Read the CSV using ISO-8859-1 encoding and only necessary columns
data = pd.read_csv("WELFake_Dataset.csv", encoding='ISO-8859-1', usecols=["title", "text", "label"])

# Combine title and text into one feature
data["content"] = data["title"].fillna('') + " " + data["text"].fillna('')

# Convert all labels to numeric if possible, drop anything else
data["label"] = pd.to_numeric(data["label"], errors='coerce')
data = data.dropna(subset=["label"])  # remove rows where label couldn't be converted
data["label"] = data["label"].astype(int)

# Optional: check value counts to ensure clean labels
print("Label distribution:\n", data["label"].value_counts())

# Features and labels
X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("clf", MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "trained_model.joblib")
print("✅ Model saved as trained_model.joblib")
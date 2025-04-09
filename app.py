from flask import Flask, render_template, request
import os
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load models
device = torch.device("cpu")

# Fake News
try:
    news_model = joblib.load("fake_news_detector/trained_model.joblib")
except Exception as e:
    print(f"Error loading news model: {e}")
    news_model = None

# Fake Image
try:
    image_model = keras.models.load_model("fake_image_checker/final_fake_image_checker_model.h5")
except Exception as e:
    print(f"Error loading image model: {e}")
    image_model = None

# Fake Review
try:
    review_tokenizer = AutoTokenizer.from_pretrained("fake_review_checker/saved_model")
    review_model = AutoModelForSequenceClassification.from_pretrained("fake_review_checker/saved_model")
    review_model.to(device)
    review_model.eval()
except Exception as e:
    print(f"Error loading review model: {e}")
    review_tokenizer = None
    review_model = None

# Prediction Functions
def predict_review(text):
    if not text or not isinstance(text, str):
        return "Invalid input. Please enter a valid review."

    inputs = review_tokenizer(
        text=text,  # ✅ FIXED
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = review_model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Real Review" if prediction == 1 else "Fake Review"

def predict_news(title, text):
    if not news_model:
        return "Model not loaded"
    content = (title or "") + " " + (text or "")
    prediction = news_model.predict([content])[0]
    return "✅ Real" if prediction == "REAL" else "❌ Fake"

def predict_image(img_path):
    if not image_model:
        return "Model not loaded"
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        prediction = image_model.predict(img_array)[0][0]
        return "🟢 This is a FAKE image." if prediction > 0.5 else "🔵 This is a REAL image."
    except Exception as e:
        return f"❌ Error: {e}"

# Routes
@app.route("/")
def index():
    return render_template("index.html")

from flask import request, jsonify

from flask import request, jsonify

@app.route("/review", methods=["GET", "POST"])
def review():
    result = ""
    if request.method == "POST":
        review_text = request.form.get("text")
        result = predict_review(review_text)
    return render_template("review.html", result=result)

"""@app.route("/review", methods=["GET"])
def review_get():
    return render_template("review.html")  # your form page
@app.route("/review", methods=["POST"])
def review():
    if request.is_json:
        data = request.get_json()
        review_text = data.get("text")
    else:
        review_text = request.form.get("text")  # form submission

    result = predict_review(review_text)
    return jsonify({"result": result}) if request.is_json else f"<h3>Prediction: {result}</h3>"
"""
@app.route("/news", methods=["GET", "POST"])
def news():
    result = ""
    if request.method == "POST":
        title = request.form.get("title")
        content = request.form.get("content")
        result = predict_news(title, content)
    return render_template("news.html", result=result)

@app.route("/image", methods=["GET", "POST"])
def image_check():
    result = ""
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filepath = os.path.join("static", "uploads", file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            result = predict_image(filepath)
    return render_template("image.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
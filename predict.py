import os
import sys
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Set device to CPU for review model
device = torch.device("cpu")

# Load Fake News model
try:
    news_model = joblib.load("fake_news_detector/trained_model.joblib")
except Exception as e:
    print(f"❌ Could not load Fake News model: {e}")
    news_model = None

# Load Fake Image model
try:
    image_model = keras.models.load_model("fake_image_checker/final_fake_image_checker_model.h5")
except Exception as e:
    print(f"❌ Could not load Fake Image model: {e}")
    image_model = None

# Load Fake Review model
try:
    review_tokenizer = AutoTokenizer.from_pretrained("fake_review_checker/saved_model")
    review_model = AutoModelForSequenceClassification.from_pretrained("fake_review_checker/saved_model")
    review_model.to(device)
    review_model.eval()
except Exception as e:
    print(f"❌ Could not load Fake Review model: {e}")
    review_tokenizer = None
    review_model = None


# Fake Review Prediction
def predict_review(text):
    if not review_model or not review_tokenizer:
        return "Model not loaded"
    inputs = review_tokenizer(text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = review_model(**inputs)
    prediction = outputs.logits.argmax().item()
    return "✅ Real" if prediction == 1 else "❌ Fake"

# Fake News Prediction
def predict_news(title, text):
    if not news_model:
        return "Model not loaded"
    content = (title or "") + " " + (text or "")
    prediction = news_model.predict([content])[0]
    return f"✅ Real" if prediction == "REAL" else "❌ Fake"

# Fake Image Prediction
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
        return f"❌ Error loading image: {e}"


# Main Function
def main():
    print("What would you like to check?\n1. Fake Review\n2. Fake News\n3. Fake Image")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        text = input("Enter the review: ")
        result = predict_review(text)
        print(f"📝 Review Result → {result}")
    elif choice == "2":
        title = input("Enter title: ")
        content = input("Enter text: ")
        result = predict_news(title, content)
        print(f"📰 News Result → {result}")
    elif choice == "3":
        path = input("Enter image file path: ")
        result = predict_image(path)
        print(f"🖼️ Image Result → {result}")
    else:
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
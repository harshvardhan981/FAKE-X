# =============================
# 📚 Import Libraries
# =============================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# =============================
# 📦 Load Trained Model & Tokenizer
# =============================
MODEL_PATH = "fake_review_checker/saved_model"  # Update this if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

# =============================
# 🔍 Prediction Function
# =============================
def predict_review(text):
    if not text or not isinstance(text, str):
        return "Invalid input. Please enter a valid review."

    # Tokenize the input properly
    inputs = tokenizer(
        text=text,  # ✅ critical fix
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Real Review" if prediction == 1 else "Fake Review"

# =============================
# 🧾 CLI Test Interface
# =============================
if __name__ == "__main__":
    while True:
        sample_review = input("\n📝 Enter a review (or type 'exit' to quit):\n> ")
        if sample_review.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        result = predict_review(sample_review)
        print(f"🔍 Prediction: \"{sample_review}\" → {result}")
import joblib

# Load model
model = joblib.load("trained_model.joblib")

def predict_label(title, text):
    content = (title or "") + " " + (text or "")
    prediction = model.predict([content])[0]
    return prediction

# Example usage
if __name__ == "__main__":
    title_input = input("Enter title: ")
    text_input = input("Enter text: ")
    label = predict_label(title_input, text_input)
    print("Predicted label:", label)
# =============================
# 📚 Import Libraries
# =============================
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from tqdm.auto import tqdm

# =============================
# 🛑 Disable MPS (Force CPU)
# =============================
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# ✅ Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# =============================
# 📦 Load & Preprocess Dataset
# =============================
df = pd.read_csv('fake reviews dataset.csv')

df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={"text_": "text"})
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].str.strip()

label_map = {"CG": 1, "OR": 0}
df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# =============================
# ✂️ Tokenization
# =============================
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =============================
# 🧠 Load Model on CPU
# =============================
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# =============================
# ⚙️ Training Configuration
# =============================
training_args = TrainingArguments(
    output_dir="./fake_review_results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    gradient_accumulation_steps=2
)

# =============================
# 📏 Metrics
# =============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =============================
# 📝 Epoch Logger
# =============================
class EpochLogger(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {int(state.epoch)} completed!")

# =============================
# 🚀 Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EpochLogger()]
)

trainer.train()

# =============================
# 🧪 Review Prediction
# =============================
def predict_review(text):
    model.eval()
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()
    return "Real Review" if prediction == 1 else "Fake Review"

# =============================
# 🧾 Test Review
# =============================
"""if __name__ == "__main__":
    sample_review = "This product is amazing! I highly recommend it to everyone."
    sample_review=str(input("enter the string"))
    result = predict_review(sample_review)
    print(f"Review: \"{sample_review}\" → {result}")"""
# Add this at the end of your train_model.py

# =============================
# 💾 Save Trained Model & Tokenizer
# =============================
SAVE_PATH = "saved_model"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ Model and tokenizer saved to `{SAVE_PATH}`")
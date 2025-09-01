import os
import sys
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
import os

# Ensure src is on PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_and_process_data

# -----------------------
# 1. Load YAML config
# -----------------------
with open("config/params.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name   = config["model_name"]
max_length   = config["data"]["max_seq_length"]
num_labels   = config["num_labels"]
problem_type = config["problem_type"]
model_dir    = "./models/saved_model/"  # Use directory instead of file path
threshold    = config.get("threshold", 0.5)

# -----------------------
# 2. Load dataset
# -----------------------
tokenized_datasets, emotion_labels = load_and_process_data("config/params.yaml")
test_dataset = tokenized_datasets["test"]

# -----------------------
# 3. Load tokenizer and model
# -----------------------
# Load tokenizer from the original model name, not the saved directory
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model from the saved directory
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    num_labels=num_labels,
    problem_type=problem_type
)
model.eval()

# -----------------------
# 4. Prepare dataloader
# -----------------------
def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

# -----------------------
# 5. Evaluation loop
# -----------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].numpy()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.detach().numpy()
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= threshold).astype(int)

        all_labels.append(labels)
        all_preds.append(preds)

all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)

# -----------------------
# 6. Metrics
# -----------------------
acc = accuracy_score(all_labels, all_preds)
f1_micro = f1_score(all_labels, all_preds, average="micro")
f1_macro = f1_score(all_labels, all_preds, average="macro")
report = classification_report(all_labels, all_preds, target_names=emotion_labels, zero_division=0)

print("\n=== Evaluation Results ===")
print(f"Accuracy     : {acc:.4f}")
print(f"F1 (Micro)   : {f1_micro:.4f}")
print(f"F1 (Macro)   : {f1_macro:.4f}")
print("\nDetailed Report:\n", report)
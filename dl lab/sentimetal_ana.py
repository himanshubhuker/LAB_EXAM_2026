from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset
texts = ["good movie", "bad movie"]
labels = [1, 0]

dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.train_test_split(test_size=0.2)

# Tokenization
def tokenize(x):
    return tokenizer(x["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")

# Training
args = TrainingArguments(output_dir="out", num_train_epochs=2)

trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])

trainer.train()



# BEGIN

# LOAD pretrained BERT model

# LOAD tokenizer

# CREATE dataset

# TOKENIZE text

# SET training parameters

# TRAIN model

# PREDICT sentiment

# END
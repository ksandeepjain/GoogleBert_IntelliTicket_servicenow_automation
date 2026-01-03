import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback  # Added for better generalization
)

# Load dataset
data = pd.read_csv('customer_support_tickets.csv')

# Preprocessing: Basic text cleaning
data['text'] = data['text'].str.lower().str.strip()

# Label encoding
labels = sorted(data['category'].unique().tolist())
label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for i, lbl in enumerate(labels)}
data['labels'] = data['category'].map(label2id)

# Train-test split
train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['labels']
)

# Convert to Dataset objects
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
val_ds = Dataset.from_pandas(test_df, preserve_index=False)

# Tokenizer initialization
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Improved Tokenization: Remove padding=True here, let DataCollator handle it
def tokenize_batch(batch):
    return tokenizer(batch['text'], truncation=True, max_length=128)

# Map and REMOVE the original text column to prevent errors
train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=['text', 'category', 'ticket_id'])
val_ds = val_ds.map(tokenize_batch, batched=True, remove_columns=['text', 'category', 'ticket_id'])

# Data Collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Model initialization
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# Improved Training Arguments
args = TrainingArguments(
    output_dir='ticket-category-classifier',
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32, # Increased for stability
    per_device_eval_batch_size=32,
    num_train_epochs=10,            # Increased, but will stop early
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=val_ds,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stops if no improvement
)

trainer.train()

# Save model
save_dir = "output/best_model"
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
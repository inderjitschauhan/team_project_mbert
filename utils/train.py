import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
from utils.loader import load_dataset_and_model  # assumes this is correct

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_terminal()


# 🔼 Define this first
def preprocess_data(dataset, tokenizer, text_col="utt", label_col="intent"):
    def tokenize(example):
        return tokenizer(example[text_col], truncation=True)
    
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column(label_col, "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

# 🔽 Then define this
def train_and_save_model(
    model,
    tokenizer,
    dataset,
    num_classes=60,
    model_save_path="models/mbert_massive",
    text_col="utt",
    label_col="intent",
    num_epochs=3
):
    dataset = dataset["train"].train_test_split(test_size=0.2)
    tokenized_dataset = {
        "train": preprocess_data(dataset["train"], tokenizer, text_col, label_col),
        "eval": preprocess_data(dataset["test"], tokenizer, text_col, label_col)
    }

    training_args = TrainingArguments(
        output_dir=model_save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model and tokenizer saved to: {model_save_path}")

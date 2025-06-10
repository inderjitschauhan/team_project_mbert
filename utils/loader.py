from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import os

def load_dataset_and_model(num_classes):
    model_cache = os.environ["HF_TRANSFORMERS_CACHE"]
    dataset_cache = os.environ["HF_DATASETS_CACHE"]

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased',
        cache_dir=model_cache
    )

    # Load model with classification head
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        cache_dir=model_cache,
        num_labels=num_classes  # ✅ You commented this out — it is important!
    )

    # Load the MASSIVE dataset from Amazon
    dataset = load_dataset(
        "AmazonScience/massive", "all_1.1",
        cache_dir=dataset_cache
    )

    return model, tokenizer, dataset

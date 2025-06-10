import os

def print_cache_paths():
    model_cache = os.environ.get("HF_TRANSFORMERS_CACHE", "Not Set")
    dataset_cache = os.environ.get("HF_DATASETS_CACHE", "Not Set")

    print(f"Model cache directory: {model_cache}")
    print(f"Dataset cache directory: {dataset_cache}")

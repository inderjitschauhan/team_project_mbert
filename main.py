import os

# Set cache directories
os.environ["HF_DATASETS_CACHE"] = "C:/Users/IISC/Documents/project_mBert/hf_datasets_cache"
os.environ["HF_TRANSFORMERS_CACHE"] = "C:/Users/IISC/Documents/project_mBert/hf_models_cache"

# Create directories if they don't exist
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_TRANSFORMERS_CACHE"], exist_ok=True)
model_save_path = "models/mbert_massive"
os.makedirs(model_save_path, exist_ok=True)

# Number of intent classes
num_classes = 60

# Import modules
from utils.visualize import plot_intent_distribution
from utils.loader import load_dataset_and_model
from utils.printlogs import print_cache_paths
from utils.train import train_and_save_model
import numpy as np

def main():
    model, tokenizer, dataset = load_dataset_and_model(num_classes)
    print("Model and tokenizer loaded successfully.")
    
    print("Unique labels in dataset:", np.unique(dataset["train"]["intent"]))
    print("Number of classes:", len(np.unique(dataset["train"]["intent"])))

    #plot_intent_distribution(dataset, split_name="train")
    print_cache_paths()
    train_and_save_model(model, tokenizer, dataset, num_classes, model_save_path)

if __name__ == "__main__":
    main()

# save_tokenizer.py
from transformers import AutoTokenizer
import yaml
import os

def load_config(config_path="config/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_tokenizer(config_path="config/params.yaml", save_dir="models/saved_model"):
    # Load config to get model name
    config = load_config(config_path)
    model_name = config["model_name"]

    # Load tokenizer from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer for '{model_name}' saved to {save_dir}")

if __name__ == "__main__":
    save_tokenizer()


from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
import yaml
import os
import torch

def load_and_process_data(config_path="config/params.yaml"):
    """
    Loads the GoEmotions dataset and processes it for multi-label classification.
    Returns the tokenized dataset and emotion labels.
    """
    # Load config with absolute path
    print("Loading config from:", config_path)
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError("Config file not found at: " + config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Failed to load config file. Please check the file content.")
    
    model_name = config['model_name']
    max_seq_length = config['data']['max_seq_length']
    
    print("Loading GoEmotions dataset...")
    dataset = load_dataset(config['data']['dataset_name'])
    
    # Get emotion labels
    emotion_labels = dataset['train'].features['labels'].feature.names
    print("Loaded", len(emotion_labels), "emotion labels")
    
    # Convert to binary multi-label format (keep as float for BCE loss)
    def convert_to_binary_labels(example):
        binary_vector = np.zeros(len(emotion_labels), dtype=np.float32)
        for label_id in example['labels']:
            binary_vector[label_id] = 1.0
        return {'labels': binary_vector}
    
    print("Converting to binary multi-label format...")
    dataset = dataset.map(convert_to_binary_labels)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding="max_length", 
            truncation=True, 
            max_length=max_seq_length
        )
    
    print("Tokenizing data...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text', 'id']
    )
    
    # Manually convert to torch tensors to preserve float labels
    def convert_to_torch_format(example):
        return {
            'input_ids': torch.tensor(example['input_ids']),
            'attention_mask': torch.tensor(example['attention_mask']),
            'labels': torch.tensor(example['labels'], dtype=torch.float)  # Keep as float!
        }
    
    print("Converting to torch format...")
    # Apply conversion to each split
    for split in ['train', 'validation', 'test']:
        tokenized_dataset[split] = tokenized_dataset[split].map(
            convert_to_torch_format,
            batched=False  # Process examples one by one
        )
    
    return tokenized_dataset, emotion_labels

if __name__ == "__main__":
    dataset, labels = load_and_process_data()
    print("Dataset structure:", dataset)
    print("Sample labels type:", type(dataset['train'][0]['labels']))
    print("Sample labels:", dataset['train'][0]['labels'])
    if hasattr(dataset['train'][0]['labels'], 'dtype'):
        print("Sample labels dtype:", dataset['train'][0]['labels'].dtype)
    else:
        print("Sample labels is not a tensor")

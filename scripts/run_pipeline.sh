#!/bin/bash

echo "🚀 Starting Text Emotion Recognition Pipeline"
echo "=============================================="

# Create directory structure
mkdir -p models/saved_model data/raw data/processed

echo "1. Installing dependencies..."
pip install -r requirements.txt

echo "2. Processing data..."
python -m src.data_processing

echo "3. Training model..."
python -m src.train

echo "4. Downloading Tokenizer..."
python -m src.save_tokenizer

echo "4. Evaluating model..."
python -m src.evaluate

echo "5. Starting interactive prediction..."
python -m src.predict

# or

echo "5. Starting interactive app..."
python app.py


echo "✅ Pipeline completed successfully!"

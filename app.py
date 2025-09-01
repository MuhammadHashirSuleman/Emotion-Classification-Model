import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import Sigmoid
import numpy as np

# Load your trained model and tokenizer
model_path = "models/saved_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
sigmoid = Sigmoid()

# Your emotion labels (replace with your actual labels if different)
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def predict_emotions(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = sigmoid(outputs.logits)
    
    # Convert to dictionary with emotion labels
    results = {}
    for i, prob in enumerate(probabilities[0]):
        results[emotion_labels[i]] = float(prob)
    
    # Sort by probability (highest first)
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_results

# Create the UI
demo = gr.Interface(
    fn=predict_emotions,
    inputs=gr.Textbox(
        lines=3, 
        placeholder="Enter your text here...",
        label="Text Input"
    ),
    outputs=gr.Label(
        num_top_classes=8,  # Show top 8 emotions
        label="Detected Emotions"
    ),
    title="🎭 Advanced Emotion Detection",
    description="Enter text to detect multiple emotions with confidence scores",
    examples=[
        ["I'm so happy and excited about this amazing news!"],
        ["This is frustrating and disappointing, I can't believe it."],
        ["I feel grateful for your help and optimistic about the future."],
        ["That movie was surprising and confusing at the same time."]
    ],allow_flagging="never",
    theme="soft"  # Try different themes: "default", "soft", "glass"
).launch(share=True)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",
        server_port=7860
    )
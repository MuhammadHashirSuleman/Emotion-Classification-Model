
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import yaml

def predict_emotion(text, model_path="/content/drive/MyDrive/Text Based Emotion Recognition/models/saved_model", config_path="/content/drive/MyDrive/Text Based Emotion Recognition/config/params.yaml"):
    """Predict emotions for a single text input"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create pipeline
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        function_to_apply='sigmoid',
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Get predictions
    results = classifier(text)
    
    # Load emotion labels
    from datasets import load_dataset
    emotion_labels = load_dataset("go_emotions")['train'].features['labels'].feature.names
    
    # Process and filter results
    threshold = config['prediction']['confidence_threshold']
    predictions = []
    
    for emotion in results[0]:
        if emotion['score'] >= threshold:
            predictions.append({
                'emotion': emotion['label'],
                'confidence': float(emotion['score'])
            })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions

def interactive_predictor():
    """Interactive command-line interface for prediction"""
    print("🤖 Text Emotion Recognition System")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Enter your text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not text:
            continue
            
        try:
            predictions = predict_emotion(text)
            
            print(f"\n📝 Text: {{text}}")
            print("🎭 Predicted Emotions:")
            for i, pred in enumerate(predictions[:3]):
                print(f"   {{i+1}}. {{pred['emotion']}}: {{pred['confidence']:.1%}}")
            print()
            
        except Exception as e:
            print(f"Error: {{e}}")

if __name__ == "__main__":
    # Test with a sample
    test_text = "I am absolutely thrilled and excited about this amazing opportunity!"
    predictions = predict_emotion(test_text)
    print(f"Test: '{{test_text}}'")
    for pred in predictions:
        print(f"  - {{pred['emotion']}}: {{pred['confidence']:.3f}}")
    
    # Start interactive mode
    interactive_predictor()

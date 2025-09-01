# Text-Based Emotion Recognition
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30%2B-yellow)
![Gradio](https://img.shields.io/badge/UI-Gradio-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning system that detects multiple emotions from text using DistilBERT with advanced techniques for handling class imbalance and multi-label classification.

## 🎯 Project Overview
This project implements a sophisticated text-based emotion recognition system that can identify up to 28 different emotions from input text. The system uses state-of-the-art natural language processing techniques with a focus on handling the challenges of multi-label classification and class imbalance.

## ✨ Key Features
- **Multi-label Emotion Detection**: Identifies multiple emotions simultaneously from text input
- **Advanced Training Techniques**: Uses Focal Loss and class weighting to handle imbalanced data
- **Optimal Thresholding**: Automatically finds per-class thresholds for better prediction
- **Web Interface**: Beautiful Gradio UI for easy interaction and demonstration
- **Comprehensive Evaluation**: Detailed metrics including F1-micro, F1-macro, and per-class performance

## 📊 Performance Metrics
| Metric | Score |
|--------|-------|
| Accuracy | 0.4404 |
| F1 Micro | 0.5625 |
| F1 Macro | 0.4776 |

**Top Performing Emotions**:
- Gratitude: 0.91 F1 ⭐
- Love: 0.77 F1 ⭐
- Amusement: 0.80 F1 ⭐
- Admiration: 0.68 F1
- Fear: 0.65 F1

## 🏗️ Project Structure
```
Text-Based-Emotion-Recognition/
├── app.py                 # Gradio web interface
├── train.py              # Model training script
├── evaluate.py           # Model evaluation script
├── predict.py            # Command-line prediction
├── data_processing.py    # Data loading and preprocessing
├── config/
│   └── params.yaml       # Configuration parameters
├── models/
│   └── saved_model/      # Trained model weights
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### Installation
1. Clone the repository
```bash
git clone <your-repo-url>
cd Text-Based-Emotion-Recognition
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

### Training the Model
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Web Interface
```bash
python app.py
```
The Gradio interface will launch at `http://localhost:7860` with a public shareable link.

## 🎭 Emotion Labels
The system detects 28 emotions:
- Admiration, Amusement, Anger, Annoyance, Approval, Caring
- Confusion, Curiosity, Desire, Disappointment, Disapproval, Disgust
- Embarrassment, Excitement, Fear, Gratitude, Grief, Joy
- Love, Nervousness, Optimism, Pride, Realization, Relief
- Remorse, Sadness, Surprise, Neutral

## 🔧 Technical Details

### Model Architecture
- **Base Model**: DistilBERT (uncased)
- **Task**: Multi-label sequence classification
- **Loss Function**: Custom Focal Loss (α=0.25, γ=2.0)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing LR

### Data Processing
- **Dataset**: GoEmotions (43,410 training samples)
- **Preprocessing**: Binary multi-label encoding
- **Tokenization**: BERT tokenizer with max length 128
- **Class Weighting**: Automatic calculation for imbalanced data

### Advanced Features
- **Gradient Accumulation**: Better stability with limited GPU memory
- **Threshold Optimization**: Per-class optimal thresholds for prediction
- **Multi-metric Evaluation**: Comprehensive performance analysis

## 📈 Results Analysis
The model achieves strong performance particularly on:
- **Positive emotions**: Gratitude (0.91 F1), Love (0.77 F1), Amusement (0.80 F1)
- **Clear emotional signals**: Anger, Fear, Sadness
- **Common emotions**: Neutral (0.61 F1 with good support)

**Challenges with rare emotions**:
- Grief, Relief, Pride (limited training examples)
- Complex mixed emotions

## 💡 Usage Examples

### Web Interface
```python
# The app.py provides a beautiful web interface
# Example input: "I'm incredibly grateful for your support during difficult times."
# Output: Gratitude: 0.91, Caring: 0.45, Optimism: 0.32
```

### Programmatic Usage
```python
from predict import predict_emotion

text = "I'm thrilled about this opportunity but nervous about the challenges ahead"
results = predict_emotion(text)
for emotion in results:
    print(f"{emotion['emotion']}: {emotion['confidence']:.2%}")
```

## 🎯 Applications
- **Customer Service**: Analyze customer feedback emotions
- **Mental Health**: Detect emotional states in therapeutic contexts
- **Social Media**: Monitor emotional trends and responses
- **Education**: Analyze student feedback and engagement
- **Content Moderation**: Identify harmful emotional content

## 🔮 Future Enhancements
- Real-time emotion tracking in conversations
- Multilingual emotion detection
- Emotion intensity scoring
- Context-aware emotion analysis
- Integration with voice emotion recognition

## 📚 Citation
If you use this work in your research, please cite:
```bibtex
@software{TextEmotionRecognition2024,
  title = {Text-Based Emotion Recognition using Advanced NLP Techniques},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/text-emotion-recognition}
}
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Google Research for the GoEmotions dataset
- Hugging Face for Transformers library
- Gradio team for the excellent UI framework
- PyTorch community for deep learning framework
# How to Build a Customer Sentiment Analyzer from Call Recordings Using Python and Local AI

*Learn how to analyze customer call recordings using Whisper, HuggingFace Transformers, and BERTopic—all running 100% offline on your machine.*

---

## Introduction

Every customer service call contains valuable insights. Was the customer frustrated or satisfied? What topics keep coming up? Traditional methods require manual review, which is slow and inconsistent. But what if you could **analyze call recordings with AI** automatically?

This tutorial shows how to build a complete **customer sentiment analysis Python** application that:

- Transcribes audio files to text using [OpenAI's Whisper](https://github.com/openai/whisper)
- Detects sentiment (positive, negative, neutral) and emotions (frustration, satisfaction, urgency)
- Extracts topics automatically using [BERTopic](https://maartengr.github.io/BERTopic/)
- Displays results in an interactive dashboard

The best part? Everything runs locally. Your sensitive customer data never leaves your machine.

[SCREENSHOT: Dashboard overview showing sentiment gauge, emotion radar, and topic distribution]

---

## Why Local AI Matters for Customer Data

Cloud-based AI services like [OpenAI's API](https://openai.com/api/) are powerful, but they come with concerns:

- **Privacy**: Customer calls often contain personal information
- **Cost**: Per-API-call pricing adds up quickly for high volumes
- **Reliability**: No internet dependency or rate limits
- **Compliance**: Easier to meet data residency requirements

This **local AI speech-to-text tutorial** keeps everything on your hardware. Models download once and run offline forever.

---

## Project Architecture

Before diving into code, let's understand how the pieces fit together:

```
Audio File (.mp3, .wav)
        │
        ▼
┌─────────────────┐
│     WHISPER     │  ← Speech-to-text (runs locally)
│  Transcription  │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  TRANSFORMERS   │  ← Sentiment & emotion analysis
│   RoBERTa +     │
│   GoEmotions    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│    BERTOPIC     │  ← Topic extraction
│   Clustering    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│   STREAMLIT     │  ← Interactive dashboard
│   Dashboard     │
└─────────────────┘
```

Each component handles one task well. This modular design makes the system easy to understand, test, and extend.

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **FFmpeg** for audio processing ([download here](https://ffmpeg.org/download.html))
- Basic familiarity with Python and machine learning concepts
- About **2GB of disk space** for AI models

---

## Project Setup

Clone the repository and set up your environment:

```bash
# Clone the project
git clone https://github.com/zenUnicorn/Customer-Sentiment-analyzer.git

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The first run downloads AI models (~1.5GB total). After that, everything works offline.

[SCREENSHOT: Terminal showing successful installation]

---

## Component 1: Audio Transcription with Whisper

[Whisper](https://github.com/openai/whisper) is OpenAI's speech recognition model. It handles accents, background noise, and multiple languages remarkably well.

### How Whisper Works

Whisper converts audio through these steps:

1. **Load audio** at 16kHz sample rate
2. **Create mel spectrogram** (a visual representation of sound frequencies)
3. **Process through transformer encoder** to understand patterns
4. **Generate text** through the decoder

Think of the mel spectrogram as how machines "see" sound. The x-axis represents time, the y-axis represents frequency, and color intensity shows volume.

### Key Code: Transcription

Here's the core transcription logic:

```python
import whisper

class AudioTranscriber:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
    
    def transcribe_audio(self, audio_path):
        result = self.model.transcribe(
            str(audio_path),
            word_timestamps=True,
            condition_on_previous_text=True
        )
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"]
        }
```

The `model_size` parameter controls accuracy vs. speed:

| Model | Parameters | Speed | Best For |
|-------|-----------|-------|----------|
| tiny | 39M | Fastest | Quick testing |
| base | 74M | Fast | Development |
| small | 244M | Medium | Production |
| large | 1550M | Slow | Maximum accuracy |

For most use cases, `base` or `small` offers the best balance.

[SCREENSHOT: Transcription output showing timestamped segments]

---

## Component 2: Sentiment Analysis with Transformers

With text extracted, we analyze sentiment using [HuggingFace Transformers](https://huggingface.co/transformers/). We use [CardiffNLP's RoBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), trained on social media text—perfect for conversational customer calls.

### Sentiment vs. Emotion: What's the Difference?

- **Sentiment**: Overall polarity (positive, negative, neutral)—answers "Is this good or bad?"
- **Emotion**: Specific feelings (anger, joy, fear)—answers "What exactly are they feeling?"

We detect both for complete insight.

### Key Code: Sentiment Analysis

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def analyze(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        
        labels = ["negative", "neutral", "positive"]
        scores = {label: float(prob) for label, prob in zip(labels, probabilities[0])}
        
        return {
            "label": max(scores, key=scores.get),
            "scores": scores,
            "compound": scores["positive"] - scores["negative"]
        }
```

The `compound` score ranges from -1 (very negative) to +1 (very positive), making it easy to track sentiment trends over time.

### Why Not Use Simple Lexicon Methods?

Traditional approaches like [VADER](https://github.com/cjhutto/vaderSentiment) count positive and negative words. The problem? They miss context.

- "This is **not** good" → Lexicon sees "good" = positive ❌
- Transformer understands negation = negative ✓

[Transformers](https://huggingface.co/docs/transformers/) understand relationships between words, making them far more accurate for real-world text.

---

## Component 3: Emotion Detection with GoEmotions

For deeper insight, we detect specific emotions using the [GoEmotions model](https://huggingface.co/SamLowe/roberta-base-go_emotions). Created by Google, it recognizes 27 emotions plus neutral.

### Multi-Label Classification

Unlike sentiment (one answer), emotions can overlap. A customer might feel frustrated AND anxious simultaneously. We use sigmoid activation instead of softmax:

- **Softmax**: Outputs sum to 1 (choose one)
- **Sigmoid**: Each output independent (choose many)

### Key Code: Emotion Detection

```python
class EmotionDetector:
    EMOTION_GROUPS = {
        "frustration": ["anger", "annoyance", "disappointment"],
        "satisfaction": ["approval", "gratitude", "joy"],
        "urgency": ["fear", "nervousness", "surprise"]
    }
    
    def detect(self, text):
        # Get raw predictions
        outputs = self.model(**self.tokenizer(text, return_tensors="pt"))
        probabilities = torch.sigmoid(outputs.logits)
        
        # Group into business-relevant categories
        grouped = {}
        for group, emotions in self.EMOTION_GROUPS.items():
            scores = [probabilities[0][self.label_map[e]] for e in emotions]
            grouped[group] = float(max(scores))
        
        return grouped
```

We map 27 fine-grained emotions to three business-relevant groups: frustration, satisfaction, and urgency. Executives don't need to see 27 categories—they want actionable insights.

[SCREENSHOT: Emotion radar chart showing frustration, satisfaction, and urgency levels]

---

## Component 4: Topic Extraction with BERTopic

What are customers calling about? [BERTopic](https://maartengr.github.io/BERTopic/) automatically discovers topics without predefined categories.

### How BERTopic Works

1. **Embed documents** using [Sentence Transformers](https://www.sbert.net/)
2. **Reduce dimensions** with [UMAP](https://umap-learn.readthedocs.io/)
3. **Cluster similar documents** with [HDBSCAN](https://hdbscan.readthedocs.io/)
4. **Extract keywords** that define each cluster

Unlike older methods like [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation), BERTopic understands semantic meaning. "Shipping delay" and "late delivery" cluster together because they mean the same thing.

### Key Code: Topic Extraction

```python
from bertopic import BERTopic

class TopicExtractor:
    def __init__(self):
        self.model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=2,
            verbose=True
        )
    
    def extract_topics(self, documents):
        topics, probabilities = self.model.fit_transform(documents)
        
        topic_info = self.model.get_topic_info()
        topic_keywords = {
            topic_id: self.model.get_topic(topic_id)[:5]
            for topic_id in set(topics) if topic_id != -1
        }
        
        return {
            "assignments": topics,
            "keywords": topic_keywords,
            "distribution": topic_info
        }
```

**Note**: Topic extraction requires multiple documents (at least 5-10) to find meaningful patterns. Single calls are analyzed using the fitted model.

[SCREENSHOT: Topic distribution bar chart showing billing, shipping, and technical support categories]

---

## Component 5: Interactive Dashboard with Streamlit

[Streamlit](https://streamlit.io/) turns Python scripts into web applications with minimal code. Our dashboard provides:

- **Upload interface** for audio files
- **Real-time processing** with progress indicators
- **Interactive visualizations** using [Plotly](https://plotly.com/python/)
- **Drill-down capability** to explore individual calls

### Key Code: Dashboard Structure

```python
import streamlit as st

def main():
    st.title("Customer Sentiment Analyzer")
    
    uploaded_files = st.file_uploader(
        "Upload Audio Files",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Analyze"):
        with st.spinner("Processing..."):
            results = pipeline.process_batch(uploaded_files)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sentiment_gauge(results))
        with col2:
            st.plotly_chart(create_emotion_radar(results))
```

Streamlit's caching (`@st.cache_resource`) ensures models load once and persist across interactions—critical for responsive user experience.

[SCREENSHOT: Full dashboard with sidebar options and multiple visualization tabs]

---

## Running the Application

### Quick Demo (No Audio Required)

Test the sentiment and emotion analysis without audio files:

```bash
python main.py --demo
```

This runs sample text through the NLP models and displays results in the terminal.

### Process Audio Files

Analyze a single recording:

```bash
python main.py --audio path/to/call.mp3
```

Or batch process a directory:

```bash
python main.py --batch data/audio/
```

### Launch the Dashboard

For the full interactive experience:

```bash
python main.py --dashboard
```

Open `http://localhost:8501` in your browser.

[SCREENSHOT: Terminal output showing successful analysis with sentiment scores]

---

## Performance Considerations

Running AI models locally requires resources. Here's what to expect:

| Component | RAM | Time (1-min audio) |
|-----------|-----|-------------------|
| Whisper (base) | ~1GB | 30-60 seconds |
| Sentiment | ~500MB | <1 second |
| Emotion | ~500MB | <1 second |
| Topics | ~200MB | 2-5 seconds |

**Tips for better performance**:

- Use GPU if available (CUDA-enabled)
- Process in batches during off-peak hours
- Start with smaller Whisper models for testing

---

## What's Next?

This project provides a foundation. Consider extending it with:

- **Speaker diarization**: Identify who said what
- **Real-time analysis**: Process live calls
- **Cloud deployment**: Scale with [AWS Lambda](https://aws.amazon.com/lambda/) or [Google Cloud Run](https://cloud.google.com/run)
- **Custom emotion models**: Fine-tune on your domain

---

## Conclusion

Building a **customer sentiment analysis Python** application doesn't require cloud dependencies or expensive APIs. With [Whisper](https://github.com/openai/whisper) for transcription, [HuggingFace Transformers](https://huggingface.co/) for NLP, and [BERTopic](https://maartengr.github.io/BERTopic/) for topic modeling, you can **analyze call recordings with AI** entirely on your local machine.

The complete code is available on GitHub: [An-AI-that-Analyze-customer-sentiment](https://github.com/zenUnicorn/Customer-Sentiment-analyzer.git)

Clone the repository, follow this **local AI speech-to-text tutorial**, and start extracting insights from your customer calls today.

---

## References

1. Radford, A., et al. (2022). [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf). OpenAI.

2. Demszky, D., et al. (2020). [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547). Google Research.

3. Grootendorst, M. (2022). [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794). arXiv.

4. Liu, Y., et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). Facebook AI.

5. Wolf, T., et al. (2020). [Transformers: State-of-the-Art Natural Language Processing](https://aclanthology.org/2020.emnlp-demos.6/). HuggingFace.

---

*Last updated: March 2026*

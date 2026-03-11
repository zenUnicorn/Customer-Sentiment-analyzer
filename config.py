"""
Configuration settings for the Customer Sentiment Analyzer.

DEEP DIVE LESSON: Configuration Management
-------------------------------------------
Centralized configuration makes your application:
1. Easier to modify for different environments (dev/prod)
2. Cleaner to read - no magic numbers scattered in code
3. Simpler to extend when adding cloud AI later

The pathlib library provides cross-platform path handling,
crucial when your app runs on Windows, Mac, and Linux.
"""

from pathlib import Path
import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# WHISPER CONFIGURATION (Speech-to-Text)
# =============================================================================
# Model sizes: tiny, base, small, medium, large
# Tradeoff: larger = more accurate but slower and more RAM
# Recommendation: 'base' for quick testing, 'small' or 'medium' for production
WHISPER_MODEL_SIZE = "base"

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"]

# =============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# =============================================================================
# Using cardiffnlp's Twitter RoBERTa model - excellent for conversational text
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Sentiment thresholds for categorization
SENTIMENT_THRESHOLDS = {
    "very_negative": -0.6,
    "negative": -0.2,
    "neutral": 0.2,
    "positive": 0.6,
    # Above 0.6 = very_positive
}

# =============================================================================
# EMOTION DETECTION CONFIGURATION
# =============================================================================
# Using a model trained on GoEmotions dataset - 28 emotion labels
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"

# Emotions we care about for customer service
TARGET_EMOTIONS = [
    "anger",        # Maps to frustration
    "annoyance",    # Maps to frustration  
    "disappointment",# Maps to frustration
    "satisfaction", # Direct mapping (approval in the model)
    "approval",     # Maps to satisfaction
    "gratitude",    # Maps to satisfaction
    "neutral",      # Neutral state
    "urgency",      # Maps from fear/nervousness
    "fear",         # Maps to urgency
    "nervousness",  # Maps to urgency
]

# Emotion groupings for simplified output
EMOTION_GROUPS = {
    "frustration": ["anger", "annoyance", "disappointment", "disgust"],
    "satisfaction": ["approval", "gratitude", "joy", "love", "admiration", "relief"],
    "urgency": ["fear", "nervousness", "surprise"],
    "neutral": ["neutral", "realization", "curiosity"]
}

# =============================================================================
# TOPIC EXTRACTION CONFIGURATION
# =============================================================================
# BERTopic settings
TOPIC_MODEL_EMBEDDING = "all-MiniLM-L6-v2"  # Sentence transformer for embeddings
MIN_TOPIC_SIZE = 2  # Minimum documents to form a topic
NR_TOPICS = "auto"  # Let BERTopic decide, or set a number

# Common customer service topics to seed/guide extraction
SEED_TOPICS = [
    "billing",
    "technical support", 
    "product inquiry",
    "complaint",
    "refund",
    "shipping",
    "account issues",
    "general inquiry"
]

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================
BATCH_SIZE = 5  # Process this many files before saving intermediate results
MAX_WORKERS = 4  # Parallel processing threads (for non-GPU tasks)

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD_PORT = 8501
DASHBOARD_THEME = "light"  # or "dark"

# Chart color schemes
COLORS = {
    "positive": "#28a745",
    "neutral": "#ffc107", 
    "negative": "#dc3545",
    "frustration": "#dc3545",
    "satisfaction": "#28a745",
    "urgency": "#fd7e14",
}

# =============================================================================
# FUTURE: CLOUD AI CONFIGURATION (OpenAI)
# =============================================================================
# Uncomment and configure when ready for cloud deployment
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# USE_CLOUD_AI = False  # Toggle between local and cloud
# OPENAI_MODEL = "gpt-4"

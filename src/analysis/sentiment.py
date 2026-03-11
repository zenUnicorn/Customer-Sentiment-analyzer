"""
Sentiment Analysis Module using Transformers.

DEEP DIVE LESSON: NLP Sentiment Analysis Architecture
=====================================================

How Transformer-based Sentiment Analysis Works:
----------------------------------------------
1. Tokenization: Text → numerical tokens the model understands
2. Embedding: Tokens → dense vectors capturing semantic meaning
3. Attention Layers: Model learns relationships between all words
4. Classification Head: Final layer outputs sentiment probabilities

The Model We Use: RoBERTa
-------------------------
RoBERTa (Robustly Optimized BERT) improves on BERT by:
- Training longer with more data
- Removing next sentence prediction task
- Using dynamic masking during training

CardiffNLP's Twitter model is fine-tuned on social media text,
which is actually perfect for conversational customer calls!

Why Not Lexicon-Based (VADER, TextBlob)?
---------------------------------------
Traditional lexicon methods count positive/negative words.
Problems:
- "not good" gets counted as positive (it has "good")
- Sarcasm is impossible to detect
- Domain-specific words are missing

Transformers understand CONTEXT. "This is sick!" gets correctly
identified as positive slang, not illness-related.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Union
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SENTIMENT_MODEL, SENTIMENT_THRESHOLDS


class SentimentAnalyzer:
    """
    Analyzes text sentiment using a pre-trained transformer model.
    
    ARCHITECTURE PATTERN: Model Wrapper Class
    -----------------------------------------
    This pattern is common in NLP applications:
    1. Encapsulate model loading and preprocessing
    2. Provide a clean API for inference
    3. Handle batching and resource management
    4. Easy to swap underlying models
    """
    
    def __init__(self, model_name: str = SENTIMENT_MODEL):
        """
        Initialize with a specific model.
        
        Args:
            model_name: HuggingFace model identifier
            
        TECHNICAL NOTE: Model Selection
        --------------------------------
        We use cardiffnlp/twitter-roberta-base-sentiment-latest because:
        1. Trained on 124M tweets - understands casual language
        2. Output: negative (0), neutral (1), positive (2)
        3. ~125M parameters - good accuracy/speed balance
        4. Downloads automatically from HuggingFace Hub
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = ["negative", "neutral", "positive"]
        
    def load_model(self) -> None:
        """
        Load the model and tokenizer.
        
        DEEP DIVE: The Two Components
        -----------------------------
        1. Tokenizer: Converts text to input format
           - Handles vocabulary lookup
           - Adds special tokens ([CLS], [SEP])
           - Truncates/pads to max length
           
        2. Model: The neural network itself
           - Processes tokenized input
           - Returns logits (raw scores)
           - We convert to probabilities with softmax
        """
        if self.model is None:
            print(f"[Sentiment] Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            
            # Set to evaluation mode - disables dropout
            self.model.eval()
            
            print(f"[Sentiment] Model loaded on {self.device}")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict with:
                - label: "positive", "negative", or "neutral"
                - score: Confidence score for the label (0-1)
                - scores: All three probabilities
                - compound_score: Single value from -1 to 1
                
        TECHNICAL: Converting Logits to Scores
        --------------------------------------
        Model outputs raw logits (can be any number).
        Softmax converts these to probabilities that sum to 1.
        
        Example:
            logits: [-2.3, 0.5, 1.8]
            softmax: [0.02, 0.15, 0.83]  # Highly positive!
        """
        self.load_model()
        
        # Tokenize with proper handling
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # RoBERTa's max context
            padding=True
        ).to(self.device)
        
        # Inference without gradient computation (faster, less memory)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert to probabilities
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get the winner
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.labels[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Calculate compound score: -1 (negative) to +1 (positive)
        # Formula: (positive_prob - negative_prob)
        compound = float(probabilities[2] - probabilities[0])
        
        return {
            "label": predicted_label,
            "score": float(confidence),
            "scores": {
                "negative": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "positive": float(probabilities[2])
            },
            "compound_score": compound,
            "sentiment_category": self._categorize_sentiment(compound)
        }
    
    def _categorize_sentiment(self, compound: float) -> str:
        """
        Convert compound score to human-readable category.
        
        Uses thresholds from config for consistency.
        """
        if compound <= SENTIMENT_THRESHOLDS["very_negative"]:
            return "very_negative"
        elif compound <= SENTIMENT_THRESHOLDS["negative"]:
            return "negative"
        elif compound <= SENTIMENT_THRESHOLDS["neutral"]:
            return "neutral"
        elif compound <= SENTIMENT_THRESHOLDS["positive"]:
            return "positive"
        else:
            return "very_positive"
    
    def analyze_segments(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for each segment of a transcript.
        
        This enables tracking sentiment OVER TIME through the call!
        Very valuable for identifying:
        - Where the call went wrong
        - When the customer got satisfied
        - Turning points in the conversation
        
        Args:
            segments: List of dicts with 'text', 'start', 'end' keys
            
        Returns:
            Segments enriched with sentiment data
        """
        self.load_model()
        
        analyzed_segments = []
        
        for segment in segments:
            text = segment.get("text", "")
            
            if text.strip():
                sentiment = self.analyze(text)
                analyzed_segment = {
                    **segment,
                    "sentiment": sentiment
                }
            else:
                analyzed_segment = {
                    **segment,
                    "sentiment": None
                }
            
            analyzed_segments.append(analyzed_segment)
        
        return analyzed_segments
    
    def get_overall_sentiment(
        self, 
        analyzed_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate overall sentiment metrics from analyzed segments.
        
        ANALYTICAL INSIGHT:
        ------------------
        We don't just average - we also look at:
        - Sentiment trajectory (improving or worsening?)
        - Volatility (stable or fluctuating?)
        - Ending sentiment (how did it conclude?)
        """
        # Filter segments with valid sentiment
        valid_segments = [
            s for s in analyzed_segments 
            if s.get("sentiment") is not None
        ]
        
        if not valid_segments:
            return {"error": "No valid segments to analyze"}
        
        # Extract compound scores
        compound_scores = [
            s["sentiment"]["compound_score"] 
            for s in valid_segments
        ]
        
        # Calculate metrics
        avg_sentiment = np.mean(compound_scores)
        sentiment_std = np.std(compound_scores)
        
        # Trajectory: compare first half to second half
        mid_point = len(compound_scores) // 2
        first_half_avg = np.mean(compound_scores[:mid_point]) if mid_point > 0 else avg_sentiment
        second_half_avg = np.mean(compound_scores[mid_point:])
        trajectory = second_half_avg - first_half_avg
        
        # Determine trajectory label
        if trajectory > 0.2:
            trajectory_label = "improving"
        elif trajectory < -0.2:
            trajectory_label = "worsening"
        else:
            trajectory_label = "stable"
        
        # Count sentiment categories
        categories = [s["sentiment"]["label"] for s in valid_segments]
        category_counts = {
            "positive": categories.count("positive"),
            "neutral": categories.count("neutral"),
            "negative": categories.count("negative")
        }
        
        return {
            "average_compound": float(avg_sentiment),
            "overall_category": self._categorize_sentiment(avg_sentiment),
            "volatility": float(sentiment_std),
            "trajectory": trajectory_label,
            "trajectory_score": float(trajectory),
            "ending_sentiment": valid_segments[-1]["sentiment"]["label"],
            "category_distribution": category_counts,
            "total_segments_analyzed": len(valid_segments)
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Quick function to analyze sentiment of text.
    
    Usage:
        from src.analysis.sentiment import analyze_sentiment
        result = analyze_sentiment("I love this product!")
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text)


if __name__ == "__main__":
    # Quick test
    print("Testing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I'm really happy with the service!",
        "This is okay, nothing special.",
        "I'm extremely frustrated and want a refund!",
        "The product works as expected.",
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} ({result['score']:.2f})")
        print(f"Compound: {result['compound_score']:.2f}")

"""
Emotion Detection Module using GoEmotions Model.

DEEP DIVE LESSON: Multi-Label Emotion Classification
====================================================

Emotion vs Sentiment - What's the Difference?
--------------------------------------------
- Sentiment: Overall polarity (good/bad/neutral) - ONE output
- Emotion: Specific feelings (anger, joy, fear) - MULTIPLE possible outputs

Example: "I can't believe how slow this shipping is getting my gift!"
- Sentiment: Negative
- Emotions: Frustration (annoyance), Urgency (time pressure), Anticipation

Why Multi-Label Classification?
------------------------------
Real emotions are complex. A customer can simultaneously feel:
- Frustrated (with the problem)
- Grateful (that you're helping)
- Anxious (about the outcome)

Our model outputs MULTIPLE emotions with confidence scores!

The GoEmotions Dataset:
----------------------
Created by Google, labeled with 27 emotion categories + neutral.
58k carefully labeled Reddit comments - captures nuanced human emotion.

Technical Architecture:
----------------------
Same as sentiment (RoBERTa base), but with 28 output neurons instead of 3.
Each neuron outputs probability of that emotion being present.
We use sigmoid (not softmax) because multiple emotions can be true!
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Union
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EMOTION_MODEL, EMOTION_GROUPS, TARGET_EMOTIONS


class EmotionDetector:
    """
    Detects multiple emotions in text using GoEmotions model.
    
    ARCHITECTURE: Multi-Label vs Multi-Class
    ----------------------------------------
    - Multi-Class (Sentiment): Exactly one class is correct
      Use softmax - outputs sum to 1
      
    - Multi-Label (Emotion): Multiple classes can be correct
      Use sigmoid - each output is independent 0-1
    """
    
    # GoEmotions label mapping (model outputs in this order)
    EMOTION_LABELS = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]
    
    def __init__(self, model_name: str = EMOTION_MODEL):
        """
        Initialize the emotion detector.
        
        The model we use (SamLowe/roberta-base-go_emotions) is specifically
        fine-tuned for nuanced emotion detection in conversational text.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Threshold for considering an emotion "present"
        # Lower = more sensitive, higher = more confident
        self.emotion_threshold = 0.3
        
    def load_model(self) -> None:
        """Load model and tokenizer."""
        if self.model is None:
            print(f"[Emotion] Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            
            self.model.eval()
            print(f"[Emotion] Model loaded on {self.device}")
    
    def detect(self, text: str, threshold: float = None) -> Dict[str, Any]:
        """
        Detect emotions in text.
        
        Args:
            text: Text to analyze
            threshold: Minimum probability to report emotion (default: 0.3)
            
        Returns:
            Dict with:
                - emotions: List of detected emotions with scores
                - primary_emotion: Strongest emotion detected
                - grouped_emotions: Emotions mapped to our target groups
                - raw_scores: All 28 emotion probabilities
                
        TECHNICAL DETAIL: Sigmoid Activation
        ------------------------------------
        Unlike softmax (for sentiment), we use sigmoid here:
        
        Softmax: [0.1, 0.2, 0.7] - Must sum to 1
        Sigmoid: [0.1, 0.8, 0.9] - Each independent
        
        This lets us say "80% angry AND 90% disappointed"
        """
        self.load_model()
        
        threshold = threshold or self.emotion_threshold
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply sigmoid for multi-label
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Build emotion scores dictionary
        all_scores = {
            label: float(prob)
            for label, prob in zip(self.EMOTION_LABELS, probabilities)
        }
        
        # Filter to detected emotions (above threshold)
        detected = [
            {"emotion": label, "score": float(prob)}
            for label, prob in zip(self.EMOTION_LABELS, probabilities)
            if prob >= threshold
        ]
        
        # Sort by confidence
        detected = sorted(detected, key=lambda x: x["score"], reverse=True)
        
        # Find primary emotion
        primary_idx = np.argmax(probabilities)
        primary_emotion = self.EMOTION_LABELS[primary_idx]
        primary_score = float(probabilities[primary_idx])
        
        # Map to grouped emotions (for simpler reporting)
        grouped = self._map_to_groups(all_scores)
        
        return {
            "emotions": detected,
            "primary_emotion": {
                "emotion": primary_emotion,
                "score": primary_score
            },
            "grouped_emotions": grouped,
            "raw_scores": all_scores,
            "text_length": len(text.split())
        }
    
    def _map_to_groups(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Map fine-grained emotions to our simplified groups.
        
        BUSINESS VALUE:
        --------------
        Executives don't want to see 28 emotions.
        They want to know: "Is the customer frustrated or satisfied?"
        
        We aggregate related emotions:
        - Frustration = max(anger, annoyance, disappointment, disgust)
        - Satisfaction = max(approval, gratitude, joy, etc.)
        - Urgency = max(fear, nervousness, surprise)
        - Neutral = neutral + realization + curiosity
        """
        grouped = {}
        
        for group_name, emotions in EMOTION_GROUPS.items():
            # Take the MAX of all emotions in the group
            # This represents "how much of this feeling is present"
            group_scores = [scores.get(e, 0.0) for e in emotions]
            grouped[group_name] = float(max(group_scores)) if group_scores else 0.0
        
        # Add aggregate score for each group (for trend analysis)
        # Using max is better than mean for emotion detection
        # because even one strong emotion signal is meaningful
        
        return grouped
    
    def detect_in_segments(
        self, 
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect emotions in transcript segments.
        
        Enables emotion tracking THROUGH THE CALL:
        - Did frustration spike at minute 3?
        - When did satisfaction appear?
        - What pattern do difficult calls follow?
        """
        self.load_model()
        
        analyzed_segments = []
        
        for segment in segments:
            text = segment.get("text", "")
            
            if text.strip():
                emotions = self.detect(text)
                analyzed_segment = {
                    **segment,
                    "emotions": emotions
                }
            else:
                analyzed_segment = {
                    **segment,
                    "emotions": None
                }
            
            analyzed_segments.append(analyzed_segment)
        
        return analyzed_segments
    
    def get_emotion_summary(
        self, 
        analyzed_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarize emotions across all segments.
        
        Provides:
        - Peak emotions (highest intensity moments)
        - Average emotion levels
        - Dominant emotion throughout call
        - Emotion timeline for visualization
        """
        valid_segments = [
            s for s in analyzed_segments 
            if s.get("emotions") is not None
        ]
        
        if not valid_segments:
            return {"error": "No valid segments to analyze"}
        
        # Collect grouped emotion scores across segments
        grouped_scores = {
            "frustration": [],
            "satisfaction": [],
            "urgency": [],
            "neutral": []
        }
        
        emotion_timeline = []
        
        for seg in valid_segments:
            grouped = seg["emotions"]["grouped_emotions"]
            
            for group_name in grouped_scores:
                grouped_scores[group_name].append(grouped.get(group_name, 0))
            
            # Build timeline entry
            emotion_timeline.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "primary": seg["emotions"]["primary_emotion"]["emotion"],
                "frustration": grouped.get("frustration", 0),
                "satisfaction": grouped.get("satisfaction", 0),
                "urgency": grouped.get("urgency", 0)
            })
        
        # Calculate averages and peaks
        summary = {
            "average_emotions": {},
            "peak_emotions": {},
            "dominant_emotion": None,
            "emotional_volatility": {}
        }
        
        max_avg = 0
        dominant = None
        
        for group_name, scores in grouped_scores.items():
            avg = float(np.mean(scores))
            peak = float(max(scores))
            std = float(np.std(scores))
            
            summary["average_emotions"][group_name] = avg
            summary["peak_emotions"][group_name] = peak
            summary["emotional_volatility"][group_name] = std
            
            if avg > max_avg:
                max_avg = avg
                dominant = group_name
        
        summary["dominant_emotion"] = dominant
        summary["emotion_timeline"] = emotion_timeline
        summary["total_segments"] = len(valid_segments)
        
        return summary


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def detect_emotions(text: str) -> Dict[str, Any]:
    """
    Quick function to detect emotions in text.
    
    Usage:
        from src.analysis.emotion import detect_emotions
        result = detect_emotions("I'm so frustrated with this delay!")
    """
    detector = EmotionDetector()
    return detector.detect(text)


if __name__ == "__main__":
    # Quick test
    print("Testing Emotion Detector...")
    detector = EmotionDetector()
    
    test_texts = [
        "I'm so frustrated! I've been waiting for an hour!",
        "Thank you so much, you've been incredibly helpful!",
        "I need this resolved urgently, it's critical for my business.",
        "Okay, I understand the process now.",
    ]
    
    for text in test_texts:
        result = detector.detect(text)
        print(f"\nText: {text}")
        print(f"Primary: {result['primary_emotion']['emotion']} ({result['primary_emotion']['score']:.2f})")
        print(f"Grouped: {result['grouped_emotions']}")

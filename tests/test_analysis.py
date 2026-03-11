"""
Test suite for Customer Sentiment Analyzer.

TESTING STRATEGY:
----------------
1. Unit tests for individual components
2. Integration tests for pipeline
3. Mock tests when models aren't needed

Run with: pytest tests/test_analysis.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_imports(self):
        """Verify config can be imported."""
        from config import (
            WHISPER_MODEL_SIZE,
            SENTIMENT_MODEL,
            EMOTION_MODEL,
            AUDIO_DIR,
            RESULTS_DIR
        )
        
        assert WHISPER_MODEL_SIZE is not None
        assert SENTIMENT_MODEL is not None
        assert EMOTION_MODEL is not None
    
    def test_directories_exist(self):
        """Verify data directories are created."""
        from config import AUDIO_DIR, TRANSCRIPTS_DIR, RESULTS_DIR
        
        assert AUDIO_DIR.exists()
        assert TRANSCRIPTS_DIR.exists()
        assert RESULTS_DIR.exists()


class TestSentimentAnalysis:
    """Test sentiment analysis module."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        from src.analysis.sentiment import SentimentAnalyzer
        return SentimentAnalyzer()
    
    def test_positive_sentiment(self, analyzer):
        """Test detection of positive sentiment."""
        result = analyzer.analyze("I absolutely love this product! It's amazing!")
        
        assert "label" in result
        assert "compound_score" in result
        assert result["label"] == "positive"
        assert result["compound_score"] > 0
    
    def test_negative_sentiment(self, analyzer):
        """Test detection of negative sentiment."""
        result = analyzer.analyze("This is terrible! I'm very disappointed and angry!")
        
        assert result["label"] == "negative"
        assert result["compound_score"] < 0
    
    def test_neutral_sentiment(self, analyzer):
        """Test detection of neutral sentiment."""
        result = analyzer.analyze("The package arrived today.")
        
        # Neutral should have compound near 0
        assert abs(result["compound_score"]) < 0.5
    
    def test_sentiment_scores_structure(self, analyzer):
        """Test that scores dictionary has correct structure."""
        result = analyzer.analyze("Hello world")
        
        assert "scores" in result
        assert "positive" in result["scores"]
        assert "neutral" in result["scores"]
        assert "negative" in result["scores"]
        
        # Scores should sum to approximately 1
        total = sum(result["scores"].values())
        assert 0.99 <= total <= 1.01


class TestEmotionDetection:
    """Test emotion detection module."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        from src.analysis.emotion import EmotionDetector
        return EmotionDetector()
    
    def test_frustration_detection(self, detector):
        """Test detection of frustration."""
        result = detector.detect("I'm so frustrated! This is ridiculous!")
        
        assert "grouped_emotions" in result
        assert result["grouped_emotions"]["frustration"] > 0.3
    
    def test_satisfaction_detection(self, detector):
        """Test detection of satisfaction."""
        result = detector.detect("Thank you so much! You've been incredibly helpful!")
        
        assert result["grouped_emotions"]["satisfaction"] > 0.3
    
    def test_urgency_detection(self, detector):
        """Test detection of urgency."""
        result = detector.detect("I need this fixed immediately! It's critical!")
        
        assert "urgency" in result["grouped_emotions"]
    
    def test_primary_emotion_exists(self, detector):
        """Test that primary emotion is identified."""
        result = detector.detect("I'm feeling various things about this.")
        
        assert "primary_emotion" in result
        assert "emotion" in result["primary_emotion"]
        assert "score" in result["primary_emotion"]


class TestTopicExtraction:
    """Test topic extraction module."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        from src.analysis.topics import TopicExtractor
        return TopicExtractor(min_topic_size=2)
    
    def test_topic_extraction_requires_documents(self, extractor):
        """Test that topic extraction needs multiple documents."""
        # With just one document, should handle gracefully
        single_doc = ["This is a test document."]
        
        result = extractor.extract_topics(single_doc)
        # Should not crash, may warn about insufficient data
        assert result is not None
    
    def test_topic_keywords_structure(self, extractor):
        """Test topic keywords have correct structure."""
        docs = [
            "I need help with billing.",
            "There's an issue with my bill.",
            "Can you help with payment?",
            "Billing question here.",
            "Payment problem.",
            "I want to ship this item.",
            "Shipping is slow.",
            "When will my order ship?",
            "Shipping status please.",
            "Track my shipment."
        ]
        
        result = extractor.extract_topics(docs)
        
        assert "topic_keywords" in result
        assert "topic_assignments" in result


class TestPipeline:
    """Test the processing pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        from src.pipeline import CallAnalysisPipeline
        
        pipeline = CallAnalysisPipeline(
            whisper_model="tiny",  # Use smallest for testing
            enable_topics=False    # Disable for faster testing
        )
        
        assert pipeline is not None
        assert pipeline.transcriber is not None
        assert pipeline.sentiment_analyzer is not None
        assert pipeline.emotion_detector is not None


class TestTranscription:
    """Test transcription module."""
    
    def test_transcriber_initialization(self):
        """Test transcriber can be initialized."""
        from src.transcription import AudioTranscriber
        
        transcriber = AudioTranscriber(model_size="tiny")
        assert transcriber is not None
        assert transcriber.model_size == "tiny"
    
    def test_supported_formats(self):
        """Test that supported formats are defined."""
        from config import SUPPORTED_AUDIO_FORMATS
        
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS
        assert ".wav" in SUPPORTED_AUDIO_FORMATS


# =============================================================================
# INTEGRATION TESTS
# =============================================================================
class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_analyze_text_only(self):
        """Test analyzing text without audio."""
        from src.analysis import analyze_sentiment, detect_emotions
        
        text = "I called about my order and the agent was very helpful!"
        
        sentiment = analyze_sentiment(text)
        emotions = detect_emotions(text)
        
        assert sentiment["label"] in ["positive", "neutral", "negative"]
        assert "grouped_emotions" in emotions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

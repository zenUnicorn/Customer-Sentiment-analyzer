"""
Processing Pipeline - Orchestrates the Full Analysis Workflow.

DEEP DIVE LESSON: Pipeline Architecture for Data Processing
==========================================================

What is a Pipeline?
------------------
A pipeline is a sequence of processing steps where the output
of one step becomes the input of the next.

Our Pipeline:
    Audio → Transcript → Sentiment → Emotions → Topics → Results

Why Pipeline Architecture?
-------------------------
1. Modularity: Each step is independent and testable
2. Flexibility: Easy to add/remove/modify steps
3. Reusability: Components work standalone or together
4. Scalability: Can parallelize independent steps
5. Debugging: Inspect output at any stage

Design Pattern: Chain of Responsibility
--------------------------------------
Each processing step:
- Takes input from previous step
- Performs its transformation
- Passes enhanced output to next step

This is similar to Unix pipes: cat file | grep pattern | sort | uniq
Each command does one thing well and passes output forward.

Batch Processing Considerations:
-------------------------------
1. Checkpointing: Save progress to resume on failure
2. Parallel Processing: Use multiprocessing for CPU-bound tasks
3. Memory Management: Don't load everything at once
4. Progress Tracking: Keep users informed
5. Error Handling: Continue gracefully on individual failures
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    AUDIO_DIR, TRANSCRIPTS_DIR, RESULTS_DIR,
    BATCH_SIZE, MAX_WORKERS, SUPPORTED_AUDIO_FORMATS
)

from src.transcription import AudioTranscriber
from src.analysis import SentimentAnalyzer, EmotionDetector, TopicExtractor


class CallAnalysisPipeline:
    """
    Orchestrates end-to-end analysis of customer call recordings.
    
    ARCHITECTURE: Facade Pattern
    ---------------------------
    This class provides a simple interface to complex subsystems.
    Users call one method; we handle all the complexity internally.
    
    Usage:
        pipeline = CallAnalysisPipeline()
        results = pipeline.process_batch(audio_files)
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        enable_topics: bool = True,
        save_intermediate: bool = True
    ):
        """
        Initialize the analysis pipeline.
        
        Args:
            whisper_model: Whisper model size for transcription
            enable_topics: Whether to run topic extraction (requires multiple docs)
            save_intermediate: Save results after each step
        """
        # Initialize components (lazy - models loaded on first use)
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.topic_extractor = TopicExtractor() if enable_topics else None
        
        self.enable_topics = enable_topics
        self.save_intermediate = save_intermediate
        
        # Processing state
        self.processed_results = []
        self.processing_errors = []
        
    def process_single_call(
        self, 
        input_source: Union[str, Path, Dict],
        call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single call recording or transcript.
        
        Args:
            input_source: Path to audio file, transcript JSON, or transcript dict
            call_id: Optional identifier for this call
            
        Returns:
            Complete analysis results for the call
            
        PIPELINE FLOW:
        -------------
        1. Load/transcribe → Get text and segments
        2. Sentiment analysis → Score each segment
        3. Emotion detection → Detect emotions per segment
        4. Aggregate → Calculate overall metrics
        5. Package → Return structured results
        """
        start_time = datetime.now()
        
        # Determine input type and get transcript
        if isinstance(input_source, dict):
            # Already a transcript dictionary
            transcript = input_source
            source_type = "dict"
            
        elif isinstance(input_source, (str, Path)):
            input_path = Path(input_source)
            
            if input_path.suffix.lower() == '.json':
                # Load existing transcript
                transcript = self.transcriber.load_transcript(input_path)
                source_type = "transcript"
                
            elif input_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS:
                # Transcribe audio file
                transcript = self.transcriber.transcribe_audio(input_path)
                source_type = "audio"
                
                if self.save_intermediate:
                    self.transcriber.save_transcript(transcript)
            else:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")
        else:
            raise ValueError(f"Invalid input type: {type(input_source)}")
        
        # Generate call ID if not provided
        if call_id is None:
            call_id = transcript.get("file_name", f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Get segments and full text
        segments = transcript.get("segments", [])
        full_text = transcript.get("text", "")
        
        # If no segments, create one from full text
        if not segments and full_text:
            segments = [{
                "id": 0,
                "start": 0,
                "end": transcript.get("duration_seconds", 0),
                "text": full_text
            }]
        
        # =================================================================
        # STEP 2: Sentiment Analysis
        # =================================================================
        print(f"[Pipeline] Analyzing sentiment for {call_id}...")
        
        analyzed_segments = self.sentiment_analyzer.analyze_segments(segments)
        overall_sentiment = self.sentiment_analyzer.get_overall_sentiment(analyzed_segments)
        
        # =================================================================
        # STEP 3: Emotion Detection
        # =================================================================
        print(f"[Pipeline] Detecting emotions for {call_id}...")
        
        emotion_segments = self.emotion_detector.detect_in_segments(segments)
        emotion_summary = self.emotion_detector.get_emotion_summary(emotion_segments)
        
        # =================================================================
        # STEP 4: Merge Segment Analysis
        # =================================================================
        # Combine sentiment and emotion data per segment
        merged_segments = []
        for i, seg in enumerate(segments):
            merged = {
                **seg,
                "sentiment": analyzed_segments[i].get("sentiment"),
                "emotions": emotion_segments[i].get("emotions")
            }
            merged_segments.append(merged)
        
        # =================================================================
        # STEP 5: Package Results
        # =================================================================
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "call_id": call_id,
            "source_type": source_type,
            "source_file": str(transcript.get("file_path", "unknown")),
            "processed_at": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            
            # Transcript data
            "transcript": {
                "full_text": full_text,
                "word_count": transcript.get("word_count", len(full_text.split())),
                "duration_seconds": transcript.get("duration_seconds", 0),
                "language": transcript.get("language", "unknown")
            },
            
            # Analysis results
            "sentiment_analysis": overall_sentiment,
            "emotion_analysis": emotion_summary,
            
            # Detailed segment analysis
            "segments": merged_segments,
            
            # Metadata
            "metadata": {
                "whisper_model": self.transcriber.model_size,
                "sentiment_model": self.sentiment_analyzer.model_name,
                "emotion_model": self.emotion_detector.model_name
            }
        }
        
        return result
    
    def process_batch(
        self, 
        input_sources: List[Union[str, Path]],
        extract_topics: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple calls in batch.
        
        Args:
            input_sources: List of audio files or transcript paths
            extract_topics: Run topic extraction across all calls
            
        Returns:
            Batch results with individual and aggregate analysis
            
        BATCH PROCESSING STRATEGY:
        -------------------------
        1. Sequential transcription (GPU-bound, can't parallelize easily)
        2. Parallel sentiment/emotion (CPU-bound, safe to parallelize)
        3. Topic extraction after all texts collected (needs full corpus)
        4. Aggregate metrics across all calls
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting batch processing of {len(input_sources)} files")
        print(f"{'='*60}\n")
        
        batch_start = datetime.now()
        results = []
        errors = []
        all_texts = []
        all_ids = []
        
        # Process each call
        for i, source in enumerate(tqdm(input_sources, desc="Processing calls")):
            try:
                call_id = Path(source).stem if isinstance(source, (str, Path)) else f"call_{i}"
                
                result = self.process_single_call(source, call_id=call_id)
                results.append(result)
                
                # Collect text for topic modeling
                if extract_topics and result["transcript"]["full_text"]:
                    all_texts.append(result["transcript"]["full_text"])
                    all_ids.append(call_id)
                
                # Save individual result
                if self.save_intermediate:
                    self._save_result(result, RESULTS_DIR / f"{call_id}_analysis.json")
                    
            except Exception as e:
                error_info = {
                    "source": str(source),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                print(f"[Pipeline] Error processing {source}: {e}")
        
        # =================================================================
        # Topic Extraction (requires all documents)
        # =================================================================
        topic_results = None
        
        if extract_topics and self.topic_extractor and len(all_texts) >= 5:
            print(f"\n[Pipeline] Extracting topics from {len(all_texts)} calls...")
            
            try:
                topic_results = self.topic_extractor.extract_topics(all_texts, all_ids)
                topic_summary = self.topic_extractor.get_topic_summary(topic_results)
                
                # Add topic assignments to individual results
                topic_assignments = {
                    ta["document_id"]: ta 
                    for ta in topic_results.get("topic_assignments", [])
                }
                
                for result in results:
                    call_id = result["call_id"]
                    if call_id in topic_assignments:
                        result["topic"] = topic_assignments[call_id]
                        
            except Exception as e:
                print(f"[Pipeline] Topic extraction failed: {e}")
                topic_results = {"error": str(e)}
        
        elif len(all_texts) < 5:
            print("[Pipeline] Skipping topics - need at least 5 documents")
        
        # =================================================================
        # Aggregate Metrics
        # =================================================================
        aggregate = self._calculate_aggregate_metrics(results)
        
        # =================================================================
        # Final Batch Results
        # =================================================================
        batch_time = (datetime.now() - batch_start).total_seconds()
        
        batch_results = {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processed_at": datetime.now().isoformat(),
            "total_processing_time_seconds": batch_time,
            "total_calls": len(input_sources),
            "successful": len(results),
            "failed": len(errors),
            
            # Aggregate analysis
            "aggregate_metrics": aggregate,
            
            # Topic analysis (if performed)
            "topic_analysis": topic_summary if topic_results and "error" not in topic_results else topic_results,
            
            # Individual results
            "results": results,
            
            # Errors
            "errors": errors
        }
        
        # Save batch results
        if self.save_intermediate:
            batch_path = RESULTS_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._save_result(batch_results, batch_path)
            print(f"\n[Pipeline] Batch results saved to: {batch_path}")
        
        return batch_results
    
    def _calculate_aggregate_metrics(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all processed calls.
        
        BUSINESS VALUE:
        --------------
        These metrics answer executive questions:
        - What's our overall customer satisfaction?
        - How many calls are frustrating for customers?
        - What's the average call length?
        """
        if not results:
            return {}
        
        # Sentiment aggregation
        sentiment_categories = []
        compound_scores = []
        
        # Emotion aggregation
        frustration_scores = []
        satisfaction_scores = []
        urgency_scores = []
        
        # Duration
        durations = []
        
        for result in results:
            # Sentiment
            sentiment = result.get("sentiment_analysis", {})
            if sentiment and "error" not in sentiment:
                sentiment_categories.append(sentiment.get("overall_category"))
                compound_scores.append(sentiment.get("average_compound", 0))
            
            # Emotions
            emotions = result.get("emotion_analysis", {})
            if emotions and "error" not in emotions:
                avg_emotions = emotions.get("average_emotions", {})
                frustration_scores.append(avg_emotions.get("frustration", 0))
                satisfaction_scores.append(avg_emotions.get("satisfaction", 0))
                urgency_scores.append(avg_emotions.get("urgency", 0))
            
            # Duration
            transcript = result.get("transcript", {})
            if transcript.get("duration_seconds", 0) > 0:
                durations.append(transcript["duration_seconds"])
        
        # Calculate stats
        import numpy as np
        
        aggregate = {
            "sentiment": {
                "average_compound": float(np.mean(compound_scores)) if compound_scores else 0,
                "std_compound": float(np.std(compound_scores)) if compound_scores else 0,
                "category_distribution": {
                    cat: sentiment_categories.count(cat) 
                    for cat in set(sentiment_categories) if cat
                }
            },
            "emotions": {
                "average_frustration": float(np.mean(frustration_scores)) if frustration_scores else 0,
                "average_satisfaction": float(np.mean(satisfaction_scores)) if satisfaction_scores else 0,
                "average_urgency": float(np.mean(urgency_scores)) if urgency_scores else 0,
                "high_frustration_calls": sum(1 for f in frustration_scores if f > 0.5),
                "high_satisfaction_calls": sum(1 for s in satisfaction_scores if s > 0.5)
            },
            "calls": {
                "average_duration_seconds": float(np.mean(durations)) if durations else 0,
                "total_duration_seconds": float(sum(durations)),
                "calls_with_negative_sentiment": sum(
                    1 for c in sentiment_categories 
                    if c in ["negative", "very_negative"]
                )
            }
        }
        
        return aggregate
    
    def _save_result(self, data: Dict, path: Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def analyze_call(audio_path: str) -> Dict[str, Any]:
    """
    Quick function to analyze a single call.
    
    Usage:
        from src.pipeline.processor import analyze_call
        result = analyze_call("call_recording.mp3")
    """
    pipeline = CallAnalysisPipeline()
    return pipeline.process_single_call(audio_path)


def analyze_calls_batch(audio_paths: List[str]) -> Dict[str, Any]:
    """
    Quick function to analyze multiple calls.
    
    Usage:
        from src.pipeline.processor import analyze_calls_batch
        result = analyze_calls_batch(["call1.mp3", "call2.mp3"])
    """
    pipeline = CallAnalysisPipeline()
    return pipeline.process_batch(audio_paths)


if __name__ == "__main__":
    print("Pipeline module loaded successfully!")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Check for any audio files
    audio_files = list(AUDIO_DIR.glob("*.*"))
    print(f"Audio files found: {len([f for f in audio_files if f.suffix.lower() in SUPPORTED_AUDIO_FORMATS])}")

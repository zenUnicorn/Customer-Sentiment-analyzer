#!/usr/bin/env python3
"""
Customer Sentiment Analyzer - Main Entry Point

DEEP DIVE: Application Entry Points
===================================

This file serves multiple purposes:
1. CLI interface for batch processing
2. Quick-start for developers
3. Integration point for automated pipelines

USAGE EXAMPLES:
--------------
# Analyze a single audio file
python main.py --audio path/to/call.mp3

# Analyze multiple files
python main.py --batch data/audio/

# Start the dashboard
python main.py --dashboard

# Analyze pre-transcribed text
python main.py --transcript path/to/transcript.json

INTEGRATION WITH WORKFLOWS:
--------------------------
This can be called from:
- Cron jobs for scheduled batch processing
- CI/CD pipelines for testing
- Import statement from other Python code
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from config import AUDIO_DIR, RESULTS_DIR, SUPPORTED_AUDIO_FORMATS


def analyze_single(file_path: str) -> dict:
    """Analyze a single audio file or transcript."""
    from src.pipeline import CallAnalysisPipeline
    
    print(f"\n{'='*60}")
    print("CUSTOMER SENTIMENT ANALYZER")
    print(f"{'='*60}")
    print(f"Processing: {file_path}\n")
    
    pipeline = CallAnalysisPipeline()
    result = pipeline.process_single_call(file_path)
    
    # Print summary
    print_result_summary(result)
    
    return result


def analyze_batch(directory: str) -> dict:
    """Analyze all audio files in a directory."""
    from src.pipeline import CallAnalysisPipeline
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find audio files
    audio_files = []
    for ext in SUPPORTED_AUDIO_FORMATS:
        audio_files.extend(dir_path.glob(f"*{ext}"))
    
    # Also find transcript files
    audio_files.extend(dir_path.glob("*.json"))
    
    if not audio_files:
        print(f"No audio or transcript files found in {directory}")
        return {}
    
    print(f"\n{'='*60}")
    print("CUSTOMER SENTIMENT ANALYZER - BATCH MODE")
    print(f"{'='*60}")
    print(f"Found {len(audio_files)} files to process\n")
    
    pipeline = CallAnalysisPipeline()
    results = pipeline.process_batch(audio_files)
    
    # Print batch summary
    print_batch_summary(results)
    
    return results


def start_dashboard():
    """Start the Streamlit dashboard."""
    import subprocess
    
    print("\n" + "="*60)
    print("Starting Customer Sentiment Analyzer Dashboard...")
    print("="*60 + "\n")
    
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.headless", "true"
    ])


def print_result_summary(result: dict):
    """Print a formatted summary of analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Transcript info
    transcript = result.get("transcript", {})
    print(f"\n📄 Transcript:")
    print(f"   Duration: {transcript.get('duration_seconds', 0):.1f} seconds")
    print(f"   Words: {transcript.get('word_count', 0)}")
    print(f"   Language: {transcript.get('language', 'unknown')}")
    
    # Sentiment
    sentiment = result.get("sentiment_analysis", {})
    print(f"\n📊 Sentiment:")
    print(f"   Overall: {sentiment.get('overall_category', 'unknown')}")
    print(f"   Compound Score: {sentiment.get('average_compound', 0):.2f}")
    print(f"   Trajectory: {sentiment.get('trajectory', 'unknown')}")
    
    # Emotions
    emotions = result.get("emotion_analysis", {})
    if emotions and "error" not in emotions:
        print(f"\n😊 Emotions:")
        print(f"   Dominant: {emotions.get('dominant_emotion', 'unknown')}")
        avg_emotions = emotions.get("average_emotions", {})
        for emotion, score in avg_emotions.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   {emotion.capitalize():12} [{bar}] {score:.2f}")
    
    print("\n" + "="*60)


def print_batch_summary(results: dict):
    """Print a formatted summary of batch results."""
    print("\n" + "="*60)
    print("BATCH ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total Calls: {results.get('total_calls', 0)}")
    print(f"   Successful: {results.get('successful', 0)}")
    print(f"   Failed: {results.get('failed', 0)}")
    print(f"   Processing Time: {results.get('total_processing_time_seconds', 0):.1f}s")
    
    agg = results.get("aggregate_metrics", {})
    
    # Sentiment
    sentiment = agg.get("sentiment", {})
    print(f"\n📈 Sentiment Overview:")
    print(f"   Average: {sentiment.get('average_compound', 0):.2f}")
    dist = sentiment.get("category_distribution", {})
    for cat, count in dist.items():
        print(f"   {cat.capitalize()}: {count} calls")
    
    # Emotions
    emotions = agg.get("emotions", {})
    print(f"\n😊 Emotion Overview:")
    print(f"   High Frustration Calls: {emotions.get('high_frustration_calls', 0)}")
    print(f"   High Satisfaction Calls: {emotions.get('high_satisfaction_calls', 0)}")
    
    # Topics
    topics = results.get("topic_analysis", {})
    if topics and "error" not in topics:
        print(f"\n🏷️ Topic Overview:")
        print(f"   Topics Found: {topics.get('num_topics_found', 0)}")
        print(f"   Categorization Rate: {topics.get('categorization_rate', 0):.1f}%")
    
    print("\n" + "="*60)
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*60 + "\n")


def demo():
    """Run a quick demo with sample text."""
    from src.analysis import analyze_sentiment, detect_emotions
    
    print("\n" + "="*60)
    print("DEMO: Quick Sentiment & Emotion Analysis")
    print("="*60 + "\n")
    
    test_texts = [
        "I'm absolutely thrilled with your customer service! The agent was so helpful and resolved my issue immediately.",
        "This is ridiculous! I've been waiting for 30 minutes and nobody can help me with this simple billing question.",
        "The product works as expected. Nothing special, but gets the job done.",
        "I need this fixed urgently! My business depends on this service being operational by tomorrow."
    ]
    
    for text in test_texts:
        print(f"\n📝 Text: \"{text[:60]}...\"" if len(text) > 60 else f"\n📝 Text: \"{text}\"")
        
        # Sentiment
        sentiment = analyze_sentiment(text)
        print(f"   Sentiment: {sentiment['label']} (score: {sentiment['compound_score']:.2f})")
        
        # Emotions
        emotions = detect_emotions(text)
        grouped = emotions['grouped_emotions']
        dominant = max(grouped, key=grouped.get)
        print(f"   Dominant Emotion: {dominant} ({grouped[dominant]:.2f})")
    
    print("\n" + "="*60)
    print("To analyze audio files, use:")
    print("  python main.py --audio your_file.mp3")
    print("  python main.py --batch data/audio/")
    print("  python main.py --dashboard")
    print("="*60 + "\n")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Customer Sentiment Analyzer - Analyze call recordings using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run quick demo
  python main.py --audio call.mp3          # Analyze single file
  python main.py --batch data/audio/       # Analyze directory
  python main.py --dashboard               # Start web dashboard
  python main.py --transcript text.json    # Analyze pre-transcribed file
        """
    )
    
    parser.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to a single audio file to analyze"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Path to directory containing audio files"
    )
    
    parser.add_argument(
        "--transcript", "-t",
        type=str,
        help="Path to pre-transcribed JSON file"
    )
    
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Start the Streamlit dashboard"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo with sample text"
    )
    
    args = parser.parse_args()
    
    # Handle arguments
    if args.demo:
        demo()
    elif args.audio:
        analyze_single(args.audio)
    elif args.transcript:
        analyze_single(args.transcript)
    elif args.batch:
        analyze_batch(args.batch)
    elif args.dashboard:
        start_dashboard()
    else:
        # No args - show help
        parser.print_help()
        print("\n💡 TIP: Start with '--demo' to see the analyzer in action!")


if __name__ == "__main__":
    main()

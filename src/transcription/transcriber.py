"""
Audio Transcription Module using OpenAI's Whisper.

DEEP DIVE LESSON: Speech-to-Text Pipeline Architecture
======================================================

How Whisper Works:
-----------------
1. Audio Loading: Converts any audio format to 16kHz mono (Whisper's expected input)
2. Feature Extraction: Converts audio waveform to mel spectrogram (visual representation of frequencies)
3. Encoder: Processes the spectrogram through transformer layers to understand audio patterns
4. Decoder: Generates text tokens auto-regressively (one word at a time)

Why Whisper is Revolutionary:
----------------------------
- Trained on 680,000 hours of multilingual audio
- Handles accents, background noise, and multiple languages
- Runs 100% locally - your customer data never leaves your machine
- Model sizes let you trade accuracy for speed

The mel spectrogram is key - it's how machines "see" sound:
- X-axis: Time
- Y-axis: Frequency (in mel scale, which mimics human hearing)
- Color intensity: Volume/energy at that frequency

This is why we can use image-processing techniques (transformers) on audio!
"""

import whisper
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from datetime import datetime

# Import our configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    WHISPER_MODEL_SIZE, 
    SUPPORTED_AUDIO_FORMATS,
    TRANSCRIPTS_DIR
)


class AudioTranscriber:
    """
    Handles audio-to-text transcription using Whisper.
    
    ARCHITECTURE NOTE:
    We use a class-based design here because:
    1. Model loading is expensive - we want to load once and reuse
    2. State management - track what's been processed
    3. Easy to extend - subclass for cloud provider later
    """
    
    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        """
        Initialize the transcriber with specified model size.
        
        Args:
            model_size: One of 'tiny', 'base', 'small', 'medium', 'large'
                       Larger = more accurate but slower
        
        TECHNICAL NOTE: Model Loading
        -----------------------------
        Whisper models are downloaded on first use to ~/.cache/whisper/
        - tiny: 39M parameters, ~1GB RAM, ~32x realtime on CPU
        - base: 74M parameters, ~1GB RAM, ~16x realtime on CPU  
        - small: 244M parameters, ~2GB RAM, ~6x realtime on CPU
        - medium: 769M parameters, ~5GB RAM, ~2x realtime on CPU
        - large: 1550M parameters, ~10GB RAM, ~1x realtime on CPU
        """
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Transcriber] Using device: {self.device}")
        
    def load_model(self) -> None:
        """
        Lazy load the Whisper model.
        
        DESIGN PATTERN: Lazy Loading
        ----------------------------
        We don't load the model in __init__ because:
        1. Faster startup if transcription isn't needed immediately
        2. Memory efficiency - only load when actually used
        3. Better error handling - can catch load failures separately
        """
        if self.model is None:
            print(f"[Transcriber] Loading Whisper '{self.model_size}' model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"[Transcriber] Model loaded successfully!")
    
    def transcribe_audio(
        self, 
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            verbose: Print progress during transcription
            
        Returns:
            Dict containing:
                - text: Full transcription
                - segments: Timestamped segments with confidence scores
                - language: Detected/specified language
                - duration: Audio duration in seconds
                
        TECHNICAL DEEP DIVE: Whisper Output Structure
        ---------------------------------------------
        Whisper provides rich output including:
        - segments: List of dicts with 'start', 'end', 'text' times
        - Each segment has token-level timestamps and confidence
        - This enables features like "jump to this moment in call"
        """
        self.load_model()
        
        audio_path = Path(audio_path)
        
        # Validate file exists and format is supported
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {SUPPORTED_AUDIO_FORMATS}"
            )
        
        print(f"[Transcriber] Transcribing: {audio_path.name}")
        
        # The magic happens here!
        # Whisper handles all preprocessing internally:
        # 1. Load audio at 16kHz
        # 2. Pad/trim to 30 seconds
        # 3. Convert to log-mel spectrogram
        # 4. Run through encoder-decoder
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=verbose,
            # These options improve accuracy for conversational audio
            condition_on_previous_text=True,  # Use context from previous segments
            word_timestamps=True,  # Get word-level timing
        )
        
        # Structure the output
        transcription = {
            "file_name": audio_path.name,
            "file_path": str(audio_path),
            "transcription_timestamp": datetime.now().isoformat(),
            "model_used": f"whisper-{self.model_size}",
            "language": result.get("language", "unknown"),
            "text": result["text"].strip(),
            "segments": [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    # Confidence isn't directly available, but we can use avg_logprob
                    # Higher (less negative) = more confident
                    "confidence": seg.get("avg_logprob", -1.0),
                }
                for seg in result.get("segments", [])
            ],
            "word_count": len(result["text"].split()),
        }
        
        # Calculate duration from last segment
        if transcription["segments"]:
            transcription["duration_seconds"] = transcription["segments"][-1]["end"]
        else:
            transcription["duration_seconds"] = 0
            
        return transcription
    
    def save_transcript(
        self, 
        transcription: Dict[str, Any], 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save transcription to JSON file.
        
        Args:
            transcription: Output from transcribe_audio()
            output_path: Where to save (default: TRANSCRIPTS_DIR)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            # Generate filename from source audio name
            source_name = Path(transcription["file_name"]).stem
            output_path = TRANSCRIPTS_DIR / f"{source_name}_transcript.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        
        print(f"[Transcriber] Saved transcript to: {output_path}")
        return output_path
    
    def load_transcript(self, transcript_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a pre-existing transcript from JSON file.
        
        This supports the requirement to work with pre-transcribed text!
        """
        transcript_path = Path(transcript_path)
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def transcribe_batch(
        self, 
        audio_paths: list[Union[str, Path]],
        save_transcripts: bool = True
    ) -> list[Dict[str, Any]]:
        """
        Transcribe multiple audio files.
        
        BATCH PROCESSING INSIGHT:
        ------------------------
        We load the model once, then process all files.
        This is much more efficient than loading per-file!
        
        For very large batches, consider:
        1. Progress tracking (we use tqdm)
        2. Checkpointing (save after each file)
        3. Error handling (continue on failure)
        """
        from tqdm import tqdm
        
        self.load_model()  # Ensure model is ready
        
        results = []
        
        for audio_path in tqdm(audio_paths, desc="Transcribing"):
            try:
                transcription = self.transcribe_audio(audio_path)
                
                if save_transcripts:
                    self.save_transcript(transcription)
                    
                results.append(transcription)
                
            except Exception as e:
                print(f"[Transcriber] Error processing {audio_path}: {e}")
                results.append({
                    "file_path": str(audio_path),
                    "error": str(e),
                    "text": None
                })
        
        return results


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def transcribe_file(audio_path: Union[str, Path], model_size: str = "base") -> str:
    """
    Quick function to transcribe a single file.
    
    Usage:
        from src.transcription.transcriber import transcribe_file
        text = transcribe_file("my_call.mp3")
    """
    transcriber = AudioTranscriber(model_size=model_size)
    result = transcriber.transcribe_audio(audio_path)
    return result["text"]


if __name__ == "__main__":
    # Quick test
    print("Transcriber module loaded successfully!")
    print(f"Whisper model size: {WHISPER_MODEL_SIZE}")
    print(f"Supported formats: {SUPPORTED_AUDIO_FORMATS}")
    print(f"Transcripts directory: {TRANSCRIPTS_DIR}")

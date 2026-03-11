# Customer Sentiment Analyzer

**An AI-powered tool that analyzes customer sentiment and topics from call recordings using 100% local processing.**

## Features
- **Audio Transcription**: Converts call recordings to text using OpenAI's Whisper.
- **Sentiment Analysis**: Detects positive, neutral, and negative sentiments.
- **Emotion Detection**: Identifies emotions like frustration, satisfaction, and urgency.
- **Topic Modeling**: Extracts key topics from conversations using BERTopic.
- **Interactive Dashboard**: Visualize insights with Streamlit.

## Why Local AI?
- **Privacy**: Customer data never leaves your machine.
- **Cost-Effective**: No API costs for processing.
- **Reliability**: Works offline without internet dependency.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zenUnicorn/Customer-Sentiment-analyzer.git
   cd Customer-Sentiment-analyzer
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install FFmpeg (required for audio processing):
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
   - Windows: [Download here](https://ffmpeg.org/download.html)

## Usage
### Quick Demo
Run a demo with sample text:
```bash
python main.py --demo
```

### Analyze Audio Files
Analyze a single audio file:
```bash
python main.py --audio path/to/call.mp3
```

Batch process a directory of audio files:
```bash
python main.py --batch data/audio/
```

### Launch the Dashboard
Start the interactive dashboard:
```bash
python main.py --dashboard
```
Open `http://localhost:8501` in your browser.

## Project Structure
```
CSST/
├── src/                # Source code
│   ├── transcription/  # Audio transcription module
│   ├── analysis/       # Sentiment, emotion, and topic analysis
│   ├── dashboard/      # Streamlit dashboard
│   └── pipeline/       # End-to-end processing pipeline
├── data/               # Sample data and transcripts
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── main.py             # Entry point for CLI and dashboard
└── README.md           # Project documentation
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## References
- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [BERTopic](https://maartengr.github.io/BERTopic/)
- [Streamlit](https://streamlit.io/)

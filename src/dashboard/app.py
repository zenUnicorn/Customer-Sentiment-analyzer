"""
Streamlit Dashboard for Customer Sentiment Analysis.

DEEP DIVE LESSON: Data Visualization & Dashboards
=================================================

Why Streamlit?
-------------
Traditional web dashboards need: HTML, CSS, JavaScript, backend API, database...
Streamlit: Write Python, get interactive web app. Perfect for data scientists!

Dashboard Design Principles:
---------------------------
1. OVERVIEW FIRST: Show summary metrics at top (like an executive summary)
2. DRILL DOWN: Allow users to explore details
3. CONTEXT: Always explain what numbers mean
4. INTERACTIVITY: Let users filter, sort, explore
5. EXPORT: Allow data download for further analysis

Our Dashboard Structure:
-----------------------
1. Sidebar: Upload files, adjust settings
2. Overview Tab: KPIs and aggregate metrics
3. Sentiment Tab: Detailed sentiment analysis
4. Emotion Tab: Emotion detection results
5. Topics Tab: Topic distribution and exploration
6. Call Details Tab: Individual call deep dive

Technical Concepts Covered:
--------------------------
- Streamlit state management
- Plotly interactive charts
- Real-time processing with progress
- Session caching for performance
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import AUDIO_DIR, RESULTS_DIR, COLORS, SUPPORTED_AUDIO_FORMATS

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Customer Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CACHED FUNCTIONS (Performance Optimization)
# =============================================================================
@st.cache_resource
def load_pipeline():
    """
    Load the analysis pipeline once and cache it.
    
    STREAMLIT CACHING:
    -----------------
    @st.cache_resource: For objects that should persist (models, connections)
    @st.cache_data: For data that should be cached (DataFrames, API responses)
    
    Without caching, models would reload on every interaction!
    """
    from src.pipeline import CallAnalysisPipeline
    return CallAnalysisPipeline(whisper_model="base")


@st.cache_data
def load_results_file(file_path: str) -> dict:
    """Load analysis results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_uploaded_files(uploaded_files):
    """
    Process uploaded audio/transcript files through the analysis pipeline.
    
    This function:
    1. Saves uploaded files to a temp directory
    2. Runs them through the analysis pipeline
    3. Stores results in session state for display
    """
    import tempfile
    import os
    
    # Show processing status in main area
    st.info("🔄 Starting analysis... This may take a few minutes for audio files.")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create temp directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = []
            
            # Save uploaded files to temp directory
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Saving file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)
                
                progress_bar.progress((i + 1) / (len(uploaded_files) * 2))  # First half: saving
            
            # Load the analysis pipeline
            status_text.text("Loading AI models... (first run downloads ~1GB of models)")
            pipeline = load_pipeline()
            
            # Process files
            if len(saved_paths) == 1:
                # Single file analysis
                status_text.text(f"Analyzing: {uploaded_files[0].name}")
                result = pipeline.process_single_call(saved_paths[0])
                progress_bar.progress(1.0)
                st.session_state.loaded_results = result
            else:
                # Batch analysis
                status_text.text(f"Batch processing {len(saved_paths)} files...")
                results = pipeline.process_batch(saved_paths, extract_topics=(len(saved_paths) >= 5))
                progress_bar.progress(1.0)
                st.session_state.loaded_results = results
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Success message
        st.success("✅ Analysis complete! Results are displayed below.")
        st.rerun()  # Refresh to show results
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)  # Show full error for debugging


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_sentiment_gauge(value: float, title: str = "Overall Sentiment") -> go.Figure:
    """
    Create a gauge chart for sentiment score.
    
    VISUALIZATION CHOICE:
    --------------------
    Gauges are intuitive for showing where a value falls on a scale.
    Everyone understands a speedometer-like display.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': "", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 2},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.2], 'color': COLORS["negative"]},
                {'range': [-0.2, 0.2], 'color': COLORS["neutral"]},
                {'range': [0.2, 1], 'color': COLORS["positive"]}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_emotion_radar(emotions: dict) -> go.Figure:
    """
    Create a radar chart for emotion visualization.
    
    WHY RADAR CHARTS FOR EMOTIONS:
    -----------------------------
    Radar charts show multiple dimensions at once.
    You can instantly see the "emotional fingerprint" of a call.
    """
    categories = list(emotions.keys())
    values = list(emotions.values())
    
    # Close the radar shape
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(41, 128, 185, 0.3)',
        line=dict(color='rgb(41, 128, 185)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


def create_sentiment_timeline(segments: list) -> go.Figure:
    """
    Create a timeline showing sentiment change through the call.
    
    INSIGHT VALUE:
    -------------
    This visualization shows WHERE problems occurred in the call.
    Did it start bad and improve? Or the opposite?
    """
    times = []
    sentiments = []
    texts = []
    
    for seg in segments:
        if seg.get("sentiment"):
            times.append(seg.get("start", 0))
            sentiments.append(seg["sentiment"]["compound_score"])
            texts.append(seg.get("text", "")[:50] + "...")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=sentiments,
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='rgb(41, 128, 185)', width=2),
        marker=dict(size=8),
        text=texts,
        hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Sentiment:</b> %{y:.2f}<br><b>Text:</b> %{text}<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    fig.add_hline(y=0.2, line_dash="dot", line_color=COLORS["positive"], annotation_text="Positive threshold")
    fig.add_hline(y=-0.2, line_dash="dot", line_color=COLORS["negative"], annotation_text="Negative threshold")
    
    fig.update_layout(
        title="Sentiment Through the Call",
        xaxis_title="Time (seconds)",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        height=400
    )
    
    return fig


def create_topic_distribution(topic_summary: dict) -> go.Figure:
    """Create a bar chart of topic distribution."""
    if not topic_summary or "topic_distribution" not in topic_summary:
        return None
    
    dist = topic_summary["topic_distribution"]
    
    df = pd.DataFrame(dist)
    
    fig = px.bar(
        df,
        x='label',
        y='percentage',
        title='Call Topics Distribution',
        labels={'label': 'Topic', 'percentage': 'Percentage of Calls'},
        color='percentage',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Sentiment AI")
        st.title("📊 Settings")
        
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Analysis Mode",
            ["Upload Files", "Load Results", "Live Demo"],
            help="Choose how to analyze calls"
        )
        
        st.markdown("---")
        
        if mode == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload Audio/Transcript Files",
                type=["mp3", "wav", "m4a", "json"],
                accept_multiple_files=True,
                help="Upload call recordings or pre-transcribed JSON files"
            )
            
            if uploaded_files:
                st.info(f"{len(uploaded_files)} files selected")
                
                if st.button("🚀 Analyze", type="primary"):
                    # Process the uploaded files
                    process_uploaded_files(uploaded_files)
                    
        elif mode == "Load Results":
            # List existing results
            result_files = list(RESULTS_DIR.glob("*.json"))
            
            if result_files:
                selected_result = st.selectbox(
                    "Select Results File",
                    options=result_files,
                    format_func=lambda x: x.stem
                )
                
                if st.button("📂 Load Results"):
                    st.session_state.loaded_results = load_results_file(str(selected_result))
            else:
                st.warning("No results files found. Process some calls first!")
                
        else:  # Live Demo
            st.info("Demo mode uses sample data to showcase features.")
            if st.button("🎮 Load Demo"):
                st.session_state.loaded_results = get_demo_data()
    
    # Main content
    st.title("🎯 Customer Sentiment Analyzer")
    st.markdown("*AI-powered analysis of customer call recordings*")
    
    # Check if we have results to display
    if "loaded_results" not in st.session_state:
        # Welcome screen
        st.markdown("""
        ## Welcome! 👋
        
        This dashboard analyzes customer call recordings to extract:
        
        - **📊 Sentiment Scores** - Overall positive/negative/neutral assessment
        - **😊 Emotion Detection** - Frustration, satisfaction, urgency levels  
        - **🏷️ Topic Extraction** - What customers are calling about
        - **📈 Actionable Insights** - Trends and patterns across calls
        
        ### Getting Started:
        
        1. **Upload Files** - Select audio recordings (.mp3, .wav) or transcripts (.json)
        2. **Click Analyze** - AI processes each call through our pipeline
        3. **Explore Results** - Use the tabs below to dive into insights
        
        Or select **Live Demo** in the sidebar to see example results!
        """)
        
        # Show architecture diagram
        with st.expander("🔧 System Architecture"):
            st.markdown("""
            ```
            ┌─────────────────────────────────────────────────────────┐
            │                    PROCESSING PIPELINE                   │
            └─────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
            │   WHISPER     │  │  TRANSFORMERS │  │   BERTOPIC    │
            │ Transcription │  │  Sentiment &  │  │    Topic      │
            │    (Local)    │  │   Emotion     │  │  Extraction   │
            └───────────────┘  └───────────────┘  └───────────────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        ▼
                            ┌───────────────────┐
                            │   DASHBOARD       │
                            │   Visualization   │
                            └───────────────────┘
            ```
            """)
        return
    
    # We have results - display them
    results = st.session_state.loaded_results
    
    # Handle both single call and batch results
    is_batch = "results" in results and isinstance(results.get("results"), list)
    
    if is_batch:
        display_batch_results(results)
    else:
        display_single_result(results)


def display_batch_results(batch_results: dict):
    """Display results for batch processing."""
    
    # Top-level metrics
    st.markdown("## 📊 Batch Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Calls",
            batch_results.get("total_calls", 0),
            help="Number of calls processed"
        )
    
    with col2:
        st.metric(
            "Successful",
            batch_results.get("successful", 0),
            delta=f"{batch_results.get('failed', 0)} failed" if batch_results.get('failed', 0) > 0 else None,
            delta_color="inverse"
        )
    
    with col3:
        agg = batch_results.get("aggregate_metrics", {})
        avg_sentiment = agg.get("sentiment", {}).get("average_compound", 0)
        st.metric(
            "Avg Sentiment",
            f"{avg_sentiment:.2f}",
            delta="Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral",
            delta_color="normal" if avg_sentiment >= 0 else "inverse"
        )
    
    with col4:
        emotions = agg.get("emotions", {})
        frustrated = emotions.get("high_frustration_calls", 0)
        st.metric(
            "High Frustration",
            frustrated,
            help="Calls with frustration score > 0.5"
        )
    
    st.markdown("---")
    
    # Tabs for detailed views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sentiment", "😊 Emotions", "🏷️ Topics", "📋 Call List"])
    
    with tab1:
        display_sentiment_tab(batch_results)
    
    with tab2:
        display_emotion_tab(batch_results)
    
    with tab3:
        display_topic_tab(batch_results)
    
    with tab4:
        display_call_list_tab(batch_results)


def display_sentiment_tab(batch_results: dict):
    """Display sentiment analysis details."""
    st.markdown("### Sentiment Analysis")
    
    agg = batch_results.get("aggregate_metrics", {}).get("sentiment", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment gauge
        avg_sentiment = agg.get("average_compound", 0)
        fig = create_sentiment_gauge(avg_sentiment, "Average Sentiment")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Category distribution
        dist = agg.get("category_distribution", {})
        if dist:
            fig = px.pie(
                values=list(dist.values()),
                names=list(dist.keys()),
                title="Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No category distribution available")


def display_emotion_tab(batch_results: dict):
    """Display emotion detection details."""
    st.markdown("### Emotion Analysis")
    
    agg = batch_results.get("aggregate_metrics", {}).get("emotions", {})
    
    if agg:
        emotions = {
            "Frustration": agg.get("average_frustration", 0),
            "Satisfaction": agg.get("average_satisfaction", 0),
            "Urgency": agg.get("average_urgency", 0)
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_emotion_radar(emotions)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Summary metrics
            st.markdown("#### Key Insights")
            
            high_frustration = agg.get("high_frustration_calls", 0)
            high_satisfaction = agg.get("high_satisfaction_calls", 0)
            
            if high_frustration > high_satisfaction:
                st.error(f"⚠️ {high_frustration} calls had high frustration levels")
            elif high_satisfaction > high_frustration:
                st.success(f"✅ {high_satisfaction} calls had high satisfaction levels")
            else:
                st.info("Mixed emotional responses across calls")
            
            # Emotion bars
            for emotion, value in emotions.items():
                st.progress(value, text=f"{emotion}: {value:.2f}")


def display_topic_tab(batch_results: dict):
    """Display topic extraction results."""
    st.markdown("### Topic Analysis")
    
    topic_analysis = batch_results.get("topic_analysis", {})
    
    if topic_analysis and "error" not in topic_analysis:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_topic_distribution(topic_analysis)
            if fig:
                st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Topic Summary")
            st.metric("Topics Found", topic_analysis.get("num_topics_found", 0))
            st.metric("Categorization Rate", f"{topic_analysis.get('categorization_rate', 0):.1f}%")
            
            st.markdown("---")
            st.markdown("#### Top Topics")
            for topic in topic_analysis.get("topic_distribution", [])[:5]:
                st.write(f"**{topic['label']}**: {topic['percentage']:.1f}%")
    else:
        st.warning("Topic analysis not available. Need at least 5 documents.")


def display_call_list_tab(batch_results: dict):
    """Display list of individual calls with details."""
    st.markdown("### Individual Calls")
    
    results = batch_results.get("results", [])
    
    if not results:
        st.info("No call results available")
        return
    
    # Create summary dataframe
    call_data = []
    for r in results:
        sentiment = r.get("sentiment_analysis", {})
        emotion = r.get("emotion_analysis", {})
        
        call_data.append({
            "Call ID": r.get("call_id", "Unknown"),
            "Duration (s)": r.get("transcript", {}).get("duration_seconds", 0),
            "Sentiment": sentiment.get("overall_category", "Unknown"),
            "Compound": sentiment.get("average_compound", 0),
            "Dominant Emotion": emotion.get("dominant_emotion", "Unknown"),
            "Topic": r.get("topic", {}).get("topic_id", "N/A")
        })
    
    df = pd.DataFrame(call_data)
    
    # Interactive table
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Compound": st.column_config.ProgressColumn(
                "Sentiment Score",
                min_value=-1,
                max_value=1
            )
        }
    )
    
    # Select call for details
    selected_call_id = st.selectbox(
        "Select call for details",
        options=[r.get("call_id") for r in results]
    )
    
    if selected_call_id:
        selected_call = next((r for r in results if r.get("call_id") == selected_call_id), None)
        
        if selected_call:
            with st.expander(f"📞 Details: {selected_call_id}", expanded=True):
                display_single_result(selected_call)


def display_single_result(result: dict):
    """Display detailed results for a single call."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment = result.get("sentiment_analysis", {})
        fig = create_sentiment_gauge(
            sentiment.get("average_compound", 0),
            "Call Sentiment"
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        emotion = result.get("emotion_analysis", {})
        avg_emotions = emotion.get("average_emotions", {})
        if avg_emotions:
            fig = create_emotion_radar(avg_emotions)
            st.plotly_chart(fig, width='stretch')
    
    with col3:
        st.markdown("#### Call Details")
        transcript = result.get("transcript", {})
        st.write(f"**Duration:** {transcript.get('duration_seconds', 0):.1f}s")
        st.write(f"**Words:** {transcript.get('word_count', 0)}")
        st.write(f"**Language:** {transcript.get('language', 'Unknown')}")
        
        sentiment = result.get("sentiment_analysis", {})
        st.write(f"**Trajectory:** {sentiment.get('trajectory', 'Unknown')}")
        st.write(f"**Ending:** {sentiment.get('ending_sentiment', 'Unknown')}")
    
    # Sentiment timeline
    segments = result.get("segments", [])
    if segments:
        st.markdown("---")
        fig = create_sentiment_timeline(segments)
        st.plotly_chart(fig, width='stretch')
    
    # Transcript with annotations
    with st.expander("📝 Full Transcript"):
        text = result.get("transcript", {}).get("full_text", "No transcript available")
        st.text_area("Transcript", value=text, height=200, disabled=True, label_visibility="hidden")


def get_demo_data() -> dict:
    """Generate demo data for showcase."""
    return {
        "batch_id": "demo_batch",
        "total_calls": 10,
        "successful": 10,
        "failed": 0,
        "aggregate_metrics": {
            "sentiment": {
                "average_compound": 0.15,
                "category_distribution": {
                    "positive": 4,
                    "neutral": 3,
                    "negative": 3
                }
            },
            "emotions": {
                "average_frustration": 0.35,
                "average_satisfaction": 0.45,
                "average_urgency": 0.25,
                "high_frustration_calls": 3,
                "high_satisfaction_calls": 5
            }
        },
        "topic_analysis": {
            "topic_distribution": [
                {"label": "billing | charges | payment", "percentage": 30},
                {"label": "shipping | delivery | tracking", "percentage": 25},
                {"label": "technical | support | issue", "percentage": 20},
                {"label": "account | login | password", "percentage": 15},
                {"label": "refund | cancel | return", "percentage": 10}
            ],
            "num_topics_found": 5,
            "categorization_rate": 85.0
        },
        "results": [
            {
                "call_id": "demo_call_001",
                "transcript": {"duration_seconds": 180, "word_count": 450, "language": "english", "full_text": "This is a demo transcript..."},
                "sentiment_analysis": {"average_compound": 0.3, "overall_category": "positive", "trajectory": "improving", "ending_sentiment": "positive"},
                "emotion_analysis": {"average_emotions": {"frustration": 0.2, "satisfaction": 0.6, "urgency": 0.1, "neutral": 0.3}, "dominant_emotion": "satisfaction"},
                "segments": [
                    {"start": 0, "end": 30, "text": "Hello, I need help with my order", "sentiment": {"compound_score": -0.1}},
                    {"start": 30, "end": 60, "text": "The agent helped resolve my issue", "sentiment": {"compound_score": 0.5}},
                    {"start": 60, "end": 90, "text": "Thank you so much for your help!", "sentiment": {"compound_score": 0.8}}
                ]
            }
        ]
    }


if __name__ == "__main__":
    main()

"""
Topic Extraction Module using BERTopic.

DEEP DIVE LESSON: Topic Modeling Architecture
=============================================

What is Topic Modeling?
----------------------
Topic modeling automatically discovers themes/subjects in text collections.
Instead of predefined categories, it FINDS patterns in your data.

Evolution of Topic Modeling:
---------------------------
1. LDA (Latent Dirichlet Allocation) - 2003
   - Bag of words approach
   - Fast but ignores word order/meaning
   - Requires pre-specifying number of topics

2. BERTopic - 2022 (What we use!)
   - Uses transformer embeddings (semantic meaning)
   - Leverages clustering (HDBSCAN)
   - Dynamic topic discovery
   - Much better topic coherence

How BERTopic Works (The Pipeline):
---------------------------------
1. EMBEDDING: Convert documents to vectors using sentence-transformers
   - Each transcript becomes a 384-dimensional vector
   - Semantically similar documents have similar vectors

2. DIMENSIONALITY REDUCTION (UMAP):
   - 384 dimensions → 5 dimensions
   - Preserves local structure (similar docs stay close)
   - Makes clustering tractable

3. CLUSTERING (HDBSCAN):
   - Finds dense regions in the reduced space
   - Handles noise (outlier documents)
   - Doesn't require pre-specifying cluster count

4. TOPIC REPRESENTATION (c-TF-IDF):
   - Extracts keywords that define each cluster
   - Class-based TF-IDF across topics
   - Creates human-readable topic labels

Why This Matters for Customer Calls:
-----------------------------------
- Discover WHAT customers are calling about
- Track topic trends over time
- Identify emerging issues before they explode
- No manual categorization needed!
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOPIC_MODEL_EMBEDDING, MIN_TOPIC_SIZE, NR_TOPICS, SEED_TOPICS


class TopicExtractor:
    """
    Extracts topics from transcripts using BERTopic.
    
    DESIGN PATTERN: Lazy Initialization
    -----------------------------------
    BERTopic model is heavy - we only initialize when needed.
    This also allows us to configure it differently per use case.
    """
    
    def __init__(
        self, 
        embedding_model: str = TOPIC_MODEL_EMBEDDING,
        min_topic_size: int = MIN_TOPIC_SIZE,
        nr_topics: Union[int, str] = NR_TOPICS
    ):
        """
        Initialize the topic extractor.
        
        Args:
            embedding_model: Sentence-transformer model for embeddings
            min_topic_size: Minimum documents to form a topic
            nr_topics: Number of topics ('auto' or integer)
            
        TECHNICAL NOTE: Model Selection
        --------------------------------
        all-MiniLM-L6-v2 is our embedding model:
        - 384 dimensions (balanced size)
        - 22M parameters (fast inference)
        - Excellent semantic understanding
        - Perfect for English text
        """
        self.embedding_model = embedding_model
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.topic_model = None
        self.fitted = False
        
    def load_model(self) -> None:
        """
        Initialize BERTopic with our configuration.
        
        DEEP DIVE: BERTopic Components
        ------------------------------
        We can customize each stage of the pipeline:
        
        1. embedding_model: How we vectorize text
        2. umap_model: Dimensionality reduction params
        3. hdbscan_model: Clustering parameters
        4. vectorizer_model: How we extract keywords
        5. representation_model: How we name topics
        """
        if self.topic_model is None:
            print(f"[Topics] Initializing BERTopic...")
            
            # Import here to avoid slow startup
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Configure embedding model
            sentence_model = SentenceTransformer(self.embedding_model)
            
            # Configure UMAP for dimensionality reduction
            # Lower n_neighbors = more local structure preserved
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            
            # Configure HDBSCAN for clustering
            # min_cluster_size = minimum docs to form a topic
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_topic_size,
                metric='euclidean',
                cluster_selection_method='eom',  # Excess of mass
                prediction_data=True
            )
            
            # Configure vectorizer for keyword extraction
            # We remove very common/rare words
            vectorizer_model = CountVectorizer(
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,  # Appear in at least 2 docs
            )
            
            # Build the model
            self.topic_model = BERTopic(
                embedding_model=sentence_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics=self.nr_topics if self.nr_topics != "auto" else None,
                verbose=True
            )
            
            print(f"[Topics] BERTopic initialized")
    
    def extract_topics(
        self, 
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract topics from a collection of documents.
        
        Args:
            documents: List of text documents (transcripts)
            doc_ids: Optional IDs for each document
            
        Returns:
            Dict with:
                - topics: Topic assignments for each document
                - topic_info: Information about each topic
                - topic_keywords: Keywords for each topic
                - document_topics: Mapping of documents to topics
                
        IMPORTANT: Minimum Data Requirement
        -----------------------------------
        BERTopic needs sufficient data to cluster:
        - Minimum ~10 documents for meaningful topics
        - More documents = better topic separation
        - Very short documents may cluster poorly
        """
        self.load_model()
        
        if len(documents) < 5:
            print("[Topics] Warning: Very few documents. Topics may not be meaningful.")
        
        # Filter empty documents
        valid_docs = []
        valid_ids = []
        
        for i, doc in enumerate(documents):
            if doc and doc.strip():
                valid_docs.append(doc)
                valid_ids.append(doc_ids[i] if doc_ids else f"doc_{i}")
        
        if len(valid_docs) == 0:
            return {"error": "No valid documents to analyze"}
        
        print(f"[Topics] Extracting topics from {len(valid_docs)} documents...")
        
        # Fit the model and get topics
        topics, probabilities = self.topic_model.fit_transform(valid_docs)
        
        self.fitted = True
        
        # Get topic information
        topic_info = self.topic_model.get_topic_info()
        
        # Build results
        # -1 means outlier (didn't fit any topic)
        topic_assignments = []
        
        for i, (doc_id, topic, prob) in enumerate(zip(valid_ids, topics, probabilities)):
            topic_assignments.append({
                "document_id": doc_id,
                "topic_id": int(topic),
                "topic_probability": float(prob) if isinstance(prob, (int, float)) else float(max(prob)),
                "is_outlier": topic == -1
            })
        
        # Get keywords for each topic
        topic_keywords = {}
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                keywords = self.topic_model.get_topic(topic_id)
                topic_keywords[topic_id] = [
                    {"word": word, "weight": float(weight)}
                    for word, weight in keywords[:10]  # Top 10 keywords
                ]
        
        # Create readable topic labels
        topic_labels = {}
        for topic_id, keywords in topic_keywords.items():
            # Use top 3 keywords as label
            label_words = [k["word"] for k in keywords[:3]]
            topic_labels[topic_id] = " | ".join(label_words)
        
        return {
            "topic_assignments": topic_assignments,
            "topic_info": topic_info.to_dict('records') if hasattr(topic_info, 'to_dict') else [],
            "topic_keywords": topic_keywords,
            "topic_labels": topic_labels,
            "num_topics": len(set(topics)) - (1 if -1 in topics else 0),
            "num_outliers": sum(1 for t in topics if t == -1),
            "total_documents": len(valid_docs)
        }
    
    def extract_single_document_topics(
        self, 
        text: str
    ) -> Dict[str, Any]:
        """
        Extract topics from a single document by chunking.
        
        For a single long transcript, we can:
        1. Split into chunks (by segments or sentences)
        2. Treat each chunk as a "document"
        3. Find what topics appear throughout the call
        
        This shows what a SINGLE call was about!
        """
        if not self.fitted:
            return {
                "error": "Model must be fitted on multiple documents first. "
                         "Call extract_topics() first with your corpus."
            }
        
        # Chunk the text into sentences
        chunks = self._chunk_text(text)
        
        if len(chunks) < 3:
            return {"error": "Text too short to extract meaningful topics"}
        
        # Transform (not fit) new documents
        topics, probabilities = self.topic_model.transform(chunks)
        
        # Aggregate topic mentions
        topic_counts = Counter(t for t in topics if t != -1)
        total_chunks = len(chunks)
        
        topics_in_document = []
        for topic_id, count in topic_counts.most_common():
            topics_in_document.append({
                "topic_id": int(topic_id),
                "mentions": count,
                "percentage": count / total_chunks * 100,
                "keywords": [
                    kw["word"] 
                    for kw in self.topic_model.get_topic(topic_id)[:5]
                ]
            })
        
        return {
            "topics": topics_in_document,
            "primary_topic": topics_in_document[0] if topics_in_document else None,
            "num_chunks_analyzed": total_chunks,
            "outlier_chunks": sum(1 for t in topics if t == -1)
        }
    
    def _chunk_text(self, text: str, chunk_size: int = 3) -> List[str]:
        """
        Split text into overlapping chunks of sentences.
        
        Args:
            text: Full text to chunk
            chunk_size: Number of sentences per chunk
            
        TECHNIQUE: Sentence Chunking
        ----------------------------
        We group sentences because:
        1. Single sentences are too short for good embeddings
        2. Paragraphs may mix multiple topics
        3. Overlapping windows catch topic transitions
        """
        # Simple sentence splitting (could use nltk for better results)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Create overlapping chunks
        chunks = []
        for i in range(0, len(sentences), chunk_size - 1):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def get_topic_summary(
        self, 
        topic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a high-level summary of topics for dashboard.
        
        OUTPUT FOR EXECUTIVES:
        ---------------------
        - What are customers calling about? (topic distribution)
        - What's trending up? (if time data available)
        - How many calls don't fit categories? (outlier rate)
        """
        if "error" in topic_results:
            return topic_results
        
        # Calculate topic distribution
        assignments = topic_results["topic_assignments"]
        labels = topic_results.get("topic_labels", {})
        
        topic_counts = Counter(
            a["topic_id"] for a in assignments 
            if not a["is_outlier"]
        )
        
        total_non_outlier = sum(topic_counts.values())
        
        distribution = []
        for topic_id, count in topic_counts.most_common():
            distribution.append({
                "topic_id": topic_id,
                "label": labels.get(topic_id, f"Topic {topic_id}"),
                "count": count,
                "percentage": count / total_non_outlier * 100 if total_non_outlier > 0 else 0
            })
        
        return {
            "topic_distribution": distribution,
            "total_calls_analyzed": topic_results["total_documents"],
            "categorized_calls": total_non_outlier,
            "uncategorized_calls": topic_results["num_outliers"],
            "categorization_rate": (total_non_outlier / topic_results["total_documents"] * 100) 
                                   if topic_results["total_documents"] > 0 else 0,
            "num_topics_found": topic_results["num_topics"]
        }


# =============================================================================
# GUIDED TOPIC EXTRACTION (Using Seed Topics)
# =============================================================================
class GuidedTopicExtractor(TopicExtractor):
    """
    Topic extraction with seed/guide topics.
    
    WHEN TO USE:
    -----------
    When you have predefined categories (billing, support, etc.)
    but want the model to also discover new topics.
    
    TECHNIQUE: Semi-Supervised Topic Modeling
    -----------------------------------------
    We give the model "hints" about expected topics:
    1. Create initial embeddings from seed topic descriptions
    2. Model uses these as anchors
    3. New topics are still discovered around them
    """
    
    def __init__(self, seed_topics: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.seed_topics = seed_topics or SEED_TOPICS
    
    def extract_topics_guided(
        self, 
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract topics with guidance from seed topics.
        
        Useful when you know your common categories but want
        flexibility to discover new ones.
        """
        self.load_model()
        
        # BERTopic's guided topic modeling uses seed keywords
        # to influence topic formation
        
        # Create seed topic representations
        # Each seed topic becomes initial keywords
        seed_topic_list = [[topic] for topic in self.seed_topics]
        
        # This tells BERTopic to look for these topics
        self.topic_model.seed_topic_list = seed_topic_list
        
        # Now extract normally - model will be influenced by seeds
        return self.extract_topics(documents, doc_ids)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def extract_document_topics(documents: List[str]) -> Dict[str, Any]:
    """
    Quick function to extract topics from documents.
    
    Usage:
        from src.analysis.topics import extract_document_topics
        result = extract_document_topics(["doc1 text", "doc2 text", ...])
    """
    extractor = TopicExtractor()
    return extractor.extract_topics(documents)


if __name__ == "__main__":
    # Quick test with sample data
    print("Testing Topic Extractor...")
    print("Note: Requires multiple documents for meaningful results.")
    
    # Sample documents (simulating call transcripts)
    sample_docs = [
        "I need help with my billing statement. There's a charge I don't recognize from last month.",
        "The product stopped working after two days. I want a refund or replacement.",
        "Can you tell me about your shipping options? I need expedited delivery.",
        "I forgot my password and can't access my account. Need to reset it.",
        "Your customer service is excellent! The agent was very helpful.",
        "The billing department charged me twice for the same order.",
        "Shipping took way too long. It was supposed to arrive in 3 days.",
        "I want to cancel my subscription and get a refund for unused months.",
        "Technical support helped me fix the issue. Great service!",
        "My account was locked after too many login attempts.",
    ]
    
    extractor = TopicExtractor(min_topic_size=2)
    result = extractor.extract_topics(sample_docs)
    
    print(f"\nTopics found: {result['num_topics']}")
    print(f"Outliers: {result['num_outliers']}")
    print("\nTopic Labels:")
    for topic_id, label in result['topic_labels'].items():
        print(f"  Topic {topic_id}: {label}")

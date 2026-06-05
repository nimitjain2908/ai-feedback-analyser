import pandas as pd
from collections import Counter
import re

def get_summary_stats(df: pd.DataFrame) -> dict:
    """Compute high-level stats from analyzed reviews."""
    
    stats = {
        "total_reviews": len(df),
        "avg_sentiment_score": round(df['llm_sentiment_score'].mean(), 2),
        "sentiment_distribution": df['llm_sentiment'].value_counts().to_dict(),
        "top_topics": df['llm_topic'].value_counts().head(5).to_dict(),
        "high_urgency_count": int((df['llm_urgency'] >= 4).sum()),
        "avg_urgency": round(df['llm_urgency'].mean(), 1),
    }
    return stats


def get_urgent_reviews(df: pd.DataFrame, min_urgency: int = 4) -> pd.DataFrame:
    """Return reviews that need immediate attention."""
    urgent = df[df['llm_urgency'] >= min_urgency].copy()
    urgent = urgent.sort_values('llm_urgency', ascending=False)
    return urgent[['review_text', 'llm_sentiment', 'llm_topic',
                    'llm_urgency', 'llm_suggested_action']]


def get_topic_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Average sentiment score per topic — for the bar chart."""
    return (df.groupby('llm_topic')['llm_sentiment_score']
              .mean()
              .round(2)
              .sort_values()
              .reset_index()
              .rename(columns={'llm_topic': 'Topic', 'llm_sentiment_score': 'Avg Sentiment'}))


def get_key_phrases(df: pd.DataFrame) -> dict:
    """Extract and count all key phrases for word cloud."""
    all_phrases = df['llm_key_phrase'].dropna().tolist()
    
    # Split phrases into individual words and clean them
    words = []
    for phrase in all_phrases:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', str(phrase).lower())
        words.extend(cleaned.split())
    
    # Remove common stopwords that add no value
    stopwords = {'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 
                 'for', 'of', 'and', 'or', 'but', 'not', 'with', 'this',
                 'that', 'was', 'are', 'be', 'has', 'had', 'have', 'i',
                 'my', 'we', 'very', 'so', 'too', 'its', 'just'}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    return dict(Counter(words).most_common(50))


def get_urgency_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for urgency vs sentiment scatter plot."""
    matrix = df[['llm_topic', 'llm_urgency', 'llm_sentiment_score', 
                 'llm_sentiment', 'llm_suggested_action', 'review_text']].copy()
    matrix = matrix.rename(columns={
        'llm_topic': 'Topic',
        'llm_urgency': 'Urgency',
        'llm_sentiment_score': 'Sentiment Score',
        'llm_sentiment': 'Sentiment',
        'llm_suggested_action': 'Suggested Action',
        'review_text': 'Review'
    })
    return matrix
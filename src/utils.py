import pandas as pd
import os

def load_reviews(filepath: str, sample_size: int = 100) -> pd.DataFrame:
    """Load and clean the reviews CSV."""
    df = pd.read_csv(filepath)
    
    # Print columns so you can see what's in your dataset
    print(f"Columns found: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Rename columns to standard names
    column_map = {
        'review_content': 'review_text',
        'review_title': 'title',
        'rating': 'rating',
        'product_name': 'product',
        'category': 'category',
    }
    
    # Only rename columns that exist
    existing_map = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=existing_map)
    
    # Drop rows with no review text
    if 'review_text' in df.columns:
        df = df.dropna(subset=['review_text'])
        df['review_text'] = df['review_text'].astype(str).str.strip()
        df = df[df['review_text'].str.len() > 20]
    
    # Take a sample
    df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} reviews for analysis")
    
    return df


def save_results(df: pd.DataFrame, output_path: str):
    """Save analyzed results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
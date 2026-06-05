import os
import json
import time
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a customer insights analyst. 
Analyze the customer review and return ONLY a JSON object with these exact fields:
- sentiment: one of "positive", "negative", "neutral", "mixed"
- sentiment_score: float from -1.0 (very negative) to 1.0 (very positive)
- topic: the main topic (e.g. "product quality", "delivery", "customer service", "pricing", "usability")
- urgency: integer 1-5 where 5 = needs immediate action
- key_phrase: the most important 5-8 word phrase from the review
- suggested_action: one concrete action the business should take (max 15 words)

Return ONLY the JSON object. No explanation. No markdown. No extra text."""

def analyze_single_review(review_text: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Review: {review_text[:500]}"}
                ],
                temperature=0.1,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())

        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"sentiment": "unknown", "sentiment_score": 0,
                        "topic": "unknown", "urgency": 0,
                        "key_phrase": "", "suggested_action": "Review manually"}
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print(f"Rate limit hit — waiting 10s...")
                time.sleep(10)
            else:
                print(f"Error on attempt {attempt+1}: {e}")
                time.sleep(2)

    return {"sentiment": "error", "sentiment_score": 0, "topic": "error",
            "urgency": 0, "key_phrase": "", "suggested_action": "Manual review needed"}


def analyze_all_reviews(df: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing reviews"):
        result = analyze_single_review(row['review_text'])
        results.append(result)
        time.sleep(delay)
    results_df = pd.DataFrame(results)
    for col in results_df.columns:
        df[f'llm_{col}'] = results_df[col].values
    return df
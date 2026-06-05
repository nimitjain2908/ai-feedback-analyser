# AI Customer Feedback Analyser

An end-to-end LLM-powered pipeline that turns raw customer reviews into 
actionable business insights — with an interactive Streamlit dashboard.

## What it does
- Ingests customer review data (CSV)
- Uses **Llama 3.3 70B via Groq API** to extract per-review: sentiment, 
  topic, urgency score, key phrase, and recommended action
- Aggregates results into trend summaries and interactive charts
- Provides a natural language query interface over the feedback corpus

## Features
- Sentiment distribution pie chart
- Sentiment by topic bar chart
- Key phrases word cloud
- Action priority matrix (urgency vs sentiment — fix top-left first)
- High urgency reviews table
- Natural language Q&A over the dataset

## Tech stack
- **LLM**: Llama 3.3 70B (via Groq free tier)
- **Backend**: Python, Pandas
- **Frontend**: Streamlit + Plotly + Matplotlib
- **Prompt design**: Structured JSON extraction with temperature=0.1

## Setup
1. Clone the repo
2. `pip install -r requirements.txt`
3. Get a free API key at console.groq.com
4. Create `.env` with `GROQ_API_KEY=your_key`
5. Add reviews CSV to `data/reviews.csv`
6. `streamlit run app.py`

## Business context
Built to demonstrate LLM-powered analytics, extending traditional NPS 
text analytics (LDA topic modelling) with modern prompt engineering and 
structured JSON extraction. Reduces manual feedback analysis time by ~80%.
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
from groq import Groq
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.utils import load_reviews, save_results
from src.analyzer import analyze_single_review
from src.aggregator import get_summary_stats, get_urgent_reviews, get_topic_sentiment, get_key_phrases, get_urgency_matrix

load_dotenv()

st.set_page_config(page_title="AI Feedback Analyser", layout="wide")
st.title("AI Customer Feedback Analyser")
st.caption("Powered by Llama 3.3 70B via Groq · Built by Nimit Jain")

# ── Sidebar controls ─────────────────────────────────────────────────
st.sidebar.header("Controls")
sample_size = st.sidebar.slider("Reviews to analyse", 10, 150, 50)
run_analysis = st.sidebar.button("▶ Run Analysis", type="primary")

output_path = "outputs/analyzed_reviews.csv"

# ── Load existing results OR run fresh analysis ───────────────────────
if run_analysis:
    with st.spinner("Loading reviews..."):
        df = load_reviews("data/reviews.csv", sample_size=sample_size)

    progress = st.progress(0, text="Analysing with LLM...")
    results = []

    for i, (_, row) in enumerate(df.iterrows()):
        result = analyze_single_review(row['review_text'])
        results.append(result)
        progress.progress((i + 1) / len(df), text=f"Analysing review {i+1}/{len(df)}...")
        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    for col in results_df.columns:
        df[f'llm_{col}'] = results_df[col].values

    save_results(df, output_path)
    st.success(f"✓ Analysis complete! {len(df)} reviews processed.")
    st.session_state['df'] = df

elif os.path.exists(output_path):
    st.session_state['df'] = pd.read_csv(output_path)
    st.info("Showing previous results. Hit 'Run Analysis' to refresh.")

# ── Dashboard ──────────────────────────────────────────────────────────
if 'df' in st.session_state:
    df = st.session_state['df']
    stats = get_summary_stats(df)

    # ── Metric cards ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analysed", stats["total_reviews"])
    c2.metric("Avg sentiment score", stats["avg_sentiment_score"],
              help="-1 = very negative, +1 = very positive")
    c3.metric("High urgency reviews", stats["high_urgency_count"],
              delta="need action" if stats["high_urgency_count"] > 0 else "none")
    c4.metric("Top topic", list(stats["top_topics"].keys())[0] if stats["top_topics"] else "N/A")

    st.divider()

    # ── Row 1: Sentiment pie + Sentiment by topic ─────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment distribution")
        sent_data = pd.DataFrame(list(stats["sentiment_distribution"].items()),
                                 columns=["Sentiment", "Count"])
        fig1 = px.pie(sent_data, values="Count", names="Sentiment",
                      color_discrete_map={"positive": "#1D9E75", "negative": "#D85A30",
                                          "neutral": "#888780", "mixed": "#BA7517"})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Sentiment by topic")
        topic_df = get_topic_sentiment(df)
        fig2 = px.bar(topic_df, x="Avg Sentiment", y="Topic", orientation="h",
                      color="Avg Sentiment", color_continuous_scale="RdYlGn",
                      range_color=[-1, 1])
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Row 2: Word cloud + Action priority matrix ────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Key phrases word cloud")
        phrases = get_key_phrases(df)
        if phrases:
            wc = WordCloud(
                width=800, height=400,
                background_color='black',
                colormap='RdYlGn',
                max_words=50
            ).generate_from_frequencies(phrases)
            fig3, ax = plt.subplots(figsize=(8, 4))
            fig3.patch.set_facecolor('black')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig3)
        else:
            st.info("No key phrases found yet.")

    with col4:
        st.subheader("Action priority matrix")
        st.caption("Top-left = urgent negatives → fix first")
        matrix_df = get_urgency_matrix(df)
        fig4 = px.scatter(
            matrix_df,
            x="Sentiment Score",
            y="Urgency",
            color="Topic",
            hover_data=["Suggested Action", "Review"],
            size_max=15,
        )
        fig4.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig4.add_hline(y=3, line_dash="dash", line_color="gray", opacity=0.5)
        fig4.update_layout(
            xaxis_range=[-1.2, 1.2],
            yaxis_range=[0.5, 5.5],
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── High urgency reviews ──────────────────────────────────────
    st.subheader("🔴 High urgency reviews (need action)")
    urgent = get_urgent_reviews(df)
    if len(urgent) > 0:
        st.dataframe(urgent, use_container_width=True, height=250)
    else:
        st.success("No high urgency reviews found.")

    st.divider()

    # ── Natural language query ────────────────────────────────────
    st.subheader("Ask a question about the feedback")
    query = st.text_input("e.g. What are customers saying about delivery?")

    if query:
        qclient = Groq(api_key=os.getenv("GROQ_API_KEY"))
        sample_reviews = df['review_text'].head(30).tolist()
        context = "\n".join([f"- {r[:200]}" for r in sample_reviews])

        with st.spinner("Thinking..."):
            response = qclient.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content":
                    f"Based on these customer reviews:\n{context}\n\nAnswer concisely: {query}"}],
                max_tokens=300
            )
        st.write(response.choices[0].message.content)

    # ── Full data table ───────────────────────────────────────────
    with st.expander("View full results table"):
        st.dataframe(df, use_container_width=True)
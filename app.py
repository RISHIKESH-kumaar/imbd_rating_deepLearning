import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/predict"  # your FastAPI server must be running

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review below and get a positive/negative sentiment prediction from your LSTM model.")

review_text = st.text_area(
    "Movie review",
    placeholder="e.g. This movie was absolutely fantastic, great acting and story!",
    height=150
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_clicked = st.button("Predict", type="primary")

# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
if predict_clicked:
    if not review_text or not review_text.strip():
        st.warning("Please enter some review text first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(API_URL, json={"text": review_text}, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]

                    if sentiment == "positive":
                        st.success(f"**Positive** 😀 (confidence: {confidence:.2%})")
                    else:
                        st.error(f"**Negative** 😞 (confidence: {confidence:.2%})")

                    st.progress(confidence)

                    with st.expander("See cleaned text sent to the model"):
                        st.code(result["cleaned_text"])

                else:
                    st.error(f"API returned an error (status {response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure it's running with `uvicorn main:app --reload` on port 8000."
                )
            except requests.exceptions.Timeout:
                st.error("The request timed out. The model may be taking too long to respond.")

# ---------------------------------------------------------------------------
# Footer / status check
# ---------------------------------------------------------------------------
st.divider()
if st.button("Check API health"):
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
        st.json(health)
    except requests.exceptions.ConnectionError:
        st.error("FastAPI server is not reachable.")
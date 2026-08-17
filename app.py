"""
app.py

Standalone Streamlit app for IMDB sentiment analysis.
Loads the trained LSTM model and tokenizer directly (no separate
FastAPI backend needed).
"""

import re
import pickle

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negation_words = {
    "not", "no", "nor", "never", "none", "neither",
    "cannot", "n't", "without", "against"
}
stop_words = stop_words - negation_words

MAX_LEN = 200
MODEL_PATH = "sentiment_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"


# ---------------------------------------------------------------------------
# Load model + tokenizer once, cached
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


model, tokenizer = load_artifacts()


# ---------------------------------------------------------------------------
# Preprocessing — must match training pipeline exactly
# ---------------------------------------------------------------------------
def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()

    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    if remove_stopwords:
        tokens = [w for w in tokens if w not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review below and get a positive/negative sentiment prediction from an LSTM model.")

review_text = st.text_area(
    "Movie review",
    placeholder="e.g. This movie was absolutely fantastic, great acting and story!",
    height=150
)

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not review_text or not review_text.strip():
        st.warning("Please enter some review text first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(review_text)
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

            prob = float(model.predict(padded, verbose=0)[0][0])
            sentiment = "positive" if prob >= 0.5 else "negative"
            confidence = prob if sentiment == "positive" else 1 - prob

        if sentiment == "positive":
            st.success(f"**Positive** 😀 (confidence: {confidence:.2%})")
        else:
            st.error(f"**Negative** 😞 (confidence: {confidence:.2%})")

        st.progress(confidence)

        with st.expander("See cleaned text sent to the model"):
            st.code(cleaned)

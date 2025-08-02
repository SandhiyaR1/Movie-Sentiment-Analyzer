import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ============ PAGE SETUP ============
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎥", layout="centered")

# ============ CUSTOM CSS ============

st.markdown("""
    <style>
    body {
        background-color: #0f0f0f;
        color: white;
    }
    .stApp {
        background-color: #0f0f0f;
    }
    h1 {
        font-size: 3.5rem;
        color: #ff00ff;
        text-shadow: 0 0 5px #ff00ff, 0 0 15px #ff00ff;
        text-align: center;
    }
    h3 {
        text-align: center;
        margin-top: -20px;
        color: #ff69b4;
        font-weight: 400;
    }
    .comment-box textarea {
        border-radius: 15px;
        border: 2px solid #ff00ff;
        background-color: #1e1e1e;
        color: white;
        font-size: 18px;
    }
    .analyze-button button {
        background: linear-gradient(45deg, #ff00ff, #ff1493);
        border: none;
        color: white;
        padding: 10px 30px;
        border-radius: 10px;
        font-size: 18px;
        box-shadow: 0 0 10px #ff00ff;
    }
    .sentiment-box {
        font-size: 1.5rem;
        text-align: center;
        margin-top: 20px;
    }
    .confidence-bar .stProgress > div > div {
        background-color: #00bfff;
    }
    </style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown("<h1>🎬 Movie Sentiment Analyzer</h1>", unsafe_allow_html=True)

# ============ LOAD TOKENIZER ============
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("❌ Tokenizer file not found. Please upload 'tokenizer.pickle'.")
    st.stop()

# ============ LOAD MODEL ============
try:
    model = load_model("newmodel.h5")
except Exception as e:
    st.error("❌ Model loading failed. Make sure 'sentiment_model.h5' is uploaded correctly.")
    st.stop()

# ============ TEXT INPUT ============
st.markdown("#### 🎤 Write your honest movie review:")
user_input = st.text_area("", height=180, placeholder="Type your review here...", key="comment_box", help="Share how the movie made you feel 🎭")

# ============ ANALYZE BUTTON ============
analyze = st.button("🔍 Analyze Review", key="analyze_button")

# ============ PREDICTION ============
if analyze:
    if user_input.strip() == "":
        st.warning("🚫 Please enter some text.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=200)
        prediction = model.predict(padded)[0][0]

        # Result
        st.markdown("---")
        if prediction > 0.6:
            sentiment = "💖 Positive"
            color = "green"
            emoji = "😊"
        elif prediction < 0.4:
            sentiment = "💔 Negative"
            color = "red"
            emoji = "😞"
        else:
            sentiment = "😐 Neutral"
            color = "yellow"
            emoji = "😐"

        st.markdown(f"<div class='sentiment-box'>🎭 Sentiment: <span style='color:{color}; font-weight:bold'>{sentiment} {emoji}</span></div>", unsafe_allow_html=True)

        # Confidence
        st.markdown("### 🧠 Confidence Level")
        st.progress(float(prediction))

        st.markdown(f"<span style='font-size:16px;'>Model Confidence: <span style='color:lime; font-weight:bold'>{prediction:.2f}</span></span>", unsafe_allow_html=True)


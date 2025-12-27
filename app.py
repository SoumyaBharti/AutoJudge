import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

st.set_page_config(
    page_title="AutoJudge",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf.pkl")
    scaler = joblib.load("scaler.pkl")
    reg = joblib.load("regressor.pkl")
    return tfidf, scaler, reg

tfidf, scaler, reg = load_models()

def combine_text(title, desc, inp, out):
    return " ".join([
        title,
        desc,
        inp,
        out
    ])

def count_symbols(text):
    symbols = ['+', '-', '*', '/', '=', '<', '>', '%']
    return sum(text.count(s) for s in symbols)

def score_to_class(score):
    if score <= 2.8:
        return "Easy"
    elif score <= 5.5:
        return "Medium"
    else:
        return "Hard"

st.title("🧠 AutoJudge: Problem Difficulty Predictor")
st.markdown("Predict programming problem difficulty using **text only**.")

st.markdown("---")

title = st.text_input("Problem Title")

description = st.text_area(
    "Problem Description",
    height=200
)

input_desc = st.text_area(
    "Input Description",
    height=150
)

output_desc = st.text_area(
    "Output Description",
    height=150
)

if st.button("Predict Difficulty"):

    if description.strip() == "":
        st.warning("Please enter a problem description.")
    else:
        full_text = combine_text(
            title,
            description,
            input_desc,
            output_desc
        )

        # TF-IDF
        X_text = tfidf.transform([full_text])

        # Handcrafted features
        text_len = len(full_text)
        symbol_count = count_symbols(full_text)

        X_hand = scaler.transform([[text_len, symbol_count]])

        # Combine
        X_final = hstack([X_text, X_hand])

        # Predict score
        score = reg.predict(X_final)[0]

        # Class
        difficulty = score_to_class(score)

        st.markdown("---")
        st.subheader("🔍 Prediction Results")
        st.write(f"**Predicted Difficulty Score:** `{score:.2f}`")
        st.write(f"**Predicted Difficulty Class:** `{difficulty}`")

        st.info(
            "⚠️ Prediction is based only on textual information. "
            "Actual difficulty may vary due to constraints and intended algorithms."
        )


import streamlit as st
import joblib
import numpy as np
import re
from scipy.sparse import hstack

st.set_page_config(
    page_title="AutoJudge",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODELS ---------------- #
@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf.pkl")
    scaler = joblib.load("scaler.pkl")
    clf = joblib.load("best_classifier.pkl")
    reg = joblib.load("best_regressor.pkl")
    return tfidf, scaler, clf, reg

tfidf, scaler, clf, reg = load_models()

# ---------------- FEATURE EXTRACTION ---------------- #
symbols = ['+', '-', '*', '/', '=', '<', '>', '%']
data_structures = ['array', 'tree', 'graph', 'stack', 'queue', 'heap', 'hash', 'matrix']

def combine_text(title, desc, inp, out):
    return " ".join([title, desc, inp, out])

def extract_handcrafted_features(text):
    text_lower = text.lower()

    features = [
        len(text),                                           # text_len
        sum(text.count(s) for s in symbols),                 # symbol_count
        text.count("\n"),                                    # line_count
        int(bool(re.search(r'constraint|limit|bound', text_lower))),
        int(bool(re.search(r'dynamic programming|dp', text_lower))),
        int(bool(re.search(r'graph|tree|dfs|bfs', text_lower))),
        int(bool(re.search(r'greedy|optimal', text_lower))),
        int(bool(re.search(r'string|substring|character', text_lower))),
        int(bool(re.search(r'array|sequence|list', text_lower))),
        len(text.split()),                                   # word_count
        int(bool(re.search(r'optimal|minimum|maximum', text_lower))),
        sum(text_lower.count(ds) for ds in data_structures), # data_structure_count
        len(re.findall(r'\d+', text)),                       # number_count
        len(re.split(r'[.!?]+', text)),                      # sentence_count
        np.mean([len(w) for w in text.split()]) if text.split() else 0,
        int(bool(re.search(r'O\(', text)))                   # big-O mention
    ]

    return np.array(features).reshape(1, -1)

# ---------------- UI ---------------- #
st.title("🧠 AutoJudge: Problem Difficulty Predictor")
st.markdown("Predict programming problem difficulty using **text only**.")
st.markdown("---")

title = st.text_input("Problem Title")
description = st.text_area("Problem Description", height=200)
input_desc = st.text_area("Input Description", height=150)
output_desc = st.text_area("Output Description", height=150)

if st.button("Predict Difficulty"):

    if not description.strip():
        st.warning("Please enter a problem description.")
    else:
        full_text = combine_text(title, description, input_desc, output_desc)

        # TF-IDF
        X_text = tfidf.transform([full_text])

        # Handcrafted features
        X_hand = extract_handcrafted_features(full_text)
        X_hand_scaled = scaler.transform(X_hand)

        # -------- CLASSIFICATION -------- #
        X_class = hstack([X_text, X_hand_scaled])
        predicted_class = clf.predict(X_class)[0]

        # -------- REGRESSION -------- #
        X_reg = np.hstack([X_text.toarray(), X_hand_scaled])
        predicted_score = reg.predict(X_reg)[0]

        # ---------------- OUTPUT ---------------- #
        st.markdown("---")
        st.subheader("🔍 Prediction Results")

        st.write(f"**Predicted Difficulty (Classifier):** `{predicted_class}`")
        st.write(f"**Predicted Difficulty Score (Regression):** `{predicted_score:.2f}`")

        st.info(
            "⚠️ Classification decides difficulty category. "
            "Regression score provides a continuous difficulty estimate."
        )







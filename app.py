import streamlit as st
import numpy as np
import pickle
import joblib
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Preprocessing setup
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

# Load model
model = load_model("sentiment_analysis.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max length
max_len = joblib.load("max_len.pkl")

label_map = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive"
}

st.title("ANN Sentiment Analyzer")

user_input = st.text_area("Enter text")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        # 🔥 SAME CLEANING AS TRAINING
        cleaned_text = preprocess_text(user_input)

        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')

        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader(f"Prediction: {label_map[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.bar_chart(prediction[0])
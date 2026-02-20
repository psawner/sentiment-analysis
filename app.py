import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


@st.cache_resource
def load_trained_model():
    return load_model("sentiment_analysis.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_max_len():
    return joblib.load("max_len.pkl")

@st.cache_resource
def load_label_classes():
    return joblib.load("label_classes.pkl")

label_classes = load_label_classes()


model = load_trained_model()
tokenizer = load_tokenizer()
max_len = load_max_len()

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center;'>Sentiment Analyzer</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: grey;'>Analyze text sentiment using a deep learning model.</p>",
    unsafe_allow_html=True
)

st.markdown("""
### Project Overview
- Model: Artificial Neural Network (ANN) with Embedding Layer  
- Task: Multi-class Sentiment Classification  
- Classes: Positive, Negative, Neutral, Irrelevant  
""")


st.markdown("---")

user_input = st.text_area("Enter text to analyze", height=150)

predict_clicked = st.button("Analyze Sentiment")

if predict_clicked:

    if user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):

            cleaned_text = preprocess_text(user_input)

            sequence = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(sequence, maxlen=max_len, padding='post')

            prediction = model.predict(padded)

            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100
            predicted_label = label_classes[predicted_class]

        st.markdown("---")

        # Prediction Card
        st.markdown("### Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Predicted Sentiment", value=predicted_label)

        with col2:
            st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.progress(int(confidence))

    
        if confidence >= 80:
            st.success("High confidence prediction.")
        elif confidence >= 60:
            st.info("Moderate confidence prediction.")
        else:
            st.warning("Low confidence. The input may be ambiguous.")

    
        st.markdown("### Interpretation")

        if predicted_label.lower() == "positive":
            st.write("The text expresses satisfaction, praise, or positive emotion.")
        elif predicted_label.lower() == "negative":
            st.write("The text expresses dissatisfaction, criticism, or negative emotion.")
        elif predicted_label.lower() == "neutral":
            st.write("The text does not contain strong emotional signals.")
        else:
            st.write("The text may not be directly relevant to the defined sentiment categories.")

        st.markdown("### Class Probabilities")
        st.bar_chart(prediction[0])

        st.markdown("---")


        st.markdown("## Model Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric("Training Accuracy", "96.5%")
        col2.metric("Validation Accuracy", "96.1%")
        col3.metric("Validation Loss", "0.19")

        st.markdown(
                """
                    **Generalization Insight:**  
                    The small gap between training accuracy (96.5%) and validation accuracy (96.1%) 
                    indicates that the model generalizes well and does not significantly overfit.
                """
        )

        st.caption(
                "Metrics were calculated on a held-out validation dataset during training."
        )


        st.markdown("---")
        
        with st.expander("Model Details & Technical Explanation"):
            st.write("""
                **Architecture**
                - Embedding Layer
                - Dense Hidden Layers
                - Softmax Output Layer

                **Preprocessing**
                    - Lowercasing
                    - Punctuation removal
                    - Tokenization
                    - Sequence padding

                **Training**
                    - Loss Function: Categorical Crossentropy
                    - Optimizer: Adam
                    - Evaluation Metrics: Accuracy

                The model outputs probability scores for each sentiment class.
                The highest probability determines the final classification.
            """)

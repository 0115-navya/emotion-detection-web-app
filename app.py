import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

emotion_emoji = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("😊 Emotion Detection Web App")
st.write("Enter a sentence and detect emotion")

text = st.text_area("Your Text")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vec = vectorizer.transform([text])
        probs = model.predict_proba(text_vec)[0]

        max_prob = np.max(probs)
        pred_index = np.argmax(probs)
        emotion = model.classes_[pred_index]

        # Confidence threshold
        if max_prob < 0.40:
            st.info("The model is not confident enough to make a prediction.")
        else:
            st.success(
                f"Emotion: {emotion} {emotion_emoji.get(emotion,'')} "
            )
            st.write(f"Confidence: {max_prob*100:.2f}%")

        # Show probability breakdown
        st.subheader("Prediction Probabilities")
        for emo, prob in zip(model.classes_, probs):
            st.write(f"{emo}: {prob*100:.2f}%")



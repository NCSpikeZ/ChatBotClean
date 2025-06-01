# app.py

import streamlit as st
import json
import os
import time

st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🎭 Real-Time Emotion Detector")
st.markdown("Shows the latest detected facial emotion from webcam.")

# Fonction pour lire le fichier JSON
def get_emotion_data():
    if os.path.exists("emotion.json"):
        with open("emotion.json", "r") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                return None
    return None

placeholder = st.empty()

# Rafraîchissement toutes les 2 secondes
while True:
    emotion_data = get_emotion_data()

    with placeholder.container():
        if emotion_data:
            st.metric(label="Current Emotion", value=emotion_data["emotion"])
            st.progress(min(int(emotion_data["score"] * 100), 100))
        else:
            st.info("Waiting for emotion data from webcam...")

    time.sleep(2)
